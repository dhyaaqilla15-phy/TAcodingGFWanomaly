# eval.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model import LSTMClassifier
from split import group_train_val_test_split
from suspected import SuspectedCfg, build_suspected_df
from standardize import load_scaler, apply_scaler

from agg_utils import (
    AggParams,
    softmax_np,
    conf_score_from_probs,
    aggregate_vessel,
    confusion_matrix_np,
    metrics_from_cm,
)


def pick_device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _torch_load_compat(path: Path, map_location="cpu") -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_npz(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    groups = data["groups"]
    lm_arr = data["label_map"]
    label_map = {int(k): str(v) for k, v in lm_arr.tolist()}
    if "scaled" not in data.files:
        print("[eval] WARNING: NPZ has no 'scaled' metadata. Prefer rerunning preprocess with the train-only scaler pipeline.")
    return X, y, groups, label_map


class AisSeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        self.X = X
        self.y = y
        self.groups = groups

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.groups[idx], idx


def save_confusion_png(cm: np.ndarray, labels: List[str], out_path: Path, normalize: bool = False) -> None:
    import matplotlib.pyplot as plt

    cm = cm.astype(np.float64)

    if normalize:
        rs = cm.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        show = cm / rs
        title = "Confusion Matrix (normalized)"
        fmt = "{:.2f}"
    else:
        show = cm
        title = "Confusion Matrix"
        fmt = "{:.0f}"

    fig = plt.figure(figsize=(10, 8))
    plt.imshow(show, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()

    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    thresh = (show.max() if show.size else 0) * 0.55
    for i in range(show.shape[0]):
        for j in range(show.shape[1]):
            val = show[i, j]
            plt.text(j, i, fmt.format(val), ha="center", va="center", fontsize=8,
                     color=("white" if val > thresh else "black"))

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _load_split_indices(out_dir: Path) -> Dict[str, np.ndarray] | None:
    p = out_dir / "split_indices.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    out = {"train_idx": d["train_idx"], "test_idx": d["test_idx"]}
    if "val_idx" in d.files:
        out["val_idx"] = d["val_idx"]
    return out


def _load_eval_scaler(model_path: Path, out_dir: Path, ckpt: dict):
    candidates: List[Path] = []
    saved = str(ckpt.get("scaler_path", "")).strip()
    if saved:
        candidates.append(Path(saved))
    candidates.extend([model_path.parent / "scaler.joblib", out_dir / "scaler.joblib"])

    seen = set()
    for p in candidates:
        p = Path(p)
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.exists():
            return load_scaler(p), p

    return None, None


def _top2_margin(avg_prob: np.ndarray) -> float:
    if avg_prob.size < 2:
        return 1.0
    s = np.sort(avg_prob)[::-1]
    return float(s[0] - s[1])


def _get_log_pi_from_ckpt(ckpt: dict) -> np.ndarray:
    if "priors" in ckpt:
        pri = np.array(ckpt["priors"], dtype=np.float32).reshape(-1)
        pri = np.clip(pri, 1e-8, None)
        return np.log(pri).astype(np.float32)

    tau = float(ckpt.get("tau", 0.0))
    la = ckpt.get("logit_adjust", None)
    if la is not None and tau > 1e-9:
        logit_adjust = np.array(la, dtype=np.float32).reshape(-1)
        return (logit_adjust / tau).astype(np.float32)

    return np.zeros((int(ckpt["num_classes"]),), dtype=np.float32)


def _task_name_from_label_map(label_map: Dict[int, str]) -> str:
    vals = {str(v).strip().lower() for v in label_map.values()}
    if vals == {"normal", "spoofing"}:
        return "spoofing"
    if vals in [{"normal", "go_dark"}, {"normal", "godark"}]:
        return "godark"
    if vals == {"fishing", "not_fishing"}:
        return "fishing"
    return "gear"


def _primary_metric_scope(task_name: str) -> str:
    return "sequence" if task_name in {"spoofing", "godark"} else "vessel"


def evaluate(
    data_npz: str | Path,
    model_path: str | Path,
    out_dir: str | Path,
    device: str = "auto",
    batch_size: int = 64,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(model_path)

    X, y, groups, label_map = load_npz(Path(data_npz))
    num_classes = int(len(label_map))
    labels = [label_map.get(i, str(i)) for i in range(num_classes)]
    task_name = _task_name_from_label_map(label_map)

    dev = pick_device(device)

    split_idx = _load_split_indices(out_dir)
    if split_idx is None:
        split = group_train_val_test_split(
            X, y, groups,
            val_size=0.05,
            test_size=test_size,
            random_state=random_state,
        )
        test_idx = split.test_idx
    else:
        test_idx = split_idx["test_idx"]

    y_test = y[test_idx]
    g_test = groups[test_idx]

    ckpt = _torch_load_compat(model_path, map_location="cpu")
    scaler, scaler_path = _load_eval_scaler(model_path, out_dir, ckpt)
    if scaler is None:
        print("[eval] WARNING: scaler.joblib not found; using X from NPZ as-is (legacy mode).")
        X_test = X[test_idx]
    else:
        print(f"[eval] scaler loaded -> {scaler_path}")
        X_test = apply_scaler(X[test_idx], scaler)

    def _build_model() -> LSTMClassifier:
        return LSTMClassifier(
            input_size=int(ckpt["input_size"]),
            hidden_size=int(ckpt["hidden_size"]),
            num_layers=int(ckpt["num_layers"]),
            num_classes=int(ckpt["num_classes"]),
            dropout=float(ckpt.get("dropout", 0.0)),
            bidirectional=bool(ckpt.get("bidirectional", False)),
            input_proj_dim=ckpt.get("input_proj_dim", None),
            embed_dim=ckpt.get("embed_dim", None),
            attention_heads=int(ckpt.get("attention_heads", 0)),
            attention_layers=int(ckpt.get("attention_layers", 0)),
            predict_coords=bool(ckpt.get("predict_coords", False)),
        )

    model = _build_model()
    model.load_state_dict(ckpt["model_state"], strict=True)
    used_cudnn_fallback = False
    if dev.type == "cuda":
        try:
            model.to(dev)
        except RuntimeError as e:
            if "CUDNN_STATUS_INTERNAL_ERROR" not in str(e):
                raise
            print("[eval] cuDNN internal error on model.to(cuda), retrying with cuDNN disabled.")
            torch.backends.cudnn.enabled = False
            torch.cuda.empty_cache()
            used_cudnn_fallback = True
            model = _build_model()
            model.load_state_dict(ckpt["model_state"], strict=True)
            model.to(dev)
    else:
        model.to(dev)
    model.eval()

    ds = AisSeqDataset(X_test, y_test, g_test)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    logits_list = []
    y_list = []
    g_list = []

    with torch.inference_mode():
        for xb, yb, gb, _ in loader:
            while True:
                try:
                    xb_t = (xb if torch.is_tensor(xb) else torch.from_numpy(xb)).float().to(dev, non_blocking=True)
                    if dev.type == "cuda":
                        with torch.amp.autocast("cuda", dtype=torch.float16):
                            lg = model(xb_t)
                    else:
                        lg = model(xb_t)

                    logits_list.append(lg.detach().cpu().float().numpy())
                    y_list.append((yb.detach().cpu().numpy() if torch.is_tensor(yb) else np.asarray(yb)).astype(np.int64))
                    g_list.append(np.asarray(gb, dtype=object))
                    break
                except RuntimeError as err:
                    if (dev.type != "cuda") or ("CUDNN_STATUS_INTERNAL_ERROR" not in str(err)) or used_cudnn_fallback:
                        raise
                    print("[eval] cuDNN internal error during inference, retrying with cuDNN disabled.")
                    torch.backends.cudnn.enabled = False
                    torch.cuda.empty_cache()
                    used_cudnn_fallback = True

    logits_np = np.concatenate(logits_list, axis=0)
    y_np = np.concatenate(y_list, axis=0)
    g_np = np.concatenate(g_list, axis=0)

    log_pi = _get_log_pi_from_ckpt(ckpt)

    # Final eval must not tune on the held-out test split. Use the tau and
    # aggregation parameters selected on validation during training.
    ck_best = AggParams(
        keep_frac=float(ckpt.get("agg_keep_frac", 0.15)),
        min_keep=int(ckpt.get("agg_min_keep", 8)),
        weight_power=float(ckpt.get("agg_weight_power", 3.0)),
        conf_mode=str(ckpt.get("agg_conf_mode", "maxprob_margin2")),
        agg_method=str(ckpt.get("agg_method", "mean_logit")),
    )
    metric_scope = str(ckpt.get("primary_metric_scope", _primary_metric_scope(task_name)))
    best_tau = float(ckpt.get("tau", 0.0))
    best_agg = ck_best

    adj_logits = logits_np - (float(best_tau) * log_pi.reshape(1, -1))
    probs = softmax_np(adj_logits)
    confs = conf_score_from_probs(probs, mode=best_agg.conf_mode)
    pred_seq = np.argmax(probs, axis=1).astype(np.int64)

    cm_seq = confusion_matrix_np(y_np, pred_seq, num_classes)
    m_seq = metrics_from_cm(cm_seq)

    g_str = g_np.astype(str)
    uniq_v = np.unique(g_str)
    v2idx: Dict[str, np.ndarray] = {v: np.where(g_str == v)[0] for v in uniq_v}

    import pandas as pd

    pv_rows = []
    vessel_details = []
    y_true_v = []
    y_pred_v = []

    for vid, idxs in v2idx.items():
        yt = int(np.bincount(y_np[idxs], minlength=num_classes).argmax())
        avg, conf_v, n_used = aggregate_vessel(adj_logits[idxs], probs[idxs], confs[idxs], best_agg)
        yp = int(np.argmax(avg))
        margin = _top2_margin(avg)

        vote = np.bincount(pred_seq[idxs], minlength=num_classes)
        maj_ratio = float(vote.max() / max(int(vote.sum()), 1))

        pv_rows.append(
            {
                "mmsi": str(vid),
                "true_id": yt,
                "true_label": label_map.get(yt, str(yt)),
                "pred_id": yp,
                "pred_label": label_map.get(yp, str(yp)),
                "confidence": float(conf_v),
                "n_sequences": int(len(idxs)),
                "n_used_for_vessel": int(n_used),
            }
        )

        vessel_details.append(
            {
                "vessel_id": str(vid),
                "pred_label": label_map.get(yp, str(yp)),
                "confidence": float(conf_v),
                "margin": float(margin),
                "majority_ratio": float(maj_ratio),
                "n_sequences": int(len(idxs)),
                "true_label": label_map.get(yt, str(yt)),
                "pred_id": int(yp),
                "true_id": int(yt),
            }
        )

        y_true_v.append(yt)
        y_pred_v.append(yp)

    cm_v = confusion_matrix_np(np.array(y_true_v), np.array(y_pred_v), num_classes)
    m_v = metrics_from_cm(cm_v)

    cm_primary = cm_seq if metric_scope == "sequence" else cm_v
    save_confusion_png(cm_primary, labels, out_dir / "confusion_matrix.png", normalize=False)
    save_confusion_png(cm_primary, labels, out_dir / "confusion_matrix_normalized.png", normalize=True)

    pv_path = out_dir / "per_vessel_predictions.csv"
    pd.DataFrame(pv_rows).sort_values(["true_label", "confidence"], ascending=[True, False]).to_csv(pv_path, index=False)

    sus_df = build_suspected_df(vessel_details, SuspectedCfg())
    sus_path = out_dir / "suspected_model.csv"
    sus_df.to_csv(sus_path, index=False)

    summary = {
        "task": task_name,
        "primary_metric_scope": metric_scope,
        "num_classes": num_classes,
        "labels": labels,
        "metrics_seq": m_seq,
        "metrics_vessel": m_v,
        "metrics": (m_seq if metric_scope == "sequence" else m_v),
        "test_sequences": int(len(y_np)),
        "test_vessels": int(len(v2idx)),
        "batch_size": int(batch_size),
        "device_used": str(dev),
        "logit_adjust_used": True,
        "tau": float(best_tau),
        "keep_frac": float(best_agg.keep_frac),
        "min_keep": int(best_agg.min_keep),
        "eval_split": "held_out_test",
        "test_tuning_used": False,
        "scaler": (str(scaler_path) if scaler_path is not None else None),
    }
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[eval] USING VALIDATION-SELECTED: tau={best_tau:.2f} agg=({best_agg.agg_method}, keep={best_agg.keep_frac}, min={best_agg.min_keep}, wp={best_agg.weight_power}, conf={best_agg.conf_mode})")
    if metric_scope == "sequence":
        print(f"[eval][SEQ] acc={m_seq['accuracy']:.4f} macro_f1={m_seq['macro_f1']:.4f} bal_acc={m_seq['balanced_acc']:.4f}")
        print(f"[eval][VESSEL-AUX] acc={m_v['accuracy']:.4f} macro_f1={m_v['macro_f1']:.4f} bal_acc={m_v['balanced_acc']:.4f}")
    else:
        print(f"[eval][VESSEL] acc={m_v['accuracy']:.4f} macro_f1={m_v['macro_f1']:.4f} bal_acc={m_v['balanced_acc']:.4f}")
        print(f"[eval][SEQ-AUX] acc={m_seq['accuracy']:.4f} macro_f1={m_seq['macro_f1']:.4f} bal_acc={m_seq['balanced_acc']:.4f}")
