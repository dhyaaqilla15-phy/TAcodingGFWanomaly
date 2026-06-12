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
from transshipment_ml import predict_transshipment_tabular_and_hybrid
from godark_event import (
    DEFAULT_GODARK_EVENT_MIN_POSITIVE_RATIO,
    DEFAULT_GODARK_EVENT_MIN_POSITIVE_WINDOWS,
    DEFAULT_GODARK_EVENT_PROB_THRESHOLD,
    DEFAULT_GODARK_EVENT_SHORT_MIN_RATIO,
    godark_event_breakdown,
    godark_event_report,
    godark_score,
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


def _load_rule_features(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    if "rule_features" not in data.files:
        return None, []
    rule_features = data["rule_features"].astype(np.float32, copy=False)
    rule_cols = [str(x) for x in data["rule_cols"].tolist()] if "rule_cols" in data.files else []
    if rule_features.ndim != 3 or rule_features.shape[-1] == 0:
        return None, rule_cols
    return rule_features, rule_cols


def _load_window_metadata(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    n = int(data["y"].shape[0])
    event_ids = data["window_event_ids"] if "window_event_ids" in data.files else np.array([""] * n, dtype=object)
    kinds = data["window_kinds"] if "window_kinds" in data.files else np.array(["unknown"] * n, dtype=object)
    return event_ids.astype(object), kinds.astype(object)


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
    if vals in [
        {"normal", "encounter", "loitering"},
        {"normal", "encounter"},
        {"normal", "loitering"},
        {"normal", "potential_transshipment"},
        {"normal", "transshipment"},
    ]:
        return "transshipment"
    if vals == {"fishing", "not_fishing"}:
        return "fishing"
    return "gear"


def _primary_metric_scope(task_name: str) -> str:
    return "sequence" if task_name in {"spoofing", "godark", "transshipment"} else "vessel"


def evaluate(
    data_npz: str | Path,
    model_path: str | Path,
    out_dir: str | Path,
    device: str = "auto",
    batch_size: int = 64,
    test_size: float = 0.2,
    random_state: int = 42,
    godark_event_prob_threshold: float | None = None,
    godark_event_min_positive_windows: int | None = None,
    godark_event_min_positive_ratio: float | None = None,
    godark_event_short_min_positive_ratio: float | None = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(model_path)

    X, y, groups, label_map = load_npz(Path(data_npz))
    rule_features, rule_cols = _load_rule_features(Path(data_npz))
    window_event_ids, window_kinds = _load_window_metadata(Path(data_npz))
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
    event_ids_test = window_event_ids[test_idx] if len(window_event_ids) == len(y) else np.array([""] * len(test_idx), dtype=object)
    kinds_test = window_kinds[test_idx] if len(window_kinds) == len(y) else np.array(["unknown"] * len(test_idx), dtype=object)

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

    godark_prob_threshold = float(
        DEFAULT_GODARK_EVENT_PROB_THRESHOLD
        if godark_event_prob_threshold is None
        else godark_event_prob_threshold
    )
    godark_min_windows = int(
        DEFAULT_GODARK_EVENT_MIN_POSITIVE_WINDOWS
        if godark_event_min_positive_windows is None
        else godark_event_min_positive_windows
    )
    godark_min_ratio = float(
        DEFAULT_GODARK_EVENT_MIN_POSITIVE_RATIO
        if godark_event_min_positive_ratio is None
        else godark_event_min_positive_ratio
    )
    godark_short_min_ratio = float(
        DEFAULT_GODARK_EVENT_SHORT_MIN_RATIO
        if godark_event_short_min_positive_ratio is None
        else godark_event_short_min_positive_ratio
    )
    if task_name == "godark":
        godark_prob_threshold = float(
            ckpt.get("godark_event_prob_threshold", godark_prob_threshold)
            if godark_event_prob_threshold is None
            else godark_event_prob_threshold
        )
        godark_min_windows = int(
            ckpt.get("godark_event_min_positive_windows", godark_min_windows)
            if godark_event_min_positive_windows is None
            else godark_event_min_positive_windows
        )
        godark_min_ratio = float(
            ckpt.get("godark_event_min_positive_ratio", godark_min_ratio)
            if godark_event_min_positive_ratio is None
            else godark_event_min_positive_ratio
        )
        godark_short_min_ratio = float(
            ckpt.get("godark_event_short_min_positive_ratio", godark_short_min_ratio)
            if godark_event_short_min_positive_ratio is None
            else godark_event_short_min_positive_ratio
        )

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
    group_id_col = "event_id" if task_name == "transshipment" else "mmsi"

    for vid, idxs in v2idx.items():
        yt = int(np.bincount(y_np[idxs], minlength=num_classes).argmax())
        avg, conf_v, n_used = aggregate_vessel(adj_logits[idxs], probs[idxs], confs[idxs], best_agg)
        yp = int(np.argmax(avg))
        margin = _top2_margin(avg)

        vote = np.bincount(pred_seq[idxs], minlength=num_classes)
        maj_ratio = float(vote.max() / max(int(vote.sum()), 1))

        pv_rows.append(
            {
                group_id_col: str(vid),
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

    pv_name = "per_event_predictions.csv" if task_name == "transshipment" else "per_vessel_predictions.csv"
    pv_path = out_dir / pv_name
    pd.DataFrame(pv_rows).sort_values(["true_label", "confidence"], ascending=[True, False]).to_csv(pv_path, index=False)

    godark_event_metrics = None
    godark_event_path = None
    godark_event_breakdown_path = None
    if task_name == "godark":
        godark_event_metrics, godark_rows = godark_event_report(
            event_ids=event_ids_test,
            kinds=kinds_test,
            y_true=y_np,
            y_pred=pred_seq,
            probs=probs,
            label_map=label_map,
            prob_threshold=godark_prob_threshold,
            min_positive_windows=godark_min_windows,
            min_positive_ratio=godark_min_ratio,
            short_min_positive_ratio=godark_short_min_ratio,
        )
        godark_event_metrics = dict(godark_event_metrics)
        godark_event_metrics["godark_score"] = godark_score(m_seq, godark_event_metrics)
        godark_event_path = out_dir / "per_godark_event_predictions.csv"
        godark_df = pd.DataFrame(godark_rows)
        if not godark_df.empty:
            godark_df = godark_df.sort_values(
                ["error_type", "max_go_dark_probability", "windows_over_threshold"],
                ascending=[True, False, False],
            )
        godark_df.to_csv(godark_event_path, index=False)
        godark_event_breakdown_path = out_dir / "godark_event_error_breakdown.csv"
        godark_breakdown_df = pd.DataFrame(godark_event_breakdown(godark_rows))
        if not godark_breakdown_df.empty:
            godark_breakdown_df = godark_breakdown_df.sort_values(
                ["fp", "fn", "event_kind_family"],
                ascending=[False, False, True],
            )
        godark_breakdown_df.to_csv(godark_event_breakdown_path, index=False)

    tx_extra = None
    tx_extra_path = None
    if task_name == "transshipment":
        tx_extra = predict_transshipment_tabular_and_hybrid(
            model_path=model_path.parent / "transshipment_tabular.joblib",
            X=X[test_idx],
            y=y_test,
            groups=g_test,
            label_map=label_map,
            rule_features=(rule_features[test_idx] if rule_features is not None else None),
            rule_cols=rule_cols,
        )
        if tx_extra is not None:
            tx_rows = []
            for i, gid in enumerate(tx_extra["groups"].astype(str).tolist()):
                yt = int(tx_extra["y_true"][i])
                pt = int(tx_extra["pred_tabular"][i])
                ph = int(tx_extra["pred_hybrid"][i])
                tx_rows.append(
                    {
                        "event_id": gid,
                        "true_id": yt,
                        "true_label": label_map.get(yt, str(yt)),
                        "tabular_pred_id": pt,
                        "tabular_pred_label": label_map.get(pt, str(pt)),
                        "hybrid_pred_id": ph,
                        "hybrid_pred_label": label_map.get(ph, str(ph)),
                        "tabular_confidence": float(tx_extra["tabular_confidence"][i]),
                        "rule_used": int(tx_extra["rule_used"][i]),
                        "rule_risk": float(tx_extra["rule_risk"][i]),
                        "rule_encounter": float(tx_extra["rule_encounter"][i]),
                        "rule_loitering": float(tx_extra["rule_loitering"][i]),
                    }
                )
            tx_extra_path = out_dir / "per_event_predictions_hybrid.csv"
            pd.DataFrame(tx_rows).sort_values(["true_label", "rule_risk"], ascending=[True, False]).to_csv(tx_extra_path, index=False)

    if task_name == "transshipment":
        sus_path = None
    else:
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
        "metrics_godark_event": godark_event_metrics,
        "metrics_tabular": (tx_extra.get("metrics_tabular") if tx_extra is not None else None),
        "metrics_hybrid": (tx_extra.get("metrics_hybrid") if tx_extra is not None else None),
        "metrics": (m_seq if metric_scope == "sequence" else m_v),
        "test_sequences": int(len(y_np)),
        "test_vessels": int(len(v2idx)),
        "test_events": int(len(v2idx)) if task_name == "transshipment" else None,
        "prediction_table": str(pv_path),
        "godark_event_prediction_table": (str(godark_event_path) if godark_event_path is not None else None),
        "godark_event_error_breakdown_table": (
            str(godark_event_breakdown_path) if godark_event_breakdown_path is not None else None
        ),
        "godark_event_decision": (
            {
                "prob_threshold": float(godark_prob_threshold),
                "min_positive_windows": int(godark_min_windows),
                "min_positive_ratio": float(godark_min_ratio),
                "short_min_positive_ratio": float(godark_short_min_ratio),
                "source": (
                    "cli_override"
                    if (
                        godark_event_prob_threshold is not None
                        or godark_event_min_positive_windows is not None
                        or godark_event_min_positive_ratio is not None
                        or godark_event_short_min_positive_ratio is not None
                    )
                    else "checkpoint_or_default"
                ),
            }
            if task_name == "godark"
            else None
        ),
        "hybrid_prediction_table": (str(tx_extra_path) if tx_extra_path is not None else None),
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
        if godark_event_metrics is not None:
            print(
                f"[eval][GODARK-EVENT] precision={godark_event_metrics['event_precision']:.4f} "
                f"recall={godark_event_metrics['event_recall']:.4f} "
                f"f1={godark_event_metrics['event_f1']:.4f} "
                f"score={godark_event_metrics['godark_score']:.4f} "
                f"thr={godark_prob_threshold:.2f} min_win={godark_min_windows}"
            )
    else:
        print(f"[eval][VESSEL] acc={m_v['accuracy']:.4f} macro_f1={m_v['macro_f1']:.4f} bal_acc={m_v['balanced_acc']:.4f}")
        print(f"[eval][SEQ-AUX] acc={m_seq['accuracy']:.4f} macro_f1={m_seq['macro_f1']:.4f} bal_acc={m_seq['balanced_acc']:.4f}")
