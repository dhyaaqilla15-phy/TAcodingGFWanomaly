# eval.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
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
    per_class_metrics_from_cm,
)
from transshipment_ml import predict_transshipment_tabular_and_hybrid
from godark_event import (
    DEFAULT_GODARK_EVENT_MEAN_PROB_THRESHOLD,
    DEFAULT_GODARK_EVENT_MIN_PRECISION,
    DEFAULT_GODARK_EVENT_MIN_POSITIVE_RATIO,
    DEFAULT_GODARK_EVENT_MIN_POSITIVE_WINDOWS,
    DEFAULT_GODARK_EVENT_PROB_THRESHOLD,
    DEFAULT_GODARK_EVENT_SHORT_MIN_RATIO,
    DEFAULT_GODARK_EVENT_USE_SHORT_RESCUE,
    godark_event_breakdown,
    godark_event_report,
    godark_score,
)


GEAR_KNOWN_LIMITATION_LABELS = {"pole_and_line", "trollers"}


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


def _label_map_from_checkpoint(ckpt: dict) -> Dict[int, str] | None:
    raw = ckpt.get("label_map", None)
    if raw is None:
        return None
    if isinstance(raw, dict):
        return {int(k): str(v) for k, v in raw.items()}
    try:
        arr = np.asarray(raw, dtype=object)
        return {int(k): str(v) for k, v in arr.tolist()}
    except Exception:
        return None


def _remap_y_to_checkpoint_labels(
    y: np.ndarray,
    data_label_map: Dict[int, str],
    ckpt_label_map: Dict[int, str] | None,
) -> tuple[np.ndarray, Dict[int, str]]:
    if not ckpt_label_map:
        return y, data_label_map

    data_name_to_id = {str(v): int(k) for k, v in data_label_map.items()}
    ckpt_name_to_id = {str(v): int(k) for k, v in ckpt_label_map.items()}
    if data_name_to_id == ckpt_name_to_id:
        return y, ckpt_label_map

    missing = sorted(set(data_name_to_id) - set(ckpt_name_to_id))
    if missing:
        raise ValueError(
            "External data contains labels not present in checkpoint label_map: "
            + ", ".join(missing)
        )

    remap = {data_id: ckpt_name_to_id[name] for name, data_id in data_name_to_id.items()}
    y_new = np.asarray([remap[int(v)] for v in y], dtype=np.int64)
    print(f"[eval] remapped data labels to checkpoint label_map: {remap}")
    return y_new, ckpt_label_map


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


def save_spoofing_detection_png(
    cm: np.ndarray,
    out_path: Path,
    *,
    normalize: bool = False,
    attack_name: str | None = None,
) -> None:
    """Save an anomaly-first binary matrix: TP/FN on the first row."""
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=np.float64)
    if cm.shape != (2, 2):
        raise ValueError(f"Spoofing detection matrix must be 2x2; got {cm.shape}.")

    # Standard order is [normal, spoofing]. Reorder both axes so spoofing is
    # first: [[TP, FN], [FP, TN]].
    focus_counts = cm[np.ix_([1, 0], [1, 0])]
    if normalize:
        row_sum = focus_counts.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        show = focus_counts / row_sum
    else:
        show = focus_counts

    suffix = f" - {attack_name}" if attack_name else ""
    title = "Spoofing Detection Matrix" + suffix
    if normalize:
        title += " (row-normalized)"

    cell_names = np.array(
        [
            ["TP\nspoofing detected", "FN\nspoofing missed"],
            ["FP\nfalse alarm", "TN\nnormal rejected"],
        ],
        dtype=object,
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(show, interpolation="nearest", cmap="OrRd", vmin=0.0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xticks([0, 1], ["spoofing", "normal"])
    ax.set_yticks([0, 1], ["spoofing", "normal"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")

    threshold = (float(show.max()) if show.size else 0.0) * 0.55
    for i in range(2):
        for j in range(2):
            if normalize:
                value_text = f"{show[i, j]:.2f}\n(n={int(focus_counts[i, j])})"
            else:
                value_text = f"n={int(focus_counts[i, j])}"
            ax.text(
                j,
                i,
                f"{cell_names[i, j]}\n{value_text}",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="semibold" if i == 0 else "normal",
                color="white" if show[i, j] > threshold else "black",
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_spoofing_focus_reports(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    kinds: np.ndarray,
    out_dir: Path,
) -> Dict[str, object]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    kinds_str = np.char.lower(np.asarray(kinds).astype(str))
    normal_mask = np.isin(kinds_str, ["normal", "normal_random"])

    reports: Dict[str, object] = {}
    overall_cm = confusion_matrix_np(y_true, y_pred, 2)
    overall_raw = out_dir / "confusion_matrix_spoofing_focus.png"
    overall_norm = out_dir / "confusion_matrix_spoofing_focus_normalized.png"
    save_spoofing_detection_png(overall_cm, overall_raw)
    save_spoofing_detection_png(overall_cm, overall_norm, normalize=True)
    reports["overall"] = {
        "counts": str(overall_raw),
        "normalized": str(overall_norm),
    }

    attack_reports: Dict[str, Dict[str, str]] = {}
    attack_names = sorted(
        set(kinds_str.tolist()) - {"", "normal", "normal_random", "unknown"}
    )
    for attack in attack_names:
        subset = normal_mask | (kinds_str == attack)
        if not subset.any():
            continue
        attack_cm = confusion_matrix_np(y_true[subset], y_pred[subset], 2)
        safe_name = "".join(
            char if char.isalnum() or char in {"-", "_"} else "_"
            for char in attack
        )
        raw_path = out_dir / f"confusion_matrix_spoofing_{safe_name}.png"
        norm_path = out_dir / (
            f"confusion_matrix_spoofing_{safe_name}_normalized.png"
        )
        save_spoofing_detection_png(
            attack_cm,
            raw_path,
            attack_name=attack,
        )
        save_spoofing_detection_png(
            attack_cm,
            norm_path,
            normalize=True,
            attack_name=attack,
        )
        attack_reports[attack] = {
            "counts": str(raw_path),
            "normalized": str(norm_path),
        }
    reports["per_attack"] = attack_reports
    return reports


def _per_class_metric_rows(cm: np.ndarray, labels: List[str], scope: str) -> List[Dict[str, object]]:
    cls = per_class_metrics_from_cm(cm)
    known_limitations = set(_known_limitation_labels("gear", labels))
    rows: List[Dict[str, object]] = []
    for i, label in enumerate(labels):
        is_known_limitation = str(label) in known_limitations
        rows.append(
            {
                "scope": str(scope),
                "class_id": int(i),
                "class_label": str(label),
                "known_limitation": bool(is_known_limitation),
                "viable_class": bool(not is_known_limitation),
                "precision": float(cls["precision"][i]) if i < len(cls["precision"]) else 0.0,
                "recall": float(cls["recall"][i]) if i < len(cls["recall"]) else 0.0,
                "f1": float(cls["f1"][i]) if i < len(cls["f1"]) else 0.0,
                "support": int(cls["support"][i]) if i < len(cls["support"]) else 0,
                "pred_count": int(cls["pred_count"][i]) if i < len(cls["pred_count"]) else 0,
            }
        )
    return rows


def _known_limitation_labels(task_name: str, labels: List[str]) -> List[str]:
    if str(task_name) != "gear":
        return []
    available = {str(label) for label in labels}
    return [label for label in sorted(GEAR_KNOWN_LIMITATION_LABELS) if label in available]


def _filtered_metrics_from_cm(cm: np.ndarray, labels: List[str], excluded_labels: List[str]) -> Dict[str, float]:
    excluded = {str(label) for label in excluded_labels}
    keep_ids = [i for i, label in enumerate(labels) if str(label) not in excluded]
    if not keep_ids:
        return {"accuracy": 0.0, "macro_f1": 0.0, "balanced_acc": 0.0, "weighted_f1": 0.0, "num_classes": 0}

    cm_f = cm.astype(np.float64)
    cls = per_class_metrics_from_cm(cm_f)
    f1 = cls["f1"][keep_ids]
    recall = cls["recall"][keep_ids]
    support = cls["support"][keep_ids]
    tp = np.diag(cm_f)[keep_ids]

    return {
        "accuracy": float(tp.sum() / max(float(support.sum()), 1.0)),
        "macro_f1": float(f1.mean()) if len(f1) else 0.0,
        "balanced_acc": float(recall.mean()) if len(recall) else 0.0,
        "weighted_f1": float((f1 * support).sum() / max(float(support.sum()), 1.0)),
        "num_classes": int(len(keep_ids)),
    }


def _load_split_indices(out_dir: Path) -> Dict[str, np.ndarray] | None:
    p = out_dir / "split_indices.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    out = {"train_idx": d["train_idx"], "test_idx": d["test_idx"]}
    if "val_idx" in d.files:
        out["val_idx"] = d["val_idx"]
    return out


def _load_split_indices_for_eval(out_dir: Path, model_path: Path) -> Dict[str, np.ndarray] | None:
    for base in [out_dir, model_path.parent]:
        split_idx = _load_split_indices(base)
        if split_idx is not None:
            return split_idx
    return None


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
    split_random_state: int | None = None,
    eval_split: str = "test",
    godark_event_prob_threshold: float | None = None,
    godark_event_mean_prob_threshold: float | None = None,
    godark_event_min_positive_windows: int | None = None,
    godark_event_min_positive_ratio: float | None = None,
    godark_event_short_min_positive_ratio: float | None = None,
    godark_event_use_short_rescue: bool | None = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(model_path)

    X, y, groups, label_map = load_npz(Path(data_npz))
    rule_features, rule_cols = _load_rule_features(Path(data_npz))
    window_event_ids, window_kinds = _load_window_metadata(Path(data_npz))
    ckpt = _torch_load_compat(model_path, map_location="cpu")
    y, label_map = _remap_y_to_checkpoint_labels(
        y.astype(np.int64),
        data_label_map=label_map,
        ckpt_label_map=_label_map_from_checkpoint(ckpt),
    )
    num_classes = int(len(label_map))
    labels = [label_map.get(i, str(i)) for i in range(num_classes)]
    task_name = _task_name_from_label_map(label_map)
    if task_name == "spoofing":
        normalized_label_map = {
            int(key): str(value).strip().lower()
            for key, value in label_map.items()
        }
        if normalized_label_map != {0: "normal", 1: "spoofing"}:
            raise ValueError(
                "Spoofing evaluation requires class 1 to be the anomaly; "
                f"got {normalized_label_map}."
            )

    dev = pick_device(device)

    eval_split_seed = int(
        random_state
        if split_random_state is None
        else split_random_state
    )
    if split_random_state is None and "split_random_state" in ckpt:
        eval_split_seed = int(ckpt.get("split_random_state", eval_split_seed))

    eval_split_key = str(eval_split).strip().lower()
    if task_name == "gear" and eval_split_key == "all":
        present_ids = set(np.unique(y).astype(int).tolist())
        missing_labels = [
            labels[i]
            for i in range(num_classes)
            if i not in present_ids
        ]
        if missing_labels:
            raise ValueError(
                "External gear evaluation is missing classes after preprocessing: "
                + ", ".join(missing_labels)
                + ". Adjust gap/seq_len/filter settings before evaluating."
            )

    if eval_split_key == "all":
        split_lookup = {
            "all": np.arange(len(y), dtype=np.int64),
        }
    else:
        split_idx = _load_split_indices_for_eval(out_dir, model_path)
        if split_idx is None:
            split = group_train_val_test_split(
                X, y, groups,
                val_size=0.05,
                test_size=test_size,
                random_state=eval_split_seed,
                mixed_label_groups=(task_name == "spoofing"),
            )
            split_lookup = {
                "train": split.train_idx,
                "val": split.val_idx,
                "validation": split.val_idx,
                "test": split.test_idx,
                "all": np.arange(len(y), dtype=np.int64),
            }
        else:
            split_lookup = {
                "train": split_idx["train_idx"],
                "val": split_idx["val_idx"],
                "validation": split_idx["val_idx"],
                "test": split_idx["test_idx"],
                "all": np.arange(len(y), dtype=np.int64),
            }

    if eval_split_key not in split_lookup:
        raise ValueError("--eval_split must be one of: train, val, validation, test, all")
    test_idx = split_lookup[eval_split_key]
    eval_split_name = "val" if eval_split_key == "validation" else eval_split_key
    if len(test_idx) == 0:
        raise ValueError(f"Selected eval split '{eval_split_name}' is empty.")

    y_test = y[test_idx]
    g_test = groups[test_idx]
    event_ids_test = window_event_ids[test_idx] if len(window_event_ids) == len(y) else np.array([""] * len(test_idx), dtype=object)
    kinds_test = window_kinds[test_idx] if len(window_kinds) == len(y) else np.array(["unknown"] * len(test_idx), dtype=object)

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
    godark_mean_prob_threshold = float(
        DEFAULT_GODARK_EVENT_MEAN_PROB_THRESHOLD
        if godark_event_mean_prob_threshold is None
        else godark_event_mean_prob_threshold
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
    godark_min_precision = float(DEFAULT_GODARK_EVENT_MIN_PRECISION)
    godark_short_min_ratio = float(
        DEFAULT_GODARK_EVENT_SHORT_MIN_RATIO
        if godark_event_short_min_positive_ratio is None
        else godark_event_short_min_positive_ratio
    )
    godark_use_short_rescue = bool(
        DEFAULT_GODARK_EVENT_USE_SHORT_RESCUE
        if godark_event_use_short_rescue is None
        else godark_event_use_short_rescue
    )
    if task_name == "godark":
        godark_prob_threshold = float(
            ckpt.get("godark_event_prob_threshold", godark_prob_threshold)
            if godark_event_prob_threshold is None
            else godark_event_prob_threshold
        )
        godark_mean_prob_threshold = float(
            ckpt.get("godark_event_mean_prob_threshold", godark_mean_prob_threshold)
            if godark_event_mean_prob_threshold is None
            else godark_event_mean_prob_threshold
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
        godark_min_precision = float(ckpt.get("godark_event_min_precision", godark_min_precision))
        godark_short_min_ratio = float(
            ckpt.get("godark_event_short_min_positive_ratio", godark_short_min_ratio)
            if godark_event_short_min_positive_ratio is None
            else godark_event_short_min_positive_ratio
        )
        godark_use_short_rescue = bool(
            ckpt.get("godark_event_use_short_rescue", godark_use_short_rescue)
            if godark_event_use_short_rescue is None
            else godark_event_use_short_rescue
        )

    adj_logits = logits_np - (float(best_tau) * log_pi.reshape(1, -1))
    probs = softmax_np(adj_logits)
    confs = conf_score_from_probs(probs, mode=best_agg.conf_mode)
    pred_seq = np.argmax(probs, axis=1).astype(np.int64)

    binary_ranking_metrics = None
    if num_classes == 2 and np.unique(y_np).size == 2:
        positive_probs = probs[:, 1].astype(np.float64)
        binary_ranking_metrics = {
            "average_precision": float(
                average_precision_score(y_np, positive_probs)
            ),
            "roc_auc": float(roc_auc_score(y_np, positive_probs)),
        }

    cm_seq = confusion_matrix_np(y_np, pred_seq, num_classes)
    m_seq = metrics_from_cm(cm_seq)
    known_limitation_labels = _known_limitation_labels(task_name, labels)
    m_seq_viable = _filtered_metrics_from_cm(cm_seq, labels, known_limitation_labels)

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
    m_v_viable = _filtered_metrics_from_cm(cm_v, labels, known_limitation_labels)

    if task_name == "gear":
        cm_primary = cm_v
        save_confusion_png(cm_primary, labels, out_dir / "confusion_matrix.png", normalize=False)
        save_confusion_png(cm_primary, labels, out_dir / "confusion_matrix_normalized.png", normalize=True)
        for stale_name in [
            "confusion_matrix_sequence.png",
            "confusion_matrix_sequence_normalized.png",
            "confusion_matrix_vessel.png",
            "confusion_matrix_vessel_normalized.png",
        ]:
            stale_path = out_dir / stale_name
            if stale_path.exists():
                stale_path.unlink()
    else:
        cm_primary = cm_seq if metric_scope == "sequence" else cm_v
        if task_name == "spoofing":
            save_spoofing_detection_png(
                cm_primary,
                out_dir / "confusion_matrix.png",
            )
            save_spoofing_detection_png(
                cm_primary,
                out_dir / "confusion_matrix_normalized.png",
                normalize=True,
            )
            save_confusion_png(
                cm_primary,
                labels,
                out_dir / "confusion_matrix_standard.png",
                normalize=False,
            )
            save_confusion_png(
                cm_primary,
                labels,
                out_dir / "confusion_matrix_standard_normalized.png",
                normalize=True,
            )
        else:
            save_confusion_png(cm_primary, labels, out_dir / "confusion_matrix.png", normalize=False)
            save_confusion_png(cm_primary, labels, out_dir / "confusion_matrix_normalized.png", normalize=True)
        save_confusion_png(cm_seq, labels, out_dir / "confusion_matrix_sequence.png", normalize=False)
        save_confusion_png(cm_seq, labels, out_dir / "confusion_matrix_sequence_normalized.png", normalize=True)
        save_confusion_png(cm_v, labels, out_dir / "confusion_matrix_vessel.png", normalize=False)
        save_confusion_png(cm_v, labels, out_dir / "confusion_matrix_vessel_normalized.png", normalize=True)

    per_class_path = out_dir / "per_class_metrics.csv"
    per_class_rows = (
        _per_class_metric_rows(cm_seq, labels, "sequence")
        + _per_class_metric_rows(cm_v, labels, "vessel")
    )
    pd.DataFrame(per_class_rows).to_csv(per_class_path, index=False)

    spoofing_attack_path = None
    spoofing_sequence_path = None
    spoofing_scenario_path = None
    spoofing_scenario_metrics = None
    spoofing_focus_reports = None
    spoofing_attack_rows = []
    if task_name == "spoofing" and len(kinds_test) == len(y_np):
        kinds_str = np.asarray(kinds_test).astype(str)
        spoofing_focus_reports = save_spoofing_focus_reports(
            y_np,
            pred_seq,
            kinds_str,
            out_dir,
        )
        normal_mask = np.isin(
            np.char.lower(kinds_str),
            ["normal", "normal_random"],
        )
        attack_names = sorted(
            {
                str(kind).strip().lower()
                for kind in kinds_str.tolist()
                if str(kind).strip().lower()
                not in {"", "normal", "normal_random", "unknown"}
            }
        )
        context_required = {"replay", "meaconing", "ghost", "mirroring"}
        for attack in attack_names:
            attack_mask = np.char.lower(kinds_str) == attack
            subset = normal_mask | attack_mask
            if not subset.any():
                continue
            cm_attack = confusion_matrix_np(
                y_np[subset],
                pred_seq[subset],
                num_classes,
            )
            metrics_attack = metrics_from_cm(cm_attack)
            tp = int(cm_attack[1, 1]) if cm_attack.shape == (2, 2) else 0
            fp = int(cm_attack[0, 1]) if cm_attack.shape == (2, 2) else 0
            fn = int(cm_attack[1, 0]) if cm_attack.shape == (2, 2) else 0
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = (
                2.0 * precision * recall / max(precision + recall, 1e-12)
            )
            spoofing_attack_rows.append(
                {
                    "attack_type": attack,
                    "identifiability": (
                        "context_required"
                        if attack in context_required
                        else "single_window_kinematic"
                    ),
                    "positive_windows": int(
                        (attack_mask & (y_np == 1)).sum()
                    ),
                    "normal_windows": int((subset & (y_np == 0)).sum()),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "accuracy": float(metrics_attack["accuracy"]),
                    "balanced_acc": float(metrics_attack["balanced_acc"]),
                    "macro_f1": float(metrics_attack["macro_f1"]),
                }
            )
        spoofing_attack_path = out_dir / "spoofing_attack_metrics.csv"
        pd.DataFrame(spoofing_attack_rows).to_csv(
            spoofing_attack_path,
            index=False,
        )
        spoofing_sequence_path = out_dir / "spoofing_sequence_predictions.csv"
        pd.DataFrame(
            {
                "source_group": g_np.astype(str),
                "scenario_id": np.asarray(event_ids_test).astype(str),
                "attack_type": kinds_str,
                "true_id": y_np.astype(int),
                "pred_id": pred_seq.astype(int),
                "spoofing_probability": probs[:, 1].astype(float),
                "correct": (y_np == pred_seq),
            }
        ).to_csv(spoofing_sequence_path, index=False)

        scenario_rows = []
        scenario_ids_str = np.asarray(event_ids_test).astype(str)
        for scenario_id in np.unique(scenario_ids_str):
            idx = np.where(scenario_ids_str == scenario_id)[0]
            scenario_true = int(np.max(y_np[idx]))
            scenario_attack = str(
                pd.Series(kinds_str[idx]).value_counts().index[0]
            )
            scenario_probs = probs[idx, 1].astype(np.float64)
            top_count = max(1, int(np.ceil(len(idx) * 0.10)))
            top_mean_probability = float(
                np.mean(np.sort(scenario_probs)[-top_count:])
            )
            scenario_pred = int(top_mean_probability >= 0.50)
            scenario_rows.append(
                {
                    "scenario_id": scenario_id,
                    "source_group": str(g_np[idx[0]]),
                    "attack_type": scenario_attack,
                    "true_id": scenario_true,
                    "pred_id": scenario_pred,
                    "top10pct_mean_spoofing_probability": top_mean_probability,
                    "n_windows": int(len(idx)),
                    "correct": scenario_true == scenario_pred,
                }
            )
        scenario_df = pd.DataFrame(scenario_rows)
        spoofing_scenario_path = out_dir / "spoofing_scenario_predictions.csv"
        scenario_df.to_csv(spoofing_scenario_path, index=False)
        if not scenario_df.empty and scenario_df["true_id"].nunique() == 2:
            scenario_true = scenario_df["true_id"].to_numpy(dtype=np.int64)
            scenario_pred = scenario_df["pred_id"].to_numpy(dtype=np.int64)
            scenario_score = scenario_df[
                "top10pct_mean_spoofing_probability"
            ].to_numpy(dtype=np.float64)
            scenario_cm = confusion_matrix_np(
                scenario_true,
                scenario_pred,
                num_classes,
            )
            spoofing_scenario_metrics = {
                **metrics_from_cm(scenario_cm),
                "average_precision": float(
                    average_precision_score(scenario_true, scenario_score)
                ),
                "roc_auc": float(
                    roc_auc_score(scenario_true, scenario_score)
                ),
                "threshold": 0.50,
                "aggregation": "mean_top_10_percent_spoofing_probability",
                "num_scenarios": int(len(scenario_df)),
            }
            scenario_raw = out_dir / "confusion_matrix_spoofing_scenario.png"
            scenario_norm = out_dir / (
                "confusion_matrix_spoofing_scenario_normalized.png"
            )
            save_spoofing_detection_png(
                scenario_cm,
                scenario_raw,
                attack_name="scenario level",
            )
            save_spoofing_detection_png(
                scenario_cm,
                scenario_norm,
                normalize=True,
                attack_name="scenario level",
            )
            if spoofing_focus_reports is not None:
                spoofing_focus_reports["scenario"] = {
                    "counts": str(scenario_raw),
                    "normalized": str(scenario_norm),
                }

    pv_name = "per_event_predictions.csv" if task_name == "transshipment" else "per_vessel_predictions.csv"
    pv_path = out_dir / pv_name
    pv_df = pd.DataFrame(pv_rows).sort_values(["true_label", "confidence"], ascending=[True, False])
    pv_df.to_csv(pv_path, index=False)

    high_conf_wrong_threshold = 0.75
    wrong_high_conf_path = out_dir / "wrong_high_confidence_predictions.csv"
    if not pv_df.empty and {"true_id", "pred_id", "confidence"}.issubset(pv_df.columns):
        wrong_high_conf_df = pv_df[
            (pv_df["true_id"].astype(int) != pv_df["pred_id"].astype(int))
            & (pv_df["confidence"].astype(float) >= high_conf_wrong_threshold)
        ].sort_values("confidence", ascending=False)
    else:
        wrong_high_conf_df = pd.DataFrame(columns=pv_df.columns)
    wrong_high_conf_df.to_csv(wrong_high_conf_path, index=False)

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
            mean_prob_threshold=godark_mean_prob_threshold,
            min_positive_windows=godark_min_windows,
            min_positive_ratio=godark_min_ratio,
            short_min_positive_ratio=godark_short_min_ratio,
            use_short_rescue=godark_use_short_rescue,
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
        "data_npz": str(data_npz),
        "model_path": str(model_path),
        "task": task_name,
        "primary_metric_scope": metric_scope,
        "num_classes": num_classes,
        "labels": labels,
        "known_limitation_labels": known_limitation_labels,
        "metrics_seq": m_seq,
        "metrics_vessel": m_v,
        "metrics_seq_viable": m_seq_viable,
        "metrics_vessel_viable": m_v_viable,
        "metrics_godark_event": godark_event_metrics,
        "metrics_tabular": (tx_extra.get("metrics_tabular") if tx_extra is not None else None),
        "metrics_hybrid": (tx_extra.get("metrics_hybrid") if tx_extra is not None else None),
        "metrics": (m_seq if metric_scope == "sequence" else m_v),
        "test_sequences": int(len(y_np)),
        "test_vessels": int(len(v2idx)),
        "test_events": int(len(v2idx)) if task_name == "transshipment" else None,
        "prediction_table": str(pv_path),
        "per_class_metrics_table": str(per_class_path),
        "wrong_high_confidence_table": str(wrong_high_conf_path),
        "wrong_high_confidence_threshold": float(high_conf_wrong_threshold),
        "wrong_high_confidence_count": int(len(wrong_high_conf_df)),
        "spoofing_attack_metrics_table": (
            str(spoofing_attack_path)
            if spoofing_attack_path is not None
            else None
        ),
        "spoofing_attack_metrics": (
            spoofing_attack_rows if task_name == "spoofing" else None
        ),
        "spoofing_sequence_predictions_table": (
            str(spoofing_sequence_path)
            if spoofing_sequence_path is not None
            else None
        ),
        "spoofing_scenario_predictions_table": (
            str(spoofing_scenario_path)
            if spoofing_scenario_path is not None
            else None
        ),
        "spoofing_scenario_metrics": spoofing_scenario_metrics,
        "spoofing_focus_confusion_matrices": spoofing_focus_reports,
        "binary_ranking_metrics": binary_ranking_metrics,
        "godark_event_prediction_table": (str(godark_event_path) if godark_event_path is not None else None),
        "godark_event_error_breakdown_table": (
            str(godark_event_breakdown_path) if godark_event_breakdown_path is not None else None
        ),
        "godark_event_decision": (
            {
                "prob_threshold": float(godark_prob_threshold),
                "mean_prob_threshold": float(godark_mean_prob_threshold),
                "min_positive_windows": int(godark_min_windows),
                "min_positive_ratio": float(godark_min_ratio),
                "short_min_positive_ratio": float(godark_short_min_ratio),
                "use_short_rescue": bool(godark_use_short_rescue),
                "checkpoint_status": ckpt.get("checkpoint_status", None),
                "checkpoint_valid": ckpt.get("checkpoint_valid", None),
                "checkpoint_invalid_reason": ckpt.get("checkpoint_invalid_reason", None),
                "validation_min_precision": float(godark_min_precision),
                "source": (
                    "cli_override"
                    if (
                        godark_event_prob_threshold is not None
                        or godark_event_mean_prob_threshold is not None
                        or godark_event_min_positive_windows is not None
                        or godark_event_min_positive_ratio is not None
                        or godark_event_short_min_positive_ratio is not None
                        or godark_event_use_short_rescue is not None
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
        "random_state": int(random_state),
        "split_random_state": int(eval_split_seed),
        "train_random_state": (
            None if ckpt.get("train_random_state", None) is None else int(ckpt.get("train_random_state"))
        ),
        "deterministic": ckpt.get("deterministic", None),
        "logit_adjust_used": True,
        "tau": float(best_tau),
        "keep_frac": float(best_agg.keep_frac),
        "min_keep": int(best_agg.min_keep),
        "eval_split": eval_split_name,
        "test_tuning_used": bool(eval_split_name not in {"test", "all"}),
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
        if known_limitation_labels:
            print(
                f"[eval][VESSEL-VIABLE] acc={m_v_viable['accuracy']:.4f} "
                f"macro_f1={m_v_viable['macro_f1']:.4f} "
                f"bal_acc={m_v_viable['balanced_acc']:.4f} "
                f"excluded={known_limitation_labels}"
            )
        print(f"[eval][SEQ-AUX] acc={m_seq['accuracy']:.4f} macro_f1={m_seq['macro_f1']:.4f} bal_acc={m_seq['balanced_acc']:.4f}")
