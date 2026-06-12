from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np


DEFAULT_GODARK_EVENT_PROB_THRESHOLD = 0.80
DEFAULT_GODARK_EVENT_MIN_POSITIVE_WINDOWS = 2
DEFAULT_GODARK_EVENT_MIN_POSITIVE_RATIO = 0.0
DEFAULT_GODARK_EVENT_MIN_RECALL = 0.70
DEFAULT_GODARK_EVENT_PROB_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
DEFAULT_GODARK_EVENT_MIN_WINDOWS_GRID = (1, 2, 3, 5, 8, 10)
DEFAULT_GODARK_EVENT_MIN_RATIO_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
DEFAULT_GODARK_EVENT_SHORT_MIN_RATIO = 0.85


def binary_prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float((2.0 * precision * recall) / max(precision + recall, 1e-12))
    return {"precision": precision, "recall": recall, "f1": f1}


def event_kind_family(kind: str) -> str:
    s = str(kind or "unknown")
    if s.startswith("hard_negative_gap"):
        return "hard_negative_gap"
    if s.startswith("hard_negative_feature"):
        return "hard_negative_feature"
    if s.startswith("positive_event"):
        return "positive_event"
    if s.startswith("normal_random"):
        return "normal_random"
    return s.split("::", 1)[0] if "::" in s else s


def _mode_text(values: np.ndarray, default: str = "unknown") -> str:
    if values.size == 0:
        return default
    vals, cnt = np.unique(values.astype(str), return_counts=True)
    return str(vals[int(np.argmax(cnt))]) if vals.size else default


def _error_type(true_event: bool, pred_event: bool) -> str:
    if true_event and pred_event:
        return "TP"
    if true_event and not pred_event:
        return "FN"
    if not true_event and pred_event:
        return "FP"
    return "TN"


def _summarize_events_for_threshold(
    event_ids: np.ndarray,
    kinds: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    prob_threshold: float,
) -> Dict[str, np.ndarray]:
    event_ids = np.asarray(event_ids).astype(str)
    kinds = np.asarray(kinds).astype(str)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    pos_prob = probs[:, 1] if probs.ndim == 2 and probs.shape[1] > 1 else y_pred.astype(float)
    prob_threshold = float(prob_threshold)

    uniq = np.unique(event_ids)
    out = {
        "event_id": uniq.astype(object),
        "event_kind": np.empty((len(uniq),), dtype=object),
        "event_kind_family": np.empty((len(uniq),), dtype=object),
        "true_event": np.zeros((len(uniq),), dtype=bool),
        "n_windows": np.zeros((len(uniq),), dtype=np.int64),
        "true_positive_windows": np.zeros((len(uniq),), dtype=np.int64),
        "pred_positive_windows": np.zeros((len(uniq),), dtype=np.int64),
        "windows_over_threshold": np.zeros((len(uniq),), dtype=np.int64),
        "positive_window_ratio": np.zeros((len(uniq),), dtype=np.float64),
        "argmax_positive_window_ratio": np.zeros((len(uniq),), dtype=np.float64),
        "max_go_dark_probability": np.zeros((len(uniq),), dtype=np.float64),
        "mean_go_dark_probability": np.zeros((len(uniq),), dtype=np.float64),
    }

    for i, eid in enumerate(uniq):
        idx = np.where(event_ids == eid)[0]
        n_windows = int(len(idx))
        kind = _mode_text(kinds[idx])
        pred_positive_windows = int(np.sum(y_pred[idx] == 1))
        windows_over_threshold = int(np.sum(pos_prob[idx] >= prob_threshold)) if n_windows else 0
        out["event_kind"][i] = kind
        out["event_kind_family"][i] = event_kind_family(kind)
        out["true_event"][i] = bool(np.any(y_true[idx] == 1))
        out["n_windows"][i] = n_windows
        out["true_positive_windows"][i] = int(np.sum(y_true[idx] == 1))
        out["pred_positive_windows"][i] = pred_positive_windows
        out["windows_over_threshold"][i] = windows_over_threshold
        out["positive_window_ratio"][i] = float(windows_over_threshold / max(n_windows, 1))
        out["argmax_positive_window_ratio"][i] = float(pred_positive_windows / max(n_windows, 1))
        out["max_go_dark_probability"][i] = float(np.max(pos_prob[idx])) if n_windows else 0.0
        out["mean_go_dark_probability"][i] = float(np.mean(pos_prob[idx])) if n_windows else 0.0

    return out


def _event_metrics_from_pred(
    true_event: np.ndarray,
    pred_event: np.ndarray,
    prob_threshold: float,
    min_positive_windows: int,
    min_positive_ratio: float,
    short_min_positive_ratio: float = DEFAULT_GODARK_EVENT_SHORT_MIN_RATIO,
) -> Dict[str, float]:
    true_event = np.asarray(true_event, dtype=bool)
    pred_event = np.asarray(pred_event, dtype=bool)
    tp = int(np.sum(true_event & pred_event))
    fp = int(np.sum((~true_event) & pred_event))
    fn = int(np.sum(true_event & (~pred_event)))
    tn = int(np.sum((~true_event) & (~pred_event)))
    prf = binary_prf(tp, fp, fn)
    return {
        "event_precision": prf["precision"],
        "event_recall": prf["recall"],
        "event_f1": prf["f1"],
        "event_tp": tp,
        "event_fp": fp,
        "event_fn": fn,
        "event_tn": tn,
        "true_events": int(tp + fn),
        "predicted_events": int(tp + fp),
        "evaluated_event_groups": int(true_event.size),
        "godark_event_prob_threshold": float(prob_threshold),
        "godark_event_min_positive_windows": int(min_positive_windows),
        "godark_event_min_positive_ratio": float(min_positive_ratio),
        "godark_event_short_min_positive_ratio": float(short_min_positive_ratio),
    }


def _rows_from_event_summary(
    summary: Dict[str, np.ndarray],
    pred_event: np.ndarray,
    label_map: Dict[int, str],
    prob_threshold: float,
    min_positive_windows: int,
    min_positive_ratio: float,
    short_min_positive_ratio: float = DEFAULT_GODARK_EVENT_SHORT_MIN_RATIO,
) -> List[dict]:
    pred_event = np.asarray(pred_event, dtype=bool)
    true_event = np.asarray(summary["true_event"], dtype=bool)
    decision_rule = (
        f"max_prob>={float(prob_threshold):.3f} AND "
        f"((windows_over_threshold>={int(min_positive_windows)} AND "
        f"positive_window_ratio>={float(min_positive_ratio):.3f}) OR "
        f"(n_windows<{int(min_positive_windows)} AND "
        f"positive_window_ratio>={float(short_min_positive_ratio):.3f}))"
    )
    rows: List[dict] = []
    for i, eid in enumerate(summary["event_id"].tolist()):
        max_prob = float(summary["max_go_dark_probability"][i])
        windows_over_threshold = int(summary["windows_over_threshold"][i])
        ratio = float(summary["positive_window_ratio"][i])
        short_event = int(summary["n_windows"][i]) < int(min_positive_windows)
        short_event_rescue = bool(short_event and ratio >= float(short_min_positive_ratio))
        true_i = bool(true_event[i])
        pred_i = bool(pred_event[i])
        rows.append(
            {
                "event_id": str(eid),
                "event_kind": str(summary["event_kind"][i]),
                "event_kind_family": str(summary["event_kind_family"][i]),
                "true_event": int(true_i),
                "pred_event": int(pred_i),
                "error_type": _error_type(true_i, pred_i),
                "true_label": label_map.get(1, "go_dark") if true_i else label_map.get(0, "normal"),
                "pred_label": label_map.get(1, "go_dark") if pred_i else label_map.get(0, "normal"),
                "n_windows": int(summary["n_windows"][i]),
                "true_positive_windows": int(summary["true_positive_windows"][i]),
                "pred_positive_windows": int(summary["pred_positive_windows"][i]),
                "windows_over_threshold": windows_over_threshold,
                "positive_window_ratio": ratio,
                "argmax_positive_window_ratio": float(summary["argmax_positive_window_ratio"][i]),
                "max_go_dark_probability": max_prob,
                "mean_go_dark_probability": float(summary["mean_go_dark_probability"][i]),
                "godark_event_prob_threshold": float(prob_threshold),
                "godark_event_min_positive_windows": int(min_positive_windows),
                "godark_event_min_positive_ratio": float(min_positive_ratio),
                "godark_event_short_min_positive_ratio": float(short_min_positive_ratio),
                "max_prob_pass": int(max_prob >= float(prob_threshold)),
                "min_windows_pass": int(windows_over_threshold >= int(min_positive_windows)),
                "min_ratio_pass": int(ratio >= float(min_positive_ratio)),
                "short_event": int(short_event),
                "short_event_rescue": int(short_event_rescue),
                "event_decision_rule": decision_rule,
            }
        )
    return rows


def godark_score(seq_metrics: Dict[str, float], event_metrics: Dict[str, float]) -> float:
    return float(
        (0.20 * float(seq_metrics.get("macro_f1", 0.0)))
        + (0.15 * float(seq_metrics.get("balanced_acc", 0.0)))
        + (0.35 * float(event_metrics.get("event_f1", 0.0)))
        + (0.15 * float(event_metrics.get("event_precision", 0.0)))
        + (0.15 * float(event_metrics.get("event_recall", 0.0)))
    )


def godark_selection_key(
    seq_metrics: Dict[str, float],
    event_metrics: Dict[str, float],
    min_event_recall: float = DEFAULT_GODARK_EVENT_MIN_RECALL,
) -> Tuple[float, ...]:
    recall = float(event_metrics.get("event_recall", 0.0))
    event_f1 = float(event_metrics.get("event_f1", 0.0))
    event_precision = float(event_metrics.get("event_precision", 0.0))
    macro_f1 = float(seq_metrics.get("macro_f1", 0.0))
    balanced_acc = float(seq_metrics.get("balanced_acc", 0.0))
    score = godark_score(seq_metrics, event_metrics)
    recall_floor_met = 1.0 if recall >= float(min_event_recall) else 0.0
    if recall_floor_met > 0.0:
        return (1.0, score, event_f1, event_precision, macro_f1, balanced_acc, recall)
    return (0.0, recall, score, event_f1, event_precision, macro_f1, balanced_acc)


def godark_event_report(
    event_ids: np.ndarray,
    kinds: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    label_map: Dict[int, str],
    prob_threshold: float = DEFAULT_GODARK_EVENT_PROB_THRESHOLD,
    min_positive_windows: int = DEFAULT_GODARK_EVENT_MIN_POSITIVE_WINDOWS,
    min_positive_ratio: float = DEFAULT_GODARK_EVENT_MIN_POSITIVE_RATIO,
    short_min_positive_ratio: float = DEFAULT_GODARK_EVENT_SHORT_MIN_RATIO,
) -> tuple[dict, List[dict]]:
    prob_threshold = float(prob_threshold)
    min_positive_windows = int(max(1, min_positive_windows))
    min_positive_ratio = float(max(0.0, min_positive_ratio))
    short_min_positive_ratio = float(max(0.0, short_min_positive_ratio))
    summary = _summarize_events_for_threshold(event_ids, kinds, y_true, y_pred, probs, prob_threshold)
    strict_event = (
        (summary["windows_over_threshold"] >= min_positive_windows)
        & (summary["positive_window_ratio"] >= min_positive_ratio)
    )
    short_event_rescue = (
        (summary["n_windows"] < min_positive_windows)
        & (summary["positive_window_ratio"] >= short_min_positive_ratio)
    )
    pred_event = (
        (summary["max_go_dark_probability"] >= prob_threshold)
        & (strict_event | short_event_rescue)
    )
    metrics = _event_metrics_from_pred(
        true_event=summary["true_event"],
        pred_event=pred_event,
        prob_threshold=prob_threshold,
        min_positive_windows=min_positive_windows,
        min_positive_ratio=min_positive_ratio,
        short_min_positive_ratio=short_min_positive_ratio,
    )
    rows = _rows_from_event_summary(
        summary=summary,
        pred_event=pred_event,
        label_map=label_map,
        prob_threshold=prob_threshold,
        min_positive_windows=min_positive_windows,
        min_positive_ratio=min_positive_ratio,
        short_min_positive_ratio=short_min_positive_ratio,
    )
    return metrics, rows


def godark_event_breakdown(rows: List[dict]) -> List[dict]:
    if not rows:
        return []

    families = sorted({str(r.get("event_kind_family", event_kind_family(r.get("event_kind", "")))) for r in rows})
    out: List[dict] = []
    for family in families:
        sub = [r for r in rows if str(r.get("event_kind_family", event_kind_family(r.get("event_kind", "")))) == family]
        tp = int(sum(r.get("error_type") == "TP" for r in sub))
        fp = int(sum(r.get("error_type") == "FP" for r in sub))
        fn = int(sum(r.get("error_type") == "FN" for r in sub))
        tn = int(sum(r.get("error_type") == "TN" for r in sub))
        prf = binary_prf(tp, fp, fn)
        out.append(
            {
                "event_kind_family": family,
                "n_events": int(len(sub)),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": prf["precision"],
                "recall": prf["recall"],
                "f1": prf["f1"],
                "mean_max_go_dark_probability": float(np.mean([float(r.get("max_go_dark_probability", 0.0)) for r in sub])),
                "mean_go_dark_probability": float(np.mean([float(r.get("mean_go_dark_probability", 0.0)) for r in sub])),
                "mean_windows_over_threshold": float(np.mean([float(r.get("windows_over_threshold", 0.0)) for r in sub])),
                "mean_positive_window_ratio": float(np.mean([float(r.get("positive_window_ratio", 0.0)) for r in sub])),
            }
        )
    return out


def pick_best_godark_event_setting(
    event_ids: np.ndarray,
    kinds: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    label_map: Dict[int, str],
    seq_metrics: Dict[str, float],
    prob_thresholds: Iterable[float] = DEFAULT_GODARK_EVENT_PROB_THRESHOLDS,
    min_windows_grid: Iterable[int] = DEFAULT_GODARK_EVENT_MIN_WINDOWS_GRID,
    min_positive_ratio: float = DEFAULT_GODARK_EVENT_MIN_POSITIVE_RATIO,
    min_positive_ratios: Iterable[float] | None = None,
    short_min_positive_ratio: float = DEFAULT_GODARK_EVENT_SHORT_MIN_RATIO,
    min_event_recall: float = DEFAULT_GODARK_EVENT_MIN_RECALL,
) -> tuple[dict, List[dict], Tuple[float, ...]]:
    best_metrics = None
    best_key = None
    best_summary = None
    best_pred_event = None
    ratio_grid = (
        [float(v) for v in min_positive_ratios]
        if min_positive_ratios is not None
        else [float(min_positive_ratio)]
    )

    for prob_threshold in prob_thresholds:
        prob_threshold = float(prob_threshold)
        summary = _summarize_events_for_threshold(event_ids, kinds, y_true, y_pred, probs, prob_threshold)
        true_event = summary["true_event"]
        max_prob_pass = summary["max_go_dark_probability"] >= prob_threshold
        for min_windows in min_windows_grid:
            min_windows = int(max(1, min_windows))
            min_windows_pass = summary["windows_over_threshold"] >= min_windows
            for ratio in ratio_grid:
                ratio = float(max(0.0, ratio))
                short_rescue = (
                    (summary["n_windows"] < min_windows)
                    & (summary["positive_window_ratio"] >= float(short_min_positive_ratio))
                )
                pred_event = max_prob_pass & ((min_windows_pass & (summary["positive_window_ratio"] >= ratio)) | short_rescue)
                metrics = _event_metrics_from_pred(
                    true_event=true_event,
                    pred_event=pred_event,
                    prob_threshold=float(prob_threshold),
                    min_positive_windows=int(min_windows),
                    min_positive_ratio=float(ratio),
                    short_min_positive_ratio=float(short_min_positive_ratio),
                )
                score = godark_score(seq_metrics, metrics)
                metrics = dict(metrics)
                metrics["godark_score"] = score
                metrics["event_recall_floor"] = float(min_event_recall)
                metrics["event_recall_floor_met"] = int(float(metrics["event_recall"]) >= float(min_event_recall))
                key = godark_selection_key(seq_metrics, metrics, min_event_recall=min_event_recall)
                if best_key is None or key > best_key:
                    best_metrics = metrics
                    best_key = key
                    best_summary = summary
                    best_pred_event = pred_event.copy()

    best_rows = []
    if best_metrics is not None and best_summary is not None and best_pred_event is not None:
        best_rows = _rows_from_event_summary(
            summary=best_summary,
            pred_event=best_pred_event,
            label_map=label_map,
            prob_threshold=float(best_metrics["godark_event_prob_threshold"]),
            min_positive_windows=int(best_metrics["godark_event_min_positive_windows"]),
            min_positive_ratio=float(best_metrics["godark_event_min_positive_ratio"]),
            short_min_positive_ratio=float(best_metrics.get("godark_event_short_min_positive_ratio", short_min_positive_ratio)),
        )

    return dict(best_metrics or {}), list(best_rows), tuple(best_key or ())
