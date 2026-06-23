from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from agg_utils import confusion_matrix_np, metrics_from_cm
from godark_event import godark_event_report, godark_score, godark_selection_key


DEFAULT_GODARK_HYBRID_BLEND = 0.50


def build_godark_context_features(X: np.ndarray) -> np.ndarray:
    """Summarize only observable pre/gap/post sequence information."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3 or X.shape[1] < 2:
        raise ValueError(f"Expected Go-Dark X with shape (N,T,F); got {X.shape}.")
    center = max(1, int(X.shape[1] // 2))
    pre = X[:, :center, :]
    post = X[:, center:, :]
    return np.concatenate(
        [
            X[:, center, :],
            pre.mean(axis=1),
            post.mean(axis=1),
            pre.std(axis=1),
            post.std(axis=1),
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def fit_godark_hardnegative_hybrid(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    neural_val_positive_prob: np.ndarray,
    event_ids_val: np.ndarray,
    kinds_val: np.ndarray,
    label_map: Dict[int, str],
    out_path: Path,
    random_state: int,
    min_event_recall: float,
    min_event_precision: float,
    source_train: np.ndarray | None = None,
    blend_weight: float = DEFAULT_GODARK_HYBRID_BLEND,
    thresholds: Iterable[float] | None = None,
) -> dict:
    """Fit a hard-negative context model and select its hybrid threshold on val."""
    y_train_arr = np.asarray(y_train, dtype=np.int64)
    fit_sample_weight = None
    source_balancing_used = False
    if source_train is not None:
        source_arr = np.asarray(source_train).astype(str)
        if source_arr.shape[0] != y_train_arr.shape[0]:
            raise ValueError("source_train length must match y_train.")
        joint = np.array(
            [f"{source}::{int(label)}" for source, label in zip(source_arr, y_train_arr)],
            dtype=object,
        )
        values, counts = np.unique(joint, return_counts=True)
        inverse_count = {
            str(value): 1.0 / float(count)
            for value, count in zip(values.tolist(), counts.tolist())
        }
        fit_sample_weight = np.array(
            [inverse_count[str(value)] for value in joint], dtype=np.float64
        )
        source_balancing_used = True

    estimator = ExtraTreesClassifier(
        n_estimators=1000,
        min_samples_leaf=3,
        max_features=0.70,
        class_weight=None if source_balancing_used else "balanced",
        random_state=int(random_state),
        n_jobs=-1,
    )
    estimator.fit(
        build_godark_context_features(X_train),
        y_train_arr,
        sample_weight=fit_sample_weight,
    )

    context_prob = estimator.predict_proba(build_godark_context_features(X_val))[:, 1]
    blend = float(np.clip(blend_weight, 0.0, 1.0))
    neural_prob = np.asarray(neural_val_positive_prob, dtype=np.float64)
    hybrid_prob = (blend * neural_prob) + ((1.0 - blend) * context_prob)
    probs = np.column_stack([1.0 - hybrid_prob, hybrid_prob])

    threshold_grid = list(thresholds or np.arange(0.20, 0.8001, 0.025).tolist())
    best_metrics = None
    best_key = None
    for threshold in threshold_grid:
        threshold = float(threshold)
        pred = (hybrid_prob >= threshold).astype(np.int64)
        seq_metrics = metrics_from_cm(confusion_matrix_np(y_val, pred, 2))
        event_metrics, _ = godark_event_report(
            event_ids=np.asarray(event_ids_val, dtype=object),
            kinds=np.asarray(kinds_val, dtype=object),
            y_true=np.asarray(y_val, dtype=np.int64),
            y_pred=pred,
            probs=probs,
            label_map=label_map,
            prob_threshold=threshold,
            mean_prob_threshold=0.0,
            min_positive_windows=1,
            min_positive_ratio=1.0,
            use_short_rescue=False,
        )
        event_metrics = dict(event_metrics)
        event_metrics["godark_score"] = godark_score(seq_metrics, event_metrics)
        key = godark_selection_key(
            seq_metrics,
            event_metrics,
            min_event_recall=float(min_event_recall),
            min_event_precision=float(min_event_precision),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_metrics = {**seq_metrics, **event_metrics}

    artifact = {
        "estimator": estimator,
        "blend_weight_neural": blend,
        "prob_threshold": float(best_metrics["godark_event_prob_threshold"]),
        "validation_metrics": dict(best_metrics),
        "feature_builder": "center_pre_post_mean_std_v1",
        "random_state": int(random_state),
        "source_balancing_used": bool(source_balancing_used),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out_path)
    return artifact


def apply_godark_hardnegative_hybrid(
    artifact_path: Path,
    X: np.ndarray,
    neural_positive_prob: np.ndarray,
) -> tuple[np.ndarray, dict]:
    artifact = joblib.load(Path(artifact_path))
    estimator = artifact["estimator"]
    context_prob = estimator.predict_proba(build_godark_context_features(X))[:, 1]
    blend = float(artifact.get("blend_weight_neural", DEFAULT_GODARK_HYBRID_BLEND))
    neural_prob = np.asarray(neural_positive_prob, dtype=np.float64)
    hybrid_prob = (blend * neural_prob) + ((1.0 - blend) * context_prob)
    return np.clip(hybrid_prob, 0.0, 1.0), artifact
