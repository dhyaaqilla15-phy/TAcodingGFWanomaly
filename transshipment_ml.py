from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np

from agg_utils import confusion_matrix_np, metrics_from_cm


RULE_THRESHOLD = 0.90
NORMAL_RULE_THRESHOLD = 0.20


def _mode_label(y: np.ndarray, num_classes: int) -> int:
    y = np.asarray(y, dtype=np.int64)
    pos = y[y > 0]
    if pos.size:
        return int(np.bincount(pos, minlength=num_classes).argmax())
    return int(np.bincount(y, minlength=num_classes).argmax())


def _feature_index(feature_cols: List[str], name: str) -> int | None:
    try:
        return feature_cols.index(name)
    except ValueError:
        return None


def build_event_feature_table(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    label_map: Dict[int, str],
    feature_cols: List[str],
    rule_features: np.ndarray | None = None,
    rule_cols: List[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, np.ndarray]]:
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups).astype(str)
    R = None if rule_features is None else np.asarray(rule_features, dtype=np.float32)
    rule_cols = list(rule_cols or [])
    num_classes = int(len(label_map))
    F = int(X.shape[-1])

    valid_idx = _feature_index(feature_cols, "valid_point")
    risk_idx = _feature_index(feature_cols, "risk_score")
    enc_idx = _feature_index(feature_cols, "encounter_rule_score")
    loi_idx = _feature_index(feature_cols, "loitering_rule_score")

    base_names = feature_cols if feature_cols and len(feature_cols) == F else [f"f{i}" for i in range(F)]
    col_names = [
        f"{name}_{stat}"
        for stat in ("mean", "std", "min", "max")
        for name in base_names
    ]
    col_names.extend(["n_windows", "n_points"])

    rows = []
    y_rows = []
    gids = []
    rule_risk = []
    rule_enc = []
    rule_loi = []

    for gid in np.unique(groups):
        idx = np.where(groups == gid)[0]
        arr = X[idx].reshape(-1, F)
        rule_arr = None
        if R is not None and R.shape[0] == X.shape[0] and R.shape[1] == X.shape[1]:
            rule_arr = R[idx].reshape(-1, R.shape[-1])
        if valid_idx is not None:
            valid = arr[:, valid_idx] > 0.5
            if valid.any():
                if rule_arr is not None and len(rule_arr) == len(arr):
                    rule_arr = rule_arr[valid]
                arr = arr[valid]
        if arr.size == 0:
            arr = X[idx].reshape(-1, F)

        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        mn = np.nanmin(arr, axis=0)
        mx = np.nanmax(arr, axis=0)
        feat = np.concatenate(
            [
                np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0),
                np.nan_to_num(std, nan=0.0, posinf=0.0, neginf=0.0),
                np.nan_to_num(mn, nan=0.0, posinf=0.0, neginf=0.0),
                np.nan_to_num(mx, nan=0.0, posinf=0.0, neginf=0.0),
                np.array([float(len(idx)), float(len(arr))], dtype=np.float32),
            ]
        )
        rows.append(feat.astype(np.float32))
        y_rows.append(_mode_label(y[idx], num_classes))
        gids.append(str(gid))
        if rule_arr is not None and rule_arr.size and rule_cols:
            r_risk_idx = _feature_index(rule_cols, "risk_score")
            r_enc_idx = _feature_index(rule_cols, "encounter_rule_score")
            r_loi_idx = _feature_index(rule_cols, "loitering_rule_score")
            rule_risk.append(float(np.nanmax(rule_arr[:, r_risk_idx])) if r_risk_idx is not None else 0.0)
            rule_enc.append(float(np.nanmax(rule_arr[:, r_enc_idx])) if r_enc_idx is not None else 0.0)
            rule_loi.append(float(np.nanmax(rule_arr[:, r_loi_idx])) if r_loi_idx is not None else 0.0)
        else:
            rule_risk.append(float(np.nanmax(arr[:, risk_idx])) if risk_idx is not None else 0.0)
            rule_enc.append(float(np.nanmax(arr[:, enc_idx])) if enc_idx is not None else 0.0)
            rule_loi.append(float(np.nanmax(arr[:, loi_idx])) if loi_idx is not None else 0.0)

    meta = {
        "risk_score": np.asarray(rule_risk, dtype=np.float32),
        "encounter_rule_score": np.asarray(rule_enc, dtype=np.float32),
        "loitering_rule_score": np.asarray(rule_loi, dtype=np.float32),
    }
    return (
        np.vstack(rows).astype(np.float32) if rows else np.zeros((0, F * 4 + 2), dtype=np.float32),
        np.asarray(y_rows, dtype=np.int64),
        np.asarray(gids, dtype=object),
        col_names,
        meta,
    )


def train_transshipment_tabular_baseline(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    label_map: Dict[int, str],
    feature_cols: List[str],
    out_path: Path,
    rule_features: np.ndarray | None = None,
    rule_cols: List[str] | None = None,
    random_state: int = 42,
) -> bool:
    from sklearn.ensemble import RandomForestClassifier

    Xev, yev, _, col_names, _ = build_event_feature_table(
        X, y, groups, label_map, feature_cols, rule_features=rule_features, rule_cols=rule_cols
    )
    if Xev.shape[0] < 3 or np.unique(yev).size < 2:
        return False

    clf = RandomForestClassifier(
        n_estimators=350,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=int(random_state),
        n_jobs=-1,
    )
    clf.fit(Xev, yev)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": clf,
            "label_map": dict(label_map),
            "feature_cols": list(feature_cols),
            "rule_cols": list(rule_cols or []),
            "event_feature_cols": col_names,
            "rule_threshold": RULE_THRESHOLD,
            "normal_rule_threshold": NORMAL_RULE_THRESHOLD,
        },
        out_path,
    )
    return True


def _rule_pred(
    risk: np.ndarray,
    enc: np.ndarray,
    loi: np.ndarray,
    label_map: Dict[int, str],
    rule_threshold: float = RULE_THRESHOLD,
    normal_threshold: float = NORMAL_RULE_THRESHOLD,
) -> np.ndarray:
    labels = {int(k): str(v).strip().lower() for k, v in label_map.items()}
    n = len(risk)
    pred = np.full((n,), -1, dtype=np.int64)

    if set(labels.values()) == {"normal", "encounter", "loitering"}:
        both = np.column_stack([enc, loi])
        best = np.argmax(both, axis=1) + 1
        score = both.max(axis=1)
        pred[score >= float(rule_threshold)] = best[score >= float(rule_threshold)]
    elif "encounter" in labels.values():
        pred[enc >= float(rule_threshold)] = 1
    elif "loitering" in labels.values():
        pred[loi >= float(rule_threshold)] = 1
    else:
        pred[risk >= float(rule_threshold)] = 1

    pred[risk <= float(normal_threshold)] = 0
    return pred


def predict_transshipment_tabular_and_hybrid(
    model_path: Path,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    label_map: Dict[int, str],
    rule_features: np.ndarray | None = None,
    rule_cols: List[str] | None = None,
) -> dict | None:
    model_path = Path(model_path)
    if not model_path.exists():
        return None
    bundle = joblib.load(model_path)
    feature_cols = list(bundle.get("feature_cols", []))
    if rule_cols is None:
        rule_cols = list(bundle.get("rule_cols", []))
    Xev, yev, gids, _, meta = build_event_feature_table(
        X, y, groups, label_map, feature_cols, rule_features=rule_features, rule_cols=rule_cols
    )
    if Xev.size == 0:
        return None

    clf = bundle["model"]
    pred_tab = clf.predict(Xev).astype(np.int64)
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(Xev)
        conf_tab = proba.max(axis=1)
    else:
        conf_tab = np.ones(len(pred_tab), dtype=np.float32)

    rule_raw = _rule_pred(
        risk=meta["risk_score"],
        enc=meta["encounter_rule_score"],
        loi=meta["loitering_rule_score"],
        label_map=label_map,
        rule_threshold=float(bundle.get("rule_threshold", RULE_THRESHOLD)),
        normal_threshold=float(bundle.get("normal_rule_threshold", NORMAL_RULE_THRESHOLD)),
    )
    pred_hybrid = pred_tab.copy()
    use_rule = rule_raw >= 0
    pred_hybrid[use_rule] = rule_raw[use_rule]

    num_classes = int(len(label_map))
    m_tab = metrics_from_cm(confusion_matrix_np(yev, pred_tab, num_classes))
    m_hybrid = metrics_from_cm(confusion_matrix_np(yev, pred_hybrid, num_classes))
    return {
        "groups": gids,
        "y_true": yev,
        "pred_tabular": pred_tab,
        "pred_hybrid": pred_hybrid,
        "tabular_confidence": conf_tab,
        "rule_used": use_rule.astype(np.int64),
        "rule_risk": meta["risk_score"],
        "rule_encounter": meta["encounter_rule_score"],
        "rule_loitering": meta["loitering_rule_score"],
        "metrics_tabular": m_tab,
        "metrics_hybrid": m_hybrid,
    }
