from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parent
PROTOCOL = "synthetic_train_only_vessel_disjoint_external_real_v1"
VARIANTS = {"syn050": 50, "syn125": 125, "syn250": 250}


def env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name, "").strip()
    path = Path(raw) if raw else default
    return path if path.is_absolute() else (ROOT / path).resolve()


INTERNAL_NPZ = env_path(
    "TRANS_INTERNAL_NPZ",
    ROOT
    / "Outputs"
    / "transshipment_external01"
    / "data_internal_trainval"
    / "processed_transshipment_trainval.npz",
)
EXTERNAL_NPZ = env_path(
    "TRANS_EXTERNAL_NPZ",
    ROOT
    / "Outputs"
    / "transshipment_external01"
    / "data_external_test"
    / "processed_transshipment.npz",
)
TUNING_ROOT = env_path(
    "TRANS_TUNING_ROOT",
    ROOT / "Outputs" / "transshipment_tuning01_internal_oof",
)
TRAIN_ROOT = env_path(
    "TRANS_TRAIN_ROOT",
    ROOT / "Outputs" / "transshipment_training01_internal_oof",
)
FOLD_ROOT = TUNING_ROOT / "folds"
RUN_ROOT = TRAIN_ROOT / "runs"
CALIBRATOR_ROOT = TUNING_ROOT / "calibrators"
WINNER_PATH = TUNING_ROOT / "winner_internal_only.json"


def seeds() -> list[int]:
    values = [
        int(value.strip())
        for value in os.environ.get("TRANS_TUNING_SEEDS", "42,43,44").split(",")
        if value.strip()
    ]
    if len(values) < 3:
        raise ValueError("TRANS_TUNING_SEEDS must contain at least three seeds.")
    return values


def epochs() -> int:
    return int(os.environ.get("TRANS_TUNING_EPOCHS", "50"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], title: str) -> None:
    print("\n" + "=" * 76)
    print(f"[trans-oof] {title}")
    print("+ " + " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def _load_internal() -> dict[str, np.ndarray]:
    if not INTERNAL_NPZ.is_file():
        raise FileNotFoundError(
            f"Missing {INTERNAL_NPZ}. Run transshipment prepare first."
        )
    with np.load(INTERNAL_NPZ, allow_pickle=True) as data:
        required = {
            "X",
            "y",
            "groups",
            "window_is_synthetic",
            "window_mmsi_a",
            "window_mmsi_b",
            "window_source_labels",
            "window_kinds",
            "transshipment_data_protocol",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise RuntimeError(f"Internal NPZ missing metadata: {missing}")
        protocol = str(np.asarray(data["transshipment_data_protocol"]).item())
        if protocol != PROTOCOL:
            raise RuntimeError(f"Unexpected internal protocol: {protocol}")
        return {key: data[key] for key in data.files}


def _valid_mmsi(value: object) -> str | None:
    text = str(value).strip()
    return None if not text or text.lower() in {"none", "nan", "-1"} else text


def event_table(data: dict[str, np.ndarray]) -> pd.DataFrame:
    groups = data["groups"].astype(str)
    y = data["y"].astype(np.int64)
    synthetic = data["window_is_synthetic"].astype(np.int8)
    mmsi_a = data["window_mmsi_a"].astype(str)
    mmsi_b = data["window_mmsi_b"].astype(str)
    sources = data["window_source_labels"].astype(str)
    kinds = data["window_kinds"].astype(str)
    rows = []
    for event_id in np.unique(groups):
        idx = np.where(groups == event_id)[0]
        kind_raw = str(kinds[idx[0]]).lower()
        rows.append(
            {
                "event_id": str(event_id),
                "y": int(np.bincount(y[idx], minlength=2).argmax()),
                "synthetic": int(np.max(synthetic[idx])),
                "mmsi_a": str(mmsi_a[idx[0]]),
                "mmsi_b": str(mmsi_b[idx[0]]),
                "source_label": str(sources[idx[0]]),
                "event_kind": (
                    "encounter" if "encounter" in kind_raw else "loitering"
                ),
                "windows": int(len(idx)),
            }
        )
    return pd.DataFrame(rows)


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def _real_components(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    real = events[events["synthetic"] == 0].copy()
    union = UnionFind()
    for row in real.itertuples(index=False):
        vessels = [v for v in (_valid_mmsi(row.mmsi_a), _valid_mmsi(row.mmsi_b)) if v]
        for vessel in vessels:
            union.find(vessel)
        if len(vessels) == 2:
            union.union(vessels[0], vessels[1])
    real["component"] = [union.find(str(value)) for value in real["mmsi_a"]]

    key_names = [
        "negative_encounter",
        "positive_encounter",
        "negative_loitering",
        "positive_loitering",
    ]
    source_names = sorted(real["source_label"].unique().tolist())
    rows = []
    for component, part in real.groupby("component"):
        row: dict[str, object] = {
            "component": str(component),
            "events": int(len(part)),
            "windows": int(part["windows"].sum()),
        }
        for name in key_names:
            label_text, kind = name.split("_")
            target = 1 if label_text == "positive" else 0
            row[name] = int(np.sum((part["y"] == target) & (part["event_kind"] == kind)))
        for source in source_names:
            row[f"source::{source}"] = int(np.sum(part["source_label"] == source))
        rows.append(row)
    return real, pd.DataFrame(rows)


def _assign_components(components: pd.DataFrame, n_folds: int = 3) -> dict[str, int]:
    key_cols = [
        "events",
        "windows",
        "negative_encounter",
        "positive_encounter",
        "negative_loitering",
        "positive_loitering",
    ]
    source_cols = [column for column in components if column.startswith("source::")]
    cols = key_cols + source_cols
    values = components[cols].to_numpy(dtype=np.float64)
    totals = np.maximum(values.sum(axis=0), 1.0)
    target = totals / float(n_folds)
    order_base = np.argsort(-components["events"].to_numpy())
    best_score: float | None = None
    best_assignment: np.ndarray | None = None
    rng = np.random.RandomState(20260623)

    for attempt in range(3000):
        loads = np.zeros((n_folds, len(cols)), dtype=np.float64)
        assignment = np.full(len(components), -1, dtype=np.int64)
        order = order_base.copy()
        if attempt:
            tail = order[3:].copy()
            noise = rng.random(len(tail))
            tail = tail[np.lexsort((noise, -components.iloc[tail]["events"].to_numpy()))]
            order = np.concatenate([order[:3], tail])

        fold_order = rng.permutation(n_folds)
        for rank, idx in enumerate(order):
            if rank < n_folds:
                fold = int(fold_order[rank])
            else:
                candidates = []
                for fold_id in range(n_folds):
                    proposed = loads.copy()
                    proposed[fold_id] += values[idx]
                    relative = (proposed - target.reshape(1, -1)) / target.reshape(1, -1)
                    # Event/type balance dominates; source-pair balance is a
                    # secondary tie breaker because some pairs are rare.
                    score = float(np.mean(relative[:, : len(key_cols)] ** 2))
                    if source_cols:
                        score += 0.15 * float(np.mean(relative[:, len(key_cols) :] ** 2))
                    score += 1e-6 * float(rng.rand())
                    candidates.append((score, fold_id))
                fold = min(candidates)[1]
            assignment[idx] = fold
            loads[fold] += values[idx]

        relative = (loads - target.reshape(1, -1)) / target.reshape(1, -1)
        score = float(np.mean(relative[:, : len(key_cols)] ** 2))
        if source_cols:
            score += 0.15 * float(np.mean(relative[:, len(key_cols) :] ** 2))
        # Every fold must contain positive encounter and loitering evidence.
        for column in ("positive_encounter", "positive_loitering"):
            col = cols.index(column)
            score += 1000.0 * float(np.sum(loads[:, col] <= 0))
        if best_score is None or score < best_score:
            best_score, best_assignment = score, assignment.copy()

    if best_assignment is None:
        raise RuntimeError("Could not assign transshipment components to folds.")
    return {
        str(component): int(fold)
        for component, fold in zip(components["component"], best_assignment)
    }


def _stable_synthetic_order(event_ids: list[str], fold: int) -> list[str]:
    return sorted(
        event_ids,
        key=lambda event: hashlib.sha256(
            f"{fold}::{event}".encode("utf-8")
        ).hexdigest(),
    )


def prepare_folds() -> None:
    data = _load_internal()
    events = event_table(data)
    real, components = _real_components(events)
    assignments = _assign_components(components, n_folds=3)
    real["fold"] = real["component"].map(assignments).astype(int)
    groups = data["groups"].astype(str)
    all_event_to_indices = {
        event: np.where(groups == event)[0] for event in np.unique(groups)
    }
    synthetic = events[events["synthetic"] == 1].copy()
    audit_rows = []
    FOLD_ROOT.mkdir(parents=True, exist_ok=True)

    for fold in range(3):
        validation_events = set(real.loc[real["fold"] == fold, "event_id"].tolist())
        train_real_events = set(real.loc[real["fold"] != fold, "event_id"].tolist())
        validation_mmsi: set[str] = set()
        for row in real[real["fold"] == fold].itertuples(index=False):
            validation_mmsi.update(
                vessel
                for vessel in (_valid_mmsi(row.mmsi_a), _valid_mmsi(row.mmsi_b))
                if vessel
            )
        safe_synthetic = []
        for row in synthetic.itertuples(index=False):
            vessels = {
                vessel
                for vessel in (_valid_mmsi(row.mmsi_a), _valid_mmsi(row.mmsi_b))
                if vessel
            }
            if not (vessels & validation_mmsi):
                safe_synthetic.append(str(row.event_id))
        safe_synthetic = _stable_synthetic_order(safe_synthetic, fold)

        val_idx = np.concatenate(
            [all_event_to_indices[event] for event in sorted(validation_events)]
        ).astype(np.int64)
        for variant, requested_synthetic in VARIANTS.items():
            selected_synthetic = safe_synthetic[:requested_synthetic]
            train_events = sorted(train_real_events | set(selected_synthetic))
            train_idx = np.concatenate(
                [all_event_to_indices[event] for event in train_events]
            ).astype(np.int64)
            path = FOLD_ROOT / variant / f"fold_{fold}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                train_idx=np.sort(train_idx),
                val_idx=np.sort(val_idx),
                test_idx=np.zeros((0,), dtype=np.int64),
            )
            train_events_frame = events[events["event_id"].isin(train_events)]
            val_events_frame = real[real["event_id"].isin(validation_events)]
            train_mmsi = {
                vessel
                for row in train_events_frame.itertuples(index=False)
                for vessel in (_valid_mmsi(row.mmsi_a), _valid_mmsi(row.mmsi_b))
                if vessel
            }
            val_mmsi = {
                vessel
                for row in val_events_frame.itertuples(index=False)
                for vessel in (_valid_mmsi(row.mmsi_a), _valid_mmsi(row.mmsi_b))
                if vessel
            }
            overlap = train_mmsi & val_mmsi
            if overlap:
                raise RuntimeError(
                    f"Fold {fold} variant {variant} leaks MMSI: {sorted(overlap)[:10]}"
                )
            audit_rows.append(
                {
                    "variant": variant,
                    "fold": fold,
                    "train_events": int(len(train_events_frame)),
                    "train_windows": int(len(train_idx)),
                    "train_synthetic_events": int(len(selected_synthetic)),
                    "safe_synthetic_available": int(len(safe_synthetic)),
                    "validation_events": int(len(val_events_frame)),
                    "validation_windows": int(len(val_idx)),
                    "validation_positive_encounter": int(
                        np.sum(
                            (val_events_frame["y"] == 1)
                            & (val_events_frame["event_kind"] == "encounter")
                        )
                    ),
                    "validation_positive_loitering": int(
                        np.sum(
                            (val_events_frame["y"] == 1)
                            & (val_events_frame["event_kind"] == "loitering")
                        )
                    ),
                    "validation_negative_encounter": int(
                        np.sum(
                            (val_events_frame["y"] == 0)
                            & (val_events_frame["event_kind"] == "encounter")
                        )
                    ),
                    "validation_negative_loitering": int(
                        np.sum(
                            (val_events_frame["y"] == 0)
                            & (val_events_frame["event_kind"] == "loitering")
                        )
                    ),
                    "validation_sources": int(
                        val_events_frame["source_label"].nunique()
                    ),
                    "synthetic_in_validation": 0,
                    "mmsi_overlap": int(len(overlap)),
                }
            )

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(TUNING_ROOT / "fold_audit.csv", index=False)
    real[["event_id", "component", "fold", "event_kind", "source_label", "y"]].to_csv(
        TUNING_ROOT / "real_event_fold_assignments.csv", index=False
    )
    manifest = {
        "protocol": "three_fold_real_event_oof_synthetic_train_only_v1",
        "internal_npz": str(INTERNAL_NPZ),
        "internal_npz_sha256": sha256(INTERNAL_NPZ),
        "folds": 3,
        "real_events": int(len(real)),
        "vessel_pair_components": int(len(components)),
        "variants": VARIANTS,
        "external_used": False,
        "training_artifact_root": str(TRAIN_ROOT),
    }
    (TUNING_ROOT / "fold_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(audit.to_string(index=False))
    print(f"[trans-oof] folds prepared -> {FOLD_ROOT}")


def _trial_dir(variant: str, seed: int, fold: int) -> Path:
    return RUN_ROOT / variant / f"seed_{seed}" / f"fold_{fold}"


def train_trial(variant: str, seed: int, fold: int) -> None:
    split_path = FOLD_ROOT / variant / f"fold_{fold}.npz"
    if not split_path.is_file():
        raise FileNotFoundError("Run prepare before search.")
    trial = _trial_dir(variant, seed, fold)
    model_dir = trial / "model_transshipment"
    val_dir = trial / "validation_eval"
    prediction_path = val_dir / "per_event_predictions.csv"
    required = [model_dir / "model.pt", val_dir / "eval_summary.json", prediction_path]
    spec = {
        "variant": variant,
        "seed": seed,
        "fold": fold,
        "epochs": epochs(),
        "architecture": "compact_bilstm_h128_attention4",
        "internal_npz_sha256": sha256(INTERNAL_NPZ),
        "external_used": False,
    }
    spec_path = trial / "trial_spec.json"
    if all(path.is_file() for path in required) and spec_path.is_file():
        existing = json.loads(spec_path.read_text(encoding="utf-8"))
        frame = pd.read_csv(prediction_path, nrows=1)
        if existing == spec and "positive_probability" in frame.columns:
            print(f"[trans-oof] reuse {variant} seed={seed} fold={fold}")
            return

    command = [
        sys.executable,
        "main.py",
        "train",
        "--data_npz",
        str(INTERNAL_NPZ),
        "--split_indices_path",
        str(split_path),
        "--out_dir",
        str(model_dir),
        "--device",
        os.environ.get("DEVICE", "auto"),
        "--random_state",
        str(seed),
        "--split_random_state",
        "20260623",
        "--train_random_state",
        str(seed),
        "--test_size",
        "0",
        "--val_size",
        "0.333333",
        "--epochs",
        str(epochs()),
        "--disable_early_stopping",
        "--batch_size",
        "128",
        "--lr",
        "0.00025",
        "--hidden_size",
        "128",
        "--num_layers",
        "1",
        "--input_proj_dim",
        "96",
        "--embed_dim",
        "128",
        "--dropout",
        "0.40",
        "--attention_heads",
        "4",
        "--attention_layers",
        "1",
        "--optimizer",
        "adamw",
        "--weight_decay",
        "0.0013",
        "--focal_gamma",
        "1.2",
        "--geo_aux_weight",
        "0",
        "--eval_after_train",
        "--validation_eval_out",
        str(val_dir),
    ]
    run(command, f"internal OOF {variant} seed={seed} fold={fold}")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Trial missing artifacts: {missing}")
    trial.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")


def binary_metrics(frame: pd.DataFrame, threshold: float, prob_col: str) -> dict:
    y = frame["true_id"].to_numpy(dtype=np.int64)
    pred = frame[prob_col].to_numpy(dtype=np.float64) >= float(threshold)
    tp = int(np.sum((y == 1) & pred))
    fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & ~pred))
    tn = int(np.sum((y == 0) & ~pred))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    specificity = tn / max(tn + fp, 1)
    normal_precision = tn / max(tn + fn, 1)
    normal_recall = specificity
    normal_f1 = 2.0 * normal_precision * normal_recall / max(
        normal_precision + normal_recall, 1e-12
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "event_f1": f1,
        "specificity": specificity,
        "balanced_acc": 0.5 * (recall + specificity),
        "accuracy": (tp + tn) / max(len(frame), 1),
        "macro_f1": 0.5 * (f1 + normal_f1),
        "positive_support": int(np.sum(y == 1)),
    }


def grouped_metrics(
    frame: pd.DataFrame, threshold: float, prob_col: str, group_col: str
) -> tuple[pd.DataFrame, dict]:
    rows = []
    for value, part in frame.groupby(group_col):
        rows.append({group_col: str(value), **binary_metrics(part, threshold, prob_col)})
    table = pd.DataFrame(rows)
    positive_rows = table[table["positive_support"] > 0]
    summary = {
        f"macro_{group_col}_macro_f1": float(table["macro_f1"].mean()),
        f"macro_{group_col}_positive_f1": float(positive_rows["event_f1"].mean()),
        f"min_{group_col}_recall": float(positive_rows["recall"].min()),
        f"macro_{group_col}_specificity": float(table["specificity"].mean()),
    }
    return table, summary


def _apply_platt(frame: pd.DataFrame, estimator: LogisticRegression) -> pd.DataFrame:
    p = np.clip(frame["positive_probability"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    x = np.log(p / (1.0 - p)).reshape(-1, 1)
    out = frame.copy()
    out["calibrated_probability"] = estimator.predict_proba(x)[:, 1]
    return out


def _fit_platt(frame: pd.DataFrame) -> LogisticRegression:
    p = np.clip(frame["positive_probability"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    x = np.log(p / (1.0 - p)).reshape(-1, 1)
    estimator = LogisticRegression(C=1.0, solver="lbfgs", random_state=20260623)
    estimator.fit(x, frame["true_id"].to_numpy(dtype=np.int64))
    return estimator


def calibrated_seed_oof(variant: str, seed: int) -> tuple[pd.DataFrame, Path]:
    fold_frames: dict[int, pd.DataFrame] = {}
    for fold in range(3):
        path = _trial_dir(variant, seed, fold) / "validation_eval" / "per_event_predictions.csv"
        frame = pd.read_csv(path)
        required = {
            "event_id",
            "true_id",
            "positive_probability",
            "event_kind",
            "source_label",
            "mmsi_a",
            "mmsi_b",
        }
        missing = required - set(frame)
        if missing:
            raise RuntimeError(f"OOF prediction missing columns {sorted(missing)}: {path}")
        frame["fold"] = fold
        fold_frames[fold] = frame
    pooled_raw = pd.concat(fold_frames.values(), ignore_index=True)
    if pooled_raw["event_id"].duplicated().any():
        raise RuntimeError(f"Duplicate OOF events for {variant} seed={seed}")

    calibrated_parts = []
    for held_fold, held in fold_frames.items():
        calibration_train = pd.concat(
            [frame for fold, frame in fold_frames.items() if fold != held_fold],
            ignore_index=True,
        )
        calibrated_parts.append(_apply_platt(held, _fit_platt(calibration_train)))
    calibrated = pd.concat(calibrated_parts, ignore_index=True)
    calibrated["seed"] = seed

    final_estimator = _fit_platt(pooled_raw)
    CALIBRATOR_ROOT.mkdir(parents=True, exist_ok=True)
    path = CALIBRATOR_ROOT / f"{variant}_seed_{seed}_platt.joblib"
    joblib.dump(
        {
            "estimator": final_estimator,
            "fit_scope": "pooled_internal_real_event_oof_only",
            "variant": variant,
            "seed": seed,
            "external_used": False,
            "internal_npz_sha256": sha256(INTERNAL_NPZ),
        },
        path,
    )
    calibrated.to_csv(
        CALIBRATOR_ROOT / f"{variant}_seed_{seed}_cross_calibrated_oof.csv",
        index=False,
    )
    return calibrated, path


def ensemble_seed_oof(seed_frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    merged = None
    probability_columns = []
    keys = ["event_id", "true_id", "event_kind", "source_label", "mmsi_a", "mmsi_b"]
    for seed, frame in seed_frames.items():
        column = f"probability_seed_{seed}"
        probability_columns.append(column)
        keep = frame[keys + ["calibrated_probability"]].rename(
            columns={"calibrated_probability": column}
        )
        merged = keep if merged is None else merged.merge(
            keep, on=keys, how="inner", validate="one_to_one"
        )
    if merged is None or merged.empty:
        raise RuntimeError("No seed OOF predictions available.")
    merged["ensemble_probability"] = merged[probability_columns].mean(axis=1)
    return merged


def select_threshold(frame: pd.DataFrame, variant: str) -> tuple[float, pd.DataFrame, dict]:
    rows = []
    best_key = None
    best_threshold = None
    for threshold in np.arange(0.05, 0.9501, 0.025):
        overall = binary_metrics(frame, threshold, "ensemble_probability")
        _, kind_summary = grouped_metrics(
            frame, threshold, "ensemble_probability", "event_kind"
        )
        _, source_summary = grouped_metrics(
            frame, threshold, "ensemble_probability", "source_label"
        )
        row = {
            "variant": variant,
            "threshold": float(threshold),
            **overall,
            **kind_summary,
            **source_summary,
        }
        floors_met = (
            overall["precision"] >= 0.80
            and overall["recall"] >= 0.70
            and kind_summary["min_event_kind_recall"] >= 0.60
        )
        row["selection_floors_met"] = int(floors_met)
        rows.append(row)
        key = (
            int(floors_met),
            kind_summary["macro_event_kind_positive_f1"],
            source_summary["macro_source_label_macro_f1"],
            overall["event_f1"],
            overall["precision"],
            overall["specificity"],
            -overall["fp"],
            -abs(float(threshold) - 0.5),
        )
        if best_key is None or key > best_key:
            best_key, best_threshold = key, float(threshold)
    sweep = pd.DataFrame(rows)
    selected = sweep[np.isclose(sweep["threshold"], best_threshold)].iloc[0].to_dict()
    return float(best_threshold), sweep, selected


def search() -> None:
    if not (TUNING_ROOT / "fold_manifest.json").is_file():
        prepare_folds()
    seed_values = seeds()
    for variant in VARIANTS:
        for seed in seed_values:
            for fold in range(3):
                train_trial(variant, seed, fold)

    ranking_rows = []
    calibrators: dict[str, dict[int, str]] = {}
    for variant in VARIANTS:
        seed_frames = {}
        calibrators[variant] = {}
        for seed in seed_values:
            seed_frame, calibrator = calibrated_seed_oof(variant, seed)
            seed_frames[seed] = seed_frame
            calibrators[variant][seed] = str(calibrator)
        ensemble = ensemble_seed_oof(seed_frames)
        # Expected coverage is checked at the event level, not windows.
        expected_events = int(
            event_table(_load_internal()).query("synthetic == 0")["event_id"].nunique()
        )
        if len(ensemble) != expected_events:
            raise RuntimeError(
                f"OOF event coverage mismatch {variant}: {len(ensemble)} != {expected_events}"
            )
        threshold, sweep, selected = select_threshold(ensemble, variant)
        sweep.to_csv(TUNING_ROOT / f"{variant}_threshold_sweep.csv", index=False)
        ensemble.to_csv(TUNING_ROOT / f"{variant}_ensemble_oof_predictions.csv", index=False)
        kind_table, _ = grouped_metrics(
            ensemble, threshold, "ensemble_probability", "event_kind"
        )
        source_table, _ = grouped_metrics(
            ensemble, threshold, "ensemble_probability", "source_label"
        )
        kind_table.to_csv(TUNING_ROOT / f"{variant}_event_kind_metrics.csv", index=False)
        source_table.to_csv(TUNING_ROOT / f"{variant}_source_metrics.csv", index=False)
        ranking_rows.append(selected)

    ranking = pd.DataFrame(ranking_rows).sort_values(
        [
            "selection_floors_met",
            "macro_event_kind_positive_f1",
            "macro_source_label_macro_f1",
            "event_f1",
            "precision",
        ],
        ascending=False,
    )
    ranking.to_csv(TUNING_ROOT / "internal_oof_variant_ranking.csv", index=False)
    winner = json.loads(ranking.iloc[0].to_json())
    winner_variant = str(winner["variant"])
    manifest = {
        "selection_basis": "three-fold real-event internal OOF only",
        "external_used_for_selection": False,
        "winner": winner,
        "seeds": seed_values,
        "folds": 3,
        "calibrators": calibrators[winner_variant],
        "internal_npz": str(INTERNAL_NPZ),
        "internal_npz_sha256": sha256(INTERNAL_NPZ),
        "threshold_policy": (
            "precision>=0.80, recall>=0.70, min event-kind recall>=0.60; "
            "then maximize event-kind/source macro performance and penalize FP"
        ),
        "model": "compact BiLSTM h128 + 4-head self-attention",
        "training_artifact_root": str(TRAIN_ROOT),
    }
    WINNER_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (TUNING_ROOT / "external_status.json").write_text(
        json.dumps(
            {
                "status": "locked_pending_explicit_evaluation",
                "external_used_during_search": False,
                "note": "Current external set was already inspected previously and is development-only.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(ranking.to_string(index=False))
    print(f"[trans-oof] internal winner -> {WINNER_PATH}")


def _bootstrap_ci(frame: pd.DataFrame, threshold: float, iterations: int = 2000) -> dict:
    rng = np.random.RandomState(20260623)
    strata = {
        key: part.index.to_numpy()
        for key, part in frame.groupby(["true_id", "event_kind"])
    }
    collected = {key: [] for key in ["precision", "recall", "event_f1", "balanced_acc", "macro_f1", "accuracy"]}
    for _ in range(iterations):
        indices = np.concatenate(
            [rng.choice(idx, size=len(idx), replace=True) for idx in strata.values()]
        )
        metric = binary_metrics(frame.loc[indices], threshold, "ensemble_probability")
        for key in collected:
            collected[key].append(float(metric[key]))
    return {
        key: {
            "lower_95": float(np.percentile(values, 2.5)),
            "upper_95": float(np.percentile(values, 97.5)),
        }
        for key, values in collected.items()
    }


def external_development() -> None:
    if not WINNER_PATH.is_file():
        raise FileNotFoundError("Run search before external evaluation.")
    if not EXTERNAL_NPZ.is_file():
        raise FileNotFoundError(EXTERNAL_NPZ)
    manifest = json.loads(WINNER_PATH.read_text(encoding="utf-8"))
    if manifest.get("external_used_for_selection") is not False:
        raise RuntimeError("Winner was not selected internal-only.")
    if manifest.get("internal_npz_sha256") != sha256(INTERNAL_NPZ):
        raise RuntimeError("Internal NPZ changed after winner selection.")
    variant = str(manifest["winner"]["variant"])
    threshold = float(manifest["winner"]["threshold"])
    seed_values = [int(value) for value in manifest["seeds"]]
    merged = None
    probability_columns = []
    keys = ["event_id", "true_id", "event_kind", "source_label", "mmsi_a", "mmsi_b"]

    for seed in seed_values:
        calibrator_path = Path(manifest["calibrators"][str(seed)])
        estimator = joblib.load(calibrator_path)["estimator"]
        for fold in range(3):
            model = _trial_dir(variant, seed, fold) / "model_transshipment" / "model.pt"
            out = TUNING_ROOT / "development_external_models" / f"seed_{seed}_fold_{fold}"
            command = [
                sys.executable,
                "main.py",
                "eval",
                "--data_npz",
                str(EXTERNAL_NPZ),
                "--model_path",
                str(model),
                "--out_dir",
                str(out),
                "--device",
                os.environ.get("DEVICE", "auto"),
                "--eval_split",
                "all",
            ]
            run(command, f"development external {variant} seed={seed} fold={fold}")
            frame = pd.read_csv(out / "per_event_predictions.csv")
            calibrated = _apply_platt(frame, estimator)
            column = f"probability_seed_{seed}_fold_{fold}"
            probability_columns.append(column)
            keep = calibrated[keys + ["calibrated_probability"]].rename(
                columns={"calibrated_probability": column}
            )
            merged = keep if merged is None else merged.merge(
                keep, on=keys, how="inner", validate="one_to_one"
            )

    if merged is None or merged.empty:
        raise RuntimeError("No external ensemble predictions.")
    merged["ensemble_probability"] = merged[probability_columns].mean(axis=1)
    merged["pred_id"] = (
        merged["ensemble_probability"].to_numpy() >= threshold
    ).astype(np.int64)
    metric = binary_metrics(merged, threshold, "ensemble_probability")
    kind_table, kind_summary = grouped_metrics(
        merged, threshold, "ensemble_probability", "event_kind"
    )
    source_table, source_summary = grouped_metrics(
        merged, threshold, "ensemble_probability", "source_label"
    )
    out_dir = TUNING_ROOT / "development_external_ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "ensemble_event_predictions.csv", index=False)
    kind_table.to_csv(out_dir / "event_kind_metrics.csv", index=False)
    source_table.to_csv(out_dir / "source_metrics.csv", index=False)
    from eval import save_transshipment_detection_png

    cm = np.array(
        [[metric["tn"], metric["fp"]], [metric["fn"], metric["tp"]]],
        dtype=np.int64,
    )
    save_transshipment_detection_png(cm, out_dir / "confusion_matrix.png")
    ci = _bootstrap_ci(merged, threshold)
    summary = {
        "role": "external-development; not untouched final holdout",
        "final_claim_allowed": False,
        "selection_used_external": False,
        "winner_variant": variant,
        "threshold_from_internal_oof": threshold,
        "ensemble": "mean calibrated probability across 3 seeds x 3 folds",
        "metrics": metric,
        "event_kind_summary": kind_summary,
        "source_summary": source_summary,
        "bootstrap_method": "2000 true-label x event-kind stratified event resamples",
        "bootstrap_95_ci": ci,
    }
    (out_dir / "ensemble_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser("Leakage-safe transshipment internal OOF tuning")
    parser.add_argument(
        "mode", choices=["prepare", "search", "external"], nargs="?", default="prepare"
    )
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare_folds()
    elif args.mode == "search":
        search()
    else:
        external_development()


if __name__ == "__main__":
    main()
