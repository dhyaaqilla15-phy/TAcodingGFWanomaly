from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parent


def env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    path = Path(value) if value else default
    return path if path.is_absolute() else (ROOT / path).resolve()


INTERNAL_NPZ = env_path(
    "GODARK_INTERNAL_NPZ",
    ROOT / "Outputs" / "godark_external01" / "data_internal_trainval" / "processed_godark.npz",
)
EXTERNAL_NPZ = env_path(
    "GODARK_EXTERNAL_NPZ",
    ROOT / "Outputs" / "godark_external01" / "data_external_test" / "processed_godark.npz",
)
TUNING_ROOT = env_path(
    "GODARK_TUNING_ROOT",
    ROOT / "Outputs" / "godark_tuning01_internal_oof",
)
TRIAL_ROOT = TUNING_ROOT / "trials"
WINNER_PATH = TUNING_ROOT / "winner_internal_only.json"
SPLIT_ROOT = TUNING_ROOT / "source_stratified_folds"
CALIBRATOR_ROOT = TUNING_ROOT / "oof_calibrators"
EXPECTED_SOURCES = {
    "drifting_longlines",
    "fixed_gear",
    "purse_seines",
    "trawlers",
}


@dataclass(frozen=True)
class Config:
    name: str
    hidden_size: int
    num_layers: int
    input_proj_dim: int
    embed_dim: int
    dropout: float
    weight_decay: float
    lr: float
    focal_gamma: float


CONFIGS = (
    Config("compact_h64", 64, 1, 64, 128, 0.40, 0.003, 2e-4, 1.0),
    Config("compact_h128", 128, 1, 96, 128, 0.40, 0.003, 2e-4, 1.0),
    Config("compact_h128_e256", 128, 1, 128, 256, 0.50, 0.003, 2e-4, 1.0),
    Config("medium_h256", 256, 1, 128, 256, 0.40, 0.003, 2e-4, 1.0),
    Config("compact_lowreg", 128, 1, 96, 128, 0.30, 0.001, 3e-4, 1.2),
    Config("compact_highreg", 128, 1, 96, 128, 0.50, 0.010, 1e-4, 2.0),
)


def selected_configs() -> tuple[Config, ...]:
    raw = os.environ.get("GODARK_CONFIG_NAMES", "").strip()
    if not raw:
        return CONFIGS
    requested = {name.strip() for name in raw.split(",") if name.strip()}
    selected = tuple(config for config in CONFIGS if config.name in requested)
    missing = requested - {config.name for config in selected}
    if missing:
        raise ValueError(f"Unknown GODARK_CONFIG_NAMES: {sorted(missing)}")
    return selected


def seeds() -> list[int]:
    raw = os.environ.get("GODARK_TUNING_SEEDS", "42,43,44")
    parsed = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if len(parsed) < 3:
        raise ValueError("GODARK_TUNING_SEEDS must contain at least three seeds.")
    return parsed


def run(command: list[str], title: str) -> None:
    print("\n" + "=" * 72)
    print(f"[godark-tuning] {title}")
    print("+ " + " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_signature(path: Path) -> str:
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=True) as split:
        for key in ("train_idx", "val_idx", "test_idx"):
            values = (
                split[key].astype(np.int64)
                if key in split.files
                else np.array([], dtype=np.int64)
            )
            digest.update(key.encode("utf-8"))
            digest.update(values.tobytes())
    return digest.hexdigest()


def validate_source_label_cells(data: np.lib.npyio.NpzFile, domain: str) -> None:
    sources = data["window_source_labels"].astype(str)
    y = data["y"].astype(np.int64)
    actual = set(sources.tolist())
    if actual != EXPECTED_SOURCES:
        raise RuntimeError(f"Unexpected {domain} source labels: {sorted(actual)}")
    incomplete = {
        source: sorted(set(y[sources == source].tolist()))
        for source in EXPECTED_SOURCES
        if set(y[sources == source].tolist()) != {0, 1}
    }
    if incomplete:
        raise RuntimeError(
            f"Every {domain} source must contain normal and go-dark labels: {incomplete}"
        )


def validate_internal_npz() -> None:
    if not INTERNAL_NPZ.is_file():
        raise FileNotFoundError(
            f"Missing {INTERNAL_NPZ}. Run: bash run_godark_external_test_pipeline.sh prepare"
        )
    with np.load(INTERNAL_NPZ, allow_pickle=True) as data:
        if "window_source_labels" not in data.files:
            raise RuntimeError(
                "Internal NPZ is stale and has no window_source_labels. Rerun prepare."
            )
        protocol = (
            str(np.asarray(data["godark_diversity_protocol"]).item())
            if "godark_diversity_protocol" in data.files
            else "missing"
        )
        if protocol != "source_label_duration_cadence_distance_position_v2":
            raise RuntimeError(
                "Internal NPZ predates the diversity-balanced generator. Rerun prepare."
            )
        validate_source_label_cells(data, "internal")


def validate_external_npz() -> None:
    if not EXTERNAL_NPZ.is_file():
        raise FileNotFoundError(f"Missing external NPZ: {EXTERNAL_NPZ}")
    with np.load(INTERNAL_NPZ, allow_pickle=True) as internal, np.load(
        EXTERNAL_NPZ, allow_pickle=True
    ) as external:
        if "window_source_labels" not in external.files:
            raise RuntimeError(
                "External NPZ is stale and has no window_source_labels. Rerun prepare."
            )
        external_protocol = (
            str(np.asarray(external["godark_diversity_protocol"]).item())
            if "godark_diversity_protocol" in external.files
            else "missing"
        )
        if external_protocol != "source_label_duration_cadence_distance_position_v2":
            raise RuntimeError(
                "External NPZ does not match the locked diversity protocol. Rerun prepare."
            )
        validate_source_label_cells(external, "external")
        internal_groups = set(internal["groups"].astype(str).tolist())
        external_groups = set(external["groups"].astype(str).tolist())
        overlap = sorted(internal_groups & external_groups)
        if overlap:
            raise RuntimeError(f"Internal/external MMSI leakage detected: {overlap[:10]}")
        if internal["X"].shape[1:] != external["X"].shape[1:]:
            raise RuntimeError("Internal/external sequence schemas differ.")
        if internal["feature_cols"].astype(str).tolist() != external[
            "feature_cols"
        ].astype(str).tolist():
            raise RuntimeError("Internal/external feature columns differ.")


def build_source_stratified_folds(seed_values: list[int]) -> dict[int, Path]:
    """Create disjoint vessel folds with every source represented in every fold."""
    n_folds = len(seed_values)
    with np.load(INTERNAL_NPZ, allow_pickle=True) as data:
        groups = data["groups"].astype(str)
        sources = data["window_source_labels"].astype(str)
        y = data["y"].astype(np.int64)
    unique_groups = np.unique(groups)
    rng = np.random.RandomState(20260623)
    fold_groups: list[list[str]] = [[] for _ in range(n_folds)]

    for source in sorted(EXPECTED_SOURCES):
        source_groups = sorted(set(groups[sources == source].tolist()))
        if len(source_groups) < n_folds:
            raise RuntimeError(
                f"Source {source} has {len(source_groups)} vessels; need >= {n_folds}."
            )
        records = []
        for group in source_groups:
            mask = groups == group
            if set(sources[mask].tolist()) != {source}:
                raise RuntimeError(f"Vessel {group} appears in multiple sources.")
            records.append(
                {
                    "group": group,
                    "positive": int(np.sum(y[mask] == 1)),
                    "total": int(np.sum(mask)),
                    "tie": float(rng.rand()),
                }
            )
        records.sort(key=lambda row: (-row["positive"], -row["total"], row["tie"]))
        stats = [{"positive": 0, "total": 0, "groups": 0} for _ in range(n_folds)]
        for order, row in enumerate(records):
            target = order if order < n_folds else min(
                range(n_folds),
                key=lambda fold: (
                    stats[fold]["positive"],
                    stats[fold]["total"],
                    stats[fold]["groups"],
                ),
            )
            fold_groups[target].append(str(row["group"]))
            stats[target]["positive"] += int(row["positive"])
            stats[target]["total"] += int(row["total"])
            stats[target]["groups"] += 1

    SPLIT_ROOT.mkdir(parents=True, exist_ok=True)
    paths: dict[int, Path] = {}
    all_indices = np.arange(len(y), dtype=np.int64)
    validation_union: list[np.ndarray] = []
    audit_rows = []
    for fold, seed in enumerate(seed_values):
        val_mask = np.isin(groups, fold_groups[fold])
        val_idx = np.where(val_mask)[0].astype(np.int64)
        train_idx = np.where(~val_mask)[0].astype(np.int64)
        for source in sorted(EXPECTED_SOURCES):
            mask = val_mask & (sources == source)
            labels = set(y[mask].tolist())
            if labels != {0, 1}:
                raise RuntimeError(
                    f"Fold {fold} source {source} lacks a class: {sorted(labels)}"
                )
            audit_rows.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "source_class": source,
                    "validation_vessels": int(len(set(groups[mask].tolist()))),
                    "validation_normal": int(np.sum(mask & (y == 0))),
                    "validation_go_dark": int(np.sum(mask & (y == 1))),
                }
            )
        path = SPLIT_ROOT / f"fold_{fold}_seed_{seed}.npz"
        np.savez_compressed(
            path,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=np.array([], dtype=np.int64),
        )
        paths[seed] = path
        validation_union.append(val_idx)
    union = np.concatenate(validation_union)
    if not np.array_equal(np.sort(union), all_indices) or len(np.unique(union)) != len(y):
        raise RuntimeError("Source-stratified folds do not form exact OOF coverage.")
    pd.DataFrame(audit_rows).to_csv(SPLIT_ROOT / "fold_source_audit.csv", index=False)
    return paths


def trial_dir(config: Config, seed: int) -> Path:
    return TRIAL_ROOT / config.name / f"seed_{seed}"


def train_trial(config: Config, seed: int, split_path: Path) -> None:
    out = trial_dir(config, seed)
    model = out / "model_godark"
    val = out / "validation_eval"
    required = (
        model / "model.pt",
        model / "godark_hardnegative_hybrid.joblib",
        val / "eval_summary.json",
        val / "per_godark_event_predictions.csv",
    )
    spec_path = out / "trial_spec.json"
    expected_spec = {
        "config": asdict(config),
        "seed": int(seed),
        "internal_npz_sha256": sha256(INTERNAL_NPZ),
        "epochs": 50,
        "early_stopping": False,
        "godark_source_balancing": True,
        "source_stratified_split_signature": split_signature(split_path),
        "external_used": False,
    }
    existing_spec = (
        json.loads(spec_path.read_text(encoding="utf-8"))
        if spec_path.is_file()
        else None
    )
    if all(path.is_file() for path in required) and existing_spec == expected_spec:
        print(f"[godark-tuning] reuse complete trial {config.name} seed={seed}")
        return

    train_cmd = [
        sys.executable,
        "main.py",
        "train",
        "--data_npz", str(INTERNAL_NPZ),
        "--out_dir", str(model),
        "--device", os.environ.get("DEVICE", "auto"),
        "--random_state", str(seed),
        "--split_random_state", str(seed),
        "--train_random_state", str(seed),
        "--split_indices_path", str(split_path),
        "--test_size", "0",
        "--val_size", "0.20",
        "--epochs", "50",
        "--disable_early_stopping",
        "--batch_size", "32",
        "--hidden_size", str(config.hidden_size),
        "--num_layers", str(config.num_layers),
        "--input_proj_dim", str(config.input_proj_dim),
        "--embed_dim", str(config.embed_dim),
        "--dropout", str(config.dropout),
        "--weight_decay", str(config.weight_decay),
        "--lr", str(config.lr),
        "--focal_gamma", str(config.focal_gamma),
        "--geo_aux_weight", "0",
        "--eval_after_train",
        "--validation_eval_out", str(val),
    ]
    run(train_cmd, f"train internal-only {config.name} seed={seed}")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Trial finished without required artifacts: {missing}")
    spec_path.write_text(json.dumps(expected_spec, indent=2), encoding="utf-8")


def binary_metrics(frame: pd.DataFrame, threshold: float) -> dict:
    y = frame["true_event"].to_numpy(dtype=np.int64)
    pred = frame["max_go_dark_probability"].to_numpy(dtype=np.float64) >= float(threshold)
    tp = int(np.sum((y == 1) & pred))
    fp = int(np.sum((y == 0) & pred))
    fn = int(np.sum((y == 1) & ~pred))
    tn = int(np.sum((y == 0) & ~pred))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    specificity = tn / max(tn + fp, 1)
    balanced_acc = 0.5 * (recall + specificity)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    normal_precision = tn / max(tn + fn, 1)
    normal_recall = tn / max(tn + fp, 1)
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
        "balanced_acc": balanced_acc,
        "accuracy": accuracy,
        "macro_f1": 0.5 * (f1 + normal_f1),
    }


def source_metrics(frame: pd.DataFrame, threshold: float) -> dict:
    rows = []
    for source, part in frame.groupby("source_class"):
        metric = binary_metrics(part, threshold)
        rows.append({"source_class": str(source), **metric})
    if {row["source_class"] for row in rows} != EXPECTED_SOURCES:
        raise RuntimeError("OOF source metrics do not cover all expected sources.")
    return {
        "rows": rows,
        "macro_source_f1": float(np.mean([row["event_f1"] for row in rows])),
        "min_source_recall": float(np.min([row["recall"] for row in rows])),
        "macro_source_recall": float(np.mean([row["recall"] for row in rows])),
    }


def fit_platt_calibrator(config: Config, pooled: pd.DataFrame) -> Path:
    probability = np.clip(
        pooled["max_go_dark_probability"].to_numpy(dtype=np.float64),
        1e-6,
        1.0 - 1e-6,
    )
    transformed = np.log(probability / (1.0 - probability)).reshape(-1, 1)
    estimator = LogisticRegression(C=1.0, solver="lbfgs", random_state=20260623)
    estimator.fit(transformed, pooled["true_event"].to_numpy(dtype=np.int64))
    CALIBRATOR_ROOT.mkdir(parents=True, exist_ok=True)
    path = CALIBRATOR_ROOT / f"{config.name}_platt.joblib"
    joblib.dump(
        {
            "estimator": estimator,
            "method": "platt_logistic_on_logit_probability",
            "fit_scope": "pooled_internal_source_stratified_oof_only",
            "external_used": False,
            "internal_npz_sha256": sha256(INTERNAL_NPZ),
        },
        path,
    )
    return path


def apply_platt_estimator(
    frame: pd.DataFrame, estimator: LogisticRegression
) -> pd.DataFrame:
    probability = np.clip(
        frame["max_go_dark_probability"].to_numpy(dtype=np.float64),
        1e-6,
        1.0 - 1e-6,
    )
    transformed = np.log(probability / (1.0 - probability)).reshape(-1, 1)
    out = frame.copy()
    out["uncalibrated_max_go_dark_probability"] = probability
    out["max_go_dark_probability"] = estimator.predict_proba(transformed)[:, 1]
    return out


def apply_platt(frame: pd.DataFrame, calibrator_path: Path) -> pd.DataFrame:
    artifact = joblib.load(calibrator_path)
    return apply_platt_estimator(frame, artifact["estimator"])


def select_oof_threshold(
    config: Config, seed_values: list[int]
) -> tuple[float, pd.DataFrame, dict, Path]:
    frames: dict[int, pd.DataFrame] = {}
    for seed in seed_values:
        frame = pd.read_csv(
            trial_dir(config, seed) / "validation_eval" / "per_godark_event_predictions.csv"
        )
        frame["oof_fold_seed"] = int(seed)
        frames[seed] = frame

    pooled_raw = pd.concat(list(frames.values()), ignore_index=True)
    if pooled_raw["event_id"].astype(str).duplicated().any():
        raise RuntimeError("Source-stratified OOF contains duplicate event IDs.")
    with np.load(INTERNAL_NPZ, allow_pickle=True) as data:
        expected_events = int(len(data["y"]))
    if len(pooled_raw) != expected_events:
        raise RuntimeError(
            f"OOF coverage mismatch: predictions={len(pooled_raw)} expected={expected_events}"
        )
    calibrator_path = fit_platt_calibrator(config, pooled_raw)
    cross_calibrated: dict[int, pd.DataFrame] = {}
    for held_seed, held_frame in frames.items():
        calibration_train = pd.concat(
            [frame for seed, frame in frames.items() if seed != held_seed],
            ignore_index=True,
        )
        probability = np.clip(
            calibration_train["max_go_dark_probability"].to_numpy(dtype=np.float64),
            1e-6,
            1.0 - 1e-6,
        )
        transformed = np.log(probability / (1.0 - probability)).reshape(-1, 1)
        estimator = LogisticRegression(C=1.0, solver="lbfgs", random_state=20260623)
        estimator.fit(
            transformed,
            calibration_train["true_event"].to_numpy(dtype=np.int64),
        )
        cross_calibrated[held_seed] = apply_platt_estimator(held_frame, estimator)
    frames = cross_calibrated
    pooled_frame = pd.concat(list(frames.values()), ignore_index=True)

    rows = []
    best_key = None
    best_threshold = None
    for threshold in np.arange(0.05, 0.9501, 0.025):
        per_seed = [binary_metrics(frames[seed], float(threshold)) for seed in seed_values]
        pooled = binary_metrics(pooled_frame, float(threshold))
        by_source = source_metrics(pooled_frame, float(threshold))
        mean_f1 = float(np.mean([row["event_f1"] for row in per_seed]))
        mean_recall = float(np.mean([row["recall"] for row in per_seed]))
        mean_precision = float(np.mean([row["precision"] for row in per_seed]))
        std_f1 = float(np.std([row["event_f1"] for row in per_seed], ddof=1))
        row = {
            "config": config.name,
            "threshold": float(threshold),
            "mean_event_f1": mean_f1,
            "std_event_f1": std_f1,
            "mean_precision": mean_precision,
            "mean_recall": mean_recall,
            "pooled_event_f1": pooled["event_f1"],
            "pooled_precision": pooled["precision"],
            "pooled_recall": pooled["recall"],
            "pooled_fp": pooled["fp"],
            "pooled_fn": pooled["fn"],
            "macro_source_f1": by_source["macro_source_f1"],
            "macro_source_recall": by_source["macro_source_recall"],
            "min_source_recall": by_source["min_source_recall"],
        }
        rows.append(row)
        source_recall_floor_met = by_source["min_source_recall"] >= 0.50
        key = (
            int(source_recall_floor_met),
            by_source["macro_source_f1"],
            by_source["min_source_recall"],
            mean_f1,
            pooled["event_f1"],
            mean_precision,
            -std_f1,
            -abs(float(threshold) - 0.5),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)

    sweep = pd.DataFrame(rows)
    selected = sweep.loc[np.isclose(sweep["threshold"], best_threshold)].iloc[0].to_dict()
    pooled_frame.to_csv(
        CALIBRATOR_ROOT / f"{config.name}_calibrated_oof_predictions.csv",
        index=False,
    )
    pd.DataFrame(
        source_metrics(pooled_frame, float(best_threshold))["rows"]
    ).to_csv(
        CALIBRATOR_ROOT / f"{config.name}_selected_source_metrics.csv",
        index=False,
    )
    return float(best_threshold), sweep, selected, calibrator_path


def search() -> None:
    validate_internal_npz()
    seed_values = seeds()
    TUNING_ROOT.mkdir(parents=True, exist_ok=True)
    split_paths = build_source_stratified_folds(seed_values)

    configs = selected_configs()
    for config in configs:
        for seed in seed_values:
            train_trial(config, seed, split_paths[seed])

    ranking = []
    all_sweeps = []
    for config in configs:
        threshold, sweep, selected, calibrator_path = select_oof_threshold(
            config, seed_values
        )
        all_sweeps.append(sweep)
        per_seed_rows = []
        calibrated_oof = pd.read_csv(
            CALIBRATOR_ROOT / f"{config.name}_calibrated_oof_predictions.csv"
        )
        for seed in seed_values:
            frame = calibrated_oof[
                calibrated_oof["oof_fold_seed"].astype(int) == int(seed)
            ].copy()
            per_seed_rows.append(binary_metrics(frame, threshold))
        ranking.append(
            {
                **asdict(config),
                "seeds": ",".join(map(str, seed_values)),
                "pooled_oof_threshold": threshold,
                "mean_event_f1": float(np.mean([r["event_f1"] for r in per_seed_rows])),
                "std_event_f1": float(np.std([r["event_f1"] for r in per_seed_rows], ddof=1)),
                "mean_precision": float(np.mean([r["precision"] for r in per_seed_rows])),
                "mean_recall": float(np.mean([r["recall"] for r in per_seed_rows])),
                "mean_balanced_acc": float(np.mean([r["balanced_acc"] for r in per_seed_rows])),
                "macro_source_f1": float(selected["macro_source_f1"]),
                "macro_source_recall": float(selected["macro_source_recall"]),
                "min_source_recall": float(selected["min_source_recall"]),
                "platt_calibrator_path": str(calibrator_path),
                "platt_calibrator_sha256": sha256(calibrator_path),
                "external_used_for_selection": False,
            }
        )

    ranking_df = pd.DataFrame(ranking).sort_values(
        ["macro_source_f1", "min_source_recall", "mean_event_f1", "std_event_f1"],
        ascending=[False, False, False, True],
    )
    ranking_df.to_csv(TUNING_ROOT / "internal_multiseed_ranking.csv", index=False)
    pd.concat(all_sweeps, ignore_index=True).to_csv(
        TUNING_ROOT / "pooled_oof_threshold_sweep.csv", index=False
    )
    winner = ranking_df.iloc[0].to_dict()
    WINNER_PATH.write_text(
        json.dumps(
            {
                "selection_scope": "internal_train_validation_only",
                "selection_metric": "macro_source_f1_with_min_source_recall_floor",
                "external_used_for_selection": False,
                "internal_npz_sha256": sha256(INTERNAL_NPZ),
                "seeds": seed_values,
                "fold_policy": "disjoint_source_stratified_vessel_oof",
                "ensemble_policy": "mean_calibrated_probability_all_fold_models",
                "winner": winner,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (TUNING_ROOT / "external_evaluation_status.json").write_text(
        json.dumps(
            {
                "status": "locked_pending_new_winner_evaluation",
                "winner_manifest_sha256": sha256(WINNER_PATH),
                "warning": "Any older external outputs in this directory are stale.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n[godark-tuning] INTERNAL WINNER")
    print(json.dumps(winner, indent=2))
    print(f"[godark-tuning] external remains locked; winner -> {WINNER_PATH}")


def bootstrap_cluster_ci(
    frame: pd.DataFrame,
    threshold: float,
    iterations: int = 2000,
) -> dict:
    work = frame.copy()
    parts = work["event_id"].astype(str).str.split("::")
    work["vessel_id"] = parts.map(
        lambda values: values[2] if len(values) > 2 else values[0]
    )
    work["cluster"] = work["source_class"].astype(str) + "::" + work["vessel_id"]
    rng = np.random.RandomState(20260623)
    collected = {key: [] for key in ["precision", "recall", "event_f1", "balanced_acc", "macro_f1", "accuracy"]}
    source_clusters = {
        source: sorted(part["cluster"].unique().tolist())
        for source, part in work.groupby("source_class")
    }
    for _ in range(int(iterations)):
        sampled_parts = []
        for source, clusters in source_clusters.items():
            chosen = rng.choice(clusters, size=len(clusters), replace=True)
            source_frame = work[work["source_class"] == source]
            for cluster in chosen:
                sampled_parts.append(source_frame[source_frame["cluster"] == cluster])
        sampled = pd.concat(sampled_parts, ignore_index=True)
        metric = binary_metrics(sampled, threshold)
        for key in collected:
            collected[key].append(float(metric[key]))
    return {
        key: {
            "lower_95": float(np.percentile(values, 2.5)),
            "upper_95": float(np.percentile(values, 97.5)),
        }
        for key, values in collected.items()
    }


def build_external_ensemble(seed_values: list[int], threshold: float) -> tuple[dict, dict]:
    merged = None
    probability_cols = []
    for seed in seed_values:
        path = (
            TUNING_ROOT
            / "final_external_winner"
            / f"seed_{seed}"
            / "external_test_eval"
            / "per_godark_event_predictions.csv"
        )
        frame = pd.read_csv(path)
        probability_col = f"probability_seed_{seed}"
        probability_cols.append(probability_col)
        keep = frame[
            ["event_id", "event_kind", "source_class", "true_event", "max_go_dark_probability"]
        ].rename(columns={"max_go_dark_probability": probability_col})
        merged = keep if merged is None else merged.merge(
            keep,
            on=["event_id", "event_kind", "source_class", "true_event"],
            how="inner",
            validate="one_to_one",
        )
    if merged is None or merged.empty:
        raise RuntimeError("No external predictions available for ensemble.")
    merged["max_go_dark_probability"] = merged[probability_cols].mean(axis=1)
    merged["pred_event"] = (
        merged["max_go_dark_probability"].to_numpy() >= float(threshold)
    ).astype(np.int64)
    merged["error_type"] = np.select(
        [
            (merged["true_event"] == 1) & (merged["pred_event"] == 1),
            (merged["true_event"] == 0) & (merged["pred_event"] == 1),
            (merged["true_event"] == 1) & (merged["pred_event"] == 0),
        ],
        ["TP", "FP", "FN"],
        default="TN",
    )
    out_dir = TUNING_ROOT / "final_external_ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "ensemble_event_predictions.csv", index=False)
    metric = binary_metrics(merged, threshold)
    by_source = source_metrics(merged, threshold)
    pd.DataFrame(by_source["rows"]).to_csv(
        out_dir / "ensemble_source_metrics.csv", index=False
    )
    from eval import save_godark_detection_png

    cm = np.array(
        [[metric["tn"], metric["fp"]], [metric["fn"], metric["tp"]]],
        dtype=np.int64,
    )
    save_godark_detection_png(
        cm,
        out_dir / "confusion_matrix.png",
        scope="event",
    )
    ci = bootstrap_cluster_ci(merged, threshold)
    summary = {
        "policy": "mean_calibrated_probability_all_fold_models",
        "policy_selected_from": "internal_source_stratified_oof_only",
        "external_used_for_policy_selection": False,
        "threshold": float(threshold),
        "seeds": seed_values,
        "metrics": metric,
        "source_metrics": by_source,
        "bootstrap_method": "2000 source-stratified vessel-cluster resamples",
        "bootstrap_95_ci": ci,
    }
    (out_dir / "ensemble_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary, ci


def external_final() -> None:
    if not WINNER_PATH.is_file():
        raise FileNotFoundError("Run internal search first; winner file is missing.")
    manifest = json.loads(WINNER_PATH.read_text(encoding="utf-8"))
    if manifest.get("external_used_for_selection") is not False:
        raise RuntimeError("Winner manifest is not internal-only.")
    validate_internal_npz()
    if manifest.get("internal_npz_sha256") != sha256(INTERNAL_NPZ):
        raise RuntimeError(
            "Internal NPZ changed after winner selection. Rerun internal search."
        )
    validate_external_npz()
    winner = manifest["winner"]
    config = next(cfg for cfg in CONFIGS if cfg.name == winner["name"])
    threshold = float(winner["pooled_oof_threshold"])
    seed_values = [int(value) for value in manifest["seeds"]]
    calibrator_path = Path(str(winner["platt_calibrator_path"]))
    if not calibrator_path.is_absolute():
        calibrator_path = (ROOT / calibrator_path).resolve()
    if not calibrator_path.is_file():
        raise FileNotFoundError(f"Missing winner OOF calibrator: {calibrator_path}")
    if sha256(calibrator_path) != str(winner["platt_calibrator_sha256"]):
        raise RuntimeError("Winner OOF calibrator hash mismatch.")

    rows = []
    for seed in seed_values:
        model = trial_dir(config, seed) / "model_godark" / "model.pt"
        out = TUNING_ROOT / "final_external_winner" / f"seed_{seed}" / "external_test_eval"
        command = [
            sys.executable,
            "main.py",
            "eval",
            "--data_npz", str(EXTERNAL_NPZ),
            "--model_path", str(model),
            "--out_dir", str(out),
            "--device", os.environ.get("DEVICE", "auto"),
            "--eval_split", "all",
            "--godark_event_prob_threshold", str(threshold),
            "--godark_calibrator_path", str(calibrator_path),
        ]
        run(command, f"FINAL EXTERNAL winner={config.name} seed={seed}")
        summary = json.loads((out / "eval_summary.json").read_text(encoding="utf-8"))
        event = summary["metrics_godark_event"]
        rows.append(
            {
                "config": config.name,
                "seed": seed,
                "threshold_from_internal_oof": threshold,
                "event_precision": event["event_precision"],
                "event_recall": event["event_recall"],
                "event_f1": event["event_f1"],
                "event_fp": event["event_fp"],
                "event_fn": event["event_fn"],
                "macro_f1": summary["metrics_seq"]["macro_f1"],
                "balanced_acc": summary["metrics_seq"]["balanced_acc"],
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(TUNING_ROOT / "final_external_winner_results.csv", index=False)
    ensemble_summary, ensemble_ci = build_external_ensemble(seed_values, threshold)
    metrics = ["event_precision", "event_recall", "event_f1", "macro_f1", "balanced_acc"]
    final_summary = {
        "winner": config.name,
        "external_used_for_selection": False,
        "methodological_note": (
            "This external set is unbiased only if it was not inspected or used "
            "to design the current iteration; otherwise treat it as external-development."
        ),
        "threshold_source": "pooled_internal_out_of_fold_validation",
        "threshold": threshold,
        "seeds": seed_values,
        "mean": {key: float(frame[key].mean()) for key in metrics},
        "std": {key: float(frame[key].std(ddof=1)) for key in metrics},
        "ensemble": ensemble_summary,
        "ensemble_bootstrap_95_ci": ensemble_ci,
    }
    (TUNING_ROOT / "final_external_summary.json").write_text(
        json.dumps(final_summary, indent=2), encoding="utf-8"
    )
    (TUNING_ROOT / "external_evaluation_status.json").write_text(
        json.dumps(
            {
                "status": "completed_for_current_winner",
                "winner_manifest_sha256": sha256(WINNER_PATH),
                "final_summary": str(TUNING_ROOT / "final_external_summary.json"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(final_summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["search", "external"])
    args = parser.parse_args()
    if args.mode == "search":
        search()
    else:
        external_final()


if __name__ == "__main__":
    main()
