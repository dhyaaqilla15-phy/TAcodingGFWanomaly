from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)

from eval import save_spoofing_detection_png


ROOT = Path(__file__).resolve().parent


def env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name, "").strip()
    path = Path(raw) if raw else default
    return path if path.is_absolute() else (ROOT / path).resolve()


OUTPUT_ROOT = env_path(
    "SPOOFING_OUTPUT_ROOT",
    ROOT / "Outputs" / "spoofing_hybrid02_context_oof_four_gear",
)
INTERNAL_GENERATED = OUTPUT_ROOT / "_generated_internal"
EXTERNAL_GENERATED = OUTPUT_ROOT / "_generated_external"
INTERNAL_PREP = OUTPUT_ROOT / "_data_internal_trainval"
EXTERNAL_PREP = OUTPUT_ROOT / "_data_external_test"
INTERNAL_NPZ = INTERNAL_PREP / "processed_spoofing.npz"
EXTERNAL_NPZ = EXTERNAL_PREP / "processed_spoofing.npz"
LOCK_PATH = OUTPUT_ROOT / ".spoofing_multiseed.lock"
DATA_AUDIT_PATH = OUTPUT_ROOT / "spoofing_data_audit.json"
SPLIT_AUDIT_PATH = OUTPUT_ROOT / "spoofing_split_audit.json"
SEMANTICS_AUDIT_PATH = OUTPUT_ROOT / "spoofing_generation_semantics_audit.json"
OOF_SPLIT_DIR = OUTPUT_ROOT / "_source_stratified_oof_splits"
OOF_DIR = OUTPUT_ROOT / "internal_oof_policy"
OOF_POLICY_PATH = OOF_DIR / "platt_scenario_policy.json"
EXTERNAL_ENSEMBLE_DIR = OUTPUT_ROOT / "final_external_ensemble"

ALL_ATTACKS = (
    "gradual_drift",
    "location_jump",
    "replay",
    "meaconing",
    "ghost",
    "mirroring",
)
ATTACKS = tuple(
    value.strip().lower()
    for value in os.environ.get(
        "SPOOFING_ATTACKS", " ".join(ALL_ATTACKS)
    ).replace(",", " ").split()
    if value.strip()
)
if not ATTACKS or set(ATTACKS) - set(ALL_ATTACKS):
    raise ValueError(f"Invalid SPOOFING_ATTACKS={ATTACKS}; allowed={ALL_ATTACKS}")
SOURCE_INCLUDE_LABELS = tuple(
    value.strip().lower()
    for value in os.environ.get(
        "SPOOFING_INCLUDE_LABELS",
        "drifting_longlines fixed_gear purse_seines trawlers",
    ).replace(",", " ").split()
    if value.strip()
)
LOCKED_SOURCE_LABELS = (
    "drifting_longlines",
    "fixed_gear",
    "purse_seines",
    "trawlers",
)
STRICT_FOUR_GEAR = os.environ.get("SPOOFING_STRICT_FOUR_GEAR", "0") == "1"
if STRICT_FOUR_GEAR and set(SOURCE_INCLUDE_LABELS) != set(LOCKED_SOURCE_LABELS):
    raise ValueError(
        "Six-attack profile requires exactly four sources: "
        f"{LOCKED_SOURCE_LABELS}; got={SOURCE_INCLUDE_LABELS}"
    )

SEEDS = tuple(
    int(value.strip())
    for value in os.environ.get("SPOOFING_SEEDS", "42,43,44").split(",")
    if value.strip()
)
EPOCHS = int(os.environ.get("SPOOFING_EPOCHS", "50"))
DISABLE_EARLY_STOPPING = (
    os.environ.get("SPOOFING_DISABLE_EARLY_STOPPING", "0") == "1"
)
OOF_MIN_SOURCE_RECALL = float(os.environ.get("SPOOFING_OOF_MIN_SOURCE_RECALL", "0.50"))
OOF_MIN_ATTACK_RECALL = float(os.environ.get("SPOOFING_OOF_MIN_ATTACK_RECALL", "0.20"))
TRAIN_LR = float(os.environ.get("SPOOFING_LR", "0.00025"))
TRAIN_HIDDEN_SIZE = int(os.environ.get("SPOOFING_HIDDEN_SIZE", "128"))
TRAIN_NUM_LAYERS = int(os.environ.get("SPOOFING_NUM_LAYERS", "2"))
TRAIN_INPUT_PROJ_DIM = int(os.environ.get("SPOOFING_INPUT_PROJ_DIM", "128"))
TRAIN_EMBED_DIM = int(os.environ.get("SPOOFING_EMBED_DIM", "192"))
TRAIN_DROPOUT = float(os.environ.get("SPOOFING_DROPOUT", "0.35"))
TRAIN_WEIGHT_DECAY = float(os.environ.get("SPOOFING_WEIGHT_DECAY", "0.0013"))
TRAIN_FOCAL_GAMMA = float(os.environ.get("SPOOFING_FOCAL_GAMMA", "1.2"))
USE_LOCATION_FEATURES = os.environ.get("SPOOFING_USE_LOCATION_FEATURES", "0") == "1"
FINAL_DRIFT_DEG = float(os.environ.get("SPOOFING_DRIFT_DEG", "0.01"))
FINAL_JUMP_DEG = float(os.environ.get("SPOOFING_JUMP_DEG", "0.50"))
FINAL_DRIFT_RATE_KMH = float(os.environ.get("SPOOFING_DRIFT_RATE_KMH", "0.033"))
FINAL_DRIFT_RATE_JITTER_FRAC = float(
    os.environ.get("SPOOFING_DRIFT_RATE_JITTER_FRAC", "0.50")
)
FINAL_MIRROR_OFFSET_MIN_DEG = float(
    os.environ.get("SPOOFING_MIRROR_OFFSET_MIN_DEG", "1.5")
)
FINAL_MIRROR_OFFSET_MAX_DEG = float(
    os.environ.get("SPOOFING_MIRROR_OFFSET_MAX_DEG", "8.0")
)
SCENARIOS_PER_ATTACK = int(os.environ.get("SPOOFING_SCENARIOS_PER_ATTACK", "3"))
REPORTED_MOTION_MODE = "preserve"
MIXED_RECOMPUTE_PROBABILITY = 0.0
INCLUDE_MATCHED_NORMAL_CONTROLS = True


def run(command: list[str], label: str) -> int:
    print(f"\n[spoofing-runner] {label}")
    print("[spoofing-runner] " + subprocess.list2cmdline(command))
    started = time.perf_counter()
    status = subprocess.run(command, cwd=ROOT, check=False).returncode
    print(
        f"[spoofing-runner] {label} exit={status} "
        f"duration={time.perf_counter() - started:.1f}s"
    )
    return status


def process_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    result = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
        capture_output=True,
        text=True,
        check=False,
    )
    return str(pid) in result.stdout


def acquire_lock() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        try:
            old_pid = int(LOCK_PATH.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            old_pid = -1
        if process_is_running(old_pid):
            raise RuntimeError(
                f"Runner spoofing lain masih aktif dengan PID {old_pid}."
            )
        LOCK_PATH.unlink(missing_ok=True)
    fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))


def generation_complete(out_dir: Path) -> bool:
    path = out_dir / "spoofed_all.csv"
    magnitude_audits = list((out_dir / "summaries").glob("magnitude_*.csv"))
    generated_attacks: set[str] = set()
    for audit_path in magnitude_audits:
        try:
            frame = pd.read_csv(audit_path, usecols=["attack_type"])
        except (OSError, ValueError, pd.errors.EmptyDataError):
            return False
        generated_attacks.update(frame["attack_type"].astype(str).str.lower())
    return (
        path.is_file()
        and path.stat().st_size > 0
        and bool(magnitude_audits)
        and generated_attacks == set(ATTACKS)
    )


def model_complete(model_dir: Path) -> bool:
    required = (
        "model.pt",
        "best_epoch.json",
        "history.json",
        "train_config.json",
        "split_indices.npz",
        "scaler.joblib",
    )
    if not all((model_dir / name).is_file() for name in required):
        return False
    try:
        history = json.loads((model_dir / "history.json").read_text())
        config = json.loads((model_dir / "train_config.json").read_text())
        best = json.loads((model_dir / "best_epoch.json").read_text())
        if not history or int(config["epochs"]) != EPOCHS:
            return False
        completed_epoch = int(history[-1]["epoch"])
        patience = int(config.get("early_stop_patience", 0))
        best_epoch = int(best["best_epoch"])
        reached_configured_end = completed_epoch >= EPOCHS
        completed_valid_early_stop = (
            patience > 0 and completed_epoch >= best_epoch + patience
        )
        return reached_configured_end or completed_valid_early_stop
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def eval_complete(eval_dir: Path) -> bool:
    return all(
        (eval_dir / name).is_file()
        for name in (
            "eval_summary.json",
            "confusion_matrix.png",
            "spoofing_attack_metrics.csv",
            "spoofing_severity_metrics.csv",
            "spoofing_sequence_predictions.csv",
            "spoofing_scenario_predictions.csv",
        )
    )


def generation_command(
    input_path: str,
    out_dir: Path,
    *,
    seed: int,
    limit_rows: int,
    max_vessels_per_file: int,
) -> list[str]:
    return [
        sys.executable,
        "main.py",
        "make_spoofing",
        "--input_path",
        input_path,
        "--out_dir",
        str(out_dir),
        "--attacks",
        *ATTACKS,
        "--seed",
        str(seed),
        "--limit_rows",
        str(limit_rows),
        "--normal_keep_frac",
        "0.50",
        "--include_labels",
        *SOURCE_INCLUDE_LABELS,
        "--exclude_labels",
        "pole_and_line",
        "trollers",
        "--max_vessels_per_file",
        str(max_vessels_per_file),
        "--min_points_per_vessel",
        "300",
        "--points_per_attack",
        "240",
        "--scenarios_per_attack",
        str(SCENARIOS_PER_ATTACK),
        "--drift_lat_deg",
        str(FINAL_DRIFT_DEG),
        "--drift_lon_deg",
        str(FINAL_DRIFT_DEG),
        "--drift_rate_kmh",
        str(FINAL_DRIFT_RATE_KMH),
        "--drift_rate_jitter_frac",
        str(FINAL_DRIFT_RATE_JITTER_FRAC),
        "--jump_lat_deg",
        str(FINAL_JUMP_DEG),
        "--jump_lon_deg",
        str(FINAL_JUMP_DEG),
        "--mirror_offset_min_deg",
        str(FINAL_MIRROR_OFFSET_MIN_DEG),
        "--mirror_offset_max_deg",
        str(FINAL_MIRROR_OFFSET_MAX_DEG),
        "--reported_motion_mode",
        REPORTED_MOTION_MODE,
        "--mixed_recompute_probability",
        str(MIXED_RECOMPUTE_PROBABILITY),
        "--include_matched_normal_controls",
        "--combine_outputs",
    ]


def preprocess_command(data_dir: Path, out_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "main.py",
        "preprocess",
        "--data_dir",
        str(data_dir),
        "--out_dir",
        str(out_dir),
        "--task",
        "spoofing",
        "--seq_len",
        "120",
        "--stride",
        "6",
        "--gap_seconds",
        "10800",
        "--min_points_per_vessel",
        "80",
        "--max_windows_per_vessel",
        "1200",
        "--max_windows_per_file",
        "0",
        "--spoofing_window_threshold",
        "0.20",
    ]
    if not USE_LOCATION_FEATURES:
        command.append("--exclude_location_features")
    return command


def train_command(seed: int, model_dir: Path, split_path: Path) -> list[str]:
    command = [
        sys.executable,
        "main.py",
        "train",
        "--data_npz",
        str(INTERNAL_NPZ),
        "--out_dir",
        str(model_dir),
        "--device",
        "cuda",
        "--random_state",
        str(seed),
        "--split_random_state",
        str(seed),
        "--train_random_state",
        str(seed),
        "--test_size",
        "0",
        "--val_size",
        "0.20",
        "--split_indices_path",
        str(split_path),
        "--epochs",
        str(EPOCHS),
        "--batch_size",
        "128",
        "--lr",
        str(TRAIN_LR),
        "--hidden_size",
        str(TRAIN_HIDDEN_SIZE),
        "--num_layers",
        str(TRAIN_NUM_LAYERS),
        "--input_proj_dim",
        str(TRAIN_INPUT_PROJ_DIM),
        "--embed_dim",
        str(TRAIN_EMBED_DIM),
        "--dropout",
        str(TRAIN_DROPOUT),
        "--attention_heads",
        "4",
        "--attention_layers",
        "1",
        "--optimizer",
        "adamw",
        "--weight_decay",
        str(TRAIN_WEIGHT_DECAY),
        "--focal_gamma",
        str(TRAIN_FOCAL_GAMMA),
    ]
    if DISABLE_EARLY_STOPPING:
        command.append("--disable_early_stopping")
    else:
        command.extend(["--early_stop_patience", "10"])
    command.extend(["--geo_aux_weight", "0"])
    return command


def eval_command(
    data_npz: Path,
    model_path: Path,
    eval_dir: Path,
    split: str,
) -> list[str]:
    return [
        sys.executable,
        "main.py",
        "eval",
        "--data_npz",
        str(data_npz),
        "--model_path",
        str(model_path),
        "--out_dir",
        str(eval_dir),
        "--device",
        "cuda",
        "--batch_size",
        "256",
        "--eval_split",
        split,
    ]


def ensure_artifact(
    command: list[str],
    label: str,
    complete,
) -> None:
    if complete():
        print(f"[spoofing-runner] {label}: complete, skip")
        return
    status = run(command, label)
    if not complete():
        raise RuntimeError(f"{label} tidak lengkap (exit={status}).")
    if status != 0:
        print(
            f"[spoofing-runner] WARNING: {label} exit={status}, tetapi "
            "artefak terverifikasi lengkap; lanjut."
        )


def prepare_internal() -> dict[int, Path]:
    ensure_artifact(
        generation_command(
            "Dataset",
            INTERNAL_GENERATED,
            seed=42,
            limit_rows=300_000,
            max_vessels_per_file=20,
        ),
        "generate internal spoofing",
        lambda: generation_complete(INTERNAL_GENERATED),
    )
    validate_generation_semantics(("internal",))
    ensure_artifact(
        preprocess_command(INTERNAL_GENERATED, INTERNAL_PREP),
        "preprocess internal spoofing",
        lambda: INTERNAL_NPZ.is_file(),
    )
    validate_preprocessed_data(require_external=False)
    return prepare_source_stratified_oof_splits()


def prepare_external_after_internal_policy() -> None:
    if not OOF_POLICY_PATH.is_file():
        raise RuntimeError(
            "External test terkunci: jalankan pooled internal OOF policy dulu. "
            f"Missing: {OOF_POLICY_PATH}"
        )
    ensure_artifact(
        generation_command(
            "Dataset_Test_Enriched",
            EXTERNAL_GENERATED,
            seed=1042,
            limit_rows=0,
            max_vessels_per_file=0,
        ),
        "generate external spoofing",
        lambda: generation_complete(EXTERNAL_GENERATED),
    )
    validate_generation_semantics(("internal", "external"))
    ensure_artifact(
        preprocess_command(EXTERNAL_GENERATED, EXTERNAL_PREP),
        "preprocess external spoofing",
        lambda: EXTERNAL_NPZ.is_file(),
    )
    validate_preprocessed_data(require_external=True)


def prepare() -> None:
    prepare_internal()


def _read_magnitude_audits(generated_dir: Path) -> pd.DataFrame:
    paths = sorted((generated_dir / "summaries").glob("magnitude_*.csv"))
    if not paths:
        raise RuntimeError(f"No magnitude audits found in {generated_dir}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def validate_generation_semantics(stages: tuple[str, ...] = ("internal", "external")) -> None:
    expected_modes = (
        {"preserve", "recompute"}
        if REPORTED_MOTION_MODE == "mixed"
        else {REPORTED_MOTION_MODE}
    )
    expected_attacks = set(ATTACKS)
    severity_scales = (
        [1.0]
        if SCENARIOS_PER_ATTACK <= 1
        else np.linspace(0.60, 1.40, SCENARIOS_PER_ATTACK).tolist()
    )
    lower_rate = (
        FINAL_DRIFT_RATE_KMH
        * min(severity_scales)
        * (1.0 - FINAL_DRIFT_RATE_JITTER_FRAC)
    )
    upper_rate = (
        FINAL_DRIFT_RATE_KMH
        * max(severity_scales)
        * (1.0 + FINAL_DRIFT_RATE_JITTER_FRAC)
    )
    audit: dict[str, object] = {
        "reported_motion_mode": REPORTED_MOTION_MODE,
        "expected_modes_per_attack": sorted(expected_modes),
        "scenarios_per_attack": SCENARIOS_PER_ATTACK,
        "severity_scales": severity_scales,
        "target_drift_rate_kmh": FINAL_DRIFT_RATE_KMH,
        "allowed_drift_rate_kmh": [lower_rate, upper_rate],
        "datasets": {},
        "problems": [],
    }
    problems: list[str] = []
    known_dirs = {
        "internal": INTERNAL_GENERATED,
        "external": EXTERNAL_GENERATED,
    }
    for name in stages:
        if name not in known_dirs:
            raise RuntimeError(f"Unknown generation stage: {name}")
        generated_dir = known_dirs[name]
        data = _read_magnitude_audits(generated_dir)
        combined_path = generated_dir / "spoofed_all.csv"
        controls = pd.read_csv(
            combined_path,
            usecols=["scenario_id", "note", "normal_control_for_attack"],
            low_memory=False,
        )
        controls = controls[
            controls["note"].astype(str).eq("matched_unmodified_control")
        ]
        control_scenarios = int(controls["scenario_id"].astype(str).nunique())
        attack_scenarios = int(
            data[data["attack_type"].astype(str).isin(expected_attacks)][
                "scenario_id"
            ].astype(str).nunique()
        )
        dataset_rows: dict[str, object] = {}
        dataset_rows["matched_controls"] = {
            "attack_scenarios": attack_scenarios,
            "control_scenarios": control_scenarios,
        }
        if INCLUDE_MATCHED_NORMAL_CONTROLS and control_scenarios != attack_scenarios:
            problems.append(
                f"{name} matched controls={control_scenarios}, attacks={attack_scenarios}"
            )
        for attack in sorted(expected_attacks):
            subset = data[data["attack_type"].astype(str).eq(attack)].copy()
            modes = set(subset["reported_motion_mode"].astype(str).tolist())
            mode_counts = {
                str(key): int(value)
                for key, value in subset["reported_motion_mode"].value_counts().items()
            }
            dataset_rows[attack] = {
                "scenarios": int(len(subset)),
                "reported_motion_mode_counts": mode_counts,
            }
            if modes != expected_modes:
                problems.append(
                    f"{name}/{attack} modes={sorted(modes)}, expected={sorted(expected_modes)}"
                )
            if attack == "gradual_drift":
                rates = pd.to_numeric(
                    subset["attack_drift_rate_kmh"], errors="coerce"
                ).dropna()
                dataset_rows[attack]["drift_rate_kmh_min"] = float(rates.min())
                dataset_rows[attack]["drift_rate_kmh_median"] = float(rates.median())
                dataset_rows[attack]["drift_rate_kmh_max"] = float(rates.max())
                tolerance = 1e-6
                if (
                    rates.empty
                    or float(rates.min()) < lower_rate - tolerance
                    or float(rates.max()) > upper_rate + tolerance
                ):
                    problems.append(
                        f"{name}/gradual_drift rate outside frozen range"
                    )
        audit["datasets"][name] = dataset_rows
    audit["problems"] = problems
    audit["valid"] = not problems
    SEMANTICS_AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if problems:
        raise RuntimeError("; ".join(problems))
    print("[spoofing-runner] generation semantics audit passed")


def validate_preprocessed_data(require_external: bool = True) -> None:
    internal = np.load(INTERNAL_NPZ, allow_pickle=True)
    internal_groups = set(internal["groups"].astype(str).tolist())
    internal_features = internal["feature_cols"].astype(str).tolist()
    internal_classes = sorted(np.unique(internal["y"]).astype(int).tolist())
    internal_positive_ratio = float(np.mean(internal["y"] == 1))
    internal_attacks = sorted(
        set(internal["window_kinds"].astype(str).tolist()) - {"normal"}
    )
    problems = []
    external = None
    external_groups: set[str] = set()
    external_features: list[str] = []
    external_classes: list[int] = []
    external_positive_ratio: float | None = None
    external_attacks: list[str] = []
    overlap: list[str] = []
    if require_external:
        external = np.load(EXTERNAL_NPZ, allow_pickle=True)
        external_groups = set(external["groups"].astype(str).tolist())
        overlap = sorted(internal_groups & external_groups)
        external_features = external["feature_cols"].astype(str).tolist()
        external_classes = sorted(np.unique(external["y"]).astype(int).tolist())
        external_positive_ratio = float(np.mean(external["y"] == 1))
        external_attacks = sorted(
            set(external["window_kinds"].astype(str).tolist()) - {"normal"}
        )
        if overlap:
            problems.append(f"source MMSI overlap internal/external: {overlap[:10]}")
        if internal_features != external_features:
            problems.append("internal/external feature schema differs")
    required_context = {
        "claimed_identity_registered",
        "claimed_history_age_log_hours",
        "claimed_prev_dt_log_hours",
        "claimed_prev_distance_log_km",
        "claimed_prev_implied_speed_log_knots",
        "claimed_concurrent_reports_log1p",
        "claimed_concurrent_spread_log_km",
        "claimed_revisit_lag_log_hours",
        "claimed_revisit_score",
    }
    missing_context = sorted(required_context - set(internal_features))
    if missing_context:
        problems.append(f"observable context features missing: {missing_context}")
    datasets_to_check: list[tuple[str, np.lib.npyio.NpzFile]] = [("internal", internal)]
    if external is not None:
        datasets_to_check.append(("external", external))
    for name, data in datasets_to_check:
        if "spoofing_balance_strata" not in data.files:
            problems.append(f"{name} spoofing balance strata missing")
    if internal_classes != [0, 1]:
        problems.append(f"both internal classes required; internal={internal_classes}")
    if require_external and external_classes != [0, 1]:
        problems.append(f"both external classes required; external={external_classes}")
    if not 0.01 <= internal_positive_ratio <= 0.50:
        problems.append(
            "internal positive-window ratio unhealthy: "
            f"{internal_positive_ratio:.3f}"
        )
    if (
        require_external
        and external_positive_ratio is not None
        and not 0.01 <= external_positive_ratio <= 0.50
    ):
        problems.append(
            "external positive-window ratio unhealthy: "
            f"{external_positive_ratio:.3f}"
        )
    expected_attacks = set(ATTACKS)
    if set(internal_attacks) != expected_attacks:
        problems.append(f"internal attacks incomplete: {internal_attacks}")
    if require_external and set(external_attacks) != expected_attacks:
        problems.append(f"external attacks incomplete: {external_attacks}")
    internal_sources = sorted(
        set(internal["window_source_labels"].astype(str).tolist())
        if "window_source_labels" in internal.files
        else set()
    )
    external_sources = (
        sorted(
            set(external["window_source_labels"].astype(str).tolist())
            if "window_source_labels" in external.files
            else set()
        )
        if external is not None
        else []
    )
    if STRICT_FOUR_GEAR:
        expected_sources = set(LOCKED_SOURCE_LABELS)
        if set(internal_sources) != expected_sources:
            problems.append(f"internal sources invalid: {internal_sources}")
        if require_external and set(external_sources) != expected_sources:
            problems.append(f"external sources invalid: {external_sources}")

    def source_attack_coverage(data: np.lib.npyio.NpzFile) -> list[dict]:
        if "window_source_labels" not in data.files:
            return []
        frame = pd.DataFrame(
            {
                "source": data["window_source_labels"].astype(str),
                "attack": data["window_kinds"].astype(str),
                "positive": data["y"].astype(np.int64),
            }
        )
        frame = frame[frame["attack"].isin(ATTACKS)]
        return [
            {
                "source": str(source),
                "attack": str(attack),
                "windows": int(len(part)),
                "positive_windows": int(part["positive"].sum()),
            }
            for (source, attack), part in frame.groupby(
                ["source", "attack"], sort=True
            )
        ]

    internal_coverage = source_attack_coverage(internal)
    external_coverage = source_attack_coverage(external) if external is not None else []
    if STRICT_FOUR_GEAR:
        expected_pairs = {
            (source, attack)
            for source in LOCKED_SOURCE_LABELS
            for attack in ATTACKS
        }
        for name, rows in (
            ("internal", internal_coverage),
            *([("external", external_coverage)] if require_external else []),
        ):
            observed = {
                (row["source"], row["attack"])
                for row in rows
                if int(row["positive_windows"]) > 0
            }
            missing_pairs = sorted(expected_pairs - observed)
            if missing_pairs:
                problems.append(
                    f"{name} source x attack positive coverage missing: "
                    f"{missing_pairs}"
                )

    audit = {
        "internal_windows": int(internal["y"].shape[0]),
        "external_windows": int(external["y"].shape[0]) if external is not None else None,
        "internal_source_groups": len(internal_groups),
        "external_source_groups": len(external_groups),
        "source_group_overlap": overlap,
        "internal_class_counts": np.bincount(
            internal["y"].astype(np.int64), minlength=2
        ).astype(int).tolist(),
        "external_class_counts": np.bincount(
            external["y"].astype(np.int64), minlength=2
        ).astype(int).tolist()
        if external is not None
        else None,
        "internal_positive_ratio": internal_positive_ratio,
        "external_positive_ratio": external_positive_ratio,
        "external_locked_until_after_internal_oof_policy": not require_external,
        "drift_rate_kmh": FINAL_DRIFT_RATE_KMH,
        "drift_rate_jitter_frac": FINAL_DRIFT_RATE_JITTER_FRAC,
        "jump_nominal_deg": FINAL_JUMP_DEG,
        "scenarios_per_attack": SCENARIOS_PER_ATTACK,
        "reported_motion_mode": REPORTED_MOTION_MODE,
        "mixed_recompute_probability": MIXED_RECOMPUTE_PROBABILITY,
        "include_matched_normal_controls": INCLUDE_MATCHED_NORMAL_CONTROLS,
        "internal_attacks": internal_attacks,
        "external_attacks": external_attacks,
        "internal_sources": internal_sources,
        "external_sources": external_sources,
        "strict_four_gear": STRICT_FOUR_GEAR,
        "internal_source_attack_coverage": internal_coverage,
        "external_source_attack_coverage": external_coverage,
        "feature_count": len(internal_features),
        "context_feature_count": len(required_context),
        "context_features": sorted(required_context),
        "location_features_used": any(
            name in internal_features
            for name in ("distance_from_shore", "distance_from_port")
        ),
        "spoofing_use_location_features_requested": bool(USE_LOCATION_FEATURES),
        "valid": not problems,
        "problems": problems,
    }
    DATA_AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if problems:
        raise RuntimeError("; ".join(problems))
    print(
        "[spoofing-runner] data audit passed: "
        f"internal_groups={len(internal_groups)} "
        f"external_groups={len(external_groups)} overlap=0"
    )


def prepare_source_stratified_oof_splits() -> dict[int, Path]:
    """Create three disjoint validation folds, each containing all four gears."""
    if len(SEEDS) != 3:
        raise RuntimeError(
            "Context OOF profile requires exactly three seeds/models so every "
            "internal vessel is validated exactly once."
        )
    with np.load(INTERNAL_NPZ, allow_pickle=True) as data:
        groups = data["groups"].astype(str)
        sources = data["window_source_labels"].astype(str)
        kinds = data["window_kinds"].astype(str)
        labels = data["y"].astype(np.int64)

    group_source: dict[str, str] = {}
    for group in np.unique(groups):
        values = sorted(set(sources[groups == group].tolist()))
        if len(values) != 1:
            raise RuntimeError(f"Internal group {group} maps to sources={values}")
        group_source[str(group)] = str(values[0])

    fold_groups: list[list[str]] = [[] for _ in range(3)]
    rng = np.random.RandomState(20260623)
    for source in LOCKED_SOURCE_LABELS:
        source_groups = sorted(
            group for group, value in group_source.items() if value == source
        )
        if len(source_groups) < 3:
            raise RuntimeError(
                f"Source {source} needs >=3 unique vessels for 3-fold OOF; "
                f"found={len(source_groups)}."
            )
        rng.shuffle(source_groups)
        for position, group in enumerate(source_groups):
            fold_groups[position % 3].append(group)

    OOF_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    result: dict[int, Path] = {}
    seen_validation: set[str] = set()
    audit_folds = []
    for fold, seed in enumerate(SEEDS):
        validation_groups = set(fold_groups[fold])
        if seen_validation & validation_groups:
            raise RuntimeError("OOF validation groups overlap across folds.")
        seen_validation.update(validation_groups)
        val_mask = np.isin(groups, sorted(validation_groups))
        val_idx = np.where(val_mask)[0].astype(np.int64)
        train_idx = np.where(~val_mask)[0].astype(np.int64)
        train_sources = sorted(set(sources[train_idx].tolist()))
        val_sources = sorted(set(sources[val_idx].tolist()))
        train_attacks = sorted(set(kinds[train_idx].tolist()) - {"normal"})
        val_attacks = sorted(set(kinds[val_idx].tolist()) - {"normal"})
        expected_sources = sorted(LOCKED_SOURCE_LABELS)
        if train_sources != expected_sources or val_sources != expected_sources:
            raise RuntimeError(
                f"Fold {fold} lacks four-source coverage: "
                f"train={train_sources}, val={val_sources}."
            )
        if set(ATTACKS) - set(train_attacks) or set(ATTACKS) - set(val_attacks):
            raise RuntimeError(
                f"Fold {fold} lacks attack coverage: train={train_attacks}, "
                f"val={val_attacks}."
            )
        if set(np.unique(labels[train_idx])) != {0, 1} or set(np.unique(labels[val_idx])) != {0, 1}:
            raise RuntimeError(f"Fold {fold} lacks both binary labels.")
        split_path = OOF_SPLIT_DIR / f"fold_{fold}_seed_{seed}.npz"
        np.savez_compressed(
            split_path,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=np.array([], dtype=np.int64),
            fold=np.array(fold, dtype=np.int64),
            seed=np.array(seed, dtype=np.int64),
        )
        result[int(seed)] = split_path
        audit_folds.append(
            {
                "fold": int(fold),
                "seed": int(seed),
                "train_windows": int(len(train_idx)),
                "validation_windows": int(len(val_idx)),
                "train_groups": int(len(set(groups[train_idx].tolist()))),
                "validation_groups": int(len(validation_groups)),
                "validation_group_ids": sorted(validation_groups),
                "train_sources": train_sources,
                "validation_sources": val_sources,
                "train_attacks": train_attacks,
                "validation_attacks": val_attacks,
            }
        )
    if seen_validation != set(group_source):
        missing = sorted(set(group_source) - seen_validation)
        raise RuntimeError(f"OOF folds do not cover every internal group: {missing}")
    (OOF_SPLIT_DIR / "oof_split_audit.json").write_text(
        json.dumps(
            {
                "protocol": "source_stratified_group_disjoint_3fold_v1",
                "all_groups_covered_once": True,
                "folds": audit_folds,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("[spoofing-runner] source-stratified 3-fold OOF splits ready")
    return result


def _binary_report(y_true: np.ndarray, y_pred: np.ndarray, probability: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    precision, recall, positive_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], zero_division=0
    )
    report = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_spoofing": float(precision[0]),
        "recall_spoofing": float(recall[0]),
        "f1_spoofing": float(positive_f1[0]),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }
    if len(np.unique(y_true)) == 2:
        report["average_precision"] = float(average_precision_score(y_true, probability))
        report["roc_auc"] = float(roc_auc_score(y_true, probability))
    return report


def _safe_logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def fit_internal_oof_policy() -> dict:
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for fold, seed in enumerate(SEEDS):
        path = OUTPUT_ROOT / f"seed_{seed}" / "validation_eval" / "spoofing_scenario_predictions.csv"
        frame = pd.read_csv(path)
        frame["fold"] = int(fold)
        frame["seed"] = int(seed)
        frames.append(frame)
    pooled = pd.concat(frames, ignore_index=True)
    if pooled["scenario_id"].duplicated().any():
        duplicates = pooled.loc[
            pooled["scenario_id"].duplicated(), "scenario_id"
        ].head().tolist()
        raise RuntimeError(f"OOF scenario predictions duplicated: {duplicates}")
    raw = pooled["top10pct_mean_spoofing_probability"].to_numpy(dtype=np.float64)
    y_true = pooled["true_id"].to_numpy(dtype=np.int64)
    with np.load(INTERNAL_NPZ, allow_pickle=True) as data:
        internal_groups = data["groups"].astype(str)
        internal_sources = data["window_source_labels"].astype(str)
    group_to_source = {}
    for group in np.unique(internal_groups):
        values, counts = np.unique(
            internal_sources[internal_groups == group], return_counts=True
        )
        group_to_source[str(group)] = str(values[int(np.argmax(counts))])
    pooled["source_label"] = pooled["source_group"].astype(str).map(group_to_source)
    if pooled["source_label"].isna().any():
        raise RuntimeError("Cannot map all OOF scenarios to a source label.")
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=20260623)
    calibrator.fit(_safe_logit(raw).reshape(-1, 1), y_true)
    calibrated = calibrator.predict_proba(_safe_logit(raw).reshape(-1, 1))[:, 1]
    candidates = np.unique(
        np.concatenate([np.linspace(0.05, 0.95, 181), calibrated])
    )
    best = None
    for threshold in candidates:
        pred = (calibrated >= float(threshold)).astype(np.int64)
        report = _binary_report(y_true, pred, calibrated)
        source_reports = []
        for source in LOCKED_SOURCE_LABELS:
            idx = pooled["source_label"].astype(str).eq(source).to_numpy()
            if not bool(idx.any()):
                raise RuntimeError(
                    f"Pooled OOF predictions contain no scenarios for source={source}."
                )
            source_reports.append(
                _binary_report(y_true[idx], pred[idx], calibrated[idx])
            )
        report["macro_source_f1"] = float(
            np.mean([row["macro_f1"] for row in source_reports])
        )
        report["min_source_recall"] = float(
            np.min([row["recall_spoofing"] for row in source_reports])
        )
        attack_reports = []
        attack_values = pooled["attack_type"].astype(str).str.lower().to_numpy()
        for attack in ATTACKS:
            idx = attack_values == attack
            if not bool(idx.any()):
                raise RuntimeError(
                    f"Pooled OOF predictions contain no scenarios for attack={attack}."
                )
            attack_reports.append(
                _binary_report(y_true[idx], pred[idx], calibrated[idx])
            )
        report["macro_attack_f1"] = float(
            np.mean([row["f1_spoofing"] for row in attack_reports])
        )
        report["min_attack_recall"] = float(
            np.min([row["recall_spoofing"] for row in attack_reports])
        )
        key = (
            int(report["min_source_recall"] >= OOF_MIN_SOURCE_RECALL),
            int(report["min_attack_recall"] >= OOF_MIN_ATTACK_RECALL),
            report["macro_source_f1"],
            report["macro_attack_f1"],
            report["min_source_recall"],
            report["min_attack_recall"],
            report["macro_f1"],
            report["balanced_accuracy"],
            report["recall_spoofing"],
            -abs(float(threshold) - 0.5),
        )
        if best is None or key > best[0]:
            best = (key, float(threshold), report)
    assert best is not None
    threshold = float(best[1])
    pred = (calibrated >= threshold).astype(np.int64)
    pooled["raw_probability"] = raw
    pooled["calibrated_probability"] = calibrated
    pooled["calibrated_pred_id"] = pred
    pooled["calibrated_correct"] = pred == y_true
    pooled.to_csv(OOF_DIR / "pooled_oof_scenario_predictions.csv", index=False)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    save_spoofing_detection_png(
        cm, OOF_DIR / "confusion_matrix.png", normalize=True
    )
    policy = {
        "protocol": "pooled_source_stratified_group_oof_platt_v1",
        "selection_data": "internal_oof_only",
        "external_used_for_selection": False,
        "probability_input": "mean_top_10_percent_spoofing_probability",
        "ensemble_rule": "mean_raw_probability_then_platt",
        "platt_input": "logit_probability",
        "platt_coefficient": float(calibrator.coef_[0, 0]),
        "platt_intercept": float(calibrator.intercept_[0]),
        "threshold": threshold,
        "threshold_objective": (
            f"min_source_recall_floor_{OOF_MIN_SOURCE_RECALL:.2f}_and_"
            f"min_attack_recall_floor_{OOF_MIN_ATTACK_RECALL:.2f}_then_"
            "macro_source_f1_macro_attack_f1_min_recalls_overall_macro_f1"
        ),
        "min_source_recall_floor": float(OOF_MIN_SOURCE_RECALL),
        "min_attack_recall_floor": float(OOF_MIN_ATTACK_RECALL),
        "num_oof_scenarios": int(len(pooled)),
        "metrics": best[2],
        "member_seeds": [int(seed) for seed in SEEDS],
    }
    OOF_POLICY_PATH.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    print(
        "[spoofing-runner] internal OOF policy locked: "
        f"threshold={threshold:.4f} macro_f1={best[2]['macro_f1']:.4f}"
    )
    return policy


def finalize_external_ensemble(policy: dict) -> None:
    EXTERNAL_ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    members = []
    for seed in SEEDS:
        path = OUTPUT_ROOT / f"seed_{seed}" / "external_test_eval" / "spoofing_scenario_predictions.csv"
        frame = pd.read_csv(path).rename(
            columns={
                "top10pct_mean_spoofing_probability": f"probability_seed_{seed}"
            }
        )
        members.append(frame)
    keys = ["scenario_id", "source_group", "attack_type", "true_id", "n_windows"]
    merged = members[0][keys + [f"probability_seed_{SEEDS[0]}"]].copy()
    for seed, frame in zip(SEEDS[1:], members[1:]):
        merged = merged.merge(
            frame[keys + [f"probability_seed_{seed}"]],
            on=keys,
            how="inner",
            validate="one_to_one",
        )
    probability_cols = [f"probability_seed_{seed}" for seed in SEEDS]
    if len(merged) != len(members[0]):
        raise RuntimeError("External ensemble members do not contain identical scenarios.")
    raw_mean = merged[probability_cols].mean(axis=1).to_numpy(dtype=np.float64)
    coefficient = float(policy["platt_coefficient"])
    intercept = float(policy["platt_intercept"])
    calibrated = 1.0 / (
        1.0 + np.exp(-(coefficient * _safe_logit(raw_mean) + intercept))
    )
    threshold = float(policy["threshold"])
    pred = (calibrated >= threshold).astype(np.int64)
    y_true = merged["true_id"].to_numpy(dtype=np.int64)
    merged["ensemble_raw_probability"] = raw_mean
    merged["calibrated_probability"] = calibrated
    merged["pred_id"] = pred
    merged["correct"] = pred == y_true
    with np.load(EXTERNAL_NPZ, allow_pickle=True) as data:
        external_groups = data["groups"].astype(str)
        external_sources = data["window_source_labels"].astype(str)
    external_group_to_source = {}
    for group in np.unique(external_groups):
        values, counts = np.unique(
            external_sources[external_groups == group], return_counts=True
        )
        external_group_to_source[str(group)] = str(values[int(np.argmax(counts))])
    merged["source_label"] = merged["source_group"].astype(str).map(
        external_group_to_source
    )
    if merged["source_label"].isna().any():
        raise RuntimeError("Cannot map all external scenarios to a source label.")
    merged.to_csv(
        EXTERNAL_ENSEMBLE_DIR / "spoofing_scenario_predictions.csv", index=False
    )
    overall = _binary_report(y_true, pred, calibrated)
    attack_rows = []
    attack_values = merged["attack_type"].astype(str).str.lower().to_numpy()
    normal_mask = attack_values == "normal"
    for attack in ATTACKS:
        subset = normal_mask | (attack_values == attack)
        attack_rows.append(
            {
                "attack_type": attack,
                **_binary_report(y_true[subset], pred[subset], calibrated[subset]),
            }
        )
    pd.DataFrame(attack_rows).to_csv(
        EXTERNAL_ENSEMBLE_DIR / "spoofing_attack_metrics.csv", index=False
    )
    source_rows = []
    for source in LOCKED_SOURCE_LABELS:
        subset = merged["source_label"].astype(str).eq(source).to_numpy()
        if not bool(subset.any()):
            raise RuntimeError(
                f"External ensemble contains no scenarios for source={source}."
            )
        source_rows.append(
            {
                "source_label": source,
                **_binary_report(y_true[subset], pred[subset], calibrated[subset]),
            }
        )
    pd.DataFrame(source_rows).to_csv(
        EXTERNAL_ENSEMBLE_DIR / "spoofing_source_metrics.csv", index=False
    )
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    save_spoofing_detection_png(
        cm, EXTERNAL_ENSEMBLE_DIR / "confusion_matrix.png", normalize=True
    )
    summary = {
        "protocol": "locked_internal_oof_calibrated_three_member_ensemble_v1",
        "selection_data": "internal_oof_only",
        "external_used_for_selection": False,
        "external_data": str(EXTERNAL_NPZ),
        "member_seeds": [int(seed) for seed in SEEDS],
        "calibration_policy": str(OOF_POLICY_PATH),
        "threshold": threshold,
        "num_external_scenarios": int(len(merged)),
        "metrics": overall,
        "prediction_table": str(
            EXTERNAL_ENSEMBLE_DIR / "spoofing_scenario_predictions.csv"
        ),
        "attack_metrics_table": str(
            EXTERNAL_ENSEMBLE_DIR / "spoofing_attack_metrics.csv"
        ),
        "source_metrics_table": str(
            EXTERNAL_ENSEMBLE_DIR / "spoofing_source_metrics.csv"
        ),
        "confusion_matrix": str(EXTERNAL_ENSEMBLE_DIR / "confusion_matrix.png"),
    }
    (EXTERNAL_ENSEMBLE_DIR / "eval_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        "[spoofing-runner] final external ensemble complete: "
        f"macro_f1={overall['macro_f1']:.4f} "
        f"recall={overall['recall_spoofing']:.4f}"
    )


def validate_multiseed_split_diversity() -> None:
    data = np.load(INTERNAL_NPZ, allow_pickle=True)
    groups = data["groups"].astype(str)
    rows = []
    validation_sets: list[tuple[str, ...]] = []
    for seed in SEEDS:
        split_path = OUTPUT_ROOT / f"seed_{seed}" / "model_spoofing" / "split_indices.npz"
        split = np.load(split_path, allow_pickle=True)
        train_idx = split["train_idx"].astype(np.int64)
        val_idx = split["val_idx"].astype(np.int64)
        train_groups = set(groups[train_idx].tolist())
        val_groups = set(groups[val_idx].tolist())
        overlap = sorted(train_groups & val_groups)
        validation_sets.append(tuple(sorted(val_groups)))
        rows.append(
            {
                "seed": int(seed),
                "train_groups": sorted(train_groups),
                "validation_groups": sorted(val_groups),
                "group_overlap": overlap,
                "train_windows": int(train_idx.size),
                "validation_windows": int(val_idx.size),
            }
        )

    unique_sets = len(set(validation_sets))
    problems = []
    if any(row["group_overlap"] for row in rows):
        problems.append("train/validation source-group overlap detected")
    if unique_sets != len(SEEDS):
        problems.append(
            f"expected {len(SEEDS)} distinct validation group sets, got {unique_sets}"
        )
    audit = {
        "requested_seeds": list(SEEDS),
        "unique_validation_group_sets": unique_sets,
        "valid": not problems,
        "problems": problems,
        "splits": rows,
    }
    SPLIT_AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if problems:
        raise RuntimeError("; ".join(problems))
    print(
        "[spoofing-runner] split audit passed: "
        f"distinct_validation_sets={unique_sets}/{len(SEEDS)}"
    )


def train_and_validate_internal() -> dict:
    split_paths = prepare_internal()
    for seed in SEEDS:
        run_dir = OUTPUT_ROOT / f"seed_{seed}"
        model_dir = run_dir / "model_spoofing"
        model_path = model_dir / "model.pt"
        val_dir = run_dir / "validation_eval"
        ensure_artifact(
            train_command(seed, model_dir, split_paths[int(seed)]),
            f"seed={seed} train",
            lambda model_dir=model_dir: model_complete(model_dir),
        )
        ensure_artifact(
            eval_command(INTERNAL_NPZ, model_path, val_dir, "val"),
            f"seed={seed} validation",
            lambda val_dir=val_dir: eval_complete(val_dir),
        )

    validate_multiseed_split_diversity()
    return fit_internal_oof_policy()


def evaluate_external(policy: dict | None = None) -> None:
    if policy is None:
        if not OOF_POLICY_PATH.is_file():
            raise RuntimeError(
                "Internal OOF policy belum ada. Jalankan stage internal dulu."
            )
        policy = json.loads(OOF_POLICY_PATH.read_text(encoding="utf-8"))
    # The external set is opened only after the complete model/calibration/
    # threshold policy has been locked from pooled internal OOF predictions.
    prepare_external_after_internal_policy()
    for seed in SEEDS:
        run_dir = OUTPUT_ROOT / f"seed_{seed}"
        model_path = run_dir / "model_spoofing" / "model.pt"
        external_dir = run_dir / "external_test_eval"
        ensure_artifact(
            eval_command(EXTERNAL_NPZ, model_path, external_dir, "all"),
            f"seed={seed} external",
            lambda external_dir=external_dir: eval_complete(external_dir),
        )
    finalize_external_ensemble(policy)


def train_and_evaluate() -> None:
    policy = train_and_validate_internal()
    evaluate_external(policy)


def status() -> None:
    print(f"output: {OUTPUT_ROOT}")
    print(f"internal_generated: {generation_complete(INTERNAL_GENERATED)}")
    print(f"external_generated: {generation_complete(EXTERNAL_GENERATED)}")
    print(f"internal_preprocessed: {INTERNAL_NPZ.is_file()}")
    print(f"external_preprocessed: {EXTERNAL_NPZ.is_file()}")
    print(f"data_audit: {DATA_AUDIT_PATH.is_file()}")
    print(f"split_audit: {SPLIT_AUDIT_PATH.is_file()}")
    print(f"generation_semantics_audit: {SEMANTICS_AUDIT_PATH.is_file()}")
    print(f"oof_splits: {(OOF_SPLIT_DIR / 'oof_split_audit.json').is_file()}")
    print(f"oof_policy: {OOF_POLICY_PATH.is_file()}")
    print(
        "final_external_ensemble: "
        f"{(EXTERNAL_ENSEMBLE_DIR / 'eval_summary.json').is_file()}"
    )
    for seed in SEEDS:
        run_dir = OUTPUT_ROOT / f"seed_{seed}"
        print(
            f"seed {seed}: "
            f"train={model_complete(run_dir / 'model_spoofing')} "
            f"val={eval_complete(run_dir / 'validation_eval')} "
            f"external={eval_complete(run_dir / 'external_test_eval')}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Leakage-safe spoofing baseline, multiseed, and external evaluation."
    )
    parser.add_argument("stage", choices=("prepare", "internal", "external", "run", "status"))
    args = parser.parse_args()
    if args.stage == "status":
        status()
        return 0

    locked = False
    try:
        acquire_lock()
        locked = True
        if args.stage == "prepare":
            prepare()
        elif args.stage == "internal":
            train_and_validate_internal()
        elif args.stage == "external":
            evaluate_external()
        else:
            train_and_evaluate()
        return 0
    except KeyboardInterrupt:
        print("\n[spoofing-runner] dihentikan pengguna")
        return 130
    except Exception as exc:
        print(f"\n[spoofing-runner] FATAL: {type(exc).__name__}: {exc}")
        return 1
    finally:
        if locked:
            LOCK_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
