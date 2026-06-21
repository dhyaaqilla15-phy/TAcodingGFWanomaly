from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "Outputs" / "spoofing_baseline01_identifiable_multiseed"
INTERNAL_GENERATED = OUTPUT_ROOT / "_generated_internal"
EXTERNAL_GENERATED = OUTPUT_ROOT / "_generated_external"
INTERNAL_PREP = OUTPUT_ROOT / "_data_internal_trainval"
EXTERNAL_PREP = OUTPUT_ROOT / "_data_external_test"
INTERNAL_NPZ = INTERNAL_PREP / "processed_spoofing.npz"
EXTERNAL_NPZ = EXTERNAL_PREP / "processed_spoofing.npz"
LOCK_PATH = OUTPUT_ROOT / ".spoofing_multiseed.lock"
DATA_AUDIT_PATH = OUTPUT_ROOT / "spoofing_data_audit.json"

SEEDS = (42, 43, 44, 45, 46)
EPOCHS = 50


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
    return (
        path.is_file()
        and path.stat().st_size > 0
        and bool(magnitude_audits)
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
        return (
            bool(history)
            and int(history[-1]["epoch"]) >= EPOCHS
            and int(config["epochs"]) == EPOCHS
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def eval_complete(eval_dir: Path) -> bool:
    return all(
        (eval_dir / name).is_file()
        for name in (
            "eval_summary.json",
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            "spoofing_attack_metrics.csv",
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
        "gradual_drift",
        "location_jump",
        "--seed",
        str(seed),
        "--limit_rows",
        str(limit_rows),
        "--normal_keep_frac",
        "0.50",
        "--exclude_labels",
        "pole_and_line",
        "trollers",
        "--max_vessels_per_file",
        str(max_vessels_per_file),
        "--min_points_per_vessel",
        "300",
        "--points_per_attack",
        "240",
        "--drift_lat_deg",
        "0.08",
        "--drift_lon_deg",
        "0.08",
        "--jump_lat_deg",
        "0.70",
        "--jump_lon_deg",
        "0.70",
        "--combine_outputs",
    ]


def preprocess_command(data_dir: Path, out_dir: Path) -> list[str]:
    return [
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
        "30000",
        "--spoofing_window_threshold",
        "0.20",
        "--exclude_location_features",
    ]


def train_command(seed: int, model_dir: Path) -> list[str]:
    return [
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
        "--epochs",
        str(EPOCHS),
        "--batch_size",
        "128",
        "--lr",
        "0.00025",
        "--hidden_size",
        "256",
        "--num_layers",
        "2",
        "--input_proj_dim",
        "128",
        "--embed_dim",
        "256",
        "--dropout",
        "0.30",
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
        "--early_stop_patience",
        "90",
        "--geo_aux_weight",
        "0",
    ]


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


def prepare() -> None:
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
    ensure_artifact(
        preprocess_command(INTERNAL_GENERATED, INTERNAL_PREP),
        "preprocess internal spoofing",
        lambda: INTERNAL_NPZ.is_file(),
    )
    ensure_artifact(
        preprocess_command(EXTERNAL_GENERATED, EXTERNAL_PREP),
        "preprocess external spoofing",
        lambda: EXTERNAL_NPZ.is_file(),
    )
    validate_preprocessed_data()


def validate_preprocessed_data() -> None:
    internal = np.load(INTERNAL_NPZ, allow_pickle=True)
    external = np.load(EXTERNAL_NPZ, allow_pickle=True)
    internal_groups = set(internal["groups"].astype(str).tolist())
    external_groups = set(external["groups"].astype(str).tolist())
    overlap = sorted(internal_groups & external_groups)
    internal_features = internal["feature_cols"].astype(str).tolist()
    external_features = external["feature_cols"].astype(str).tolist()
    internal_classes = sorted(np.unique(internal["y"]).astype(int).tolist())
    external_classes = sorted(np.unique(external["y"]).astype(int).tolist())
    internal_positive_ratio = float(np.mean(internal["y"] == 1))
    external_positive_ratio = float(np.mean(external["y"] == 1))
    internal_attacks = sorted(
        set(internal["window_kinds"].astype(str).tolist()) - {"normal"}
    )
    external_attacks = sorted(
        set(external["window_kinds"].astype(str).tolist()) - {"normal"}
    )
    problems = []
    if overlap:
        problems.append(f"source MMSI overlap internal/external: {overlap[:10]}")
    if internal_features != external_features:
        problems.append("internal/external feature schema differs")
    if internal_classes != [0, 1] or external_classes != [0, 1]:
        problems.append(
            f"both classes required; internal={internal_classes} "
            f"external={external_classes}"
        )
    if not 0.10 <= internal_positive_ratio <= 0.90:
        problems.append(
            "internal positive-window ratio unhealthy: "
            f"{internal_positive_ratio:.3f}"
        )
    if not 0.10 <= external_positive_ratio <= 0.90:
        problems.append(
            "external positive-window ratio unhealthy: "
            f"{external_positive_ratio:.3f}"
        )
    expected_attacks = {"gradual_drift", "location_jump"}
    if not expected_attacks.issubset(internal_attacks):
        problems.append(f"internal attacks incomplete: {internal_attacks}")
    if not expected_attacks.issubset(external_attacks):
        problems.append(f"external attacks incomplete: {external_attacks}")

    audit = {
        "internal_windows": int(internal["y"].shape[0]),
        "external_windows": int(external["y"].shape[0]),
        "internal_source_groups": len(internal_groups),
        "external_source_groups": len(external_groups),
        "source_group_overlap": overlap,
        "internal_class_counts": np.bincount(
            internal["y"].astype(np.int64), minlength=2
        ).astype(int).tolist(),
        "external_class_counts": np.bincount(
            external["y"].astype(np.int64), minlength=2
        ).astype(int).tolist(),
        "internal_positive_ratio": internal_positive_ratio,
        "external_positive_ratio": external_positive_ratio,
        "internal_attacks": internal_attacks,
        "external_attacks": external_attacks,
        "feature_count": len(internal_features),
        "location_features_used": any(
            name in internal_features
            for name in ("distance_from_shore", "distance_from_port")
        ),
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


def train_and_evaluate() -> None:
    prepare()
    for seed in SEEDS:
        run_dir = OUTPUT_ROOT / f"seed_{seed}"
        model_dir = run_dir / "model_spoofing"
        model_path = model_dir / "model.pt"
        val_dir = run_dir / "validation_eval"
        external_dir = run_dir / "external_test_eval"
        ensure_artifact(
            train_command(seed, model_dir),
            f"seed={seed} train",
            lambda model_dir=model_dir: model_complete(model_dir),
        )
        ensure_artifact(
            eval_command(INTERNAL_NPZ, model_path, val_dir, "val"),
            f"seed={seed} validation",
            lambda val_dir=val_dir: eval_complete(val_dir),
        )
        ensure_artifact(
            eval_command(EXTERNAL_NPZ, model_path, external_dir, "all"),
            f"seed={seed} external",
            lambda external_dir=external_dir: eval_complete(external_dir),
        )


def status() -> None:
    print(f"output: {OUTPUT_ROOT}")
    print(f"internal_generated: {generation_complete(INTERNAL_GENERATED)}")
    print(f"external_generated: {generation_complete(EXTERNAL_GENERATED)}")
    print(f"internal_preprocessed: {INTERNAL_NPZ.is_file()}")
    print(f"external_preprocessed: {EXTERNAL_NPZ.is_file()}")
    print(f"data_audit: {DATA_AUDIT_PATH.is_file()}")
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
    parser.add_argument("stage", choices=("prepare", "run", "status"))
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
