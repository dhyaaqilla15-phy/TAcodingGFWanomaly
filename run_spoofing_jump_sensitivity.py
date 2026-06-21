from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "Outputs" / "spoofing_tuning01_jump_magnitude_seed42"
LOCK_PATH = OUTPUT_ROOT / ".spoofing_jump_sensitivity.lock"
SUMMARY_PATH = OUTPUT_ROOT / "jump_sensitivity_summary.csv"

SIMULATION_SEED = 42
MODEL_SEED = 42
EPOCHS = 50
DRIFT_DEG = 0.05
JUMP_CONFIGS = {
    "jump_010deg": 0.10,
    "jump_030deg": 0.30,
    "jump_050deg": 0.50,
    "jump_080deg": 0.80,
}


def run(command: list[str], label: str) -> int:
    print(f"\n[spoof-jump] {label}")
    print("[spoof-jump] " + subprocess.list2cmdline(command))
    started = time.perf_counter()
    status = subprocess.run(command, cwd=ROOT, check=False).returncode
    print(
        f"[spoof-jump] {label} exit={status} "
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
            raise RuntimeError(f"Runner lain masih aktif dengan PID {old_pid}.")
        LOCK_PATH.unlink(missing_ok=True)
    fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))


def config_payload(name: str, jump_deg: float) -> dict:
    return {
        "name": name,
        "simulation_seed": SIMULATION_SEED,
        "model_seed": MODEL_SEED,
        "drift_deg": DRIFT_DEG,
        "jump_deg": float(jump_deg),
        "points_per_attack": 240,
        "normal_keep_frac": 0.50,
        "seq_len": 120,
        "stride": 6,
        "gap_seconds": 10800,
        "spoofing_window_threshold": 0.20,
        "epochs": EPOCHS,
        "external_test_used": False,
    }


def ensure_config(run_dir: Path, payload: dict) -> None:
    path = run_dir / "experiment_config.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != payload:
            raise RuntimeError(
                f"Konfigurasi {run_dir.name} berubah. Gunakan output directory "
                "baru agar hasil lama tidak tercampur."
            )
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def generation_complete(generated_dir: Path) -> bool:
    combined = generated_dir / "spoofed_all.csv"
    audits = list((generated_dir / "summaries").glob("magnitude_*.csv"))
    return combined.is_file() and combined.stat().st_size > 0 and bool(audits)


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
            "spoofing_attack_metrics.csv",
            "spoofing_sequence_predictions.csv",
            "spoofing_scenario_predictions.csv",
        )
    )


def ensure_artifact(command: list[str], label: str, complete) -> None:
    if complete():
        print(f"[spoof-jump] {label}: complete, skip")
        return
    status = run(command, label)
    if not complete():
        raise RuntimeError(f"{label} tidak lengkap (exit={status}).")
    if status != 0:
        print(
            f"[spoof-jump] WARNING: {label} exit={status}, tetapi artefak "
            "terverifikasi lengkap; lanjut."
        )


def generation_command(generated_dir: Path, jump_deg: float) -> list[str]:
    return [
        sys.executable,
        "main.py",
        "make_spoofing",
        "--input_path",
        "Dataset",
        "--out_dir",
        str(generated_dir),
        "--attacks",
        "gradual_drift",
        "location_jump",
        "--seed",
        str(SIMULATION_SEED),
        "--limit_rows",
        "300000",
        "--normal_keep_frac",
        "0.50",
        "--exclude_labels",
        "pole_and_line",
        "trollers",
        "--max_vessels_per_file",
        "20",
        "--min_points_per_vessel",
        "300",
        "--points_per_attack",
        "240",
        "--drift_lat_deg",
        str(DRIFT_DEG),
        "--drift_lon_deg",
        str(DRIFT_DEG),
        "--jump_lat_deg",
        str(jump_deg),
        "--jump_lon_deg",
        str(jump_deg),
        "--combine_outputs",
    ]


def preprocess_command(generated_dir: Path, prep_dir: Path) -> list[str]:
    return [
        sys.executable,
        "main.py",
        "preprocess",
        "--data_dir",
        str(generated_dir),
        "--out_dir",
        str(prep_dir),
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


def train_command(npz_path: Path, model_dir: Path) -> list[str]:
    return [
        sys.executable,
        "main.py",
        "train",
        "--data_npz",
        str(npz_path),
        "--out_dir",
        str(model_dir),
        "--device",
        "cuda",
        "--random_state",
        str(MODEL_SEED),
        "--split_random_state",
        str(MODEL_SEED),
        "--train_random_state",
        str(MODEL_SEED),
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


def eval_command(npz_path: Path, model_path: Path, eval_dir: Path) -> list[str]:
    return [
        sys.executable,
        "main.py",
        "eval",
        "--data_npz",
        str(npz_path),
        "--model_path",
        str(model_path),
        "--out_dir",
        str(eval_dir),
        "--device",
        "cuda",
        "--batch_size",
        "256",
        "--eval_split",
        "val",
    ]


def validate_npz(npz_path: Path) -> None:
    data = np.load(npz_path, allow_pickle=True)
    y = data["y"].astype(np.int64)
    groups = data["groups"].astype(str)
    classes = sorted(np.unique(y).astype(int).tolist())
    kinds = set(data["window_kinds"].astype(str).tolist())
    features = set(data["feature_cols"].astype(str).tolist())
    if classes != [0, 1]:
        raise RuntimeError(f"NPZ harus memiliki dua kelas; ditemukan {classes}.")
    if not {"gradual_drift", "location_jump"}.issubset(kinds):
        raise RuntimeError(f"Jenis serangan tidak lengkap: {sorted(kinds)}.")
    if {"distance_from_shore", "distance_from_port"} & features:
        raise RuntimeError("Fitur lokasi absolut tidak boleh masuk spoofing.")
    positive_count = int(np.sum(y == 1))
    positive_ratio = float(np.mean(y == 1))
    positive_groups = int(np.unique(groups[y == 1]).size)
    if positive_count < 200 or positive_groups < 5:
        raise RuntimeError(
            "Window spoofing tidak cukup: "
            f"count={positive_count}, source_groups={positive_groups}."
        )
    if not 0.01 <= positive_ratio <= 0.20:
        raise RuntimeError(
            "Prevalensi validation tidak sesuai desain sensitivity study: "
            f"{positive_ratio:.3f}."
        )


def preprocessed_complete(npz_path: Path) -> bool:
    if not npz_path.is_file() or npz_path.stat().st_size <= 0:
        return False
    try:
        validate_npz(npz_path)
        return True
    except (OSError, KeyError, ValueError, RuntimeError):
        return False


def prepare_one(name: str, jump_deg: float) -> tuple[Path, Path, Path, Path]:
    run_dir = OUTPUT_ROOT / name
    generated_dir = run_dir / "generated_internal"
    prep_dir = run_dir / "data_internal_trainval"
    npz_path = prep_dir / "processed_spoofing.npz"
    model_dir = run_dir / "model_spoofing"
    eval_dir = run_dir / "validation_eval"
    ensure_config(run_dir, config_payload(name, jump_deg))
    ensure_artifact(
        generation_command(generated_dir, jump_deg),
        f"{name} generate",
        lambda: generation_complete(generated_dir),
    )
    ensure_artifact(
        preprocess_command(generated_dir, prep_dir),
        f"{name} preprocess",
        lambda: preprocessed_complete(npz_path),
    )
    validate_npz(npz_path)
    return npz_path, model_dir, eval_dir, generated_dir


def run_all() -> None:
    for name, jump_deg in JUMP_CONFIGS.items():
        npz_path, model_dir, eval_dir, _ = prepare_one(name, jump_deg)
        ensure_artifact(
            train_command(npz_path, model_dir),
            f"{name} train",
            lambda model_dir=model_dir: model_complete(model_dir),
        )
        ensure_artifact(
            eval_command(npz_path, model_dir / "model.pt", eval_dir),
            f"{name} validation",
            lambda eval_dir=eval_dir: eval_complete(eval_dir),
        )
        write_summary()


def magnitude_stats(generated_dir: Path) -> dict[str, float]:
    frames = [pd.read_csv(path) for path in sorted(
        (generated_dir / "summaries").glob("magnitude_*.csv")
    )]
    if not frames:
        return {}
    data = pd.concat(frames, ignore_index=True)
    values = data.loc[
        data["attack_type"].astype(str).eq("location_jump"),
        "attack_displacement_km",
    ].astype(float)
    if values.empty:
        return {}
    return {
        "jump_km_min": float(values.min()),
        "jump_km_median": float(values.median()),
        "jump_km_max": float(values.max()),
    }


def read_attack_metric(path: Path, attack: str) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("attack_type") == attack:
                return row
    return {}


def write_summary() -> None:
    rows = []
    for name, jump_deg in JUMP_CONFIGS.items():
        run_dir = OUTPUT_ROOT / name
        eval_dir = run_dir / "validation_eval"
        summary_path = eval_dir / "eval_summary.json"
        attack_path = eval_dir / "spoofing_attack_metrics.csv"
        if not summary_path.is_file() or not attack_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        attack = read_attack_metric(attack_path, "location_jump")
        ranking = summary.get("binary_ranking_metrics") or {}
        row = {
            "config": name,
            "jump_nominal_deg": jump_deg,
            **magnitude_stats(run_dir / "generated_internal"),
            "val_macro_f1": summary["metrics_seq"]["macro_f1"],
            "val_balanced_acc": summary["metrics_seq"]["balanced_acc"],
            "val_average_precision": ranking.get("average_precision"),
            "val_roc_auc": ranking.get("roc_auc"),
            "jump_precision": attack.get("precision"),
            "jump_recall": attack.get("recall"),
            "jump_f1": attack.get("f1"),
            "jump_positive_windows": attack.get("positive_windows"),
            "external_test_used": False,
        }
        rows.append(row)
    if rows:
        pd.DataFrame(rows).sort_values("jump_nominal_deg").to_csv(
            SUMMARY_PATH,
            index=False,
        )
        print(f"[spoof-jump] summary -> {SUMMARY_PATH}")


def status() -> None:
    print(f"output: {OUTPUT_ROOT}")
    for name, jump_deg in JUMP_CONFIGS.items():
        run_dir = OUTPUT_ROOT / name
        generated_dir = run_dir / "generated_internal"
        npz_path = run_dir / "data_internal_trainval" / "processed_spoofing.npz"
        print(
            f"{name} ({jump_deg:.2f} deg): "
            f"generated={generation_complete(generated_dir)} "
            f"preprocessed={npz_path.is_file()} "
            f"train={model_complete(run_dir / 'model_spoofing')} "
            f"validation={eval_complete(run_dir / 'validation_eval')}"
        )
    write_summary()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Internal-validation sensitivity study for spoofing jump magnitude."
    )
    parser.add_argument("stage", choices=("prepare", "run", "status", "summary"))
    args = parser.parse_args()
    if args.stage == "status":
        status()
        return 0
    if args.stage == "summary":
        write_summary()
        return 0

    locked = False
    try:
        acquire_lock()
        locked = True
        if args.stage == "prepare":
            for name, jump_deg in JUMP_CONFIGS.items():
                prepare_one(name, jump_deg)
        else:
            run_all()
        return 0
    except KeyboardInterrupt:
        print("\n[spoof-jump] dihentikan pengguna")
        return 130
    except Exception as exc:
        print(f"\n[spoof-jump] FATAL: {type(exc).__name__}: {exc}")
        return 1
    finally:
        if locked:
            LOCK_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
