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

import run_spoofing_jump_sensitivity as shared


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "Outputs" / "spoofing_tuning02_drift_magnitude_seed42"
SUMMARY_PATH = OUTPUT_ROOT / "drift_sensitivity_summary.csv"
LOCK_PATH = OUTPUT_ROOT / ".spoofing_drift_sensitivity.lock"
REUSED_005_ROOT = OUTPUT_ROOT / "drift_005deg"

SIMULATION_SEED = 42
MODEL_SEED = 42
EPOCHS = 50
JUMP_DEG = 0.50
DRIFT_CONFIGS = {
    "drift_001deg": 0.01,
    "drift_003deg": 0.03,
    "drift_005deg": 0.05,
    "drift_008deg": 0.08,
}


def run_dir_for(name: str) -> Path:
    if name == "drift_005deg":
        return REUSED_005_ROOT
    return OUTPUT_ROOT / name


def run(command: list[str], label: str) -> int:
    print(f"\n[spoof-drift] {label}")
    print("[spoof-drift] " + subprocess.list2cmdline(command))
    started = time.perf_counter()
    status = subprocess.run(command, cwd=ROOT, check=False).returncode
    print(
        f"[spoof-drift] {label} exit={status} "
        f"duration={time.perf_counter() - started:.1f}s"
    )
    return status


def acquire_lock() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        try:
            old_pid = int(LOCK_PATH.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            old_pid = -1
        if shared.process_is_running(old_pid):
            raise RuntimeError(f"Runner drift lain masih aktif dengan PID {old_pid}.")
        LOCK_PATH.unlink(missing_ok=True)
    fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))


def config_payload(name: str, drift_deg: float) -> dict:
    return {
        "name": name,
        "simulation_seed": SIMULATION_SEED,
        "model_seed": MODEL_SEED,
        "drift_deg": float(drift_deg),
        "jump_deg": JUMP_DEG,
        "points_per_attack": 240,
        "normal_keep_frac": 0.50,
        "seq_len": 120,
        "stride": 6,
        "gap_seconds": 10800,
        "spoofing_window_threshold": 0.20,
        "epochs": EPOCHS,
        "external_test_used": False,
        "reused_existing_run": name == "drift_005deg",
    }


def generation_command(generated_dir: Path, drift_deg: float) -> list[str]:
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
        str(drift_deg),
        "--drift_lon_deg",
        str(drift_deg),
        "--jump_lat_deg",
        str(JUMP_DEG),
        "--jump_lon_deg",
        str(JUMP_DEG),
        "--combine_outputs",
    ]


def ensure_config(run_dir: Path, payload: dict) -> None:
    if run_dir == REUSED_005_ROOT:
        existing = json.loads((run_dir / "experiment_config.json").read_text(encoding="utf-8"))
        comparable = {
            key: existing[key]
            for key in (
                "simulation_seed", "model_seed", "drift_deg", "jump_deg",
                "points_per_attack", "normal_keep_frac", "seq_len", "stride",
                "gap_seconds", "spoofing_window_threshold", "epochs",
                "external_test_used",
            )
        }
        expected = {key: payload[key] for key in comparable}
        if comparable != expected:
            raise RuntimeError("Existing drift 0.05 run is not configuration-compatible.")
        return
    shared.ensure_config(run_dir, payload)


def prepare_and_run(name: str, drift_deg: float) -> None:
    run_dir = run_dir_for(name)
    ensure_config(run_dir, config_payload(name, drift_deg))
    if run_dir == REUSED_005_ROOT:
        if not shared.model_complete(run_dir / "model_spoofing"):
            raise RuntimeError("Reusable drift 0.05 checkpoint is incomplete.")
        if not shared.eval_complete(run_dir / "validation_eval"):
            raise RuntimeError("Reusable drift 0.05 evaluation is incomplete.")
        print("[spoof-drift] drift_005deg: reuse completed jump_050deg run")
        return

    generated_dir = run_dir / "generated_internal"
    prep_dir = run_dir / "data_internal_trainval"
    npz_path = prep_dir / "processed_spoofing.npz"
    model_dir = run_dir / "model_spoofing"
    eval_dir = run_dir / "validation_eval"

    if not shared.generation_complete(generated_dir):
        status = run(generation_command(generated_dir, drift_deg), f"{name} generate")
        if not shared.generation_complete(generated_dir):
            raise RuntimeError(f"{name} generation incomplete (exit={status}).")
    if not shared.preprocessed_complete(npz_path):
        status = run(shared.preprocess_command(generated_dir, prep_dir), f"{name} preprocess")
        if not shared.preprocessed_complete(npz_path):
            raise RuntimeError(f"{name} preprocessing incomplete (exit={status}).")
    shared.validate_npz(npz_path)
    if not shared.model_complete(model_dir):
        status = run(shared.train_command(npz_path, model_dir), f"{name} train")
        if not shared.model_complete(model_dir):
            raise RuntimeError(f"{name} training incomplete (exit={status}).")
        if status != 0:
            print(
                f"[spoof-drift] WARNING: {name} train exit={status}, but all "
                "required checkpoint artifacts are complete; continuing."
            )
    if not shared.eval_complete(eval_dir):
        status = run(
            shared.eval_command(npz_path, model_dir / "model.pt", eval_dir),
            f"{name} validation",
        )
        if not shared.eval_complete(eval_dir):
            raise RuntimeError(f"{name} validation incomplete (exit={status}).")

    if not shared.model_complete(model_dir) or not shared.eval_complete(eval_dir):
        raise RuntimeError(f"Artifacts incomplete after run: {name}")


def drift_stats(generated_dir: Path) -> dict[str, float]:
    frames = [
        pd.read_csv(path)
        for path in sorted((generated_dir / "summaries").glob("magnitude_*.csv"))
    ]
    if not frames:
        return {}
    data = pd.concat(frames, ignore_index=True)
    drift = data[data["attack_type"].astype(str).eq("gradual_drift")]
    output: dict[str, float] = {}
    for column in ("attack_displacement_km", "attack_duration_hours", "attack_drift_rate_kmh"):
        if column not in drift or drift[column].dropna().empty:
            continue
        values = pd.to_numeric(drift[column], errors="coerce").dropna()
        output[f"{column}_min"] = float(values.min())
        output[f"{column}_median"] = float(values.median())
        output[f"{column}_max"] = float(values.max())
    return output


def read_attack_metric(path: Path, attack: str) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("attack_type") == attack:
                return row
    return {}


def write_summary() -> None:
    rows = []
    for name, drift_deg in DRIFT_CONFIGS.items():
        run_dir = run_dir_for(name)
        eval_dir = run_dir / "validation_eval"
        summary_path = eval_dir / "eval_summary.json"
        attack_path = eval_dir / "spoofing_attack_metrics.csv"
        if not summary_path.is_file() or not attack_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        attack = read_attack_metric(attack_path, "gradual_drift")
        ranking = summary.get("binary_ranking_metrics") or {}
        rows.append(
            {
                "config": name,
                "drift_nominal_deg": drift_deg,
                "jump_nominal_deg": JUMP_DEG,
                **drift_stats(run_dir / "generated_internal"),
                "val_macro_f1": summary["metrics_seq"]["macro_f1"],
                "val_balanced_acc": summary["metrics_seq"]["balanced_acc"],
                "val_average_precision": ranking.get("average_precision"),
                "val_roc_auc": ranking.get("roc_auc"),
                "drift_precision": attack.get("precision"),
                "drift_recall": attack.get("recall"),
                "drift_f1": attack.get("f1"),
                "drift_positive_windows": attack.get("positive_windows"),
                "reused_existing_run": name == "drift_005deg",
                "external_test_used": False,
            }
        )
    if rows:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).sort_values("drift_nominal_deg").to_csv(
            SUMMARY_PATH, index=False
        )
        print(f"[spoof-drift] summary -> {SUMMARY_PATH}")


def status() -> None:
    for name, drift_deg in DRIFT_CONFIGS.items():
        run_dir = run_dir_for(name)
        print(
            f"{name} ({drift_deg:.2f} deg): "
            f"generated={shared.generation_complete(run_dir / 'generated_internal')} "
            f"preprocessed={shared.preprocessed_complete(run_dir / 'data_internal_trainval' / 'processed_spoofing.npz')} "
            f"train={shared.model_complete(run_dir / 'model_spoofing')} "
            f"validation={shared.eval_complete(run_dir / 'validation_eval')}"
        )
    write_summary()


def main() -> int:
    parser = argparse.ArgumentParser(description="Gradual-drift magnitude sensitivity study.")
    parser.add_argument("stage", choices=("run", "status", "summary"))
    args = parser.parse_args()
    if args.stage == "status":
        status()
        return 0
    if args.stage == "summary":
        write_summary()
        return 0

    acquire_lock()
    try:
        for name, drift_deg in DRIFT_CONFIGS.items():
            prepare_and_run(name, drift_deg)
            write_summary()
    finally:
        LOCK_PATH.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
