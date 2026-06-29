from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TUNING_ROOT = Path(
    os.environ.get(
        "SPOOFING_TUNING_ROOT",
        ROOT / "Outputs" / "spoofing_hparam_tuning01_internal_oof",
    )
)
if not TUNING_ROOT.is_absolute():
    TUNING_ROOT = (ROOT / TUNING_ROOT).resolve()

TRIALS_ROOT = TUNING_ROOT / "trials"
SUMMARY_CSV = TUNING_ROOT / "tuning_summary.csv"
SUMMARY_JSON = TUNING_ROOT / "tuning_summary.json"
WINNER_JSON = TUNING_ROOT / "winner_internal_only.json"
LOCK_PATH = TUNING_ROOT / ".spoofing_hparam_tuning.lock"


@dataclass(frozen=True)
class Config:
    name: str
    hidden_size: int = 128
    input_proj_dim: int = 128
    embed_dim: int = 192
    dropout: float = 0.35
    weight_decay: float = 0.0013
    focal_gamma: float = 1.2
    lr: float = 0.00025
    min_source_recall: float = 0.50
    min_attack_recall: float = 0.20


CONFIGS = (
    Config("baseline_h128"),
    Config("drop040_wd002", dropout=0.40, weight_decay=0.0020),
    Config("drop030_wd001", dropout=0.30, weight_decay=0.0010),
    Config("lr1e4_drop035", lr=0.00010, dropout=0.35),
    Config("lr35e5_drop035", lr=0.00035, dropout=0.35),
    Config(
        "compact_h96_reg",
        hidden_size=96,
        input_proj_dim=96,
        embed_dim=128,
        dropout=0.45,
        weight_decay=0.0030,
        focal_gamma=1.0,
    ),
    Config(
        "h160_drop040",
        hidden_size=160,
        input_proj_dim=128,
        embed_dim=192,
        dropout=0.40,
        weight_decay=0.0020,
    ),
    Config("nofocal_drop035", focal_gamma=0.0, dropout=0.35),
)
CONFIG_BY_NAME = {config.name: config for config in CONFIGS}


def selected_configs() -> tuple[Config, ...]:
    raw = os.environ.get("SPOOFING_TUNING_CONFIGS", "").strip()
    if not raw:
        return CONFIGS
    requested = [name.strip() for name in raw.replace(";", ",").split(",") if name.strip()]
    missing = [name for name in requested if name not in CONFIG_BY_NAME]
    if missing:
        raise ValueError(f"Unknown SPOOFING_TUNING_CONFIGS: {missing}")
    return tuple(CONFIG_BY_NAME[name] for name in requested)


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
    TUNING_ROOT.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        try:
            old_pid = int(LOCK_PATH.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            old_pid = -1
        if process_is_running(old_pid):
            raise RuntimeError(
                f"Runner tuning spoofing lain masih aktif dengan PID {old_pid}."
            )
        LOCK_PATH.unlink(missing_ok=True)
    fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))


def trial_root(config: Config) -> Path:
    return TRIALS_ROOT / config.name


def policy_path(config: Config) -> Path:
    return trial_root(config) / "internal_oof_policy" / "platt_scenario_policy.json"


def trial_complete(config: Config) -> bool:
    path = policy_path(config)
    if not path.is_file():
        return False
    try:
        policy = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(policy.get("metrics")) and policy.get("external_used_for_selection") is False


def env_for_config(config: Config) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "SPOOFING_OUTPUT_ROOT": str(trial_root(config)),
            "SPOOFING_HIDDEN_SIZE": str(config.hidden_size),
            "SPOOFING_INPUT_PROJ_DIM": str(config.input_proj_dim),
            "SPOOFING_EMBED_DIM": str(config.embed_dim),
            "SPOOFING_DROPOUT": str(config.dropout),
            "SPOOFING_WEIGHT_DECAY": str(config.weight_decay),
            "SPOOFING_FOCAL_GAMMA": str(config.focal_gamma),
            "SPOOFING_LR": str(config.lr),
            "SPOOFING_OOF_MIN_SOURCE_RECALL": str(config.min_source_recall),
            "SPOOFING_OOF_MIN_ATTACK_RECALL": str(config.min_attack_recall),
        }
    )
    return env


def run_trial(config: Config) -> None:
    if trial_complete(config):
        print(f"[spoofing-hparam] {config.name}: complete, skip")
        return
    command = [sys.executable, "run_spoofing_multiseed.py", "internal"]
    print("\n" + "=" * 72)
    print(f"[spoofing-hparam] trial={config.name}")
    print("[spoofing-hparam] " + subprocess.list2cmdline(command))
    print(f"[spoofing-hparam] output={trial_root(config)}")
    started = time.perf_counter()
    status = subprocess.run(command, cwd=ROOT, env=env_for_config(config), check=False).returncode
    print(
        f"[spoofing-hparam] trial={config.name} exit={status} "
        f"duration={time.perf_counter() - started:.1f}s"
    )
    if status != 0 or not trial_complete(config):
        raise RuntimeError(f"Trial {config.name} tidak lengkap (exit={status}).")


def load_trial_row(config: Config) -> dict[str, object]:
    policy = json.loads(policy_path(config).read_text(encoding="utf-8"))
    metrics = policy["metrics"]
    row: dict[str, object] = {
        **asdict(config),
        "output_root": str(trial_root(config)),
        "threshold": float(policy["threshold"]),
        "threshold_objective": str(policy["threshold_objective"]),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "precision_spoofing": float(metrics["precision_spoofing"]),
        "recall_spoofing": float(metrics["recall_spoofing"]),
        "f1_spoofing": float(metrics["f1_spoofing"]),
        "average_precision": float(metrics.get("average_precision", 0.0)),
        "roc_auc": float(metrics.get("roc_auc", 0.0)),
        "macro_source_f1": float(metrics.get("macro_source_f1", 0.0)),
        "min_source_recall": float(metrics.get("min_source_recall", 0.0)),
        "macro_attack_f1": float(metrics.get("macro_attack_f1", 0.0)),
        "min_attack_recall": float(metrics.get("min_attack_recall", 0.0)),
        "tp": int(metrics["tp"]),
        "fp": int(metrics["fp"]),
        "fn": int(metrics["fn"]),
        "tn": int(metrics["tn"]),
    }
    return row


def selection_key(row: dict[str, object]) -> tuple[float, ...]:
    return (
        float(row["min_attack_recall"]),
        float(row["min_source_recall"]),
        float(row["macro_attack_f1"]),
        float(row["macro_source_f1"]),
        float(row["macro_f1"]),
        float(row["average_precision"]),
        float(row["roc_auc"]),
    )


def write_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    rows_sorted = sorted(rows, key=selection_key, reverse=True)
    TUNING_ROOT.mkdir(parents=True, exist_ok=True)
    if rows_sorted:
        with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows_sorted[0].keys()))
            writer.writeheader()
            writer.writerows(rows_sorted)
    summary = {
        "selection_data": "internal_oof_only",
        "external_used_for_selection": False,
        "selection_key": [
            "min_attack_recall",
            "min_source_recall",
            "macro_attack_f1",
            "macro_source_f1",
            "macro_f1",
            "average_precision",
            "roc_auc",
        ],
        "num_trials": len(rows_sorted),
        "trials": rows_sorted,
        "winner": rows_sorted[0] if rows_sorted else None,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    WINNER_JSON.write_text(json.dumps(summary["winner"], indent=2), encoding="utf-8")
    return summary


def run_all() -> None:
    configs = selected_configs()
    for config in configs:
        run_trial(config)
    rows = [load_trial_row(config) for config in configs]
    summary = write_summary(rows)
    winner = summary["winner"]
    if winner is None:
        raise RuntimeError("Tidak ada trial tuning yang selesai.")
    print("\n[spoofing-hparam] winner internal-only:")
    print(json.dumps(winner, indent=2))
    print("\n[spoofing-hparam] final external command for locked winner:")
    print(
        f'SPOOFING_OUTPUT_ROOT="{winner["output_root"]}" '
        "python run_spoofing_multiseed.py external"
    )


def status() -> None:
    print(f"tuning_root: {TUNING_ROOT}")
    for config in selected_configs():
        print(f"{config.name}: complete={trial_complete(config)} output={trial_root(config)}")
    print(f"summary_csv: {SUMMARY_CSV.is_file()}")
    print(f"winner_json: {WINNER_JSON.is_file()}")
    if WINNER_JSON.is_file():
        print(WINNER_JSON.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Internal-only hyperparameter tuning for spoofing."
    )
    parser.add_argument("stage", choices=("run", "status"))
    args = parser.parse_args()
    if args.stage == "status":
        status()
        return 0
    locked = False
    try:
        acquire_lock()
        locked = True
        run_all()
        return 0
    finally:
        if locked:
            LOCK_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
