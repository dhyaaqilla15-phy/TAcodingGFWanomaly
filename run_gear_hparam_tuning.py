from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "Outputs" / "gear_tuning06_internal_hparam_gap12h_opfilter"
BASELINE_ROOT = (
    ROOT
    / "Outputs"
    / "gear_tuning04_gap12h_opfilter_1to12_geo0_multiseed"
)
INTERNAL_DIR = BASELINE_ROOT / "_data_internal_trainval"
EXTERNAL_DIR = BASELINE_ROOT / "_data_external_test"
INTERNAL_NPZ = INTERNAL_DIR / "processed_gear.npz"
EXTERNAL_NPZ = EXTERNAL_DIR / "processed_gear.npz"
SEARCH_DIR = OUTPUT_ROOT / "stage1_search"
CONFIRM_DIR = OUTPUT_ROOT / "stage2_multiseed"
FINAL_DIR = OUTPUT_ROOT / "stage3_external_final"
SEARCH_SUMMARY = OUTPUT_ROOT / "stage1_search_summary.json"
CANDIDATES_PATH = OUTPUT_ROOT / "stage1_top3_candidates.json"
CONFIRM_SUMMARY = OUTPUT_ROOT / "stage2_multiseed_summary.json"
WINNER_PATH = OUTPUT_ROOT / "stage2_winner.json"
LOCK_PATH = OUTPUT_ROOT / ".gear_hparam_tuning.lock"

SEEDS = (42, 43, 44, 45, 46)
EPOCHS = 50
TOP_K = 3


@dataclass(frozen=True)
class Config:
    name: str
    lr: float = 2.5e-4
    hidden_size: int = 384
    dropout: float = 0.30
    weight_decay: float = 1.3e-3
    class_weight_power: float = 1.0
    focal_gamma: float = 1.2
    geo_aux_weight: float = 0.0


# Small, targeted search. Both geo0 and the original geo auxiliary setting are
# compared on identical current preprocessing before tuning regularization and
# minority-class loss.
CONFIGS = (
    Config("baseline_geo0"),
    Config("baseline_geo003", geo_aux_weight=0.03),
    Config("lr_1e4", lr=1.0e-4),
    Config("lr_5e4", lr=5.0e-4),
    Config("hidden_256", hidden_size=256),
    Config("dropout_020", dropout=0.20),
    Config("dropout_040", dropout=0.40),
    Config("classweight_050", class_weight_power=0.50),
    Config("classweight_075", class_weight_power=0.75),
    Config("focal_000", focal_gamma=0.0),
    Config("focal_200", focal_gamma=2.0),
    Config("cw075_focal200", class_weight_power=0.75, focal_gamma=2.0),
    Config(
        "lr1e4_drop020_cw075",
        lr=1.0e-4,
        dropout=0.20,
        class_weight_power=0.75,
    ),
)
CONFIG_BY_NAME = {cfg.name: cfg for cfg in CONFIGS}


def run(command: list[str], label: str) -> int:
    print(f"\n[gear-tuning] {label}")
    print("[gear-tuning] " + subprocess.list2cmdline(command))
    started = time.perf_counter()
    status = subprocess.run(command, cwd=ROOT, check=False).returncode
    print(
        f"[gear-tuning] {label} exit={status} "
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
                f"Runner tuning lain masih aktif dengan PID {old_pid}."
            )
        LOCK_PATH.unlink(missing_ok=True)
    fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))


def preprocess_command(data_dir: str, out_dir: Path, external: bool) -> list[str]:
    command = [
        sys.executable,
        "main.py",
        "preprocess",
        "--data_dir",
        data_dir,
        "--out_dir",
        str(out_dir),
        "--task",
        "gear",
        "--exclude_labels",
        "unknown",
        "pole_and_line",
        "trollers",
        "--seq_len",
        "120",
        "--stride",
        "6",
        "--gap_seconds",
        "43200",
        "--min_points_per_vessel",
        "80",
        "--min_windows_per_vessel",
        "0",
        "--max_windows_per_vessel",
        "1200",
        "--max_windows_per_file",
        "20000",
        "--use_operational_filter",
        "--op_speed_min",
        "1",
        "--op_speed_max",
        "12",
    ]
    if external:
        command.append("--no_jump_filter")
    return command


def ensure_internal_data() -> None:
    if not INTERNAL_NPZ.is_file():
        status = run(
            preprocess_command("Dataset", INTERNAL_DIR, external=False),
            "preprocess internal train/validation",
        )
        if not INTERNAL_NPZ.is_file():
            raise RuntimeError(
                "Preprocess internal tidak menghasilkan processed_gear.npz "
                f"(exit={status})."
            )
        if status != 0:
            print(
                "[gear-tuning] WARNING: preprocess internal exit nonzero, "
                "tetapi processed_gear.npz lengkap; lanjut."
            )


def ensure_external_data() -> None:
    if not EXTERNAL_NPZ.is_file():
        status = run(
            preprocess_command(
                "Dataset_Test_Enriched",
                EXTERNAL_DIR,
                external=True,
            ),
            "preprocess external final",
        )
        if not EXTERNAL_NPZ.is_file():
            raise RuntimeError(
                "Preprocess external tidak menghasilkan processed_gear.npz "
                f"(exit={status})."
            )
        if status != 0:
            print(
                "[gear-tuning] WARNING: preprocess external exit nonzero, "
                "tetapi processed_gear.npz lengkap; lanjut."
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
        history = json.loads((model_dir / "history.json").read_text(encoding="utf-8"))
        config = json.loads(
            (model_dir / "train_config.json").read_text(encoding="utf-8")
        )
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
        )
    )


def train_command(cfg: Config, model_dir: Path, seed: int) -> list[str]:
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
        str(cfg.lr),
        "--hidden_size",
        str(cfg.hidden_size),
        "--num_layers",
        "2",
        "--input_proj_dim",
        "256",
        "--embed_dim",
        "512",
        "--dropout",
        str(cfg.dropout),
        "--optimizer",
        "adamw",
        "--weight_decay",
        str(cfg.weight_decay),
        "--early_stop_patience",
        "90",
        "--focal_gamma",
        str(cfg.focal_gamma),
        "--geo_aux_weight",
        str(cfg.geo_aux_weight),
        "--gear_minority_f1_weight",
        "0.03",
        "--gear_class_weight_power",
        str(cfg.class_weight_power),
        "--gear_class_weight_max",
        "10",
        "--gear_tau_max",
        "0.6",
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


def train_and_validate(cfg: Config, seed: int, run_dir: Path) -> dict[str, float]:
    if cfg.name == "baseline_geo0":
        run_dir = BASELINE_ROOT / f"seed_{seed}"
    model_dir = run_dir / "model_gear"
    validation_dir = run_dir / "validation_eval"
    if not model_complete(model_dir):
        status = run(
            train_command(cfg, model_dir, seed),
            f"{cfg.name} seed={seed} train",
        )
        if not model_complete(model_dir):
            raise RuntimeError(
                f"{cfg.name} seed={seed} training tidak lengkap "
                f"(exit={status})."
            )
        if status != 0:
            print(
                f"[gear-tuning] WARNING: {cfg.name} seed={seed} exit={status} "
                "setelah penyimpanan, tetapi seluruh artefak training lengkap; "
                "lanjut ke validation."
            )
    if not eval_complete(validation_dir):
        status = run(
            eval_command(
                INTERNAL_NPZ,
                model_dir / "model.pt",
                validation_dir,
                "val",
            ),
            f"{cfg.name} seed={seed} validation",
        )
        if not eval_complete(validation_dir):
            raise RuntimeError(
                f"{cfg.name} seed={seed} validation tidak lengkap "
                f"(exit={status})."
            )
        if status != 0:
            print(
                f"[gear-tuning] WARNING: validation exit={status}, tetapi "
                "summary dan confusion matrix lengkap; lanjut."
            )
    summary = json.loads(
        (validation_dir / "eval_summary.json").read_text(encoding="utf-8")
    )
    return summary["metrics_vessel"]


def ranking_key(row: dict[str, object]) -> tuple[float, float, float]:
    return (
        float(row["mean_macro_f1"]),
        float(row["mean_balanced_acc"]),
        float(row["mean_accuracy"]),
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stage_search() -> None:
    ensure_internal_data()
    rows: list[dict[str, object]] = []
    for cfg in CONFIGS:
        metrics = train_and_validate(
            cfg,
            seed=42,
            run_dir=SEARCH_DIR / cfg.name / "seed_42",
        )
        rows.append(
            {
                "config": cfg.name,
                **asdict(cfg),
                "mean_macro_f1": float(metrics["macro_f1"]),
                "mean_balanced_acc": float(metrics["balanced_acc"]),
                "mean_accuracy": float(metrics["accuracy"]),
            }
        )
    rows.sort(key=ranking_key, reverse=True)
    SEARCH_SUMMARY.write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )
    write_csv(OUTPUT_ROOT / "stage1_search_summary.csv", rows)
    top = rows[:TOP_K]
    CANDIDATES_PATH.write_text(json.dumps(top, indent=2), encoding="utf-8")
    print("\n[gear-tuning] TOP 3 INTERNAL SEARCH")
    for rank, row in enumerate(top, start=1):
        print(
            f"  {rank}. {row['config']} "
            f"macro_f1={float(row['mean_macro_f1']):.4f}"
        )
    print("[gear-tuning] External test BELUM digunakan.")


def load_candidates() -> list[Config]:
    if not CANDIDATES_PATH.is_file():
        raise RuntimeError("Jalankan tahap search terlebih dahulu.")
    rows = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))
    return [CONFIG_BY_NAME[str(row["config"])] for row in rows]


def stage_confirm() -> None:
    ensure_internal_data()
    candidates = load_candidates()
    rows: list[dict[str, object]] = []
    for cfg in candidates:
        per_seed: list[dict[str, float]] = []
        for seed in SEEDS:
            if seed == 42:
                run_dir = SEARCH_DIR / cfg.name / "seed_42"
            else:
                run_dir = CONFIRM_DIR / cfg.name / f"seed_{seed}"
            metrics = train_and_validate(cfg, seed=seed, run_dir=run_dir)
            per_seed.append(
                {
                    "seed": seed,
                    "macro_f1": float(metrics["macro_f1"]),
                    "balanced_acc": float(metrics["balanced_acc"]),
                    "accuracy": float(metrics["accuracy"]),
                }
            )
        n = float(len(per_seed))
        row: dict[str, object] = {
            "config": cfg.name,
            **asdict(cfg),
            "mean_macro_f1": sum(r["macro_f1"] for r in per_seed) / n,
            "std_macro_f1": statistics.stdev(
                r["macro_f1"] for r in per_seed
            ),
            "mean_balanced_acc": sum(r["balanced_acc"] for r in per_seed) / n,
            "std_balanced_acc": statistics.stdev(
                r["balanced_acc"] for r in per_seed
            ),
            "mean_accuracy": sum(r["accuracy"] for r in per_seed) / n,
            "per_seed": per_seed,
        }
        rows.append(row)
    rows.sort(key=ranking_key, reverse=True)
    CONFIRM_SUMMARY.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    flat_rows = [
        {k: v for k, v in row.items() if k != "per_seed"}
        for row in rows
    ]
    write_csv(OUTPUT_ROOT / "stage2_multiseed_summary.csv", flat_rows)
    WINNER_PATH.write_text(json.dumps(rows[0], indent=2), encoding="utf-8")
    print("\n[gear-tuning] PEMENANG VALIDATION INTERNAL")
    print(
        f"  {rows[0]['config']} "
        f"mean_macro_f1={float(rows[0]['mean_macro_f1']):.4f}"
    )
    print("[gear-tuning] External test BELUM digunakan.")


def winner_model_dir(config_name: str, seed: int) -> Path:
    if config_name == "baseline_geo0":
        return BASELINE_ROOT / f"seed_{seed}" / "model_gear"
    if seed == 42:
        return SEARCH_DIR / config_name / "seed_42" / "model_gear"
    return CONFIRM_DIR / config_name / f"seed_{seed}" / "model_gear"


def stage_external() -> None:
    if not WINNER_PATH.is_file():
        raise RuntimeError("Jalankan tahap confirm sampai selesai terlebih dahulu.")
    winner = json.loads(WINNER_PATH.read_text(encoding="utf-8"))
    config_name = str(winner["config"])
    ensure_external_data()
    metrics_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        model_dir = winner_model_dir(config_name, seed)
        if not model_complete(model_dir):
            raise RuntimeError(
                f"Model final {config_name} seed {seed} belum lengkap."
            )
        eval_dir = FINAL_DIR / config_name / f"seed_{seed}" / "external_test_eval"
        if not eval_complete(eval_dir):
            status = run(
                eval_command(
                    EXTERNAL_NPZ,
                    model_dir / "model.pt",
                    eval_dir,
                    "all",
                ),
                f"FINAL external {config_name} seed={seed}",
            )
            if not eval_complete(eval_dir):
                raise RuntimeError(
                    f"External eval {config_name} seed={seed} tidak lengkap "
                    f"(exit={status})."
                )
            if status != 0:
                print(
                    f"[gear-tuning] WARNING: external eval exit={status}, "
                    "tetapi summary dan confusion matrix lengkap; lanjut."
                )
        summary = json.loads(
            (eval_dir / "eval_summary.json").read_text(encoding="utf-8")
        )
        metrics_rows.append(
            {
                "seed": seed,
                **summary["metrics_vessel"],
                "test_vessels": int(summary["test_vessels"]),
                "test_sequences": int(summary["test_sequences"]),
            }
        )
    n = float(len(metrics_rows))
    final_summary = {
        "config": config_name,
        "selection_source": "internal_validation_only",
        "external_used_for_selection": False,
        "per_seed": metrics_rows,
        "mean_metrics_vessel": {
            key: sum(float(row[key]) for row in metrics_rows) / n
            for key in ("accuracy", "macro_f1", "balanced_acc", "weighted_f1")
        },
    }
    (FINAL_DIR / "final_external_summary.json").parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    (FINAL_DIR / "final_external_summary.json").write_text(
        json.dumps(final_summary, indent=2),
        encoding="utf-8",
    )
    write_csv(FINAL_DIR / "final_external_per_seed.csv", metrics_rows)
    print("\n[gear-tuning] FINAL EXTERNAL COMPLETE")
    print(json.dumps(final_summary["mean_metrics_vessel"], indent=2))


def print_status() -> None:
    print(f"output: {OUTPUT_ROOT}")
    print(f"internal_data: {INTERNAL_NPZ.is_file()}")
    print(f"stage1_search: {SEARCH_SUMMARY.is_file()}")
    print(f"stage1_top3: {CANDIDATES_PATH.is_file()}")
    print(f"stage2_confirm: {CONFIRM_SUMMARY.is_file()}")
    print(f"stage2_winner: {WINNER_PATH.is_file()}")
    print(
        "stage3_external: "
        f"{(FINAL_DIR / 'final_external_summary.json').is_file()}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Leakage-safe gear hyperparameter tuning pipeline."
    )
    parser.add_argument(
        "stage",
        choices=("search", "confirm", "external", "all", "status"),
    )
    args = parser.parse_args()
    if args.stage == "status":
        print_status()
        return 0
    locked = False
    try:
        acquire_lock()
        locked = True
        if args.stage == "search":
            stage_search()
        elif args.stage == "confirm":
            stage_confirm()
        elif args.stage == "external":
            stage_external()
        elif args.stage == "all":
            stage_search()
            stage_confirm()
            stage_external()
        return 0
    except KeyboardInterrupt:
        print("\n[gear-tuning] Dihentikan pengguna.")
        return 130
    except Exception as exc:
        print(f"\n[gear-tuning] FATAL: {type(exc).__name__}: {exc}")
        return 1
    finally:
        if locked:
            LOCK_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
