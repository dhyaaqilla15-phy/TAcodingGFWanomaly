from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "Outputs"
LOCK_PATH = OUTPUTS / ".gear_overnight_python.lock"
LOG_DIR = OUTPUTS / "logs"
SEEDS = (42, 43, 44, 45, 46)
EPOCHS = 50

VARIANTS = (
    {
        "name": "geo0",
        "root": OUTPUTS / "gear_tuning04_gap12h_opfilter_1to12_geo0_multiseed",
        "location_features": True,
    },
    {
        "name": "motion_only",
        "root": OUTPUTS / "gear_tuning05_gap12h_opfilter_1to12_geo0_motiononly_multiseed",
        "location_features": False,
    },
)


class Tee:
    def __init__(self, path: Path):
        self.file = path.open("a", encoding="utf-8", buffering=1)

    def write(self, message: str) -> None:
        print(message, flush=True)
        self.file.write(message + "\n")

    def close(self) -> None:
        self.file.close()


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
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        try:
            old_pid = int(LOCK_PATH.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            old_pid = -1
        if process_is_running(old_pid):
            raise RuntimeError(
                f"Runner lain masih aktif dengan PID {old_pid}. "
                "Jangan jalankan command dua kali."
            )
        LOCK_PATH.unlink(missing_ok=True)

    fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))


def training_complete(model_out: Path) -> bool:
    required = (
        "model.pt",
        "history.json",
        "best_epoch.json",
        "train_config.json",
        "training_curves.png",
        "split_indices.npz",
        "scaler.joblib",
    )
    if not all((model_out / name).is_file() for name in required):
        return False
    try:
        history = json.loads((model_out / "history.json").read_text(encoding="utf-8"))
        config = json.loads((model_out / "train_config.json").read_text(encoding="utf-8"))
        return (
            bool(history)
            and int(history[-1].get("epoch", 0)) >= EPOCHS
            and int(config.get("epochs", 0)) == EPOCHS
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def eval_complete(eval_out: Path) -> bool:
    return all(
        (eval_out / name).is_file()
        for name in (
            "eval_summary.json",
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
        )
    )


def print_status() -> None:
    for variant in VARIANTS:
        name = str(variant["name"])
        root_out = Path(variant["root"])
        print(f"{name}: {root_out}")
        for seed in SEEDS:
            run_out = root_out / f"seed_{seed}"
            train_ok = training_complete(run_out / "model_gear")
            val_ok = eval_complete(run_out / "validation_eval")
            ext_ok = eval_complete(run_out / "external_test_eval")
            if not train_ok:
                next_step = "train"
            elif not val_ok:
                next_step = "validation_eval"
            elif not ext_ok:
                next_step = "external_eval"
            else:
                next_step = "complete"
            print(
                f"  seed {seed}: train={train_ok} val={val_ok} "
                f"external={ext_ok} next={next_step}"
            )


def run_command(
    command: list[str],
    tee: Tee,
    *,
    label: str,
    retries: int = 0,
) -> int:
    for attempt in range(1, retries + 2):
        tee.write(f"[runner] {label} attempt={attempt}/{retries + 1}")
        tee.write("[runner] command: " + subprocess.list2cmdline(command))
        started = time.perf_counter()
        # Inherit the real console so tqdm can update one line in place.
        # Capturing stdout through a pipe makes carriage returns look like
        # hundreds of separate progress lines on Windows terminals.
        process = subprocess.Popen(
            command,
            cwd=ROOT,
        )
        status = process.wait()
        tee.write(
            f"[runner] {label} exit={status} "
            f"duration={time.perf_counter() - started:.1f}s"
        )
        if status == 0:
            return 0
        if attempt <= retries:
            tee.write(f"[runner] {label} retry in 10 seconds")
            time.sleep(10)
    return status


def preprocess_command(
    data_dir: str,
    out_dir: Path,
    *,
    location_features: bool,
    external: bool,
) -> list[str]:
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
    if not location_features:
        command.append("--exclude_location_features")
    return command


def train_command(data_npz: Path, model_out: Path, seed: int) -> list[str]:
    return [
        sys.executable,
        "main.py",
        "train",
        "--data_npz",
        str(data_npz),
        "--out_dir",
        str(model_out),
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
        "--hidden_size",
        "384",
        "--num_layers",
        "2",
        "--input_proj_dim",
        "256",
        "--embed_dim",
        "512",
        "--dropout",
        "0.30",
        "--optimizer",
        "adamw",
        "--weight_decay",
        "0.0013",
        "--early_stop_patience",
        "90",
        "--geo_aux_weight",
        "0",
        "--gear_minority_f1_weight",
        "0.03",
        "--gear_class_weight_power",
        "1.0",
        "--gear_class_weight_max",
        "10.0",
        "--gear_tau_max",
        "0.6",
    ]


def eval_command(
    data_npz: Path,
    model_path: Path,
    eval_out: Path,
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
        str(eval_out),
        "--device",
        "cuda",
        "--batch_size",
        "256",
        "--eval_split",
        split,
    ]


def run_variant(variant: dict[str, object], tee: Tee) -> None:
    name = str(variant["name"])
    root_out = Path(variant["root"])
    location_features = bool(variant["location_features"])
    internal_out = root_out / "_data_internal_trainval"
    external_out = root_out / "_data_external_test"
    internal_npz = internal_out / "processed_gear.npz"
    external_npz = external_out / "processed_gear.npz"

    tee.write(f"[runner] VARIANT START: {name}")
    root_out.mkdir(parents=True, exist_ok=True)

    if not internal_npz.is_file():
        status = run_command(
            preprocess_command(
                "Dataset",
                internal_out,
                location_features=location_features,
                external=False,
            ),
            tee,
            label=f"{name} preprocess internal",
        )
        if status != 0 or not internal_npz.is_file():
            raise RuntimeError(f"{name}: preprocess internal gagal")
    else:
        tee.write(f"[runner] {name}: reuse {internal_npz}")

    if not external_npz.is_file():
        status = run_command(
            preprocess_command(
                "Dataset_Test_Enriched",
                external_out,
                location_features=location_features,
                external=True,
            ),
            tee,
            label=f"{name} preprocess external",
        )
        if status != 0 or not external_npz.is_file():
            raise RuntimeError(f"{name}: preprocess external gagal")
    else:
        tee.write(f"[runner] {name}: reuse {external_npz}")

    for seed in SEEDS:
        run_out = root_out / f"seed_{seed}"
        model_out = run_out / "model_gear"
        model_path = model_out / "model.pt"
        validation_out = run_out / "validation_eval"
        external_eval_out = run_out / "external_test_eval"

        tee.write(f"[runner] {name} seed={seed} START")
        if training_complete(model_out):
            tee.write(f"[runner] {name} seed={seed}: training complete, skip")
        else:
            status = run_command(
                train_command(internal_npz, model_out, seed),
                tee,
                label=f"{name} seed={seed} train",
            )
            if not training_complete(model_out):
                raise RuntimeError(
                    f"{name} seed={seed}: training incomplete (exit={status})"
                )
            tee.write(f"[runner] {name} seed={seed}: training verified")

        if eval_complete(validation_out):
            tee.write(f"[runner] {name} seed={seed}: validation eval complete, skip")
        else:
            status = run_command(
                eval_command(internal_npz, model_path, validation_out, "val"),
                tee,
                label=f"{name} seed={seed} validation eval",
                retries=2,
            )
            if status != 0 or not eval_complete(validation_out):
                raise RuntimeError(f"{name} seed={seed}: validation eval gagal")

        if eval_complete(external_eval_out):
            tee.write(f"[runner] {name} seed={seed}: external eval complete, skip")
        else:
            status = run_command(
                eval_command(external_npz, model_path, external_eval_out, "all"),
                tee,
                label=f"{name} seed={seed} external eval",
                retries=2,
            )
            if status != 0 or not eval_complete(external_eval_out):
                raise RuntimeError(f"{name} seed={seed}: external eval gagal")

        tee.write(f"[runner] {name} seed={seed} COMPLETE")

    tee.write(f"[runner] VARIANT COMPLETE: {name}")


def main() -> int:
    if "--status" in sys.argv[1:]:
        print_status()
        return 0

    acquire_lock()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"gear_python_{datetime.now():%Y%m%d_%H%M%S}.log"
    tee = Tee(log_path)
    try:
        tee.write(f"[runner] PID={os.getpid()} log={log_path}")
        for variant in VARIANTS:
            run_variant(variant, tee)
        tee.write("[runner] ALL VARIANTS COMPLETE")
        return 0
    except KeyboardInterrupt:
        tee.write("[runner] interrupted by user")
        return 130
    except Exception as exc:
        tee.write(f"[runner] FATAL: {type(exc).__name__}: {exc}")
        return 1
    finally:
        tee.close()
        LOCK_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
