from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
LOCK_PATH = ROOT / "GODARK_MODEL_LOCK.json"
MANIPULATION_ROOT = ROOT / "Outputs" / "godark_manipulation_tuning01"
RUN_ROOT = MANIPULATION_ROOT / "runs" / "count_3"
WINNER_PATH = RUN_ROOT / "winner_internal_only.json"
VALIDATION_SUMMARY_PATH = (
    MANIPULATION_ROOT / "best_selected" / "validation_oof" / "validation_summary.json"
)
TEST_SUMMARY_PATH = (
    MANIPULATION_ROOT
    / "best_selected"
    / "test_external_ensemble"
    / "test_summary.json"
)
OUTPUT_DIR = ROOT / "Outputs" / "godark_final_locked"
OUTPUT_PATH = OUTPUT_DIR / "godark_final_compact_h128_ensemble.h5"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_bytes(group: h5py.Group, name: str, data: bytes) -> h5py.Dataset:
    return group.create_dataset(name, data=np.frombuffer(data, dtype=np.uint8))


def main() -> None:
    lock_manifest = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    winner_manifest = json.loads(WINNER_PATH.read_text(encoding="utf-8"))
    winner = winner_manifest["winner"]
    data_variant = str(lock_manifest["data_variant"])
    model_variant = str(lock_manifest["model_config"])
    seeds = [int(seed) for seed in winner_manifest["seeds"]]

    if lock_manifest.get("status") != "LOCKED":
        raise RuntimeError("Refusing export: GODARK_MODEL_LOCK.json is not LOCKED.")
    if winner_manifest.get("external_used_for_selection") is not False:
        raise RuntimeError("Refusing export: winner was not selected from internal validation only.")
    if data_variant != "count_3" or model_variant != "compact_h128" or seeds != [42, 43, 44]:
        raise RuntimeError(
            "Unexpected locked winner: "
            f"data={data_variant}, model={model_variant}, seeds={seeds}"
        )
    if not np.isclose(
        float(lock_manifest["decision_threshold"]),
        float(winner["pooled_oof_threshold"]),
    ):
        raise RuntimeError("Lock threshold and internal winner threshold disagree.")

    calibrator_path = RUN_ROOT / "oof_calibrators" / f"{model_variant}_platt.joblib"
    required = [
        LOCK_PATH,
        WINNER_PATH,
        VALIDATION_SUMMARY_PATH,
        TEST_SUMMARY_PATH,
        calibrator_path,
    ]
    for seed in seeds:
        model_dir = RUN_ROOT / "trials" / model_variant / f"seed_{seed}" / "model_godark"
        required.extend([model_dir / "model.pt", model_dir / "scaler.joblib"])
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing locked GoDark artifacts:\n" + "\n".join(missing))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = OUTPUT_PATH.with_suffix(".h5.tmp")
    if temporary_path.exists():
        temporary_path.unlink()

    with h5py.File(temporary_path, "w") as h5:
        h5.attrs["artifact_type"] = "pytorch_godark_ensemble_bundle"
        h5.attrs["framework"] = "PyTorch"
        h5.attrs["keras_load_model_compatible"] = False
        h5.attrs["task"] = "go_dark_detection"
        h5.attrs["data_variant"] = data_variant
        h5.attrs["model_variant"] = model_variant
        h5.attrs["ensemble_policy"] = winner_manifest["ensemble_policy"]
        h5.attrs["decision_threshold"] = float(lock_manifest["decision_threshold"])
        h5.attrs["seeds"] = np.asarray(seeds, dtype=np.int64)
        h5.attrs["source_selection"] = "internal_source_stratified_oof_only"

        metadata = h5.create_group("metadata")
        metadata.create_dataset(
            "godark_model_lock_json",
            data=LOCK_PATH.read_text(encoding="utf-8"),
            dtype=h5py.string_dtype("utf-8"),
        )
        metadata.create_dataset(
            "winner_internal_only_json",
            data=WINNER_PATH.read_text(encoding="utf-8"),
            dtype=h5py.string_dtype("utf-8"),
        )
        metadata.create_dataset(
            "official_validation_summary_json",
            data=VALIDATION_SUMMARY_PATH.read_text(encoding="utf-8"),
            dtype=h5py.string_dtype("utf-8"),
        )
        metadata.create_dataset(
            "official_test_summary_json",
            data=TEST_SUMMARY_PATH.read_text(encoding="utf-8"),
            dtype=h5py.string_dtype("utf-8"),
        )

        models = h5.create_group("models")
        for seed in seeds:
            model_dir = RUN_ROOT / "trials" / model_variant / f"seed_{seed}" / "model_godark"
            checkpoint_path = model_dir / "model.pt"
            scaler_path = model_dir / "scaler.joblib"
            checkpoint_bytes = checkpoint_path.read_bytes()
            scaler_bytes = scaler_path.read_bytes()
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

            seed_group = models.create_group(f"seed_{seed}")
            seed_group.attrs["seed"] = seed
            seed_group.attrs["checkpoint_sha256"] = sha256_bytes(checkpoint_bytes)
            seed_group.attrs["scaler_sha256"] = sha256_bytes(scaler_bytes)
            seed_group.attrs["checkpoint_source"] = checkpoint_path.relative_to(ROOT).as_posix()
            seed_group.attrs["scaler_source"] = scaler_path.relative_to(ROOT).as_posix()
            write_bytes(seed_group, "pytorch_checkpoint_pt", checkpoint_bytes)
            write_bytes(seed_group, "scaler_joblib", scaler_bytes)

            checkpoint_metadata = {
                key: value for key, value in checkpoint.items() if key != "model_state"
            }
            checkpoint_metadata["scaler_path"] = f"/models/seed_{seed}/scaler_joblib"
            seed_group.create_dataset(
                "checkpoint_metadata_json",
                data=json.dumps(checkpoint_metadata, default=json_default, sort_keys=True),
                dtype=h5py.string_dtype("utf-8"),
            )

            state_group = seed_group.create_group("state_dict")
            for parameter_name, tensor in checkpoint["model_state"].items():
                array = tensor.detach().cpu().numpy()
                dataset = state_group.create_dataset(parameter_name, data=array)
                dataset.attrs["torch_dtype"] = str(tensor.dtype)

        artifacts = h5.create_group("ensemble_artifacts")
        calibrator_bytes = calibrator_path.read_bytes()
        calibrator = write_bytes(artifacts, "platt_calibrator_joblib", calibrator_bytes)
        calibrator.attrs["sha256"] = sha256_bytes(calibrator_bytes)
        calibrator.attrs["source"] = calibrator_path.relative_to(ROOT).as_posix()

    os.replace(temporary_path, OUTPUT_PATH)
    output_hash = sha256_bytes(OUTPUT_PATH.read_bytes())
    hash_path = OUTPUT_PATH.with_suffix(".h5.sha256")
    hash_path.write_text(f"{output_hash}  {OUTPUT_PATH.name}\n", encoding="ascii")

    with h5py.File(OUTPUT_PATH, "r") as h5:
        packaged_seeds = sorted(int(name.removeprefix("seed_")) for name in h5["models"])
        if packaged_seeds != seeds:
            raise RuntimeError(f"HDF5 verification failed: {packaged_seeds} != {seeds}")
        for seed in seeds:
            if len(h5[f"models/seed_{seed}/state_dict"]) == 0:
                raise RuntimeError(f"HDF5 verification failed: empty state_dict seed {seed}")

    print(f"[godark-h5] exported: {OUTPUT_PATH}")
    print(f"[godark-h5] seeds: {seeds}")
    print(f"[godark-h5] sha256: {output_hash}")
    print(f"[godark-h5] size_bytes: {OUTPUT_PATH.stat().st_size}")


if __name__ == "__main__":
    main()
