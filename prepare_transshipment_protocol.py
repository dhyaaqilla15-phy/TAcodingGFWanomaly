from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from data_preparation import (
    DEFAULT_TRANSSHIPMENT_SOURCE_INCLUDE_LABELS,
    TRANSSHIPMENT_GEAR_TO_ID,
)


WINDOW_KEYS = (
    "X",
    "y",
    "groups",
    "coords",
    "window_event_ids",
    "window_kinds",
    "window_source_labels",
    "window_is_synthetic",
    "window_mmsi_a",
    "window_mmsi_b",
    "rule_features",
)


def _read_unique_mmsi(path: Path) -> np.ndarray:
    values: set[int] = set()
    for chunk in pd.read_csv(path, usecols=["mmsi"], chunksize=500_000):
        numeric = (
            pd.to_numeric(chunk["mmsi"], errors="coerce")
            .dropna()
            .round()
            .astype("int64")
        )
        values.update(int(v) for v in numeric.tolist())
    return np.asarray(sorted(values), dtype=np.int64)


def _source_mmsi(data_dir: Path) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    owners: Dict[int, str] = {}
    for source in DEFAULT_TRANSSHIPMENT_SOURCE_INCLUDE_LABELS:
        path = data_dir / f"{source}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Required transshipment source is missing: {path}")
        ids = _read_unique_mmsi(path)
        if ids.size < 2:
            raise RuntimeError(f"Source {source} has fewer than two vessels: {ids.size}")
        for mmsi in ids.tolist():
            previous = owners.get(int(mmsi))
            if previous is not None and previous != source:
                raise RuntimeError(
                    f"MMSI {mmsi} occurs in both {previous} and {source}; source labels are contaminated."
                )
            owners[int(mmsi)] = source
        result[source] = ids
    return result


def split_mmsi(
    internal_dir: Path,
    external_dir: Path,
    out_dir: Path,
    val_fraction: float,
    seed: int,
) -> None:
    if not 0.05 <= float(val_fraction) <= 0.40:
        raise ValueError("--val_fraction must be between 0.05 and 0.40")
    internal = _source_mmsi(internal_dir)
    external = _source_mmsi(external_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(int(seed))
    rows = []
    for source in DEFAULT_TRANSSHIPMENT_SOURCE_INCLUDE_LABELS:
        ids = internal[source].copy()
        rng.shuffle(ids)
        n_val = max(1, int(round(len(ids) * float(val_fraction))))
        n_val = min(n_val, len(ids) - 1)
        val_ids = set(int(v) for v in ids[:n_val].tolist())
        for mmsi in sorted(int(v) for v in ids.tolist()):
            rows.append(
                {
                    "mmsi": mmsi,
                    "source": source,
                    "gear_id": int(TRANSSHIPMENT_GEAR_TO_ID[source]),
                    "split": "validation" if mmsi in val_ids else "train",
                    "seed": int(seed),
                }
            )

    manifest = pd.DataFrame(rows).sort_values(["split", "source", "mmsi"])
    train = manifest[manifest["split"] == "train"].copy()
    validation = manifest[manifest["split"] == "validation"].copy()
    train_set = set(train["mmsi"].astype(int))
    val_set = set(validation["mmsi"].astype(int))
    external_set = {
        int(v) for ids in external.values() for v in ids.tolist()
    }
    if train_set & val_set:
        raise RuntimeError("Internal train/validation MMSI overlap.")
    if (train_set | val_set) & external_set:
        raise RuntimeError("Internal/external MMSI overlap.")

    manifest.to_csv(out_dir / "internal_mmsi_split_manifest.csv", index=False)
    train.to_csv(out_dir / "train_mmsi.csv", index=False)
    validation.to_csv(out_dir / "validation_mmsi.csv", index=False)

    audit = (
        manifest.groupby(["split", "source"], sort=True)
        .agg(vessels=("mmsi", "nunique"))
        .reset_index()
    )
    external_rows = [
        {"split": "external_test", "source": source, "vessels": int(len(ids))}
        for source, ids in external.items()
    ]
    audit = pd.concat([audit, pd.DataFrame(external_rows)], ignore_index=True)
    audit.to_csv(out_dir / "mmsi_split_source_audit.csv", index=False)
    (out_dir / "mmsi_split_audit.json").write_text(
        json.dumps(
            {
                "protocol": "source_stratified_vessel_disjoint_train_validation_external_v1",
                "seed": int(seed),
                "validation_fraction": float(val_fraction),
                "allowed_sources": list(DEFAULT_TRANSSHIPMENT_SOURCE_INCLUDE_LABELS),
                "train_vessels": int(len(train_set)),
                "validation_vessels": int(len(val_set)),
                "external_vessels": int(len(external_set)),
                "train_validation_overlap": 0,
                "internal_external_overlap": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(audit.to_string(index=False))
    print(f"[transshipment-protocol] MMSI manifests -> {out_dir}")


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _require_window_metadata(data: Dict[str, np.ndarray], name: str) -> None:
    required = set(WINDOW_KEYS)
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"{name} NPZ is missing protocol metadata: {missing}")
    n = int(len(data["y"]))
    bad = [key for key in WINDOW_KEYS if int(data[key].shape[0]) != n]
    if bad:
        raise ValueError(f"{name} NPZ has misaligned window arrays: {bad}")


def _mmsi_set(data: Dict[str, np.ndarray]) -> set[str]:
    values: set[str] = set()
    for key in ("window_mmsi_a", "window_mmsi_b"):
        for raw in data[key].astype(str).tolist():
            value = str(raw).strip()
            if value and value.lower() not in {"none", "nan", "-1"}:
                values.add(value)
    return values


def _class_counts(y: np.ndarray) -> Dict[str, int]:
    values, counts = np.unique(np.asarray(y, dtype=np.int64), return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts)}


def _summary(data: Dict[str, np.ndarray]) -> dict:
    return {
        "windows": int(len(data["y"])),
        "events": int(np.unique(data["groups"].astype(str)).size),
        "vessels": int(len(_mmsi_set(data))),
        "synthetic_windows": int(np.asarray(data["window_is_synthetic"]).astype(int).sum()),
        "classes": _class_counts(data["y"]),
        "sources": sorted(set(data["window_source_labels"].astype(str).tolist())),
    }


def _same_array(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(a, b)


def combine_audit(
    train_npz: Path,
    validation_npz: Path,
    external_npz: Path,
    out_dir: Path,
) -> None:
    train = _load_npz(train_npz)
    validation = _load_npz(validation_npz)
    external = _load_npz(external_npz)
    for name, data in (("train", train), ("validation", validation), ("external", external)):
        _require_window_metadata(data, name)

    for key in ("feature_cols", "rule_cols", "label_map"):
        if not _same_array(train[key], validation[key]) or not _same_array(train[key], external[key]):
            raise ValueError(f"Train/validation/external schema mismatch for {key}.")

    train_synthetic = np.asarray(train["window_is_synthetic"]).astype(int)
    val_synthetic = np.asarray(validation["window_is_synthetic"]).astype(int)
    external_synthetic = np.asarray(external["window_is_synthetic"]).astype(int)
    if int(train_synthetic.sum()) == 0:
        raise RuntimeError("Training NPZ contains no synthetic encounter augmentation.")
    if bool(np.any(val_synthetic)):
        raise RuntimeError("Synthetic encounter leaked into internal validation.")
    if bool(np.any(external_synthetic)):
        raise RuntimeError("Synthetic encounter leaked into pure external test.")

    train_mmsi = _mmsi_set(train)
    val_mmsi = _mmsi_set(validation)
    external_mmsi = _mmsi_set(external)
    # Artificial partner IDs are created only in training and cannot create
    # validation leakage. All observed MMSI must nevertheless stay disjoint.
    if train_mmsi & val_mmsi:
        raise RuntimeError(f"Train/validation MMSI overlap: {sorted(train_mmsi & val_mmsi)[:10]}")
    if (train_mmsi | val_mmsi) & external_mmsi:
        raise RuntimeError(
            "Internal/external MMSI overlap: "
            f"{sorted((train_mmsi | val_mmsi) & external_mmsi)[:10]}"
        )

    present_val = set(np.unique(validation["y"]).astype(int).tolist())
    present_external = set(np.unique(external["y"]).astype(int).tolist())
    expected = set(np.unique(train["y"]).astype(int).tolist())
    if not expected.issubset(present_val):
        raise RuntimeError(
            "Real internal validation is missing classes required by training: "
            f"train={sorted(expected)}, validation={sorted(present_val)}. "
            "Do not replace this with synthetic validation; adjust the MMSI split seed."
        )
    if not expected.issubset(present_external):
        raise RuntimeError(
            "Pure external test is missing classes required by training: "
            f"train={sorted(expected)}, external={sorted(present_external)}. "
            "Do not add synthetic events to external; obtain more real external candidates."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    n_train = int(len(train["y"]))
    n_val = int(len(validation["y"]))
    payload: Dict[str, np.ndarray] = {}
    for key, value in train.items():
        if key in WINDOW_KEYS:
            left = value
            right = validation[key]
            if key in {"groups", "window_event_ids"}:
                left = np.asarray([f"train::{x}" for x in left.astype(str)], dtype=object)
                right = np.asarray([f"validation::{x}" for x in right.astype(str)], dtype=object)
            payload[key] = np.concatenate([left, right], axis=0)
        else:
            payload[key] = value
    payload["transshipment_data_protocol"] = np.array(
        "synthetic_train_only_vessel_disjoint_external_real_v1", dtype=object
    )
    combined_path = out_dir / "processed_transshipment_trainval.npz"
    np.savez_compressed(combined_path, **payload)
    split_path = out_dir / "split_indices.npz"
    np.savez_compressed(
        split_path,
        train_idx=np.arange(n_train, dtype=np.int64),
        val_idx=np.arange(n_train, n_train + n_val, dtype=np.int64),
        test_idx=np.zeros((0,), dtype=np.int64),
    )

    audit = {
        "protocol": "synthetic_train_only_vessel_disjoint_external_real_v1",
        "train": _summary(train),
        "validation_real_internal": _summary(validation),
        "test_real_external": _summary(external),
        "train_validation_mmsi_overlap": 0,
        "internal_external_mmsi_overlap": 0,
        "validation_synthetic_windows": 0,
        "external_synthetic_windows": 0,
        "combined_trainval_npz": str(combined_path.resolve()),
        "split_indices": str(split_path.resolve()),
        "external_npz": str(external_npz.resolve()),
    }
    (out_dir / "transshipment_protocol_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))
    print(f"[transshipment-protocol] combined train/validation -> {combined_path}")


def main() -> None:
    parser = argparse.ArgumentParser("Prepare leakage-safe transshipment protocol")
    sub = parser.add_subparsers(dest="command", required=True)

    split = sub.add_parser("split-mmsi")
    split.add_argument("--internal_dir", required=True)
    split.add_argument("--external_dir", required=True)
    split.add_argument("--out_dir", required=True)
    split.add_argument("--val_fraction", type=float, default=0.20)
    split.add_argument("--seed", type=int, default=42)

    combine = sub.add_parser("combine-audit")
    combine.add_argument("--train_npz", required=True)
    combine.add_argument("--validation_npz", required=True)
    combine.add_argument("--external_npz", required=True)
    combine.add_argument("--out_dir", required=True)

    args = parser.parse_args()
    if args.command == "split-mmsi":
        split_mmsi(
            Path(args.internal_dir),
            Path(args.external_dir),
            Path(args.out_dir),
            float(args.val_fraction),
            int(args.seed),
        )
    else:
        combine_audit(
            Path(args.train_npz),
            Path(args.validation_npz),
            Path(args.external_npz),
            Path(args.out_dir),
        )


if __name__ == "__main__":
    main()
