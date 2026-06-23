from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
MASTER_NPZ = (
    ROOT
    / "Outputs"
    / "godark_external01"
    / "data_internal_trainval"
    / "processed_godark.npz"
)
OUT_ROOT = ROOT / "Outputs" / "godark_manipulation_tuning01"
DATA_ROOT = OUT_ROOT / "variant_data"
EXPECTED_PROTOCOL = "source_label_duration_cadence_distance_position_v2"
EXPECTED_SOURCES = {
    "drifting_longlines",
    "fixed_gear",
    "purse_seines",
    "trawlers",
}


VARIANTS = (
    {"name": "count_1", "factor": "event_count", "level": "1", "max_event_order": 1},
    {"name": "count_2", "factor": "event_count", "level": "2", "max_event_order": 2},
    {"name": "count_3", "factor": "event_count", "level": "3", "max_event_order": 3},
    {"name": "position_early", "factor": "trajectory_position", "level": "early", "position": 0},
    {"name": "position_middle", "factor": "trajectory_position", "level": "middle", "position": 1},
    {"name": "position_late", "factor": "trajectory_position", "level": "late", "position": 2},
)


def variant_npz(name: str) -> Path:
    return DATA_ROOT / name / "processed_godark.npz"


def tuning_root(name: str) -> Path:
    return OUT_ROOT / "runs" / name


def validate_master() -> None:
    if not MASTER_NPZ.is_file():
        raise FileNotFoundError(
            f"Missing {MASTER_NPZ}. Run: bash run_godark_external_test_pipeline.sh prepare"
        )
    with np.load(MASTER_NPZ, allow_pickle=True) as data:
        required = {
            "X", "y", "groups", "window_source_labels",
            "window_event_orders", "window_position_strata",
            "godark_diversity_protocol",
        }
        missing = required - set(data.files)
        if missing:
            raise RuntimeError(
                f"Master NPZ predates manipulation metadata {sorted(missing)}. Rerun prepare."
            )
        protocol = str(np.asarray(data["godark_diversity_protocol"]).item())
        if protocol != EXPECTED_PROTOCOL:
            raise RuntimeError(f"Unexpected master protocol: {protocol}. Rerun prepare.")
        if set(data["window_source_labels"].astype(str).tolist()) != EXPECTED_SOURCES:
            raise RuntimeError("Master NPZ does not contain exactly the four EDA sources.")


def selection_mask(data: np.lib.npyio.NpzFile, variant: dict) -> np.ndarray:
    y = data["y"].astype(np.int64)
    positive = y == 1
    keep_positive = positive.copy()
    if "max_event_order" in variant:
        keep_positive &= data["window_event_orders"].astype(np.int64) <= int(
            variant["max_event_order"]
        )
    if "position" in variant:
        keep_positive &= data["window_position_strata"].astype(np.int64) == int(
            variant["position"]
        )
    # Use the same complete hard-negative pool in every variant. Source-aware
    # sampling handles class balance, while the positive manipulation factor is
    # the only intended experimental difference.
    return (~positive) | keep_positive


def audit_variant(path: Path, variant: dict) -> dict:
    with np.load(path, allow_pickle=True) as data:
        y = data["y"].astype(np.int64)
        source = data["window_source_labels"].astype(str)
        groups = data["groups"].astype(str)
        rows = []
        for source_class in sorted(EXPECTED_SOURCES):
            mask = source == source_class
            positive = mask & (y == 1)
            rows.append(
                {
                    "variant": variant["name"],
                    "factor": variant["factor"],
                    "level": variant["level"],
                    "source_class": source_class,
                    "normal_events": int(np.sum(mask & (y == 0))),
                    "go_dark_events": int(np.sum(positive)),
                    "vessels": int(len(set(groups[mask].tolist()))),
                    "positive_vessels": int(len(set(groups[positive].tolist()))),
                }
            )
        incomplete = [
            row for row in rows
            if row["normal_events"] == 0
            or row["go_dark_events"] == 0
            or row["positive_vessels"] < 3
        ]
        if incomplete:
            raise RuntimeError(
                f"Variant {variant['name']} cannot support three source-stratified folds: {incomplete}"
            )
        return {"sequences": int(len(y)), "rows": rows}


def prepare() -> None:
    validate_master()
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    all_audit_rows = []
    with np.load(MASTER_NPZ, allow_pickle=True) as master:
        n = int(len(master["y"]))
        for variant in VARIANTS:
            keep = selection_mask(master, variant)
            payload = {}
            for key in master.files:
                value = master[key]
                payload[key] = value[keep] if value.ndim >= 1 and value.shape[0] == n else value
            payload["manipulation_variant"] = np.array(
                variant["name"], dtype=object
            )
            path = variant_npz(variant["name"])
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **payload)
            audit = audit_variant(path, variant)
            all_audit_rows.extend(audit["rows"])
            print(
                f"[manipulation] prepared {variant['name']} "
                f"sequences={audit['sequences']} -> {path}"
            )
    pd.DataFrame(all_audit_rows).to_csv(
        OUT_ROOT / "variant_data_audit.csv", index=False
    )


def run_internal_searches() -> None:
    validate_master()
    for variant in VARIANTS:
        data_path = variant_npz(variant["name"])
        if not data_path.is_file():
            raise FileNotFoundError(f"Missing {data_path}. Run prepare first.")
        audit_variant(data_path, variant)
        env = os.environ.copy()
        env.update(
            {
                "GODARK_INTERNAL_NPZ": str(data_path),
                "GODARK_TUNING_ROOT": str(tuning_root(variant["name"])),
                "GODARK_CONFIG_NAMES": "compact_h128",
                "GODARK_TUNING_SEEDS": "42,43,44",
            }
        )
        command = [sys.executable, "run_godark_hparam_tuning.py", "search"]
        print(f"[manipulation] internal-only search -> {variant['name']}")
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    summarize()


def summarize() -> None:
    rows = []
    for variant in VARIANTS:
        winner_path = tuning_root(variant["name"]) / "winner_internal_only.json"
        if not winner_path.is_file():
            continue
        manifest = json.loads(winner_path.read_text(encoding="utf-8"))
        if manifest.get("external_used_for_selection") is not False:
            raise RuntimeError(f"Invalid winner scope for {variant['name']}")
        winner = manifest["winner"]
        rows.append(
            {
                "variant": variant["name"],
                "factor": variant["factor"],
                "level": variant["level"],
                "model": winner["name"],
                "threshold": winner["pooled_oof_threshold"],
                "macro_source_f1": winner["macro_source_f1"],
                "min_source_recall": winner["min_source_recall"],
                "mean_event_f1": winner["mean_event_f1"],
                "std_event_f1": winner["std_event_f1"],
                "mean_precision": winner["mean_precision"],
                "mean_recall": winner["mean_recall"],
                "external_used": False,
            }
        )
    if not rows:
        raise FileNotFoundError("No manipulation winner manifests found.")
    frame = pd.DataFrame(rows).sort_values(
        ["factor", "macro_source_f1", "min_source_recall"],
        ascending=[True, False, False],
    )
    frame["rank_within_factor"] = frame.groupby("factor").cumcount() + 1
    frame.to_csv(OUT_ROOT / "manipulation_sensitivity_summary.csv", index=False)
    best = (
        frame.sort_values(
            ["macro_source_f1", "min_source_recall", "mean_event_f1"],
            ascending=[False, False, False],
        )
        .iloc[0]
        .to_dict()
    )
    (OUT_ROOT / "manipulation_sensitivity_best_internal.json").write_text(
        json.dumps(
            {
                "selection_scope": "internal_source_stratified_oof_only",
                "external_used": False,
                "interpretation": (
                    "Sensitivity/ablation result; do not tune manipulation settings "
                    "against the external test."
                ),
                "best_internal_variant": best,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(frame.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Internal-only Go-Dark manipulation point sensitivity study."
    )
    parser.add_argument("mode", choices=["prepare", "search", "summarize", "all"])
    args = parser.parse_args()
    if args.mode in {"prepare", "all"}:
        prepare()
    if args.mode in {"search", "all"}:
        run_internal_searches()
    elif args.mode == "summarize":
        summarize()


if __name__ == "__main__":
    main()
