from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "Outputs" / "spoofing_tuning01_jump_magnitude_seed42"
OUTPUT_ROOT = SOURCE_ROOT / "cross_severity_analysis"
DETAIL_ROOT = OUTPUT_ROOT / "_temporary_eval_details"

CONFIGS = {
    "jump_010deg": 0.10,
    "jump_030deg": 0.30,
    "jump_050deg": 0.50,
    "jump_080deg": 0.80,
}


def _paths(name: str) -> tuple[Path, Path, Path]:
    run_dir = SOURCE_ROOT / name
    return (
        run_dir / "data_internal_trainval" / "processed_spoofing.npz",
        run_dir / "model_spoofing" / "model.pt",
        run_dir / "model_spoofing" / "split_indices.npz",
    )


def validate_alignment() -> None:
    reference_arrays: dict[str, np.ndarray] | None = None
    reference_splits: dict[str, np.ndarray] | None = None
    array_keys = ("y", "groups", "window_kinds", "window_event_ids")
    split_keys = ("train_idx", "val_idx", "test_idx")

    for name in CONFIGS:
        npz_path, model_path, split_path = _paths(name)
        for required in (npz_path, model_path, split_path):
            if not required.is_file():
                raise FileNotFoundError(f"Required completed artifact missing: {required}")

        with np.load(npz_path, allow_pickle=True) as data:
            arrays = {key: data[key] for key in array_keys}
        with np.load(split_path, allow_pickle=True) as split:
            splits = {key: split[key] for key in split_keys}

        if reference_arrays is None:
            reference_arrays = arrays
            reference_splits = splits
            continue

        assert reference_splits is not None
        mismatched_arrays = [
            key for key in array_keys
            if not np.array_equal(reference_arrays[key], arrays[key])
        ]
        mismatched_splits = [
            key for key in split_keys
            if not np.array_equal(reference_splits[key], splits[key])
        ]
        if mismatched_arrays or mismatched_splits:
            raise RuntimeError(
                f"Cross-severity inputs are not aligned for {name}: "
                f"arrays={mismatched_arrays}, splits={mismatched_splits}"
            )


def _attack_rows(path: Path) -> dict[str, dict[str, object]]:
    frame = pd.read_csv(path)
    return {
        str(row["attack_type"]): row.to_dict()
        for _, row in frame.iterrows()
    }


def evaluate_pair(train_name: str, eval_name: str) -> tuple[dict, dict]:
    eval_npz, _, _ = _paths(eval_name)
    _, model_path, _ = _paths(train_name)
    out_dir = DETAIL_ROOT / f"train_{train_name}__eval_{eval_name}"
    command = [
        sys.executable,
        "main.py",
        "eval",
        "--data_npz",
        str(eval_npz),
        "--model_path",
        str(model_path),
        "--out_dir",
        str(out_dir),
        "--device",
        "cuda",
        "--batch_size",
        "256",
        "--eval_split",
        "val",
    ]
    print(f"[cross-severity] train={train_name} eval={eval_name}")
    subprocess.run(command, cwd=ROOT, check=True)

    summary_path = out_dir / "eval_summary.json"
    attack_path = out_dir / "spoofing_attack_metrics.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    attacks = _attack_rows(attack_path)
    jump = attacks["location_jump"]
    drift = attacks["gradual_drift"]
    ranking = summary.get("binary_ranking_metrics") or {}

    row = {
        "train_config": train_name,
        "eval_config": eval_name,
        "train_jump_nominal_deg": CONFIGS[train_name],
        "eval_jump_nominal_deg": CONFIGS[eval_name],
        "validation_windows": int(summary["test_sequences"]),
        "seq_accuracy": float(summary["metrics_seq"]["accuracy"]),
        "seq_macro_f1": float(summary["metrics_seq"]["macro_f1"]),
        "seq_balanced_accuracy": float(summary["metrics_seq"]["balanced_acc"]),
        "average_precision": float(ranking["average_precision"]),
        "roc_auc": float(ranking["roc_auc"]),
        "gradual_drift_precision": float(drift["precision"]),
        "gradual_drift_recall": float(drift["recall"]),
        "gradual_drift_f1": float(drift["f1"]),
        "location_jump_precision": float(jump["precision"]),
        "location_jump_recall": float(jump["recall"]),
        "location_jump_f1": float(jump["f1"]),
        "location_jump_positive_windows": int(jump["positive_windows"]),
        "normal_windows": int(jump["normal_windows"]),
        "external_test_used": False,
    }
    detail = {
        "train_config": train_name,
        "eval_config": eval_name,
        "metrics_seq": summary["metrics_seq"],
        "binary_ranking_metrics": ranking,
        "spoofing_scenario_metrics": summary.get("spoofing_scenario_metrics"),
        "per_attack": attacks,
    }
    return row, detail


def write_heatmap(results: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    ordered = list(CONFIGS)
    matrix = (
        results.pivot(
            index="train_config",
            columns="eval_config",
            values="location_jump_f1",
        )
        .reindex(index=ordered, columns=ordered)
        .to_numpy(dtype=float)
    )
    labels = [f"{CONFIGS[name]:.2f}°" for name in ordered]
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=ax, label="Location-jump F1")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Evaluation jump severity")
    ax.set_ylabel("Training jump severity")
    ax.set_title("Cross-severity validation: location-jump F1")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.3f}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] >= 0.65 else "black",
            )
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "cross_severity_location_jump_f1.png", dpi=220)
    plt.close(fig)


def write_outputs(rows: list[dict], details: list[dict]) -> None:
    results = pd.DataFrame(rows).sort_values(
        ["train_jump_nominal_deg", "eval_jump_nominal_deg"]
    )
    results.to_csv(OUTPUT_ROOT / "cross_severity_results.csv", index=False)

    robustness = (
        results.groupby(
            ["train_config", "train_jump_nominal_deg"], as_index=False
        )
        .agg(
            mean_location_jump_f1=("location_jump_f1", "mean"),
            worst_location_jump_f1=("location_jump_f1", "min"),
            mean_location_jump_recall=("location_jump_recall", "mean"),
            worst_location_jump_recall=("location_jump_recall", "min"),
            mean_seq_macro_f1=("seq_macro_f1", "mean"),
            worst_seq_macro_f1=("seq_macro_f1", "min"),
            mean_average_precision=("average_precision", "mean"),
            worst_average_precision=("average_precision", "min"),
        )
        .sort_values(
            ["worst_location_jump_f1", "mean_location_jump_f1"],
            ascending=False,
        )
    )
    robustness.insert(0, "robustness_rank", range(1, len(robustness) + 1))
    robustness.to_csv(OUTPUT_ROOT / "cross_severity_robustness_summary.csv", index=False)
    (OUTPUT_ROOT / "cross_severity_details.json").write_text(
        json.dumps(details, indent=2),
        encoding="utf-8",
    )
    write_heatmap(results)


def run() -> None:
    validate_alignment()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if DETAIL_ROOT.exists():
        shutil.rmtree(DETAIL_ROOT)
    DETAIL_ROOT.mkdir(parents=True)

    rows: list[dict] = []
    details: list[dict] = []
    completed = False
    try:
        for train_name in CONFIGS:
            for eval_name in CONFIGS:
                row, detail = evaluate_pair(train_name, eval_name)
                rows.append(row)
                details.append(detail)
        write_outputs(rows, details)
        completed = True
    finally:
        if completed and DETAIL_ROOT.exists():
            shutil.rmtree(DETAIL_ROOT)

    print(f"[cross-severity] results -> {OUTPUT_ROOT}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate each spoofing jump checkpoint on every aligned severity."
    )
    parser.add_argument("stage", nargs="?", default="run", choices=("run", "status"))
    args = parser.parse_args()
    if args.stage == "run":
        run()
    else:
        for name in (
            "cross_severity_results.csv",
            "cross_severity_robustness_summary.csv",
            "cross_severity_details.json",
            "cross_severity_location_jump_f1.png",
        ):
            path = OUTPUT_ROOT / name
            print(f"{name}: {path.is_file()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
