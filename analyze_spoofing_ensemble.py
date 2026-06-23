from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from agg_utils import confusion_matrix_np, metrics_from_cm
from eval import save_spoofing_detection_png


DEFAULT_ROOT = (
    Path(__file__).resolve().parent
    / "Outputs"
    / "spoofing_baseline04_paired_position_only_multiseed"
)
SEEDS = (42, 43, 44, 45, 46)


def _binary_positive_metrics(cm: np.ndarray) -> dict[str, float | int]:
    cm = np.asarray(cm, dtype=np.int64)
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def select_sequence_threshold_from_validation(run_root: Path) -> tuple[float, dict[int, float]]:
    grid = np.linspace(0.05, 0.95, 181)
    selected: dict[int, float] = {}
    for seed in SEEDS:
        path = (
            run_root
            / f"seed_{seed}"
            / "validation_eval"
            / "spoofing_sequence_predictions.csv"
        )
        frame = pd.read_csv(path)
        y_true = frame["true_id"].to_numpy(dtype=np.int64)
        scores = frame["spoofing_probability"].to_numpy(dtype=float)
        candidates = []
        for threshold in grid:
            y_pred = (scores >= threshold).astype(np.int64)
            candidates.append(
                (
                    float(f1_score(y_true, y_pred, average="macro")),
                    -abs(float(threshold) - 0.50),
                    float(threshold),
                )
            )
        selected[int(seed)] = max(candidates)[-1]
    # A median is robust to one unstable split selecting an extreme threshold.
    return float(np.median(list(selected.values()))), selected


def load_aligned_predictions(
    run_root: Path,
    eval_dir_name: str,
    threshold: float,
) -> pd.DataFrame:
    frames = []
    key_cols = ["source_group", "scenario_id", "attack_type", "true_id"]
    for seed in SEEDS:
        path = (
            run_root
            / f"seed_{seed}"
            / eval_dir_name
            / "spoofing_sequence_predictions.csv"
        )
        if not path.is_file():
            raise FileNotFoundError(f"Missing seed prediction table: {path}")
        frame = pd.read_csv(path)
        frames.append(frame)

    reference_keys = frames[0][key_cols].astype(str)
    for seed, frame in zip(SEEDS[1:], frames[1:]):
        if not reference_keys.equals(frame[key_cols].astype(str)):
            raise RuntimeError(
                f"Seed {seed} predictions are not row-aligned; refusing ensemble."
            )

    output = frames[0][key_cols].copy()
    probability_cols = []
    for seed, frame in zip(SEEDS, frames):
        column = f"spoofing_probability_seed{seed}"
        output[column] = frame["spoofing_probability"].astype(float)
        probability_cols.append(column)
    output["spoofing_probability_ensemble"] = output[probability_cols].mean(axis=1)
    output["pred_id_ensemble"] = (
        output["spoofing_probability_ensemble"] >= float(threshold)
    ).astype(int)
    output["correct_ensemble"] = (
        output["true_id"].astype(int) == output["pred_id_ensemble"]
    )
    return output


def analyze(run_root: Path, eval_dir_name: str, out_dir: Path) -> None:
    threshold, per_seed_thresholds = select_sequence_threshold_from_validation(run_root)
    predictions = load_aligned_predictions(run_root, eval_dir_name, threshold)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = predictions["true_id"].to_numpy(dtype=np.int64)
    y_pred = predictions["pred_id_ensemble"].to_numpy(dtype=np.int64)
    scores = predictions["spoofing_probability_ensemble"].to_numpy(dtype=float)
    cm = confusion_matrix_np(y_true, y_pred, 2)
    sequence_metrics = {
        **metrics_from_cm(cm),
        **_binary_positive_metrics(cm),
        "average_precision": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "threshold": float(threshold),
        "num_windows": int(len(predictions)),
    }

    normal_mask = predictions["attack_type"].astype(str).str.lower().isin(
        ["normal", "normal_random"]
    )
    per_attack = []
    for attack in ("gradual_drift", "location_jump"):
        subset = normal_mask | predictions["attack_type"].astype(str).str.lower().eq(attack)
        attack_true = predictions.loc[subset, "true_id"].to_numpy(dtype=np.int64)
        attack_pred = predictions.loc[subset, "pred_id_ensemble"].to_numpy(dtype=np.int64)
        attack_cm = confusion_matrix_np(attack_true, attack_pred, 2)
        per_attack.append(
            {
                "attack_type": attack,
                **metrics_from_cm(attack_cm),
                **_binary_positive_metrics(attack_cm),
            }
        )

    scenario_rows = []
    for scenario_id, group in predictions.groupby("scenario_id", sort=True):
        values = group["spoofing_probability_ensemble"].to_numpy(dtype=float)
        top_count = max(1, int(np.ceil(len(values) * 0.10)))
        score = float(np.mean(np.sort(values)[-top_count:]))
        scenario_rows.append(
            {
                "scenario_id": str(scenario_id),
                "source_group": str(group["source_group"].iloc[0]),
                "attack_type": str(group["attack_type"].mode().iloc[0]),
                "true_id": int(group["true_id"].max()),
                "spoofing_probability_ensemble": score,
                "pred_id_ensemble": int(score >= 0.50),
                "n_windows": int(len(group)),
            }
        )
    scenarios = pd.DataFrame(scenario_rows)
    scenario_true = scenarios["true_id"].to_numpy(dtype=np.int64)
    scenario_pred = scenarios["pred_id_ensemble"].to_numpy(dtype=np.int64)
    scenario_scores = scenarios["spoofing_probability_ensemble"].to_numpy(dtype=float)
    scenario_cm = confusion_matrix_np(scenario_true, scenario_pred, 2)
    scenario_metrics = {
        **metrics_from_cm(scenario_cm),
        **_binary_positive_metrics(scenario_cm),
        "average_precision": float(
            average_precision_score(scenario_true, scenario_scores)
        ),
        "roc_auc": float(roc_auc_score(scenario_true, scenario_scores)),
        "threshold": 0.50,
        "aggregation": "mean_top_10_percent_ensemble_probability",
        "num_scenarios": int(len(scenarios)),
    }

    predictions.to_csv(out_dir / "ensemble_sequence_predictions.csv", index=False)
    scenarios.to_csv(out_dir / "ensemble_scenario_predictions.csv", index=False)
    pd.DataFrame(per_attack).to_csv(out_dir / "ensemble_per_attack_metrics.csv", index=False)
    save_spoofing_detection_png(
        cm,
        out_dir / "confusion_matrix.png",
        normalize=True,
        attack_name="5-seed probability ensemble",
    )

    evaluation_role = (
        "diagnostic_external_not_pristine_holdout"
        if eval_dir_name == "external_test_eval"
        else "aligned_evaluation"
    )
    summary = {
        "run_root": str(run_root),
        "evaluation_directory": eval_dir_name,
        "evaluation_role": evaluation_role,
        "seeds": list(SEEDS),
        "fusion": "arithmetic_mean_spoofing_probability",
        "threshold_selection": "median_of_per_seed_validation_macro_f1_optima",
        "validation_thresholds_per_seed": per_seed_thresholds,
        "sequence_metrics": sequence_metrics,
        "per_attack_metrics": per_attack,
        "scenario_metrics": scenario_metrics,
        "reporting_warning": (
            "This external set informed baseline02 design and is diagnostic, "
            "not an untouched final holdout."
            if evaluation_role == "diagnostic_external_not_pristine_holdout"
            else None
        ),
    }
    (out_dir / "ensemble_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[spoofing-ensemble] summary -> {out_dir / 'ensemble_summary.json'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fuse aligned spoofing predictions across seeds.")
    parser.add_argument("--run_root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--eval_dir_name", default="external_test_eval")
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()
    out_dir = args.out_dir or (args.run_root / "ensemble_external_diagnostic")
    analyze(args.run_root, str(args.eval_dir_name), out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
