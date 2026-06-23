from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from agg_utils import confusion_matrix_np, metrics_from_cm, per_class_metrics_from_cm


ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_ROOT = (
    ROOT / "Outputs" / "spoofing_baseline04_paired_position_only_multiseed"
)
SEEDS = (42, 43, 44, 45, 46)
ATTACKS = ("gradual_drift", "location_jump")


def positive_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores >= float(threshold)).astype(np.int64)
    cm = confusion_matrix_np(y_true, y_pred, 2)
    cls = per_class_metrics_from_cm(cm)
    tn, fp, fn, tp = [int(value) for value in cm.ravel()]
    output = {
        **metrics_from_cm(cm),
        "precision": float(cls["precision"][1]),
        "recall": float(cls["recall"][1]),
        "f1": float(cls["f1"][1]),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "positive_support": int((y_true == 1).sum()),
    }
    if np.unique(y_true).size == 2:
        output["average_precision"] = float(average_precision_score(y_true, scores))
        output["roc_auc"] = float(roc_auc_score(y_true, scores))
    else:
        output["average_precision"] = float("nan")
        output["roc_auc"] = float("nan")
    return output


def select_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, dict]:
    candidates = []
    for threshold in np.linspace(0.05, 0.95, 181):
        metrics = positive_metrics(y_true, scores, float(threshold))
        key = (
            metrics["f1"],
            metrics["balanced_acc"],
            -abs(float(threshold) - 0.50),
        )
        candidates.append((key, float(threshold), metrics))
    _, threshold, metrics = max(candidates, key=lambda item: item[0])
    return threshold, metrics


def prediction_path(run_root: Path, seed: int, split: str, level: str) -> Path:
    name = (
        "spoofing_sequence_predictions.csv"
        if level == "sequence"
        else "spoofing_scenario_predictions.csv"
    )
    return run_root / f"seed_{seed}" / split / name


def load_predictions(run_root: Path, seed: int, split: str, level: str) -> pd.DataFrame:
    path = prediction_path(run_root, seed, split, level)
    if not path.is_file():
        raise FileNotFoundError(f"Missing prediction table: {path}")
    return pd.read_csv(path)


def choose_global_threshold(run_root: Path, level: str) -> tuple[float, dict[int, dict]]:
    score_col = (
        "spoofing_probability"
        if level == "sequence"
        else "top10pct_mean_spoofing_probability"
    )
    selections: dict[int, dict] = {}
    for seed in SEEDS:
        frame = load_predictions(run_root, seed, "validation_eval", level)
        threshold, metrics = select_threshold(
            frame["true_id"].to_numpy(dtype=np.int64),
            frame[score_col].to_numpy(dtype=float),
        )
        selections[seed] = {"threshold": threshold, **metrics}
    threshold = float(np.median([item["threshold"] for item in selections.values()]))
    return threshold, selections


def evaluate_level(
    run_root: Path,
    split: str,
    level: str,
    threshold: float,
) -> pd.DataFrame:
    score_col = (
        "spoofing_probability"
        if level == "sequence"
        else "top10pct_mean_spoofing_probability"
    )
    rows = []
    for seed in SEEDS:
        frame = load_predictions(run_root, seed, split, level)
        metrics = positive_metrics(
            frame["true_id"].to_numpy(dtype=np.int64),
            frame[score_col].to_numpy(dtype=float),
            threshold,
        )
        rows.append({"split": split, "level": level, "seed": seed, **metrics})
    return pd.DataFrame(rows)


def evaluate_attacks(
    run_root: Path,
    split: str,
    threshold: float,
) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        frame = load_predictions(run_root, seed, split, "sequence")
        attack_names = frame["attack_type"].astype(str).str.lower()
        normal = attack_names.isin(["normal", "normal_random"])
        for attack in ATTACKS:
            subset = frame.loc[normal | attack_names.eq(attack)]
            metrics = positive_metrics(
                subset["true_id"].to_numpy(dtype=np.int64),
                subset["spoofing_probability"].to_numpy(dtype=float),
                threshold,
            )
            rows.append(
                {"split": split, "level": "sequence", "seed": seed,
                 "attack_type": attack, **metrics}
            )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metric_cols = [
        "precision", "recall", "f1", "accuracy", "balanced_acc", "macro_f1",
        "average_precision", "roc_auc", "positive_support", "n",
    ]
    rows = []
    for keys, group in frame.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["seeds"] = int(group["seed"].nunique())
        for col in metric_cols:
            values = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = float(values.mean())
            row[f"{col}_std"] = float(values.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def analyze(run_root: Path, out_dir: Path) -> None:
    for seed in SEEDS:
        for split in ("validation_eval", "external_test_eval"):
            for level in ("sequence", "scenario"):
                path = prediction_path(run_root, seed, split, level)
                if not path.is_file():
                    raise FileNotFoundError(f"Incomplete baseline04 output: {path}")

    sequence_threshold, sequence_selections = choose_global_threshold(
        run_root, "sequence"
    )
    scenario_threshold, scenario_selections = choose_global_threshold(
        run_root, "scenario"
    )

    level_frames = []
    attack_frames = []
    for split in ("validation_eval", "external_test_eval"):
        level_frames.extend(
            [
                evaluate_level(run_root, split, "sequence", sequence_threshold),
                evaluate_level(run_root, split, "scenario", scenario_threshold),
            ]
        )
        attack_frames.append(evaluate_attacks(run_root, split, sequence_threshold))

    metrics_by_seed = pd.concat(level_frames, ignore_index=True)
    attacks_by_seed = pd.concat(attack_frames, ignore_index=True)
    metrics_summary = summarize(metrics_by_seed, ["split", "level"])
    attacks_summary = summarize(attacks_by_seed, ["split", "level", "attack_type"])

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_by_seed.to_csv(out_dir / "metrics_by_seed.csv", index=False)
    metrics_summary.to_csv(out_dir / "metrics_mean_std.csv", index=False)
    attacks_by_seed.to_csv(out_dir / "per_attack_metrics_by_seed.csv", index=False)
    attacks_summary.to_csv(out_dir / "per_attack_metrics_mean_std.csv", index=False)

    threshold_report = {
        "selection_data": "validation_only",
        "selection_objective": "positive_class_f1_then_balanced_accuracy",
        "aggregation": "median_of_five_per_seed_optima",
        "sequence_threshold": sequence_threshold,
        "scenario_threshold": scenario_threshold,
        "sequence_per_seed": sequence_selections,
        "scenario_per_seed": scenario_selections,
    }
    (out_dir / "threshold_selection.json").write_text(
        json.dumps(threshold_report, indent=2), encoding="utf-8"
    )

    summary = {
        "run_root": str(run_root.resolve()),
        "baseline": "baseline04_paired_position_only_seq120",
        "status": "reporting_baseline",
        "attacks_in_scope": list(ATTACKS),
        "sequence_threshold": sequence_threshold,
        "scenario_threshold": scenario_threshold,
        "external_role": "diagnostic_not_pristine_holdout",
        "primary_reporting_level": "sequence_and_scenario",
        "excluded_reporting_level": "vessel",
        "reason_vessel_excluded": (
            "Validation source vessels do not provide meaningful positive vessel labels; "
            "spoofing is evaluated at sequence and scenario level."
        ),
        "files": {
            "metrics_by_seed": "metrics_by_seed.csv",
            "metrics_mean_std": "metrics_mean_std.csv",
            "per_attack_by_seed": "per_attack_metrics_by_seed.csv",
            "per_attack_mean_std": "per_attack_metrics_mean_std.csv",
            "threshold_selection": "threshold_selection.json",
        },
    }
    (out_dir / "final_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[spoofing-final] sequence threshold: {sequence_threshold:.3f}")
    print(f"[spoofing-final] scenario threshold: {scenario_threshold:.3f}")
    print(f"[spoofing-final] outputs: {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create validation-thresholded five-seed spoofing analysis."
    )
    parser.add_argument("--run_root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()
    out_dir = args.out_dir or (args.run_root / "final_analysis")
    analyze(args.run_root, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
