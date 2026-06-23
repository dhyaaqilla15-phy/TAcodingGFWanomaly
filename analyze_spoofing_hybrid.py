from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _find_column(frame: pd.DataFrame, candidates: list[str], role: str) -> str:
    for column in candidates:
        if column in frame.columns:
            return column
    raise ValueError(f"Missing {role}; expected one of {candidates}")


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = (scores >= float(threshold)).astype(np.int64)
    output = {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "tn": int(((y_true == 0) & (predicted == 0)).sum()),
        "fp": int(((y_true == 0) & (predicted == 1)).sum()),
        "fn": int(((y_true == 1) & (predicted == 0)).sum()),
        "tp": int(((y_true == 1) & (predicted == 1)).sum()),
        "threshold": float(threshold),
    }
    # AP/ROC require a ranking score. A logical-OR decision has only {0,1}
    # values and must not be presented as a calibrated/ranking model score.
    if np.unique(y_true).size == 2 and np.unique(scores).size > 2:
        output["average_precision"] = float(average_precision_score(y_true, scores))
        output["roc_auc"] = float(roc_auc_score(y_true, scores))
    return output


def analyze(
    bilstm_path: Path,
    context_path: Path,
    out_dir: Path,
    bilstm_threshold: float,
    context_threshold: float,
) -> None:
    bilstm = pd.read_csv(bilstm_path)
    context = pd.read_csv(context_path)
    key = "scenario_id"
    if key not in bilstm.columns or key not in context.columns:
        raise ValueError("Both branches must contain scenario_id.")
    if bilstm[key].duplicated().any() or context[key].duplicated().any():
        raise ValueError("Each branch must contain exactly one row per scenario_id.")

    bilstm_ids = set(bilstm[key].astype(str))
    context_ids = set(context[key].astype(str))
    if bilstm_ids != context_ids:
        missing_context = sorted(bilstm_ids - context_ids)[:10]
        missing_bilstm = sorted(context_ids - bilstm_ids)[:10]
        raise RuntimeError(
            "Refusing unaligned hybrid evaluation. Both detectors must score the "
            "same scenarios. "
            f"missing_context={missing_context}, missing_bilstm={missing_bilstm}"
        )

    bilstm_score_col = _find_column(
        bilstm,
        ["top10pct_mean_spoofing_probability", "spoofing_probability"],
        "BiLSTM score",
    )
    context_score_col = _find_column(
        context,
        ["context_score"],
        "context score",
    )
    true_bilstm_col = _find_column(bilstm, ["true_id"], "BiLSTM true label")
    true_context_col = _find_column(context, ["true_id"], "context true label")

    bilstm_attack_col = _find_column(
        bilstm, ["attack_type", "true_attack"], "BiLSTM attack label"
    )
    context_attack_col = _find_column(
        context, ["true_attack", "attack_type"], "context attack label"
    )
    left = bilstm.rename(
        columns={
            true_bilstm_col: "true_id_bilstm",
            bilstm_score_col: "bilstm_score",
            bilstm_attack_col: "attack_type_bilstm",
        }
    ).copy()
    right = context.rename(
        columns={
            true_context_col: "true_id_context",
            context_score_col: "context_score",
            context_attack_col: "attack_type_context",
        }
    ).copy()
    left[key] = left[key].astype(str)
    right[key] = right[key].astype(str)
    merged = left.merge(right, on=key, how="inner", suffixes=("_bilstm_extra", "_context_extra"))
    y_bilstm = merged["true_id_bilstm"].to_numpy(dtype=np.int64)
    y_context = merged["true_id_context"].to_numpy(dtype=np.int64)
    if not np.array_equal(y_bilstm, y_context):
        bad = merged.loc[y_bilstm != y_context, key].head(10).tolist()
        raise RuntimeError(f"True-label mismatch between branches: {bad}")

    bilstm_score = merged["bilstm_score"].to_numpy(dtype=float)
    context_score = merged["context_score"].to_numpy(dtype=float)
    bilstm_alert = bilstm_score >= float(bilstm_threshold)
    context_alert = context_score >= float(context_threshold)
    # Logical OR is represented as a binary system score. It is not presented
    # as a calibrated probability.
    hybrid_score = np.maximum(
        bilstm_alert.astype(float), context_alert.astype(float)
    )
    merged["bilstm_alert"] = bilstm_alert.astype(int)
    merged["context_alert"] = context_alert.astype(int)
    merged["hybrid_score"] = hybrid_score
    merged["hybrid_pred_id"] = (hybrid_score >= 0.5).astype(int)

    attack_col = "attack_type_context"
    per_attack = []
    normal_names = {"normal", "normal_random"}
    attack_names = merged[attack_col].astype(str).str.lower()
    for attack in sorted(set(attack_names) - normal_names):
        subset = attack_names.isin(normal_names | {attack})
        per_attack.append(
            {
                "attack_type": attack,
                **binary_metrics(y_bilstm[subset], hybrid_score[subset], 0.5),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "hybrid_scenario_predictions.csv", index=False)
    pd.DataFrame(per_attack).to_csv(out_dir / "hybrid_per_attack_metrics.csv", index=False)
    summary = {
        "fusion": "logical_or_of_aligned_scenario_alerts",
        "score_warning": "hybrid_score is a decision score, not a calibrated probability",
        "ranking_metrics_omitted": True,
        "bilstm_threshold": float(bilstm_threshold),
        "context_threshold": float(context_threshold),
        "scenarios": int(len(merged)),
        "metrics": binary_metrics(y_bilstm, hybrid_score, 0.5),
    }
    (out_dir / "hybrid_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[hybrid] aligned_scenarios={len(merged)} f1={summary['metrics']['f1']:.3f}")
    print(f"[hybrid] outputs={out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fuse aligned BiLSTM and context scenario predictions."
    )
    parser.add_argument("--bilstm_predictions", type=Path, required=True)
    parser.add_argument("--context_predictions", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--bilstm_threshold", type=float, default=0.45)
    parser.add_argument("--context_threshold", type=float, default=0.50)
    args = parser.parse_args()
    analyze(
        args.bilstm_predictions,
        args.context_predictions,
        args.out_dir,
        args.bilstm_threshold,
        args.context_threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
