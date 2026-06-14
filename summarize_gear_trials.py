from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _per_class_f1(path: Path, scope: str = "vessel") -> dict[str, float]:
    if not path.exists():
        return {}
    out: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("scope") != scope:
                continue
            label = str(row.get("class_label", ""))
            try:
                out[label] = float(row.get("f1", 0.0))
            except ValueError:
                out[label] = 0.0
    return out


def _row_for(model_dir: Path) -> dict[str, object]:
    eval_summary = _load_json(model_dir / "eval_summary.json")
    best = _load_json(model_dir / "best_epoch.json")
    train = _load_json(model_dir / "train_config.json")
    per_class = _per_class_f1(model_dir / "per_class_metrics.csv", scope="vessel")

    run_name = model_dir.parent.name
    if run_name == "gear" and model_dir.parent.parent.name:
        run_name = model_dir.parent.parent.name

    mv = eval_summary.get("metrics_vessel") or {}
    ms = eval_summary.get("metrics_seq") or {}
    return {
        "run": run_name,
        "seed": train.get("random_state"),
        "epochs": train.get("epochs"),
        "geo_aux_weight": train.get("geo_aux_weight"),
        "gear_minority_f1_weight": train.get("gear_minority_f1_weight"),
        "gear_class_weight_power": train.get("gear_class_weight_power"),
        "gear_class_weight_max": train.get("gear_class_weight_max"),
        "gear_tau_max": train.get("gear_tau_max"),
        "best_epoch": best.get("best_epoch"),
        "best_val_macro_f1": best.get("best_val_macro_f1"),
        "best_val_balanced_acc": best.get("best_val_balanced_acc"),
        "best_val_accuracy": best.get("best_val_accuracy"),
        "best_val_rare_mean_f1": best.get("best_val_rare_mean_f1"),
        "best_val_min_present_f1": best.get("best_val_min_present_f1"),
        "tau": best.get("tau"),
        "agg_method": best.get("agg_method"),
        "agg_keep_frac": best.get("agg_keep_frac"),
        "agg_min_keep": best.get("agg_min_keep"),
        "test_vessels": eval_summary.get("test_vessels"),
        "test_macro_f1": mv.get("macro_f1"),
        "test_balanced_acc": mv.get("balanced_acc"),
        "test_accuracy": mv.get("accuracy"),
        "test_weighted_f1": mv.get("weighted_f1"),
        "seq_macro_f1": ms.get("macro_f1"),
        "wrong_high_confidence_count": eval_summary.get("wrong_high_confidence_count"),
        "drifting_longlines_f1": per_class.get("drifting_longlines"),
        "fixed_gear_f1": per_class.get("fixed_gear"),
        "pole_and_line_f1": per_class.get("pole_and_line"),
        "purse_seines_f1": per_class.get("purse_seines"),
        "trawlers_f1": per_class.get("trawlers"),
        "trollers_f1": per_class.get("trollers"),
    }


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/gear_structured_trials/runs")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else root.parent / "summary.csv"
    model_dirs = sorted(root.glob("**/model_gear"))
    rows = [_row_for(p) for p in model_dirs if (p / "eval_summary.json").exists()]

    fieldnames = [
        "run",
        "seed",
        "epochs",
        "geo_aux_weight",
        "gear_minority_f1_weight",
        "gear_class_weight_power",
        "gear_class_weight_max",
        "gear_tau_max",
        "best_epoch",
        "best_val_macro_f1",
        "best_val_balanced_acc",
        "best_val_accuracy",
        "best_val_rare_mean_f1",
        "best_val_min_present_f1",
        "tau",
        "agg_method",
        "agg_keep_frac",
        "agg_min_keep",
        "test_vessels",
        "test_macro_f1",
        "test_balanced_acc",
        "test_accuracy",
        "test_weighted_f1",
        "seq_macro_f1",
        "wrong_high_confidence_count",
        "drifting_longlines_f1",
        "fixed_gear_f1",
        "pole_and_line_f1",
        "purse_seines_f1",
        "trawlers_f1",
        "trollers_f1",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[summarize_gear_trials] wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
