from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_manifest(directory: Path, allowed_labels: set[str]) -> dict:
    files = sorted(directory.glob("*.csv"))
    selected = [p for p in files if p.stem.strip().lower() in allowed_labels]
    excluded = [p.name for p in files if p.stem.strip().lower() not in allowed_labels]
    selected_rows = [
        {
            "file": path.name,
            "bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
        }
        for path in selected
    ]
    return {
        "selected_files": selected_rows,
        "excluded_files": excluded,
        "selected_labels": sorted(p.stem.strip().lower() for p in selected),
        "missing_allowed_labels": sorted(
            allowed_labels - {p.stem.strip().lower() for p in selected}
        ),
    }


def _npz_summary(path: Path) -> tuple[dict, set[str], list[str]]:
    data = np.load(path, allow_pickle=True)
    y = data["y"].astype(np.int64)
    groups = {str(x) for x in data["groups"].tolist()}
    feature_cols = [str(x) for x in data["feature_cols"].tolist()]
    source_labels = (
        data["window_source_labels"].astype(str)
        if "window_source_labels" in data.files
        else np.array(["unknown"] * len(y), dtype=str)
    )
    source_values, source_counts = np.unique(source_labels, return_counts=True)
    source_class_counts = {
        str(source): {
            str(int(label)): int(count)
            for label, count in zip(
                *np.unique(y[source_labels == source], return_counts=True)
            )
        }
        for source in source_values.tolist()
    }
    labels, counts = np.unique(y, return_counts=True)
    summary = {
        "path": str(path),
        "sha256": _sha256(path),
        "sequences": int(len(y)),
        "vessels": int(len(groups)),
        "class_counts": {
            str(int(label)): int(count)
            for label, count in zip(labels.tolist(), counts.tolist())
        },
        "seq_len": int(data["X"].shape[1]),
        "feature_count": int(data["X"].shape[2]),
        "feature_cols": feature_cols,
        "source_counts": {
            str(label): int(count)
            for label, count in zip(source_values.tolist(), source_counts.tolist())
        },
        "source_class_counts": source_class_counts,
        "godark_min_distance_from_shore_nm": float(
            np.asarray(data["godark_min_distance_from_shore_nm"]).item()
        ),
        "godark_diversity_protocol": (
            str(np.asarray(data["godark_diversity_protocol"]).item())
            if "godark_diversity_protocol" in data.files
            else "missing"
        ),
    }
    return summary, groups, feature_cols


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit that Go-Dark test data is a pure external domain."
    )
    parser.add_argument("--internal_npz", type=Path, required=True)
    parser.add_argument("--external_npz", type=Path, required=True)
    parser.add_argument("--internal_source_dir", type=Path, required=True)
    parser.add_argument("--external_source_dir", type=Path, required=True)
    parser.add_argument("--allowed_labels", nargs="+", required=True)
    parser.add_argument("--expected_shore_nm", type=float, required=True)
    parser.add_argument("--split_indices", type=Path)
    parser.add_argument("--external_eval_summary", type=Path)
    parser.add_argument("--out_path", type=Path, required=True)
    args = parser.parse_args()

    allowed_labels = {str(x).strip().lower() for x in args.allowed_labels}
    internal_source_manifest = _source_manifest(
        args.internal_source_dir, allowed_labels
    )
    external_source_manifest = _source_manifest(
        args.external_source_dir, allowed_labels
    )
    internal, internal_groups, internal_features = _npz_summary(args.internal_npz)
    external, external_groups, external_features = _npz_summary(args.external_npz)
    overlap = sorted(internal_groups & external_groups)

    checks = {
        "no_mmsi_overlap": len(overlap) == 0,
        "feature_schema_identical": internal_features == external_features,
        "sequence_length_identical": internal["seq_len"] == external["seq_len"],
        "shore_filter_identical": (
            internal["godark_min_distance_from_shore_nm"]
            == external["godark_min_distance_from_shore_nm"]
        ),
        "shore_filter_matches_protocol": (
            internal["godark_min_distance_from_shore_nm"]
            == float(args.expected_shore_nm)
            and external["godark_min_distance_from_shore_nm"]
            == float(args.expected_shore_nm)
        ),
        "external_has_both_classes": set(external["class_counts"]) == {"0", "1"},
        "internal_has_all_allowed_sources": not bool(
            internal_source_manifest["missing_allowed_labels"]
        ),
        "external_has_all_allowed_sources": not bool(
            external_source_manifest["missing_allowed_labels"]
        ),
        "internal_npz_has_only_allowed_sources": (
            set(internal["source_counts"]) == allowed_labels
        ),
        "external_npz_has_only_allowed_sources": (
            set(external["source_counts"]) == allowed_labels
        ),
        "internal_each_source_has_both_classes": all(
            set(internal["source_class_counts"].get(source, {})) == {"0", "1"}
            for source in allowed_labels
        ),
        "external_each_source_has_both_classes": all(
            set(external["source_class_counts"].get(source, {})) == {"0", "1"}
            for source in allowed_labels
        ),
        "internal_diversity_protocol_current": (
            internal["godark_diversity_protocol"]
            == "source_label_duration_cadence_distance_position_v2"
        ),
        "external_diversity_protocol_current": (
            external["godark_diversity_protocol"]
            == "source_label_duration_cadence_distance_position_v2"
        ),
        "internal_test_split_empty": None,
        "internal_train_val_disjoint": None,
        "internal_train_val_group_disjoint": None,
        "internal_train_val_cover_all": None,
        "external_eval_uses_all": None,
        "external_eval_not_tuned": None,
        "external_eval_uses_external_npz": None,
    }

    split_summary = None
    if args.split_indices is not None and args.split_indices.exists():
        split = np.load(args.split_indices, allow_pickle=True)
        train_idx = split["train_idx"].astype(np.int64)
        val_idx = split["val_idx"].astype(np.int64)
        test_idx = split["test_idx"].astype(np.int64)
        checks["internal_test_split_empty"] = len(test_idx) == 0
        checks["internal_train_val_disjoint"] = not bool(
            np.intersect1d(train_idx, val_idx).size
        )
        internal_data = np.load(args.internal_npz, allow_pickle=True)
        internal_groups_array = internal_data["groups"].astype(str)
        checks["internal_train_val_group_disjoint"] = not bool(
            set(internal_groups_array[train_idx].tolist())
            & set(internal_groups_array[val_idx].tolist())
        )
        checks["internal_train_val_cover_all"] = (
            len(np.union1d(train_idx, val_idx)) == internal["sequences"]
        )
        split_summary = {
            "train_sequences": int(len(train_idx)),
            "validation_sequences": int(len(val_idx)),
            "test_sequences": int(len(test_idx)),
        }

    eval_summary = None
    if args.external_eval_summary is not None and args.external_eval_summary.exists():
        eval_summary = json.loads(args.external_eval_summary.read_text(encoding="utf-8"))
        checks["external_eval_uses_all"] = eval_summary.get("eval_split") == "all"
        checks["external_eval_not_tuned"] = not bool(
            eval_summary.get("test_tuning_used", True)
        )
        evaluated_npz = Path(str(eval_summary.get("data_npz", "")))
        if not evaluated_npz.is_absolute():
            evaluated_npz = (Path.cwd() / evaluated_npz).resolve()
        checks["external_eval_uses_external_npz"] = (
            evaluated_npz == args.external_npz.resolve()
        )

    failed = [name for name, passed in checks.items() if passed is False]
    report = {
        "protocol": "internal_train_validation_only__external_test_all",
        "internal_source": {
            "directory": str(args.internal_source_dir),
            **internal_source_manifest,
        },
        "external_source": {
            "directory": str(args.external_source_dir),
            **external_source_manifest,
        },
        "allowed_labels": sorted(allowed_labels),
        "expected_shore_nm": float(args.expected_shore_nm),
        "internal_processed": internal,
        "external_processed": external,
        "mmsi_overlap_count": int(len(overlap)),
        "mmsi_overlap": overlap,
        "internal_split": split_summary,
        "external_eval_summary": eval_summary,
        "checks": checks,
        "status": "valid" if not failed else "invalid",
        "failed_checks": failed,
    }

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[godark-audit] status={report['status']} -> {args.out_path}")
    if failed:
        raise RuntimeError(
            "Go-Dark external protocol audit failed: " + ", ".join(failed)
        )


if __name__ == "__main__":
    main()
