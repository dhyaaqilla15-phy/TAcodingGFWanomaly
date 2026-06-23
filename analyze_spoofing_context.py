from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from agg_utils import confusion_matrix_np, metrics_from_cm, per_class_metrics_from_cm
from dataload import read_ais_csv
from spoofing_simulator import _prep_base_df


CONTEXT_ATTACKS = ("replay", "meaconing", "ghost", "mirroring")
KINEMATIC_ATTACKS = ("gradual_drift", "location_jump")
ALL_ATTACKS = (*KINEMATIC_ATTACKS, *CONTEXT_ATTACKS)


@dataclass(frozen=True)
class ContextThresholds:
    # Guard bands for the current simulator threat model. They are explicit
    # research parameters, not values claimed by the reference paper.
    same_time_residual_deg: float = 1e-7
    history_shape_residual_deg: float = 5e-5
    translation_offset_deg: float = 0.75
    lag_seconds: float = 60.0
    simultaneous_seconds: float = 3 * 3600.0


def clean_id(value: object) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def load_generated(generated_dir: Path) -> pd.DataFrame:
    paths = sorted(
        path for path in generated_dir.glob("spoofed_*.csv")
        if path.name != "spoofed_all.csv"
    )
    if not paths:
        raise FileNotFoundError(f"No spoofed_*.csv found in {generated_dir}")
    usecols = [
        "mmsi", "claimed_mmsi", "timestamp", "lat", "lon", "scenario_id",
        "attack_type", "is_spoofing",
    ]
    frames = [pd.read_csv(path, usecols=usecols) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    frame["claimed_mmsi"] = frame["claimed_mmsi"].map(clean_id)
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    frame["lat"] = pd.to_numeric(frame["lat"], errors="coerce")
    frame["lon"] = pd.to_numeric(frame["lon"], errors="coerce")
    return frame.dropna(subset=["timestamp", "lat", "lon"]).copy()


def load_reference(
    reference_dir: Path,
    wanted_ids: set[str],
    limit_rows: int,
) -> dict[str, pd.DataFrame]:
    parts: dict[str, list[pd.DataFrame]] = {}
    for path in sorted(reference_dir.glob("*.csv")):
        if path.stem.lower() in {"pole_and_line", "trollers"}:
            continue
        frame = _prep_base_df(read_ais_csv(path, limit_rows=limit_rows, chunksize=0))
        frame["mmsi_key"] = frame["mmsi"].map(clean_id)
        frame = frame[frame["mmsi_key"].isin(wanted_ids)]
        for vessel, group in frame.groupby("mmsi_key", sort=False):
            parts.setdefault(str(vessel), []).append(
                group[["timestamp", "lat", "lon"]].copy()
            )
    output = {}
    for vessel, frames in parts.items():
        output[vessel] = (
            pd.concat(frames, ignore_index=True)
            .sort_values("timestamp")
            .drop_duplicates("timestamp", keep="last")
            .reset_index(drop=True)
        )
    return output


def perturb_inputs(
    generated: pd.DataFrame,
    references: dict[str, pd.DataFrame],
    *,
    coordinate_noise_std_deg: float,
    history_keep_frac: float,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rng = np.random.RandomState(int(random_state))
    output_generated = generated.copy()
    noise_std = max(0.0, float(coordinate_noise_std_deg))
    if noise_std > 0.0:
        output_generated["lat"] = (
            output_generated["lat"].to_numpy(dtype=float)
            + rng.normal(0.0, noise_std, size=len(output_generated))
        )
        output_generated["lon"] = (
            output_generated["lon"].to_numpy(dtype=float)
            + rng.normal(0.0, noise_std, size=len(output_generated))
        )

    keep_frac = float(np.clip(history_keep_frac, 0.05, 1.0))
    output_references: dict[str, pd.DataFrame] = {}
    for vessel, frame in references.items():
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        if keep_frac >= 1.0 or len(frame) < 3:
            output_references[vessel] = frame.copy()
            continue
        keep = max(3, int(round(len(frame) * keep_frac)))
        # A trusted archive can be incomplete at either end. Select a stable
        # contiguous retained interval instead of deleting random rows, which
        # would manufacture impossible AIS cadence.
        local_rng = np.random.RandomState(
            int(random_state) + sum(ord(char) for char in str(vessel))
        )
        start = int(local_rng.randint(0, max(1, len(frame) - keep + 1)))
        output_references[vessel] = frame.iloc[start:start + keep].reset_index(drop=True)
    return output_generated, output_references


def fingerprint(coords: np.ndarray) -> bytes:
    delta = np.diff(np.asarray(coords, dtype=float), axis=0)
    return np.round(delta, 5).astype(np.float32).tobytes()


class HistoryMatcher:
    def __init__(self, references: dict[str, pd.DataFrame]):
        self.references = references
        self._indices: dict[tuple[str, int], dict[bytes, list[int]]] = {}

    def _index(self, vessel: str, length: int) -> dict[bytes, list[int]]:
        key = (vessel, int(length))
        if key in self._indices:
            return self._indices[key]
        frame = self.references[vessel]
        coords = frame[["lat", "lon"]].to_numpy(dtype=float)
        index: dict[bytes, list[int]] = {}
        for start in range(max(0, len(coords) - length + 1)):
            code = fingerprint(coords[start:start + length])
            index.setdefault(code, []).append(start)
        self._indices[key] = index
        return index

    def match(self, vessel: str, scenario: pd.DataFrame) -> dict | None:
        if vessel not in self.references:
            return None
        scenario = scenario.sort_values("timestamp")
        coords = scenario[["lat", "lon"]].to_numpy(dtype=float)
        length = len(coords)
        if length < 3 or len(self.references[vessel]) < length:
            return None
        starts = self._index(vessel, length).get(fingerprint(coords), [])
        if not starts:
            # Approximate translation-invariant fallback. It compares eight
            # evenly spaced shape anchors, then fully evaluates the closest
            # candidates. This makes the context branch testable under small
            # coordinate noise without using attack labels or original_mmsi.
            ref_coords_all = self.references[vessel][["lat", "lon"]].to_numpy(dtype=float)
            count = len(ref_coords_all) - length + 1
            if count <= 0:
                return None
            sample_offsets = np.unique(
                np.linspace(0, length - 1, num=min(8, length), dtype=int)
            )
            candidate_starts = np.arange(count, dtype=np.int64)
            sampled = ref_coords_all[
                candidate_starts[:, None] + sample_offsets[None, :]
            ]
            sampled = sampled - sampled[:, :1, :]
            target = coords[sample_offsets] - coords[sample_offsets[0]]
            errors = np.sqrt(np.mean((sampled - target[None, :, :]) ** 2, axis=(1, 2)))
            top_count = min(8, len(errors))
            starts = np.argpartition(errors, top_count - 1)[:top_count].tolist()
        ref = self.references[vessel]
        best = None
        for start in starts:
            candidate = ref.iloc[start:start + length]
            ref_coords = candidate[["lat", "lon"]].to_numpy(dtype=float)
            offsets = coords - ref_coords
            offset = np.median(offsets, axis=0)
            residual = offsets - offset
            residual_rmse = float(np.sqrt(np.mean(residual * residual)))
            item = {
                "start": int(start),
                "residual_rmse_deg": residual_rmse,
                "lat_offset_deg": float(offset[0]),
                "lon_offset_deg": float(offset[1]),
                "reference_timestamps": candidate["timestamp"].to_numpy(dtype=float),
            }
            if best is None or residual_rmse < best["residual_rmse_deg"]:
                best = item
        return best


def scenario_groups(frame: pd.DataFrame, seq_len: int) -> list[tuple[str, pd.DataFrame, str]]:
    groups: list[tuple[str, pd.DataFrame, str]] = []
    attacks = frame[frame["attack_type"].astype(str).str.lower().isin(ALL_ATTACKS)]
    for scenario_id, group in attacks.groupby("scenario_id", sort=True):
        groups.append((str(scenario_id), group.sort_values("timestamp"), str(group["attack_type"].iloc[0]).lower()))

    normal = frame[frame["attack_type"].astype(str).str.lower().eq("normal")]
    for scenario_id, group in normal.groupby("scenario_id", sort=True):
        group = group.sort_values("timestamp")
        # One context window per explicitly generated normal/control scenario.
        # Keeping the original scenario_id is required for alignment-safe
        # fusion with the BiLSTM branch.
        if len(group) >= seq_len:
            groups.append((str(scenario_id), group.iloc[:seq_len].copy(), "normal"))
    return groups


def classify_context(
    group: pd.DataFrame,
    matcher: HistoryMatcher,
    thresholds: ContextThresholds,
) -> tuple[str, dict]:
    claimed = clean_id(group["claimed_mmsi"].iloc[0])
    if claimed not in matcher.references:
        return "ghost", {
            "rule": "claimed_identity_not_in_registry",
            "context_score": 1.0,
        }

    # Mirroring and legitimate reports can be checked directly at equal
    # timestamps. This is more robust than a rounded trajectory fingerprint
    # when adding a large coordinate offset changes floating-point rounding.
    ordered = group.sort_values("timestamp")
    reference = matcher.references[claimed]
    aligned = ordered[["timestamp", "lat", "lon"]].merge(
        reference,
        on="timestamp",
        how="inner",
        suffixes=("_scenario", "_reference"),
    )
    if len(aligned) >= max(3, int(np.ceil(0.90 * len(ordered)))):
        offsets = np.column_stack(
            [
                aligned["lat_scenario"].to_numpy(dtype=float)
                - aligned["lat_reference"].to_numpy(dtype=float),
                aligned["lon_scenario"].to_numpy(dtype=float)
                - aligned["lon_reference"].to_numpy(dtype=float),
            ]
        )
        offset = np.median(offsets, axis=0)
        residual = offsets - offset
        residual_rmse = float(np.sqrt(np.mean(residual * residual)))
        offset_norm = float(np.hypot(offset[0], offset[1]))
        # Same-time normal/mirroring transformations are exact in this
        # simulator. Keep this tolerance tight so a slow vessel under
        # meaconing is not mistaken for an unchanged normal report.
        if residual_rmse <= float(thresholds.same_time_residual_deg):
            predicted = (
                "mirroring"
                if offset_norm > float(thresholds.translation_offset_deg)
                else "normal"
            )
            return predicted, {
                "rule": (
                    "simultaneous_translation_invariant_clone"
                    if predicted == "mirroring"
                    else "same_identity_same_time_reference_match"
                ),
                "time_shift_seconds": 0.0,
                "offset_norm_deg": offset_norm,
                "lat_offset_deg": float(offset[0]),
                "lon_offset_deg": float(offset[1]),
                "residual_rmse_deg": residual_rmse,
                "context_score": 1.0 if predicted != "normal" else 0.0,
            }

    match = matcher.match(claimed, group)
    if (
        match is None
        or match["residual_rmse_deg"]
        > float(thresholds.history_shape_residual_deg)
    ):
        residual = None if match is None else float(match["residual_rmse_deg"])
        return "normal", {
            "rule": "no_history_shape_match",
            "residual_rmse_deg": residual,
            "context_score": 0.0,
        }

    scenario_ts = group.sort_values("timestamp")["timestamp"].to_numpy(dtype=float)
    ref_ts = match.pop("reference_timestamps")
    time_shift = float(np.median(scenario_ts - ref_ts))
    offset_norm = float(np.hypot(match["lat_offset_deg"], match["lon_offset_deg"]))
    if (
        offset_norm > float(thresholds.translation_offset_deg)
        and abs(time_shift) <= float(thresholds.simultaneous_seconds)
    ):
        predicted = "mirroring"
        rule = "simultaneous_translation_invariant_clone"
    elif (
        abs(time_shift) > float(thresholds.lag_seconds)
        and float(scenario_ts.min()) > float(reference["timestamp"].max())
    ):
        predicted = "replay"
        rule = "historical_trajectory_reappears_in_future"
    elif abs(time_shift) > float(thresholds.lag_seconds):
        predicted = "meaconing"
        rule = "historical_positions_reported_with_lag"
    else:
        predicted = "normal"
        rule = "same_identity_same_time_reference_match"
    return predicted, {
        "rule": rule,
        "time_shift_seconds": time_shift,
        "offset_norm_deg": offset_norm,
        "context_score": 1.0 if predicted != "normal" else 0.0,
        **match,
    }


def analyze(
    generated_dir: Path,
    reference_dir: Path,
    out_dir: Path,
    limit_rows: int,
    seq_len: int,
    thresholds: ContextThresholds,
    coordinate_noise_std_deg: float,
    history_keep_frac: float,
    random_state: int,
    required_scenarios_csv: Path | None,
) -> None:
    generated = load_generated(generated_dir)
    registered = set(
        generated.loc[generated["attack_type"].astype(str).str.lower().ne("ghost"), "claimed_mmsi"]
        .map(clean_id)
        .tolist()
    )
    references = load_reference(reference_dir, registered, limit_rows)
    generated, references = perturb_inputs(
        generated,
        references,
        coordinate_noise_std_deg=coordinate_noise_std_deg,
        history_keep_frac=history_keep_frac,
        random_state=random_state,
    )
    matcher = HistoryMatcher(references)

    required_ids: set[str] | None = None
    if required_scenarios_csv is not None:
        required_frame = pd.read_csv(required_scenarios_csv, usecols=["scenario_id"])
        required_ids = set(required_frame["scenario_id"].astype(str))

    rows = []
    for scenario_id, group, true_attack in scenario_groups(generated, seq_len):
        if required_ids is not None and str(scenario_id) not in required_ids:
            continue
        predicted_attack, evidence = classify_context(group, matcher, thresholds)
        true_id = int(true_attack != "normal")
        pred_id = int(predicted_attack != "normal")
        rows.append(
            {
                "scenario_id": scenario_id,
                "claimed_mmsi": clean_id(group["claimed_mmsi"].iloc[0]),
                "true_attack": true_attack,
                "predicted_attack": predicted_attack,
                "true_id": true_id,
                "pred_id": pred_id,
                "correct_binary": bool(true_id == pred_id),
                "correct_attack": bool(true_attack == predicted_attack),
                "n_points": int(len(group)),
                **evidence,
            }
        )
    predictions = pd.DataFrame(rows)
    if predictions.empty:
        raise RuntimeError("No context scenarios were available for analysis.")
    if required_ids is not None:
        produced_ids = set(predictions["scenario_id"].astype(str))
        if produced_ids != required_ids:
            missing = sorted(required_ids - produced_ids)[:10]
            extra = sorted(produced_ids - required_ids)[:10]
            raise RuntimeError(
                "Context/BiLSTM scenario alignment failed: "
                f"missing={missing}, extra={extra}"
            )

    cm = confusion_matrix_np(
        predictions["true_id"].to_numpy(dtype=np.int64),
        predictions["pred_id"].to_numpy(dtype=np.int64),
        2,
    )
    cls = per_class_metrics_from_cm(cm)
    binary = {
        **metrics_from_cm(cm),
        "precision": float(cls["precision"][1]),
        "recall": float(cls["recall"][1]),
        "f1": float(cls["f1"][1]),
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
    }
    per_attack = []
    for attack in ALL_ATTACKS:
        subset = predictions[predictions["true_attack"].isin(["normal", attack])]
        attack_rows = predictions[predictions["true_attack"].eq(attack)]
        per_attack.append(
            {
                "attack_type": attack,
                "scenarios": int(len(attack_rows)),
                "binary_recall": float((attack_rows["pred_id"] == 1).mean()),
                "attack_identification_accuracy": float(
                    (attack_rows["predicted_attack"] == attack).mean()
                ),
                "normal_false_positive_rate": float(
                    (subset.loc[subset["true_attack"].eq("normal"), "pred_id"] == 1).mean()
                ),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_dir / "context_scenario_predictions.csv", index=False)
    pd.DataFrame(per_attack).to_csv(out_dir / "context_per_attack_metrics.csv", index=False)
    summary = {
        "detector": "trusted-history context rule baseline",
        "training_required": False,
        "generated_dir": str(generated_dir.resolve()),
        "reference_dir": str(reference_dir.resolve()),
        "registered_reference_vessels": int(len(references)),
        "thresholds": asdict(thresholds),
        "robustness_condition": {
            "coordinate_noise_std_deg": float(coordinate_noise_std_deg),
            "history_keep_frac": float(history_keep_frac),
            "random_state": int(random_state),
        },
        "binary_metrics": binary,
        "attack_identification_accuracy": float(predictions["correct_attack"].mean()),
        "scope_warning": (
            "Simulator-grounded baseline requiring a trusted identity registry and "
            "trusted trajectory history; do not compare directly with the kinematic LSTM."
        ),
    }
    (out_dir / "context_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[context] scenarios={len(predictions)} binary_f1={binary['f1']:.3f}")
    print(f"[context] outputs={out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate four context-required spoofing attacks.")
    parser.add_argument("--generated_dir", type=Path, required=True)
    parser.add_argument("--reference_dir", type=Path, default=Path("Dataset"))
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--limit_rows", type=int, default=300_000)
    parser.add_argument("--seq_len", type=int, default=120)
    parser.add_argument("--coordinate_noise_std_deg", type=float, default=0.0)
    parser.add_argument("--history_keep_frac", type=float, default=1.0)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--same_time_residual_deg", type=float, default=1e-7)
    parser.add_argument("--history_shape_residual_deg", type=float, default=5e-5)
    parser.add_argument("--translation_offset_deg", type=float, default=0.75)
    parser.add_argument("--lag_seconds", type=float, default=60.0)
    parser.add_argument("--simultaneous_seconds", type=float, default=10800.0)
    parser.add_argument(
        "--required_scenarios_csv",
        type=Path,
        default=None,
        help="Optional BiLSTM scenario table; context output is restricted to exactly these IDs.",
    )
    args = parser.parse_args()
    thresholds = ContextThresholds(
        same_time_residual_deg=float(args.same_time_residual_deg),
        history_shape_residual_deg=float(args.history_shape_residual_deg),
        translation_offset_deg=float(args.translation_offset_deg),
        lag_seconds=float(args.lag_seconds),
        simultaneous_seconds=float(args.simultaneous_seconds),
    )
    analyze(
        args.generated_dir,
        args.reference_dir,
        args.out_dir,
        args.limit_rows,
        args.seq_len,
        thresholds,
        args.coordinate_noise_std_deg,
        args.history_keep_frac,
        args.random_state,
        args.required_scenarios_csv,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
