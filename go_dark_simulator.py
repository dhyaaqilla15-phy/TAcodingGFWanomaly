from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from dataload import read_ais_csv
from data_preparation import DEFAULT_SOURCE_EXCLUDE_LABELS, haversine_km_np, bearing_deg_np


NAUTICAL_MILE_KM = 1.852
DEFAULT_GODARK_SOURCE_INCLUDE_LABELS = (
    "drifting_longlines",
    "fixed_gear",
    "purse_seines",
    "trawlers",
)


@dataclass
class GoDarkSimCfg:
    """
    Generator kandidat suspected AIS disabling sintetis.
    
    Desain candidate-gap mengikuti Welch et al. (2022), Science Advances,
    DOI 10.1126/sciadv.abq2109. Segmen lintasan kontinu dihilangkan dan
    konteks sebelum/sesudahnya diperlakukan sebagai satu event. Generator ini
    tidak mengklaim membuktikan niat atau aktivitas ilegal kapal.
    
    Catatan:
    - Hapus segmen trajectory di tengah perjalanan → observed AIS punya gap besar
    - Output positif adalah suspected gap sintetis, bukan bukti niat ilegal.
    """

    seed: int = 42

    # Biar aman untuk dataset besar
    limit_rows: int = 0
    chunksize: int = 0
    sample_frac: float = 0.0
    include_labels: Sequence[str] = DEFAULT_GODARK_SOURCE_INCLUDE_LABELS
    exclude_labels: Sequence[str] = DEFAULT_SOURCE_EXCLUDE_LABELS

    # Vessel dan event sampling
    max_vessels_per_file: int = 20
    min_points_per_vessel: int = 120
    events_per_vessel: int = 1
    diversity_candidate_pool: int = 48

    # Panjang segmen yang sengaja dihilangkan
    min_hidden_points: int = 20
    max_hidden_points: int = 120

    # Gap candidate. Default 12 jam mengikuti Welch et al. (2022).
    min_dark_seconds: int = 12 * 3600
    max_dark_seconds: int = 7 * 24 * 3600
    min_hidden_distance_km: float = 0.5
    max_source_gap_seconds: int = 3 * 3600
    context_points_each_side: int = 60

    # Geographic context - analyze jauh dari pantai untuk reduce false positives
    min_distance_from_shore_nm: float = 50.0
    shore_distance_unit: str = "meters"

    # Konteks perilaku sebelum disappearance.
    min_pre_gap_speed_knots: float = 1.0        # Kapal harus moving sebelum gap
    max_pre_gap_speed_knots: float = 50.0
    
    # Ping frequency - proxy untuk signal presence
    ping_window_seconds: int = 12 * 3600
    min_ping_count_prev_window: int = 0

    # Kompatibilitas CLI lama; event sekarang hanya memakai dua boundary.
    label_before_points: int = 1
    label_after_points: int = 1

    # Metadata audit saja, bukan fitur label yang dimasukkan ke model.
    # Seberapa ekstrem perubahan behavior sebelum gap
    speed_change_threshold_knots: float = 5.0   # Sudden speed change
    course_change_threshold_deg: float = 45.0   # Sudden course change
    
    # Output
    combine_outputs: bool = False


def _label_set(labels: Sequence[str] | None) -> set[str]:
    return {str(label).strip().lower() for label in (labels or []) if str(label).strip()}


def _input_csvs(
    input_path: Path,
    include_labels: Sequence[str] | None,
    exclude_labels: Sequence[str] | None,
) -> List[Path]:
    included = _label_set(include_labels)
    excluded = _label_set(exclude_labels)
    if input_path.is_dir():
        csvs = sorted(input_path.glob("*.csv"))
    else:
        csvs = [input_path]

    if included:
        skipped = [p for p in csvs if p.stem.strip().lower() not in included]
        csvs = [p for p in csvs if p.stem.strip().lower() in included]
        if skipped:
            names = ", ".join(p.name for p in skipped)
            print(f"[godark] include source labels={sorted(included)} skipped={names}")

    if excluded:
        skipped = [p for p in csvs if p.stem.strip().lower() in excluded]
        csvs = [p for p in csvs if p.stem.strip().lower() not in excluded]
        if skipped:
            names = ", ".join(p.name for p in skipped)
            print(f"[godark] exclude source labels={sorted(excluded)} skipped={names}")
    return csvs


def _prep_base_df(df: pd.DataFrame) -> pd.DataFrame:
    need = ["mmsi", "timestamp", "lat", "lon"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' after column normalization. Columns={list(df.columns)}")

    df = df.copy()
    df["mmsi"] = pd.to_numeric(df["mmsi"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["mmsi", "timestamp", "lat", "lon"]).copy()

    df["mmsi"] = df["mmsi"].astype("int64")
    df["timestamp"] = df["timestamp"].round().astype("int64")
    df = df[(df["lat"].between(-90, 90)) & (df["lon"].between(-180, 180))].copy()

    if "speed" not in df.columns:
        df["speed"] = 0.0
    if "course" not in df.columns:
        df["course"] = 0.0
    if "distance_from_shore" not in df.columns:
        df["distance_from_shore"] = -1.0
    if "distance_from_port" not in df.columns:
        df["distance_from_port"] = -1.0
    if "is_fishing" not in df.columns:
        df["is_fishing"] = -1.0
    if "source" not in df.columns:
        df["source"] = "unknown"

    for c in ["speed", "course", "distance_from_shore", "distance_from_port", "is_fishing"]:
        fill = -1.0 if c not in ["speed", "course"] else 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(fill)

    df["speed"] = df["speed"].clip(0.0, 50.0)
    df["course"] = df["course"].fillna(0.0) % 360.0
    df = df.sort_values(["mmsi", "timestamp"]).drop_duplicates(["mmsi", "timestamp"], keep="last")
    return df.reset_index(drop=True)


def _read_input_csv(path: Path, cfg: GoDarkSimCfg) -> pd.DataFrame:
    df = read_ais_csv(path, limit_rows=cfg.limit_rows, chunksize=cfg.chunksize)
    df = _prep_base_df(df)

    if cfg.sample_frac and 0.0 < cfg.sample_frac < 1.0:
        # Sampling baris merusak kontinuitas temporal. Ambil vessel utuh.
        vessels = np.sort(df["mmsi"].unique())
        rng = np.random.RandomState(int(cfg.seed))
        keep_n = max(1, int(round(len(vessels) * float(cfg.sample_frac))))
        keep = rng.choice(vessels, size=min(keep_n, len(vessels)), replace=False)
        df = df[df["mmsi"].isin(keep)].copy()
        df = df.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)

    return df


def _choose_vessels(df: pd.DataFrame, cfg: GoDarkSimCfg, rng: np.random.RandomState) -> List[int]:
    counts = df.groupby("mmsi").size()
    eligible = counts[counts >= int(cfg.min_points_per_vessel)].index.to_numpy(dtype=np.int64)
    if eligible.size == 0:
        return []
    max_v = int(cfg.max_vessels_per_file)
    if max_v > 0 and eligible.size > max_v:
        eligible = rng.choice(eligible, size=max_v, replace=False)
    return [int(x) for x in eligible.tolist()]


def _event_metrics(g: pd.DataFrame, start_idx: int, end_idx: int) -> Tuple[int, float, float, float]:
    """
    start_idx = index titik terakhir sebelum AIS gelap.
    end_idx = index titik pertama setelah kapal muncul kembali.
    
    Returns: (dark_seconds, distance_km, implied_knots, bearing_deg)
    """
    ts0 = int(g.loc[start_idx, "timestamp"])
    ts1 = int(g.loc[end_idx, "timestamp"])
    dark_seconds = max(1, ts1 - ts0)

    km = float(haversine_km_np(
        np.array([g.loc[start_idx, "lat"]], dtype=float),
        np.array([g.loc[start_idx, "lon"]], dtype=float),
        np.array([g.loc[end_idx, "lat"]], dtype=float),
        np.array([g.loc[end_idx, "lon"]], dtype=float),
    )[0])

    implied_knots = (km / float(dark_seconds)) * (3600.0 / NAUTICAL_MILE_KM)

    bearing = float(bearing_deg_np(
        np.array([g.loc[start_idx, "lat"]], dtype=float),
        np.array([g.loc[start_idx, "lon"]], dtype=float),
        np.array([g.loc[end_idx, "lat"]], dtype=float),
        np.array([g.loc[end_idx, "lon"]], dtype=float),
    )[0])

    return dark_seconds, km, implied_knots, bearing


def _compute_behavioral_markers(
    g: pd.DataFrame,
    start_idx: int,
    cfg: GoDarkSimCfg,
) -> dict:
    """
    Compute behavioral anomaly markers sebelum gap untuk detectability score.
    Metadata deskriptif untuk audit konteks sebelum gap.
    
    Returns dict dengan keys:
    - speed_before_gap: speed kapal sebelum disappearance
    - course_change_magnitude: perubahan heading sebelum gap
    - pre_gap_activity_level: berapa banyak activity sebelum gap
    - behavior_anomaly_score: 0-100 indicator (higher = more suspicious)
    """
    window_size = max(3, min(10, start_idx // 2))
    start_window = max(0, start_idx - window_size)
    
    # Extract speed dan course dari window sebelum gap
    speeds = pd.to_numeric(g.loc[start_window:start_idx, "speed"], errors="coerce").fillna(0)
    courses = pd.to_numeric(g.loc[start_window:start_idx, "course"], errors="coerce").fillna(0)
    
    speed_before = float(speeds.iloc[-1]) if len(speeds) > 0 else 0.0
    speed_mean = float(speeds.mean()) if len(speeds) > 0 else 0.0
    
    # Course change (shortest angle)
    if len(courses) >= 2:
        course_before = float(courses.iloc[-1])
        course_prev = float(courses.iloc[-2])
        dc = (course_before - course_prev) % 360.0
        course_change = min(dc, 360.0 - dc)
    else:
        course_change = 0.0
    
    # Activity level (timestamp gaps)
    if len(g.loc[start_window:start_idx]) > 1:
        ts_diffs = g.loc[start_window:start_idx, "timestamp"].diff().dropna()
        activity = 1.0 / (1.0 + ts_diffs.mean() / 600.0) if len(ts_diffs) > 0 else 0.5
    else:
        activity = 0.5
    
    # Intentionality score (0-100)
    # Higher = more suspicious pattern
    anomaly_score = 0.0
    
    # Criteria 1: Speed consistency before gap (sudden stop/slowdown = suspicious)
    if speed_mean > cfg.min_pre_gap_speed_knots and speed_before < speed_mean * 0.5:
        anomaly_score += 30.0
    elif speed_before < cfg.min_pre_gap_speed_knots:
        anomaly_score += 15.0
    
    # Criteria 2: Course stability (erratic course = suspicious)
    if course_change > cfg.course_change_threshold_deg:
        anomaly_score += 25.0
    
    # Criteria 3: Activity level (too quiet = suspicious)
    if activity < 0.3:
        anomaly_score += 20.0
    
    # Criteria 4: High speed before gap (trying to escape = suspicious)
    if speed_before > 20.0:
        anomaly_score += 15.0
    
    return {
        "speed_before_gap": float(speed_before),
        "speed_mean_window": float(speed_mean),
        "course_change_deg": float(course_change),
        "activity_level_0to1": float(activity),
        "behavior_anomaly_score": min(100.0, max(0.0, anomaly_score)),
    }


def _distance_from_shore_km(g: pd.DataFrame, unit: str = "meters") -> np.ndarray:
    """
    Mengubah distance_from_shore menjadi km dengan satuan eksplisit.
    Nilai <0 dianggap tidak tersedia.
    """
    raw = pd.to_numeric(
        g.get("distance_from_shore", pd.Series(-1.0, index=g.index)),
        errors="coerce",
    )
    x = np.asarray(raw.fillna(-1.0), dtype=np.float64).copy()
    x[~np.isfinite(x)] = -1.0

    valid = x[x >= 0]
    if valid.size == 0:
        return np.full(len(g), np.nan, dtype=np.float64)

    unit_norm = str(unit).strip().lower()
    if unit_norm not in {"meters", "kilometers"}:
        raise ValueError("shore_distance_unit harus 'meters' atau 'kilometers'")
    if unit_norm == "meters":
        x = x / 1000.0

    x[x < 0] = np.nan
    return x.astype(np.float64, copy=False)


def _shore_ok(shore_km: np.ndarray, start_idx: int, end_idx: int, cfg: GoDarkSimCfg) -> bool:
    min_nm = float(cfg.min_distance_from_shore_nm)
    if min_nm <= 0:
        return True

    min_km = min_nm * NAUTICAL_MILE_KM
    vals = np.array([shore_km[start_idx], shore_km[end_idx]], dtype=float)

    # Filter tidak boleh diam-diam lolos jika konteks pantai tidak tersedia.
    if np.isnan(vals).all():
        return False

    valid_vals = vals[~np.isnan(vals)]
    return bool(valid_vals.size > 0 and np.nanmin(valid_vals) >= min_km)


def _source_segment_is_continuous(
    g: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    cfg: GoDarkSimCfg,
) -> bool:
    ts = g.loc[start_idx:end_idx, "timestamp"].to_numpy(dtype=np.int64)
    if ts.size < 2:
        return False
    dt = np.diff(ts)
    return bool(np.all(dt > 0) and np.max(dt) <= int(cfg.max_source_gap_seconds))


def _prev_ping_count_ok(g: pd.DataFrame, start_idx: int, cfg: GoDarkSimCfg) -> Tuple[bool, int]:
    threshold = int(cfg.min_ping_count_prev_window)

    if threshold <= 0:
        return True, 0

    ts = g["timestamp"].to_numpy(dtype=np.int64)
    t0 = int(ts[start_idx])
    lo_t = t0 - int(cfg.ping_window_seconds)

    lo = int(np.searchsorted(ts, lo_t, side="left"))
    hi = int(np.searchsorted(ts, t0, side="left"))

    count = max(0, hi - lo)
    return bool(count >= threshold), int(count)


def _find_event(
    g: pd.DataFrame,
    cfg: GoDarkSimCfg,
    rng: np.random.RandomState,
    used: np.ndarray,
    shore_km: np.ndarray,
    diversity_counts: dict[tuple[int, int, int, int], int],
) -> Tuple[int, int, int] | None:
    n = len(g)

    min_hidden = max(1, int(cfg.min_hidden_points))
    max_hidden = max(min_hidden, int(cfg.max_hidden_points))
    context = max(1, int(cfg.context_points_each_side))
    label_pad = 2 * context

    # Butuh titik sebelum gap, titik tersembunyi, titik muncul kembali, dan label setelahnya.
    if n < min_hidden + label_pad + 4:
        return None

    candidates: list[dict] = []
    seen: set[tuple[int, int]] = set()
    max_attempts = max(700, int(cfg.diversity_candidate_pool) * 12)
    for _ in range(max_attempts):
        hidden_len = int(rng.randint(min_hidden, max_hidden + 1))

        if hidden_len + label_pad + 4 >= n:
            hidden_len = max(min_hidden, n - label_pad - 4)

        if hidden_len < min_hidden:
            return None

        start_min = max(1, context - 1)
        start_max = n - hidden_len - context - 1

        if start_max <= start_min:
            continue

        start_idx = int(rng.randint(start_min, start_max + 1))
        end_idx = start_idx + hidden_len + 1

        if end_idx >= n:
            continue

        # Jangan overlap dengan event lain.
        lo = max(0, start_idx - context + 1)
        hi = min(n, end_idx + context)

        if used[lo:hi].any():
            continue

        dark_seconds, km, _, _ = _event_metrics(g, start_idx, end_idx)

        if dark_seconds < int(cfg.min_dark_seconds):
            continue

        if dark_seconds > int(cfg.max_dark_seconds):
            continue

        if km < float(cfg.min_hidden_distance_km):
            continue

        if not _source_segment_is_continuous(g, start_idx, end_idx, cfg):
            continue

        speed_before = float(pd.to_numeric(g.loc[start_idx, "speed"], errors="coerce"))
        if not (
            float(cfg.min_pre_gap_speed_knots)
            <= speed_before
            <= float(cfg.max_pre_gap_speed_knots)
        ):
            continue

        if not _shore_ok(shore_km, start_idx, end_idx, cfg):
            continue

        ok_ping, ping_count = _prev_ping_count_ok(g, start_idx, cfg)

        if not ok_ping:
            continue
        key = (start_idx, end_idx)
        if key in seen:
            continue
        seen.add(key)
        cadence_per_hour = float(ping_count) / max(
            float(cfg.ping_window_seconds) / 3600.0, 1e-6
        )
        # Fixed, interpretable strata avoid learning only the most common
        # synthetic severity. They are protocol constants, not fitted on the
        # external domain.
        duration_bin = int(np.digitize(dark_seconds, [24 * 3600, 72 * 3600]))
        cadence_bin = int(np.digitize(cadence_per_hour, [2.0, 6.0]))
        distance_bin = int(np.digitize(km, [10.0, 100.0]))
        position_fraction = float((start_idx + end_idx) / (2.0 * max(n - 1, 1)))
        position_bin = int(np.digitize(position_fraction, [1.0 / 3.0, 2.0 / 3.0]))
        candidates.append(
            {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "ping_count": ping_count,
                "lo": lo,
                "hi": hi,
                "cell": (duration_bin, cadence_bin, distance_bin, position_bin),
            }
        )
        if len(candidates) >= int(cfg.diversity_candidate_pool):
            break

    if not candidates:
        return None
    minimum = min(diversity_counts.get(item["cell"], 0) for item in candidates)
    balanced = [
        item for item in candidates
        if diversity_counts.get(item["cell"], 0) == minimum
    ]
    selected = balanced[int(rng.randint(0, len(balanced)))]
    used[int(selected["lo"]):int(selected["hi"])] = True
    cell = selected["cell"]
    diversity_counts[cell] = diversity_counts.get(cell, 0) + 1
    return (
        int(selected["start_idx"]),
        int(selected["end_idx"]),
        int(selected["ping_count"]),
    )


def _init_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["is_go_dark"] = 0
    df["label"] = "Normal"
    df["event_type"] = "normal"
    df["attack_type"] = "normal"
    df["original_mmsi"] = df["mmsi"].astype(str)
    df["go_dark_event_id"] = "normal"
    df["event_phase"] = "normal"

    df["gap_start_timestamp"] = np.nan
    df["gap_end_timestamp"] = np.nan
    df["dark_duration_seconds"] = 0
    df["hidden_points"] = 0
    df["hidden_distance_km"] = 0.0
    df["reappearance_step_km"] = 0.0
    df["implied_speed_knots"] = 0.0
    df["gap_bearing"] = 0.0
    df["ping_count_prev_window"] = 0
    df["min_distance_from_shore_nm_used"] = 0.0
    df["distance_from_shore_km_normalized"] = np.nan
    df["candidate_generation_method"] = "none"
    
    # Metadata perilaku untuk audit; tidak otomatis menjadi fitur model.
    df["speed_before_gap"] = 0.0
    df["speed_mean_window"] = 0.0
    df["course_change_deg"] = 0.0
    df["activity_level_0to1"] = 0.0
    df["behavior_anomaly_score"] = 0.0
    
    df["generation_method"] = "original_ais"
    df["note"] = "original_ais"

    return df


def _apply_go_dark_events_to_vessel(
    g: pd.DataFrame,
    cfg: GoDarkSimCfg,
    rng: np.random.RandomState,
    file_stem: str,
    diversity_counts: dict[tuple[int, int, int, int], int],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    g = g.sort_values("timestamp").reset_index(drop=True).copy()
    g = _init_label_columns(g)

    used = np.zeros(len(g), dtype=bool)
    shore_km = _distance_from_shore_km(g, cfg.shore_distance_unit)
    g["distance_from_shore_km_normalized"] = shore_km

    hidden_indices: set[int] = set()
    event_rows = []
    hidden_parts = []

    vessel = int(g["mmsi"].iloc[0])

    for event_no in range(1, int(cfg.events_per_vessel) + 1):
        found = _find_event(
            g, cfg, rng, used, shore_km, diversity_counts
        )

        if found is None:
            continue

        start_idx, end_idx, ping_count = found
        hidden_idx = list(range(start_idx + 1, end_idx))

        if not hidden_idx:
            continue

        event_id = f"{file_stem}_godark_{vessel}_{event_no:03d}"
        dark_seconds, km, implied_knots, bearing = _event_metrics(g, start_idx, end_idx)
        
        # Compute behavioral markers untuk analysis suspicious pattern
        behavior_markers = _compute_behavioral_markers(g, start_idx, cfg)

        # Hidden truth adalah lintasan yang sengaja dibuat tidak terlihat.
        hidden = g.iloc[hidden_idx].copy()
        hidden["is_go_dark"] = 1
        hidden["label"] = "HiddenDuringGoDark"
        hidden["event_type"] = "go_dark_hidden_truth"
        hidden["attack_type"] = "go_dark_hidden_truth"
        hidden["go_dark_event_id"] = event_id
        hidden["event_phase"] = "hidden_segment"
        hidden["gap_start_timestamp"] = int(g.loc[start_idx, "timestamp"])
        hidden["gap_end_timestamp"] = int(g.loc[end_idx, "timestamp"])
        hidden["dark_duration_seconds"] = int(dark_seconds)
        hidden["hidden_points"] = int(len(hidden_idx))
        hidden["hidden_distance_km"] = float(km)
        hidden["reappearance_step_km"] = float(km)
        hidden["implied_speed_knots"] = float(implied_knots)
        hidden["gap_bearing"] = float(bearing)
        hidden["ping_count_prev_window"] = int(ping_count)
        hidden["min_distance_from_shore_nm_used"] = float(cfg.min_distance_from_shore_nm)
        # Assign behavioral markers
        hidden["speed_before_gap"] = behavior_markers["speed_before_gap"]
        hidden["speed_mean_window"] = behavior_markers["speed_mean_window"]
        hidden["course_change_deg"] = behavior_markers["course_change_deg"]
        hidden["activity_level_0to1"] = behavior_markers["activity_level_0to1"]
        hidden["behavior_anomaly_score"] = behavior_markers["behavior_anomaly_score"]
        hidden["generation_method"] = "synthetic_go_dark_gap_injection"
        hidden["note"] = "ground_truth_removed_points_not_seen_by_ais"
        hidden_parts.append(hidden)

        # Hanya boundary yang dapat diamati diberi metadata. Preprocessing
        # membentuk satu sample event dari konteks sebelum dan sesudah gap.
        for phase, indices in [("pre_gap_boundary", [start_idx]), ("post_gap_boundary", [end_idx])]:
            if not indices:
                continue

            g.loc[indices, "is_go_dark"] = 1
            g.loc[indices, "label"] = "GoDark"
            g.loc[indices, "event_type"] = "go_dark"
            g.loc[indices, "attack_type"] = "go_dark"
            g.loc[indices, "go_dark_event_id"] = event_id
            g.loc[indices, "event_phase"] = phase
            g.loc[indices, "gap_start_timestamp"] = int(g.loc[start_idx, "timestamp"])
            g.loc[indices, "gap_end_timestamp"] = int(g.loc[end_idx, "timestamp"])
            g.loc[indices, "dark_duration_seconds"] = int(dark_seconds)
            g.loc[indices, "hidden_points"] = int(len(hidden_idx))
            g.loc[indices, "hidden_distance_km"] = float(km)
            g.loc[indices, "reappearance_step_km"] = float(km)
            g.loc[indices, "implied_speed_knots"] = float(implied_knots)
            g.loc[indices, "gap_bearing"] = float(bearing)
            g.loc[indices, "ping_count_prev_window"] = int(ping_count)
            g.loc[indices, "min_distance_from_shore_nm_used"] = float(cfg.min_distance_from_shore_nm)
            # Assign behavioral markers untuk boundary points juga
            g.loc[indices, "speed_before_gap"] = behavior_markers["speed_before_gap"]
            g.loc[indices, "speed_mean_window"] = behavior_markers["speed_mean_window"]
            g.loc[indices, "course_change_deg"] = behavior_markers["course_change_deg"]
            g.loc[indices, "activity_level_0to1"] = behavior_markers["activity_level_0to1"]
            g.loc[indices, "behavior_anomaly_score"] = behavior_markers["behavior_anomaly_score"]
            g.loc[indices, "generation_method"] = "synthetic_go_dark_gap_injection"
            g.loc[indices, "candidate_generation_method"] = "rule_gap_context"
            g.loc[indices, "note"] = "synthetic_suspected_ais_disabling_boundary"

        start_shore = shore_km[start_idx] if not np.isnan(shore_km[start_idx]) else np.nan
        end_shore = shore_km[end_idx] if not np.isnan(shore_km[end_idx]) else np.nan

        event_rows.append({
            "go_dark_event_id": event_id,
            "mmsi": vessel,
            "gap_start_timestamp": int(g.loc[start_idx, "timestamp"]),
            "gap_end_timestamp": int(g.loc[end_idx, "timestamp"]),
            "dark_duration_seconds": int(dark_seconds),
            "hidden_points": int(len(hidden_idx)),
            "hidden_distance_km": float(km),
            "reappearance_step_km": float(km),
            "implied_speed_knots": float(implied_knots),
            "gap_bearing": float(bearing),
            "ping_count_prev_window": int(ping_count),
            "ping_window_seconds": int(cfg.ping_window_seconds),
            "min_ping_count_prev_window": int(cfg.min_ping_count_prev_window),
            "start_distance_from_shore_km": float(start_shore) if not np.isnan(start_shore) else np.nan,
            "end_distance_from_shore_km": float(end_shore) if not np.isnan(end_shore) else np.nan,
            "min_distance_from_shore_nm_used": float(cfg.min_distance_from_shore_nm),
            "shore_distance_unit_input": str(cfg.shore_distance_unit),
            "max_source_gap_seconds": int(cfg.max_source_gap_seconds),
            "context_points_each_side": int(cfg.context_points_each_side),
            # Metadata perilaku sebelum gap.
            "speed_before_gap": behavior_markers["speed_before_gap"],
            "speed_mean_window": behavior_markers["speed_mean_window"],
            "course_change_deg": behavior_markers["course_change_deg"],
            "activity_level_0to1": behavior_markers["activity_level_0to1"],
            "behavior_anomaly_score": behavior_markers["behavior_anomaly_score"],
            "generation_method": "synthetic_go_dark_gap_injection",
            "trajectory_position_fraction": float(
                (start_idx + end_idx) / (2.0 * max(len(g) - 1, 1))
            ),
            "trajectory_position_stratum": int(
                np.digitize(
                    (start_idx + end_idx) / (2.0 * max(len(g) - 1, 1)),
                    [1.0 / 3.0, 2.0 / 3.0],
                )
            ),
            "lat_before": float(g.loc[start_idx, "lat"]),
            "lon_before": float(g.loc[start_idx, "lon"]),
            "lat_after": float(g.loc[end_idx, "lat"]),
            "lon_after": float(g.loc[end_idx, "lon"]),
        })

        hidden_indices.update(hidden_idx)

    observed = g.drop(index=sorted(hidden_indices)).reset_index(drop=True)
    events = pd.DataFrame(event_rows)

    if hidden_parts:
        hidden_truth = pd.concat(hidden_parts, ignore_index=True, sort=False)
    else:
        hidden_truth = pd.DataFrame(columns=g.columns)

    return observed, events, hidden_truth


def generate_go_dark_for_file(csv_path: Path, out_dir: Path, cfg: GoDarkSimCfg) -> Path:
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    events_dir = out_dir / "events"
    hidden_dir = out_dir / "hidden_truth"
    summary_dir = out_dir / "summaries"
    events_dir.mkdir(parents=True, exist_ok=True)
    hidden_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    stable_file_hash = int.from_bytes(
        hashlib.blake2b(csv_path.stem.encode("utf-8"), digest_size=4).digest(),
        byteorder="little",
        signed=False,
    )
    rng = np.random.RandomState((int(cfg.seed) + stable_file_hash) % (2**32 - 1))

    print(f"[godark] read: {csv_path}")

    df = _read_input_csv(csv_path, cfg)

    if df.empty:
        raise RuntimeError(f"No valid rows in {csv_path}")
    # Persist the EDA source class through the combined CSV and NPZ. This is
    # metadata for source-aware sampling/auditing, never an input feature.
    df["godark_source_class"] = csv_path.stem.strip().lower()

    vessels = _choose_vessels(df, cfg, rng)

    if not vessels:
        raise RuntimeError(
            f"No eligible vessel with >= {cfg.min_points_per_vessel} points. "
            "Coba kecilkan --min_points_per_vessel atau naikkan --limit_rows."
        )

    selected = df[df["mmsi"].isin(vessels)].copy()

    observed_parts = []
    event_parts = []
    hidden_parts = []
    diversity_counts: dict[tuple[int, int, int, int], int] = {}

    for _, g in selected.groupby("mmsi", sort=False):
        obs, events, hidden = _apply_go_dark_events_to_vessel(
            g=g,
            cfg=cfg,
            rng=rng,
            file_stem=csv_path.stem,
            diversity_counts=diversity_counts,
        )

        observed_parts.append(obs)

        if not events.empty:
            event_parts.append(events)

        if not hidden.empty:
            hidden_parts.append(hidden)

    observed = pd.concat(observed_parts, ignore_index=True, sort=False)
    observed = observed.sort_values(["mmsi", "timestamp", "event_type"]).reset_index(drop=True)

    if observed["is_go_dark"].sum() == 0:
        raise RuntimeError(
            "Tidak ada event go-dark yang berhasil dibuat. Coba turunkan --min_dark_seconds, "
            "--min_hidden_distance_km, --min_distance_from_shore_nm, --min_ping_count_prev_window, "
            "atau --min_points_per_vessel."
        )

    # Jangan forward/back-fill semua kolom. Itu dapat memalsukan konteks
    # pantai, port, dan fishing; derived features dihitung ulang saat preprocess.
    out_path = out_dir / f"godark_{csv_path.stem}.csv"
    observed.to_csv(out_path, index=False)

    events_df = pd.concat(event_parts, ignore_index=True, sort=False) if event_parts else pd.DataFrame()
    if not events_df.empty:
        events_df["duration_stratum"] = np.digitize(
            pd.to_numeric(events_df["dark_duration_seconds"], errors="coerce").fillna(0),
            [24 * 3600, 72 * 3600],
        )
        cadence_per_hour = pd.to_numeric(
            events_df["ping_count_prev_window"], errors="coerce"
        ).fillna(0) / max(float(cfg.ping_window_seconds) / 3600.0, 1e-6)
        events_df["cadence_stratum"] = np.digitize(cadence_per_hour, [2.0, 6.0])
        events_df["distance_stratum"] = np.digitize(
            pd.to_numeric(events_df["hidden_distance_km"], errors="coerce").fillna(0),
            [10.0, 100.0],
        )
    events_path = events_dir / f"events_godark_{csv_path.stem}.csv"
    events_df.to_csv(events_path, index=False)
    if not events_df.empty:
        diversity_audit = (
            events_df.groupby(
                [
                    "duration_stratum",
                    "cadence_stratum",
                    "distance_stratum",
                    "trajectory_position_stratum",
                ],
                dropna=False,
            )
            .agg(events=("go_dark_event_id", "nunique"), vessels=("mmsi", "nunique"))
            .reset_index()
        )
        diversity_audit.to_csv(
            summary_dir / f"diversity_godark_{csv_path.stem}.csv", index=False
        )

    hidden_truth = pd.concat(hidden_parts, ignore_index=True, sort=False) if hidden_parts else pd.DataFrame()
    hidden_path = hidden_dir / f"hidden_truth_godark_{csv_path.stem}.csv"
    hidden_truth.to_csv(hidden_path, index=False)

    summary = (
        observed.groupby(["label", "event_type", "event_phase"], dropna=False)
        .agg(rows=("mmsi", "size"), vessels=("mmsi", "nunique"))
        .reset_index()
        .sort_values(["label", "event_type", "event_phase"])
    )

    summary_path = summary_dir / f"summary_godark_{csv_path.stem}.csv"
    summary.to_csv(summary_path, index=False)

    print(f"[godark] Saved observed dataset: {out_path}")
    print(f"[godark] Saved event table: {events_path}")
    print(f"[godark] Saved hidden truth: {hidden_path}")
    print(f"[godark] Saved summary: {summary_path}")
    print(summary.to_string(index=False))

    return out_path


def generate_go_dark_dataset(input_path: Path, out_dir: Path, cfg: GoDarkSimCfg) -> List[Path]:
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csvs = _input_csvs(input_path, cfg.include_labels, cfg.exclude_labels)

    if not csvs:
        raise FileNotFoundError(f"No CSV found in {input_path}")

    outputs: List[Path] = []

    for p in csvs:
        outputs.append(generate_go_dark_for_file(p, out_dir, cfg))

    if cfg.combine_outputs and len(outputs) > 1:
        combined_path = out_dir / "godark_all.csv"
        first = True

        for p in outputs:
            for chunk in pd.read_csv(p, chunksize=300_000):
                chunk.to_csv(
                    combined_path,
                    mode="w" if first else "a",
                    index=False,
                    header=first,
                )
                first = False

        print(f"[godark] Saved combined dataset: {combined_path}")
        outputs.append(combined_path)

    return outputs
