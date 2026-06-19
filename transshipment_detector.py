from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from dataload import read_ais_csv
from data_preparation import DEFAULT_SOURCE_EXCLUDE_LABELS, haversine_km_np

try:
    from sklearn.neighbors import BallTree
except Exception:  # pragma: no cover - sklearn is in requirements, fallback is defensive.
    BallTree = None


EARTH_RADIUS_KM = 6371.0088
NAUTICAL_MILE_KM = 1.852


TRANS_FEATURE_COLS = [
    "event_mode_id",
    "distance_between_km",
    "speed_a",
    "speed_b",
    "speed_pair_mean",
    "relative_speed_knots",
    "course_diff_deg",
    "same_direction_score",
    "lat_mid",
    "lon_mid",
    "shore_km_min",
    "port_km_min",
    "duration_nearby_minutes",
    "event_duration_minutes",
    "both_slow",
    "is_offshore",
    "is_port_far",
    "is_fishing_a",
    "is_fishing_b",
    "gear_a_id",
    "gear_b_id",
    "loitering_spatial_range_km",
    "loitering_start_end_km",
    "loitering_compactness",
    "loitering_turn_rate_abs",
    "loitering_duration_minutes",
    "encounter_rule_score",
    "loitering_rule_score",
    "risk_score",
    "valid_point",
]

TRANS_OUTPUT_COLUMNS = [
    "event_id",
    "event_kind",
    "label",
    "class_id",
    "is_transshipment",
    "timestamp",
    "mmsi_a",
    "mmsi_b",
    "pair_id",
    "lat_a",
    "lon_a",
    "lat_b",
    "lon_b",
    "gear_a_label",
    "gear_b_label",
    *TRANS_FEATURE_COLS,
]


@dataclass
class TransshipmentCfg:
    """
    Weak-label AIS transshipment candidate detector.

    Encounter defaults follow recent GFW-aligned literature:
    <=0.5 km, >=2 hours, <2 knots, away from anchorage/port proxy.
    Loitering defaults follow carrier-loitering practice:
    <2 knots, >=8 hours, >=20 nautical miles offshore.
    """

    seed: int = 42
    mode: str = "both"  # both | encounter | loitering

    limit_rows: int = 0
    chunksize: int = 0
    sample_frac: float = 0.0
    exclude_labels: Sequence[str] = DEFAULT_SOURCE_EXCLUDE_LABELS

    max_vessels_per_file: int = 60
    min_points_per_vessel: int = 40

    grid_minutes: int = 10
    max_interp_gap_minutes: int = 90

    encounter_distance_km: float = 0.5
    encounter_candidate_distance_km: float = 2.0
    encounter_min_hours: float = 2.0
    encounter_max_speed_knots: float = 2.0
    encounter_min_port_km: float = 10.0
    encounter_merge_gap_minutes: int = 30

    loitering_min_hours: float = 8.0
    loitering_max_speed_knots: float = 2.0
    loitering_min_shore_nm: float = 20.0
    loitering_candidate_speed_knots: float = 4.0
    loitering_merge_gap_minutes: int = 30

    normal_min_hours: float = 0.5
    max_encounter_events_per_file: int = 250
    max_loitering_events_per_file: int = 250
    max_normal_events_per_file: int = 500
    synthetic_encounters_per_file: int = 0

    combine_outputs: bool = False


def _wrap_lon(lon: np.ndarray) -> np.ndarray:
    return ((lon + 180.0) % 360.0) - 180.0


def _label_set(labels: Sequence[str] | None) -> set[str]:
    return {str(label).strip().lower() for label in (labels or []) if str(label).strip()}


def _input_csvs(input_path: Path, exclude_labels: Sequence[str] | None) -> List[Path]:
    excluded = _label_set(exclude_labels)
    if input_path.is_dir():
        csvs = sorted(input_path.glob("*.csv"))
    else:
        csvs = [input_path]

    if excluded:
        skipped = [p for p in csvs if p.stem.strip().lower() in excluded]
        csvs = [p for p in csvs if p.stem.strip().lower() not in excluded]
        if skipped:
            names = ", ".join(p.name for p in skipped)
            print(f"[transshipment] exclude source labels={sorted(excluded)} skipped={names}")
    return csvs


def _angle_diff_deg(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    d = (np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) % 360.0
    return np.abs(((d + 180.0) % 360.0) - 180.0).astype(float)


def _distance_col_km(s: pd.Series, n: int) -> np.ndarray:
    x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64, copy=True)
    if x.size != n:
        x = np.full((n,), np.nan, dtype=np.float64)
    x[~np.isfinite(x)] = np.nan
    valid = x[x >= 0.0]
    if valid.size == 0:
        return np.full((n,), np.nan, dtype=np.float64)
    if float(np.nanmedian(valid)) > 1000.0:
        x = x / 1000.0
    x[x < 0.0] = np.nan
    return x.astype(np.float64, copy=False)


def _prep_base_df(df: pd.DataFrame, gear_label: str, gear_id: int) -> pd.DataFrame:
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
        df["distance_from_shore"] = np.nan
    if "distance_from_port" not in df.columns:
        df["distance_from_port"] = np.nan
    if "is_fishing" not in df.columns:
        df["is_fishing"] = -1.0
    if "source" not in df.columns:
        df["source"] = "unknown"

    df["speed"] = pd.to_numeric(df["speed"], errors="coerce").fillna(0.0).clip(0.0, 50.0)
    df["course"] = pd.to_numeric(df["course"], errors="coerce").fillna(0.0) % 360.0
    df["is_fishing"] = pd.to_numeric(df["is_fishing"], errors="coerce").fillna(-1.0)
    df["shore_km"] = _distance_col_km(df["distance_from_shore"], len(df))
    df["port_km"] = _distance_col_km(df["distance_from_port"], len(df))
    df["gear_label"] = str(gear_label)
    df["gear_id"] = int(gear_id)

    df = df.sort_values(["mmsi", "timestamp"]).drop_duplicates(["mmsi", "timestamp"], keep="last")
    return df.reset_index(drop=True)


def _read_input_path(input_path: Path, cfg: TransshipmentCfg) -> pd.DataFrame:
    input_path = Path(input_path)
    csvs = _input_csvs(input_path, cfg.exclude_labels)
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {input_path}")

    gear_labels = sorted({p.stem.strip().lower() for p in csvs})
    gear_to_id = {g: i for i, g in enumerate(gear_labels)}

    parts: List[pd.DataFrame] = []
    for p in csvs:
        gear = p.stem.strip().lower()
        print(f"[transshipment] read: {p}")
        df = read_ais_csv(p, limit_rows=int(cfg.limit_rows), chunksize=int(cfg.chunksize))
        df = _prep_base_df(df, gear_label=gear, gear_id=gear_to_id[gear])
        if not df.empty:
            parts.append(df)

    if not parts:
        raise RuntimeError(f"No valid AIS rows in {input_path}")

    df_all = pd.concat(parts, ignore_index=True, sort=False)
    if cfg.sample_frac and 0.0 < float(cfg.sample_frac) < 1.0:
        df_all = df_all.sample(frac=float(cfg.sample_frac), random_state=int(cfg.seed)).copy()
    df_all = df_all.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)
    return df_all


def _choose_vessels(df: pd.DataFrame, cfg: TransshipmentCfg, rng: np.random.RandomState) -> List[int]:
    counts = df.groupby("mmsi").size()
    eligible = counts[counts >= int(cfg.min_points_per_vessel)].index.to_numpy(dtype=np.int64)
    if eligible.size == 0:
        return []
    max_v = int(cfg.max_vessels_per_file)
    if max_v > 0 and eligible.size > max_v:
        eligible = rng.choice(eligible, size=max_v, replace=False)
    return [int(x) for x in eligible.tolist()]


def _interp_angle_deg(vals: np.ndarray, lo: np.ndarray, hi: np.ndarray, frac: np.ndarray) -> np.ndarray:
    rad = np.deg2rad(vals.astype(float) % 360.0)
    s = np.sin(rad)
    c = np.cos(rad)
    si = s[lo] * (1.0 - frac) + s[hi] * frac
    ci = c[lo] * (1.0 - frac) + c[hi] * frac
    return (np.rad2deg(np.arctan2(si, ci)) + 360.0) % 360.0


def _regularize_vessel(g: pd.DataFrame, cfg: TransshipmentCfg) -> pd.DataFrame:
    g = g.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    if len(g) < 2:
        return pd.DataFrame()

    grid_sec = max(60, int(cfg.grid_minutes) * 60)
    max_gap = max(grid_sec, int(cfg.max_interp_gap_minutes) * 60)
    ts = g["timestamp"].to_numpy(dtype=np.int64)
    start = int(np.ceil(ts[0] / grid_sec) * grid_sec)
    end = int(np.floor(ts[-1] / grid_sec) * grid_sec)
    if end < start:
        return pd.DataFrame()

    grid = np.arange(start, end + grid_sec, grid_sec, dtype=np.int64)
    idx = np.searchsorted(ts, grid, side="left")
    exact = (idx < len(ts)) & (ts[np.clip(idx, 0, len(ts) - 1)] == grid)
    lo = np.where(exact, idx, idx - 1)
    hi = np.where(exact, idx, idx)
    valid = (lo >= 0) & (hi < len(ts))
    lo = lo[valid]
    hi = hi[valid]
    grid = grid[valid]
    if grid.size == 0:
        return pd.DataFrame()

    dt = (ts[hi] - ts[lo]).astype(float)
    valid_gap = (dt <= float(max_gap)) | (dt == 0.0)
    lo = lo[valid_gap]
    hi = hi[valid_gap]
    grid = grid[valid_gap]
    dt = dt[valid_gap]
    if grid.size == 0:
        return pd.DataFrame()

    frac = np.zeros_like(dt, dtype=float)
    nz = dt > 0.0
    frac[nz] = (grid[nz] - ts[lo[nz]]) / dt[nz]
    frac = np.clip(frac, 0.0, 1.0)

    def interp_num(col: str) -> np.ndarray:
        vals = g[col].to_numpy(dtype=float)
        return vals[lo] * (1.0 - frac) + vals[hi] * frac

    out = pd.DataFrame(
        {
            "timestamp": grid.astype(np.int64),
            "mmsi": int(g["mmsi"].iloc[0]),
            "lat": interp_num("lat"),
            "lon": _wrap_lon(interp_num("lon")),
            "speed": interp_num("speed"),
            "course": _interp_angle_deg(g["course"].to_numpy(dtype=float), lo, hi, frac),
            "shore_km": interp_num("shore_km"),
            "port_km": interp_num("port_km"),
            "is_fishing": interp_num("is_fishing"),
            "gear_id": int(g["gear_id"].iloc[0]),
            "gear_label": str(g["gear_label"].iloc[0]),
        }
    )
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=["lat", "lon", "speed", "course"])


def _regularize_all(df: pd.DataFrame, cfg: TransshipmentCfg) -> pd.DataFrame:
    rng = np.random.RandomState(int(cfg.seed))
    vessels = _choose_vessels(df, cfg, rng)
    if not vessels:
        raise RuntimeError(
            f"No eligible vessel with >= {cfg.min_points_per_vessel} points. "
            "Coba kecilkan --min_points_per_vessel atau naikkan --limit_rows."
        )

    parts = []
    for vessel in vessels:
        g = df[df["mmsi"] == int(vessel)]
        rg = _regularize_vessel(g, cfg)
        if not rg.empty:
            parts.append(rg)

    if not parts:
        raise RuntimeError("No regularized vessel tracks were created. Try larger --limit_rows or --max_interp_gap_minutes.")

    reg = pd.concat(parts, ignore_index=True, sort=False)
    reg = reg.sort_values(["timestamp", "mmsi"]).reset_index(drop=True)
    print(
        f"[transshipment] regularized rows={len(reg)} vessels={reg['mmsi'].nunique()} "
        f"grid_minutes={cfg.grid_minutes}"
    )
    return reg


def _vx(speed: float, course: float) -> float:
    return float(speed) * float(np.cos(np.deg2rad(course)))


def _vy(speed: float, course: float) -> float:
    return float(speed) * float(np.sin(np.deg2rad(course)))


def _min_valid(a: float, b: float) -> float:
    vals = [float(x) for x in [a, b] if np.isfinite(float(x)) and float(x) >= 0.0]
    return float(min(vals)) if vals else -1.0


def _pairs_within_radius(g: pd.DataFrame, radius_km: float) -> List[Tuple[int, int, float]]:
    if len(g) < 2:
        return []
    lat = g["lat"].to_numpy(dtype=float)
    lon = g["lon"].to_numpy(dtype=float)

    if BallTree is not None:
        coords = np.deg2rad(np.column_stack([lat, lon]))
        tree = BallTree(coords, metric="haversine")
        neigh = tree.query_radius(coords, r=float(radius_km) / EARTH_RADIUS_KM)
        pairs: List[Tuple[int, int, float]] = []
        for i, arr in enumerate(neigh):
            for j in arr:
                j = int(j)
                if j <= i:
                    continue
                d = float(haversine_km_np(np.array([lat[i]]), np.array([lon[i]]), np.array([lat[j]]), np.array([lon[j]]))[0])
                if d <= float(radius_km):
                    pairs.append((i, j, d))
        return pairs

    pairs = []
    for i in range(len(g) - 1):
        d = haversine_km_np(
            np.full(len(g) - i - 1, lat[i]),
            np.full(len(g) - i - 1, lon[i]),
            lat[i + 1:],
            lon[i + 1:],
        )
        for off, dist in enumerate(d.tolist(), start=1):
            if float(dist) <= float(radius_km):
                pairs.append((i, i + off, float(dist)))
    return pairs


def _base_feature_row(
    *,
    timestamp: int,
    event_mode_id: int,
    mmsi_a: int,
    mmsi_b: int | None,
    pair_id: str,
    lat_a: float,
    lon_a: float,
    lat_b: float,
    lon_b: float,
    speed_a: float,
    speed_b: float,
    course_a: float,
    course_b: float,
    shore_a: float,
    shore_b: float,
    port_a: float,
    port_b: float,
    is_fishing_a: float,
    is_fishing_b: float,
    gear_a_id: int,
    gear_b_id: int,
    gear_a_label: str,
    gear_b_label: str,
    distance_between_km: float,
) -> dict:
    course_diff = float(_angle_diff_deg(course_a, course_b))
    rel_speed = float(np.hypot(_vx(speed_a, course_a) - _vx(speed_b, course_b), _vy(speed_a, course_a) - _vy(speed_b, course_b)))
    speed_pair = float((float(speed_a) + float(speed_b)) / 2.0)
    same_dir = float((np.cos(np.deg2rad(course_diff)) + 1.0) / 2.0)
    shore_min = _min_valid(shore_a, shore_b)
    port_min = _min_valid(port_a, port_b)

    return {
        "event_id": "",
        "event_kind": "candidate",
        "label": "Normal",
        "class_id": 0,
        "is_transshipment": 0,
        "timestamp": int(timestamp),
        "mmsi_a": int(mmsi_a),
        "mmsi_b": "" if mmsi_b is None else int(mmsi_b),
        "pair_id": str(pair_id),
        "lat_a": float(lat_a),
        "lon_a": float(lon_a),
        "lat_b": float(lat_b) if np.isfinite(float(lat_b)) else np.nan,
        "lon_b": float(lon_b) if np.isfinite(float(lon_b)) else np.nan,
        "gear_a_label": str(gear_a_label),
        "gear_b_label": str(gear_b_label),
        "event_mode_id": int(event_mode_id),
        "distance_between_km": float(distance_between_km),
        "speed_a": float(speed_a),
        "speed_b": float(speed_b),
        "speed_pair_mean": speed_pair,
        "relative_speed_knots": rel_speed,
        "course_diff_deg": course_diff,
        "same_direction_score": same_dir,
        "lat_mid": float((float(lat_a) + float(lat_b)) / 2.0) if np.isfinite(float(lat_b)) else float(lat_a),
        "lon_mid": float(_wrap_lon(np.array([(float(lon_a) + float(lon_b)) / 2.0]))[0]) if np.isfinite(float(lon_b)) else float(lon_a),
        "shore_km_min": shore_min,
        "port_km_min": port_min,
        "duration_nearby_minutes": 0.0,
        "event_duration_minutes": 0.0,
        "both_slow": 0,
        "is_offshore": 0,
        "is_port_far": 0,
        "is_fishing_a": float(is_fishing_a),
        "is_fishing_b": float(is_fishing_b),
        "gear_a_id": int(gear_a_id),
        "gear_b_id": int(gear_b_id),
        "loitering_spatial_range_km": 0.0,
        "loitering_start_end_km": 0.0,
        "loitering_compactness": 0.0,
        "loitering_turn_rate_abs": 0.0,
        "loitering_duration_minutes": 0.0,
        "encounter_rule_score": 0.0,
        "loitering_rule_score": 0.0,
        "risk_score": 0.0,
        "valid_point": 1.0,
    }


def _duration_minutes(seg: pd.DataFrame, grid_minutes: int) -> float:
    if seg.empty:
        return 0.0
    if len(seg) == 1:
        return float(grid_minutes)
    return float((int(seg["timestamp"].iloc[-1]) - int(seg["timestamp"].iloc[0])) / 60.0 + float(grid_minutes))


def _encounter_risk(seg: pd.DataFrame, cfg: TransshipmentCfg, duration_min: float) -> float:
    med_dist = float(np.nanmedian(seg["distance_between_km"]))
    med_speed = float(np.nanmedian(seg["speed_pair_mean"]))
    port_vals = seg["port_km_min"].to_numpy(dtype=float)
    valid_port = port_vals[port_vals >= 0.0]
    port_med = float(np.nanmedian(valid_port)) if valid_port.size else float(cfg.encounter_min_port_km)

    prox = 1.0 - np.clip(med_dist / max(float(cfg.encounter_candidate_distance_km), 1e-6), 0.0, 1.0)
    dur = np.clip(duration_min / max(float(cfg.encounter_min_hours) * 60.0, 1.0), 0.0, 1.0)
    slow = 1.0 - np.clip(med_speed / max(float(cfg.encounter_max_speed_knots) * 2.0, 1e-6), 0.0, 1.0)
    port = np.clip(port_med / max(float(cfg.encounter_min_port_km), 1e-6), 0.0, 1.0)
    return float(np.clip((0.35 * prox) + (0.25 * dur) + (0.25 * slow) + (0.15 * port), 0.0, 1.0))


def _select_parts(
    parts: List[pd.DataFrame],
    summaries: List[dict],
    positive_class_id: int,
    max_positive: int,
    max_normal: int,
    rng: np.random.RandomState,
) -> Tuple[List[pd.DataFrame], List[dict]]:
    if not parts:
        return [], []

    pos = [i for i, s in enumerate(summaries) if int(s.get("class_id", 0)) == int(positive_class_id)]
    neg = [i for i, s in enumerate(summaries) if int(s.get("class_id", 0)) == 0]

    if max_positive > 0 and len(pos) > max_positive:
        pos = rng.choice(pos, size=int(max_positive), replace=False).tolist()
    if max_normal > 0 and len(neg) > max_normal:
        neg = rng.choice(neg, size=int(max_normal), replace=False).tolist()

    keep = sorted(pos + neg)
    return [parts[i] for i in keep], [summaries[i] for i in keep]


def _build_encounter_events(reg: pd.DataFrame, cfg: TransshipmentCfg, rng: np.random.RandomState, run_stem: str) -> Tuple[List[pd.DataFrame], List[dict]]:
    candidate_rows: List[dict] = []
    radius = max(float(cfg.encounter_candidate_distance_km), float(cfg.encounter_distance_km))

    cols = [
        "mmsi",
        "timestamp",
        "lat",
        "lon",
        "speed",
        "course",
        "shore_km",
        "port_km",
        "is_fishing",
        "gear_id",
        "gear_label",
    ]
    vessel_tracks = [
        (int(v), g[cols].sort_values("timestamp").reset_index(drop=True))
        for v, g in reg.groupby("mmsi", sort=False)
        if len(g) > 0
    ]
    vessel_tracks = sorted(vessel_tracks, key=lambda x: x[0])
    if len(vessel_tracks) < 2:
        return [], []

    n_pairs = int(len(vessel_tracks) * (len(vessel_tracks) - 1) / 2)
    print(
        f"[transshipment] encounter scan: vessels={len(vessel_tracks)} pairs={n_pairs} "
        f"radius_km={radius}",
        flush=True,
    )

    checked_pairs = 0
    for ia in range(len(vessel_tracks) - 1):
        mmsi_a, ga = vessel_tracks[ia]
        for ib in range(ia + 1, len(vessel_tracks)):
            mmsi_b, gb = vessel_tracks[ib]
            checked_pairs += 1
            if n_pairs >= 100 and checked_pairs % 100 == 0:
                print(
                    f"[transshipment] encounter scan progress: {checked_pairs}/{n_pairs} "
                    f"pairs, candidate_points={len(candidate_rows)}",
                    flush=True,
                )

            merged = ga.merge(
                gb,
                on="timestamp",
                how="inner",
                suffixes=("_a", "_b"),
                copy=False,
            )
            if merged.empty:
                continue

            lat_a = merged["lat_a"].to_numpy(dtype=float)
            lon_a = merged["lon_a"].to_numpy(dtype=float)
            lat_b = merged["lat_b"].to_numpy(dtype=float)
            lon_b = merged["lon_b"].to_numpy(dtype=float)

            lat_gate = np.abs(lat_a - lat_b) <= (radius / 110.574)
            cos_lat = np.maximum(0.15, np.abs(np.cos(np.deg2rad((lat_a + lat_b) / 2.0))))
            lon_delta = np.abs(_wrap_lon(lon_a - lon_b))
            lon_gate = lon_delta <= (radius / (111.320 * cos_lat))
            gate = lat_gate & lon_gate
            if not bool(gate.any()):
                continue

            dist = np.full((len(merged),), np.inf, dtype=np.float64)
            dist[gate] = haversine_km_np(lat_a[gate], lon_a[gate], lat_b[gate], lon_b[gate])
            hit_idx = np.where(dist <= radius)[0]
            if hit_idx.size == 0:
                continue

            pair_id = f"{mmsi_a}__{mmsi_b}"
            for idx in hit_idx.tolist():
                r = merged.iloc[int(idx)]
                d = float(dist[int(idx)])
                ts = int(r["timestamp"])
                candidate_rows.append(
                    _base_feature_row(
                        timestamp=int(ts),
                        event_mode_id=1,
                        mmsi_a=mmsi_a,
                        mmsi_b=mmsi_b,
                        pair_id=pair_id,
                        lat_a=float(r["lat_a"]),
                        lon_a=float(r["lon_a"]),
                        lat_b=float(r["lat_b"]),
                        lon_b=float(r["lon_b"]),
                        speed_a=float(r["speed_a"]),
                        speed_b=float(r["speed_b"]),
                        course_a=float(r["course_a"]),
                        course_b=float(r["course_b"]),
                        shore_a=float(r["shore_km_a"]),
                        shore_b=float(r["shore_km_b"]),
                        port_a=float(r["port_km_a"]),
                        port_b=float(r["port_km_b"]),
                        is_fishing_a=float(r["is_fishing_a"]),
                        is_fishing_b=float(r["is_fishing_b"]),
                        gear_a_id=int(r["gear_id_a"]),
                        gear_b_id=int(r["gear_id_b"]),
                        gear_a_label=str(r["gear_label_a"]),
                        gear_b_label=str(r["gear_label_b"]),
                        distance_between_km=d,
                    )
                )

    if not candidate_rows:
        print("[transshipment] encounter scan done: candidate_points=0", flush=True)
        return [], []

    print(
        f"[transshipment] encounter scan done: candidate_points={len(candidate_rows)}",
        flush=True,
    )

    cand = pd.DataFrame(candidate_rows)
    parts: List[pd.DataFrame] = []
    summaries: List[dict] = []
    grid_sec = max(60, int(cfg.grid_minutes) * 60)
    merge_gap_sec = max(grid_sec, int(cfg.encounter_merge_gap_minutes) * 60)

    event_no = 0
    for pair_id, pg in cand.groupby("pair_id", sort=False):
        pg = pg.sort_values("timestamp").reset_index(drop=True)
        gaps = pg["timestamp"].diff().fillna(0).to_numpy(dtype=float) > float(merge_gap_sec)
        seg_ids = np.cumsum(gaps.astype(int))
        for _, seg in pg.groupby(seg_ids, sort=False):
            seg = seg.sort_values("timestamp").reset_index(drop=True)
            duration = _duration_minutes(seg, int(cfg.grid_minutes))
            if duration < float(cfg.normal_min_hours) * 60.0:
                continue

            med_dist = float(np.nanmedian(seg["distance_between_km"]))
            med_speed = float(np.nanmedian(seg["speed_pair_mean"]))
            close_rate = float(np.mean(seg["distance_between_km"].to_numpy(dtype=float) <= float(cfg.encounter_distance_km)))
            slow_rate = float(np.mean(seg["speed_pair_mean"].to_numpy(dtype=float) < float(cfg.encounter_max_speed_knots)))

            port_vals = seg["port_km_min"].to_numpy(dtype=float)
            valid_port = port_vals[port_vals >= 0.0]
            port_ok = bool(valid_port.size == 0 or np.nanmedian(valid_port) >= float(cfg.encounter_min_port_km))

            positive = (
                med_dist <= float(cfg.encounter_distance_km)
                and duration >= float(cfg.encounter_min_hours) * 60.0
                and med_speed < float(cfg.encounter_max_speed_knots)
                and close_rate >= 0.80
                and slow_rate >= 0.80
                and port_ok
            )

            event_no += 1
            event_id = f"{run_stem}_encounter_{event_no:06d}"
            risk = _encounter_risk(seg, cfg, duration)
            seg = seg.copy()
            seg["event_id"] = event_id
            seg["event_kind"] = "encounter"
            seg["label"] = "Encounter" if positive else "Normal"
            seg["class_id"] = 1 if positive else 0
            seg["is_transshipment"] = 1 if positive else 0
            seg["duration_nearby_minutes"] = float(duration)
            seg["event_duration_minutes"] = float(duration)
            seg["both_slow"] = ((seg["speed_a"] < float(cfg.encounter_max_speed_knots)) & (seg["speed_b"] < float(cfg.encounter_max_speed_knots))).astype(int)
            seg["is_port_far"] = (seg["port_km_min"].to_numpy(dtype=float) >= float(cfg.encounter_min_port_km)).astype(int)
            seg["is_offshore"] = seg["is_port_far"]
            seg["encounter_rule_score"] = risk
            seg["risk_score"] = risk

            parts.append(seg[TRANS_OUTPUT_COLUMNS])
            summaries.append(
                {
                    "event_id": event_id,
                    "event_kind": "encounter",
                    "label": "Encounter" if positive else "Normal",
                    "class_id": int(1 if positive else 0),
                    "pair_id": str(pair_id),
                    "mmsi_a": str(seg["mmsi_a"].iloc[0]),
                    "mmsi_b": str(seg["mmsi_b"].iloc[0]),
                    "start_timestamp": int(seg["timestamp"].min()),
                    "end_timestamp": int(seg["timestamp"].max()),
                    "duration_minutes": float(duration),
                    "n_points": int(len(seg)),
                    "median_distance_km": med_dist,
                    "median_speed_knots": med_speed,
                    "close_rate": close_rate,
                    "slow_rate": slow_rate,
                    "risk_score": float(risk),
                    "rule_note": "distance<=0.5km duration>=2h speed<2kn port>=10km" if positive else "hard_negative_near_encounter",
                }
            )

    return _select_parts(
        parts,
        summaries,
        positive_class_id=1,
        max_positive=int(cfg.max_encounter_events_per_file),
        max_normal=int(cfg.max_normal_events_per_file),
        rng=rng,
    )


def _safe_offshore_value(value: float, minimum: float) -> float:
    try:
        v = float(value)
    except Exception:
        v = np.nan
    if not np.isfinite(v) or v < minimum:
        return float(minimum + 5.0)
    return float(v)


def _synthetic_partner_point(
    lat: float,
    lon: float,
    cfg: TransshipmentCfg,
    rng: np.random.RandomState,
) -> Tuple[float, float, float]:
    radius = float(rng.uniform(0.08, max(0.09, float(cfg.encounter_distance_km) * 0.85)))
    bearing = float(rng.uniform(0.0, 2.0 * np.pi))
    dlat = (radius * np.cos(bearing)) / 111.32
    cos_lat = max(0.15, abs(float(np.cos(np.deg2rad(lat)))))
    dlon = (radius * np.sin(bearing)) / (111.32 * cos_lat)
    return float(np.clip(lat + dlat, -89.999, 89.999)), float(_wrap_lon(np.array([lon + dlon]))[0]), radius


def _contiguous_track_segments(g: pd.DataFrame, grid_minutes: int, min_ticks: int) -> List[pd.DataFrame]:
    g = g.sort_values("timestamp").reset_index(drop=True)
    if len(g) < int(min_ticks):
        return []
    grid_sec = max(60, int(grid_minutes) * 60)
    max_step = int(round(grid_sec * 1.5))
    gaps = g["timestamp"].diff().fillna(0).to_numpy(dtype=float) > float(max_step)
    seg_ids = np.cumsum(gaps.astype(int))
    out = []
    for _, seg in g.groupby(seg_ids, sort=False):
        if len(seg) >= int(min_ticks):
            out.append(seg.reset_index(drop=True))
    return out


def _build_synthetic_encounter_events(
    reg: pd.DataFrame,
    cfg: TransshipmentCfg,
    rng: np.random.RandomState,
    run_stem: str,
) -> Tuple[List[pd.DataFrame], List[dict]]:
    n_events = int(max(0, cfg.synthetic_encounters_per_file))
    if n_events <= 0:
        return [], []

    grid_minutes = max(1, int(cfg.grid_minutes))
    ticks = max(2, int(np.ceil(float(cfg.encounter_min_hours) * 60.0 / float(grid_minutes))))
    vessel_groups: List[Tuple[int, pd.DataFrame]] = []
    for v, g in reg.groupby("mmsi", sort=False):
        for seg in _contiguous_track_segments(g, grid_minutes=grid_minutes, min_ticks=ticks):
            vessel_groups.append((int(v), seg))
    if len(vessel_groups) < 1:
        return [], []

    parts: List[pd.DataFrame] = []
    summaries: List[dict] = []
    min_port = float(cfg.encounter_min_port_km)
    event_no = 0

    for _ in range(n_events):
        vessel_a, ga = vessel_groups[int(rng.randint(0, len(vessel_groups)))]
        start_max = len(ga) - ticks
        if start_max < 0:
            continue
        start = int(rng.randint(0, start_max + 1)) if start_max > 0 else 0
        seg_a = ga.iloc[start:start + ticks].copy().reset_index(drop=True)
        if seg_a.empty:
            continue

        if len(vessel_groups) >= 2:
            pool = [(v, g) for v, g in vessel_groups if int(v) != int(vessel_a)]
            vessel_b, gb = pool[int(rng.randint(0, len(pool)))] if pool else vessel_groups[int(rng.randint(0, len(vessel_groups)))]
            if len(gb) >= ticks:
                start_b_max = len(gb) - ticks
                start_b = int(rng.randint(0, start_b_max + 1)) if start_b_max > 0 else 0
                seg_b = gb.iloc[start_b:start_b + ticks].copy().reset_index(drop=True)
            else:
                vessel_b = 900_000_000_000 + int(rng.randint(0, 99_999_999))
                seg_b = seg_a.copy()
        else:
            vessel_b = 900_000_000_000 + int(rng.randint(0, 99_999_999))
            seg_b = seg_a.copy()

        event_no += 1
        event_id = f"{run_stem}_synthetic_encounter_{event_no:06d}"
        rows: List[dict] = []
        distances = []
        speeds = []

        for k, (_, a) in enumerate(seg_a.iterrows()):
            b = seg_b.iloc[min(k, len(seg_b) - 1)] if len(seg_b) else a
            lat_b, lon_b, d_km = _synthetic_partner_point(float(a["lat"]), float(a["lon"]), cfg, rng)
            speed_a = float(min(float(a["speed"]), max(0.1, float(cfg.encounter_max_speed_knots) * 0.75)))
            speed_b = float(min(float(b["speed"]), max(0.1, float(cfg.encounter_max_speed_knots) * 0.75)))
            course_b = (float(a["course"]) + float(rng.uniform(-12.0, 12.0))) % 360.0
            port_a = _safe_offshore_value(float(a.get("port_km", np.nan)), min_port)
            port_b = _safe_offshore_value(float(b.get("port_km", np.nan)), min_port)
            shore_a = _safe_offshore_value(float(a.get("shore_km", np.nan)), 0.0)
            shore_b = _safe_offshore_value(float(b.get("shore_km", np.nan)), 0.0)
            rows.append(
                _base_feature_row(
                    timestamp=int(a["timestamp"]),
                    event_mode_id=1,
                    mmsi_a=int(vessel_a),
                    mmsi_b=int(vessel_b),
                    pair_id=f"{int(vessel_a)}__{int(vessel_b)}",
                    lat_a=float(a["lat"]),
                    lon_a=float(a["lon"]),
                    lat_b=lat_b,
                    lon_b=lon_b,
                    speed_a=speed_a,
                    speed_b=speed_b,
                    course_a=float(a["course"]),
                    course_b=course_b,
                    shore_a=shore_a,
                    shore_b=shore_b,
                    port_a=port_a,
                    port_b=port_b,
                    is_fishing_a=float(a.get("is_fishing", -1.0)),
                    is_fishing_b=float(b.get("is_fishing", -1.0)),
                    gear_a_id=int(a.get("gear_id", -1)),
                    gear_b_id=int(b.get("gear_id", -1)),
                    gear_a_label=str(a.get("gear_label", "unknown")),
                    gear_b_label=str(b.get("gear_label", "unknown")),
                    distance_between_km=float(d_km),
                )
            )
            distances.append(float(d_km))
            speeds.append(float((speed_a + speed_b) / 2.0))

        seg = pd.DataFrame(rows)
        duration = _duration_minutes(seg, grid_minutes)
        risk = _encounter_risk(seg, cfg, duration)
        seg["event_id"] = event_id
        seg["event_kind"] = "encounter"
        seg["label"] = "Encounter"
        seg["class_id"] = 1
        seg["is_transshipment"] = 1
        seg["duration_nearby_minutes"] = float(duration)
        seg["event_duration_minutes"] = float(duration)
        seg["both_slow"] = 1
        seg["is_port_far"] = 1
        seg["is_offshore"] = 1
        seg["encounter_rule_score"] = float(risk)
        seg["risk_score"] = float(risk)

        parts.append(seg[TRANS_OUTPUT_COLUMNS])
        summaries.append(
            {
                "event_id": event_id,
                "event_kind": "encounter",
                "label": "Encounter",
                "class_id": 1,
                "pair_id": str(seg["pair_id"].iloc[0]),
                "mmsi_a": str(int(vessel_a)),
                "mmsi_b": str(int(vessel_b)),
                "start_timestamp": int(seg["timestamp"].min()),
                "end_timestamp": int(seg["timestamp"].max()),
                "duration_minutes": float(duration),
                "n_points": int(len(seg)),
                "median_distance_km": float(np.nanmedian(distances)) if distances else 0.0,
                "median_speed_knots": float(np.nanmedian(speeds)) if speeds else 0.0,
                "close_rate": 1.0,
                "slow_rate": 1.0,
                "risk_score": float(risk),
                "rule_note": "synthetic_positive_encounter_for_balanced_training",
            }
        )

    return parts, summaries


def _track_distance_km(lat: np.ndarray, lon: np.ndarray) -> float:
    if len(lat) < 2:
        return 0.0
    d = haversine_km_np(lat[:-1], lon[:-1], lat[1:], lon[1:])
    return float(np.nansum(d))


def _spatial_range_km(lat: np.ndarray, lon: np.ndarray) -> float:
    if len(lat) < 2:
        return 0.0
    lat0 = np.full(len(lat), float(np.nanmean(lat)))
    lon0 = np.full(len(lon), float(np.nanmean(lon)))
    d = haversine_km_np(lat0, lon0, lat, lon)
    return float(np.nanmax(d) * 2.0)


def _loitering_risk(seg: pd.DataFrame, cfg: TransshipmentCfg, duration_min: float, compactness: float) -> float:
    speed_col = "speed_a" if "speed_a" in seg.columns else "speed"
    shore_col = "shore_km_min" if "shore_km_min" in seg.columns else "shore_km"
    med_speed = float(np.nanmedian(seg[speed_col]))
    shore_vals = seg[shore_col].to_numpy(dtype=float)
    valid_shore = shore_vals[shore_vals >= 0.0]
    shore_med = float(np.nanmedian(valid_shore)) if valid_shore.size else float(cfg.loitering_min_shore_nm) * NAUTICAL_MILE_KM
    shore_min_km = float(cfg.loitering_min_shore_nm) * NAUTICAL_MILE_KM

    dur = np.clip(duration_min / max(float(cfg.loitering_min_hours) * 60.0, 1.0), 0.0, 1.0)
    slow = 1.0 - np.clip(med_speed / max(float(cfg.loitering_candidate_speed_knots), 1e-6), 0.0, 1.0)
    offshore = np.clip(shore_med / max(shore_min_km, 1e-6), 0.0, 1.0)
    compact = 1.0 - np.clip(float(compactness), 0.0, 1.0)
    return float(np.clip((0.30 * dur) + (0.30 * slow) + (0.25 * offshore) + (0.15 * compact), 0.0, 1.0))


def _build_loitering_events(reg: pd.DataFrame, cfg: TransshipmentCfg, rng: np.random.RandomState, run_stem: str) -> Tuple[List[pd.DataFrame], List[dict]]:
    parts: List[pd.DataFrame] = []
    summaries: List[dict] = []
    grid_sec = max(60, int(cfg.grid_minutes) * 60)
    merge_gap_sec = max(grid_sec, int(cfg.loitering_merge_gap_minutes) * 60)
    shore_min_km = float(cfg.loitering_min_shore_nm) * NAUTICAL_MILE_KM
    candidate_speed = max(float(cfg.loitering_candidate_speed_knots), float(cfg.loitering_max_speed_knots))
    event_no = 0

    for vessel, vg in reg.groupby("mmsi", sort=False):
        vg = vg.sort_values("timestamp").reset_index(drop=True)
        shore = vg["shore_km"].to_numpy(dtype=float)
        offshore_candidate = (shore >= shore_min_km) | (~np.isfinite(shore)) | (shore < 0.0)
        candidate = (vg["speed"].to_numpy(dtype=float) <= candidate_speed) & offshore_candidate
        if not candidate.any():
            continue

        idx = np.where(candidate)[0]
        cand = vg.iloc[idx].copy().reset_index(drop=True)
        gaps = cand["timestamp"].diff().fillna(0).to_numpy(dtype=float) > float(merge_gap_sec)
        seg_ids = np.cumsum(gaps.astype(int))

        for _, raw_seg in cand.groupby(seg_ids, sort=False):
            raw_seg = raw_seg.sort_values("timestamp").reset_index(drop=True)
            duration = _duration_minutes(raw_seg, int(cfg.grid_minutes))
            if duration < float(cfg.normal_min_hours) * 60.0:
                continue

            lat = raw_seg["lat"].to_numpy(dtype=float)
            lon = raw_seg["lon"].to_numpy(dtype=float)
            track_km = _track_distance_km(lat, lon)
            start_end = float(haversine_km_np(np.array([lat[0]]), np.array([lon[0]]), np.array([lat[-1]]), np.array([lon[-1]]))[0]) if len(lat) >= 2 else 0.0
            compactness = float(start_end / max(track_km, 1e-6))
            spatial_range = _spatial_range_km(lat, lon)

            course = raw_seg["course"].to_numpy(dtype=float)
            if len(course) >= 2:
                dc = _angle_diff_deg(course[1:], course[:-1])
                turn_rate = float(np.nanmean(dc) / max(float(cfg.grid_minutes), 1.0))
            else:
                turn_rate = 0.0

            med_speed = float(np.nanmedian(raw_seg["speed"]))
            shore_vals = raw_seg["shore_km"].to_numpy(dtype=float)
            valid_shore = shore_vals[shore_vals >= 0.0]
            shore_ok = bool(valid_shore.size == 0 or np.nanmedian(valid_shore) >= shore_min_km)
            positive = (
                duration >= float(cfg.loitering_min_hours) * 60.0
                and med_speed < float(cfg.loitering_max_speed_knots)
                and shore_ok
            )

            event_no += 1
            event_id = f"{run_stem}_loitering_{event_no:06d}"
            risk = _loitering_risk(raw_seg, cfg, duration, compactness)

            rows: List[dict] = []
            for _, r in raw_seg.iterrows():
                rows.append(
                    _base_feature_row(
                        timestamp=int(r["timestamp"]),
                        event_mode_id=2,
                        mmsi_a=int(vessel),
                        mmsi_b=None,
                        pair_id=f"{int(vessel)}__loitering",
                        lat_a=float(r["lat"]),
                        lon_a=float(r["lon"]),
                        lat_b=np.nan,
                        lon_b=np.nan,
                        speed_a=float(r["speed"]),
                        speed_b=0.0,
                        course_a=float(r["course"]),
                        course_b=float(r["course"]),
                        shore_a=float(r["shore_km"]),
                        shore_b=np.nan,
                        port_a=float(r["port_km"]),
                        port_b=np.nan,
                        is_fishing_a=float(r["is_fishing"]),
                        is_fishing_b=-1.0,
                        gear_a_id=int(r["gear_id"]),
                        gear_b_id=-1,
                        gear_a_label=str(r["gear_label"]),
                        gear_b_label="none",
                        distance_between_km=0.0,
                    )
                )

            seg = pd.DataFrame(rows)
            seg["event_id"] = event_id
            seg["event_kind"] = "loitering"
            seg["label"] = "Loitering" if positive else "Normal"
            seg["class_id"] = 2 if positive else 0
            seg["is_transshipment"] = 1 if positive else 0
            seg["event_duration_minutes"] = float(duration)
            seg["duration_nearby_minutes"] = 0.0
            seg["loitering_duration_minutes"] = float(duration)
            seg["loitering_spatial_range_km"] = float(spatial_range)
            seg["loitering_start_end_km"] = float(start_end)
            seg["loitering_compactness"] = float(np.clip(compactness, 0.0, 1.0))
            seg["loitering_turn_rate_abs"] = float(turn_rate)
            seg["is_offshore"] = (seg["shore_km_min"].to_numpy(dtype=float) >= shore_min_km).astype(int)
            seg["loitering_rule_score"] = float(risk)
            seg["risk_score"] = float(risk)

            parts.append(seg[TRANS_OUTPUT_COLUMNS])
            summaries.append(
                {
                    "event_id": event_id,
                    "event_kind": "loitering",
                    "label": "Loitering" if positive else "Normal",
                    "class_id": int(2 if positive else 0),
                    "pair_id": f"{int(vessel)}__loitering",
                    "mmsi_a": str(int(vessel)),
                    "mmsi_b": "",
                    "start_timestamp": int(seg["timestamp"].min()),
                    "end_timestamp": int(seg["timestamp"].max()),
                    "duration_minutes": float(duration),
                    "n_points": int(len(seg)),
                    "median_distance_km": 0.0,
                    "median_speed_knots": med_speed,
                    "spatial_range_km": float(spatial_range),
                    "compactness": float(np.clip(compactness, 0.0, 1.0)),
                    "risk_score": float(risk),
                    "rule_note": "speed<2kn duration>=8h shore>=20nm" if positive else "hard_negative_loitering_like",
                }
            )

    return _select_parts(
        parts,
        summaries,
        positive_class_id=2,
        max_positive=int(cfg.max_loitering_events_per_file),
        max_normal=int(cfg.max_normal_events_per_file),
        rng=rng,
    )


def generate_transshipment_dataset(input_path: Path, out_dir: Path, cfg: TransshipmentCfg) -> List[Path]:
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    events_dir = out_dir / "events"
    summaries_dir = out_dir / "summaries"
    events_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    mode = str(cfg.mode).strip().lower()
    if mode not in {"both", "encounter", "loitering"}:
        raise ValueError("--mode must be one of: both, encounter, loitering")

    rng = np.random.RandomState(int(cfg.seed))
    df = _read_input_path(input_path, cfg)
    reg = _regularize_all(df, cfg)

    run_stem = "all" if input_path.is_dir() else input_path.stem
    parts: List[pd.DataFrame] = []
    summaries: List[dict] = []

    if mode in {"both", "encounter"}:
        p, s = _build_encounter_events(reg, cfg, rng, run_stem)
        parts.extend(p)
        summaries.extend(s)
        p, s = _build_synthetic_encounter_events(reg, cfg, rng, run_stem)
        parts.extend(p)
        summaries.extend(s)

    if mode in {"both", "loitering"}:
        p, s = _build_loitering_events(reg, cfg, rng, run_stem)
        parts.extend(p)
        summaries.extend(s)

    if parts:
        merged = pd.concat(parts, ignore_index=True, sort=False)
        merged = merged.sort_values(["event_kind", "event_id", "timestamp"]).reset_index(drop=True)
    else:
        merged = pd.DataFrame(columns=TRANS_OUTPUT_COLUMNS)

    event_summary = pd.DataFrame(summaries)
    if event_summary.empty:
        event_summary = pd.DataFrame(
            columns=[
                "event_id",
                "event_kind",
                "label",
                "class_id",
                "pair_id",
                "mmsi_a",
                "mmsi_b",
                "start_timestamp",
                "end_timestamp",
                "duration_minutes",
                "n_points",
                "median_distance_km",
                "median_speed_knots",
                "risk_score",
                "rule_note",
            ]
        )

    out_name = "transshipment_all.csv" if (input_path.is_dir() or cfg.combine_outputs) else f"transshipment_{input_path.stem}.csv"
    out_path = out_dir / out_name
    events_path = events_dir / f"events_{out_name}"
    summary_path = summaries_dir / "summary_transshipment.csv"

    merged.to_csv(out_path, index=False)
    event_summary.to_csv(events_path, index=False)

    summary = (
        merged.groupby(["event_kind", "label", "class_id"], dropna=False)
        .agg(rows=("event_id", "size"), events=("event_id", "nunique"))
        .reset_index()
        .sort_values(["event_kind", "class_id", "label"])
        if not merged.empty
        else pd.DataFrame(columns=["event_kind", "label", "class_id", "rows", "events"])
    )
    summary.to_csv(summary_path, index=False)

    print(f"[transshipment] Saved dataset: {out_path}")
    print(f"[transshipment] Saved events: {events_path}")
    print(f"[transshipment] Saved summary: {summary_path}")
    if not summary.empty:
        print(summary.to_string(index=False))
    else:
        print("[transshipment] No candidate events were generated.")

    return [out_path]
