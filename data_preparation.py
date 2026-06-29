from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from dataload import read_ais_csv, infer_label_from_filename


DEFAULT_SOURCE_EXCLUDE_LABELS = ("pole_and_line", "trollers")
DEFAULT_GEAR_EXCLUDE_LABELS = ("unknown", *DEFAULT_SOURCE_EXCLUDE_LABELS)
DEFAULT_TRANSSHIPMENT_SOURCE_INCLUDE_LABELS = (
    "drifting_longlines",
    "fixed_gear",
    "purse_seines",
    "trawlers",
)
DEFAULT_SPOOFING_SOURCE_INCLUDE_LABELS = (
    "drifting_longlines",
    "fixed_gear",
    "purse_seines",
    "trawlers",
)
TRANSSHIPMENT_GEAR_TO_ID = {
    label: idx for idx, label in enumerate(DEFAULT_TRANSSHIPMENT_SOURCE_INCLUDE_LABELS)
}


SEQ_FEATURE_COLS = [
    "speed",
    "vx",
    "vy",
    "dspeed",
    "accel",
    "dcourse",
    "turn_rate",
    "abs_dcourse",
    "step_km",
    "step_km_raw",
    "dt",
    "dt_raw_seconds",
    "dt_log",
    "implied_speed_knots_raw",
    "distance_from_shore",
    "distance_from_port",
    "pos_speed_knots",
    "dpos_speed",
    "pos_bearing_sin",
    "pos_bearing_cos",
    "bearing_error",
    "curvature",
    "pos_speed_ma5",
    "pos_speed_std5",
    "abs_turn_ma5",
    "curvature_ma5",
]

# Context features are computed only from fields observable by a detector:
# claimed identity, report time, and reported position.  Attack labels,
# original_mmsi, scenario_id, and simulator magnitude are deliberately excluded
# from X; those fields are allowed only as split/balancing/audit metadata.
SPOOFING_CONTEXT_FEATURE_COLS = [
    "claimed_identity_registered",
    "claimed_history_age_log_hours",
    "claimed_prev_dt_log_hours",
    "claimed_prev_distance_log_km",
    "claimed_prev_implied_speed_log_knots",
    "claimed_concurrent_reports_log1p",
    "claimed_concurrent_spread_log_km",
    "claimed_revisit_lag_log_hours",
    "claimed_revisit_score",
]

LOCATION_FEATURE_COLS = [
    "distance_from_shore",
    "distance_from_port",
]


@dataclass
class PreprocessCfg:
    task: str = "gear"

    # windowing
    seq_len: int = 120
    stride: int = 6

    # segment split
    gap_seconds: int = 10800  # 3 jam

    # jump filter
    max_implied_knots: float = 42.0
    apply_jump_filter: bool = True

    # sanity
    max_speed_knots: float = 50.0

    # dataset balancing
    max_windows_per_vessel: int = 1200
    min_windows_per_vessel: int = 0
    min_points_per_vessel: int = 80
    max_windows_per_file: int = 20000
    balance_gear_classes: bool = False

    # optional filter
    use_operational_filter: bool = False
    op_speed_min: float = 1.0
    op_speed_max: float = 12.0
    use_location_features: bool = True

    # khusus task=spoofing
    spoofing_window_threshold: float = 0.20

    # khusus task=godark: rule-based candidate filter sebelum BiLSTM.
    godark_min_distance_from_shore_nm: float = 50.0
    godark_ping_window_seconds: int = 12 * 3600
    godark_min_ping_count_prev_window: int = 14

    # khusus task=transshipment
    transshipment_target: str = "multiclass"
    transshipment_feature_mode: str = "fair"


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

TRANS_RULE_SCORE_COLS = ["encounter_rule_score", "loitering_rule_score", "risk_score"]
TRANS_MODEL_FEATURE_COLS_FAIR = [c for c in TRANS_FEATURE_COLS if c not in TRANS_RULE_SCORE_COLS]


def _transshipment_feature_cols(mode: str) -> List[str]:
    mode = str(mode or "fair").strip().lower()
    if mode in {"full", "with_rule", "with_rules"}:
        return list(TRANS_FEATURE_COLS)
    return list(TRANS_MODEL_FEATURE_COLS_FAIR)


def _normalize_exclude_labels(task: str, exclude_labels: Optional[List[str]]) -> List[str]:
    labels = [str(label) for label in (exclude_labels or []) if str(label).strip()]
    if str(task) != "gear":
        return labels

    seen = set(labels)
    for label in DEFAULT_GEAR_EXCLUDE_LABELS:
        if label not in seen:
            labels.append(label)
            seen.add(label)
    return labels


def _sequence_feature_cols(cfg: PreprocessCfg) -> List[str]:
    cols = (
        list(SEQ_FEATURE_COLS)
        if cfg.use_location_features
        else [c for c in SEQ_FEATURE_COLS if c not in LOCATION_FEATURE_COLS]
    )
    if str(cfg.task) == "spoofing":
        cols.extend(SPOOFING_CONTEXT_FEATURE_COLS)
    return cols


def _select_spoofing_cap_indices(
    y: np.ndarray,
    kinds: np.ndarray,
    max_count: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    n = int(len(y))
    if max_count <= 0 or n <= max_count:
        return np.arange(n, dtype=np.int64)

    y = np.asarray(y, dtype=np.int64)
    kinds = np.asarray(kinds).astype(str)
    positive_idx = np.where(y == 1)[0]
    normal_idx = np.where(y == 0)[0]

    # Keep attack diversity first. Positive spoofing scenarios are scarce and
    # must not disappear through a global random cap dominated by normal data.
    positive_budget = min(
        int(positive_idx.size),
        max(1, int(round(max_count * 0.50))),
    )
    if positive_idx.size <= positive_budget:
        normal_budget = max_count - int(positive_idx.size)
        if normal_idx.size > normal_budget:
            normal_idx = rng.choice(
                normal_idx,
                size=normal_budget,
                replace=False,
            )
        keep = np.concatenate(
            [positive_idx.astype(np.int64), normal_idx.astype(np.int64)]
        )
        rng.shuffle(keep)
        return keep

    selected_positive: List[np.ndarray] = []
    attack_names = sorted(np.unique(kinds[positive_idx]).tolist())
    remaining_budget = positive_budget
    for pos, attack in enumerate(attack_names):
        idx = positive_idx[kinds[positive_idx] == attack]
        attacks_left = len(attack_names) - pos
        take = min(
            int(idx.size),
            max(1, remaining_budget // max(attacks_left, 1)),
        )
        if idx.size > take:
            idx = rng.choice(idx, size=take, replace=False)
        selected_positive.append(np.asarray(idx, dtype=np.int64))
        remaining_budget -= int(len(idx))

    pos_keep = (
        np.concatenate(selected_positive)
        if selected_positive
        else np.zeros((0,), dtype=np.int64)
    )
    normal_budget = max_count - int(pos_keep.size)
    if normal_idx.size > normal_budget:
        normal_idx = rng.choice(
            normal_idx,
            size=normal_budget,
            replace=False,
        )
    keep = np.concatenate([pos_keep, np.asarray(normal_idx, dtype=np.int64)])
    rng.shuffle(keep)
    return keep


def timestamp_to_epoch_seconds(values: pd.Series) -> pd.Series:
    ts_num = pd.to_numeric(values, errors="coerce")
    if ts_num.isna().any():
        ts_dt = pd.to_datetime(values, errors="coerce", utc=True)
        if ts_dt.isna().any():
            ts_dt_mixed = pd.to_datetime(values, errors="coerce", utc=True, format="mixed")
            ts_dt = ts_dt.fillna(ts_dt_mixed)
        ts_ns = ts_dt.to_numpy(dtype="datetime64[ns]").astype("int64")
        ts_iso = pd.Series(ts_ns // 10**9, index=values.index).where(ts_dt.notna(), np.nan)
        ts_num = ts_num.fillna(ts_iso)
    return ts_num


def haversine_km_np(lat1, lon1, lat2, lon2) -> np.ndarray:
    r = 6371.0088
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    return r * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def bearing_deg_np(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Bearing dari titik 1 ke titik 2 dalam derajat 0 sampai 360."""
    lat1r = np.deg2rad(lat1)
    lat2r = np.deg2rad(lat2)
    dlon = np.deg2rad(lon2 - lon1)

    y = np.sin(dlon) * np.cos(lat2r)
    x = (
        np.cos(lat1r) * np.sin(lat2r)
        - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    )

    brng = np.rad2deg(np.arctan2(y, x))
    brng = (brng + 360.0) % 360.0
    return brng


def _add_spoofing_observable_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add causal identity/history context without using simulator labels.

    The feature contract mirrors information available to an operational
    detector that maintains a registry and a time-ordered AIS message cache.
    No attack_type, label, original_mmsi, scenario_id, or injected magnitude is
    consulted here.
    """
    if df.empty:
        return df

    out = df.copy()
    stale_context = [
        column
        for column in SPOOFING_CONTEXT_FEATURE_COLS
        if column != "claimed_identity_registered" and column in out.columns
    ]
    if stale_context:
        out = out.drop(columns=stale_context)
    if "claimed_mmsi" in out.columns:
        claimed = out["claimed_mmsi"].astype(str)
    else:
        claimed = out["mmsi"].astype(str)
    out["_claimed_identity"] = claimed

    if "claimed_identity_registered" in out.columns:
        registered = pd.to_numeric(
            out["claimed_identity_registered"], errors="coerce"
        ).fillna(0.0)
    else:
        # Backward-compatible fallback: identities that also appear as a
        # native report identity are considered present in the local registry.
        native = set(out["mmsi"].astype(str).tolist())
        registered = claimed.isin(native).astype(float)
    out["claimed_identity_registered"] = registered.clip(0.0, 1.0).astype("float32")

    keys = ["_claimed_identity", "timestamp"]
    simultaneous_count = out.groupby(keys, sort=False)["mmsi"].transform("size")
    median_lat = out.groupby(keys, sort=False)["lat"].transform("median")
    median_lon = out.groupby(keys, sort=False)["lon"].transform("median")
    distance_to_center = haversine_km_np(
        out["lat"].to_numpy(dtype=float),
        out["lon"].to_numpy(dtype=float),
        median_lat.to_numpy(dtype=float),
        median_lon.to_numpy(dtype=float),
    )
    out["_simultaneous_distance_km"] = distance_to_center
    simultaneous_spread = out.groupby(keys, sort=False)[
        "_simultaneous_distance_km"
    ].transform("max")

    centroids = (
        out.groupby(keys, sort=False, as_index=False)
        .agg(_ctx_lat=("lat", "median"), _ctx_lon=("lon", "median"))
        .sort_values(["_claimed_identity", "timestamp"])
    )
    by_claim = centroids.groupby("_claimed_identity", sort=False)
    centroids["_prev_timestamp"] = by_claim["timestamp"].shift(1)
    centroids["_prev_lat"] = by_claim["_ctx_lat"].shift(1)
    centroids["_prev_lon"] = by_claim["_ctx_lon"].shift(1)
    centroids["_first_timestamp"] = by_claim["timestamp"].transform("min")

    prev_valid = centroids["_prev_timestamp"].notna()
    prev_distance = np.zeros(len(centroids), dtype=np.float64)
    if bool(prev_valid.any()):
        prev_distance[prev_valid.to_numpy()] = haversine_km_np(
            centroids.loc[prev_valid, "_prev_lat"].to_numpy(dtype=float),
            centroids.loc[prev_valid, "_prev_lon"].to_numpy(dtype=float),
            centroids.loc[prev_valid, "_ctx_lat"].to_numpy(dtype=float),
            centroids.loc[prev_valid, "_ctx_lon"].to_numpy(dtype=float),
        )
    prev_dt = (
        centroids["timestamp"] - centroids["_prev_timestamp"]
    ).fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    prev_speed_knots = np.divide(
        prev_distance * (3600.0 / 1.852),
        np.maximum(prev_dt, 1.0),
    )

    # Approximate causal revisit memory using ~1 km spatial cells. Updates are
    # committed per timestamp so simultaneous reports cannot masquerade as
    # historical revisits.
    revisit_lag = np.zeros(len(centroids), dtype=np.float64)
    for _, idx in centroids.groupby("_claimed_identity", sort=False).groups.items():
        last_seen: Dict[Tuple[int, int], float] = {}
        local = centroids.loc[list(idx)].sort_values("timestamp")
        for timestamp, rows in local.groupby("timestamp", sort=True):
            pending: List[Tuple[int, int]] = []
            for row_idx, row in rows.iterrows():
                cell = (
                    int(np.round(float(row["_ctx_lat"]) * 100.0)),
                    int(np.round(float(row["_ctx_lon"]) * 100.0)),
                )
                if cell in last_seen:
                    revisit_lag[int(row_idx)] = max(
                        0.0, float(timestamp) - float(last_seen[cell])
                    )
                pending.append(cell)
            for cell in pending:
                last_seen[cell] = float(timestamp)

    centroids["claimed_history_age_log_hours"] = np.log1p(
        ((centroids["timestamp"] - centroids["_first_timestamp"]) / 3600.0)
        .clip(lower=0.0)
    )
    centroids["claimed_prev_dt_log_hours"] = np.log1p(prev_dt / 3600.0)
    centroids["claimed_prev_distance_log_km"] = np.log1p(
        np.clip(prev_distance, 0.0, 20000.0)
    )
    centroids["claimed_prev_implied_speed_log_knots"] = np.log1p(
        np.clip(prev_speed_knots, 0.0, 10000.0)
    )
    centroids["claimed_revisit_lag_log_hours"] = np.log1p(revisit_lag / 3600.0)
    centroids["claimed_revisit_score"] = (
        revisit_lag >= 3600.0
    ).astype(np.float32)

    context_cols = [
        "claimed_history_age_log_hours",
        "claimed_prev_dt_log_hours",
        "claimed_prev_distance_log_km",
        "claimed_prev_implied_speed_log_knots",
        "claimed_revisit_lag_log_hours",
        "claimed_revisit_score",
    ]
    out = out.merge(
        centroids[keys + context_cols],
        on=keys,
        how="left",
        validate="many_to_one",
    )
    out["claimed_concurrent_reports_log1p"] = np.log1p(
        simultaneous_count.to_numpy(dtype=float).clip(min=1.0) - 1.0
    ).astype(np.float32)
    out["claimed_concurrent_spread_log_km"] = np.log1p(
        np.clip(simultaneous_spread.to_numpy(dtype=float), 0.0, 20000.0)
    ).astype(np.float32)
    for column in SPOOFING_CONTEXT_FEATURE_COLS:
        out[column] = (
            pd.to_numeric(out[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype("float32")
        )
    return out.drop(columns=["_claimed_identity", "_simultaneous_distance_km"])


def clean_and_derive(df: pd.DataFrame, cfg: PreprocessCfg) -> pd.DataFrame:
    need = ["mmsi", "timestamp", "lat", "lon"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df = df.dropna(subset=["mmsi", "timestamp", "lat", "lon"]).copy()

    df["timestamp"] = timestamp_to_epoch_seconds(df["timestamp"])
    df = df.dropna(subset=["timestamp"]).copy()
    df["timestamp"] = df["timestamp"].astype("int64")

    df["mmsi"] = pd.to_numeric(df["mmsi"], errors="coerce")
    df = df.dropna(subset=["mmsi"]).copy()
    df["mmsi"] = df["mmsi"].astype("int64").astype(str)

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"]).copy()
    df = df[(df["lat"].between(-90, 90)) & (df["lon"].between(-180, 180))].copy()

    if str(cfg.task) == "spoofing":
        df = _add_spoofing_observable_context(df)

    if "speed" in df.columns:
        df["speed"] = pd.to_numeric(df["speed"], errors="coerce")
    else:
        df["speed"] = np.nan

    if "course" in df.columns:
        df["course"] = pd.to_numeric(df["course"], errors="coerce").fillna(0.0)
    else:
        df["course"] = 0.0

    if "distance_from_shore" not in df.columns:
        df["distance_from_shore"] = np.nan
    if "distance_from_port" not in df.columns:
        df["distance_from_port"] = np.nan

    df["distance_from_shore"] = pd.to_numeric(
        df["distance_from_shore"], errors="coerce"
    ).fillna(-1.0)
    df["distance_from_port"] = pd.to_numeric(
        df["distance_from_port"], errors="coerce"
    ).fillna(-1.0)

    df["speed"] = pd.to_numeric(df["speed"], errors="coerce").fillna(0.0)
    df = df[(df["speed"] >= 0) & (df["speed"] <= cfg.max_speed_knots)].copy()

    if cfg.use_operational_filter:
        df = df[
            (df["speed"] >= cfg.op_speed_min)
            & (df["speed"] <= cfg.op_speed_max)
        ].copy()

    df = df.sort_values(["mmsi", "timestamp"])
    df = df.drop_duplicates(subset=["mmsi", "timestamp"], keep="last")

    # Derivative and rolling features must not cross a trajectory gap. Go-dark
    # intentionally keeps the gap inside a sequence because the gap is the
    # signal being detected.
    if cfg.task == "godark":
        df["_motion_segment"] = 0
        motion_group_cols = ["mmsi"]
    else:
        gap_break = (
            df.groupby("mmsi", sort=False)["timestamp"]
            .diff()
            .gt(float(cfg.gap_seconds))
        )
        df["_motion_segment"] = (
            gap_break.groupby(df["mmsi"], sort=False)
            .cumsum()
            .fillna(0)
            .astype("int64")
        )
        motion_group_cols = ["mmsi", "_motion_segment"]

    motion_groups = df.groupby(motion_group_cols, sort=False)
    rolling_levels = list(range(len(motion_group_cols)))

    # dt per step dalam detik.
    # Untuk task go-dark, gap panjang adalah sinyal penting.
    # Jadi dt tidak boleh dipotong ke default 3 jam.
    raw_dt = motion_groups["timestamp"].diff().fillna(1.0).astype("float32")

    df["dt_raw_seconds"] = raw_dt.clip(
        lower=1.0,
        upper=float(30 * 24 * 3600),
    ).astype("float32")

    dt_clip_upper = float(cfg.gap_seconds)
    if cfg.task == "godark":
        dt_clip_upper = max(dt_clip_upper, float(7 * 24 * 3600))

    df["dt"] = raw_dt.clip(
        lower=1.0,
        upper=dt_clip_upper,
    ).astype("float32")

    # course encode
    course_deg = df["course"].astype("float32") % 360.0
    cr = np.deg2rad(course_deg)
    df["course_sin"] = np.sin(cr).astype("float32")
    df["course_cos"] = np.cos(cr).astype("float32")

    # perubahan speed
    df["dspeed"] = motion_groups["speed"].diff().fillna(0).astype("float32")

    # perubahan course dengan shortest angle
    prev_course = motion_groups["course"].shift(1).astype("float32")
    dc = (course_deg - prev_course) % 360.0
    dc = ((dc + 180.0) % 360.0) - 180.0
    df["dcourse"] = dc.fillna(0).astype("float32")
    df["abs_dcourse"] = np.abs(df["dcourse"]).astype("float32")

    # jarak antar titik
    prev_lat = motion_groups["lat"].shift(1)
    prev_lon = motion_groups["lon"].shift(1)
    mask = prev_lat.notna() & prev_lon.notna()

    step_km = np.zeros(len(df), dtype=np.float32)
    if mask.any():
        step_km[mask.to_numpy()] = haversine_km_np(
            prev_lat[mask].to_numpy(),
            prev_lon[mask].to_numpy(),
            df.loc[mask, "lat"].to_numpy(),
            df.loc[mask, "lon"].to_numpy(),
        ).astype(np.float32)

    # step_km versi aman untuk fitur umum
    # step_km_raw versi lebih besar untuk menangkap pola reappearance go-dark
    df["step_km_raw"] = np.clip(step_km, 0.0, 500.0).astype(np.float32)
    df["step_km"] = np.clip(step_km, 0.0, 25.0).astype(np.float32)

    df["dt_log"] = np.log1p(df["dt"].astype("float32")).astype("float32")

    implied_raw = (df["step_km_raw"] / df["dt"].clip(lower=1.0)) * (3600.0 / 1.852)
    df["implied_speed_knots_raw"] = implied_raw.clip(0.0, 500.0).astype("float32")

    # velocity components dari AIS speed dan course
    spd = df["speed"].astype("float32")
    df["vx"] = (spd * df["course_cos"]).astype("float32")
    df["vy"] = (spd * df["course_sin"]).astype("float32")

    # akselerasi dan turn rate per menit
    dt = df["dt"].astype("float32")
    df["accel"] = (df["dspeed"] / dt * 60.0).astype("float32")
    df["turn_rate"] = (df["dcourse"] / dt * 60.0).astype("float32")
    df["accel"] = df["accel"].clip(-30.0, 30.0)
    df["turn_rate"] = df["turn_rate"].clip(-180.0, 180.0)

    # speed dari perubahan posisi
    pos_speed = (df["step_km"] / df["dt"].clip(lower=1.0)) * (3600.0 / 1.852)
    df["pos_speed_knots"] = pos_speed.clip(0.0, cfg.max_speed_knots).astype("float32")
    df["dpos_speed"] = (
        motion_groups["pos_speed_knots"].diff().fillna(0).astype("float32")
    )

    # bearing dari perubahan posisi
    pos_bearing = np.zeros(len(df), dtype=np.float32)
    if mask.any():
        pos_bearing[mask.to_numpy()] = bearing_deg_np(
            prev_lat[mask].to_numpy(),
            prev_lon[mask].to_numpy(),
            df.loc[mask, "lat"].to_numpy(),
            df.loc[mask, "lon"].to_numpy(),
        ).astype(np.float32)

    pos_bearing = (pos_bearing % 360.0).astype(np.float32)
    br = np.deg2rad(pos_bearing.astype(np.float32))
    df["pos_bearing_sin"] = np.sin(br).astype("float32")
    df["pos_bearing_cos"] = np.cos(br).astype("float32")
    df.loc[~mask, ["pos_bearing_sin", "pos_bearing_cos"]] = 0.0

    berr = np.zeros(len(df), dtype=np.float32)
    if mask.any():
        berr_valid = (
            course_deg.loc[mask].to_numpy(dtype=np.float32)
            - pos_bearing[mask.to_numpy()]
        ) % 360.0
        berr[mask.to_numpy()] = ((berr_valid + 180.0) % 360.0) - 180.0
    df["bearing_error"] = berr.astype("float32")

    curv = df["abs_dcourse"] / (df["step_km"] + 1e-3)
    df["curvature"] = curv.clip(0.0, 500.0).astype("float32")

    # rolling stats
    g = df.groupby(motion_group_cols, sort=False)

    df["pos_speed_ma5"] = (
        g["pos_speed_knots"]
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=rolling_levels, drop=True)
        .astype("float32")
    )

    df["pos_speed_std5"] = (
        g["pos_speed_knots"]
        .rolling(5, min_periods=1)
        .std()
        .reset_index(level=rolling_levels, drop=True)
        .fillna(0.0)
        .astype("float32")
    )

    abs_turn = df["turn_rate"].abs().astype("float32")
    df["abs_turn_ma5"] = (
        abs_turn.groupby(
            [df[c] for c in motion_group_cols],
            sort=False,
        )
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=rolling_levels, drop=True)
        .astype("float32")
    )

    df["curvature_ma5"] = (
        g["curvature"]
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=rolling_levels, drop=True)
        .astype("float32")
    )

    df["pos_speed_std5"] = df["pos_speed_std5"].clip(0.0, 20.0)
    df["abs_turn_ma5"] = df["abs_turn_ma5"].clip(0.0, 180.0)
    df["curvature_ma5"] = df["curvature_ma5"].clip(0.0, 500.0)

    return df


def filter_jumps(df: pd.DataFrame, cfg: PreprocessCfg) -> pd.DataFrame:
    if len(df) < 2:
        return df

    m = df["mmsi"].to_numpy()
    ts = df["timestamp"].to_numpy()
    lat = df["lat"].to_numpy()
    lon = df["lon"].to_numpy()

    keep = np.ones(len(df), dtype=bool)
    start = 0

    for i in range(1, len(df) + 1):
        if i == len(df) or m[i] != m[start]:
            if i - start >= 2:
                idx = np.arange(start, i)

                dt = ts[idx[1:]] - ts[idx[:-1]]
                valid = (dt > 0) & (dt <= cfg.gap_seconds)

                if valid.any():
                    lat1 = lat[idx[:-1]][valid]
                    lon1 = lon[idx[:-1]][valid]
                    lat2 = lat[idx[1:]][valid]
                    lon2 = lon[idx[1:]][valid]

                    d_km = haversine_km_np(lat1, lon1, lat2, lon2)
                    implied_knots = (d_km / dt[valid]) * 3600.0 / 1.852

                    bad = implied_knots > cfg.max_implied_knots
                    if bad.any():
                        bad_pos = np.where(valid)[0][bad]
                        keep[idx[bad_pos + 1]] = False

            start = i

    return df[keep].copy()


def _transshipment_class_id(row: pd.Series) -> int:
    if "class_id" in row.index:
        v = pd.to_numeric(pd.Series([row.get("class_id")]), errors="coerce").fillna(0).iloc[0]
        return int(np.clip(int(v), 0, 2))

    label = str(row.get("label", "")).strip().lower()
    kind = str(row.get("event_kind", "")).strip().lower()
    is_tx = int(pd.to_numeric(pd.Series([row.get("is_transshipment", 0)]), errors="coerce").fillna(0).iloc[0])
    if not is_tx:
        return 0
    if "loiter" in label or "loiter" in kind:
        return 2
    if "encounter" in label or "encounter" in kind:
        return 1
    return 1


def _transshipment_target_from_cfg(df: pd.DataFrame, cfg: PreprocessCfg) -> str:
    target = str(cfg.transshipment_target or "multiclass").strip().lower()
    if target == "auto":
        cls = set(pd.to_numeric(df["class_id"], errors="coerce").fillna(0).astype(int).unique().tolist())
        positives = sorted(c for c in cls if c > 0)
        if positives == [1]:
            return "encounter"
        if positives == [2]:
            return "loitering"
        if positives:
            return "multiclass"
        return "any"
    return target


def _map_transshipment_y(class_ids: np.ndarray, target: str) -> np.ndarray:
    y = np.asarray(class_ids, dtype=np.int64)
    if target == "encounter":
        return (y == 1).astype(np.int64)
    if target == "loitering":
        return (y == 2).astype(np.int64)
    if target == "any":
        return (y > 0).astype(np.int64)
    return y.astype(np.int64)


def _transshipment_label_map_for_target(target: str) -> Dict[int, str]:
    if target == "encounter":
        return {0: "normal", 1: "encounter"}
    if target == "loitering":
        return {0: "normal", 1: "loitering"}
    if target == "any":
        return {0: "normal", 1: "potential_transshipment"}
    return {0: "normal", 1: "encounter", 2: "loitering"}


def _clip_log1p_series(s: pd.Series, upper: float) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return np.log1p(x.clip(lower=0.0, upper=float(upper)))


def _stabilize_transshipment_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["duration_nearby_minutes", "event_duration_minutes", "loitering_duration_minutes"]:
        if col in df.columns:
            # 7 hari sudah lebih dari cukup sebagai sinyal durasi event, sisanya jadi outlier numerik.
            df[col] = _clip_log1p_series(df[col], upper=7 * 24 * 60)

    for col in ["shore_km_min", "port_km_min"]:
        if col in df.columns:
            df[col] = _clip_log1p_series(df[col], upper=500.0)

    for col in ["loitering_spatial_range_km", "loitering_start_end_km"]:
        if col in df.columns:
            df[col] = _clip_log1p_series(df[col], upper=200.0)

    for col in ["distance_between_km"]:
        if col in df.columns:
            df[col] = _clip_log1p_series(df[col], upper=20.0)

    for col in ["speed_a", "speed_b", "speed_pair_mean", "relative_speed_knots"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).clip(lower=0.0, upper=50.0)

    return df


def _target_from_present_transshipment_classes(classes: set[int]) -> str:
    positives = sorted(c for c in classes if c > 0)
    if positives == [1]:
        return "encounter"
    if positives == [2]:
        return "loitering"
    if positives:
        return "multiclass"
    return "any"


def _infer_transshipment_auto_target(csvs: List[Path], limit_rows: int = 0, chunksize: int = 0) -> str:
    classes: set[int] = set()
    for p in csvs:
        df = read_ais_csv(p, limit_rows=limit_rows, chunksize=chunksize)
        if df.empty:
            continue
        if "class_id" in df.columns:
            cls = (
                pd.to_numeric(df["class_id"], errors="coerce")
                .fillna(0)
                .astype(int)
                .clip(lower=0, upper=2)
            )
        else:
            cls = df.apply(_transshipment_class_id, axis=1).astype(int)
        classes.update(int(x) for x in cls.unique().tolist())
    return _target_from_present_transshipment_classes(classes)


def _stable_random_state(key: str, seed: int = 42) -> np.random.RandomState:
    h = 2166136261
    for ch in str(key):
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return np.random.RandomState((h + int(seed)) & 0xFFFFFFFF)


def _select_godark_windows(
    candidates: List[dict],
    max_windows: int,
    rng: np.random.RandomState,
) -> List[dict]:
    positives = [c for c in candidates if int(c["y"]) == 1]
    negatives = [c for c in candidates if int(c["y"]) == 0]
    cap = len(candidates) if max_windows <= 0 else int(max_windows)
    if len(positives) > cap:
        idx = rng.choice(len(positives), size=cap, replace=False)
        positives = [positives[int(i)] for i in idx.tolist()]
    remaining = max(0, cap - len(positives))
    # Maksimal 3 gap alami per positif; vessel tanpa positif tetap menyumbang
    # sedikit hard negatives untuk menguji false alarm.
    negative_budget = min(
        len(negatives),
        remaining,
        max(10, 3 * len(positives)) if positives else min(20, remaining),
    )
    if len(negatives) > negative_budget:
        idx = rng.choice(len(negatives), size=negative_budget, replace=False)
        negatives = [negatives[int(i)] for i in idx.tolist()]
    keep = positives + negatives
    return sorted(keep, key=lambda c: int(c["order"]))


def _select_godark_indices(y: np.ndarray, kinds: np.ndarray, max_windows: int, rng: np.random.RandomState) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    kinds = np.asarray(kinds).astype(str)
    if max_windows <= 0 or len(y) <= int(max_windows):
        return np.arange(len(y), dtype=np.int64)

    pos = np.where(y == 1)[0]
    hard_feature = np.where((y == 0) & np.char.startswith(kinds.astype(str), "hard_negative_feature"))[0]
    hard_gap = np.where((y == 0) & np.char.startswith(kinds.astype(str), "hard_negative_gap"))[0]
    hard_other = np.where(
        (y == 0)
        & np.char.startswith(kinds.astype(str), "hard_negative")
        & ~np.char.startswith(kinds.astype(str), "hard_negative_feature")
        & ~np.char.startswith(kinds.astype(str), "hard_negative_gap")
    )[0]
    normal = np.where((y == 0) & ~np.char.startswith(kinds.astype(str), "hard_negative"))[0]

    keep = pos.tolist()
    remaining = int(max_windows) - len(keep)
    if remaining <= 0:
        return np.sort(rng.choice(pos, size=int(max_windows), replace=False)).astype(np.int64)

    hard_take = min(remaining, max(remaining // 2, min(remaining, len(pos) * 3 if len(pos) else remaining // 2)))
    if hard_take > 0:
        feature_take = min(len(hard_feature), max(1, int(round(hard_take * 0.70))))
        if feature_take > 0:
            feature_pick = (
                rng.choice(hard_feature, size=feature_take, replace=False)
                if len(hard_feature) > feature_take else hard_feature
            )
            keep.extend([int(i) for i in feature_pick.tolist()])

        gap_pool = np.concatenate([hard_gap, hard_other], axis=0) if len(hard_gap) or len(hard_other) else np.array([], dtype=np.int64)
        gap_take = int(max(0, hard_take - feature_take))
        if gap_take > 0 and len(gap_pool):
            gap_pick = rng.choice(gap_pool, size=min(gap_take, len(gap_pool)), replace=False) if len(gap_pool) > gap_take else gap_pool
            keep.extend([int(i) for i in gap_pick.tolist()])

    remaining = int(max_windows) - len(keep)
    if remaining > 0 and len(normal):
        normal_pick = rng.choice(normal, size=min(remaining, len(normal)), replace=False)
        keep.extend([int(i) for i in normal_pick.tolist()])

    return np.sort(np.asarray(keep, dtype=np.int64))


def _build_godark_event_candidates(
    g: pd.DataFrame,
    feat: np.ndarray,
    coords: np.ndarray,
    feat_cols: List[str],
    cfg: PreprocessCfg,
    vessel_id: str,
) -> List[dict]:
    """Bentuk tepat satu context window untuk setiap gap yang dapat diamati."""
    ts = g["timestamp"].to_numpy(dtype=np.int64)
    event_ids = (
        g.get("go_dark_event_id", pd.Series("normal", index=g.index))
        .astype(str)
        .to_numpy()
    )
    y_point = g["y_point"].to_numpy(dtype=np.int64)
    source_class = str(
        g.get(
            "godark_source_class",
            pd.Series("unknown", index=g.index),
        ).iloc[0]
    ).strip().lower()
    shore_km = pd.to_numeric(
        g.get(
            "distance_from_shore_km_normalized",
            pd.Series(np.nan, index=g.index),
        ),
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    seq_len = int(cfg.seq_len)
    pre_len = seq_len // 2
    post_len = seq_len - pre_len
    min_shore_km = float(cfg.godark_min_distance_from_shore_nm) * 1.852
    min_ping = max(0, int(cfg.godark_min_ping_count_prev_window))
    ping_window = max(1, int(cfg.godark_ping_window_seconds))
    gap_idx = np.where(np.diff(ts) >= int(cfg.gap_seconds))[0]
    out: List[dict] = []

    for order, x_raw in enumerate(gap_idx.tolist()):
        x = int(x_raw)
        pre_start = x - pre_len + 1
        post_end = x + 1 + post_len
        if pre_start < 0 or post_end > len(g):
            continue

        # Aturan kandidat diterapkan sama pada positif dan negatif. Missing
        # shore context tidak boleh diam-diam dianggap lolos.
        if min_shore_km > 0.0:
            boundary_shore = shore_km[[x, x + 1]]
            if (not np.all(np.isfinite(boundary_shore))) or float(np.min(boundary_shore)) < min_shore_km:
                continue

        left_t = int(ts[x])
        ping_lo = int(np.searchsorted(ts, left_t - ping_window, side="left"))
        ping_count = max(0, x - ping_lo)
        if min_ping > 0 and ping_count < min_ping:
            continue

        left_event = str(event_ids[x])
        right_event = str(event_ids[x + 1])
        true_event = bool(
            left_event not in {"", "normal", "nan", "none"}
            and left_event == right_event
            and (int(y_point[x]) == 1 or int(y_point[x + 1]) == 1)
        )

        window = np.concatenate(
            [feat[pre_start:x + 1], feat[x + 1:post_end]],
            axis=0,
        )
        coord_window = np.concatenate(
            [coords[pre_start:x + 1], coords[x + 1:post_end]],
            axis=0,
        ).copy()
        coord_window[:, 3] = 0.0
        coord_window[pre_len, 3] = float(true_event)

        # ID berasal dari boundary yang terlihat, bukan go_dark_event_id label.
        observable_event_id = (
            f"gap::{source_class}::{vessel_id}::{int(ts[x])}::{int(ts[x + 1])}"
        )
        trajectory_position_fraction = float((x + 0.5) / max(len(g) - 1, 1))
        trajectory_position_stratum = int(
            np.digitize(trajectory_position_fraction, [1.0 / 3.0, 2.0 / 3.0])
        )
        event_order = 0
        if true_event:
            try:
                event_order = int(left_event.rsplit("_", 1)[-1])
            except (TypeError, ValueError):
                event_order = 1
        out.append(
            {
                "order": int(order),
                "X": window.astype(np.float32, copy=False),
                "coords": coord_window.astype(np.float64, copy=False),
                "y": int(true_event),
                "event_id": observable_event_id,
                "kind": "positive_event" if true_event else "hard_negative_gap",
                "source_class": source_class,
                "event_order": int(event_order),
                "trajectory_position_stratum": int(trajectory_position_stratum),
                "ping_count_prev_window": int(ping_count),
            }
        )

    return out


def build_transshipment_sequences_from_df(
    df: pd.DataFrame,
    cfg: PreprocessCfg,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    feat_cols = _transshipment_feature_cols(cfg.transshipment_feature_mode)
    need = [
        "event_id",
        "timestamp",
        "lat_mid",
        "lon_mid",
        "gear_a_label",
        "gear_b_label",
        "gear_a_id",
        "gear_b_id",
        "event_kind",
        "is_synthetic",
        "mmsi_a",
        "mmsi_b",
    ]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"task=transshipment but required column '{c}' not found")

    df = df.copy()
    allowed_sources = set(DEFAULT_TRANSSHIPMENT_SOURCE_INCLUDE_LABELS)
    gear_a = df["gear_a_label"].astype(str).str.strip().str.lower()
    gear_b = df["gear_b_label"].astype(str).str.strip().str.lower()
    invalid_a = sorted(set(gear_a.unique()) - allowed_sources)
    invalid_b = sorted(set(gear_b.unique()) - allowed_sources - {"", "none", "nan"})
    if invalid_a or invalid_b:
        raise ValueError(
            "task=transshipment only accepts source gear labels "
            f"{sorted(allowed_sources)}; invalid gear_a={invalid_a}, gear_b={invalid_b}"
        )
    df["gear_a_label"] = gear_a
    df["gear_b_label"] = gear_b
    gear_a_id = pd.to_numeric(df["gear_a_id"], errors="coerce")
    gear_b_id = pd.to_numeric(df["gear_b_id"], errors="coerce")
    expected_a_id = gear_a.map(TRANSSHIPMENT_GEAR_TO_ID).astype(float)
    expected_b_id = gear_b.map(TRANSSHIPMENT_GEAR_TO_ID).fillna(-1).astype(float)
    bad_a_id = gear_a_id.isna() | gear_a_id.ne(expected_a_id)
    bad_b_id = gear_b_id.isna() | gear_b_id.ne(expected_b_id)
    if bool(bad_a_id.any()) or bool(bad_b_id.any()):
        raise ValueError(
            "task=transshipment gear IDs do not match the locked mapping "
            f"{TRANSSHIPMENT_GEAR_TO_ID}; bad gear_a rows={int(bad_a_id.sum())}, "
            f"bad gear_b rows={int(bad_b_id.sum())}"
        )
    df["event_id"] = df["event_id"].astype(str)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["lat_mid"] = pd.to_numeric(df["lat_mid"], errors="coerce")
    df["lon_mid"] = pd.to_numeric(df["lon_mid"], errors="coerce")
    df = df.dropna(subset=["event_id", "timestamp", "lat_mid", "lon_mid"]).copy()
    if df.empty:
        return (
            np.zeros((0, cfg.seq_len, len(feat_cols)), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=object),
            np.zeros((0, cfg.seq_len, 4), dtype=np.float64),
            np.zeros((0, cfg.seq_len, len(TRANS_RULE_SCORE_COLS)), dtype=np.float32),
            {
                "window_is_synthetic": np.zeros((0,), dtype=np.int8),
                "window_mmsi_a": np.zeros((0,), dtype=object),
                "window_mmsi_b": np.zeros((0,), dtype=object),
                "window_source_labels": np.zeros((0,), dtype=object),
                "window_kinds": np.zeros((0,), dtype=object),
            },
        )

    if "class_id" not in df.columns:
        df["class_id"] = df.apply(_transshipment_class_id, axis=1)
    else:
        df["class_id"] = (
            pd.to_numeric(df["class_id"], errors="coerce")
            .fillna(0)
            .astype(int)
            .clip(lower=0, upper=2)
        )
    target = _transshipment_target_from_cfg(df, cfg)

    for c in TRANS_FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    df["valid_point"] = pd.to_numeric(df["valid_point"], errors="coerce").fillna(1.0)
    df[TRANS_FEATURE_COLS] = df[TRANS_FEATURE_COLS].fillna(0.0)
    df = _stabilize_transshipment_features(df)
    df = df.sort_values(["event_id", "timestamp"]).reset_index(drop=True)

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    g_list: List[str] = []
    coord_list: List[np.ndarray] = []
    rule_list: List[np.ndarray] = []
    synthetic_list: List[int] = []
    mmsi_a_list: List[str] = []
    mmsi_b_list: List[str] = []
    source_list: List[str] = []
    kind_list: List[str] = []

    min_points = max(1, int(cfg.min_points_per_vessel))
    seq_len = int(cfg.seq_len)
    stride = max(1, int(cfg.stride))

    for event_id, g in df.groupby("event_id", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        if len(g) < min_points:
            continue

        # Read rule scores from the current event itself. Using the reset
        # per-event index against a global DataFrame can silently mix scores
        # from different events.
        rule_g = g[TRANS_RULE_SCORE_COLS].to_numpy(dtype=np.float32)
        feat = g[feat_cols].to_numpy(dtype=np.float32)
        class_ids = g["class_id"].to_numpy(dtype=np.int64)
        y_mapped = _map_transshipment_y(class_ids, target)
        y_point = (y_mapped > 0).astype(np.int64)
        coords = np.column_stack(
            [
                g["timestamp"].to_numpy(dtype=np.float64),
                g["lat_mid"].to_numpy(dtype=np.float64),
                g["lon_mid"].to_numpy(dtype=np.float64),
                y_point.astype(np.float64),
            ]
        )
        y_event = int(np.bincount(y_mapped, minlength=3).argmax())
        is_synthetic = int(
            pd.to_numeric(g["is_synthetic"], errors="coerce")
            .fillna(0)
            .astype(int)
            .max()
            > 0
        )
        event_kind = str(g["event_kind"].iloc[0]).strip().lower()
        gear_a_label = str(g["gear_a_label"].iloc[0]).strip().lower()
        gear_b_label = str(g["gear_b_label"].iloc[0]).strip().lower()
        source_label = (
            gear_a_label
            if gear_b_label in {"", "none", "nan"}
            else "__".join(sorted([gear_a_label, gear_b_label]))
        )

        def mmsi_text(value: object) -> str:
            numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            return "" if pd.isna(numeric) else str(int(round(float(numeric))))

        mmsi_a = mmsi_text(g["mmsi_a"].iloc[0])
        mmsi_b = mmsi_text(g["mmsi_b"].iloc[0])
        window_kind = (
            "synthetic_encounter"
            if is_synthetic
            else f"real_{event_kind}_{'positive' if y_event > 0 else 'negative'}"
        )

        def append_event_metadata() -> None:
            synthetic_list.append(is_synthetic)
            mmsi_a_list.append(mmsi_a)
            mmsi_b_list.append(mmsi_b)
            source_list.append(source_label)
            kind_list.append(window_kind)

        windows_made = 0
        if len(g) < seq_len:
            pad = seq_len - len(g)
            feat_pad = np.zeros((pad, len(feat_cols)), dtype=np.float32)
            valid_idx = feat_cols.index("valid_point")
            feat_pad[:, valid_idx] = 0.0
            rule_pad = np.zeros((pad, len(TRANS_RULE_SCORE_COLS)), dtype=np.float32)
            coord_pad = np.zeros((pad, 4), dtype=np.float64)
            if len(coords) > 0:
                coord_pad[:, 0] = coords[-1, 0]
                coord_pad[:, 1] = coords[-1, 1]
                coord_pad[:, 2] = coords[-1, 2]
                coord_pad[:, 3] = 0.0
            X_list.append(np.concatenate([feat, feat_pad], axis=0))
            rule_list.append(np.concatenate([rule_g, rule_pad], axis=0))
            coord_list.append(np.concatenate([coords, coord_pad], axis=0))
            y_list.append(y_event)
            g_list.append(str(event_id))
            append_event_metadata()
            continue

        for i in range(0, len(g) - seq_len + 1, stride):
            window = feat[i:i + seq_len]
            rule_window = rule_g[i:i + seq_len]
            coord_window = coords[i:i + seq_len]
            y_win_vals = y_mapped[i:i + seq_len]
            y_win = int(np.bincount(y_win_vals, minlength=3).argmax())
            if np.any(y_win_vals > 0):
                positives = y_win_vals[y_win_vals > 0]
                if positives.size:
                    y_win = int(np.bincount(positives, minlength=3).argmax())

            X_list.append(window)
            rule_list.append(rule_window)
            coord_list.append(coord_window)
            y_list.append(y_win)
            g_list.append(str(event_id))
            append_event_metadata()
            windows_made += 1
            if cfg.max_windows_per_vessel > 0 and windows_made >= int(cfg.max_windows_per_vessel):
                break

    if not X_list:
        return (
            np.zeros((0, seq_len, len(feat_cols)), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=object),
            np.zeros((0, seq_len, 4), dtype=np.float64),
            np.zeros((0, seq_len, len(TRANS_RULE_SCORE_COLS)), dtype=np.float32),
            {
                "window_is_synthetic": np.zeros((0,), dtype=np.int8),
                "window_mmsi_a": np.zeros((0,), dtype=object),
                "window_mmsi_b": np.zeros((0,), dtype=object),
                "window_source_labels": np.zeros((0,), dtype=object),
                "window_kinds": np.zeros((0,), dtype=object),
            },
        )

    return (
        np.stack(X_list).astype(np.float32),
        np.array(y_list, dtype=np.int64),
        np.array(g_list, dtype=object),
        np.stack(coord_list).astype(np.float64),
        np.stack(rule_list).astype(np.float32),
        {
            "window_is_synthetic": np.asarray(synthetic_list, dtype=np.int8),
            "window_mmsi_a": np.asarray(mmsi_a_list, dtype=object),
            "window_mmsi_b": np.asarray(mmsi_b_list, dtype=object),
            "window_source_labels": np.asarray(source_list, dtype=object),
            "window_kinds": np.asarray(kind_list, dtype=object),
        },
    )


def build_sequences_from_df(
    df: pd.DataFrame,
    cfg: PreprocessCfg,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    df = clean_and_derive(df, cfg)

    if cfg.apply_jump_filter:
        filtered = filter_jumps(df, cfg)
        if len(filtered) != len(df):
            # Removing a jump changes the predecessor of the following point,
            # so all derivative and rolling features must be recomputed.
            df = clean_and_derive(filtered, cfg)
        else:
            df = filtered

    if cfg.task == "fishing":
        if "is_fishing" not in df.columns:
            raise ValueError("task=fishing but 'is_fishing' not found")
        df = df[df["is_fishing"] != -1].copy()
        df["y_point"] = (df["is_fishing"] >= 0.5).astype("int8")

    elif cfg.task == "spoofing":
        if "is_spoofing" in df.columns:
            df["y_point"] = (
                pd.to_numeric(df["is_spoofing"], errors="coerce")
                .fillna(0)
                .astype("int8")
            )
        elif "label" in df.columns:
            df["y_point"] = (
                df["label"].astype(str).str.lower().eq("spoofed").astype("int8")
            )
        elif "attack_type" in df.columns:
            df["y_point"] = (
                ~df["attack_type"].astype(str).str.lower().eq("normal")
            ).astype("int8")
        else:
            raise ValueError(
                "task=spoofing but no 'is_spoofing', 'label', or 'attack_type' column found"
            )
        if "is_spoofing_event" in df.columns:
            df["y_spoof_event"] = (
                pd.to_numeric(df["is_spoofing_event"], errors="coerce")
                .fillna(0)
                .astype("int8")
            )
        else:
            # Backward compatibility for previously generated spoofing CSVs.
            df["y_spoof_event"] = df["y_point"]

    elif cfg.task == "godark":
        if "is_go_dark" in df.columns:
            df["y_point"] = (
                pd.to_numeric(df["is_go_dark"], errors="coerce")
                .fillna(0)
                .astype("int8")
            )
        elif "label" in df.columns:
            df["y_point"] = (
                df["label"]
                .astype(str)
                .str.lower()
                .isin(["godark", "go_dark"])
                .astype("int8")
            )
        elif "event_type" in df.columns:
            df["y_point"] = (
                df["event_type"].astype(str).str.lower().eq("go_dark").astype("int8")
            )
        elif "attack_type" in df.columns:
            df["y_point"] = (
                df["attack_type"].astype(str).str.lower().eq("go_dark").astype("int8")
            )
        else:
            raise ValueError(
                "task=godark but no 'is_go_dark', 'label', 'event_type', or 'attack_type' column found"
            )

    else:
        df["y_point"] = 0

    feat_cols = _sequence_feature_cols(cfg)

    df = df.dropna(subset=feat_cols).copy()

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    g_list: List[str] = []
    coord_list: List[np.ndarray] = []
    event_id_list: List[str] = []
    kind_list: List[str] = []
    source_list: List[str] = []
    event_order_list: List[int] = []
    position_stratum_list: List[int] = []
    balance_stratum_list: List[str] = []

    for vessel_id, g in df.groupby("mmsi", sort=False):
        g = g.sort_values("timestamp")

        if len(g) < cfg.min_points_per_vessel:
            continue

        ts = g["timestamp"].to_numpy()
        y_point = g["y_point"].to_numpy(dtype=np.int64)
        spoof_event = (
            g["y_spoof_event"].to_numpy(dtype=np.int64)
            if cfg.task == "spoofing"
            else np.zeros(len(g), dtype=np.int64)
        )

        # Task umum memotong trajectory pada gap. GoDark membentuk context
        # event khusus di bawah dan tidak menggunakan segment ini.
        raw_gaps = np.where((ts[1:] - ts[:-1]) > cfg.gap_seconds)[0]
        gaps = raw_gaps

        starts = [0] + [int(x + 1) for x in gaps]
        ends = [int(x + 1) for x in gaps] + [len(g)]

        feat = g[feat_cols].to_numpy(dtype=np.float32)
        coords = g[["timestamp", "lat", "lon", "y_point"]].to_numpy(dtype=np.float64)
        split_group_id = str(vessel_id)
        spoof_scenario_ids = np.array([str(vessel_id)] * len(g), dtype=object)
        spoof_attack_types = np.array(["normal"] * len(g), dtype=object)
        spoof_source_label = "unknown"
        if cfg.task == "spoofing":
            if "original_mmsi" in g.columns:
                source_ids = g["original_mmsi"].dropna().astype(str).unique()
                if source_ids.size != 1:
                    raise ValueError(
                        "Each spoofing scenario trajectory must map to exactly "
                        f"one original_mmsi; got {source_ids.tolist()}."
                    )
                split_group_id = str(source_ids[0])
            if "scenario_id" in g.columns:
                spoof_scenario_ids = g["scenario_id"].astype(str).to_numpy()
            if "attack_type" in g.columns:
                spoof_attack_types = (
                    g["attack_type"].astype(str).str.lower().to_numpy()
                )
            if "source_label" not in g.columns:
                raise ValueError(
                    "task=spoofing requires source_label so the locked four-gear "
                    "protocol can be audited. Regenerate spoofing data."
                )
            source_values = sorted(
                {
                    str(value).strip().lower()
                    for value in g["source_label"].dropna().tolist()
                    if str(value).strip()
                }
            )
            if len(source_values) != 1:
                raise ValueError(
                    "Each spoofing scenario must map to exactly one source_label; "
                    f"got {source_values}."
                )
            spoof_source_label = source_values[0]
            if spoof_source_label not in DEFAULT_SPOOFING_SOURCE_INCLUDE_LABELS:
                raise ValueError(
                    "task=spoofing only accepts source labels "
                    f"{list(DEFAULT_SPOOFING_SOURCE_INCLUDE_LABELS)}; "
                    f"got {spoof_source_label!r}."
                )
        event_ids = (
            g.get("go_dark_event_id", pd.Series("normal", index=g.index))
            .astype(str)
            .to_numpy()
        )

        if cfg.task == "godark":
            candidates = _build_godark_event_candidates(
                g=g,
                feat=feat,
                coords=coords,
                feat_cols=feat_cols,
                cfg=cfg,
                vessel_id=str(vessel_id),
            )

            rng = _stable_random_state(str(vessel_id))
            selected = _select_godark_windows(candidates, int(cfg.max_windows_per_vessel), rng)
            for item in selected:
                X_list.append(item["X"])
                y_list.append(int(item["y"]))
                g_list.append(vessel_id)
                coord_list.append(item["coords"])
                event_id_list.append(str(item["event_id"]))
                kind_list.append(str(item["kind"]))
                source_list.append(str(item.get("source_class", "unknown")))
                event_order_list.append(int(item.get("event_order", 0)))
                position_stratum_list.append(
                    int(item.get("trajectory_position_stratum", -1))
                )
                balance_stratum_list.append("not_applicable")
            continue

        win_count = 0

        for s, e in zip(starts, ends):
            if e - s < cfg.seq_len:
                continue

            seg_feat = feat[s:e]
            seg_y = y_point[s:e]

            for i in range(0, len(seg_feat) - cfg.seq_len + 1, cfg.stride):
                window = seg_feat[i:i + cfg.seq_len]
                coord_window = coords[s + i:s + i + cfg.seq_len]

                if cfg.task == "fishing":
                    y_win = int(seg_y[i:i + cfg.seq_len].mean() >= 0.5)

                elif cfg.task == "spoofing":
                    attack_values = spoof_attack_types[
                        s + i:s + i + cfg.seq_len
                    ]
                    attack_kind = str(
                        pd.Series(attack_values).value_counts().index[0]
                    ).lower()
                    if attack_kind == "location_jump":
                        # With absolute location features disabled, a constant
                        # post-jump translation is indistinguishable from a
                        # normal track. Only windows spanning the jump boundary
                        # carry a motion anomaly.
                        y_win = int(
                            spoof_event[s + i:s + i + cfg.seq_len].max() >= 1
                        )
                    else:
                        y_win = int(
                            seg_y[i:i + cfg.seq_len].mean()
                            >= float(cfg.spoofing_window_threshold)
                        )

                elif cfg.task == "godark":
                    # Go-dark adalah event boundary.
                    # Titik positifnya sedikit, jadi pakai max, bukan mean threshold.
                    y_win = int(seg_y[i:i + cfg.seq_len].max() >= 1)

                else:
                    y_win = 0

                X_list.append(window)
                y_list.append(y_win)
                g_list.append(split_group_id)
                coord_list.append(coord_window)
                if cfg.task == "spoofing":
                    scenario_values = spoof_scenario_ids[
                        s + i:s + i + cfg.seq_len
                    ]
                    attack_values = spoof_attack_types[
                        s + i:s + i + cfg.seq_len
                    ]
                    event_id_list.append(str(scenario_values[0]))
                    attack_counts = pd.Series(attack_values).value_counts()
                    kind_list.append(str(attack_counts.index[0]))
                    raw_window = g.iloc[s + i:s + i + cfg.seq_len]
                    attack_ref = str(attack_counts.index[0]).strip().lower()
                    if attack_ref == "normal" and "normal_control_for_attack" in raw_window:
                        controls = (
                            raw_window["normal_control_for_attack"]
                            .astype(str)
                            .str.strip()
                            .str.lower()
                        )
                        controls = controls[~controls.isin(["", "nan", "none"])]
                        control_counts = controls.value_counts(dropna=True)
                        if len(control_counts.index) > 0:
                            attack_ref = str(control_counts.index[0])
                    duration_hours = float(
                        pd.to_numeric(
                            raw_window.get(
                                "attack_duration_hours",
                                pd.Series(0.0, index=raw_window.index),
                            ),
                            errors="coerce",
                        ).fillna(0.0).median()
                    )
                    displacement_km = float(
                        pd.to_numeric(
                            raw_window.get(
                                "attack_displacement_km",
                                pd.Series(0.0, index=raw_window.index),
                            ),
                            errors="coerce",
                        ).fillna(0.0).median()
                    )
                    raw_ts = raw_window["timestamp"].to_numpy(dtype=np.float64)
                    cadence_seconds = float(
                        np.median(np.diff(raw_ts)[np.diff(raw_ts) > 0])
                    ) if len(raw_ts) > 1 and np.any(np.diff(raw_ts) > 0) else 0.0
                    duration_bin = (
                        "short" if duration_hours < 6.0
                        else "medium" if duration_hours < 24.0
                        else "long"
                    )
                    cadence_bin = (
                        "dense" if cadence_seconds < 300.0
                        else "medium" if cadence_seconds < 1800.0
                        else "sparse"
                    )
                    magnitude_bin = (
                        "small" if displacement_km < 10.0
                        else "medium" if displacement_km < 100.0
                        else "large"
                    )
                    balance_stratum_list.append(
                        "::".join(
                            [
                                spoof_source_label,
                                attack_ref or "normal_random",
                                str(int(y_win)),
                                duration_bin,
                                cadence_bin,
                                magnitude_bin,
                            ]
                        )
                    )
                else:
                    event_id_list.append(str(vessel_id))
                    kind_list.append(
                        "positive_event" if y_win else "normal_random"
                    )
                    balance_stratum_list.append("not_applicable")
                source_list.append(
                    spoof_source_label if cfg.task == "spoofing" else "unknown"
                )
                event_order_list.append(0)
                position_stratum_list.append(-1)
                win_count += 1

                if (
                    cfg.max_windows_per_vessel > 0
                    and win_count >= cfg.max_windows_per_vessel
                ):
                    break

            if (
                cfg.max_windows_per_vessel > 0
                and win_count >= cfg.max_windows_per_vessel
            ):
                break

    F = len(feat_cols)

    if not X_list:
        return (
            np.zeros((0, cfg.seq_len, F), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=object),
            np.zeros((0, cfg.seq_len, 4), dtype=np.float64),
            np.zeros((0,), dtype=object),
            np.zeros((0,), dtype=object),
            np.zeros((0,), dtype=object),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=object),
        )

    return (
        np.stack(X_list),
        np.array(y_list, dtype=np.int64),
        np.array(g_list, dtype=object),
        np.stack(coord_list).astype(np.float64),
        np.array(event_id_list, dtype=object),
        np.array(kind_list, dtype=object),
        np.array(source_list, dtype=object),
        np.array(event_order_list, dtype=np.int64),
        np.array(position_stratum_list, dtype=np.int64),
        np.array(balance_stratum_list, dtype=object),
    )


def build_sequences_to_npz(
    data_dir: Path,
    out_dir: Path,
    task: str = "gear",
    exclude_labels: Optional[List[str]] = None,
    limit_rows: int = 0,
    chunksize: int = 0,
    seq_len: int = 120,
    stride: int = 6,
    gap_seconds: int = 10800,
    max_implied_knots: float = 42.0,
    min_points_per_vessel: int = 80,
    min_windows_per_vessel: int = 0,
    max_windows_per_vessel: int = 1200,
    max_windows_per_file: int = 20000,
    balance_gear_classes: bool = False,
    use_operational_filter: bool = False,
    op_speed_min: float = 1.0,
    op_speed_max: float = 12.0,
    use_location_features: bool = True,
    spoofing_window_threshold: float = 0.20,
    godark_min_distance_from_shore_nm: float = 50.0,
    godark_ping_window_seconds: int = 12 * 3600,
    godark_min_ping_count_prev_window: int = 14,
    transshipment_target: str = "multiclass",
    transshipment_feature_mode: str = "fair",
    apply_jump_filter: Optional[bool] = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if apply_jump_filter is None:
        apply_jump_filter = task not in ["spoofing", "godark", "transshipment"]

    allow_spoofing_location_features = (
        os.environ.get("SPOOFING_ALLOW_LOCATION_FEATURES", "0") == "1"
    )
    if (
        str(task) == "spoofing"
        and bool(use_location_features)
        and not allow_spoofing_location_features
    ):
        print(
            "[preprocess] WARNING: disabling distance_from_shore and "
            "distance_from_port for spoofing. Synthetic coordinate attacks do "
            "not have recomputed geospatial distance rasters."
        )
        use_location_features = False
    elif (
        str(task) == "spoofing"
        and bool(use_location_features)
        and allow_spoofing_location_features
    ):
        print(
            "[preprocess] WARNING: spoofing location features explicitly enabled "
            "by SPOOFING_ALLOW_LOCATION_FEATURES=1. Use only for controlled "
            "ablation because synthetic coordinate attacks may expose a "
            "geospatial shortcut."
        )

    if str(task) == "godark" and bool(use_location_features):
        print(
            "[preprocess] GoDark location features disabled. Shore distance "
            "is used only as an equal candidate filter to prevent label shortcut."
        )
        use_location_features = False

    if task == "transshipment" and int(min_points_per_vessel) == 80:
        min_points_per_vessel = 3

    if str(task) == "gear" and int(limit_rows) > 0:
        print(
            "[preprocess] WARNING: task=gear is using limit_rows="
            f"{int(limit_rows)}. This reads only the first rows of each gear CSV "
            "and can leave too few vessels for stable vessel-level validation."
        )

    cfg = PreprocessCfg(
        task=task,
        seq_len=int(seq_len),
        stride=int(stride),
        gap_seconds=int(gap_seconds),
        max_implied_knots=float(max_implied_knots),
        min_points_per_vessel=int(min_points_per_vessel),
        min_windows_per_vessel=int(min_windows_per_vessel),
        max_windows_per_vessel=int(max_windows_per_vessel),
        max_windows_per_file=int(max_windows_per_file),
        balance_gear_classes=bool(balance_gear_classes),
        use_operational_filter=bool(use_operational_filter),
        op_speed_min=float(op_speed_min),
        op_speed_max=float(op_speed_max),
        use_location_features=bool(use_location_features),
        spoofing_window_threshold=float(spoofing_window_threshold),
        godark_min_distance_from_shore_nm=float(godark_min_distance_from_shore_nm),
        godark_ping_window_seconds=int(godark_ping_window_seconds),
        godark_min_ping_count_prev_window=int(godark_min_ping_count_prev_window),
        transshipment_target=str(transshipment_target),
        transshipment_feature_mode=str(transshipment_feature_mode),
        apply_jump_filter=bool(apply_jump_filter),
    )

    exclude_labels = _normalize_exclude_labels(task, exclude_labels)

    csvs = sorted(list(Path(data_dir).glob("*.csv")))

    if task == "spoofing":
        combined = Path(data_dir) / "spoofed_all.csv"
        if combined.exists():
            csvs = [combined]
        else:
            csvs = [p for p in csvs if p.name.startswith("spoofed_")]

    if task == "godark":
        combined = Path(data_dir) / "godark_all.csv"
        if combined.exists():
            csvs = [combined]
        else:
            csvs = [p for p in csvs if p.name.startswith("godark_")]

    if task == "transshipment":
        combined = Path(data_dir) / "transshipment_all.csv"
        if combined.exists():
            csvs = [combined]
        else:
            csvs = [p for p in csvs if p.name.startswith("transshipment_")]

    if not csvs:
        raise FileNotFoundError(f"No CSV found in {data_dir}")

    if task == "transshipment" and str(cfg.transshipment_target).strip().lower() == "auto":
        cfg.transshipment_target = _infer_transshipment_auto_target(
            csvs=csvs,
            limit_rows=int(limit_rows),
            chunksize=int(chunksize),
        )
        print(f"[preprocess] transshipment_target auto -> {cfg.transshipment_target}")

    all_X, all_y, all_groups, all_coords = [], [], [], []
    all_window_event_ids, all_window_kinds, all_window_source_labels = [], [], []
    all_window_event_orders, all_window_position_strata = [], []
    all_spoofing_balance_strata = []
    all_window_is_synthetic, all_window_mmsi_a, all_window_mmsi_b = [], [], []
    all_rule_features = []
    gear_to_id: Dict[str, int] = {}
    next_id = 0

    print(f"[preprocess] files={len(csvs)} exclude={exclude_labels}")
    print(
        f"[preprocess] seq_len={cfg.seq_len} stride={cfg.stride} "
        f"gap={cfg.gap_seconds} max_knots={cfg.max_implied_knots} "
        f"apply_jump_filter={cfg.apply_jump_filter}"
    )
    print(
        f"[preprocess] max_windows_per_vessel={cfg.max_windows_per_vessel} "
        f"min_points_per_vessel={cfg.min_points_per_vessel} "
        f"min_windows_per_vessel={cfg.min_windows_per_vessel}"
    )
    print(f"[preprocess] max_windows_per_file={cfg.max_windows_per_file}")
    if cfg.use_operational_filter:
        print(
            "[preprocess] operational_filter enabled: "
            f"speed between {cfg.op_speed_min:g} and {cfg.op_speed_max:g} knots"
        )
    print(
        "[preprocess] location features "
        f"{'enabled' if cfg.use_location_features else 'disabled'}"
    )
    if task == "godark":
        print(
            "[preprocess] GoDark event candidates: "
            f"gap>={cfg.gap_seconds}s shore>={cfg.godark_min_distance_from_shore_nm:g}nm "
            f"pre_gap_pings>={cfg.godark_min_ping_count_prev_window} "
            f"within={cfg.godark_ping_window_seconds}s; one context window/event"
        )

    for p in tqdm(csvs, desc="files"):
        stem = infer_label_from_filename(p)

        if stem in exclude_labels:
            continue

        df = read_ais_csv(p, limit_rows=limit_rows, chunksize=chunksize)

        try:
            if task == "transshipment":
                X, y, groups, coords, rule_features, trans_meta = (
                    build_transshipment_sequences_from_df(df, cfg)
                )
                window_event_ids = groups.astype(object)
                window_kinds = trans_meta["window_kinds"]
                window_source_labels = trans_meta["window_source_labels"]
                window_is_synthetic = trans_meta["window_is_synthetic"]
                window_mmsi_a = trans_meta["window_mmsi_a"]
                window_mmsi_b = trans_meta["window_mmsi_b"]
                window_event_orders = np.zeros(len(y), dtype=np.int64)
                window_position_strata = np.full(len(y), -1, dtype=np.int64)
                spoofing_balance_strata = np.full(
                    len(y), "not_applicable", dtype=object
                )
            else:
                (
                    X,
                    y,
                    groups,
                    coords,
                    window_event_ids,
                    window_kinds,
                    window_source_labels,
                    window_event_orders,
                    window_position_strata,
                    spoofing_balance_strata,
                ) = build_sequences_from_df(df, cfg)
                rule_features = None
                window_is_synthetic = np.zeros(len(y), dtype=np.int8)
                window_mmsi_a = np.full(len(y), "", dtype=object)
                window_mmsi_b = np.full(len(y), "", dtype=object)
        except ValueError as exc:
            if task in ["spoofing", "godark", "transshipment"]:
                print(f"[preprocess] skip {p.name}: {exc}")
                continue
            raise

        if len(X) == 0:
            continue

        if task == "gear":
            if stem not in gear_to_id:
                gear_to_id[stem] = next_id
                next_id += 1

            y = np.full((len(X),), gear_to_id[stem], dtype=np.int64)

        else:
            y = y.astype(np.int64)

        if cfg.max_windows_per_file and len(X) > cfg.max_windows_per_file:
            rng = np.random.RandomState(42)
            if task == "godark":
                idx = _select_godark_indices(y, window_kinds, int(cfg.max_windows_per_file), rng)
            elif task == "spoofing":
                idx = _select_spoofing_cap_indices(
                    y,
                    window_kinds,
                    int(cfg.max_windows_per_file),
                    rng,
                )
            else:
                idx = rng.choice(len(X), size=cfg.max_windows_per_file, replace=False)

            X = X[idx]
            y = y[idx]
            groups = groups[idx]
            coords = coords[idx]
            window_event_ids = window_event_ids[idx]
            window_kinds = window_kinds[idx]
            window_source_labels = window_source_labels[idx]
            window_event_orders = window_event_orders[idx]
            window_position_strata = window_position_strata[idx]
            spoofing_balance_strata = spoofing_balance_strata[idx]
            window_is_synthetic = window_is_synthetic[idx]
            window_mmsi_a = window_mmsi_a[idx]
            window_mmsi_b = window_mmsi_b[idx]
            if rule_features is not None:
                rule_features = rule_features[idx]

        all_X.append(X)
        all_y.append(y)
        all_groups.append(groups)
        all_coords.append(coords)
        all_window_event_ids.append(window_event_ids)
        all_window_kinds.append(window_kinds)
        all_window_source_labels.append(window_source_labels)
        all_window_event_orders.append(window_event_orders)
        all_window_position_strata.append(window_position_strata)
        all_spoofing_balance_strata.append(spoofing_balance_strata)
        all_window_is_synthetic.append(window_is_synthetic)
        all_window_mmsi_a.append(window_mmsi_a)
        all_window_mmsi_b.append(window_mmsi_b)
        if rule_features is not None:
            all_rule_features.append(rule_features)

    if not all_X:
        raise RuntimeError("No sequences created. Coba kecilin seq_len atau cek file.")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    groups = np.concatenate(all_groups, axis=0)
    coords = np.concatenate(all_coords, axis=0)
    window_event_ids = np.concatenate(all_window_event_ids, axis=0) if all_window_event_ids else np.array([], dtype=object)
    window_kinds = np.concatenate(all_window_kinds, axis=0) if all_window_kinds else np.array([], dtype=object)
    window_source_labels = (
        np.concatenate(all_window_source_labels, axis=0)
        if all_window_source_labels
        else np.array([], dtype=object)
    )
    window_event_orders = (
        np.concatenate(all_window_event_orders, axis=0)
        if all_window_event_orders
        else np.array([], dtype=np.int64)
    )
    window_position_strata = (
        np.concatenate(all_window_position_strata, axis=0)
        if all_window_position_strata
        else np.array([], dtype=np.int64)
    )
    spoofing_balance_strata = (
        np.concatenate(all_spoofing_balance_strata, axis=0)
        if all_spoofing_balance_strata
        else np.full((len(X),), "not_applicable", dtype=object)
    )
    window_is_synthetic = (
        np.concatenate(all_window_is_synthetic, axis=0)
        if all_window_is_synthetic
        else np.zeros((len(X),), dtype=np.int8)
    )
    window_mmsi_a = (
        np.concatenate(all_window_mmsi_a, axis=0)
        if all_window_mmsi_a
        else np.full((len(X),), "", dtype=object)
    )
    window_mmsi_b = (
        np.concatenate(all_window_mmsi_b, axis=0)
        if all_window_mmsi_b
        else np.full((len(X),), "", dtype=object)
    )
    rule_features = (
        np.concatenate(all_rule_features, axis=0)
        if all_rule_features
        else np.zeros((len(X), X.shape[1], 0), dtype=np.float32)
    )

    if cfg.min_windows_per_vessel > 0:
        groups_str = groups.astype(str)
        uniq, counts = np.unique(groups_str, return_counts=True)
        keep_groups = set(uniq[counts >= int(cfg.min_windows_per_vessel)].tolist())
        keep_mask = np.array([str(g) in keep_groups for g in groups_str], dtype=bool)
        removed_vessels = int(uniq.size - len(keep_groups))
        removed_windows = int((~keep_mask).sum())
        if removed_vessels > 0:
            print(
                "[preprocess] removed vessels below min_windows_per_vessel: "
                f"vessels={removed_vessels} windows={removed_windows}"
            )
            X = X[keep_mask]
            y = y[keep_mask]
            groups = groups[keep_mask]
            coords = coords[keep_mask]
            window_event_ids = window_event_ids[keep_mask]
            window_kinds = window_kinds[keep_mask]
            window_source_labels = window_source_labels[keep_mask]
            window_event_orders = window_event_orders[keep_mask]
            window_position_strata = window_position_strata[keep_mask]
            spoofing_balance_strata = spoofing_balance_strata[keep_mask]
            window_is_synthetic = window_is_synthetic[keep_mask]
            window_mmsi_a = window_mmsi_a[keep_mask]
            window_mmsi_b = window_mmsi_b[keep_mask]
            if rule_features.shape[-1] > 0:
                rule_features = rule_features[keep_mask]

    if task == "gear":
        counts = np.bincount(y.astype(np.int64), minlength=len(gear_to_id))
        print("[preprocess] gear class windows before balance:", {label: int(counts[idx]) for label, idx in gear_to_id.items()})

        if cfg.balance_gear_classes and len(counts) > 1:
            min_count = int(counts[counts > 0].min())
            rng = np.random.RandomState(42)
            keep_parts = []
            for cls in range(len(counts)):
                cls_idx = np.where(y == cls)[0]
                if cls_idx.size == 0:
                    continue
                if cls_idx.size > min_count:
                    cls_idx = rng.choice(cls_idx, size=min_count, replace=False)
                keep_parts.append(cls_idx)

            if keep_parts:
                keep_idx = np.concatenate(keep_parts)
                rng.shuffle(keep_idx)
                X = X[keep_idx]
                y = y[keep_idx]
                groups = groups[keep_idx]
                coords = coords[keep_idx]
                window_event_ids = window_event_ids[keep_idx]
                window_kinds = window_kinds[keep_idx]
                window_source_labels = window_source_labels[keep_idx]
                window_event_orders = window_event_orders[keep_idx]
                window_position_strata = window_position_strata[keep_idx]
                spoofing_balance_strata = spoofing_balance_strata[keep_idx]
                window_is_synthetic = window_is_synthetic[keep_idx]
                window_mmsi_a = window_mmsi_a[keep_idx]
                window_mmsi_b = window_mmsi_b[keep_idx]
                if rule_features.shape[-1] > 0:
                    rule_features = rule_features[keep_idx]

                counts = np.bincount(y.astype(np.int64), minlength=len(gear_to_id))
                print("[preprocess] gear class windows after balance:", {label: int(counts[idx]) for label, idx in gear_to_id.items()})

        label_map = {v: k for k, v in gear_to_id.items()}
    elif task == "spoofing":
        label_map = {0: "normal", 1: "spoofing"}
    elif task == "godark":
        label_map = {0: "normal", 1: "go_dark"}
    elif task == "transshipment":
        target_for_map = str(cfg.transshipment_target or "multiclass").strip().lower()
        if target_for_map == "auto":
            present = set(np.unique(y.astype(np.int64)).tolist())
            if present <= {0, 1} and 1 in present:
                # In auto mode the builder has already collapsed a single
                # positive source class to binary. Use a neutral label.
                target_for_map = "any"
            else:
                target_for_map = "multiclass"
        label_map = _transshipment_label_map_for_target(target_for_map)
    else:
        label_map = {0: "not_fishing", 1: "fishing"}

    out_path = out_dir / f"processed_{task}.npz"

    save_payload = {
        "X": X.astype(np.float32),
        "y": y,
        "groups": groups,
        "coords": coords,
        "coord_cols": np.array(
            ["timestamp", "lat", "lon", "y_point"], dtype=object
        ),
        "window_event_ids": window_event_ids.astype(object),
        "window_kinds": window_kinds.astype(object),
        "feature_cols": np.array(
            _transshipment_feature_cols(cfg.transshipment_feature_mode)
            if task == "transshipment"
            else _sequence_feature_cols(cfg),
            dtype=object,
        ),
        "rule_features": rule_features.astype(np.float32),
        "rule_cols": np.array(
            TRANS_RULE_SCORE_COLS if task == "transshipment" else [],
            dtype=object,
        ),
        "transshipment_target": np.array(
            str(cfg.transshipment_target), dtype=object
        ),
        "transshipment_feature_mode": np.array(
            str(cfg.transshipment_feature_mode), dtype=object
        ),
        "gap_seconds": np.array(int(cfg.gap_seconds), dtype=np.int64),
        "seq_len": np.array(int(cfg.seq_len), dtype=np.int64),
        "stride": np.array(int(cfg.stride), dtype=np.int64),
        "use_operational_filter": np.array(bool(cfg.use_operational_filter)),
        "op_speed_min": np.array(float(cfg.op_speed_min), dtype=np.float32),
        "op_speed_max": np.array(float(cfg.op_speed_max), dtype=np.float32),
        "use_location_features": np.array(bool(cfg.use_location_features)),
        "label_map": np.array(list(label_map.items()), dtype=object),
        "scaled": np.array(False),
    }
    if task == "godark":
        save_payload.update(
            {
                "window_source_labels": window_source_labels.astype(object),
                "window_event_orders": window_event_orders.astype(np.int64),
                "window_position_strata": window_position_strata.astype(np.int64),
                "godark_sample_scope": np.array(
                    "one_observable_gap_context_window_per_event", dtype=object
                ),
                "godark_min_distance_from_shore_nm": np.array(
                    float(cfg.godark_min_distance_from_shore_nm),
                    dtype=np.float32,
                ),
                "godark_ping_window_seconds": np.array(
                    int(cfg.godark_ping_window_seconds), dtype=np.int64
                ),
                "godark_min_ping_count_prev_window": np.array(
                    int(cfg.godark_min_ping_count_prev_window), dtype=np.int64
                ),
                "godark_diversity_protocol": np.array(
                    "source_label_duration_cadence_distance_position_v2", dtype=object
                ),
            }
        )
    if task == "spoofing":
        present_sources = sorted(set(window_source_labels.astype(str).tolist()))
        invalid_sources = sorted(
            set(present_sources) - set(DEFAULT_SPOOFING_SOURCE_INCLUDE_LABELS)
        )
        if invalid_sources:
            raise RuntimeError(
                "Spoofing source lock violated; invalid sources="
                f"{invalid_sources}."
            )
        present_attacks = sorted(
            set(window_kinds.astype(str).tolist()) - {"normal"}
        )
        save_payload.update(
            {
                "window_source_labels": window_source_labels.astype(object),
                "spoofing_balance_strata": spoofing_balance_strata.astype(object),
                "spoofing_attack_types": np.asarray(
                    present_attacks, dtype=object
                ),
                "spoofing_source_labels": np.asarray(
                    present_sources, dtype=object
                ),
                "spoofing_data_protocol": np.array(
                    "four_gear_six_attack_context_hybrid_oof_v2", dtype=object
                ),
                "spoofing_context_feature_cols": np.asarray(
                    SPOOFING_CONTEXT_FEATURE_COLS, dtype=object
                ),
            }
        )
    if task == "transshipment":
        save_payload.update(
            {
                "window_source_labels": window_source_labels.astype(object),
                "window_is_synthetic": window_is_synthetic.astype(np.int8),
                "window_mmsi_a": window_mmsi_a.astype(object),
                "window_mmsi_b": window_mmsi_b.astype(object),
                "transshipment_data_protocol": np.array(
                    "synthetic_train_only_vessel_disjoint_external_real_v1",
                    dtype=object,
                ),
            }
        )
    np.savez_compressed(out_path, **save_payload)

    print(f"[preprocess] Saved: {out_path}")
    print(f"[preprocess] X={X.shape} y={y.shape} classes={len(set(y.tolist()))}")
    print("[preprocess] scaler will be fit on train split during train.")
    print("[preprocess] label_map:", label_map)
