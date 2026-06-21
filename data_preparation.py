from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from dataload import read_ais_csv, infer_label_from_filename


DEFAULT_SOURCE_EXCLUDE_LABELS = ("pole_and_line", "trollers")
DEFAULT_GEAR_EXCLUDE_LABELS = ("unknown", *DEFAULT_SOURCE_EXCLUDE_LABELS)


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
    if cfg.use_location_features:
        return list(SEQ_FEATURE_COLS)
    return [c for c in SEQ_FEATURE_COLS if c not in LOCATION_FEATURE_COLS]


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


def _mode_text(values: np.ndarray, default: str = "normal") -> str:
    vals = [str(v) for v in values.tolist() if str(v) and str(v).lower() not in {"normal", "nan", "none"}]
    if not vals:
        return default
    uniq, cnt = np.unique(np.asarray(vals, dtype=object), return_counts=True)
    return str(uniq[int(np.argmax(cnt))])


def _stable_random_state(key: str, seed: int = 42) -> np.random.RandomState:
    h = 2166136261
    for ch in str(key):
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return np.random.RandomState((h + int(seed)) & 0xFFFFFFFF)


def _is_godark_hard_negative(window: np.ndarray, feat_cols: List[str], cfg: PreprocessCfg) -> bool:
    if window.size == 0:
        return False
    col = {name: i for i, name in enumerate(feat_cols)}
    gap_thr = max(1800.0, min(float(cfg.gap_seconds), 24.0 * 3600.0))

    dt_idx = col.get("dt_raw_seconds")
    if dt_idx is not None and float(np.nanmax(window[:, dt_idx])) >= gap_thr:
        return True

    step_idx = col.get("step_km_raw")
    if step_idx is not None and float(np.nanmax(window[:, step_idx])) >= 5.0:
        return True

    implied_idx = col.get("implied_speed_knots_raw")
    if implied_idx is not None and float(np.nanmax(window[:, implied_idx])) >= 20.0:
        return True

    return False


def _select_godark_windows(
    candidates: List[dict],
    max_windows: int,
    rng: np.random.RandomState,
) -> List[dict]:
    if max_windows <= 0 or len(candidates) <= int(max_windows):
        return candidates

    positives = [c for c in candidates if int(c["y"]) == 1]
    hard_feature = [c for c in candidates if int(c["y"]) == 0 and str(c.get("kind", "")).startswith("hard_negative_feature")]
    hard_gap = [c for c in candidates if int(c["y"]) == 0 and str(c.get("kind", "")).startswith("hard_negative_gap")]
    hard_other = [
        c for c in candidates
        if int(c["y"]) == 0
        and str(c.get("kind", "")).startswith("hard_negative")
        and not str(c.get("kind", "")).startswith(("hard_negative_feature", "hard_negative_gap"))
    ]
    normal = [c for c in candidates if int(c["y"]) == 0 and not str(c.get("kind", "")).startswith("hard_negative")]

    keep: List[dict] = []
    keep.extend(positives)

    remaining = int(max_windows) - len(keep)
    if remaining <= 0:
        idx = rng.choice(len(positives), size=int(max_windows), replace=False)
        return [positives[int(i)] for i in sorted(idx.tolist(), key=lambda j: positives[j]["order"])]

    hard_limit = max(remaining // 2, min(remaining, len(positives) * 3 if positives else remaining // 2))
    hard_limit = min(remaining, hard_limit)
    if hard_limit > 0:
        feature_limit = min(len(hard_feature), max(1, int(round(hard_limit * 0.70))))
        if feature_limit > 0:
            idx = (
                rng.choice(len(hard_feature), size=feature_limit, replace=False)
                if len(hard_feature) > feature_limit else np.arange(len(hard_feature))
            )
            keep.extend([hard_feature[int(i)] for i in idx.tolist()])

        hard_remaining = int(max(0, hard_limit - feature_limit))
        gap_pool = hard_gap + hard_other
        if hard_remaining > 0 and gap_pool:
            idx = (
                rng.choice(len(gap_pool), size=min(hard_remaining, len(gap_pool)), replace=False)
                if len(gap_pool) > hard_remaining else np.arange(len(gap_pool))
            )
            keep.extend([gap_pool[int(i)] for i in idx.tolist()])

    remaining = int(max_windows) - len(keep)
    if remaining > 0 and normal:
        idx = rng.choice(len(normal), size=min(remaining, len(normal)), replace=False)
        keep.extend([normal[int(i)] for i in idx.tolist()])

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


def _append_godark_candidate(
    out: List[dict],
    *,
    order: int,
    feat: np.ndarray,
    coords: np.ndarray,
    y_vals: np.ndarray,
    event_ids: np.ndarray,
    kind: str,
    vessel_id: str,
) -> None:
    y_win = int(np.max(y_vals) >= 1)
    if y_win:
        event_id = _mode_text(event_ids[y_vals > 0], default=f"normal::{vessel_id}")
    elif str(kind).startswith("hard_negative"):
        event_id = f"{kind}::{vessel_id}"
    else:
        event_id = f"normal::{vessel_id}"
    if y_win:
        kind = "positive_event"
    out.append(
        {
            "order": int(order),
            "X": feat.astype(np.float32, copy=False),
            "coords": coords.astype(np.float64, copy=False),
            "y": int(y_win),
            "event_id": str(event_id),
            "kind": str(kind),
        }
    )


def build_transshipment_sequences_from_df(
    df: pd.DataFrame,
    cfg: PreprocessCfg,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feat_cols = _transshipment_feature_cols(cfg.transshipment_feature_mode)
    need = ["event_id", "timestamp", "lat_mid", "lon_mid"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"task=transshipment but required column '{c}' not found")

    df = df.copy()
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
    rule_df = df[TRANS_RULE_SCORE_COLS].copy()
    df = _stabilize_transshipment_features(df)
    df = df.sort_values(["event_id", "timestamp"])
    rule_df = rule_df.loc[df.index].fillna(0.0)

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    g_list: List[str] = []
    coord_list: List[np.ndarray] = []
    rule_list: List[np.ndarray] = []

    min_points = max(1, int(cfg.min_points_per_vessel))
    seq_len = int(cfg.seq_len)
    stride = max(1, int(cfg.stride))

    for event_id, g in df.groupby("event_id", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        if len(g) < min_points:
            continue

        rule_g = rule_df.loc[g.index, TRANS_RULE_SCORE_COLS].to_numpy(dtype=np.float32)
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
        )

    return (
        np.stack(X_list).astype(np.float32),
        np.array(y_list, dtype=np.int64),
        np.array(g_list, dtype=object),
        np.stack(coord_list).astype(np.float64),
        np.stack(rule_list).astype(np.float32),
    )


def build_sequences_from_df(
    df: pd.DataFrame,
    cfg: PreprocessCfg,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

        # Biasanya sequence dipotong ketika ada gap besar.
        # Untuk go-dark, gap besar yang memang diberi label GoDark tidak boleh dipotong.
        # Alasannya, model perlu melihat titik sebelum blackout dan titik reappearance.
        raw_gaps = np.where((ts[1:] - ts[:-1]) > cfg.gap_seconds)[0]

        if cfg.task == "godark" and len(raw_gaps) > 0:
            event_ids = (
                g.get("go_dark_event_id", pd.Series("normal", index=g.index))
                .astype(str)
                .to_numpy()
            )

            keep_gap = []

            for x in raw_gaps:
                left_is_event = y_point[x] == 1
                right_is_event = y_point[x + 1] == 1
                same_event = (
                    (event_ids[x] != "normal")
                    and (event_ids[x] == event_ids[x + 1])
                )

                # keep_gap=True artinya gap ini adalah synthetic go-dark.
                # Jadi gap ini tidak dipakai untuk memecah sequence.
                keep_gap.append(bool(left_is_event or right_is_event or same_event))

            gaps = np.array(
                [x for x, keep in zip(raw_gaps, keep_gap) if not keep],
                dtype=int,
            )
        else:
            gaps = raw_gaps

        starts = [0] + [int(x + 1) for x in gaps]
        ends = [int(x + 1) for x in gaps] + [len(g)]

        feat = g[feat_cols].to_numpy(dtype=np.float32)
        coords = g[["timestamp", "lat", "lon", "y_point"]].to_numpy(dtype=np.float64)
        split_group_id = str(vessel_id)
        spoof_scenario_ids = np.array([str(vessel_id)] * len(g), dtype=object)
        spoof_attack_types = np.array(["normal"] * len(g), dtype=object)
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
        event_ids = (
            g.get("go_dark_event_id", pd.Series("normal", index=g.index))
            .astype(str)
            .to_numpy()
        )

        if cfg.task == "godark":
            keep_gap = []
            for x in raw_gaps:
                left_is_event = y_point[x] == 1
                right_is_event = y_point[x + 1] == 1
                same_event = (
                    (event_ids[x] != "normal")
                    and (event_ids[x] == event_ids[x + 1])
                )
                keep_gap.append(bool(left_is_event or right_is_event or same_event))

            gaps = np.array(
                [x for x, keep in zip(raw_gaps, keep_gap) if not keep],
                dtype=int,
            )
            starts = [0] + [int(x + 1) for x in gaps]
            ends = [int(x + 1) for x in gaps] + [len(g)]

            candidates: List[dict] = []
            order = 0
            for s, e in zip(starts, ends):
                if e - s < cfg.seq_len:
                    continue
                for i in range(s, e - cfg.seq_len + 1, cfg.stride):
                    window = feat[i:i + cfg.seq_len]
                    coord_window = coords[i:i + cfg.seq_len]
                    y_vals = y_point[i:i + cfg.seq_len]
                    event_vals = event_ids[i:i + cfg.seq_len]
                    kind = "hard_negative_feature" if (
                        int(np.max(y_vals) >= 1) == 0 and _is_godark_hard_negative(window, feat_cols, cfg)
                    ) else "normal_random"
                    _append_godark_candidate(
                        candidates,
                        order=order,
                        feat=window,
                        coords=coord_window,
                        y_vals=y_vals,
                        event_ids=event_vals,
                        kind=kind,
                        vessel_id=str(vessel_id),
                    )
                    order += 1

            non_event_gap_idx = [int(x) for x, keep in zip(raw_gaps, keep_gap) if not keep]
            for gap_no, x in enumerate(non_event_gap_idx):
                lo = max(0, int(x) - cfg.seq_len + 2)
                hi = min(int(x) + 1, len(g) - cfg.seq_len)
                if hi < lo:
                    continue
                starts_gap = list(range(lo, hi + 1, max(1, cfg.stride)))
                if hi not in starts_gap:
                    starts_gap.append(hi)
                for i in starts_gap:
                    window = feat[i:i + cfg.seq_len]
                    coord_window = coords[i:i + cfg.seq_len]
                    y_vals = y_point[i:i + cfg.seq_len]
                    if int(np.max(y_vals) >= 1) == 1:
                        continue
                    event_vals = event_ids[i:i + cfg.seq_len]
                    _append_godark_candidate(
                        candidates,
                        order=order,
                        feat=window,
                        coords=coord_window,
                        y_vals=y_vals,
                        event_ids=event_vals,
                        kind=f"hard_negative_gap::{gap_no}",
                        vessel_id=str(vessel_id),
                    )
                    order += 1

            rng = _stable_random_state(str(vessel_id))
            selected = _select_godark_windows(candidates, int(cfg.max_windows_per_vessel), rng)
            for item in selected:
                X_list.append(item["X"])
                y_list.append(int(item["y"]))
                g_list.append(vessel_id)
                coord_list.append(item["coords"])
                event_id_list.append(str(item["event_id"]))
                kind_list.append(str(item["kind"]))
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
                else:
                    event_id_list.append(str(vessel_id))
                    kind_list.append(
                        "positive_event" if y_win else "normal_random"
                    )
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
        )

    return (
        np.stack(X_list),
        np.array(y_list, dtype=np.int64),
        np.array(g_list, dtype=object),
        np.stack(coord_list).astype(np.float64),
        np.array(event_id_list, dtype=object),
        np.array(kind_list, dtype=object),
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
    transshipment_target: str = "multiclass",
    transshipment_feature_mode: str = "fair",
    apply_jump_filter: Optional[bool] = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if apply_jump_filter is None:
        apply_jump_filter = task not in ["spoofing", "godark", "transshipment"]

    if str(task) == "spoofing" and bool(use_location_features):
        print(
            "[preprocess] WARNING: disabling distance_from_shore and "
            "distance_from_port for spoofing. Synthetic coordinate attacks do "
            "not have recomputed geospatial distance rasters."
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
    all_window_event_ids, all_window_kinds = [], []
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

    for p in tqdm(csvs, desc="files"):
        stem = infer_label_from_filename(p)

        if stem in exclude_labels:
            continue

        df = read_ais_csv(p, limit_rows=limit_rows, chunksize=chunksize)

        try:
            if task == "transshipment":
                X, y, groups, coords, rule_features = build_transshipment_sequences_from_df(df, cfg)
                window_event_ids = groups.astype(object)
                window_kinds = np.array(["positive_event" if int(v) > 0 else "normal_random" for v in y], dtype=object)
            else:
                X, y, groups, coords, window_event_ids, window_kinds = build_sequences_from_df(df, cfg)
                rule_features = None
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
            if rule_features is not None:
                rule_features = rule_features[idx]

        all_X.append(X)
        all_y.append(y)
        all_groups.append(groups)
        all_coords.append(coords)
        all_window_event_ids.append(window_event_ids)
        all_window_kinds.append(window_kinds)
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

    np.savez_compressed(
        out_path,
        X=X.astype(np.float32),
        y=y,
        groups=groups,
        coords=coords,
        coord_cols=np.array(["timestamp", "lat", "lon", "y_point"], dtype=object),
        window_event_ids=window_event_ids.astype(object),
        window_kinds=window_kinds.astype(object),
        feature_cols=np.array(
            _transshipment_feature_cols(cfg.transshipment_feature_mode)
            if task == "transshipment"
            else _sequence_feature_cols(cfg),
            dtype=object,
        ),
        rule_features=rule_features.astype(np.float32),
        rule_cols=np.array(TRANS_RULE_SCORE_COLS if task == "transshipment" else [], dtype=object),
        transshipment_target=np.array(str(cfg.transshipment_target), dtype=object),
        transshipment_feature_mode=np.array(str(cfg.transshipment_feature_mode), dtype=object),
        gap_seconds=np.array(int(cfg.gap_seconds), dtype=np.int64),
        seq_len=np.array(int(cfg.seq_len), dtype=np.int64),
        stride=np.array(int(cfg.stride), dtype=np.int64),
        use_operational_filter=np.array(bool(cfg.use_operational_filter)),
        op_speed_min=np.array(float(cfg.op_speed_min), dtype=np.float32),
        op_speed_max=np.array(float(cfg.op_speed_max), dtype=np.float32),
        use_location_features=np.array(bool(cfg.use_location_features)),
        label_map=np.array(list(label_map.items()), dtype=object),
        scaled=np.array(False),
    )

    print(f"[preprocess] Saved: {out_path}")
    print(f"[preprocess] X={X.shape} y={y.shape} classes={len(set(y.tolist()))}")
    print("[preprocess] scaler will be fit on train split during train.")
    print("[preprocess] label_map:", label_map)
