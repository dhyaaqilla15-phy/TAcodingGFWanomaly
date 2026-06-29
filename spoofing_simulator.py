from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

from dataload import read_ais_csv
from data_preparation import (
    DEFAULT_SOURCE_EXCLUDE_LABELS,
    DEFAULT_SPOOFING_SOURCE_INCLUDE_LABELS,
    bearing_deg_np,
    haversine_km_np,
)


DEFAULT_ATTACKS = ["gradual_drift", "location_jump", "replay", "meaconing", "ghost", "mirroring"]
DEFAULT_DETECTABLE_ATTACKS = ["gradual_drift", "location_jump"]
CONTEXT_REQUIRED_ATTACKS = {"replay", "meaconing", "ghost", "mirroring"}


@dataclass
class SpoofingSimCfg:
    """
    Konfigurasi generator spoofing.

    Desainnya mengikuti ide paper: data AIS asli dibersihkan, lalu disisipkan titik
    manipulasi GPS dengan label Normal/Spoofed. Output CSV tetap membawa kolom
    asli (mmsi, timestamp, lat, lon, speed, course, dst.) supaya pipeline lama masih
    bisa dipakai.
    """

    attacks: List[str] = field(default_factory=lambda: DEFAULT_DETECTABLE_ATTACKS.copy())
    seed: int = 42

    # sampling agar aman untuk dataset besar
    limit_rows: int = 0
    chunksize: int = 0
    sample_frac: float = 0.0
    normal_keep_frac: float = 1.0
    include_labels: Sequence[str] = DEFAULT_SPOOFING_SOURCE_INCLUDE_LABELS
    exclude_labels: Sequence[str] = DEFAULT_SOURCE_EXCLUDE_LABELS

    # banyaknya vessel/points yang dimanipulasi
    max_vessels_per_file: int = 20
    min_points_per_vessel: int = 80
    points_per_attack: int = 120
    scenarios_per_attack: int = 1
    max_attack_gap_seconds: int = 3 * 3600

    # parameter spoofing
    drift_lat_deg: float = 0.08
    drift_lon_deg: float = 0.08
    # When positive, control gradual-drift severity by physical rate instead
    # of a fixed total coordinate offset. This keeps internal and external
    # scenarios comparable when their segment durations differ.
    drift_rate_kmh: float = 0.0
    drift_rate_jitter_frac: float = 0.25
    jump_lat_deg: float = 0.70
    jump_lon_deg: float = 0.70
    replay_delay_seconds: int = 6 * 3600
    meacon_lag_steps: int = 8
    ghost_offset_min_deg: float = 1.5
    ghost_offset_max_deg: float = 8.0
    mirror_offset_min_deg: float = 1.5
    mirror_offset_max_deg: float = 8.0
    # AIS position spoofing does not imply that reported SOG/COG are always
    # altered. Preserve and recompute are both plausible threat-model variants.
    reported_motion_mode: str = "preserve"
    mixed_recompute_probability: float = 0.50
    include_matched_normal_controls: bool = False

    # output
    combine_outputs: bool = False


def _as_list(attacks: Sequence[str] | str | None) -> List[str]:
    if attacks is None:
        return DEFAULT_DETECTABLE_ATTACKS.copy()
    if isinstance(attacks, str):
        attacks = [a.strip() for a in attacks.replace(",", " ").split() if a.strip()]
    out = []
    for a in attacks:
        a = str(a).strip().lower()
        if a:
            out.append(a)
    bad = sorted(set(out) - set(DEFAULT_ATTACKS))
    if bad:
        raise ValueError(f"Unknown attack(s): {bad}. Pilihan: {DEFAULT_ATTACKS}")
    return out or DEFAULT_DETECTABLE_ATTACKS.copy()


def _stable_seed(base_seed: int, text: str) -> int:
    digest = hashlib.sha256(str(text).encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return int((int(base_seed) + offset) % (2**32 - 1))


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
        available = {path.stem.strip().lower() for path in csvs}
        missing = sorted(included - available)
        if missing:
            raise FileNotFoundError(
                f"Required spoofing source CSVs missing from {input_path}: {missing}"
            )
    else:
        csvs = [input_path]

    if included:
        skipped = [p for p in csvs if p.stem.strip().lower() not in included]
        csvs = [p for p in csvs if p.stem.strip().lower() in included]
        if skipped:
            names = ", ".join(p.name for p in skipped)
            print(f"[spoof] include source labels={sorted(included)} skipped={names}")

    if excluded:
        skipped = [p for p in csvs if p.stem.strip().lower() in excluded]
        csvs = [p for p in csvs if p.stem.strip().lower() not in excluded]
        if skipped:
            names = ", ".join(p.name for p in skipped)
            print(f"[spoof] exclude source labels={sorted(excluded)} skipped={names}")
    return csvs


def _wrap_lon(lon: np.ndarray) -> np.ndarray:
    return ((lon + 180.0) % 360.0) - 180.0


def _clip_lat(lat: np.ndarray) -> np.ndarray:
    return np.clip(lat, -89.999, 89.999)


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
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1.0 if c != "speed" and c != "course" else 0.0)

    df["speed"] = df["speed"].clip(0.0, 50.0)
    df["course"] = df["course"].fillna(0.0) % 360.0

    df = df.sort_values(["mmsi", "timestamp"]).drop_duplicates(["mmsi", "timestamp"], keep="last")
    return df.reset_index(drop=True)


def _read_input_csv(path: Path, cfg: SpoofingSimCfg) -> pd.DataFrame:
    if cfg.chunksize and cfg.chunksize > 0:
        df = read_ais_csv(path, limit_rows=cfg.limit_rows, chunksize=cfg.chunksize)
    else:
        df = read_ais_csv(path, limit_rows=cfg.limit_rows, chunksize=0)

    df = _prep_base_df(df)

    if cfg.sample_frac and 0.0 < cfg.sample_frac < 1.0:
        vessels = np.sort(df["mmsi"].unique())
        keep_count = max(1, int(round(len(vessels) * float(cfg.sample_frac))))
        rng = np.random.RandomState(_stable_seed(int(cfg.seed), path.stem))
        keep_vessels = rng.choice(
            vessels,
            size=min(keep_count, len(vessels)),
            replace=False,
        )
        # Sample complete vessels, not random rows. Random row sampling destroys
        # temporal continuity and creates artificial gaps in normal tracks.
        df = df[df["mmsi"].isin(keep_vessels)].copy()
        df = df.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)

    return df


def _choose_vessels(df: pd.DataFrame, cfg: SpoofingSimCfg, rng: np.random.RandomState) -> List[int]:
    counts = df.groupby("mmsi").size()
    eligible = counts[counts >= int(cfg.min_points_per_vessel)].index.to_numpy(dtype=np.int64)
    if eligible.size == 0:
        return []
    max_v = int(cfg.max_vessels_per_file)
    if max_v > 0 and eligible.size > max_v:
        eligible = rng.choice(eligible, size=max_v, replace=False)
    return [int(x) for x in eligible.tolist()]


def _segment(
    g: pd.DataFrame,
    n_points: int,
    rng: np.random.RandomState,
    extra: int = 0,
    max_gap_seconds: int = 3 * 3600,
) -> pd.DataFrame:
    g = g.sort_values("timestamp").reset_index(drop=True)
    need = int(n_points) + int(extra)
    if len(g) < need:
        return pd.DataFrame(columns=g.columns)
    ts = g["timestamp"].to_numpy(dtype=np.int64)
    gap = np.diff(ts)
    breaks = np.where((gap <= 0) | (gap > int(max_gap_seconds)))[0] + 1
    starts = np.r_[0, breaks]
    ends = np.r_[breaks, len(g)]
    candidates: list[tuple[int, int]] = []
    total_starts = 0
    for start, end in zip(starts, ends):
        count = int(end - start - need + 1)
        if count > 0:
            candidates.append((int(start), count))
            total_starts += count
    if total_starts <= 0:
        return pd.DataFrame(columns=g.columns)
    selected = int(rng.randint(0, total_starts))
    for run_start, count in candidates:
        if selected < count:
            start = run_start + selected
            return g.iloc[start:start + need].copy().reset_index(drop=True)
        selected -= count
    raise RuntimeError("Failed to select a contiguous spoofing segment.")


def _recompute_speed_course(seg: pd.DataFrame) -> pd.DataFrame:
    """Hitung ulang speed/course dari lat/lon/timestamp agar manipulasi tetap konsisten."""
    seg = seg.sort_values(["mmsi", "timestamp"]).copy()
    seg["speed"] = pd.to_numeric(seg["speed"], errors="coerce").fillna(0.0).astype(float)
    seg["course"] = pd.to_numeric(seg["course"], errors="coerce").fillna(0.0).astype(float)
    for _, idx in seg.groupby("mmsi", sort=False).groups.items():
        idx = np.array(list(idx), dtype=object)
        if len(idx) < 2:
            continue
        lat = seg.loc[idx, "lat"].to_numpy(dtype=float)
        lon = seg.loc[idx, "lon"].to_numpy(dtype=float)
        ts = seg.loc[idx, "timestamp"].to_numpy(dtype=np.int64)

        dt = np.diff(ts).astype(float)
        dt = np.where(dt <= 0, 1.0, dt)
        km = haversine_km_np(lat[:-1], lon[:-1], lat[1:], lon[1:])
        speed_knots = (km / dt) * (3600.0 / 1.852)
        bearing = bearing_deg_np(lat[:-1], lon[:-1], lat[1:], lon[1:])

        new_speed = seg.loc[idx, "speed"].to_numpy(dtype=float, copy=True)
        new_course = seg.loc[idx, "course"].to_numpy(dtype=float, copy=True)
        new_speed[1:] = np.clip(speed_knots, 0.0, 50.0)
        new_course[1:] = bearing % 360.0
        if len(new_speed) > 1:
            new_speed[0] = new_speed[1]
            new_course[0] = new_course[1]
        seg.loc[idx, "speed"] = new_speed
        seg.loc[idx, "course"] = new_course
    return seg


def _finish_attack(
    seg: pd.DataFrame,
    attack: str,
    original_mmsi: int,
    scenario_id: str,
    scenario_mmsi: int,
    spoof_mask: np.ndarray | None = None,
    event_mask: np.ndarray | None = None,
    magnitude: dict[str, float] | None = None,
    reported_motion_mode: str = "preserve",
) -> pd.DataFrame:
    if seg.empty:
        return seg
    # Every synthetic scenario needs its own trajectory ID. Reusing the source
    # MMSI and timestamp would make preprocess drop either the normal or
    # spoofed row as a duplicate.
    seg["mmsi"] = int(scenario_mmsi)
    motion_mode = str(reported_motion_mode).strip().lower()
    if motion_mode not in {"preserve", "recompute"}:
        raise ValueError(f"Unsupported reported_motion_mode: {reported_motion_mode}")
    if motion_mode == "recompute":
        seg = _recompute_speed_course(seg)
    if spoof_mask is None:
        spoof_mask = np.ones(len(seg), dtype=bool)
    spoof_mask = np.asarray(spoof_mask, dtype=bool)
    if spoof_mask.shape[0] != len(seg):
        raise ValueError("spoof_mask length must match attack segment length.")
    if event_mask is None:
        event_mask = spoof_mask
    event_mask = np.asarray(event_mask, dtype=bool)
    if event_mask.shape[0] != len(seg):
        raise ValueError("event_mask length must match attack segment length.")
    seg["is_spoofing"] = spoof_mask.astype("int8")
    seg["is_spoofing_event"] = event_mask.astype("int8")
    seg["label"] = np.where(spoof_mask, "Spoofed", "Normal")
    seg["attack_type"] = attack
    seg["original_mmsi"] = str(original_mmsi)
    # ``mmsi`` is a unique synthetic trajectory key so scenarios cannot
    # overwrite one another during preprocessing. ``claimed_mmsi`` represents
    # the identity visible to a context detector. Replay, meaconing, and
    # mirroring impersonate the source identity; a ghost uses an unregistered
    # identity.
    seg["claimed_mmsi"] = str(
        scenario_mmsi if attack == "ghost" else original_mmsi
    )
    seg["scenario_id"] = scenario_id
    seg["note"] = f"synthetic_{attack}"
    seg["identifiability"] = (
        "context_required"
        if attack in CONTEXT_REQUIRED_ATTACKS
        else "single_window_kinematic"
    )
    seg["reported_motion_mode"] = motion_mode
    seg["attack_points"] = int(len(seg))
    seg["attack_duration_hours"] = float(
        max(
            0.0,
            (float(seg["timestamp"].max()) - float(seg["timestamp"].min()))
            / 3600.0,
        )
    )
    for key, value in (magnitude or {}).items():
        seg[key] = float(value)
    displacement_km = float(seg.get("attack_displacement_km", pd.Series([0.0])).iloc[0])
    duration_hours = float(seg["attack_duration_hours"].iloc[0])
    seg["attack_drift_rate_kmh"] = (
        displacement_km / duration_hours
        if attack == "gradual_drift" and duration_hours > 0.0
        else 0.0
    )
    return seg


def _signed_offset(rng: np.random.RandomState, value: float) -> float:
    sign = -1.0 if rng.rand() < 0.5 else 1.0
    mag = float(value) * (0.65 + 0.70 * rng.rand())
    return sign * mag


def _signed_uniform_offset(rng: np.random.RandomState, min_value: float, max_value: float) -> float:
    min_value = float(max(0.0, min_value))
    max_value = float(max(max_value, min_value))
    sign = -1.0 if rng.rand() < 0.5 else 1.0
    return sign * float(rng.uniform(min_value, max_value))


def _select_reported_motion_mode(
    cfg: SpoofingSimCfg,
    rng: np.random.RandomState,
) -> str:
    mode = str(cfg.reported_motion_mode).strip().lower()
    if mode in {"preserve", "recompute"}:
        return mode
    if mode == "mixed":
        probability = float(np.clip(cfg.mixed_recompute_probability, 0.0, 1.0))
        return "recompute" if float(rng.rand()) < probability else "preserve"
    raise ValueError(
        "reported_motion_mode must be one of: preserve, recompute, mixed; "
        f"got {cfg.reported_motion_mode!r}"
    )


def _destination_lat_lon(
    lat_deg: float,
    lon_deg: float,
    distance_km: float,
    bearing_rad: float,
) -> tuple[float, float]:
    """Spherical destination used to turn a drift rate into an endpoint."""
    radius_km = 6371.0088
    angular = max(0.0, float(distance_km)) / radius_km
    lat1 = np.deg2rad(float(lat_deg))
    lon1 = np.deg2rad(float(lon_deg))
    lat2 = np.arcsin(
        np.sin(lat1) * np.cos(angular)
        + np.cos(lat1) * np.sin(angular) * np.cos(float(bearing_rad))
    )
    lon2 = lon1 + np.arctan2(
        np.sin(float(bearing_rad)) * np.sin(angular) * np.cos(lat1),
        np.cos(angular) - np.sin(lat1) * np.sin(lat2),
    )
    return float(np.rad2deg(lat2)), float(_wrap_lon(np.array([np.rad2deg(lon2)]))[0])


def _attack_gradual_drift(
    g: pd.DataFrame,
    vessel: int,
    cfg: SpoofingSimCfg,
    rng: np.random.RandomState,
    sid: str,
    scenario_mmsi: int,
) -> pd.DataFrame:
    seg = _segment(
        g,
        cfg.points_per_attack,
        rng,
        max_gap_seconds=cfg.max_attack_gap_seconds,
    )
    if seg.empty:
        return seg
    frac = np.linspace(0.0, 1.0, len(seg), dtype=float)
    original_end_lat = float(seg["lat"].iloc[-1])
    original_end_lon = float(seg["lon"].iloc[-1])
    duration_hours = max(
        0.0,
        (float(seg["timestamp"].iloc[-1]) - float(seg["timestamp"].iloc[0]))
        / 3600.0,
    )
    target_rate = max(0.0, float(cfg.drift_rate_kmh))
    applied_rate = 0.0
    if target_rate > 0.0 and duration_hours > 0.0:
        jitter = float(np.clip(cfg.drift_rate_jitter_frac, 0.0, 0.95))
        applied_rate = target_rate * float(rng.uniform(1.0 - jitter, 1.0 + jitter))
        target_distance_km = applied_rate * duration_hours
        end_lat, end_lon = _destination_lat_lon(
            original_end_lat,
            original_end_lon,
            target_distance_km,
            bearing_rad=float(rng.uniform(0.0, 2.0 * np.pi)),
        )
        lat_offset = end_lat - original_end_lat
        lon_offset = float(
            ((end_lon - original_end_lon + 180.0) % 360.0) - 180.0
        )
    else:
        lat_offset = _signed_offset(rng, cfg.drift_lat_deg)
        lon_offset = _signed_offset(rng, cfg.drift_lon_deg)
    seg["lat"] = _clip_lat(seg["lat"].to_numpy(dtype=float) + frac * lat_offset)
    seg["lon"] = _wrap_lon(seg["lon"].to_numpy(dtype=float) + frac * lon_offset)
    return _finish_attack(
        seg,
        "gradual_drift",
        vessel,
        sid,
        scenario_mmsi,
        spoof_mask=frac > 0.0,
        magnitude={
            "attack_nominal_lat_deg": cfg.drift_lat_deg,
            "attack_nominal_lon_deg": cfg.drift_lon_deg,
            "attack_applied_lat_deg": lat_offset,
            "attack_applied_lon_deg": lon_offset,
            "attack_target_drift_rate_kmh": target_rate,
            "attack_applied_drift_rate_kmh": applied_rate,
            "attack_displacement_km": float(
                haversine_km_np(
                    np.array([original_end_lat]),
                    np.array([original_end_lon]),
                    np.array([float(seg["lat"].iloc[-1])]),
                    np.array([float(seg["lon"].iloc[-1])]),
                )[0]
            ),
        },
        reported_motion_mode=_select_reported_motion_mode(cfg, rng),
    )


def _attack_location_jump(
    g: pd.DataFrame,
    vessel: int,
    cfg: SpoofingSimCfg,
    rng: np.random.RandomState,
    sid: str,
    scenario_mmsi: int,
) -> pd.DataFrame:
    seg = _segment(
        g,
        cfg.points_per_attack,
        rng,
        max_gap_seconds=cfg.max_attack_gap_seconds,
    )
    if seg.empty:
        return seg
    cut_lo = max(1, len(seg) // 3)
    cut_hi = max(cut_lo + 1, (len(seg) * 2) // 3)
    cut = int(rng.randint(cut_lo, cut_hi))
    lat = seg["lat"].to_numpy(dtype=float, copy=True)
    lon = seg["lon"].to_numpy(dtype=float, copy=True)
    original_cut_lat = float(lat[cut])
    original_cut_lon = float(lon[cut])
    lat_offset = _signed_offset(rng, cfg.jump_lat_deg)
    lon_offset = _signed_offset(rng, cfg.jump_lon_deg)
    lat[cut:] = _clip_lat(lat[cut:] + lat_offset)
    lon[cut:] = _wrap_lon(lon[cut:] + lon_offset)
    seg["lat"] = lat
    seg["lon"] = lon
    spoof_mask = np.zeros(len(seg), dtype=bool)
    spoof_mask[cut:] = True
    event_mask = np.zeros(len(seg), dtype=bool)
    event_mask[cut] = True
    return _finish_attack(
        seg,
        "location_jump",
        vessel,
        sid,
        scenario_mmsi,
        spoof_mask=spoof_mask,
        event_mask=event_mask,
        magnitude={
            "attack_nominal_lat_deg": cfg.jump_lat_deg,
            "attack_nominal_lon_deg": cfg.jump_lon_deg,
            "attack_applied_lat_deg": lat_offset,
            "attack_applied_lon_deg": lon_offset,
            "attack_displacement_km": float(
                haversine_km_np(
                    np.array([original_cut_lat]),
                    np.array([original_cut_lon]),
                    np.array([float(lat[cut])]),
                    np.array([float(lon[cut])]),
                )[0]
            ),
        },
        reported_motion_mode=_select_reported_motion_mode(cfg, rng),
    )


def _attack_replay(
    g: pd.DataFrame,
    vessel: int,
    cfg: SpoofingSimCfg,
    rng: np.random.RandomState,
    sid: str,
    scenario_mmsi: int,
) -> pd.DataFrame:
    seg = _segment(
        g,
        cfg.points_per_attack,
        rng,
        max_gap_seconds=cfg.max_attack_gap_seconds,
    )
    if seg.empty:
        return seg
    # Preserve the trusted source time so a paired unmodified control can be
    # reconstructed after the replay is moved into the future.
    seg["reference_timestamp"] = seg["timestamp"].astype("int64")
    start_new = int(g["timestamp"].max()) + int(cfg.replay_delay_seconds) + int(rng.randint(0, 1800))
    delta = start_new - int(seg["timestamp"].iloc[0])
    seg["timestamp"] = seg["timestamp"].astype("int64") + int(delta)
    return _finish_attack(
        seg,
        "replay",
        vessel,
        sid,
        scenario_mmsi,
        magnitude={
            "attack_replay_delay_seconds": float(delta),
            "attack_displacement_km": 0.0,
        },
        reported_motion_mode=_select_reported_motion_mode(cfg, rng),
    )


def _attack_meaconing(
    g: pd.DataFrame,
    vessel: int,
    cfg: SpoofingSimCfg,
    rng: np.random.RandomState,
    sid: str,
    scenario_mmsi: int,
) -> pd.DataFrame:
    lag = max(1, int(cfg.meacon_lag_steps))
    seg = _segment(
        g,
        cfg.points_per_attack,
        rng,
        extra=lag,
        max_gap_seconds=cfg.max_attack_gap_seconds,
    )
    if seg.empty or len(seg) <= lag:
        return pd.DataFrame(columns=g.columns)

    delayed = seg.iloc[lag:].copy().reset_index(drop=True)
    old = seg.iloc[:-lag].copy().reset_index(drop=True)
    # timestamp tetap timestamp saat ini, tapi posisi yang dibaca adalah posisi lama.
    current_lat = delayed["lat"].to_numpy(dtype=float, copy=True)
    current_lon = delayed["lon"].to_numpy(dtype=float, copy=True)
    delayed["lat"] = old["lat"].to_numpy(dtype=float)
    delayed["lon"] = old["lon"].to_numpy(dtype=float)
    if "speed" in delayed.columns:
        delayed["speed"] = old["speed"].to_numpy(dtype=float)
    if "course" in delayed.columns:
        delayed["course"] = old["course"].to_numpy(dtype=float)
    return _finish_attack(
        delayed,
        "meaconing",
        vessel,
        sid,
        scenario_mmsi,
        magnitude={
            "attack_meacon_lag_steps": float(lag),
            "attack_displacement_km": float(
                np.median(
                    haversine_km_np(
                        current_lat,
                        current_lon,
                        delayed["lat"].to_numpy(dtype=float),
                        delayed["lon"].to_numpy(dtype=float),
                    )
                )
            ),
        },
        reported_motion_mode=_select_reported_motion_mode(cfg, rng),
    )


def _attack_ghost(
    g: pd.DataFrame,
    vessel: int,
    cfg: SpoofingSimCfg,
    rng: np.random.RandomState,
    sid: str,
    scenario_mmsi: int,
) -> pd.DataFrame:
    seg = _segment(
        g,
        cfg.points_per_attack,
        rng,
        max_gap_seconds=cfg.max_attack_gap_seconds,
    )
    if seg.empty:
        return seg

    # Kapal palsu dibuat dari pola gerak yang masuk akal, lalu dipindahkan ke area lain.
    min_off = float(cfg.ghost_offset_min_deg)
    max_off = float(max(cfg.ghost_offset_max_deg, min_off))
    lat_off = _signed_offset(rng, rng.uniform(min_off, max_off))
    lon_off = _signed_offset(rng, rng.uniform(min_off, max_off))

    original_lat = seg["lat"].to_numpy(dtype=float, copy=True)
    original_lon = seg["lon"].to_numpy(dtype=float, copy=True)
    seg["lat"] = _clip_lat(original_lat + lat_off)
    seg["lon"] = _wrap_lon(original_lon + lon_off)
    seg["source"] = seg["source"].astype(str) + "_ghost"
    return _finish_attack(
        seg,
        "ghost",
        vessel,
        sid,
        scenario_mmsi,
        magnitude={
            "attack_displacement_km": float(
                np.median(
                    haversine_km_np(
                        original_lat,
                        original_lon,
                        seg["lat"].to_numpy(dtype=float),
                        seg["lon"].to_numpy(dtype=float),
                    )
                )
            )
        },
        reported_motion_mode=_select_reported_motion_mode(cfg, rng),
    )


def _attack_mirroring(
    g: pd.DataFrame,
    vessel: int,
    cfg: SpoofingSimCfg,
    rng: np.random.RandomState,
    sid: str,
    scenario_mmsi: int,
) -> pd.DataFrame:
    seg = _segment(
        g,
        cfg.points_per_attack,
        rng,
        max_gap_seconds=cfg.max_attack_gap_seconds,
    )
    if seg.empty:
        return seg

    lat_off = _signed_uniform_offset(rng, cfg.mirror_offset_min_deg, cfg.mirror_offset_max_deg)
    lon_off = _signed_uniform_offset(rng, cfg.mirror_offset_min_deg, cfg.mirror_offset_max_deg)

    original_lat = seg["lat"].to_numpy(dtype=float, copy=True)
    original_lon = seg["lon"].to_numpy(dtype=float, copy=True)
    seg["lat"] = _clip_lat(original_lat + lat_off)
    seg["lon"] = _wrap_lon(original_lon + lon_off)
    seg["source"] = seg["source"].astype(str) + "_mirroring"
    return _finish_attack(
        seg,
        "mirroring",
        vessel,
        sid,
        scenario_mmsi,
        magnitude={
            "attack_displacement_km": float(
                np.median(
                    haversine_km_np(
                        original_lat,
                        original_lon,
                        seg["lat"].to_numpy(dtype=float),
                        seg["lon"].to_numpy(dtype=float),
                    )
                )
            )
        },
        reported_motion_mode=_select_reported_motion_mode(cfg, rng),
    )


def _sample_contiguous_normal(
    df: pd.DataFrame,
    keep_frac: float,
    seed: int,
) -> pd.DataFrame:
    frac = float(keep_frac)
    if frac >= 1.0:
        return df.copy()
    if frac <= 0.0:
        return df.iloc[0:0].copy()

    parts = []
    for vessel, group in df.groupby("mmsi", sort=False):
        group = group.sort_values("timestamp")
        keep = max(1, int(round(len(group) * frac)))
        if keep >= len(group):
            parts.append(group)
            continue
        rng = np.random.RandomState(_stable_seed(seed, f"normal::{vessel}"))
        start = int(rng.randint(0, len(group) - keep + 1))
        parts.append(group.iloc[start:start + keep])
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()


def _matched_normal_control(
    source_track: pd.DataFrame,
    attacked: pd.DataFrame,
    *,
    original_mmsi: int,
    scenario_id: str,
    scenario_mmsi: int,
    attack_type: str,
) -> pd.DataFrame:
    """Return the exact unmodified source segment paired with one attack."""
    timestamp_column = (
        "reference_timestamp"
        if "reference_timestamp" in attacked.columns
        else "timestamp"
    )
    timestamps = attacked[timestamp_column].to_numpy(dtype=np.int64)
    source = source_track.drop_duplicates("timestamp", keep="last").set_index("timestamp")
    missing = [int(ts) for ts in timestamps if int(ts) not in source.index]
    if missing:
        raise RuntimeError(
            f"Cannot build matched normal control; missing timestamps={missing[:5]}"
        )
    control = source.loc[timestamps].reset_index().copy()
    if len(control) != len(attacked):
        raise RuntimeError("Matched normal control length differs from attack segment.")
    control["mmsi"] = int(scenario_mmsi)
    control["is_spoofing"] = 0
    control["is_spoofing_event"] = 0
    control["label"] = "Normal"
    control["attack_type"] = "normal"
    control["original_mmsi"] = str(original_mmsi)
    control["claimed_mmsi"] = str(original_mmsi)
    control["scenario_id"] = str(scenario_id)
    control["note"] = "matched_unmodified_control"
    control["identifiability"] = "normal"
    control["reported_motion_mode"] = "original_matched"
    control["normal_control_for_attack"] = str(attack_type)
    control["attack_nominal_lat_deg"] = 0.0
    control["attack_nominal_lon_deg"] = 0.0
    control["attack_applied_lat_deg"] = 0.0
    control["attack_applied_lon_deg"] = 0.0
    control["attack_displacement_km"] = 0.0
    control["attack_points"] = 0
    control["attack_duration_hours"] = float(
        max(0.0, (float(timestamps.max()) - float(timestamps.min())) / 3600.0)
    )
    control["attack_drift_rate_kmh"] = 0.0
    control["attack_target_drift_rate_kmh"] = 0.0
    control["attack_applied_drift_rate_kmh"] = 0.0
    return control


def generate_spoofing_for_file(csv_path: Path, out_dir: Path, cfg: SpoofingSimCfg) -> Path:
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = out_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(_stable_seed(int(cfg.seed), csv_path.stem))
    attacks = _as_list(cfg.attacks)
    context_attacks = sorted(set(attacks) & CONTEXT_REQUIRED_ATTACKS)
    if context_attacks:
        print(
            "[spoof] WARNING: context-required attacks requested: "
            f"{context_attacks}. They are not identifiable reliably from an "
            "isolated kinematic window and must be reported separately."
        )

    print(f"[spoof] read: {csv_path}")
    df = _read_input_csv(csv_path, cfg)
    if df.empty:
        raise RuntimeError(f"No valid rows in {csv_path}")
    source_label = csv_path.stem.strip().lower()
    if source_label not in _label_set(cfg.include_labels):
        raise RuntimeError(
            f"Spoofing source {source_label!r} is outside locked sources "
            f"{sorted(_label_set(cfg.include_labels))}."
        )
    df["source_label"] = source_label

    normal = _sample_contiguous_normal(
        df,
        keep_frac=float(cfg.normal_keep_frac),
        seed=int(cfg.seed),
    )

    normal["is_spoofing"] = 0
    normal["is_spoofing_event"] = 0
    normal["label"] = "Normal"
    normal["attack_type"] = "normal"
    normal["original_mmsi"] = normal["mmsi"].astype(str)
    normal["claimed_mmsi"] = normal["mmsi"].astype(str)
    normal["scenario_id"] = "normal::" + normal["original_mmsi"]
    normal["note"] = "original_ais"
    normal["identifiability"] = "normal"
    normal["reported_motion_mode"] = "original"
    normal["normal_control_for_attack"] = ""
    normal["attack_nominal_lat_deg"] = 0.0
    normal["attack_nominal_lon_deg"] = 0.0
    normal["attack_applied_lat_deg"] = 0.0
    normal["attack_applied_lon_deg"] = 0.0
    normal["attack_displacement_km"] = 0.0
    normal["attack_points"] = 0
    normal["attack_duration_hours"] = 0.0
    normal["attack_drift_rate_kmh"] = 0.0
    normal["attack_target_drift_rate_kmh"] = 0.0
    normal["attack_applied_drift_rate_kmh"] = 0.0

    vessels = _choose_vessels(df, cfg, rng)
    if not vessels:
        raise RuntimeError(
            f"No eligible vessel with >= {cfg.min_points_per_vessel} points. "
            "Coba kecilkan --min_points_per_vessel atau naikkan --limit_rows."
        )

    spoof_parts = []
    scenario_base = 900_000_000_000_000 + int(rng.randint(0, 50_000_000))
    scenario_no = 0

    scenarios_per_attack = max(1, int(cfg.scenarios_per_attack))
    for vessel in vessels:
        g = df[df["mmsi"] == int(vessel)].sort_values("timestamp").reset_index(drop=True)
        for attack in attacks:
            configured_mode = str(cfg.reported_motion_mode).strip().lower()
            if configured_mode == "mixed":
                probability = float(
                    np.clip(cfg.mixed_recompute_probability, 0.0, 1.0)
                )
                recompute_count = int(round(scenarios_per_attack * probability))
                recompute_count = int(
                    np.clip(recompute_count, 0, scenarios_per_attack)
                )
                scenario_modes = (
                    ["recompute"] * recompute_count
                    + ["preserve"] * (scenarios_per_attack - recompute_count)
                )
                rng.shuffle(scenario_modes)
            else:
                scenario_modes = [configured_mode] * scenarios_per_attack

            for replicate, scenario_mode in enumerate(scenario_modes):
                scenario_no += 1
                sid = (
                    f"{csv_path.stem}_{attack}_{scenario_no:05d}"
                    f"_rep{replicate + 1:02d}"
                )
                scenario_mmsi = scenario_base + scenario_no
                # Deterministic severity grid: every source/attack/vessel gets
                # comparable mild, medium, and strong variants.  This is data
                # diversity for internal validation, not tuning on external.
                if scenarios_per_attack <= 1:
                    severity_scale = 1.0
                else:
                    severity_scale = float(
                        np.linspace(0.60, 1.40, scenarios_per_attack)[replicate]
                    )
                scenario_cfg = replace(
                    cfg,
                    reported_motion_mode=scenario_mode,
                    drift_lat_deg=float(cfg.drift_lat_deg) * severity_scale,
                    drift_lon_deg=float(cfg.drift_lon_deg) * severity_scale,
                    drift_rate_kmh=float(cfg.drift_rate_kmh) * severity_scale,
                    jump_lat_deg=float(cfg.jump_lat_deg) * severity_scale,
                    jump_lon_deg=float(cfg.jump_lon_deg) * severity_scale,
                    replay_delay_seconds=max(
                        60, int(round(float(cfg.replay_delay_seconds) * severity_scale))
                    ),
                    meacon_lag_steps=max(
                        1, int(round(float(cfg.meacon_lag_steps) * severity_scale))
                    ),
                    ghost_offset_min_deg=float(cfg.ghost_offset_min_deg) * severity_scale,
                    ghost_offset_max_deg=float(cfg.ghost_offset_max_deg) * severity_scale,
                    mirror_offset_min_deg=float(cfg.mirror_offset_min_deg) * severity_scale,
                    mirror_offset_max_deg=float(cfg.mirror_offset_max_deg) * severity_scale,
                )
                if attack == "gradual_drift":
                    part = _attack_gradual_drift(
                        g, vessel, scenario_cfg, rng, sid, scenario_mmsi
                    )
                elif attack == "location_jump":
                    part = _attack_location_jump(
                        g, vessel, scenario_cfg, rng, sid, scenario_mmsi
                    )
                elif attack == "replay":
                    part = _attack_replay(
                        g, vessel, scenario_cfg, rng, sid, scenario_mmsi
                    )
                elif attack == "meaconing":
                    part = _attack_meaconing(
                        g, vessel, scenario_cfg, rng, sid, scenario_mmsi
                    )
                elif attack == "ghost":
                    part = _attack_ghost(
                        g, vessel, scenario_cfg, rng, sid, scenario_mmsi
                    )
                elif attack == "mirroring":
                    part = _attack_mirroring(
                        g, vessel, scenario_cfg, rng, sid, scenario_mmsi
                    )
                else:
                    raise ValueError(attack)

                if not part.empty:
                    spoof_parts.append(part)
                    if bool(cfg.include_matched_normal_controls):
                        control = _matched_normal_control(
                            g,
                            part,
                            original_mmsi=vessel,
                            scenario_id=f"control::{sid}",
                            scenario_mmsi=scenario_base + 100_000_000 + scenario_no,
                            attack_type=attack,
                        )
                        spoof_parts.append(control)

    merged_parts = [normal] + spoof_parts
    merged = pd.concat(merged_parts, ignore_index=True, sort=False)
    # Trusted identity-registry side channel.  It is derived only from the
    # identities present in the source dataset, never from attack labels.
    registered_identities = set(df["mmsi"].astype(str).tolist())
    merged["claimed_identity_registered"] = (
        merged["claimed_mmsi"].astype(str).isin(registered_identities).astype("int8")
    )
    merged = merged.sort_values(["timestamp", "mmsi", "attack_type"]).reset_index(drop=True)

    out_path = out_dir / f"spoofed_{csv_path.stem}.csv"
    merged.to_csv(out_path, index=False)

    summary = (
        merged.groupby(["label", "attack_type"], dropna=False)
        .agg(rows=("mmsi", "size"), vessels=("mmsi", "nunique"))
        .reset_index()
        .sort_values(["label", "attack_type"])
    )
    summary_path = summary_dir / f"summary_{csv_path.stem}.csv"
    summary.to_csv(summary_path, index=False)

    magnitude_cols = [
        "scenario_id",
        "original_mmsi",
        "attack_type",
        "attack_nominal_lat_deg",
        "attack_nominal_lon_deg",
        "attack_applied_lat_deg",
        "attack_applied_lon_deg",
        "attack_displacement_km",
        "attack_points",
        "attack_duration_hours",
        "attack_drift_rate_kmh",
        "attack_target_drift_rate_kmh",
        "attack_applied_drift_rate_kmh",
        "attack_replay_delay_seconds",
        "attack_meacon_lag_steps",
        "reported_motion_mode",
    ]
    magnitude_summary = (
        merged.loc[merged["attack_type"] != "normal", magnitude_cols]
        .drop_duplicates(subset=["scenario_id"])
        .sort_values(["attack_type", "scenario_id"])
    )
    magnitude_path = summary_dir / f"magnitude_{csv_path.stem}.csv"
    magnitude_summary.to_csv(magnitude_path, index=False)

    print(f"[spoof] Saved dataset: {out_path}")
    print(f"[spoof] Saved summary: {summary_path}")
    print(f"[spoof] Saved magnitude audit: {magnitude_path}")
    print(summary.to_string(index=False))
    return out_path


def generate_spoofing_dataset(input_path: Path, out_dir: Path, cfg: SpoofingSimCfg) -> List[Path]:
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csvs = _input_csvs(input_path, cfg.include_labels, cfg.exclude_labels)

    if not csvs:
        raise FileNotFoundError(f"No CSV found in {input_path}")

    outputs: List[Path] = []
    for p in csvs:
        outputs.append(generate_spoofing_for_file(p, out_dir, cfg))

    if cfg.combine_outputs and len(outputs) > 1:
        combined_path = out_dir / "spoofed_all.csv"
        first = True
        for p in outputs:
            for chunk in pd.read_csv(p, chunksize=300_000):
                chunk.to_csv(combined_path, mode="w" if first else "a", index=False, header=first)
                first = False
        print(f"[spoof] Saved combined dataset: {combined_path}")
        outputs.append(combined_path)

    return outputs
