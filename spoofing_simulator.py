from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

from dataload import read_ais_csv
from data_preparation import haversine_km_np, bearing_deg_np


DEFAULT_ATTACKS = ["gradual_drift", "location_jump", "replay", "meaconing", "ghost", "mirroring"]


@dataclass
class SpoofingSimCfg:
    """
    Konfigurasi generator spoofing.

    Desainnya mengikuti ide paper: data AIS asli dibersihkan, lalu disisipkan titik
    manipulasi GPS dengan label Normal/Spoofed. Output CSV tetap membawa kolom
    asli (mmsi, timestamp, lat, lon, speed, course, dst.) supaya pipeline lama masih
    bisa dipakai.
    """

    attacks: List[str] = field(default_factory=lambda: DEFAULT_ATTACKS.copy())
    seed: int = 42

    # sampling agar aman untuk dataset besar
    limit_rows: int = 0
    chunksize: int = 0
    sample_frac: float = 0.0
    normal_keep_frac: float = 1.0

    # banyaknya vessel/points yang dimanipulasi
    max_vessels_per_file: int = 20
    min_points_per_vessel: int = 80
    points_per_attack: int = 120

    # parameter spoofing
    drift_lat_deg: float = 0.08
    drift_lon_deg: float = 0.08
    jump_lat_deg: float = 0.70
    jump_lon_deg: float = 0.70
    replay_delay_seconds: int = 6 * 3600
    meacon_lag_steps: int = 8
    ghost_offset_min_deg: float = 1.5
    ghost_offset_max_deg: float = 8.0
    mirror_offset_min_deg: float = 1.5
    mirror_offset_max_deg: float = 8.0

    # output
    combine_outputs: bool = False


def _as_list(attacks: Sequence[str] | str | None) -> List[str]:
    if attacks is None:
        return DEFAULT_ATTACKS.copy()
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
    return out or DEFAULT_ATTACKS.copy()


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
        df = df.sample(frac=float(cfg.sample_frac), random_state=int(cfg.seed)).copy()
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


def _segment(g: pd.DataFrame, n_points: int, rng: np.random.RandomState, extra: int = 0) -> pd.DataFrame:
    g = g.sort_values("timestamp").reset_index(drop=True)
    need = int(n_points) + int(extra)
    if len(g) < need:
        return pd.DataFrame(columns=g.columns)
    start_max = len(g) - need
    start = int(rng.randint(0, start_max + 1)) if start_max > 0 else 0
    return g.iloc[start:start + need].copy().reset_index(drop=True)


def _recompute_speed_course(seg: pd.DataFrame) -> pd.DataFrame:
    """Hitung ulang speed/course dari lat/lon/timestamp agar manipulasi tetap konsisten."""
    seg = seg.sort_values(["mmsi", "timestamp"]).copy()
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


def _finish_attack(seg: pd.DataFrame, attack: str, original_mmsi: int, scenario_id: str) -> pd.DataFrame:
    if seg.empty:
        return seg
    seg = _recompute_speed_course(seg)
    seg["is_spoofing"] = 1
    seg["label"] = "Spoofed"
    seg["attack_type"] = attack
    seg["original_mmsi"] = str(original_mmsi)
    seg["scenario_id"] = scenario_id
    seg["note"] = f"synthetic_{attack}"
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


def _attack_gradual_drift(g: pd.DataFrame, vessel: int, cfg: SpoofingSimCfg, rng: np.random.RandomState, sid: str) -> pd.DataFrame:
    seg = _segment(g, cfg.points_per_attack, rng)
    if seg.empty:
        return seg
    frac = np.linspace(0.0, 1.0, len(seg), dtype=float)
    lat_offset = _signed_offset(rng, cfg.drift_lat_deg)
    lon_offset = _signed_offset(rng, cfg.drift_lon_deg)
    seg["lat"] = _clip_lat(seg["lat"].to_numpy(dtype=float) + frac * lat_offset)
    seg["lon"] = _wrap_lon(seg["lon"].to_numpy(dtype=float) + frac * lon_offset)
    return _finish_attack(seg, "gradual_drift", vessel, sid)


def _attack_location_jump(g: pd.DataFrame, vessel: int, cfg: SpoofingSimCfg, rng: np.random.RandomState, sid: str) -> pd.DataFrame:
    seg = _segment(g, cfg.points_per_attack, rng)
    if seg.empty:
        return seg
    cut_lo = max(1, len(seg) // 3)
    cut_hi = max(cut_lo + 1, (len(seg) * 2) // 3)
    cut = int(rng.randint(cut_lo, cut_hi))
    lat = seg["lat"].to_numpy(dtype=float, copy=True)
    lon = seg["lon"].to_numpy(dtype=float, copy=True)
    lat[cut:] = _clip_lat(lat[cut:] + _signed_offset(rng, cfg.jump_lat_deg))
    lon[cut:] = _wrap_lon(lon[cut:] + _signed_offset(rng, cfg.jump_lon_deg))
    seg["lat"] = lat
    seg["lon"] = lon
    return _finish_attack(seg, "location_jump", vessel, sid)


def _attack_replay(g: pd.DataFrame, vessel: int, cfg: SpoofingSimCfg, rng: np.random.RandomState, sid: str) -> pd.DataFrame:
    seg = _segment(g, cfg.points_per_attack, rng)
    if seg.empty:
        return seg
    start_new = int(g["timestamp"].max()) + int(cfg.replay_delay_seconds) + int(rng.randint(0, 1800))
    delta = start_new - int(seg["timestamp"].iloc[0])
    seg["timestamp"] = seg["timestamp"].astype("int64") + int(delta)
    return _finish_attack(seg, "replay", vessel, sid)


def _attack_meaconing(g: pd.DataFrame, vessel: int, cfg: SpoofingSimCfg, rng: np.random.RandomState, sid: str) -> pd.DataFrame:
    lag = max(1, int(cfg.meacon_lag_steps))
    seg = _segment(g, cfg.points_per_attack, rng, extra=lag)
    if seg.empty or len(seg) <= lag:
        return pd.DataFrame(columns=g.columns)

    delayed = seg.iloc[lag:].copy().reset_index(drop=True)
    old = seg.iloc[:-lag].copy().reset_index(drop=True)
    # timestamp tetap timestamp saat ini, tapi posisi yang dibaca adalah posisi lama.
    delayed["lat"] = old["lat"].to_numpy(dtype=float)
    delayed["lon"] = old["lon"].to_numpy(dtype=float)
    if "speed" in delayed.columns:
        delayed["speed"] = old["speed"].to_numpy(dtype=float)
    if "course" in delayed.columns:
        delayed["course"] = old["course"].to_numpy(dtype=float)
    return _finish_attack(delayed, "meaconing", vessel, sid)


def _attack_ghost(g: pd.DataFrame, vessel: int, cfg: SpoofingSimCfg, rng: np.random.RandomState, sid: str, ghost_id: int) -> pd.DataFrame:
    seg = _segment(g, cfg.points_per_attack, rng)
    if seg.empty:
        return seg

    # Kapal palsu dibuat dari pola gerak yang masuk akal, lalu dipindahkan ke area lain.
    min_off = float(cfg.ghost_offset_min_deg)
    max_off = float(max(cfg.ghost_offset_max_deg, min_off))
    lat_off = _signed_offset(rng, rng.uniform(min_off, max_off))
    lon_off = _signed_offset(rng, rng.uniform(min_off, max_off))

    seg["mmsi"] = int(ghost_id)
    seg["lat"] = _clip_lat(seg["lat"].to_numpy(dtype=float) + lat_off)
    seg["lon"] = _wrap_lon(seg["lon"].to_numpy(dtype=float) + lon_off)
    seg["source"] = seg["source"].astype(str) + "_ghost"
    return _finish_attack(seg, "ghost", vessel, sid)


def _attack_mirroring(g: pd.DataFrame, vessel: int, cfg: SpoofingSimCfg, rng: np.random.RandomState, sid: str) -> pd.DataFrame:
    seg = _segment(g, cfg.points_per_attack, rng)
    if seg.empty:
        return seg

    lat_off = _signed_uniform_offset(rng, cfg.mirror_offset_min_deg, cfg.mirror_offset_max_deg)
    lon_off = _signed_uniform_offset(rng, cfg.mirror_offset_min_deg, cfg.mirror_offset_max_deg)

    seg["lat"] = _clip_lat(seg["lat"].to_numpy(dtype=float) + lat_off)
    seg["lon"] = _wrap_lon(seg["lon"].to_numpy(dtype=float) + lon_off)
    seg["source"] = seg["source"].astype(str) + "_mirroring"
    return _finish_attack(seg, "mirroring", vessel, sid)


def generate_spoofing_for_file(csv_path: Path, out_dir: Path, cfg: SpoofingSimCfg) -> Path:
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = out_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(int(cfg.seed) + (abs(hash(csv_path.stem)) % 100_000))
    attacks = _as_list(cfg.attacks)

    print(f"[spoof] read: {csv_path}")
    df = _read_input_csv(csv_path, cfg)
    if df.empty:
        raise RuntimeError(f"No valid rows in {csv_path}")

    normal = df.copy()
    if 0.0 < float(cfg.normal_keep_frac) < 1.0:
        normal = normal.sample(frac=float(cfg.normal_keep_frac), random_state=int(cfg.seed)).copy()
    elif float(cfg.normal_keep_frac) <= 0.0:
        normal = normal.iloc[0:0].copy()

    normal["is_spoofing"] = 0
    normal["label"] = "Normal"
    normal["attack_type"] = "normal"
    normal["original_mmsi"] = normal["mmsi"].astype(str)
    normal["scenario_id"] = "normal"
    normal["note"] = "original_ais"

    vessels = _choose_vessels(df, cfg, rng)
    if not vessels:
        raise RuntimeError(
            f"No eligible vessel with >= {cfg.min_points_per_vessel} points. "
            "Coba kecilkan --min_points_per_vessel atau naikkan --limit_rows."
        )

    spoof_parts = []
    ghost_base = 900_000_000_000_000 + int(rng.randint(0, 50_000_000))
    scenario_no = 0

    for vessel in vessels:
        g = df[df["mmsi"] == int(vessel)].sort_values("timestamp").reset_index(drop=True)
        for attack in attacks:
            scenario_no += 1
            sid = f"{csv_path.stem}_{attack}_{scenario_no:05d}"
            if attack == "gradual_drift":
                part = _attack_gradual_drift(g, vessel, cfg, rng, sid)
            elif attack == "location_jump":
                part = _attack_location_jump(g, vessel, cfg, rng, sid)
            elif attack == "replay":
                part = _attack_replay(g, vessel, cfg, rng, sid)
            elif attack == "meaconing":
                part = _attack_meaconing(g, vessel, cfg, rng, sid)
            elif attack == "ghost":
                ghost_id = ghost_base + scenario_no
                part = _attack_ghost(g, vessel, cfg, rng, sid, ghost_id=ghost_id)
            elif attack == "mirroring":
                part = _attack_mirroring(g, vessel, cfg, rng, sid)
            else:
                raise ValueError(attack)

            if not part.empty:
                spoof_parts.append(part)

    merged_parts = [normal] + spoof_parts
    merged = pd.concat(merged_parts, ignore_index=True, sort=False)
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

    print(f"[spoof] Saved dataset: {out_path}")
    print(f"[spoof] Saved summary: {summary_path}")
    print(summary.to_string(index=False))
    return out_path


def generate_spoofing_dataset(input_path: Path, out_dir: Path, cfg: SpoofingSimCfg) -> List[Path]:
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        csvs = sorted(input_path.glob("*.csv"))
    else:
        csvs = [input_path]

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
