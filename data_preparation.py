from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from dataload import read_ais_csv, infer_label_from_filename


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

    # khusus task=spoofing
    spoofing_window_threshold: float = 0.20


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

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
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

    # dt per step dalam detik.
    # Untuk task go-dark, gap panjang adalah sinyal penting.
    # Jadi dt tidak boleh dipotong ke default 3 jam.
    raw_dt = df.groupby("mmsi")["timestamp"].diff().fillna(1.0).astype("float32")

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
    df["dspeed"] = df.groupby("mmsi")["speed"].diff().fillna(0).astype("float32")

    # perubahan course dengan shortest angle
    prev_course = df.groupby("mmsi")["course"].shift(1).astype("float32")
    dc = (course_deg - prev_course) % 360.0
    dc = ((dc + 180.0) % 360.0) - 180.0
    df["dcourse"] = dc.fillna(0).astype("float32")
    df["abs_dcourse"] = np.abs(df["dcourse"]).astype("float32")

    # jarak antar titik
    prev_lat = df.groupby("mmsi")["lat"].shift(1)
    prev_lon = df.groupby("mmsi")["lon"].shift(1)
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
        df.groupby("mmsi")["pos_speed_knots"].diff().fillna(0).astype("float32")
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

    berr = (course_deg - pos_bearing) % 360.0
    berr = ((berr + 180.0) % 360.0) - 180.0
    df["bearing_error"] = berr.astype("float32")

    curv = df["abs_dcourse"] / (df["step_km"] + 1e-3)
    df["curvature"] = curv.clip(0.0, 500.0).astype("float32")

    # rolling stats
    g = df.groupby("mmsi", sort=False)

    df["pos_speed_ma5"] = (
        g["pos_speed_knots"]
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .astype("float32")
    )

    df["pos_speed_std5"] = (
        g["pos_speed_knots"]
        .rolling(5, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
        .astype("float32")
    )

    abs_turn = df["turn_rate"].abs().astype("float32")
    df["abs_turn_ma5"] = (
        abs_turn.groupby(df["mmsi"], sort=False)
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .astype("float32")
    )

    df["curvature_ma5"] = (
        g["curvature"]
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
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


def build_sequences_from_df(
    df: pd.DataFrame,
    cfg: PreprocessCfg,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = clean_and_derive(df, cfg)

    if cfg.apply_jump_filter:
        df = filter_jumps(df, cfg)

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

    feat_cols = [
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

    df = df.dropna(subset=feat_cols).copy()

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    g_list: List[str] = []
    coord_list: List[np.ndarray] = []

    for vessel_id, g in df.groupby("mmsi", sort=False):
        g = g.sort_values("timestamp")

        if len(g) < cfg.min_points_per_vessel:
            continue

        ts = g["timestamp"].to_numpy()
        y_point = g["y_point"].to_numpy(dtype=np.int64)

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
                g_list.append(vessel_id)
                coord_list.append(coord_window)
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
        )

    return (
        np.stack(X_list),
        np.array(y_list, dtype=np.int64),
        np.array(g_list, dtype=object),
        np.stack(coord_list).astype(np.float64),
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
    spoofing_window_threshold: float = 0.20,
    apply_jump_filter: Optional[bool] = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if apply_jump_filter is None:
        apply_jump_filter = task not in ["spoofing", "godark"]

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
        spoofing_window_threshold=float(spoofing_window_threshold),
        apply_jump_filter=bool(apply_jump_filter),
    )

    exclude_labels = exclude_labels or []

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

    if not csvs:
        raise FileNotFoundError(f"No CSV found in {data_dir}")

    all_X, all_y, all_groups, all_coords = [], [], [], []
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

    for p in tqdm(csvs, desc="files"):
        stem = infer_label_from_filename(p)

        if stem in exclude_labels:
            continue

        df = read_ais_csv(p, limit_rows=limit_rows, chunksize=chunksize)

        try:
            X, y, groups, coords = build_sequences_from_df(df, cfg)
        except ValueError as exc:
            if task in ["spoofing", "godark"]:
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
            idx = rng.choice(len(X), size=cfg.max_windows_per_file, replace=False)

            X = X[idx]
            y = y[idx]
            groups = groups[idx]
            coords = coords[idx]

        all_X.append(X)
        all_y.append(y)
        all_groups.append(groups)
        all_coords.append(coords)

    if not all_X:
        raise RuntimeError("No sequences created. Coba kecilin seq_len atau cek file.")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    groups = np.concatenate(all_groups, axis=0)
    coords = np.concatenate(all_coords, axis=0)

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

                counts = np.bincount(y.astype(np.int64), minlength=len(gear_to_id))
                print("[preprocess] gear class windows after balance:", {label: int(counts[idx]) for label, idx in gear_to_id.items()})

        label_map = {v: k for k, v in gear_to_id.items()}
    elif task == "spoofing":
        label_map = {0: "normal", 1: "spoofing"}
    elif task == "godark":
        label_map = {0: "normal", 1: "go_dark"}
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
        label_map=np.array(list(label_map.items()), dtype=object),
        scaled=np.array(False),
    )

    print(f"[preprocess] Saved: {out_path}")
    print(f"[preprocess] X={X.shape} y={y.shape} classes={len(set(y.tolist()))}")
    print("[preprocess] scaler will be fit on train split during train.")
    print("[preprocess] label_map:", label_map)
