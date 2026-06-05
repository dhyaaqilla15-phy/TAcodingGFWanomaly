from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def _unique_path(base_path: Path) -> Path:
    if not base_path.exists():
        return base_path
    stem = base_path.stem
    suf = base_path.suffix
    parent = base_path.parent
    i = 1
    while True:
        cand = parent / f"{stem}_{i}{suf}"
        if not cand.exists():
            return cand
        i += 1


def _pick_col(df: pd.DataFrame, *names: str) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in cols:
            return cols[n.lower()]
    return None


def _nice_step(span: float) -> float:
    if span <= 2:
        return 0.2
    if span <= 5:
        return 0.5
    if span <= 10:
        return 1.0
    if span <= 20:
        return 2.0
    if span <= 50:
        return 5.0
    return 10.0


def _iter_csv_inputs(path: Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(p for p in path.glob("*.csv") if p.is_file())
        if not files:
            raise FileNotFoundError(f"No CSV files found in directory: {path}")
        return files
    raise FileNotFoundError(f"Input path not found: {path}")


def _standardize_chunk(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Convert kolom ke format standar."""
    c_mmsi = _pick_col(df, "mmsi", "vessel_id", "id")
    c_ts = _pick_col(df, "timestamp", "time", "ts")
    c_lat = _pick_col(df, "lat", "latitude")
    c_lon = _pick_col(df, "lon", "lng", "longitude")
    c_spd = _pick_col(df, "speed", "sog")

    meta = {"c_spd": c_spd}

    need = [c_mmsi, c_ts, c_lat, c_lon]
    if any(x is None for x in need):
        raise ValueError(f"CSV missing required columns. Found: {list(df.columns)}")

    df = df.dropna(subset=[c_mmsi, c_ts, c_lat, c_lon]).copy()

    df[c_mmsi] = pd.to_numeric(df[c_mmsi], errors="coerce")
    df = df.dropna(subset=[c_mmsi]).copy()

    df["mmsi_std"] = df[c_mmsi].astype("int64").astype(str)
    df["ts_std"] = pd.to_numeric(df[c_ts], errors="coerce")
    df["lat_std"] = pd.to_numeric(df[c_lat], errors="coerce")
    df["lon_std"] = pd.to_numeric(df[c_lon], errors="coerce")

    df = df.dropna(subset=["ts_std", "lat_std", "lon_std"]).copy()

    if c_spd is not None:
        df["speed_std"] = pd.to_numeric(df[c_spd], errors="coerce").fillna(0.0)
    else:
        df["speed_std"] = 0.0

    return df[["mmsi_std", "ts_std", "lat_std", "lon_std", "speed_std"]], meta


def plot_trajectory_from_df(
    g: pd.DataFrame,
    title: str,
    out_path: Path,
    max_points: int = 6000,
    color_by: str = "speed",  # speed | time | none
    cmap: str = "viridis",
) -> Path:
    g = g.sort_values("ts_std")
    if len(g) > max_points:
        g = g.iloc[:max_points].copy()

    x = g["lon_std"].to_numpy()
    y = g["lat_std"].to_numpy()

    x_span = float(np.nanmax(x) - np.nanmin(x)) if len(x) else 1.0
    y_span = float(np.nanmax(y) - np.nanmin(y)) if len(y) else 1.0
    span = max(x_span, y_span)

    step = _nice_step(span)
    ratio = x_span / max(y_span, 1e-6)

    base_h = 7.0
    w = max(9.0, min(18.0, base_h * ratio))

    fig = plt.figure(figsize=(w, base_h))

    plt.plot(x, y, linewidth=0.8, alpha=0.20)

    if color_by == "time":
        t = g["ts_std"].to_numpy()
        c = (t - t.min()) / (t.max() - t.min()) if t.max() > t.min() else np.zeros_like(t, dtype=float)
        sc = plt.scatter(x, y, s=8, c=c, cmap=cmap, alpha=0.95)
        plt.colorbar(sc, label="Time (normalized)")
    elif color_by == "speed":
        s = g["speed_std"].to_numpy()
        sc = plt.scatter(x, y, s=8, c=s, cmap=cmap, alpha=0.95)
        plt.colorbar(sc, label="Speed")
    else:
        plt.scatter(x, y, s=8, alpha=0.95)

    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.grid(True, alpha=0.25)

    try:
        ax.set_aspect("equal", adjustable="box")
    except Exception:
        pass

    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path = _unique_path(out_path)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_trajectory_from_csv(
    csv_path: Path,
    out_dir: Path,
    sample_vessel: str = "",
    max_points: int = 6000,
    color_by: str = "speed",
    cmap: str = "viridis",
) -> Path:
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df, _ = _standardize_chunk(df)

    if sample_vessel:
        vessel = str(sample_vessel)
        g = df[df["mmsi_std"] == vessel].copy()
        if g.empty:
            raise ValueError("sample_vessel not found in file")
    else:
        vessel = df["mmsi_std"].value_counts().index[0]
        g = df[df["mmsi_std"] == vessel].copy()

    out_path = out_dir / f"trajectory_{csv_path.stem}_{vessel}_{color_by}.png"
    out_path = plot_trajectory_from_df(
        g=g,
        title=f"Trajectory vessel={vessel} ({csv_path.stem})",
        out_path=out_path,
        max_points=max_points,
        color_by=color_by,
        cmap=cmap,
    )

    print(f"[plot] Saved {out_path}")
    return out_path


def plot_all_from_csv(
    csv_path: Path,
    out_dir: Path,
    chunksize: int = 300_000,
    max_points_per_vessel: int = 6000,
    color_by: str = "speed",
    cmap: str = "viridis",
) -> Path:
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    csv_files = _iter_csv_inputs(csv_path)

    for one_csv in csv_files:
        _plot_all_single_csv(
            csv_path=one_csv,
            out_dir=out_dir,
            chunksize=chunksize,
            max_points_per_vessel=max_points_per_vessel,
            color_by=color_by,
            cmap=cmap,
        )

    if csv_path.is_dir():
        print(f"[plot_all] done. processed_files={len(csv_files)} root={csv_path}")
        return out_dir / "plots_all"
    return out_dir / "plots_all" / csv_path.stem


def _plot_all_single_csv(
    csv_path: Path,
    out_dir: Path,
    chunksize: int = 300_000,
    max_points_per_vessel: int = 6000,
    color_by: str = "speed",
    cmap: str = "viridis",
) -> Path:
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)

    tmp_dir = out_dir / "_tmp_vessels" / csv_path.stem
    plot_dir = out_dir / "plots_all" / csv_path.stem

    tmp_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}

    print(f"[plot_all] streaming split per-vessel: {csv_path}")

    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        chunk_std, _ = _standardize_chunk(chunk)

        for vessel, g in chunk_std.groupby("mmsi_std", sort=False):
            prev = counts.get(vessel, 0)
            if prev >= max_points_per_vessel:
                continue

            remaining = max_points_per_vessel - prev
            if len(g) > remaining:
                g = g.iloc[:remaining].copy()

            f = tmp_dir / f"{vessel}.csv"
            header = not f.exists()
            g.to_csv(f, mode="a", header=header, index=False)

            counts[vessel] = prev + len(g)

    counts_path = out_dir / "plots_all" / csv_path.stem / f"vessel_counts_{csv_path.stem}.csv"
    pd.DataFrame({"mmsi": list(counts.keys()), "points_saved": list(counts.values())}).sort_values(
        "points_saved", ascending=False
    ).to_csv(counts_path, index=False)

    print(f"[plot_all] saved vessel counts: {counts_path}")
    print(f"[plot_all] start plotting {len(counts)} vessels...")

    n_ok = 0
    for vessel_file in sorted(tmp_dir.glob("*.csv")):
        vessel = vessel_file.stem
        g = pd.read_csv(vessel_file)
        g = g.dropna(subset=["ts_std", "lat_std", "lon_std"]).copy()
        if len(g) < 2:
            continue

        out_path = plot_dir / f"trajectory_{csv_path.stem}_{vessel}_{color_by}.png"
        out_path = plot_trajectory_from_df(
            g=g,
            title=f"Trajectory vessel={vessel} ({csv_path.stem})",
            out_path=out_path,
            max_points=max_points_per_vessel,
            color_by=color_by,
            cmap=cmap,
        )
        n_ok += 1

    print(f"[plot_all] done. images_saved={n_ok} in {plot_dir}")
    return plot_dir


def heatmap_from_csv(
    csv_path: Path,
    out_dir: Path,
    bins: int = 350,
    chunksize: int = 500_000,
    log_scale: bool = True,
) -> Path:
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lat_min = np.inf
    lat_max = -np.inf
    lon_min = np.inf
    lon_max = -np.inf

    print(f"[heatmap] pass1 bounds: {csv_path}")

    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        chunk_std, _ = _standardize_chunk(chunk)
        lat = chunk_std["lat_std"].to_numpy()
        lon = chunk_std["lon_std"].to_numpy()
        if len(lat) == 0:
            continue
        lat_min = min(lat_min, float(np.nanmin(lat)))
        lat_max = max(lat_max, float(np.nanmax(lat)))
        lon_min = min(lon_min, float(np.nanmin(lon)))
        lon_max = max(lon_max, float(np.nanmax(lon)))

    if not np.isfinite(lat_min) or not np.isfinite(lon_min):
        raise RuntimeError("No valid lat/lon found for heatmap")

    lat_pad = max((lat_max - lat_min) * 0.01, 1e-6)
    lon_pad = max((lon_max - lon_min) * 0.01, 1e-6)
    lat_min -= lat_pad
    lat_max += lat_pad
    lon_min -= lon_pad
    lon_max += lon_pad

    H = np.zeros((bins, bins), dtype=np.float64)

    print(f"[heatmap] pass2 histogram bins={bins}")

    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        chunk_std, _ = _standardize_chunk(chunk)
        lat = chunk_std["lat_std"].to_numpy()
        lon = chunk_std["lon_std"].to_numpy()
        if len(lat) == 0:
            continue

        h, _, _ = np.histogram2d(
            lat,
            lon,
            bins=bins,
            range=[[lat_min, lat_max], [lon_min, lon_max]],
        )
        H += h

    Z = np.log1p(H) if log_scale else H

    fig = plt.figure(figsize=(10, 7))
    plt.imshow(Z, origin="lower", extent=[lon_min, lon_max, lat_min, lat_max], aspect="auto")
    plt.colorbar(label=("log(1+count)" if log_scale else "count"))
    plt.title(f"Heatmap density ({csv_path.stem})")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()

    out_path = out_dir / f"heatmap_{csv_path.stem}.png"
    out_path = _unique_path(out_path)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    print(f"[heatmap] Saved {out_path}")
    return out_path
