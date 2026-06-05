from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_trajectory import _standardize_chunk, _unique_path


PHASE_COLORS = {
    "pre_blackout": "#f08c00",
    "reappearance": "#1971c2",
    "hidden_segment": "#c1121f",
}


def _format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    hours = seconds / 3600.0
    if hours >= 24.0:
        return f"{hours / 24.0:.1f} d"
    if hours >= 1.0:
        return f"{hours:.1f} h"
    return f"{seconds / 60.0:.0f} min"


def _add_direction_arrows(ax, g: pd.DataFrame, color: str, count: int = 3) -> None:
    gg = g.sort_values("ts_std")
    if len(gg) < 4:
        return
    xs = gg["lon_std"].to_numpy(dtype=float)
    ys = gg["lat_std"].to_numpy(dtype=float)
    for frac in np.linspace(0.25, 0.85, count):
        i = int(np.clip(round(frac * (len(xs) - 2)), 0, len(xs) - 2))
        dx = xs[i + 1] - xs[i]
        dy = ys[i + 1] - ys[i]
        if not np.isfinite(dx + dy) or (abs(dx) + abs(dy) <= 1e-12):
            continue
        ax.annotate(
            "",
            xy=(xs[i + 1], ys[i + 1]),
            xytext=(xs[i], ys[i]),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.1, alpha=0.85),
        )


def _set_readable_extent(ax, parts: List[pd.DataFrame]) -> None:
    non_empty = [p for p in parts if p is not None and not p.empty]
    if not non_empty:
        return
    all_g = pd.concat(non_empty, ignore_index=True, sort=False)
    lon = all_g["lon_std"].to_numpy(dtype=float)
    lat = all_g["lat_std"].to_numpy(dtype=float)
    lon = lon[np.isfinite(lon)]
    lat = lat[np.isfinite(lat)]
    if lon.size == 0 or lat.size == 0:
        return
    lon_min, lon_max = float(lon.min()), float(lon.max())
    lat_min, lat_max = float(lat.min()), float(lat.max())
    lon_pad = max((lon_max - lon_min) * 0.12, 0.01)
    lat_pad = max((lat_max - lat_min) * 0.12, 0.01)
    ax.set_xlim(lon_min - lon_pad, lon_max + lon_pad)
    ax.set_ylim(lat_min - lat_pad, lat_max + lat_pad)


def _load_go_dark_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    std, _ = _standardize_chunk(df)

    out = std.copy()
    if "is_go_dark" in df.columns:
        out["is_go_dark"] = pd.to_numeric(df["is_go_dark"], errors="coerce").fillna(0).astype(int).to_numpy()
    elif "label" in df.columns:
        out["is_go_dark"] = df["label"].astype(str).str.lower().eq("godark").astype(int).to_numpy()
    else:
        out["is_go_dark"] = 0

    for col, default in [
        ("event_type", "normal"),
        ("event_phase", "normal"),
        ("go_dark_event_id", "normal"),
        ("original_mmsi", None),
    ]:
        if col in df.columns:
            out[col] = df[col].astype(str).to_numpy()
        elif default is not None:
            out[col] = default

    if "original_mmsi" not in out.columns:
        out["original_mmsi"] = out["mmsi_std"].astype(str)

    for col in [
        "gap_start_timestamp",
        "gap_end_timestamp",
        "dark_duration_seconds",
        "hidden_distance_km",
        "implied_speed_knots",
    ]:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy()
        else:
            out[col] = 0

    return out


def _choose_vessel(df: pd.DataFrame, sample_vessel: str = "") -> str:
    if sample_vessel:
        return str(sample_vessel)
    dark = df[df["is_go_dark"] == 1]
    if not dark.empty:
        return str(dark["original_mmsi"].value_counts().index[0])
    return str(df["mmsi_std"].value_counts().index[0])


def _load_hidden_truth_for_event(csv_path: Path, vessel: str, event_id: str = "all") -> pd.DataFrame:
    hidden_dir = Path(csv_path).parent / "hidden_truth"
    if not hidden_dir.exists():
        return pd.DataFrame()

    parts = []
    for p in sorted(hidden_dir.glob("hidden_truth_*.csv")):
        try:
            h = _load_go_dark_csv(p)
        except Exception:
            continue
        keep = (h["mmsi_std"].astype(str) == str(vessel)) | (
            h["original_mmsi"].astype(str) == str(vessel)
        )
        if event_id and str(event_id).lower() != "all":
            keep = keep & h["go_dark_event_id"].astype(str).eq(str(event_id))
        h = h[keep].copy()
        if not h.empty:
            parts.append(h)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True, sort=False).sort_values("ts_std")


def plot_go_dark_overlay_from_csv(
    csv_path: Path,
    out_dir: Path,
    sample_vessel: str = "",
    event_id: str = "all",
    max_points: int = 9000,
) -> Path:
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_go_dark_csv(csv_path)
    vessel = _choose_vessel(df, sample_vessel=sample_vessel)

    g = df[(df["mmsi_std"].astype(str) == vessel) | (df["original_mmsi"].astype(str) == vessel)].copy()
    if event_id and event_id.lower() != "all":
        keep = (g["go_dark_event_id"].astype(str) == event_id) | (g["go_dark_event_id"].astype(str) == "normal")
        g = g[keep].copy()

    if g.empty:
        raise ValueError(f"No points found for vessel/original_mmsi={vessel}")

    g = g.sort_values("ts_std").reset_index(drop=True)
    if len(g) > max_points:
        if event_id and event_id.lower() != "all":
            event_mask = g["go_dark_event_id"].astype(str).eq(str(event_id))
            event_pos = np.where(event_mask.to_numpy())[0]
            if event_pos.size > 0:
                center_start = int(event_pos.min())
                center_end = int(event_pos.max()) + 1
                event_len = center_end - center_start
                room = max(int(max_points) - event_len, 0)
                before = room // 2
                after = room - before
                start = max(0, center_start - before)
                end = min(len(g), center_end + after)
                if end - start < int(max_points):
                    start = max(0, end - int(max_points))
                    end = min(len(g), start + int(max_points))
                g = g.iloc[start:end].copy()
        if len(g) > max_points:
            g = g.iloc[:max_points].copy()

    normal = g[g["is_go_dark"] == 0].sort_values("ts_std")
    dark = g[g["is_go_dark"] == 1].sort_values("ts_std")
    hidden = _load_hidden_truth_for_event(csv_path, vessel=vessel, event_id=event_id)

    fig, ax = plt.subplots(figsize=(11, 7.5))
    if not normal.empty:
        ax.plot(
            normal["lon_std"],
            normal["lat_std"],
            linewidth=1.0,
            color="#6c757d",
            alpha=0.48,
            label="Visible AIS trajectory",
        )
        ax.scatter(
            normal["lon_std"],
            normal["lat_std"],
            s=10,
            color="#adb5bd",
            alpha=0.50,
            label="Normal visible AIS",
        )
        _add_direction_arrows(ax, normal, "#495057", count=2)

    if not dark.empty:
        for phase, gg in dark.groupby("event_phase", sort=False):
            phase_key = str(phase).lower().strip()
            color = PHASE_COLORS.get(phase_key, "#f08c00")
            marker = "^" if phase_key == "pre_blackout" else "X"
            ax.scatter(
                gg["lon_std"],
                gg["lat_std"],
                s=72,
                color=color,
                marker=marker,
                alpha=0.96,
                edgecolor="white",
                linewidth=0.9,
                label=f"Go-dark boundary: {phase}",
                zorder=6,
            )

    if not hidden.empty:
        ax.plot(
            hidden["lon_std"],
            hidden["lat_std"],
            linestyle="--",
            linewidth=2.0,
            color="#c1121f",
            alpha=0.92,
            label="Hidden truth path (not visible in AIS)",
            zorder=4,
        )
        ax.scatter(
            hidden["lon_std"],
            hidden["lat_std"],
            s=18,
            color="#c1121f",
            alpha=0.70,
            edgecolor="white",
            linewidth=0.25,
            label="Hidden AIS points",
            zorder=5,
        )
        _add_direction_arrows(ax, hidden, "#c1121f", count=3)

    # Garis gap: hubungkan titik sebelum mati dan titik reappearance per event.
    event_groups = dark[dark["go_dark_event_id"].astype(str) != "normal"].groupby("go_dark_event_id", sort=False)
    info_lines = []
    for eid, gg in event_groups:
        gap_start = float(gg["gap_start_timestamp"].iloc[0])
        gap_end = float(gg["gap_end_timestamp"].iloc[0])
        before = g[g["ts_std"] == gap_start]
        after = g[g["ts_std"] == gap_end]
        if before.empty or after.empty:
            continue
        x = [float(before["lon_std"].iloc[-1]), float(after["lon_std"].iloc[0])]
        y = [float(before["lat_std"].iloc[-1]), float(after["lat_std"].iloc[0])]
        duration_hr = float(gg["dark_duration_seconds"].iloc[0]) / 3600.0
        ax.plot(
            x,
            y,
            linestyle=":",
            linewidth=2.2,
            color="#212529",
            alpha=0.88,
            label=f"Observed AIS gap {duration_hr:.1f} h",
            zorder=3,
        )
        info_lines = [
            f"Event: {eid}",
            f"AIS dark duration: {_format_duration(float(gg['dark_duration_seconds'].iloc[0]))}",
            f"Hidden distance: {float(gg['hidden_distance_km'].iloc[0]):.1f} km",
            f"Implied speed: {float(gg['implied_speed_knots'].iloc[0]):.1f} kn",
        ]

    ax.set_title(f"Go-Dark AIS Gap Simulation | vessel/original_mmsi={vessel}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.22)
    ax.set_aspect("equal", adjustable="box")
    _set_readable_extent(ax, [g, hidden])
    if info_lines:
        ax.text(
            0.02,
            0.98,
            "\n".join(info_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#ced4da", alpha=0.92),
        )
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_handles = []
    uniq_labels = []
    for h, lab in zip(handles, labels):
        if lab in seen:
            continue
        seen.add(lab)
        uniq_handles.append(h)
        uniq_labels.append(lab)
    ax.legend(uniq_handles, uniq_labels, fontsize=8, loc="lower right", framealpha=0.92)
    fig.tight_layout()

    suffix = event_id if event_id and event_id.lower() != "all" else "all"
    out_path = _unique_path(out_dir / f"trajectory_godark_{csv_path.stem}_{vessel}_{suffix}.png")
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    print(f"[plot_godark] Saved {out_path}")
    return out_path


def heatmap_go_dark_from_csv(csv_path: Path, out_dir: Path, bins: int = 300, log_scale: bool = True) -> List[Path]:
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_go_dark_csv(csv_path)
    paths: List[Path] = []

    lat_all = df["lat_std"].to_numpy(dtype=float)
    lon_all = df["lon_std"].to_numpy(dtype=float)
    lat_min, lat_max = float(np.nanmin(lat_all)), float(np.nanmax(lat_all))
    lon_min, lon_max = float(np.nanmin(lon_all)), float(np.nanmax(lon_all))
    lat_pad = max((lat_max - lat_min) * 0.01, 1e-6)
    lon_pad = max((lon_max - lon_min) * 0.01, 1e-6)
    extent = [lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad]

    for name, part in [("normal", df[df["is_go_dark"] == 0]), ("go_dark", df[df["is_go_dark"] == 1])]:
        if part.empty:
            continue
        h, _, _ = np.histogram2d(
            part["lat_std"].to_numpy(dtype=float),
            part["lon_std"].to_numpy(dtype=float),
            bins=int(bins),
            range=[[extent[2], extent[3]], [extent[0], extent[1]]],
        )
        z = np.log1p(h) if log_scale else h
        fig = plt.figure(figsize=(10, 7))
        plt.imshow(z, origin="lower", extent=extent, aspect="auto")
        plt.colorbar(label="log(1+count)" if log_scale else "count")
        plt.title(f"Heatmap {name} density ({csv_path.stem})")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.tight_layout()
        out_path = _unique_path(out_dir / f"heatmap_{name}_{csv_path.stem}.png")
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        print(f"[heatmap_godark] Saved {out_path}")
        paths.append(out_path)

    return paths


def plot_go_dark_examples_from_csv(
    csv_path: Path,
    out_dir: Path,
    num_examples: int = 6,
    max_points: int = 9000,
) -> Path:
    """
    Plot multiple go-dark event examples.
    Membuat output contoh go-dark yang langsung dibagi menjadi folder per event:

        out_dir/
          event_1/
          event_2/
          event_3/
          event_4/
          event_5/
          event_6/
          examples_summary_<nama_csv>.csv

    Setiap folder event berisi 1 gambar contoh trajectory untuk event tersebut.
    """
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_go_dark_csv(csv_path)
    dark = df[df["is_go_dark"] == 1].copy()

    if dark.empty:
        raise ValueError(f"No go-dark rows found in {csv_path}")

    # Get unique event IDs
    event_ids = sorted(
        [e for e in dark["go_dark_event_id"].astype(str).unique() if e.lower() != "normal"]
    )[:num_examples]

    rows = []

    for idx, event_id in enumerate(event_ids, 1):
        event_name = f"event_{idx}"
        event_dir = out_dir / event_name
        event_dir.mkdir(parents=True, exist_ok=True)

        part = dark[dark["go_dark_event_id"].astype(str) == event_id].copy()

        if part.empty:
            rows.append(
                {
                    "event_num": idx,
                    "event_id": event_id,
                    "status": "skipped_no_rows",
                    "example_vessel": "",
                    "dark_rows": 0,
                    "output_dir": str(event_dir),
                    "output_path": "",
                }
            )
            continue

        vessel = str(part["original_mmsi"].value_counts().index[0])

        try:
            out_path = plot_go_dark_overlay_from_csv(
                csv_path=csv_path,
                out_dir=event_dir,
                sample_vessel=vessel,
                event_id=event_id,
                max_points=max_points,
            )

            rows.append(
                {
                    "event_num": idx,
                    "event_id": event_id,
                    "status": "saved",
                    "example_vessel": vessel,
                    "dark_rows": int(len(part[part["original_mmsi"].astype(str) == vessel])),
                    "output_dir": str(event_dir),
                    "output_path": str(out_path),
                }
            )
        except Exception as e:
            rows.append(
                {
                    "event_num": idx,
                    "event_id": event_id,
                    "status": f"error_{str(e)[:30]}",
                    "example_vessel": vessel,
                    "dark_rows": 0,
                    "output_dir": str(event_dir),
                    "output_path": "",
                }
            )

    summary = pd.DataFrame(rows)
    summary_path = _unique_path(out_dir / f"examples_summary_{csv_path.stem}.csv")
    summary.to_csv(summary_path, index=False)
    print(f"[plot_godark_examples] Summary saved to {summary_path}")
    return summary_path
