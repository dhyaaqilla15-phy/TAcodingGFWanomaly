from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_preparation import haversine_km_np
from plot_trajectory import _standardize_chunk, _unique_path


ATTACK_DISPLAY = {
    "gradual_drift": "gradual drift",
    "location_jump": "location jump",
    "replay": "replay",
    "meaconing": "meaconing",
    "ghost": "ghost vessel",
    "mirroring": "mirroring (shifted copy)",
}

ATTACK_COLORS = {
    "gradual_drift": "#d94801",
    "location_jump": "#c1121f",
    "replay": "#7b2cbf",
    "meaconing": "#0b7285",
    "ghost": "#2b8a3e",
    "mirroring": "#e67700",
}


def _attack_display_name(attack: str) -> str:
    key = str(attack).lower().strip()
    return ATTACK_DISPLAY.get(key, key.replace("_", " "))


def _format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    hours = seconds / 3600.0
    if hours >= 24.0:
        return f"{hours / 24.0:.1f} d"
    if hours >= 1.0:
        return f"{hours:.1f} h"
    return f"{seconds / 60.0:.0f} min"


def _track_distance_km(g: pd.DataFrame) -> float:
    if len(g) < 2:
        return 0.0
    gg = g.sort_values("ts_std")
    lat = gg["lat_std"].to_numpy(dtype=float)
    lon = gg["lon_std"].to_numpy(dtype=float)
    return float(np.nansum(haversine_km_np(lat[:-1], lon[:-1], lat[1:], lon[1:])))


def _add_direction_arrows(ax, g: pd.DataFrame, color: str, count: int = 3) -> None:
    gg = g.sort_values("ts_std")
    if len(gg) < 4:
        return
    xs = gg["lon_std"].to_numpy(dtype=float)
    ys = gg["lat_std"].to_numpy(dtype=float)
    positions = np.linspace(0.25, 0.85, count)
    for frac in positions:
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


def _set_readable_extent(ax, g: pd.DataFrame) -> None:
    lon = g["lon_std"].to_numpy(dtype=float)
    lat = g["lat_std"].to_numpy(dtype=float)
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


def _load_spoofing_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    std, _ = _standardize_chunk(df)

    out = std.copy()

    if "is_spoofing" in df.columns:
        out["is_spoofing"] = (
            pd.to_numeric(df["is_spoofing"], errors="coerce")
            .fillna(0)
            .astype(int)
            .to_numpy()
        )
    elif "label" in df.columns:
        out["is_spoofing"] = (
            df["label"]
            .astype(str)
            .str.lower()
            .isin(["spoofed", "spoofing"])
            .astype(int)
            .to_numpy()
        )
    else:
        out["is_spoofing"] = 0

    if "attack_type" in df.columns:
        out["attack_type"] = df["attack_type"].astype(str).to_numpy()
    else:
        out["attack_type"] = np.where(
            out["is_spoofing"].to_numpy() == 1,
            "spoofing",
            "normal",
        )

    if "original_mmsi" in df.columns:
        out["original_mmsi"] = df["original_mmsi"].astype(str).to_numpy()
    else:
        out["original_mmsi"] = out["mmsi_std"].astype(str).to_numpy()

    return out


def _choose_vessel(df: pd.DataFrame, sample_vessel: str = "") -> str:
    if sample_vessel:
        return str(sample_vessel)

    spoof = df[df["is_spoofing"] == 1]
    if not spoof.empty:
        return str(spoof["original_mmsi"].value_counts().index[0])

    return str(df["mmsi_std"].value_counts().index[0])


def plot_spoofing_overlay_from_csv(
    csv_path: Path,
    out_dir: Path,
    sample_vessel: str = "",
    attack_type: str = "all",
    max_points: int = 8000,
    title_prefix: str = "",
    output_prefix: str = "trajectory_spoofing",
) -> Path:
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_spoofing_csv(csv_path)
    vessel = _choose_vessel(df, sample_vessel=sample_vessel)

    g = df[
        (df["mmsi_std"].astype(str) == vessel)
        | (df["original_mmsi"].astype(str) == vessel)
    ].copy()

    if attack_type and attack_type.lower() != "all":
        keep = (
            g["attack_type"].astype(str).str.lower().eq(attack_type.lower())
            | g["attack_type"].astype(str).str.lower().eq("normal")
        )
        g = g[keep].copy()

    if g.empty:
        raise ValueError(f"No points found for vessel/original_mmsi={vessel}")

    g = g.sort_values("ts_std").reset_index(drop=True)

    if len(g) > max_points:
        if attack_type and attack_type.lower() != "all":
            attack_mask = g["attack_type"].astype(str).str.lower().eq(attack_type.lower())
            attack_pos = np.where(attack_mask.to_numpy())[0]
            if attack_pos.size > 0:
                center_start = int(attack_pos.min())
                center_end = int(attack_pos.max()) + 1
                attack_len = center_end - center_start
                room = max(int(max_points) - attack_len, 0)
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

    normal = g[g["is_spoofing"] == 0].sort_values("ts_std")
    spoofed = g[g["is_spoofing"] == 1].sort_values("ts_std")

    fig, ax = plt.subplots(figsize=(11, 7.5))

    if not normal.empty:
        ax.plot(
            normal["lon_std"],
            normal["lat_std"],
            linewidth=1.0,
            color="#6c757d",
            alpha=0.50,
            label="Normal AIS trajectory",
        )
        ax.scatter(
            normal["lon_std"],
            normal["lat_std"],
            s=10,
            color="#adb5bd",
            alpha=0.55,
            label="Normal points",
        )
        ax.scatter(
            normal["lon_std"].iloc[0],
            normal["lat_std"].iloc[0],
            s=45,
            color="#495057",
            marker="o",
            edgecolor="white",
            linewidth=0.8,
            label="Normal start",
            zorder=5,
        )
        _add_direction_arrows(ax, normal, "#495057", count=2)

    info_lines = []
    if not spoofed.empty:
        for at, gg in spoofed.groupby("attack_type", sort=False):
            gg = gg.sort_values("ts_std")
            at_key = str(at).lower().strip()
            label_name = _attack_display_name(str(at))
            color = ATTACK_COLORS.get(at_key, "#c1121f")
            ax.plot(
                gg["lon_std"],
                gg["lat_std"],
                linewidth=2.2,
                color=color,
                alpha=0.92,
                label=f"Spoofed line: {label_name}",
            )
            ax.scatter(
                gg["lon_std"],
                gg["lat_std"],
                s=20,
                color=color,
                alpha=0.90,
                edgecolor="white",
                linewidth=0.25,
                label=f"Spoofed points: {label_name}",
            )
            ax.scatter(
                gg["lon_std"].iloc[0],
                gg["lat_std"].iloc[0],
                s=95,
                color=color,
                marker="^",
                edgecolor="white",
                linewidth=1.0,
                label="Attack start",
                zorder=6,
            )
            ax.scatter(
                gg["lon_std"].iloc[-1],
                gg["lat_std"].iloc[-1],
                s=95,
                color=color,
                marker="X",
                edgecolor="white",
                linewidth=1.0,
                label="Attack end",
                zorder=6,
            )
            _add_direction_arrows(ax, gg, color, count=4)

            duration = float(gg["ts_std"].max() - gg["ts_std"].min()) if len(gg) else 0.0
            info_lines = [
                f"Attack: {label_name}",
                f"Spoofed points: {len(gg)}",
                f"Attack duration: {_format_duration(duration)}",
                f"Spoofed track length: {_track_distance_km(gg):.1f} km",
            ]

            if at_key in {"ghost", "mirroring"} and not normal.empty:
                start_km = float(
                    haversine_km_np(
                        np.array([normal["lat_std"].iloc[0]], dtype=float),
                        np.array([normal["lon_std"].iloc[0]], dtype=float),
                        np.array([gg["lat_std"].iloc[0]], dtype=float),
                        np.array([gg["lon_std"].iloc[0]], dtype=float),
                    )[0]
                )
                info_lines.append(f"Start offset vs normal: {start_km:.1f} km")

    ax.set_title(
        f"{title_prefix}Normal vs Spoofed Trajectory | "
        f"vessel/original_mmsi={vessel}"
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.22)
    ax.set_aspect("equal", adjustable="box")
    _set_readable_extent(ax, g)
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

    suffix = attack_type.lower() if attack_type else "all"
    safe_vessel = str(vessel).replace("/", "_").replace("\\", "_")

    out_path = _unique_path(
        out_dir / f"{output_prefix}_{csv_path.stem}_{safe_vessel}_{suffix}.png"
    )

    fig.savefig(out_path, dpi=240)
    plt.close(fig)

    print(f"[plot_spoofing] Saved {out_path}")
    return out_path


def heatmap_spoofing_from_csv(
    csv_path: Path,
    out_dir: Path,
    bins: int = 300,
    log_scale: bool = True,
) -> List[Path]:
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_spoofing_csv(csv_path)
    paths: List[Path] = []

    lat_all = df["lat_std"].to_numpy(dtype=float)
    lon_all = df["lon_std"].to_numpy(dtype=float)

    lat_min = float(np.nanmin(lat_all))
    lat_max = float(np.nanmax(lat_all))
    lon_min = float(np.nanmin(lon_all))
    lon_max = float(np.nanmax(lon_all))

    lat_pad = max((lat_max - lat_min) * 0.01, 1e-6)
    lon_pad = max((lon_max - lon_min) * 0.01, 1e-6)

    extent = [
        lon_min - lon_pad,
        lon_max + lon_pad,
        lat_min - lat_pad,
        lat_max + lat_pad,
    ]

    parts = [
        ("normal", df[df["is_spoofing"] == 0]),
        ("spoofed", df[df["is_spoofing"] == 1]),
    ]

    for name, part in parts:
        if part.empty:
            continue

        h, _, _ = np.histogram2d(
            part["lat_std"].to_numpy(dtype=float),
            part["lon_std"].to_numpy(dtype=float),
            bins=int(bins),
            range=[
                [extent[2], extent[3]],
                [extent[0], extent[1]],
            ],
        )

        z = np.log1p(h) if log_scale else h

        fig = plt.figure(figsize=(10, 7))
        plt.imshow(
            z,
            origin="lower",
            extent=extent,
            aspect="auto",
        )
        plt.colorbar(label="log(1+count)" if log_scale else "count")
        plt.title(f"Heatmap {name} density ({csv_path.stem})")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.tight_layout()

        out_path = _unique_path(out_dir / f"heatmap_{name}_{csv_path.stem}.png")
        fig.savefig(out_path, dpi=220)
        plt.close(fig)

        print(f"[heatmap_spoofing] Saved {out_path}")
        paths.append(out_path)

    return paths


def plot_spoofing_examples_from_csv(
    csv_path: Path,
    out_dir: Path,
    attacks: Optional[List[str]] = None,
    max_points: int = 8000,
) -> Path:
    """
    Membuat output contoh spoofing yang langsung dibagi menjadi folder per attack:

        out_dir/
          gradual_drift/
          location_jump/
          replay/
          meaconing/
          ghost/
          mirroring/
          examples_summary_<nama_csv>.csv

    Setiap folder attack berisi 1 gambar contoh trajectory untuk attack tersebut.
    """

    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_spoofing_csv(csv_path)
    spoofed = df[df["is_spoofing"] == 1].copy()

    if spoofed.empty:
        raise ValueError(f"No spoofed rows found in {csv_path}")

    if attacks is None or len(attacks) == 0:
        attacks = [
            a
            for a in sorted(spoofed["attack_type"].dropna().astype(str).unique())
            if a.lower() != "normal"
        ]

    rows = []

    for attack in attacks:
        attack_name = str(attack).lower().strip()

        attack_dir = out_dir / attack_name
        attack_dir.mkdir(parents=True, exist_ok=True)

        part = spoofed[
            spoofed["attack_type"].astype(str).str.lower() == attack_name
        ].copy()

        if part.empty:
            rows.append(
                {
                    "attack_type": attack_name,
                    "status": "skipped_no_rows",
                    "example_original_mmsi": "",
                    "spoofed_rows": 0,
                    "output_dir": str(attack_dir),
                    "output_path": "",
                }
            )
            continue

        vessel_scores = []
        for vid, gg in part.groupby(part["original_mmsi"].astype(str), sort=False):
            lat_span = float(gg["lat_std"].max() - gg["lat_std"].min()) if len(gg) else 0.0
            lon_span = float(gg["lon_std"].max() - gg["lon_std"].min()) if len(gg) else 0.0
            unique_points = int(gg[["lat_std", "lon_std"]].drop_duplicates().shape[0])
            vessel_scores.append((max(lat_span, lon_span), unique_points, len(gg), str(vid)))

        vessel_scores.sort(reverse=True)
        vessel = vessel_scores[0][3]

        title = f"CONTOH {_attack_display_name(attack_name).upper()} | "
        output_prefix = f"example_{attack_name}"

        out_path = plot_spoofing_overlay_from_csv(
            csv_path=csv_path,
            out_dir=attack_dir,
            sample_vessel=vessel,
            attack_type=attack_name,
            max_points=max_points,
            title_prefix=title,
            output_prefix=output_prefix,
        )

        rows.append(
            {
                "attack_type": attack_name,
                "status": "saved",
                "example_original_mmsi": vessel,
                "spoofed_rows": int(
                    len(part[part["original_mmsi"].astype(str) == vessel])
                ),
                "output_dir": str(attack_dir),
                "output_path": str(out_path),
            }
        )

    summary = pd.DataFrame(rows)
    summary_path = _unique_path(out_dir / f"examples_summary_{csv_path.stem}.csv")
    summary.to_csv(summary_path, index=False)

    print(f"[plot_spoofing_examples] Saved {summary_path}")
    return summary_path
