from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from plot_trajectory import _unique_path


def _load_transshipment_csv(csv_path: Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path} is empty.")
    need = ["event_id", "event_kind", "timestamp", "lat_a", "lon_a", "lat_mid", "lon_mid"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {csv_path}")
    for c in ["timestamp", "lat_a", "lon_a", "lat_b", "lon_b", "lat_mid", "lon_mid", "risk_score", "class_id"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["timestamp", "lat_a", "lon_a"]).copy()


def _pick_event(df: pd.DataFrame, event_id: str = "", prefer_positive: bool = True) -> str:
    if event_id:
        if event_id not in set(df["event_id"].astype(str)):
            raise ValueError(f"event_id={event_id} not found")
        return str(event_id)

    d = df.copy()
    if prefer_positive and "class_id" in d.columns:
        pos = d[pd.to_numeric(d["class_id"], errors="coerce").fillna(0).astype(int) > 0]
        if not pos.empty:
            d = pos

    counts = d.groupby("event_id").size().sort_values(ascending=False)
    if counts.empty:
        raise ValueError("No event_id found in transshipment CSV.")
    return str(counts.index[0])


def _event_title(g: pd.DataFrame) -> str:
    kind = str(g["event_kind"].iloc[0])
    label = str(g.get("label", pd.Series([""])).iloc[0])
    risk = float(pd.to_numeric(g.get("risk_score", pd.Series([0.0])), errors="coerce").fillna(0).max())
    start = int(pd.to_numeric(g["timestamp"], errors="coerce").min())
    end = int(pd.to_numeric(g["timestamp"], errors="coerce").max())
    hours = max(0.0, (end - start) / 3600.0)
    return f"{kind.title()} | {label} | duration={hours:.2f}h | risk={risk:.2f}"


def plot_transshipment_event_from_csv(
    csv_path: Path,
    out_dir: Path,
    event_id: str = "",
    max_points: int = 2000,
) -> Path:
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_transshipment_csv(csv_path)
    chosen = _pick_event(df, event_id=event_id)
    g = df[df["event_id"].astype(str) == str(chosen)].sort_values("timestamp").copy()
    if len(g) > int(max_points):
        idx = np.linspace(0, len(g) - 1, int(max_points)).round().astype(int)
        g = g.iloc[idx].copy()

    kind = str(g["event_kind"].iloc[0]).lower()
    fig = plt.figure(figsize=(10, 7))

    plt.plot(g["lon_a"], g["lat_a"], color="#1f77b4", linewidth=1.4, alpha=0.85, label="Vessel A")
    plt.scatter(g["lon_a"], g["lat_a"], color="#1f77b4", s=12, alpha=0.75)

    has_b = "lat_b" in g.columns and "lon_b" in g.columns and g["lat_b"].notna().any() and g["lon_b"].notna().any()
    if kind == "encounter" and has_b:
        plt.plot(g["lon_b"], g["lat_b"], color="#d62728", linewidth=1.4, alpha=0.85, label="Vessel B")
        plt.scatter(g["lon_b"], g["lat_b"], color="#d62728", s=12, alpha=0.75)
        for _, r in g.iloc[:: max(1, len(g) // 12)].iterrows():
            if np.isfinite(r.get("lat_b", np.nan)) and np.isfinite(r.get("lon_b", np.nan)):
                plt.plot([r["lon_a"], r["lon_b"]], [r["lat_a"], r["lat_b"]], color="#8b949e", linewidth=0.8, alpha=0.35)
    else:
        plt.scatter(g["lon_mid"], g["lat_mid"], color="#e3901a", s=18, alpha=0.75, label="Loitering position")

    plt.scatter(g["lon_mid"].iloc[[0]], g["lat_mid"].iloc[[0]], color="#2ca02c", s=70, marker="o", label="Start")
    plt.scatter(g["lon_mid"].iloc[[-1]], g["lat_mid"].iloc[[-1]], color="#111111", s=70, marker="x", label="End")

    plt.title(_event_title(g))
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()

    safe_id = str(chosen).replace("/", "_").replace("\\", "_")
    out_path = _unique_path(out_dir / f"transshipment_{safe_id}.png")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[plot_transshipment] Saved {out_path}")
    return out_path


def plot_transshipment_examples_from_csv(
    csv_path: Path,
    out_dir: Path,
    num_examples: int = 6,
    max_points: int = 2000,
) -> List[Path]:
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _load_transshipment_csv(csv_path)

    d = df.copy()
    if "class_id" in d.columns:
        pos = d[pd.to_numeric(d["class_id"], errors="coerce").fillna(0).astype(int) > 0]
        if not pos.empty:
            d = pos

    event_ids = d.groupby("event_id").size().sort_values(ascending=False).index.astype(str).tolist()
    event_ids = event_ids[: max(1, int(num_examples))]
    return [
        plot_transshipment_event_from_csv(csv_path, out_dir, event_id=eid, max_points=max_points)
        for eid in event_ids
    ]


def heatmap_transshipment_from_csv(
    csv_path: Path,
    out_dir: Path,
    bins: int = 300,
    log_scale: bool = True,
) -> Path:
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _load_transshipment_csv(csv_path)
    df = df.dropna(subset=["lat_mid", "lon_mid"]).copy()
    if df.empty:
        raise ValueError("No lat_mid/lon_mid values available for heatmap.")

    fig = plt.figure(figsize=(10, 7))
    weights = None
    if log_scale:
        weights = np.ones(len(df), dtype=float)
    norm = LogNorm() if log_scale else None
    plt.hist2d(df["lon_mid"], df["lat_mid"], bins=int(bins), weights=weights, cmap="magma", norm=norm)
    plt.colorbar(label="Candidate density")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Transshipment Candidate Heatmap")
    plt.tight_layout()

    out_path = _unique_path(out_dir / f"transshipment_heatmap_{csv_path.stem}.png")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[heatmap_transshipment] Saved {out_path}")
    return out_path
