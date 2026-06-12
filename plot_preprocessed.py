from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

from plot_trajectory import _unique_path


def _load_label_map(data: np.lib.npyio.NpzFile) -> Dict[int, str]:
    if "label_map" not in data:
        return {}
    return {int(k): str(v) for k, v in data["label_map"].tolist()}


def _infer_task(npz_path: Path, label_map: Dict[int, str], task: str = "auto") -> str:
    task = str(task or "auto").strip().lower()
    if task != "auto":
        return task

    vals = {str(v).strip().lower() for v in label_map.values()}
    if vals == {"normal", "spoofing"}:
        return "spoofing"
    if vals in [{"normal", "go_dark"}, {"normal", "godark"}]:
        return "godark"
    if vals in [{"normal", "encounter", "loitering"}, {"normal", "transshipment"}, {"normal", "potential_transshipment"}]:
        return "transshipment"

    name = npz_path.name.lower()
    if "spoof" in name:
        return "spoofing"
    if "godark" in name or "go_dark" in name:
        return "godark"
    if "transshipment" in name:
        return "transshipment"
    return "trajectory"


def plot_preprocessed_trajectory_from_npz(
    npz_path: Path,
    out_dir: Path,
    task: str = "auto",
    sample_vessel: str = "",
    max_windows: int = 12,
    only_anomaly: bool = True,
) -> Path:
    """
    Plot window trajectory yang sudah melewati preprocess.

    NPZ lama yang belum punya `coords` perlu dibuat ulang dengan command preprocess
    terbaru. Kolom coords: timestamp, lat, lon, y_point.
    """
    npz_path = Path(npz_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path, allow_pickle=True)
    if "coords" not in data:
        raise ValueError(
            f"{npz_path} belum punya coords. Jalankan ulang preprocess dulu supaya "
            "processed_*.npz menyimpan timestamp/lat/lon/y_point."
        )

    y = data["y"].astype(np.int64)
    groups = data["groups"].astype(str)
    coords = data["coords"].astype(float)
    label_map = _load_label_map(data)
    task_name = _infer_task(npz_path, label_map, task=task)

    if coords.ndim != 3 or coords.shape[-1] < 4:
        raise ValueError(f"coords di {npz_path} tidak valid. Shape={coords.shape}")

    idx = np.arange(len(y))
    if sample_vessel:
        idx = idx[groups == str(sample_vessel)]
    if only_anomaly:
        idx = idx[y[idx] == 1]
    if idx.size == 0 and only_anomaly:
        idx = np.arange(len(y))
        if sample_vessel:
            idx = idx[groups == str(sample_vessel)]

    if idx.size == 0:
        raise ValueError(f"Tidak ada window yang bisa diplot dari {npz_path}")

    if not sample_vessel:
        anomaly_idx = idx[y[idx] == 1]
        source_idx = anomaly_idx if anomaly_idx.size else idx
        vals, cnt = np.unique(groups[source_idx], return_counts=True)
        vessel = str(vals[int(np.argmax(cnt))])
        idx = idx[groups[idx] == vessel]
    else:
        vessel = str(sample_vessel)

    if only_anomaly and np.any(y[idx] == 1):
        idx = idx[y[idx] == 1]

    idx = idx[: max(1, int(max_windows))]

    anomaly_color = "#f85149" if task_name == "spoofing" else "#e3901a" if task_name == "godark" else "#6f42c1"
    anomaly_label = (
        "Spoofing point"
        if task_name == "spoofing"
        else "Go-dark point"
        if task_name == "godark"
        else "Transshipment candidate point"
    )
    title_task = "Spoofing" if task_name == "spoofing" else "Go-Dark" if task_name == "godark" else "Transshipment" if task_name == "transshipment" else task_name.title()

    fig = plt.figure(figsize=(10, 7))
    used_labels = set()

    for n, wi in enumerate(idx, start=1):
        w = coords[wi]
        ts = w[:, 0]
        lat = w[:, 1]
        lon = w[:, 2]
        point_y = w[:, 3] >= 0.5

        order = np.argsort(ts)
        ts = ts[order]
        lat = lat[order]
        lon = lon[order]
        point_y = point_y[order]

        line_label = "Preprocessed window" if "Preprocessed window" not in used_labels else None
        plt.plot(lon, lat, linewidth=0.9, alpha=0.28, color="#8b949e", label=line_label)
        used_labels.add("Preprocessed window")

        normal = ~point_y
        if normal.any():
            label = "Normal point" if "Normal point" not in used_labels else None
            plt.scatter(lon[normal], lat[normal], s=10, alpha=0.45, color="#4493f8", label=label)
            used_labels.add("Normal point")

        if point_y.any():
            label = anomaly_label if anomaly_label not in used_labels else None
            plt.scatter(lon[point_y], lat[point_y], s=22, alpha=0.9, color=anomaly_color, label=label)
            used_labels.add(anomaly_label)

        if task_name == "godark" and len(ts) > 1:
            gaps = np.where((ts[1:] - ts[:-1]) > 3 * 3600)[0]
            for gi in gaps:
                label = "AIS gap in window" if "AIS gap in window" not in used_labels else None
                plt.plot(
                    [lon[gi], lon[gi + 1]],
                    [lat[gi], lat[gi + 1]],
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.75,
                    color=anomaly_color,
                    label=label,
                )
                used_labels.add("AIS gap in window")

    plt.title(f"{title_task} Preprocessed Manipulated Trajectory | vessel={vessel} | windows={len(idx)}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()

    safe_vessel = str(vessel).replace("/", "_").replace("\\", "_")
    out_path = _unique_path(out_dir / f"preprocessed_{task_name}_{npz_path.stem}_{safe_vessel}.png")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    print(f"[plot_preprocessed] Saved {out_path}")
    return out_path
