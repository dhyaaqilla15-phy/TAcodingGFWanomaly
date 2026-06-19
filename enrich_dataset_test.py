from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "Dataset_Test"
OUTPUT_DIR = ROOT / "Dataset_Test_Enriched"
NE_DIR = ROOT / "data_external" / "naturalearth"
COAST_SHP = NE_DIR / "ne_10m_coastline" / "ne_10m_coastline.shp"
PORTS_SHP = NE_DIR / "ne_10m_ports" / "ne_10m_ports.shp"

EARTH_RADIUS_KM = 6371.0088


def _iter_shp_records(path: Path):
    data = path.read_bytes()
    offset = 100
    while offset + 8 <= len(data):
        _rec_no, content_len_words = struct.unpack(">2i", data[offset : offset + 8])
        offset += 8
        content_len = content_len_words * 2
        rec = data[offset : offset + content_len]
        offset += content_len
        if len(rec) >= 4:
            yield rec


def _read_point_shp(path: Path) -> np.ndarray:
    pts: list[tuple[float, float]] = []
    for rec in _iter_shp_records(path):
        shape_type = struct.unpack("<i", rec[:4])[0]
        if shape_type == 1 and len(rec) >= 20:
            x, y = struct.unpack("<2d", rec[4:20])
            pts.append((float(y), float(x)))
    return np.asarray(pts, dtype=np.float64)


def _read_polyline_vertices(path: Path, densify_deg: float = 0.05) -> np.ndarray:
    pts: list[tuple[float, float]] = []
    for rec in _iter_shp_records(path):
        shape_type = struct.unpack("<i", rec[:4])[0]
        if shape_type not in {3, 13, 23}:
            continue
        num_parts, num_points = struct.unpack("<2i", rec[36:44])
        parts_start = 44
        points_start = parts_start + 4 * num_parts
        parts = list(struct.unpack(f"<{num_parts}i", rec[parts_start:points_start]))
        parts.append(num_points)
        raw = np.frombuffer(rec[points_start : points_start + 16 * num_points], dtype="<f8")
        xy = raw.reshape(-1, 2)

        for start, end in zip(parts[:-1], parts[1:]):
            part = xy[start:end]
            if len(part) == 0:
                continue
            pts.extend((float(lat), float(lon)) for lon, lat in part)
            for a, b in zip(part[:-1], part[1:]):
                lon1, lat1 = a
                lon2, lat2 = b
                steps = int(max(abs(lon2 - lon1), abs(lat2 - lat1)) / densify_deg)
                if steps <= 1:
                    continue
                for j in range(1, steps):
                    t = j / steps
                    pts.append((float(lat1 + (lat2 - lat1) * t), float(lon1 + (lon2 - lon1) * t)))

    return np.asarray(pts, dtype=np.float64)


def _latlon_to_unit(lat_lon: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(lat_lon[:, 0])
    lon = np.deg2rad(lat_lon[:, 1])
    clat = np.cos(lat)
    return np.column_stack((clat * np.cos(lon), clat * np.sin(lon), np.sin(lat)))


def _query_haversine_km(tree: cKDTree, ref_lat_lon: np.ndarray, query_lat_lon: np.ndarray) -> np.ndarray:
    q_unit = _latlon_to_unit(query_lat_lon)
    _dist, idx = tree.query(q_unit, k=1)
    nearest = ref_lat_lon[idx]
    lat1 = np.deg2rad(query_lat_lon[:, 0])
    lon1 = np.deg2rad(query_lat_lon[:, 1])
    lat2 = np.deg2rad(nearest[:, 0])
    lon2 = np.deg2rad(nearest[:, 1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return (2 * EARTH_RADIUS_KM * np.arcsin(np.minimum(1.0, np.sqrt(a)))).astype("float32")


def _gear_filename(path: Path) -> str:
    name = path.name
    if name.startswith("purse_seine"):
        return "purse_seines.csv"
    for label in ["drifting_longlines", "fixed_gear", "trawlers"]:
        if name.startswith(label):
            return f"{label}.csv"
    return path.name


def main() -> None:
    if not INPUT_DIR.exists():
        raise SystemExit(f"Input folder not found: {INPUT_DIR}")
    if not COAST_SHP.exists() or not PORTS_SHP.exists():
        raise SystemExit("Natural Earth shapefiles not found. Download/extract coastline and ports first.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[enrich] loading Natural Earth coastline...")
    coast_lat_lon = _read_polyline_vertices(COAST_SHP)
    print(f"[enrich] coastline reference points: {len(coast_lat_lon):,}")
    coast_tree = cKDTree(_latlon_to_unit(coast_lat_lon))

    print("[enrich] loading Natural Earth ports...")
    port_lat_lon = _read_point_shp(PORTS_SHP)
    print(f"[enrich] port reference points: {len(port_lat_lon):,}")
    port_tree = cKDTree(_latlon_to_unit(port_lat_lon))

    for src in sorted(INPUT_DIR.glob("*.csv")):
        df = pd.read_csv(src)
        required = {"mmsi", "timestamp", "lat", "lon", "speed", "course", "source"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{src.name} missing required columns: {missing}")

        lat_lon = df[["lat", "lon"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        ok = np.isfinite(lat_lon).all(axis=1)

        df["distance_from_shore"] = -1.0
        df["distance_from_port"] = -1.0
        if ok.any():
            df.loc[ok, "distance_from_shore"] = _query_haversine_km(coast_tree, coast_lat_lon, lat_lon[ok]) * 1000.0
            df.loc[ok, "distance_from_port"] = _query_haversine_km(port_tree, port_lat_lon, lat_lon[ok]) * 1000.0

        df["speed"] = pd.to_numeric(df["speed"], errors="coerce").fillna(0.0).astype("float32")
        df["course"] = pd.to_numeric(df["course"], errors="coerce").fillna(0.0).astype("float32")
        speed = df["speed"]
        df["is_fishing"] = ((speed >= 0.5) & (speed <= 12.0)).astype("int8")

        out_cols = [
            "mmsi",
            "timestamp",
            "distance_from_shore",
            "distance_from_port",
            "speed",
            "course",
            "lat",
            "lon",
            "is_fishing",
            "source",
        ]
        extra_cols = [c for c in df.columns if c not in out_cols]
        out = df[out_cols + extra_cols]
        dst = OUTPUT_DIR / _gear_filename(src)
        out.to_csv(dst, index=False)
        print(
            f"[enrich] wrote {dst.relative_to(ROOT)} rows={len(out):,} "
            f"shore_m=({out['distance_from_shore'].min():.2f},{out['distance_from_shore'].max():.2f}) "
            f"port_m=({out['distance_from_port'].min():.2f},{out['distance_from_port'].max():.2f}) "
            f"is_fishing_rate={out['is_fishing'].mean():.3f}"
        )


if __name__ == "__main__":
    main()
