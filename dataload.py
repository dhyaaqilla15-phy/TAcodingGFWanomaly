from __future__ import annotations

from pathlib import Path
import pandas as pd

_COL_ALIASES = {
    "mmsi": ["mmsi", "MMSI", "vessel_id", "vesselid", "ship_id", "id"],
    "timestamp": [
        "timestamp",
        "ts",
        "time",
        "t",
        "epoch",
        "datetime",
        "base_datetime",
        "basedatetime",
        "BaseDateTime",
    ],
    "lat": ["lat", "latitude", "LAT", "y"],
    "lon": ["lon", "lng", "longitude", "LON", "x"],
    "speed": ["speed", "sog", "SOG", "knots", "Speed"],
    "course": ["course", "cog", "COG", "Course"],
    "distance_from_shore": [
        "distance_from_shore",
        "dist_shore",
        "shore_dist",
        "distance_to_shore",
    ],
    "distance_from_port": ["distance_from_port", "dist_port", "port_dist", "distance_to_port"],
    "is_fishing": ["is_fishing", "fishing", "label", "isFishing"],
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.lower() for c in df.columns})

    rename = {}
    for std, cands in _COL_ALIASES.items():
        for c in cands:
            c = c.lower()
            if c in df.columns:
                rename[c] = std
                break

    df = df.rename(columns=rename)

    # timestamp string -> epoch seconds
    if "timestamp" in df.columns and df["timestamp"].dtype == "object":
        t = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["timestamp"] = (t.view("int64") // 10**9).astype("Int64")

    # numeric timestamp: detect ms -> seconds
    if "timestamp" in df.columns and df["timestamp"].dtype != "object":
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        if ts.notna().any():
            med = float(ts.dropna().median())
            if med > 1e11:  # ms epoch
                df["timestamp"] = (ts / 1000.0).round().astype("Int64")
            else:
                df["timestamp"] = ts.round().astype("Int64")

    return df


def read_ais_csv(path: Path, limit_rows: int = 0, chunksize: int = 0) -> pd.DataFrame:
    path = Path(path)

    if chunksize and chunksize > 0:
        dfs = []
        read_rows = 0
        for chunk in pd.read_csv(path, chunksize=chunksize):
            chunk = _normalize_columns(chunk)
            dfs.append(chunk)
            read_rows += len(chunk)
            if limit_rows and read_rows >= limit_rows:
                break

        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        if limit_rows and len(df) > limit_rows:
            df = df.iloc[:limit_rows].copy()
        return df

    df = pd.read_csv(path)
    df = _normalize_columns(df)
    if limit_rows and limit_rows > 0:
        df = df.iloc[:limit_rows].copy()
    return df


def infer_label_from_filename(path: Path) -> str:
    return Path(path).stem.strip().lower()
