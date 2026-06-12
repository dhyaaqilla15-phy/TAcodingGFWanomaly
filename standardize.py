from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
from sklearn.preprocessing import RobustScaler


def fit_robust_scaler(X: np.ndarray, save_path: Path) -> RobustScaler:
    """
    X shape: (N, T, F)  
    Fit scaler di seluruh titik (N*T, F)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    N, T, F = X.shape
    X2 = X.reshape(N * T, F)
    scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
    scaler.fit(X2)
    joblib.dump(scaler, save_path)
    return scaler


def load_scaler(path: Path) -> RobustScaler:
    return joblib.load(Path(path))


def apply_scaler(X: np.ndarray, scaler: RobustScaler, clip_value: float = 50.0) -> np.ndarray:
    N, T, F = X.shape
    X2 = X.reshape(N * T, F)
    Xs = scaler.transform(X2).reshape(N, T, F).astype(np.float32)
    if clip_value and float(clip_value) > 0.0:
        Xs = np.clip(Xs, -float(clip_value), float(clip_value)).astype(np.float32)
    return Xs
