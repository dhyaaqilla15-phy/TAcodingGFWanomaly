from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict

import numpy as np
import torch


@dataclass(frozen=True)
class DeterminismConfig:
    enabled: bool = True
    warn_only: bool = True
    cublas_workspace_config: str = ":4096:8"
    cudnn_benchmark: bool = False
    cudnn_deterministic: bool = True
    use_deterministic_algorithms: bool = True


DEFAULT_DETERMINISM_CONFIG = DeterminismConfig()


def apply_determinism(
    seed: int,
    enabled: bool = DEFAULT_DETERMINISM_CONFIG.enabled,
    warn_only: bool = DEFAULT_DETERMINISM_CONFIG.warn_only,
) -> Dict[str, Any]:
    """Apply one training seed policy and return the exact settings used."""
    seed = int(seed)
    cfg = DeterminismConfig(enabled=bool(enabled), warn_only=bool(warn_only))

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if cfg.enabled:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", cfg.cublas_workspace_config)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False if cfg.enabled else True
    torch.backends.cudnn.deterministic = bool(cfg.enabled)

    deterministic_algorithms_applied = False
    deterministic_algorithms_error = None
    try:
        torch.use_deterministic_algorithms(
            bool(cfg.enabled and cfg.use_deterministic_algorithms),
            warn_only=bool(cfg.warn_only),
        )
        deterministic_algorithms_applied = True
    except TypeError:
        try:
            torch.use_deterministic_algorithms(bool(cfg.enabled and cfg.use_deterministic_algorithms))
            deterministic_algorithms_applied = True
        except Exception as exc:
            deterministic_algorithms_error = str(exc)
    except Exception as exc:
        deterministic_algorithms_error = str(exc)

    settings = asdict(cfg)
    settings.update(
        {
            "seed": seed,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "torch_deterministic_algorithms_applied": deterministic_algorithms_applied,
            "torch_deterministic_algorithms_error": deterministic_algorithms_error,
            "torch_cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "torch_cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        }
    )
    return settings
