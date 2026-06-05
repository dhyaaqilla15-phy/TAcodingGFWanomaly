# train.py
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Iterator, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from model import LSTMClassifier
from split import group_train_val_test_split
from standardize import fit_robust_scaler, apply_scaler
from agg_utils import (
    AggParams,
    confusion_matrix_np,
    metrics_from_cm,
    pick_best_tau_and_agg_by_vessel,
)


class AisSeqDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray | None = None,
        coords: np.ndarray | None = None,
    ):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.groups = None if groups is None else np.asarray(groups).astype(str)
        self.coords = None if coords is None else torch.from_numpy(coords).float()

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        items = [self.X[idx], self.y[idx]]
        if self.groups is not None:
            items.append(self.groups[idx])
        if self.coords is not None:
            items.append(self.coords[idx])
        return tuple(items)


def load_npz(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    groups = data["groups"]
    coords = data["coords"] if "coords" in data.files else None
    lm_arr = data["label_map"]
    label_map = {int(k): str(v) for k, v in lm_arr.tolist()}
    if "scaled" not in data.files:
        print("[train] WARNING: NPZ has no 'scaled' metadata. Rerun preprocess with this code to avoid legacy global-scaler leakage.")
    elif bool(np.asarray(data["scaled"]).item()):
        print("[train] WARNING: NPZ is already scaled. Train-only scaler will be applied on top of it.")
    return X, y, groups, label_map, coords


def pick_device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _mode_int(a: np.ndarray) -> int:
    if a.size == 0:
        return 0
    vals, cnt = np.unique(a.astype(np.int64), return_counts=True)
    return int(vals[int(np.argmax(cnt))])


def _vessel_labels(y: np.ndarray, groups: np.ndarray) -> Dict[str, int]:
    out: Dict[str, int] = {}
    g = np.asarray(groups).astype(str)
    for vid in np.unique(g):
        idx = np.where(g == vid)[0]
        out[str(vid)] = _mode_int(y[idx])
    return out


def _prior_from_vessels(vessel_label: Dict[str, int], num_classes: int) -> np.ndarray:
    counts = np.zeros((num_classes,), dtype=np.float64)
    for _, c in vessel_label.items():
        c = int(c)
        if 0 <= c < num_classes:
            counts[c] += 1.0
    counts[counts < 1] = 1.0
    pi = counts / counts.sum()
    return pi.astype(np.float32)


def _weights_from_vessel_counts(vessel_label: Dict[str, int], num_classes: int, beta: float = 0.9995) -> torch.Tensor:
    counts = np.zeros((num_classes,), dtype=np.float64)
    for _, c in vessel_label.items():
        c = int(c)
        if 0 <= c < num_classes:
            counts[c] += 1.0
    counts[counts < 1] = 1.0
    eff = 1.0 - np.power(beta, counts)
    w = (1.0 - beta) / eff
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)

def _prior_from_window_counts(y: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=num_classes).astype(np.float64)
    counts[counts < 1] = 1.0
    pi = counts / counts.sum()
    return pi.astype(np.float32)


def _weights_from_window_counts(y: np.ndarray, num_classes: int, beta: float = 0.9995) -> torch.Tensor:
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=num_classes).astype(np.float64)
    counts[counts < 1] = 1.0
    eff = 1.0 - np.power(beta, counts)
    w = (1.0 - beta) / eff
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def _has_all_classes_in_vessel_modes(vessel_label: Dict[str, int], num_classes: int) -> bool:
    vals = {int(v) for v in vessel_label.values()}
    return all(c in vals for c in range(num_classes))


def _task_name_from_label_map(label_map: Dict[int, str]) -> str:
    vals = {str(v).strip().lower() for v in label_map.values()}
    if vals == {"normal", "spoofing"}:
        return "spoofing"
    if vals in [{"normal", "go_dark"}, {"normal", "godark"}]:
        return "godark"
    if vals == {"fishing", "not_fishing"}:
        return "fishing"
    return "gear"


def _primary_metric_scope(task_name: str) -> str:
    return "sequence" if task_name in {"spoofing", "godark"} else "vessel"


def _pick_best_tau_by_sequence(
    logits_np: np.ndarray,
    y_np: np.ndarray,
    log_pi: np.ndarray,
    tau_list: List[float],
    num_classes: int,
) -> Tuple[float, Dict[str, float]]:
    best_tau = None
    best_m = None

    for tau in tau_list:
        adj = logits_np - (float(tau) * log_pi.reshape(1, -1))
        pred = np.argmax(adj, axis=1).astype(np.int64)
        cm = confusion_matrix_np(y_np, pred, num_classes)
        m = metrics_from_cm(cm)

        if best_tau is None:
            best_tau, best_m = float(tau), m
        else:
            if m["macro_f1"] > best_m["macro_f1"] + 1e-9:
                best_tau, best_m = float(tau), m
            elif abs(m["macro_f1"] - best_m["macro_f1"]) <= 1e-9 and m["balanced_acc"] > best_m["balanced_acc"] + 1e-9:
                best_tau, best_m = float(tau), m

    return float(best_tau), dict(best_m)


class VesselBalancedBatchSampler:
    def __init__(
        self,
        y: np.ndarray,
        groups: np.ndarray,
        num_classes: int,
        batch_size: int,
        vessels_per_batch: int = 24,
        seed: int = 42,
    ):
        self.y = np.asarray(y).astype(np.int64)
        self.groups = np.asarray(groups).astype(str)
        self.num_classes = int(num_classes)
        self.batch_size = int(batch_size)
        self.vessels_per_batch = int(max(1, vessels_per_batch))
        self.seed = int(seed)

        self.v2idx: Dict[str, np.ndarray] = {}
        for v in np.unique(self.groups):
            self.v2idx[str(v)] = np.where(self.groups == str(v))[0]

        self.v_label = _vessel_labels(self.y, self.groups)
        self.c2v: Dict[int, List[str]] = {c: [] for c in range(self.num_classes)}
        for v, c in self.v_label.items():
            c = int(c)
            if 0 <= c < self.num_classes:
                self.c2v[c].append(v)

        self.num_batches = int(np.ceil(len(self.y) / max(self.batch_size, 1)))

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.RandomState(self.seed + np.random.randint(0, 10_000))

        base = self.vessels_per_batch // self.num_classes
        rem = self.vessels_per_batch % self.num_classes
        v_per_class = [base + (1 if c < rem else 0) for c in range(self.num_classes)]
        v_per_class = [max(1, k) for k in v_per_class]

        win_per_vessel = max(1, self.batch_size // self.vessels_per_batch)

        for _ in range(self.num_batches):
            chosen_vessels: List[str] = []
            for c in range(self.num_classes):
                pool = self.c2v.get(c, [])
                if len(pool) == 0:
                    continue
                k = v_per_class[c]
                vs = rng.choice(pool, size=k, replace=(len(pool) < k)).tolist()
                chosen_vessels.extend(vs)

            if not chosen_vessels:
                yield rng.choice(len(self.y), size=self.batch_size, replace=True).tolist()
                continue

            batch_idx: List[int] = []
            for v in chosen_vessels:
                idxs = self.v2idx.get(v)
                if idxs is None or idxs.size == 0:
                    continue
                pick = rng.choice(idxs, size=win_per_vessel, replace=(idxs.size < win_per_vessel))
                batch_idx.extend(pick.tolist())

            if len(batch_idx) < self.batch_size:
                extra = rng.choice(batch_idx, size=self.batch_size - len(batch_idx), replace=True).tolist()
                batch_idx.extend(extra)
            elif len(batch_idx) > self.batch_size:
                batch_idx = batch_idx[: self.batch_size]

            yield batch_idx


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {}
        self.backup = {}
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if torch.is_tensor(v) and v.dtype.is_floating_point:
                    self.shadow[k] = v.detach().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        sd = model.state_dict()
        for k, v in sd.items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, model: torch.nn.Module):
        self.backup = {}
        sd = model.state_dict()
        for k in self.shadow:
            self.backup[k] = sd[k].detach().clone()
            sd[k].copy_(self.shadow[k])

    @torch.no_grad()
    def restore(self, model: torch.nn.Module):
        sd = model.state_dict()
        for k, v in self.backup.items():
            sd[k].copy_(v)
        self.backup = {}


@dataclass
class AugCfg:
    p_aug: float = 0.40
    p_shift: float = 0.20
    shift_max: int = 10
    p_time_mask: float = 0.12
    time_mask_frac: Tuple[float, float] = (0.05, 0.10)
    p_feat_mask: float = 0.08
    feat_mask_frac: Tuple[float, float] = (0.05, 0.12)
    p_jitter: float = 0.20
    noise_std: float = 0.008


def augment_batch(x: torch.Tensor, cfg: AugCfg) -> torch.Tensor:
    if x.numel() == 0:
        return x
    if torch.rand(1, device=x.device).item() > cfg.p_aug:
        return x

    out = x.clone()
    B, T, Fdim = out.shape

    if cfg.p_shift > 0 and torch.rand(1, device=out.device).item() < cfg.p_shift:
        sh = int(torch.randint(-cfg.shift_max, cfg.shift_max + 1, (1,), device=out.device).item())
        if sh != 0:
            out = torch.roll(out, shifts=sh, dims=1)

    if cfg.p_time_mask > 0 and torch.rand(1, device=out.device).item() < cfg.p_time_mask and T >= 8:
        lo, hi = cfg.time_mask_frac
        frac = float(lo + (hi - lo) * torch.rand(1, device=out.device).item())
        L = max(1, int(T * frac))
        s = int(torch.randint(0, max(T - L + 1, 1), (1,), device=out.device).item())
        out[:, s:s + L, :] = 0.0

    if cfg.p_feat_mask > 0 and torch.rand(1, device=out.device).item() < cfg.p_feat_mask and Fdim >= 4:
        lo, hi = cfg.feat_mask_frac
        frac = float(lo + (hi - lo) * torch.rand(1, device=out.device).item())
        K = max(1, int(Fdim * frac))
        cols = torch.randperm(Fdim, device=out.device)[:K]
        out[:, :, cols] = 0.0

    if cfg.p_jitter > 0 and torch.rand(1, device=out.device).item() < cfg.p_jitter:
        out = out + torch.randn_like(out) * cfg.noise_std

    return out


def focal_ce_from_logits(
    logits: torch.Tensor,
    y: torch.Tensor,
    class_w: torch.Tensor | None,
    gamma: float = 1.5,
) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=1)
    pt = torch.exp(logp.gather(1, y.view(-1, 1)).squeeze(1)).clamp(1e-6, 1.0)
    ce = F.nll_loss(logp, y, weight=class_w, reduction="none")
    loss = ((1.0 - pt) ** float(gamma)) * ce
    return loss.mean()


def haversine_km_torch(pred_latlon: torch.Tensor, true_latlon: torch.Tensor) -> torch.Tensor:
    pred = pred_latlon.float()
    true = true_latlon.float()

    lat1 = pred[:, 0].clamp(-90.0, 90.0) * (math.pi / 180.0)
    lon1 = pred[:, 1].clamp(-180.0, 180.0) * (math.pi / 180.0)
    lat2 = true[:, 0].clamp(-90.0, 90.0) * (math.pi / 180.0)
    lon2 = true[:, 1].clamp(-180.0, 180.0) * (math.pi / 180.0)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        torch.sin(dlat * 0.5).pow(2)
        + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon * 0.5).pow(2)
    )
    a = a.clamp(0.0, 1.0)
    return 6371.0088 * 2.0 * torch.atan2(torch.sqrt(a), torch.sqrt((1.0 - a).clamp_min(1e-12)))


def haversine_aux_loss(
    pred_latlon: torch.Tensor | None,
    coords: torch.Tensor | None,
    scale_km: float = 1000.0,
) -> torch.Tensor:
    if pred_latlon is None:
        raise ValueError("pred_latlon is None; build the model with predict_coords=True.")
    if coords is None:
        return pred_latlon.sum() * 0.0

    target = coords[:, -1, 1:3].float()
    valid = (
        torch.isfinite(target).all(dim=1)
        & target[:, 0].ge(-90.0)
        & target[:, 0].le(90.0)
        & target[:, 1].ge(-180.0)
        & target[:, 1].le(180.0)
    )
    if not bool(valid.any()):
        return pred_latlon.sum() * 0.0

    km = haversine_km_torch(pred_latlon[valid], target[valid])
    denom = math.log1p(max(float(scale_km), 1.0))
    return torch.log1p(km).mean() / denom


def train_from_npz(
    data_npz: Path,
    out_dir: Path,
    epochs: int = 320,
    batch_size: int = 128,
    lr: float = 2.5e-4,
    hidden_size: int = 384,
    num_layers: int = 2,
    input_proj_dim: int | None = 256,
    embed_dim: int | None = 512,
    dropout: float = 0.40,
    bidirectional: bool = True,
    optimizer_name: str = "adamw",
    weight_decay: float = 1.3e-3,
    sgd_momentum: float = 0.9,
    attention_heads: int = 4,
    attention_layers: int = 1,
    geo_aux_weight: float = 0.03,
    geo_aux_scale_km: float = 1000.0,
    test_size: float = 0.2,
    val_size: float = 0.15,
    random_state: int = 42,
    device: str = "auto",
    early_stop_patience: int = 90,
    use_ema: bool = True,
    ema_decay: float = 0.999,
    vessels_per_batch: int = 24,
    aug_cfg: AugCfg = AugCfg(),
    use_focal: bool = True,
    focal_gamma: float = 1.2,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_t0 = time.perf_counter()

    dev = pick_device(device)
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    X, y, groups, label_map, coords = load_npz(Path(data_npz))
    y = y.astype(np.int64)
    coords = None if coords is None else coords.astype(np.float32, copy=False)
    if coords is not None and int(coords.shape[0]) != int(X.shape[0]):
        print("[train] WARNING: coords length does not match X; Haversine auxiliary loss is disabled.")
        coords = None
    num_classes = int(len(set(y.tolist())))
    input_size = int(X.shape[-1])
    task_name = _task_name_from_label_map(label_map)
    metric_scope = _primary_metric_scope(task_name)
    use_geo_aux = bool(float(geo_aux_weight) > 0.0 and coords is not None)
    if float(geo_aux_weight) > 0.0 and coords is None:
        print("[train] WARNING: NPZ has no coords; Haversine auxiliary loss is disabled.")

    print(f"[train] task={task_name} primary_metric_scope={metric_scope}")
    print(
        f"[train] optimizer={optimizer_name} lr={lr} batch_size={batch_size} "
        f"epochs={epochs} hidden_size={hidden_size} num_layers={num_layers} "
        f"input_proj_dim={input_proj_dim} embed_dim={embed_dim} dropout={dropout} "
        f"attention_heads={attention_heads} attention_layers={attention_layers} "
        f"geo_aux_weight={(float(geo_aux_weight) if use_geo_aux else 0.0)}"
    )

    split = group_train_val_test_split(
        X, y, groups,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        stratify_groups=True,
        max_tries=400,
    )
    np.savez_compressed(
        out_dir / "split_indices.npz",
        train_idx=split.train_idx,
        val_idx=split.val_idx,
        test_idx=split.test_idx,
    )

    scaler_path = out_dir / "scaler.joblib"
    scaler = fit_robust_scaler(X[split.train_idx], scaler_path)

    X_train = apply_scaler(X[split.train_idx], scaler)
    X_val = apply_scaler(X[split.val_idx], scaler)
    y_train, g_train = y[split.train_idx], groups[split.train_idx]
    y_val, g_val = y[split.val_idx], groups[split.val_idx]

    print(
        f"[train] split windows train={len(split.train_idx)} val={len(split.val_idx)} test={len(split.test_idx)}"
    )
    print(
        f"[train] split vessels train={np.unique(groups[split.train_idx]).size} "
        f"val={np.unique(groups[split.val_idx]).size} test={np.unique(groups[split.test_idx]).size}"
    )
    print(f"[train] scaler fit on train split only -> {scaler_path}")

    vlab = _vessel_labels(y_train, np.asarray(g_train).astype(str))
    use_vessel_balancing = _has_all_classes_in_vessel_modes(vlab, num_classes)
    if use_vessel_balancing:
        pri_np = _prior_from_vessels(vlab, num_classes)
        class_w = _weights_from_vessel_counts(vlab, num_classes).to(dev)
    else:
        # Untuk task spoofing, satu vessel bisa punya window Normal dan Spoofing sekaligus.
        # Kalau label mode per-vessel tidak mencakup semua kelas, fallback ke distribusi window
        # supaya kelas spoofing tidak hilang dari class weight / sampler.
        print("[train] vessel-mode labels do not cover all classes; using window-level priors/weights.")
        pri_np = _prior_from_window_counts(y_train, num_classes)
        class_w = _weights_from_window_counts(y_train, num_classes).to(dev)

    pri = torch.tensor(pri_np, dtype=torch.float32, device=dev)
    log_pi = torch.log(pri.clamp_min(1e-8))
    log_pi_np = log_pi.detach().cpu().numpy().astype(np.float32)

    # Sweep kasar + refinement rendah (sering jadi sweet spot).
    tau_coarse = [round(0.3 * i, 2) for i in range(0, 18)]  # 0.0..5.1
    tau_fine = [round(0.1 * i, 2) for i in range(1, 11)]    # 0.1..1.0
    tau_candidates = sorted(set(tau_coarse + tau_fine))

    # Reduced grid: fokus pada kombinasi yang paling stabil
    # Mengurangi overfitting pada validation set dengan limiting search space
    agg_grid = [
        AggParams(keep_frac=0.12, min_keep=6,  weight_power=3.0, conf_mode="maxprob_margin2", agg_method="mean_logit"),
        AggParams(keep_frac=0.15, min_keep=8,  weight_power=3.0, conf_mode="maxprob_margin2", agg_method="mean_logit"),
        AggParams(keep_frac=0.18, min_keep=8,  weight_power=3.0, conf_mode="maxprob_margin2", agg_method="mean_logit"),
        AggParams(keep_frac=0.15, min_keep=8,  weight_power=2.0, conf_mode="maxprob_margin",  agg_method="mean_prob"),
    ]

    coords_train = coords[split.train_idx] if use_geo_aux else None
    train_ds = AisSeqDataset(X_train, y_train, coords=coords_train)
    val_ds = AisSeqDataset(X_val, y_val, groups=g_val)

    if use_vessel_balancing:
        batch_sampler = VesselBalancedBatchSampler(
            y=y_train,
            groups=g_train,
            num_classes=num_classes,
            batch_size=batch_size,
            vessels_per_batch=vessels_per_batch,
            seed=random_state,
        )
        train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=0)
    else:
        cw_cpu = class_w.detach().cpu().numpy().astype(np.float64)
        sample_weights = cw_cpu[np.asarray(y_train, dtype=np.int64)]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=int(len(sample_weights)),
            replacement=True,
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)

    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    def _build_model() -> LSTMClassifier:
        return LSTMClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            bidirectional=bidirectional,
            input_proj_dim=input_proj_dim,
            embed_dim=embed_dim,
            attention_heads=int(attention_heads),
            attention_layers=int(attention_layers),
            predict_coords=bool(use_geo_aux),
        )

    used_cudnn_fallback = False
    model = _build_model()
    if dev.type == "cuda":
        try:
            model = model.to(dev)
        except RuntimeError as e:
            if "CUDNN_STATUS_INTERNAL_ERROR" not in str(e):
                raise
            print("[train] cuDNN internal error on model.to(cuda), retrying with cuDNN disabled.")
            torch.backends.cudnn.enabled = False
            torch.cuda.empty_cache()
            used_cudnn_fallback = True
            model = _build_model().to(dev)
    else:
        model = model.to(dev)

    optimizer_name = str(optimizer_name).strip().lower()
    if optimizer_name == "adamw":
        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optim = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=sgd_momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )
    else:
        raise ValueError(f"Unknown optimizer_name='{optimizer_name}'. Use one of: adamw, adam, sgd.")

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optim,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.12,
        div_factor=10.0,
        final_div_factor=80.0,
        anneal_strategy="cos",
    )

    use_amp = (dev.type == "cuda")
    if use_amp:
        scaler = torch.amp.GradScaler("cuda")
        autocast_ctx = lambda: torch.amp.autocast("cuda", dtype=torch.float16)
    else:
        scaler = None
        autocast_ctx = lambda: torch.autocast("cpu", enabled=False)

    ema = EMA(model, decay=ema_decay) if use_ema else None

    best_macro = -1.0
    best_epoch = 0
    best_tau = 0.0
    best_agg = agg_grid[0]

    best_path = out_dir / "model.pt"
    hist_rows = []
    no_improve = 0
    best_balanced_acc = 0.0
    best_accuracy = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        tr_losses = []
        tr_cls_losses = []
        tr_geo_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]"):
            while True:
                try:
                    xb = batch[0]
                    yb = batch[1]
                    cb = batch[2] if (use_geo_aux and len(batch) > 2) else None

                    xb = xb.to(dev, non_blocking=True)
                    yb = yb.to(dev, non_blocking=True)
                    if cb is not None:
                        cb = cb.to(dev, non_blocking=True)

                    xb = augment_batch(xb, aug_cfg)
                    optim.zero_grad(set_to_none=True)

                    if use_amp:
                        with autocast_ctx():
                            if use_geo_aux:
                                logits, pred_latlon = model.forward_with_aux(xb)
                            else:
                                logits, pred_latlon = model(xb), None

                            if use_focal:
                                loss_cls = focal_ce_from_logits(logits, yb, class_w, gamma=float(focal_gamma))
                            else:
                                loss_cls = F.cross_entropy(logits, yb, weight=class_w, label_smoothing=0.02)

                        loss_geo = logits.new_tensor(0.0)
                        if use_geo_aux and cb is not None and pred_latlon is not None:
                            with torch.amp.autocast("cuda", enabled=False):
                                loss_geo = haversine_aux_loss(
                                    pred_latlon.float(),
                                    cb.float(),
                                    scale_km=float(geo_aux_scale_km),
                                )
                        loss = loss_cls + (float(geo_aux_weight) * loss_geo)

                        scaler.scale(loss).backward()
                        scaler.unscale_(optim)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scale_before = scaler.get_scale()
                        scaler.step(optim)
                        scaler.update()
                        if scaler.get_scale() >= scale_before:
                            scheduler.step()
                    else:
                        if use_geo_aux:
                            logits, pred_latlon = model.forward_with_aux(xb)
                        else:
                            logits, pred_latlon = model(xb), None

                        if use_focal:
                            loss_cls = focal_ce_from_logits(logits, yb, class_w, gamma=float(focal_gamma))
                        else:
                            loss_cls = F.cross_entropy(logits, yb, weight=class_w, label_smoothing=0.02)
                        loss_geo = logits.new_tensor(0.0)
                        if use_geo_aux and cb is not None and pred_latlon is not None:
                            loss_geo = haversine_aux_loss(
                                pred_latlon,
                                cb,
                                scale_km=float(geo_aux_scale_km),
                            )
                        loss = loss_cls + (float(geo_aux_weight) * loss_geo)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optim.step()
                        scheduler.step()

                    if ema is not None:
                        ema.update(model)

                    tr_losses.append(float(loss.item()))
                    tr_cls_losses.append(float(loss_cls.item()))
                    tr_geo_losses.append(float(loss_geo.item()))
                    break
                except RuntimeError as err:
                    if (dev.type != "cuda") or ("CUDNN_STATUS_INTERNAL_ERROR" not in str(err)) or used_cudnn_fallback:
                        raise
                    print("[train] cuDNN internal error during training step, retrying with cuDNN disabled.")
                    torch.backends.cudnn.enabled = False
                    torch.cuda.empty_cache()
                    used_cudnn_fallback = True

        train_loss = float(np.mean(tr_losses)) if tr_losses else float("nan")
        train_cls_loss = float(np.mean(tr_cls_losses)) if tr_cls_losses else float("nan")
        train_geo_loss = float(np.mean(tr_geo_losses)) if tr_geo_losses else 0.0

        # VAL:
        # - gear/fishing: metric utama tetap vessel-level
        # - spoofing/go-dark: metric utama sequence-level karena satu vessel bisa berisi
        #   jauh lebih banyak window normal daripada window anomali
        model.eval()
        if ema is not None:
            ema.apply_shadow(model)

        logits_chunks = []
        y_chunks = []
        g_chunks = []

        with torch.no_grad():
            for xb, yb, gb in val_loader:
                while True:
                    try:
                        xb = xb.to(dev, non_blocking=True)
                        if use_amp and dev.type == "cuda":
                            with torch.amp.autocast("cuda", dtype=torch.float16):
                                lg = model(xb)
                        else:
                            lg = model(xb)

                        logits_chunks.append(lg.detach().cpu().float().numpy())
                        y_chunks.append(yb.detach().cpu().numpy().astype(np.int64))
                        g_chunks.append(np.asarray(gb, dtype=object))
                        break
                    except RuntimeError as err:
                        if (dev.type != "cuda") or ("CUDNN_STATUS_INTERNAL_ERROR" not in str(err)) or used_cudnn_fallback:
                            raise
                        print("[train] cuDNN internal error during validation step, retrying with cuDNN disabled.")
                        torch.backends.cudnn.enabled = False
                        torch.cuda.empty_cache()
                        used_cudnn_fallback = True

        logits_np = np.concatenate(logits_chunks, axis=0)
        y_np = np.concatenate(y_chunks, axis=0)
        g_np = np.concatenate(g_chunks, axis=0)

        if metric_scope == "sequence":
            tau_ep, m_ep = _pick_best_tau_by_sequence(
                logits_np=logits_np,
                y_np=y_np,
                log_pi=log_pi_np,
                tau_list=tau_candidates,
                num_classes=num_classes,
            )
            agg_ep = agg_grid[0]
        else:
            tau_ep, agg_ep, m_ep = pick_best_tau_and_agg_by_vessel(
                logits_np=logits_np,
                y_np=y_np,
                g_np=g_np,
                log_pi=log_pi_np,
                tau_list=tau_candidates,
                agg_grid=agg_grid,
                num_classes=num_classes,
            )

        if ema is not None:
            ema.restore(model)

        lr_now = float(optim.param_groups[0]["lr"])
        if metric_scope == "sequence":
            print(
                f"[epoch {epoch}] train_loss={train_loss:.4f} cls={train_cls_loss:.4f} geo={train_geo_loss:.4f} "
                f"VAL(seq) macro_f1={m_ep['macro_f1']:.4f} bal_acc={m_ep['balanced_acc']:.4f} acc={m_ep['accuracy']:.4f} "
                f"tau={tau_ep:.2f} lr={lr_now:.2e}"
            )
        else:
            print(
                f"[epoch {epoch}] train_loss={train_loss:.4f} cls={train_cls_loss:.4f} geo={train_geo_loss:.4f} "
                f"VAL(vessel) macro_f1={m_ep['macro_f1']:.4f} bal_acc={m_ep['balanced_acc']:.4f} acc={m_ep['accuracy']:.4f} "
                f"tau={tau_ep:.2f} agg=({agg_ep.agg_method}, keep={agg_ep.keep_frac}, min={agg_ep.min_keep}, wp={agg_ep.weight_power}, conf={agg_ep.conf_mode}) "
                f"lr={lr_now:.2e}"
            )

        hist_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_cls_loss": train_cls_loss,
                "train_geo_loss": train_geo_loss,
                "task": task_name,
                "val_metric_scope": metric_scope,
                "val_macro_f1": m_ep["macro_f1"],
                "val_balanced_acc": m_ep["balanced_acc"],
                "val_acc": m_ep["accuracy"],
                "val_macro_f1_seq": (m_ep["macro_f1"] if metric_scope == "sequence" else None),
                "val_balanced_acc_seq": (m_ep["balanced_acc"] if metric_scope == "sequence" else None),
                "val_acc_seq": (m_ep["accuracy"] if metric_scope == "sequence" else None),
                "val_macro_f1_vessel": (m_ep["macro_f1"] if metric_scope == "vessel" else None),
                "val_balanced_acc_vessel": (m_ep["balanced_acc"] if metric_scope == "vessel" else None),
                "val_acc_vessel": (m_ep["accuracy"] if metric_scope == "vessel" else None),
                "lr": lr_now,
                "best_tau": float(tau_ep),
                "agg_keep_frac": float(agg_ep.keep_frac),
                "agg_min_keep": int(agg_ep.min_keep),
                "agg_weight_power": float(agg_ep.weight_power),
                "agg_conf_mode": str(agg_ep.conf_mode),
                "agg_method": str(agg_ep.agg_method),
            }
        )

        improved = (m_ep["macro_f1"] > best_macro + 1e-6)
        if improved:
            best_macro = float(m_ep["macro_f1"])
            best_epoch = int(epoch)
            best_tau = float(tau_ep)
            best_agg = agg_ep
            best_balanced_acc = float(m_ep["balanced_acc"])
            best_accuracy = float(m_ep["accuracy"])
            no_improve = 0

            logit_adjust = (best_tau * log_pi).detach().cpu().numpy().astype(np.float32)

            # Metric val dihitung pakai EMA shadow; simpan checkpoint dengan bobot yang sama.
            if ema is not None:
                ema.apply_shadow(model)

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_size": input_size,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "input_proj_dim": input_proj_dim,
                    "embed_dim": embed_dim,
                    "dropout": dropout,
                    "bidirectional": bidirectional,
                    "attention_heads": int(attention_heads),
                    "attention_layers": int(attention_layers),
                    "predict_coords": bool(use_geo_aux),
                    "geo_aux_weight": float(geo_aux_weight) if use_geo_aux else 0.0,
                    "geo_aux_scale_km": float(geo_aux_scale_km),
                    "num_classes": num_classes,
                    "label_map": label_map,
                    "task": task_name,
                    "primary_metric_scope": metric_scope,
                    "optimizer_name": optimizer_name,
                    "weight_decay": float(weight_decay),
                    "sgd_momentum": float(sgd_momentum),
                    "scaler_path": str(scaler_path),
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": float(best_macro),
                    "best_val_balanced_acc": float(best_balanced_acc),
                    "best_val_accuracy": float(best_accuracy),
                    "tau": float(best_tau),
                    "priors": pri.detach().cpu().numpy().astype(np.float32),
                    "logit_adjust": logit_adjust,
                    "agg_keep_frac": float(best_agg.keep_frac),
                    "agg_min_keep": int(best_agg.min_keep),
                    "agg_weight_power": float(best_agg.weight_power),
                    "agg_conf_mode": str(best_agg.conf_mode),
                    "agg_method": str(best_agg.agg_method),
                },
                best_path,
            )
            if ema is not None:
                ema.restore(model)
            print(f"[train] new BEST epoch={epoch} macro_f1({metric_scope})={best_macro:.4f} saved -> {best_path}")
        else:
            no_improve += 1

        if no_improve >= early_stop_patience:
            print(f"[train] Early stopping: no improvement for {early_stop_patience} epochs.")
            break

    (out_dir / "history.json").write_text(json.dumps(hist_rows, indent=2), encoding="utf-8")
    (out_dir / "best_epoch.json").write_text(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "task": task_name,
                "primary_metric_scope": metric_scope,
                "optimizer_name": optimizer_name,
                "weight_decay": float(weight_decay),
                "sgd_momentum": float(sgd_momentum),
                "attention_heads": int(attention_heads),
                "attention_layers": int(attention_layers),
                "geo_aux_weight": float(geo_aux_weight) if use_geo_aux else 0.0,
                "geo_aux_scale_km": float(geo_aux_scale_km),
                "best_val_macro_f1": best_macro,
                "best_val_balanced_acc": best_balanced_acc,
                "best_val_accuracy": best_accuracy,
                "tau": best_tau,
                "agg_keep_frac": float(best_agg.keep_frac),
                "agg_min_keep": int(best_agg.min_keep),
                "agg_weight_power": float(best_agg.weight_power),
                "agg_conf_mode": str(best_agg.conf_mode),
                "agg_method": str(best_agg.agg_method),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    train_duration_seconds = float(time.perf_counter() - train_t0)
    (out_dir / "train_config.json").write_text(
        json.dumps(
            {
                "task": task_name,
                "primary_metric_scope": metric_scope,
                "data_npz": str(data_npz),
                "device": str(dev),
                "epochs": int(epochs),
                "batch_size": int(batch_size),
                "lr": float(lr),
                "hidden_size": int(hidden_size),
                "num_layers": int(num_layers),
                "input_proj_dim": (None if input_proj_dim is None else int(input_proj_dim)),
                "embed_dim": (None if embed_dim is None else int(embed_dim)),
                "dropout": float(dropout),
                "bidirectional": bool(bidirectional),
                "optimizer_name": optimizer_name,
                "weight_decay": float(weight_decay),
                "sgd_momentum": float(sgd_momentum),
                "attention_heads": int(attention_heads),
                "attention_layers": int(attention_layers),
                "geo_aux_weight": float(geo_aux_weight) if use_geo_aux else 0.0,
                "geo_aux_scale_km": float(geo_aux_scale_km),
                "test_size": float(test_size),
                "val_size": float(val_size),
                "random_state": int(random_state),
                "early_stop_patience": int(early_stop_patience),
                "use_ema": bool(use_ema),
                "ema_decay": float(ema_decay),
                "use_focal": bool(use_focal),
                "focal_gamma": float(focal_gamma),
                "train_duration_seconds": train_duration_seconds,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[train] duration_seconds={train_duration_seconds:.2f}")
    print(f"[train] Saved best model: {best_path}")
