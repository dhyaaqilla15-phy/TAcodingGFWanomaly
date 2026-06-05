# agg_utils.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.clip(ez.sum(axis=1, keepdims=True), 1e-12, None)


def top2_margin_from_probs(probs: np.ndarray) -> np.ndarray:
    if probs.size == 0:
        return np.zeros((0,), dtype=np.float64)
    part = np.partition(probs, kth=probs.shape[1] - 2, axis=1)
    top2 = part[:, -2:]
    top1 = top2.max(axis=1)
    top2v = top2.min(axis=1)
    return (top1 - top2v).astype(np.float64)


def conf_score_from_probs(probs: np.ndarray, mode: str = "maxprob_margin") -> np.ndarray:
    """
    Confidence score untuk ranking window.
    mode:
      - maxprob
      - margin
      - maxprob_margin
      - maxprob_margin2  (maxprob * margin^2)  <-- lebih selektif
    """
    if probs.size == 0:
        return np.zeros((0,), dtype=np.float64)

    mx = probs.max(axis=1).astype(np.float64)
    if mode == "maxprob":
        return mx

    m = top2_margin_from_probs(probs)
    if mode == "margin":
        return m

    if mode == "maxprob_margin2":
        return (mx * (m * m)).astype(np.float64)

    return (mx * m).astype(np.float64)


@dataclass
class AggParams:
    keep_frac: float = 0.15
    min_keep: int = 8
    weight_power: float = 3.0
    conf_mode: str = "maxprob_margin2"
    agg_method: str = "mean_logit"  # mean_logit | geom_prob | mean_prob


def aggregate_vessel(
    adj_logits: np.ndarray,  # (N,C)
    probs: np.ndarray,       # (N,C)
    confs: np.ndarray,       # (N,)
    p: AggParams,
) -> Tuple[np.ndarray, float, int]:
    """
    Pilih top-K window berdasarkan confs,
    lalu agregasi dengan metode p.agg_method.
    """
    n = int(confs.size)
    if n <= 0:
        return np.zeros((probs.shape[1],), dtype=np.float64), 0.0, 0

    order = np.argsort(confs)[::-1]
    k = max(int(p.min_keep), int(round(n * float(p.keep_frac))))
    k = min(k, n)
    sel = order[:k]

    w = np.power(np.clip(confs[sel], 1e-6, 1.0), float(p.weight_power)).astype(np.float64)
    w = w / float(np.clip(w.sum(), 1e-12, None))

    if p.agg_method == "mean_prob":
        avg = (probs[sel].astype(np.float64) * w.reshape(-1, 1)).sum(axis=0)
        avg = np.clip(avg, 1e-12, None)
        avg = avg / float(avg.sum())
        return avg, float(avg.max()), int(k)

    if p.agg_method == "geom_prob":
        pp = np.clip(probs[sel].astype(np.float64), 1e-12, 1.0)
        avg_logp = (np.log(pp) * w.reshape(-1, 1)).sum(axis=0)
        a = avg_logp - float(np.max(avg_logp))
        expa = np.exp(a)
        avg = expa / float(np.clip(expa.sum(), 1e-12, None))
        return avg, float(avg.max()), int(k)

    # default: mean_logit (biasanya paling stabil)
    lg = adj_logits[sel].astype(np.float64)
    avg_logit = (lg * w.reshape(-1, 1)).sum(axis=0, keepdims=True)  # (1,C)
    avg = softmax_np(avg_logit)[0].astype(np.float64)
    return avg, float(avg.max()), int(k)


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        t = int(t)
        p = int(p)
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def metrics_from_cm(cm: np.ndarray) -> Dict[str, float]:
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    pred_count = cm.sum(axis=0)
    total = cm.sum()

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = tp / pred_count
        recall = tp / support
        f1 = 2 * precision * recall / (precision + recall)

    precision = np.nan_to_num(precision, nan=0.0)
    recall = np.nan_to_num(recall, nan=0.0)
    f1 = np.nan_to_num(f1, nan=0.0)

    acc = float(tp.sum() / max(total, 1.0))
    macro_f1 = float(f1.mean()) if len(f1) else 0.0
    balanced_acc = float(recall.mean()) if len(recall) else 0.0
    weighted_f1 = float((f1 * support).sum() / max(support.sum(), 1.0))
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "balanced_acc": balanced_acc,
        "weighted_f1": weighted_f1,
    }


def pick_best_tau_and_agg_by_vessel(
    logits_np: np.ndarray,     # (N,C)
    y_np: np.ndarray,          # (N,)
    g_np: np.ndarray,          # (N,) object/str
    log_pi: np.ndarray,        # (C,)
    tau_list: List[float],
    agg_grid: List[AggParams],
    num_classes: int,
) -> Tuple[float, AggParams, Dict[str, float]]:
    g_str = np.asarray(g_np).astype(str)
    uniq = np.unique(g_str)
    v2idx = {v: np.where(g_str == v)[0] for v in uniq}

    best_tau = None
    best_agg = None
    best_m = None

    for p in agg_grid:
        for tau in tau_list:
            adj = logits_np - (float(tau) * log_pi.reshape(1, -1))
            probs = softmax_np(adj).astype(np.float64)
            confs = conf_score_from_probs(probs, mode=p.conf_mode)

            y_true_v = []
            y_pred_v = []

            for v, idxs in v2idx.items():
                yt = int(np.bincount(y_np[idxs], minlength=num_classes).argmax())
                avg, _, _ = aggregate_vessel(adj[idxs], probs[idxs], confs[idxs], p)
                yp = int(np.argmax(avg))
                y_true_v.append(yt)
                y_pred_v.append(yp)

            cm = confusion_matrix_np(np.array(y_true_v), np.array(y_pred_v), num_classes)
            m = metrics_from_cm(cm)

            if best_tau is None:
                best_tau, best_agg, best_m = float(tau), p, m
            else:
                if m["macro_f1"] > best_m["macro_f1"] + 1e-9:
                    best_tau, best_agg, best_m = float(tau), p, m
                elif abs(m["macro_f1"] - best_m["macro_f1"]) <= 1e-9 and m["balanced_acc"] > best_m["balanced_acc"] + 1e-9:
                    best_tau, best_agg, best_m = float(tau), p, m

    return float(best_tau), best_agg, dict(best_m)