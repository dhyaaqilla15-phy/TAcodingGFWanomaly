from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import pandas as pd


@dataclass
class SuspectedCfg:
    # aturan dari output model
    conf_low: float = 0.55
    margin_low: float = 0.10
    majority_low: float = 0.70
    min_sequences: int = 5

    # khusus eval (kalau ada true_label)
    confident_misclass: float = 0.75

    # proxy kualitas dari fitur hasil olah model, fokus spoofing trajectory saja
    jump_ratio_high: float = 0.01
    max_implied_knots_high: float = 80.0
    max_step_km_high: float = 35.0
    speed_outlier_ratio_high: float = 0.01
    course_mismatch_ratio_high: float = 0.15

    flag_unknown_label: bool = True


def _looks_unknown(label: str) -> bool:
    t = (label or "").strip().lower()
    return ("unknown" in t) or (t == "unk") or (t == "other")


def _fmt_pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def build_suspected_df(
    vessel_details: Iterable[Dict[str, Any]],
    cfg: SuspectedCfg | None = None,
) -> pd.DataFrame:
    """
    Input minimal:
      - vessel_id (str)
      - pred_label (str)
      - confidence (float)
      - margin (float)
      - majority_ratio (float)
      - n_sequences (int)
      - true_label (optional)
      - pred_id / true_id (optional)

    Input proxy tambahan dari data yang SUDAH diolah model:
      - jump_ratio, max_implied_knots, max_step_km, speed_outlier_ratio
      - teleport_count, freeze_count, course_mismatch_ratio

    Output CSV columns:
      vessel_id, jenis, alasan, is_spoofing,
      teleport_count, freeze_count, course_mismatch_ratio

    Catatan:
      Fungsi suspected ini tetap fokus ke proxy spoofing trajectory.
      Untuk go-dark, gunakan make_godark dan preprocess --task godark.
    """
    cfg = cfg or SuspectedCfg()
    rows: List[Dict[str, Any]] = []

    for r in vessel_details:
        vid = str(r.get("vessel_id", ""))
        pred_label = str(r.get("pred_label", ""))

        conf = float(r.get("confidence", 0.0))
        margin = float(r.get("margin", 0.0))
        maj = float(r.get("majority_ratio", 1.0))
        nseq = int(r.get("n_sequences", 0))

        jump_ratio = float(r.get("jump_ratio", 0.0))
        max_implied = float(r.get("max_implied_knots", 0.0))
        max_step_km = float(r.get("max_step_km", 0.0))
        speed_outlier_ratio = float(r.get("speed_outlier_ratio", 0.0))

        teleport_count = int(r.get("teleport_count", 0))
        freeze_count = int(r.get("freeze_count", 0))
        course_mismatch_ratio = float(r.get("course_mismatch_ratio", 0.0))

        true_label = r.get("true_label", None)
        true_label = None if true_label is None else str(true_label)
        pred_id = r.get("pred_id", None)
        true_id = r.get("true_id", None)

        reasons: List[str] = []

        if conf < cfg.conf_low:
            reasons.append(f"Model kurang yakin (confidence {conf:.3f} < {cfg.conf_low})")

        if margin < cfg.margin_low:
            reasons.append(f"Dua tebakan teratas hampir seri (selisih top1-top2 {margin:.3f} < {cfg.margin_low})")

        if nseq >= cfg.min_sequences and maj < cfg.majority_low:
            reasons.append(f"Prediksi tidak konsisten antar window (mayoritas cuma {maj:.3f} < {cfg.majority_low})")

        if cfg.flag_unknown_label and _looks_unknown(pred_label):
            reasons.append("Label hasil prediksi terdeteksi 'unknown/other'")

        if (jump_ratio >= cfg.jump_ratio_high) or (max_implied >= cfg.max_implied_knots_high) or (max_step_km >= cfg.max_step_km_high):
            parts = []
            if jump_ratio > 0:
                parts.append(f"teleport {_fmt_pct(jump_ratio)} langkah")
            if max_implied > 0:
                parts.append(f"max implied {max_implied:.1f} kn")
            if max_step_km > 0:
                parts.append(f"max langkah {max_step_km:.1f} km")
            reasons.append("Data lompat/teleport ({}).".format(", ".join(parts)))

        if speed_outlier_ratio >= cfg.speed_outlier_ratio_high:
            reasons.append(f"Speed outlier cukup sering ({_fmt_pct(speed_outlier_ratio)})")

        if true_label is not None and (pred_id is not None) and (true_id is not None):
            try:
                if int(pred_id) != int(true_id) and conf >= cfg.confident_misclass:
                    reasons.append(
                        f"Model yakin tapi beda dengan label asli (true={true_label}, pred={pred_label}, conf={conf:.3f} >= {cfg.confident_misclass})."
                    )
            except Exception:
                pass

        is_spoofing = int(
            (teleport_count > 0)
            or (freeze_count > 0)
            or (course_mismatch_ratio >= cfg.course_mismatch_ratio_high)
        )

        if is_spoofing:
            reasons.append("Spoofing terdeteksi (teleport/freeze/course mismatch)")

        if reasons:
            rows.append(
                {
                    "vessel_id": vid,
                    "jenis": pred_label,
                    "alasan": "; ".join(reasons),
                    "is_spoofing": is_spoofing,
                    "teleport_count": teleport_count,
                    "freeze_count": freeze_count,
                    "course_mismatch_ratio": course_mismatch_ratio,
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "vessel_id",
            "jenis",
            "alasan",
            "is_spoofing",
            "teleport_count",
            "freeze_count",
            "course_mismatch_ratio",
        ],
    )
