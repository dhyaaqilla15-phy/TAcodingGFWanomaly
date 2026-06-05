from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
)


@dataclass
class MetricsResult:
    report_text: str
    confusion: np.ndarray
    f1_macro: float
    f1_micro: float
    precision_macro: float
    recall_macro: float
    balanced_acc: float


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    label_map: Optional[Dict[int, str]] = None,
) -> MetricsResult:
    target_names = [label_map[i] for i in sorted(label_map.keys())] if label_map else None

    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred)
    f1m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1u = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    bal = float(balanced_accuracy_score(y_true, y_pred))

    return MetricsResult(
        report_text=report,
        confusion=cm,
        f1_macro=f1m,
        f1_micro=f1u,
        precision_macro=prec,
        recall_macro=rec,
        balanced_acc=bal,
    )
