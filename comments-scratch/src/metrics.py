"""
Metrics utilities for the Social Media Comments challenge (scratch).

Kaggle evaluation metric: Macro F1-score.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def _as_1d_labels(y: Iterable) -> np.ndarray:
    """
    Convert an input vector to a 1D numpy array of labels.
    If a 2D array is provided (e.g., probabilities), argmax(axis=1) is used.
    """

    if isinstance(y, pd.Series):
        y = y.to_numpy()
    arr = np.asarray(y)
    if arr.ndim == 2:
        if arr.shape[1] == 1:
            arr = arr.reshape(-1)
        else:
            arr = arr.argmax(axis=1)
    return arr.reshape(-1)


def macro_f1_score(y_true: Iterable, y_pred: Iterable, labels: Sequence | None = None) -> float:
    yt = _as_1d_labels(y_true)
    yp = _as_1d_labels(y_pred)
    if len(yt) != len(yp):
        raise ValueError(f"y_true and y_pred length mismatch: {len(yt)} != {len(yp)}")
    if labels is None:
        return float(f1_score(yt, yp, average="macro", zero_division=0))
    lbls = list(labels)
    return float(f1_score(yt, yp, average="macro", labels=lbls, zero_division=0))


