"""
Metrics utilities for the Call Center challenge.

Kaggle evaluation metric: Macro F1-score.

This module provides:
- macro_f1_score: consistent Macro F1 computation
- per_class_f1_report: per-class precision/recall/f1/support as a DataFrame
- confusion_matrix_df: confusion matrix as a DataFrame (optionally normalized)
- evaluate: single entrypoint used by training scripts/notebooks
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


def _as_1d_labels(y: Iterable) -> np.ndarray:
    """
    Convert an input vector to a 1D numpy array of labels.

    - If a 2D array is provided (e.g., probabilities), argmax(axis=1) is used.
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


def _resolve_labels(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence | None) -> list:
    if labels is None:
        # Match sklearn's default behavior when labels=None: use the union of labels in y_true and y_pred.
        return sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    return list(labels)


def macro_f1_score(y_true: Iterable, y_pred: Iterable, labels: Sequence | None = None) -> float:
    """
    Compute Macro F1-score.

    Parameters
    - y_true: true class labels
    - y_pred: predicted labels OR class probabilities (n_samples, n_classes)
    - labels: optional explicit label ordering (e.g. [0,1,2,3,4,5])
    """
    yt = _as_1d_labels(y_true)
    yp = _as_1d_labels(y_pred)
    if len(yt) != len(yp):
        raise ValueError(f"y_true and y_pred length mismatch: {len(yt)} != {len(yp)}")
    if labels is None:
        return float(f1_score(yt, yp, average="macro", zero_division=0))
    lbls = list(labels)
    return float(f1_score(yt, yp, average="macro", labels=lbls, zero_division=0))


def per_class_f1_report(y_true: Iterable, y_pred: Iterable, labels: Sequence | None = None) -> pd.DataFrame:
    """
    Per-class precision/recall/f1/support report as a DataFrame.
    """
    yt = _as_1d_labels(y_true)
    yp = _as_1d_labels(y_pred)
    if len(yt) != len(yp):
        raise ValueError(f"y_true and y_pred length mismatch: {len(yt)} != {len(yp)}")
    lbls = _resolve_labels(yt, yp, labels)
    p, r, f1, s = precision_recall_fscore_support(yt, yp, labels=lbls, zero_division=0)
    df = pd.DataFrame(
        {
            "precision": p,
            "recall": r,
            "f1": f1,
            "support": s,
        },
        index=pd.Index(lbls, name="class"),
    )
    return df


def confusion_matrix_df(
    y_true: Iterable,
    y_pred: Iterable,
    labels: Sequence | None = None,
    *,
    normalize: str | None = None,
) -> pd.DataFrame:
    """
    Confusion matrix as a DataFrame.

    Parameters
    - normalize: None, "true", "pred", or "all" (same as sklearn)
    """
    yt = _as_1d_labels(y_true)
    yp = _as_1d_labels(y_pred)
    if len(yt) != len(yp):
        raise ValueError(f"y_true and y_pred length mismatch: {len(yt)} != {len(yp)}")
    lbls = _resolve_labels(yt, yp, labels)
    cm = confusion_matrix(yt, yp, labels=lbls, normalize=normalize)
    return pd.DataFrame(cm, index=pd.Index(lbls, name="true"), columns=pd.Index(lbls, name="pred"))


def evaluate(y_true: Iterable, y_pred: Iterable, labels: Sequence | None = None) -> float:
    """
    Single entrypoint for training/offline evaluation.

    Returns Macro F1-score (float).
    """
    return macro_f1_score(y_true=y_true, y_pred=y_pred, labels=labels)


