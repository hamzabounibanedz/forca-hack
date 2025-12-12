"""
Social media comments challenge — schema detection & validation.

Copied (and kept compatible) with `forca-hack/comments/src/schema.py` so scratch
experiments remain schema-robust across environments (encoding quirks, header variants).
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

from config import CommentsConfig


def normalize_col_name(name: str) -> str:
    """
    Normalize a column name to ASCII snake-ish form for robust matching:
      "Réseau Social" -> "reseau_social"
      "Commentaire client" -> "commentaire_client"
    """

    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def read_csv_robust(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """
    Read CSV with a small encoding fallback chain (helpful when teams use different OS defaults).
    """

    p = Path(path)
    last_err: Exception | None = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(p, encoding=enc, **kwargs)
        except UnicodeDecodeError as e:
            last_err = e
            continue
    # Last resort: keep pipeline moving
    try:
        return pd.read_csv(p, encoding="utf-8", encoding_errors="replace", **kwargs)  # type: ignore[arg-type]
    except TypeError:
        if last_err:
            raise last_err
        raise


def guess_raw_to_canon(df: pd.DataFrame, cfg: CommentsConfig, *, is_train: bool) -> dict[str, str]:
    cols = list(df.columns)
    norm_map = {c: normalize_col_name(c) for c in cols}

    def _pick_by_norm(candidates: set[str]) -> str | None:
        for c in cols:
            if norm_map[c] in candidates:
                return c
        return None

    id_raw = _pick_by_norm({"id", "idx", "index"})
    if id_raw is None:
        id_raw = next((c for c in cols if "id" in norm_map[c]), None)

    platform_raw = _pick_by_norm({"platform", "reseau_social", "reseaux_social", "reseau"})
    if platform_raw is None:
        platform_raw = next((c for c in cols if "social" in norm_map[c] or "platform" in norm_map[c]), None)

    text_raw = _pick_by_norm({"comment", "commentaire", "commentaire_client", "message", "texte", "content"})
    if text_raw is None:
        text_raw = next((c for c in cols if "comment" in norm_map[c] or "message" in norm_map[c]), None)

    target_raw: str | None = None
    if is_train:
        target_raw = _pick_by_norm({"class", "classe", "label", "target", "y"})
        if target_raw is None:
            target_raw = next((c for c in cols if norm_map[c] in {"class_int", "classid"}), None)

    missing = []
    if id_raw is None:
        missing.append("id")
    if platform_raw is None:
        missing.append("platform")
    if text_raw is None:
        missing.append("comment/text")
    if is_train and target_raw is None:
        missing.append("label/class")
    if missing:
        raise ValueError(
            "Unable to detect required columns. "
            f"Missing: {missing}. "
            f"Columns={cols} Normalized={norm_map}"
        )

    mapping = {
        str(id_raw): cfg.id_col,
        str(platform_raw): cfg.platform_col,
        str(text_raw): cfg.text_col,
    }
    if is_train and target_raw is not None:
        mapping[str(target_raw)] = cfg.target_col
    return mapping


def rename_raw_columns(df: pd.DataFrame, cfg: CommentsConfig, *, is_train: bool) -> pd.DataFrame:
    """
    Rename raw columns to canonical names (id/platform/comment/label).
    """

    mapping = guess_raw_to_canon(df, cfg, is_train=is_train)
    return df.copy().rename(columns=mapping)


def validate_file_exists(path: str | Path) -> None:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")


def build_label_mapping(y_raw: pd.Series) -> tuple[dict[int, int], dict[int, int]]:
    """
    Build stable label<->id mapping.
    - labels are the raw Kaggle labels (often 1..9)
    - ids are 0..(n_classes-1) used internally
    """

    y = pd.to_numeric(y_raw, errors="coerce").dropna().astype(int)
    labels_sorted = sorted(y.unique().tolist())
    label2id = {lbl: i for i, lbl in enumerate(labels_sorted)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return label2id, id2label


