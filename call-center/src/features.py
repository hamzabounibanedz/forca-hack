"""
Call Center challenge — feature engineering & final column contract.

This module defines EXACTLY what the model sees, and guarantees that train/test
feature columns match perfectly.

Contract (high-level)
---------------------
- Keep `id` for alignment (NOT a feature)
- Keep `sn` for traceability (NOT a feature by default)
- Use:
  - categorical columns: `cfg.cat_cols`
  - text column: `cfg.text_col` (service_content)
  - engineered numeric columns:
      * time features from `cfg.datetime_col`
      * lightweight text helper features + telecom keyword flags
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from config import CallCenterConfig

# --- Engineered feature names (explicit) ---
TIME_FEATURES: tuple[str, ...] = (
    "handle_year",
    "handle_month",
    "handle_day",
    "handle_dayofyear",
    "handle_dayofweek",
    "handle_hour",
    "handle_minute",
    "handle_iso_week",
    # Single monotonic time feature (helps linear models capture ordering across months)
    "handle_time_ordinal",
    "handle_is_weekend",
    "handle_time_missing",
)
TEXT_NUM_FEATURES_BASE: tuple[str, ...] = (
    "text_len",
    "word_count",
    "digit_count",
    "has_num_token",
    "has_phone_token",
    # Speed/value patterns (help separate class 4)
    "mbps_value_count",
    "mbps_value_max",
    "mo_value_count",
    "mo_value_max",
)

# Task D: explicitly model subs_level missingness (UNK after preprocessing)
SUBS_LEVEL_FEATURES: tuple[str, ...] = ("subs_level_missing",)

# Telecom keyword flags (keep small & high-signal; these are common in the dataset)
KEYWORD_SPECS: tuple[tuple[str, str], ...] = (
    ("has_pon", r"\bpon\b"),
    ("has_los", r"\blos\b"),
    # --- Class 3 bottleneck helpers (fiber signal) ---
    # NOTE: these are evaluated on the *cleaned* text (lower + accents removed).
    ("has_los_rouge", r"\blos\s+rouge\b"),
    ("has_los_allume", r"\blos\s+allum\w*\b"),
    ("has_los_eteint", r"\blos\s+eteint\w*\b"),
    ("has_pon_stable", r"\bpon\s+stable\b"),
    ("has_pon_instable", r"\bpon\s+instable\b"),
    ("has_pon_clignote", r"\bpon\s+clignot\w*\b"),
    ("has_signal_bad", r"(?:\blos\s+(?:rouge|allum\w*)\b)|(?:\bpon\s+(?:clignot\w*|instable|rouge)\b)"),

    # --- Class 4 bottleneck helpers (throughput / speed issues) ---
    ("has_debit", r"\bdebit\b"),
    ("has_chute_debit", r"\bchute\s+de\s+debit\b"),
    ("has_mbps", r"\bmbps\b"),
    ("has_download", r"\bdownload\b"),
    ("has_upload", r"\bupload\b"),
    ("has_imei", r"\bimei\b"),
    ("has_gps", r"\bgps\b"),

    ("has_adsl", r"\badsl\b"),
    ("has_ftth", r"\bftth\b"),
    ("has_fttx", r"\bfttx\b"),
    ("has_4g", r"\b4g\b"),
    ("has_ont", r"\bont\b"),
    ("has_modem", r"\bmodem\b"),
)


def add_time_features(df: pd.DataFrame, *, datetime_col: str = "handle_time") -> pd.DataFrame:
    """
    Create time features from parsed datetime:
    - handle_year, handle_month, handle_day
    - handle_dayofyear, handle_dayofweek, handle_hour, handle_minute, handle_iso_week
    - handle_time_ordinal (days since Unix epoch, missing=-1)
    - handle_is_weekend, handle_time_missing

    Missing/invalid datetimes are encoded as -1 (float32) for stability in classical models.
    """
    if datetime_col not in df.columns:
        raise KeyError(f"Missing datetime column '{datetime_col}' in df.")

    out = df.copy()
    dt = out[datetime_col]
    if not pd.api.types.is_datetime64_any_dtype(dt):
        dt = pd.to_datetime(dt, errors="coerce")
    missing = dt.isna()

    year = dt.dt.year.astype("float32").fillna(-1.0)
    month = dt.dt.month.astype("float32").fillna(-1.0)
    day = dt.dt.day.astype("float32").fillna(-1.0)
    dayofyear = dt.dt.dayofyear.astype("float32").fillna(-1.0)
    dow = dt.dt.dayofweek.astype("float32").fillna(-1.0)
    hour = dt.dt.hour.astype("float32").fillna(-1.0)
    minute = dt.dt.minute.astype("float32").fillna(-1.0)

    # ISO week (UInt32 with NA) -> float32
    iso_week = dt.dt.isocalendar().week.astype("float32").fillna(-1.0)
    is_weekend = np.where(dow >= 0, pd.Series(dow).isin([5.0, 6.0]).to_numpy(), -1.0).astype("float32")
    time_missing = missing.astype("float32")

    # Single monotonic time feature (helps linear models capture ordering across months)
    # Encode as days since Unix epoch (1970-01-01). Missing -> -1.
    # Note: dt.astype('int64') yields nanoseconds; divide by (86400*1e9) -> days.
    dt_ns = dt.astype("int64")  # NaT -> min int64 sentinel
    # Replace NaT sentinel with NaN before scaling to avoid huge negative.
    dt_ns = dt_ns.where(~missing, other=np.nan)
    time_ordinal = (dt_ns / (86400.0 * 1e9)).astype("float32").fillna(-1.0)

    out["handle_year"] = year
    out["handle_month"] = month
    out["handle_day"] = day
    out["handle_dayofyear"] = dayofyear
    out["handle_dayofweek"] = dow
    out["handle_hour"] = hour
    out["handle_minute"] = minute
    out["handle_iso_week"] = iso_week
    out["handle_time_ordinal"] = time_ordinal
    out["handle_is_weekend"] = is_weekend
    out["handle_time_missing"] = time_missing
    return out


def add_text_helper_features(
    df: pd.DataFrame,
    *,
    text_col: str = "service_content",
    keyword_specs: Iterable[tuple[str, str]] = KEYWORD_SPECS,
) -> pd.DataFrame:
    """
    Lightweight numeric helper features from text:
    - text_len, word_count, digit_count
    - has_num_token, has_phone_token
    - telecom keyword flags (has_pon, has_los, ...)
    """
    if text_col not in df.columns:
        raise KeyError(f"Missing text column '{text_col}' in df.")

    out = df.copy()
    s = out[text_col].astype("string").fillna("")

    out["text_len"] = s.str.len().astype("float32")
    out["word_count"] = s.str.count(r"\S+").astype("float32")
    out["digit_count"] = s.str.count(r"\d").astype("float32")
    out["has_num_token"] = s.str.contains("<NUM>", regex=False, na=False).astype("float32")
    out["has_phone_token"] = s.str.contains("<PHONE>", regex=False, na=False).astype("float32")

    for feat_name, pattern in keyword_specs:
        out[feat_name] = s.str.contains(pattern, regex=True, case=False, na=False).astype("float32")

    # --- Numeric speed extractions (class 4 helpers) ---
    # These are cheap signals that complement TF-IDF / CatBoost text features.
    # We default missing to 0 (not -1) because "no speed value mentioned" is a meaningful state.
    mbps = s.str.extractall(r"\b(\d+(?:[.,]\d+)?)\s*mbps\b")
    if len(mbps.index):
        mbps_s = mbps[0].astype("string").str.replace(",", ".", regex=False)
        mbps_val = pd.to_numeric(mbps_s, errors="coerce")
        mbps_max = mbps_val.groupby(level=0).max()
        mbps_cnt = mbps_val.groupby(level=0).size()
    else:
        mbps_max = pd.Series(dtype="float64")
        mbps_cnt = pd.Series(dtype="int64")
    out["mbps_value_count"] = mbps_cnt.reindex(out.index, fill_value=0).astype("float32")
    out["mbps_value_max"] = mbps_max.reindex(out.index).fillna(0.0).astype("float32")

    mo = s.str.extractall(r"\b(\d{1,4})\s*mo\b")
    if len(mo.index):
        mo_val = pd.to_numeric(mo[0], errors="coerce")
        mo_max = mo_val.groupby(level=0).max()
        mo_cnt = mo_val.groupby(level=0).size()
    else:
        mo_max = pd.Series(dtype="float64")
        mo_cnt = pd.Series(dtype="int64")
    out["mo_value_count"] = mo_cnt.reindex(out.index, fill_value=0).astype("float32")
    out["mo_value_max"] = mo_max.reindex(out.index).fillna(0.0).astype("float32")

    return out


def engineer_features(df: pd.DataFrame, cfg: CallCenterConfig) -> pd.DataFrame:
    """
    Apply all feature engineering on a cleaned canonical dataframe.
    """
    out = add_time_features(df, datetime_col=cfg.datetime_col)
    out = add_text_helper_features(out, text_col=cfg.text_col)
    # Task D: add missing indicator for subs_level (don't guess; keep explicit signal)
    if "subs_level" in out.columns:
        out["subs_level_missing"] = (out["subs_level"].astype("string") == "UNK").astype("float32")
    else:
        out["subs_level_missing"] = np.nan
    return out


def build_feature_spec(
    cfg: CallCenterConfig,
    *,
    include_sn_as_feature: bool = False,
    include_handle_time_as_feature: bool = False,
) -> dict:
    """
    Define explicit feature lists (single source of truth).
    """
    cat_cols = list(cfg.cat_cols)
    text_col = cfg.text_col

    keyword_flag_cols = [name for name, _ in KEYWORD_SPECS]
    num_cols = list(TIME_FEATURES) + list(TEXT_NUM_FEATURES_BASE) + keyword_flag_cols + list(SUBS_LEVEL_FEATURES)

    feature_cols: list[str] = []
    feature_cols.extend(cat_cols)
    feature_cols.append(text_col)
    feature_cols.extend(num_cols)

    if include_handle_time_as_feature:
        # Not recommended: datetime should generally not be fed as raw to classical models.
        feature_cols.append(cfg.datetime_col)

    if include_sn_as_feature and "sn" not in feature_cols:
        feature_cols.append("sn")

    spec = {
        "id_col": cfg.id_col,
        "target_col": cfg.target_col,
        "datetime_col": cfg.datetime_col,
        "cat_cols": cat_cols,
        "text_col": text_col,
        "text_cols": [text_col],  # convenience for CatBoost scripts
        "num_cols": num_cols,
        "feature_cols": feature_cols,
        "keyword_specs": list(KEYWORD_SPECS),
        "notes": {
            "id_is_feature": False,
            "sn_is_feature": bool(include_sn_as_feature),
            "handle_time_is_feature": bool(include_handle_time_as_feature),
        },
    }
    return spec


def load_feature_spec(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"feature spec must be a JSON object, got: {type(obj)}")
    return obj


def normalize_feature_spec(spec: dict[str, Any], cfg: CallCenterConfig) -> dict[str, Any]:
    """
    Accept both:
    - New spec format (has 'feature_cols')
    - Legacy format (cat_cols + text_cols + num_cols)
    And return a normalized spec with:
      cat_cols, text_col, text_cols, num_cols, feature_cols, datetime_col
    """
    out: dict[str, Any] = dict(spec)

    out.setdefault("datetime_col", cfg.datetime_col)
    out.setdefault("cat_cols", list(cfg.cat_cols))

    # Text col can be either 'text_col' or 'text_cols' (legacy scripts).
    if "text_col" not in out:
        text_cols = out.get("text_cols")
        if isinstance(text_cols, list) and text_cols:
            out["text_col"] = str(text_cols[0])
        else:
            out["text_col"] = cfg.text_col
    out["text_cols"] = [out["text_col"]]

    out.setdefault("num_cols", [])

    if "feature_cols" not in out or not isinstance(out["feature_cols"], list) or not out["feature_cols"]:
        out["feature_cols"] = [*list(out["cat_cols"]), out["text_col"], *list(out["num_cols"])]

    return out


def prepare_X_from_spec(
    df: pd.DataFrame,
    cfg: CallCenterConfig,
    spec: dict[str, Any],
    *,
    strict: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Build a model-ready feature frame X from a cleaned dataframe using an explicit feature spec.
    Returns (X, normalized_spec).
    """
    spec2 = normalize_feature_spec(spec, cfg)
    feat = engineer_features(df.copy(), cfg)

    feature_cols = list(spec2["feature_cols"])
    if strict:
        missing = [c for c in feature_cols if c not in feat.columns]
        if missing:
            raise KeyError(f"Missing engineered feature columns: {missing}")
    else:
        feat = _ensure_columns(feat, feature_cols, fill_value=np.nan)

    # Stable fills for CatBoost
    for c in spec2["cat_cols"]:
        feat[c] = feat[c].astype("string").fillna("UNK")
    feat[spec2["text_col"]] = feat[spec2["text_col"]].astype("string").fillna("UNK")

    X = feat[feature_cols].copy()
    return X, spec2


def prepare_train_X_y(
    df_train: pd.DataFrame,
    cfg: CallCenterConfig,
    *,
    spec: dict[str, Any] | None = None,
    include_sn_as_feature: bool = False,
    include_handle_time_as_feature: bool = False,
    strict: bool = True,
    save_spec_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """
    Build (X_train, y_train, spec) for training, with an optional saved feature spec JSON.
    """
    spec0 = (
        spec
        if spec is not None
        else build_feature_spec(
            cfg,
            include_sn_as_feature=include_sn_as_feature,
            include_handle_time_as_feature=include_handle_time_as_feature,
        )
    )
    X, spec2 = prepare_X_from_spec(df_train, cfg, spec0, strict=strict)
    if cfg.target_col not in df_train.columns:
        raise KeyError(f"Train is missing target column '{cfg.target_col}'.")
    y = pd.to_numeric(df_train[cfg.target_col], errors="coerce")

    if save_spec_path is not None:
        p = Path(save_spec_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(spec2, ensure_ascii=False, indent=2), encoding="utf-8")

    return X, y, spec2


def _ensure_columns(df: pd.DataFrame, cols: Iterable[str], *, fill_value) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = fill_value
    return out


def prepare_train_test_matrices(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    cfg: CallCenterConfig,
    *,
    include_sn_as_feature: bool = False,
    include_handle_time_as_feature: bool = False,
    strict: bool = True,
    verbose: bool = True,
    save_spec_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict]:
    """
    Produce:
    - X_train, y_train, X_test
    - explicit feature spec (cat_cols, text_col, num_cols, feature_cols)

    Guarantees train/test feature columns match exactly (same names + same order).
    """
    spec = build_feature_spec(
        cfg,
        include_sn_as_feature=include_sn_as_feature,
        include_handle_time_as_feature=include_handle_time_as_feature,
    )

    train_feat = engineer_features(df_train.copy(), cfg)
    test_feat = engineer_features(df_test.copy(), cfg)

    # Ensure required columns exist (optional non-strict backfill).
    feature_cols = spec["feature_cols"]
    if strict:
        missing_train = [c for c in feature_cols if c not in train_feat.columns]
        missing_test = [c for c in feature_cols if c not in test_feat.columns]
        if missing_train:
            raise KeyError(f"Train is missing expected feature columns: {missing_train}")
        if missing_test:
            raise KeyError(f"Test is missing expected feature columns: {missing_test}")
    else:
        train_feat = _ensure_columns(train_feat, feature_cols, fill_value=np.nan)
        test_feat = _ensure_columns(test_feat, feature_cols, fill_value=np.nan)

    # Safety fills for model-friendly dtypes
    for c in spec["cat_cols"]:
        train_feat[c] = train_feat[c].astype("string").fillna("UNK")
        test_feat[c] = test_feat[c].astype("string").fillna("UNK")
    train_feat[spec["text_col"]] = train_feat[spec["text_col"]].astype("string").fillna("UNK")
    test_feat[spec["text_col"]] = test_feat[spec["text_col"]].astype("string").fillna("UNK")

    X_train = train_feat[feature_cols].copy()
    X_test = test_feat[feature_cols].copy()

    # Target
    if cfg.target_col not in train_feat.columns:
        raise KeyError(f"Train is missing target column '{cfg.target_col}'.")
    y_train = pd.to_numeric(train_feat[cfg.target_col], errors="coerce")

    # Final hard guarantee (DoD)
    if list(X_train.columns) != list(X_test.columns):
        raise AssertionError(
            "Train/test feature columns mismatch. "
            f"train={list(X_train.columns)}, test={list(X_test.columns)}"
        )

    if verbose:
        print("Feature contract:")
        print(f"- cat_cols ({len(spec['cat_cols'])}): {spec['cat_cols']}")
        print(f"- text_col: {spec['text_col']}")
        print(f"- num_cols ({len(spec['num_cols'])}): {spec['num_cols']}")
        print(f"- total feature_cols ({len(spec['feature_cols'])})")

    if save_spec_path is not None:
        p = Path(save_spec_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")

    return X_train, y_train, X_test, spec


