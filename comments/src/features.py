"""
Social media comments challenge — feature engineering.

Transformer models mostly need the cleaned text.
Classical models (TF-IDF + LR / CatBoost) can benefit from a few lightweight numeric/flag features.

We keep features:
- cheap to compute
- language-agnostic where possible
- explicitly named and stable (reproducible across runs)
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from config import CommentsConfig


NUM_FEATURES: tuple[str, ...] = (
    "len_chars",
    "len_tokens",
    "digit_count",
    "punct_count",
    "has_phone_token",
    "has_url_token",
    "has_email_token",
    "has_id_token",
)


KEYWORD_SPECS: tuple[tuple[str, str], ...] = (
    # Technologies
    ("has_fibre", r"\b(?:fibre|ftth|fttx|idoom_fibre)\b|(?:الفيبر|فيبر|فايبر)"),
    ("has_adsl", r"\b(?:adsl|vdsl)\b"),
    ("has_4g", r"\b(?:3g|4g|5g)\b"),
    ("has_modem", r"\b(?:modem)\b|(?:مودام|مودم)"),
    # Portal / app (Class 9)
    ("has_portal", r"\b(?:espace_client|espace\s*client|application|app|login|mot\s*de\s*passe)\b|(?:فضاء\s*الزبون|فضاء\s*الزبائن|الولوج|حساب|كلمة\s*السر)"),
    # Performance / QoS
    ("has_ping", r"\bping\b"),
    ("has_debit", r"\bdebit\b|(?:تدفق|سرعه|البينغ)"),
    ("has_mbps", r"\b(?:mbps|gbps)\b"),
    # Pricing / offers (Class 7)
    ("has_prices", r"\b(?:prix|tarif|cher|promo|offre|offres)\b|(?:سعر|الاسعار|الأسعار|اسعار|العروض)"),
    # Outage / complaint style
    ("has_outage", r"\b(?:panne|coupure|down|hs|marche\s+pas|ne\s+marche\s+pas)\b|(?:انقطاع|عطل|مقطوع|ماكاش|مكانش|walou)"),
    ("has_support_numbers", r"\b(?:100|12|1500)\b"),
)


def add_text_features(
    df: pd.DataFrame,
    cfg: CommentsConfig,
    *,
    text_col: str | None = None,
    keyword_specs: Iterable[tuple[str, str]] = KEYWORD_SPECS,
) -> pd.DataFrame:
    """
    Add numeric + boolean flags from cleaned text.
    """
    col = text_col or cfg.text_col
    if col not in df.columns:
        raise KeyError(f"Missing text column '{col}' in df.")

    out = df.copy()
    s = out[col].astype("string").fillna("")

    out["len_chars"] = s.str.len().astype("float32")
    out["len_tokens"] = s.str.count(r"\S+").astype("float32")
    out["digit_count"] = s.str.count(r"\d").astype("float32")

    # Count punctuation roughly (Latin + Arabic punctuation)
    out["punct_count"] = s.str.count(r"[!?.,،؛؟]").astype("float32")

    out["has_phone_token"] = s.str.contains(cfg.phone_token, regex=False, na=False).astype("float32")
    out["has_url_token"] = s.str.contains(cfg.url_token, regex=False, na=False).astype("float32")
    out["has_email_token"] = s.str.contains(cfg.email_token, regex=False, na=False).astype("float32")
    out["has_id_token"] = s.str.contains(cfg.id_token, regex=False, na=False).astype("float32")

    for feat_name, pattern in keyword_specs:
        out[feat_name] = s.str.contains(pattern, regex=True, case=False, na=False).astype("float32")

    # Convenience aggregate
    kw_cols = [name for name, _ in keyword_specs]
    if kw_cols:
        out["n_tech_flags"] = np.clip(out[kw_cols].sum(axis=1), 0, 100).astype("float32")
    else:
        out["n_tech_flags"] = 0.0

    return out


def engineer_features(df: pd.DataFrame, cfg: CommentsConfig) -> pd.DataFrame:
    """
    Apply all feature engineering on a cleaned canonical dataframe.
    """
    return add_text_features(df, cfg)


def build_feature_spec(cfg: CommentsConfig) -> dict:
    kw_cols = [name for name, _ in KEYWORD_SPECS]
    num_cols = list(NUM_FEATURES) + kw_cols + ["n_tech_flags"]
    return {
        "id_col": cfg.id_col,
        "platform_col": cfg.platform_col,
        "text_col": cfg.text_col,
        "target_col": cfg.target_col,
        "num_cols": num_cols,
        "keyword_flag_cols": kw_cols,
    }


