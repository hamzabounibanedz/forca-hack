"""
Social media comments challenge — preprocessing (scratch).

This is intentionally compatible with `forca-hack/comments/src/preprocess.py` so
the scratch experts operate on the same cleaned signal as the transformer pipeline.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd

import schema
from config import CommentsConfig

UNK = "UNK"

# ---- PII / noise regex ----
_URL_RE = re.compile(r"(?:https?://\S+|www\.\S+)", flags=re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", flags=re.IGNORECASE)

# Candidate numeric chunk (digits with optional separators) used to detect phone numbers.
_PHONE_CANDIDATE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s\-.]{6,}\d)(?!\d)")

# Normalization helpers
_SPACES_RE = re.compile(r"\s+")
_SEPARATORS_RE = re.compile(r"[\/_|]+")  # //, /, _, | -> space
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e]")

# Punctuation squashing (Latin + Arabic)
_REPEAT_PUNCT_RE = re.compile(r"([!?.,،؛؟])\1{2,}")

# Latin elongation: keep at most 2 repeats (e.g., "sloooow" -> "sloow")
_LATIN_ELONG_RE = re.compile(r"([a-z])\1{2,}")

# Script boundary helpers (Arabic <-> Latin). Important for code-switching like "لfibre".
_AR_LETTERS = r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]"
_LAT_LETTERS = r"[A-Za-z]"
_AR_LAT_BOUNDARY_RE = re.compile(rf"(?<={_AR_LETTERS})(?={_LAT_LETTERS})|(?<={_LAT_LETTERS})(?={_AR_LETTERS})")

# Arabic diacritics + tatweel
_ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_ARABIC_TATWEEL_RE = re.compile(r"\u0640")


# ---- Telecom canonicalization (small + high-signal) ----
_TELECOM_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Internet technologies
    (re.compile(r"\b(a\s*d\s*s\s*l|adsl)\b", flags=re.IGNORECASE), "adsl"),
    (re.compile(r"\b(v\s*d\s*s\s*l|vdsl)\b", flags=re.IGNORECASE), "vdsl"),
    (re.compile(r"\b(ftth)\b", flags=re.IGNORECASE), "ftth"),
    (re.compile(r"\b(fttx)\b", flags=re.IGNORECASE), "fttx"),
    (re.compile(r"\b(4\s*g|4g)\b", flags=re.IGNORECASE), "4g"),
    (re.compile(r"\b(3\s*g|3g)\b", flags=re.IGNORECASE), "3g"),
    (re.compile(r"\b(5\s*g|5g)\b", flags=re.IGNORECASE), "5g"),
    # Brand/product
    (re.compile(r"\b(idoom\s*fibre|idoomfibre)\b", flags=re.IGNORECASE), "idoom_fibre"),
    (re.compile(r"\b(idoom)\b", flags=re.IGNORECASE), "idoom"),
    # Customer portal (Class 9 signal)
    (re.compile(r"\b(espace\s*client|espaceclient)\b", flags=re.IGNORECASE), "espace_client"),
    (re.compile(r"(?:فضاء\s*الزبون|فضاء\s*الزبائن)", flags=re.IGNORECASE), "espace_client"),
    # Fibre variants (French/English + common Arabic loanword)
    (re.compile(r"\b(fibre\s*optique|fiber\s*optic|fiber|fibre)\b", flags=re.IGNORECASE), "fibre"),
    (re.compile(r"(?:الفيبر|فيبر|فايبر)", flags=re.IGNORECASE), "fibre"),
    # Hardware
    (re.compile(r"\b(modem|modam|modem[e]?)\b", flags=re.IGNORECASE), "modem"),
    (re.compile(r"(?:مودام|مودم)", flags=re.IGNORECASE), "modem"),
    (re.compile(r"\b(wi\s*fi\s*6|wifi\s*6|wifi6)\b", flags=re.IGNORECASE), "wifi6"),
    # Performance
    (re.compile(r"\b(ping)\b", flags=re.IGNORECASE), "ping"),
    (re.compile(r"\b(debit|débit)\b", flags=re.IGNORECASE), "debit"),
    (re.compile(r"\b(mbps|mb/s|mega|megabit)\b", flags=re.IGNORECASE), "mbps"),
    (re.compile(r"\b(gbps|gb/s|giga|gigabit)\b", flags=re.IGNORECASE), "gbps"),
)


def _remove_accents_latin(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


def _collapse_spaces(text: str) -> str:
    return _SPACES_RE.sub(" ", text).strip()


def _normalize_arabic(text: str) -> str:
    """
    Conservative Arabic normalization:
    - remove tatweel and diacritics
    - normalize common letter variants (أإآ -> ا, ى -> ي)
    """

    t = text
    t = _ARABIC_TATWEEL_RE.sub("", t)
    t = _ARABIC_DIACRITICS_RE.sub("", t)
    t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    t = t.replace("ى", "ي")
    return t


def _looks_like_phone(digits_only: str) -> bool:
    """
    Heuristic phone detector (Algeria-friendly):
    - local: 9 or 10 digits starting with 0
    - local mobile (no leading 0): 9 digits starting with 5/6/7
    - country code: 213 / 00213 / +213 with 8-10 digits after
    """

    d = digits_only
    if not d:
        return False

    if d.startswith("00213"):
        d = d[2:]  # -> 213...
    if d.startswith("213"):
        rest = d[3:]
        return 8 <= len(rest) <= 10

    if len(d) in {9, 10} and d.startswith("0"):
        return True
    if len(d) == 9 and d[0] in {"5", "6", "7"}:
        return True
    return False


def _mask_phones(text: str, *, token: str) -> str:
    def _repl(m: re.Match) -> str:
        chunk = m.group(0)
        digits = re.sub(r"\D", "", chunk)
        return token if _looks_like_phone(digits) else chunk

    return _PHONE_CANDIDATE_RE.sub(_repl, text)


def mask_pii(text: str, cfg: CommentsConfig) -> str:
    t = text
    t = _URL_RE.sub(cfg.url_token, t)
    t = _EMAIL_RE.sub(cfg.email_token, t)
    t = _mask_phones(t, token=cfg.phone_token)
    if cfg.mask_long_ids:
        t = re.sub(rf"\d{{{cfg.long_id_min_digits},}}", cfg.id_token, t)
    return t


def canonicalize_telecom_tokens(text: str) -> str:
    t = text
    for pat, repl in _TELECOM_PATTERNS:
        t = pat.sub(repl, t)
    return t


def reduce_elongations(text: str) -> str:
    return _LATIN_ELONG_RE.sub(r"\1\1", text)


def clean_platform(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return UNK
    v = str(value).strip()
    if not v:
        return UNK

    v = _collapse_spaces(v)
    v = _remove_accents_latin(v)
    v = v.lower()

    v2 = v.replace(" ", "")
    if v2 == "facebookfacebook":
        v = "facebook"
    elif v2 == "twittertwitter":
        v = "twitter"
    elif v2 == "instagraminstagram":
        v = "instagram"

    v = re.sub(r"[^a-z0-9]+", "_", v).strip("_")
    return v or UNK


def clean_comment(text: Any, cfg: CommentsConfig) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    t = str(text)

    # Unicode normalization: keeps Arabic + Latin stable, fixes some compatibility chars.
    t = unicodedata.normalize("NFKC", t)
    t = t.replace("\r", " ").replace("\n", " ")
    t = _ZERO_WIDTH_RE.sub("", t)

    # Normalize separators that often stick words together
    t = _SEPARATORS_RE.sub(" ", t)

    if cfg.mask_pii:
        t = mask_pii(t, cfg)

    if cfg.normalize_arabic:
        t = _normalize_arabic(t)

    if cfg.lower:
        t = t.lower()

    if cfg.remove_accents_latin:
        t = _remove_accents_latin(t)

    if cfg.canonicalize_telecom:
        t = canonicalize_telecom_tokens(t)

    # Ensure placeholders are surrounded by spaces (helps tokenizers and TF-IDF)
    if cfg.mask_pii:
        for tok in (cfg.phone_token, cfg.email_token, cfg.url_token, cfg.id_token):
            t = t.replace(tok, f" {tok} ")

    # Insert spaces between Arabic and Latin script boundaries (e.g., "لfibre" -> "ل fibre")
    t = _AR_LAT_BOUNDARY_RE.sub(" ", t)

    # Squash excessive punctuation and elongations
    t = _REPEAT_PUNCT_RE.sub(r"\1", t)
    if cfg.reduce_elongations:
        t = reduce_elongations(t)

    if cfg.collapse_whitespace:
        t = _collapse_spaces(t)

    return t


def preprocess_comments_df(df_raw: pd.DataFrame, cfg: CommentsConfig, *, is_train: bool) -> pd.DataFrame:
    """
    Raw -> canonical columns -> cleaned text/platform.
    """

    df = schema.rename_raw_columns(df_raw, cfg, is_train=is_train)
    out = df.copy()

    if cfg.keep_raw_cols:
        out[f"{cfg.platform_col}_raw"] = out[cfg.platform_col]
        out[f"{cfg.text_col}_raw"] = out[cfg.text_col]

    out[cfg.platform_col] = out[cfg.platform_col].map(clean_platform)
    out[cfg.text_col] = out[cfg.text_col].map(lambda x: clean_comment(x, cfg))

    out[cfg.id_col] = pd.to_numeric(out[cfg.id_col], errors="coerce").astype("Int64")
    if is_train and cfg.target_col in out.columns:
        out[cfg.target_col] = pd.to_numeric(out[cfg.target_col], errors="coerce").astype("Int64")

    return out


