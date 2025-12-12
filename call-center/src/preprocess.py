"""
Call Center challenge — preprocessing.

Goals
-----
- Clean data without destroying telecom signal (keep tokens like: pon, los, adsl, fttx, ftth, 4g, ...)
- Mask PII in free text (phone numbers) and long numeric identifiers
- Normalize messy categorical strings (trim, collapse spaces, fill missing with "UNK")
- Parse datetime safely for handle_time

This file is intentionally self-contained because it's executed from Colab
via `make_clean_dataset.py`.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

import pandas as pd

from config import CallCenterConfig

UNK = "UNK"

# Candidate numeric chunk (digits with optional separators) used to detect phone numbers.
_PHONE_CANDIDATE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s\-.]{6,}\d)(?!\d)")
_LONG_DIGITS_RE = re.compile(r"\d{6,}")
_SEPARATORS_RE = re.compile(r"[\/_|]+")  # //, /, _, | -> space
_COLON_BETWEEN_WORDS_RE = re.compile(r"(?<=\w):(?=\w)")  # e.g., "nc:0655" -> "nc 0655"
_BRACKET_SUFFIX_RE = re.compile(r"\s*\[[^\]]+\]\s*$")  # e.g., "oss sdm [used by system]" -> "oss sdm"
_ACTEL_CODE_RE = re.compile(r"\(([^()]*)\)\s*$")  # trailing (...) code at end of ACTEL label


def _remove_accents(text: str) -> str:
    # NFKD splits accented characters into base + combining mark.
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


def _collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def rename_raw_columns(df: pd.DataFrame, cfg: CallCenterConfig) -> pd.DataFrame:
    """
    Rename raw CSV columns to canonical names using `cfg.raw_to_canon`.
    """
    if cfg.raw_to_canon is None:
        raise ValueError("cfg.raw_to_canon is None. Use config.default_config() or provide mapping.")
    return df.copy().rename(columns=cfg.raw_to_canon)


def _canonicalize_cat_value(value: str, *, cfg: CallCenterConfig, col_name: str) -> str:
    """
    Normalize a single categorical value conservatively.
    """
    v = _collapse_spaces(value)
    if not v:
        return UNK

    if cfg.remove_accents:
        v = _remove_accents(v)

    # Lowercasing reduces duplicate categories; keep UNK in uppercase.
    if cfg.lower:
        v = v.lower()

    # Task C: reduce dept_traitement variants by removing generic bracket suffixes.
    # (Only strips a bracketed suffix at the end of the string.)
    if col_name == "dept_traitement":
        v = _BRACKET_SUFFIX_RE.sub("", v).strip()
        if not v:
            return UNK

    # Obvious duplicates (requested DoD example).
    # We keep this very small & conservative on purpose.
    if col_name == "customer_level":
        # Task C: unify résidentiel/residentiel/residential (+ common typos starting with "resid")
        # Examples observed: residantiel, residenciel, residenriel, residental, ... -> residential
        key = v.lower()
        if key.startswith("resid"):
            return "residential" if cfg.lower else "Residential"

    return v


def clean_categoricals(
    df: pd.DataFrame, cfg: CallCenterConfig, *, cat_cols: Iterable[str] | None = None
) -> pd.DataFrame:
    """
    Clean categorical columns:
    - convert to string
    - strip + collapse spaces
    - fill missing with "UNK"
    - normalize a small set of obvious duplicates
    """
    df = df.copy()
    cols = list(cat_cols) if cat_cols is not None else list(cfg.cat_cols)

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing categorical columns in df: {missing}")

    for c in cols:
        s = df[c].astype("string")
        # Preserve missingness (avoid 'nan' string).
        s = s.fillna(pd.NA)
        s = s.str.strip()
        s = s.str.replace(r"\s+", " ", regex=True)
        s = s.replace({"": pd.NA})
        s = s.fillna(UNK)

        # Apply conservative normalization value-by-value.
        s = s.map(lambda x, _c=c: x if x == UNK else _canonicalize_cat_value(str(x), cfg=cfg, col_name=_c))

        df[c] = s
    return df


def parse_handle_time(df: pd.DataFrame, cfg: CallCenterConfig, *, datetime_col: str | None = None) -> pd.DataFrame:
    """
    Parse `handle_time` into a standardized pandas datetime dtype.
    - safe parsing with coercion on failures
    - failures become NaT
    """
    df = df.copy()
    col = datetime_col or cfg.datetime_col
    if col not in df.columns:
        raise KeyError(f"Missing datetime column '{col}' in df.")

    s = df[col]
    if pd.api.types.is_datetime64_any_dtype(s):
        df[col] = s
        return df

    # First pass: robust parsing (ISO timestamps seen in the dataset).
    dt = pd.to_datetime(s, errors="coerce")
    # Fallback pass: try day-first in case some rows are in dd/mm/yyyy format.
    if dt.isna().any():
        dt2 = pd.to_datetime(s, errors="coerce", dayfirst=True)
        dt = dt.fillna(dt2)

    df[col] = dt
    return df


def _looks_like_phone(digits_only: str) -> bool:
    """
    Heuristic phone detector:
    - local: 9 or 10 digits starting with 0 (covers Algerian fixed and mobile formats)
    - local mobile without leading 0: 9 digits starting with 5/6/7
    - country code: 213 / 00213 / +213 with 8-10 digits after
    """
    d = digits_only
    if not d:
        return False

    # Normalize common country-code prefixes.
    if d.startswith("00213"):
        d = d[2:]  # -> 213...
    if d.startswith("213"):
        rest = d[3:]
        # Country code form usually drops the leading 0 -> 8-10 digits remain.
        if 8 <= len(rest) <= 10:
            return True
        return False

    # Default local formats
    if len(d) in {9, 10} and d.startswith("0"):
        return True
    # Some people omit the leading 0 (9 digits). Keep it conservative: only mobile prefixes.
    if len(d) == 9 and d[0] in {"5", "6", "7"}:
        return True

    return False


def _mask_phones_in_text(text: str) -> str:
    """
    Replace phone-number-like chunks with <PHONE> (keeps the rest unchanged).
    """

    def _repl(m: re.Match) -> str:
        chunk = m.group(0)
        digits = re.sub(r"\D", "", chunk)
        return "<PHONE>" if _looks_like_phone(digits) else chunk

    return _PHONE_CANDIDATE_RE.sub(_repl, text)

def _has_unmasked_phone(text: str) -> bool:
    """
    Returns True if a text still contains a phone-like number chunk (based on our heuristic).
    Useful as a post-cleaning sanity check.
    """
    for m in _PHONE_CANDIDATE_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if _looks_like_phone(digits):
            return True
    return False


def assert_no_unmasked_phones(
    df: pd.DataFrame,
    cfg: CallCenterConfig,
    *,
    text_col: str | None = None,
    max_examples: int = 5,
) -> None:
    """
    DoD helper: ensure no raw phone-like numbers remain after cleaning.
    Raises ValueError with examples if any are found.
    """
    col = text_col or cfg.text_col
    if col not in df.columns:
        raise KeyError(f"Missing text column '{col}' in df.")
    s = df[col].astype("string").fillna("")
    candidate = s.str.contains(_PHONE_CANDIDATE_RE, regex=True, na=False)
    if not candidate.any():
        return

    cand_s = s.loc[candidate]
    bad_mask = cand_s.map(lambda x: _has_unmasked_phone(str(x)))
    if bad_mask.any():
        ex_rows = df.loc[cand_s.index[bad_mask], [c for c in [cfg.id_col, col] if c in df.columns]].head(
            int(max_examples)
        )
        raise ValueError(
            "Found unmasked phone-like numbers after cleaning. "
            f"examples=\n{ex_rows.to_string(index=False)}"
        )


def clean_service_content(df: pd.DataFrame, cfg: CallCenterConfig, *, text_col: str | None = None) -> pd.DataFrame:
    """
    Clean free text (`service_content`):
    - normalize whitespace
    - mask phone numbers -> <PHONE>
    - mask long digit sequences (>=6 digits) -> <NUM>
    - optionally remove accents if cfg.remove_accents=True
    - do NOT remove telecom tokens (pon/los/adsl/fttx/...) — we keep all words; we only normalize.

    Quick before/after examples (DoD)
    -------------------------------
    - "pon stable ... 0559233222"                -> "pon stable ... <PHONE>"
    - "nc 0661224207 // ns2507310000009820"      -> "nc <PHONE> ns <NUM>"
    - "NC:0655462945 / refuse la pause"          -> "nc <PHONE> refuse la pause"
    """
    df = df.copy()
    col = text_col or cfg.text_col
    if col not in df.columns:
        raise KeyError(f"Missing text column '{col}' in df.")

    s = df[col].astype("string").fillna("")
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    s = s.replace({"": UNK})

    # Normalize case/accents BEFORE masking to reduce vocabulary, but keep our inserted tokens uppercase.
    if cfg.lower:
        s = s.str.lower()
    if cfg.remove_accents:
        s = s.map(lambda x: x if x == UNK else _remove_accents(str(x)))

    # Normalize separators that hurt tokenization.
    # NOTE: do this BEFORE phone masking to improve detection in formats like "NC:0655/462/945".
    s = s.str.replace(_SEPARATORS_RE, " ", regex=True)
    s = s.str.replace(_COLON_BETWEEN_WORDS_RE, " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()

    # Mask phone numbers then long numeric identifiers.
    s = s.map(lambda x: x if x == UNK else _mask_phones_in_text(str(x)))
    s = s.str.replace(_LONG_DIGITS_RE, "<NUM>", regex=True)

    # Make sure tokens are separable by simple whitespace tokenizers:
    # "ns<NUM>" -> "ns <NUM>", "nc:<PHONE>" -> "nc <PHONE>"
    s = s.str.replace(r"(?<!\s)(<PHONE>|<NUM>)(?!\s)", r" \1 ", regex=True)

    # Final whitespace normalization.
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    s = s.replace({"": UNK})

    df[col] = s
    # Hard gate: refuse to proceed if any phone-like pattern remains.
    assert_no_unmasked_phones(df, cfg, text_col=col, max_examples=10)
    return df


def add_actel_code_feature(
    df: pd.DataFrame,
    cfg: CallCenterConfig,
    *,
    actel_col: str = "actel",
    out_col: str = "actel_code",
) -> pd.DataFrame:
    """
    Extract ACTEL code inside trailing parentheses, e.g.:
      "ACTEL Boumerdes (22)" -> "22"
      "ACTEL Imama (J9)"     -> "J9"
      "Yaghmouracen...(03)"  -> "03"

    If no code is present, leaves missing; downstream categorical cleaning will fill UNK.
    """
    if actel_col not in df.columns:
        raise KeyError(f"Missing ACTEL column '{actel_col}' in df.")

    out = df.copy()
    s = out[actel_col].astype("string").fillna("")
    code = s.str.extract(_ACTEL_CODE_RE, expand=False).astype("string")
    code = code.str.strip().replace({"": pd.NA})
    out[out_col] = code
    return out


def preprocess_callcenter_df(df: pd.DataFrame, cfg: CallCenterConfig) -> pd.DataFrame:
    """
    Full preprocessing pipeline (raw -> cleaned canonical df).
    """
    out = rename_raw_columns(df, cfg)
    # Derived high-cardinality helper (created early so it participates in categorical cleaning).
    out = add_actel_code_feature(out, cfg, actel_col="actel", out_col="actel_code")
    out = parse_handle_time(out, cfg)
    out = clean_categoricals(out, cfg)
    out = clean_service_content(out, cfg)

    # Keep contract columns (+ target if present)
    keep = list(cfg.keep_cols)
    if cfg.target_col in out.columns:
        keep.append(cfg.target_col)

    missing = [c for c in keep if c not in out.columns]
    if missing:
        raise KeyError(f"After preprocessing, missing expected canonical columns: {missing}")

    out = out[keep].copy()

    if not cfg.keep_sn_in_outputs and "sn" in out.columns:
        out = out.drop(columns=["sn"])

    return out


