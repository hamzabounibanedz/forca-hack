"""
Sanity checks for the comments preprocessing pipeline.

Goal: catch data issues early (schema mismatches, broken cleaning, PII leakage).

Usage:
python forca-hack/comments/src/validate_preprocess.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd

import preprocess
import schema
from config import CommentsConfig, guess_local_data_dir, guess_team_dir, resolve_comments_data_dir


_URL_RE = re.compile(r"(?:https?://\S+|www\.\S+)", flags=re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", flags=re.IGNORECASE)
_LONG_DIGITS_RE = re.compile(r"\d{6,}")


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        raise AssertionError(msg)


def _check_df(df: pd.DataFrame, cfg: CommentsConfig, *, is_train: bool) -> dict[str, Any]:
    out: dict[str, Any] = {}
    _assert(cfg.id_col in df.columns, f"Missing id_col='{cfg.id_col}'")
    _assert(cfg.platform_col in df.columns, f"Missing platform_col='{cfg.platform_col}'")
    _assert(cfg.text_col in df.columns, f"Missing text_col='{cfg.text_col}'")
    if is_train:
        _assert(cfg.target_col in df.columns, f"Missing target_col='{cfg.target_col}'")

    out["rows"] = int(len(df))
    out["id_missing"] = int(df[cfg.id_col].isna().sum())
    out["id_unique"] = int(df[cfg.id_col].nunique(dropna=False))
    out["id_dupes"] = int(out["rows"] - out["id_unique"])

    # No empty platform
    plat = df[cfg.platform_col].astype("string").fillna("")
    out["platform_empty"] = int((plat.str.strip() == "").sum())

    # No null comment (can be empty string, but not NaN)
    txt = df[cfg.text_col].astype("string")
    out["text_missing"] = int(txt.isna().sum())
    out["text_empty"] = int((txt.fillna("").str.strip() == "").sum())

    # PII leaks (best-effort)
    s = txt.fillna("")
    out["url_leaks"] = int(s.str.contains(_URL_RE, regex=True, na=False).sum())
    out["email_leaks"] = int(s.str.contains(_EMAIL_RE, regex=True, na=False).sum())
    out["long_digit_leaks_ge6"] = int(s.str.contains(_LONG_DIGITS_RE, regex=True, na=False).sum())

    # Phone leak check (heuristic)
    try:
        preprocess.assert_no_unmasked_phones(df, cfg)
        out["phone_leaks"] = 0
    except Exception:
        out["phone_leaks"] = -1  # indicates failure; see exception in caller

    if is_train:
        y = df[cfg.target_col].astype("Int64")
        out["label_missing"] = int(y.isna().sum())
        out["labels"] = sorted([int(x) for x in y.dropna().unique().tolist()])
        out["n_classes"] = int(y.dropna().nunique()) if len(y.dropna()) else 0

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--train_filename", type=str, default="train.csv")
    parser.add_argument("--test_filename", type=str, default="test_file.csv")
    args = parser.parse_args()

    cfg = CommentsConfig()

    if args.data_dir:
        data_dir = Path(args.data_dir)
    elif args.team_dir:
        data_dir = resolve_comments_data_dir(Path(args.team_dir))
    else:
        team = guess_team_dir()
        data_dir = resolve_comments_data_dir(team) if team else guess_local_data_dir()

    train_path = data_dir / args.train_filename
    test_path = data_dir / args.test_filename
    schema.validate_file_exists(train_path)
    schema.validate_file_exists(test_path)

    df_train_raw = schema.read_csv_robust(train_path)
    df_test_raw = schema.read_csv_robust(test_path)

    # Schema checks on raw
    schema.run_all_checks(df_train=df_train_raw, df_test=df_test_raw, cfg=cfg)

    # Clean
    tr = preprocess.preprocess_comments_df(df_train_raw, cfg, is_train=True)
    te = preprocess.preprocess_comments_df(df_test_raw, cfg, is_train=False)

    print("## Cleaned train checks")
    print(_check_df(tr, cfg, is_train=True))
    print("\n## Cleaned test checks")
    print(_check_df(te, cfg, is_train=False))

    # Hard assertions
    _assert(tr[cfg.id_col].isna().sum() == 0, "Train has missing ids after cleaning")
    _assert(te[cfg.id_col].isna().sum() == 0, "Test has missing ids after cleaning")
    _assert(tr[cfg.id_col].nunique() == len(tr), "Train has duplicate ids")
    _assert(te[cfg.id_col].nunique() == len(te), "Test has duplicate ids")

    # If present, enforce no raw URL/email left
    _assert(int(tr[cfg.text_col].astype('string').fillna('').str.contains(_URL_RE, regex=True).sum()) == 0, "URL leaks in train cleaned text")
    _assert(int(te[cfg.text_col].astype('string').fillna('').str.contains(_URL_RE, regex=True).sum()) == 0, "URL leaks in test cleaned text")
    _assert(int(tr[cfg.text_col].astype('string').fillna('').str.contains(_EMAIL_RE, regex=True).sum()) == 0, "Email leaks in train cleaned text")
    _assert(int(te[cfg.text_col].astype('string').fillna('').str.contains(_EMAIL_RE, regex=True).sum()) == 0, "Email leaks in test cleaned text")

    # Phone leak check raises with examples if any
    preprocess.assert_no_unmasked_phones(tr, cfg)
    preprocess.assert_no_unmasked_phones(te, cfg)

    print("\nAll preprocessing sanity checks passed.")


if __name__ == "__main__":
    main()


