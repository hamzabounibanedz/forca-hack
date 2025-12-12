"""
Schema and sanity checks for the Call Center challenge.

These checks should run BEFORE preprocessing/training to prevent silent bugs:
- wrong folder / wrong file
- column mismatch between train/test
- submission format mismatch
- duplicated ids
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd

from config import CallCenterConfig


class SchemaError(ValueError):
    pass


def _missing_cols(df: pd.DataFrame, required: list[str]) -> list[str]:
    cols = set(df.columns.tolist())
    return [c for c in required if c not in cols]


def validate_file_exists(path: Path) -> None:
    if not path.exists():
        raise SchemaError(f"Missing file: {path}")


def validate_submission_format(df_sub: pd.DataFrame, cfg: CallCenterConfig) -> None:
    # Kaggle contract for this competition (do not "configure" this away).
    expected = ["id", "class_int"]
    got = df_sub.columns.tolist()
    if got != expected:
        raise SchemaError(f"sample_submission columns mismatch. expected={expected}, got={got}")

    # Also ensure our config matches the Kaggle contract (prevents subtle downstream bugs).
    cfg_expected = [cfg.id_col_raw, cfg.target_col_raw]
    if cfg_expected != expected:
        raise SchemaError(
            "Config id/target column names do not match Kaggle contract. "
            f"expected={expected}, cfg={cfg_expected}"
        )


def validate_raw_columns(df_train: pd.DataFrame, df_test: pd.DataFrame, cfg: CallCenterConfig) -> None:
    # Train must have target
    if cfg.target_col_raw not in df_train.columns:
        raise SchemaError(f"Train is missing target column '{cfg.target_col_raw}'.")
    if cfg.target_col_raw in df_test.columns:
        raise SchemaError(f"Test should NOT contain target column '{cfg.target_col_raw}'.")

    # Both must have id
    if cfg.id_col_raw not in df_train.columns or cfg.id_col_raw not in df_test.columns:
        raise SchemaError(f"Missing id column '{cfg.id_col_raw}' in train or test.")

    # Raw feature columns must match (ignoring target)
    train_cols = [c for c in df_train.columns if c != cfg.target_col_raw]
    test_cols = df_test.columns.tolist()
    if train_cols != test_cols:
        raise SchemaError(
            "Train/test feature columns mismatch (order or names differ). "
            f"train(no target)={train_cols}, test={test_cols}"
        )

    # Ensure we can rename everything we expect
    missing_train = _missing_cols(df_train, list(cfg.raw_to_canon.keys()))
    # Test doesn't have target so ignore it in required list
    required_test = [c for c in cfg.raw_to_canon.keys() if c != cfg.target_col_raw]
    missing_test = _missing_cols(df_test, required_test)
    if missing_train:
        raise SchemaError(f"Train missing expected raw columns: {missing_train}")
    if missing_test:
        raise SchemaError(f"Test missing expected raw columns: {missing_test}")


def validate_ids(df_train: pd.DataFrame, df_test: pd.DataFrame, df_sub: pd.DataFrame, cfg: CallCenterConfig) -> None:
    # Missing ids
    for name, df in (("train", df_train), ("test", df_test), ("sample_submission", df_sub)):
        if df[cfg.id_col_raw].isna().any():
            examples = df.loc[df[cfg.id_col_raw].isna()].head(10)
            raise SchemaError(f"{name} has missing ids. Examples:\n{examples}")

    # Uniqueness
    if df_train[cfg.id_col_raw].duplicated().any():
        dup_ids = df_train.loc[df_train[cfg.id_col_raw].duplicated(), cfg.id_col_raw].head(10).tolist()
        raise SchemaError(f"Duplicate ids found in train. Examples: {dup_ids}")
    if df_test[cfg.id_col_raw].duplicated().any():
        dup_ids = df_test.loc[df_test[cfg.id_col_raw].duplicated(), cfg.id_col_raw].head(10).tolist()
        raise SchemaError(f"Duplicate ids found in test. Examples: {dup_ids}")

    # Sample submission must match test ids count and ideally same set
    if len(df_sub) != len(df_test):
        raise SchemaError(f"sample_submission rows ({len(df_sub)}) != test rows ({len(df_test)}).")

    # Enforce strict ordering match (Kaggle submission requirement)
    sub_ids = df_sub[cfg.id_col_raw].astype(str).tolist()
    test_ids = df_test[cfg.id_col_raw].astype(str).tolist()
    if sub_ids != test_ids:
        first_mismatch = next((i for i, (a, b) in enumerate(zip(sub_ids, test_ids)) if a != b), None)
        extra_in_sub = sorted(set(sub_ids) - set(test_ids))[:10]
        extra_in_test = sorted(set(test_ids) - set(sub_ids))[:10]
        # Kaggle usually expects same ordering; enforce strict match for safety.
        raise SchemaError(
            "sample_submission ids do not match test ids order exactly."
            + (f" First mismatch at row={first_mismatch}: sub_id={sub_ids[first_mismatch]} vs test_id={test_ids[first_mismatch]}." if first_mismatch is not None else "")
            + (f" Extra ids in sample_submission (first 10): {extra_in_sub}." if extra_in_sub else "")
            + (f" Extra ids in test (first 10): {extra_in_test}." if extra_in_test else "")
        )


def validate_target_values(df_train: pd.DataFrame, cfg: CallCenterConfig) -> None:
    y = pd.to_numeric(df_train[cfg.target_col_raw], errors="coerce")
    if y.isna().any():
        bad = df_train.loc[y.isna(), [cfg.id_col_raw, cfg.target_col_raw]].head(10)
        raise SchemaError(f"Target has non-numeric values. Examples:\n{bad}")
    # We don't hardcode [0..5] to stay robust; just print unique count in summary.


def run_all_checks(
    df_train: pd.DataFrame, df_test: pd.DataFrame, df_sub: pd.DataFrame, cfg: CallCenterConfig
) -> None:
    validate_submission_format(df_sub, cfg)
    validate_raw_columns(df_train, df_test, cfg)
    validate_ids(df_train, df_test, df_sub, cfg)
    validate_target_values(df_train, cfg)


def build_run_summary_raw(
    df_train: pd.DataFrame, df_test: pd.DataFrame, cfg: CallCenterConfig
) -> dict:
    """
    Lightweight stats to save alongside cleaned data.
    """
    y = pd.to_numeric(df_train[cfg.target_col_raw], errors="coerce")
    summary = {
        "rows": {"train": int(len(df_train)), "test": int(len(df_test))},
        "columns": {
            "train": df_train.columns.tolist(),
            "test": df_test.columns.tolist(),
        },
        "missing_rate_train": df_train.isna().mean().to_dict(),
        "missing_rate_test": df_test.isna().mean().to_dict(),
        "target_distribution": y.value_counts(dropna=False).to_dict(),
        "config": asdict(cfg),
    }
    return summary


def _read_csv_with_safe_dtypes(path: Path, cfg: CallCenterConfig) -> pd.DataFrame:
    """
    Read CSVs in a way that makes id comparisons stable across files.

    - `id` is read as pandas string dtype (prevents int vs float vs object surprises).
    - We avoid forcing target dtype here so we can produce a clean SchemaError with examples later.
    """
    return pd.read_csv(path, dtype={cfg.id_col_raw: "string"}, low_memory=False)


def write_json(data: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def run_checks_and_write_summary_from_csvs(
    *,
    train_csv: Path,
    test_csv: Path,
    sample_submission_csv: Path,
    cfg: CallCenterConfig,
    summary_out_path: Path | None = None,
    verbose: bool = True,
) -> dict:
    """
    Colab-friendly entrypoint:
    - Validates raw CSV schema + ids + submission contract
    - Builds raw run summary (missing rates + target distribution)
    - Optionally writes it to `summary_out_path` (e.g. .../outputs/callcenter/clean/run_summary_raw.json)

    Prints "All checks passed" on success; raises SchemaError with a clear message on failure.
    """
    validate_file_exists(train_csv)
    validate_file_exists(test_csv)
    validate_file_exists(sample_submission_csv)

    df_train = _read_csv_with_safe_dtypes(train_csv, cfg)
    df_test = _read_csv_with_safe_dtypes(test_csv, cfg)
    df_sub = _read_csv_with_safe_dtypes(sample_submission_csv, cfg)

    run_all_checks(df_train=df_train, df_test=df_test, df_sub=df_sub, cfg=cfg)
    summary = build_run_summary_raw(df_train=df_train, df_test=df_test, cfg=cfg)

    if summary_out_path is not None:
        write_json(summary, summary_out_path)

    if verbose:
        print("All checks passed")
        if summary_out_path is not None:
            print(f"run_summary_raw.json written to: {summary_out_path}")

    return summary

