"""
Clean dataset generator for the Social Media Comments challenge.

This module is designed to run in Google Colab (with Drive mounted) or locally.

Source of truth
---------------
This file does NOT implement its own cleaning/feature logic.
It calls:
- `preprocess.preprocess_comments_df()`  (raw -> cleaned canonical df)
- `features.engineer_features()`         (adds engineered columns)

Outputs (default)
-----------------
forca-hack/outputs/comments/clean/
  - train_clean.csv
  - test_clean.csv
  - run_summary.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import features
import preprocess
import schema
from config import CommentsConfig, guess_local_data_dir, guess_repo_root, guess_team_dir, resolve_comments_data_dir, resolve_comments_output_dir


def _missing_stats(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    miss_count = df.isna().sum()
    miss_rate = df.isna().mean()
    return {
        "count": {k: int(v) for k, v in miss_count.to_dict().items()},
        "rate": {k: float(v) for k, v in miss_rate.to_dict().items()},
    }


def _prepare_clean_df(df_raw: pd.DataFrame, cfg: CommentsConfig, *, is_train: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Raw -> cleaned canonical -> engineered features -> final ordered output columns.
    """
    df = preprocess.preprocess_comments_df(df_raw, cfg, is_train=is_train)
    preprocess.assert_no_unmasked_phones(df, cfg)

    # Feature engineering
    df = features.engineer_features(df, cfg)
    spec = features.build_feature_spec(cfg)

    # Final column order
    ordered: list[str] = [cfg.id_col]
    if is_train:
        ordered.append(cfg.target_col)

    # Keep raw cols if requested
    if cfg.keep_raw_cols:
        for c in (f"{cfg.platform_col}_raw", f"{cfg.text_col}_raw"):
            if c in df.columns and c not in ordered:
                ordered.append(c)

    for c in (cfg.platform_col, cfg.text_col):
        if c in df.columns and c not in ordered:
            ordered.append(c)

    for c in spec["num_cols"]:
        if c in df.columns and c not in ordered:
            ordered.append(c)

    df = df[ordered].copy()
    meta: dict[str, Any] = {
        "columns": df.columns.tolist(),
        "feature_spec": spec,
    }
    return df, meta


def make_clean_dataset(
    *,
    team_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    train_filename: str = "train.csv",
    test_filename: str = "test_file.csv",
    cfg: CommentsConfig | None = None,
) -> dict[str, str]:
    """
    Generate cleaned datasets and write them to an output folder.
    Returns a dict with output paths (as strings).
    """
    cfg = cfg or CommentsConfig()

    # ----- Resolve input/output paths -----
    if team_dir is None:
        team_dir_resolved = guess_team_dir()
    else:
        team_dir_resolved = Path(team_dir)

    if data_dir is not None:
        data_dir_path = Path(data_dir)
    elif team_dir_resolved is not None:
        data_dir_path = resolve_comments_data_dir(team_dir_resolved)
    else:
        data_dir_path = guess_local_data_dir()

    if output_dir is not None:
        output_dir_path = Path(output_dir)
    elif team_dir_resolved is not None:
        output_dir_path = resolve_comments_output_dir(team_dir_resolved) / "clean"
    else:
        output_dir_path = guess_repo_root() / "outputs" / "comments" / "clean"

    train_path = data_dir_path / train_filename
    test_path = data_dir_path / test_filename

    schema.validate_file_exists(train_path)
    schema.validate_file_exists(test_path)

    output_dir_path.mkdir(parents=True, exist_ok=True)
    train_out = output_dir_path / cfg.clean_train_name
    test_out = output_dir_path / cfg.clean_test_name
    summary_out = output_dir_path / cfg.run_summary_name

    # ----- Read raw (robust encoding) -----
    df_train = schema.read_csv_robust(train_path)
    df_test = schema.read_csv_robust(test_path)

    # ----- Validate schema -----
    checks = schema.run_all_checks(df_train=df_train, df_test=df_test, cfg=cfg)

    # ----- Clean + engineer -----
    train_clean, train_meta = _prepare_clean_df(df_train, cfg, is_train=True)
    test_clean, test_meta = _prepare_clean_df(df_test, cfg, is_train=False)

    # ----- Write outputs -----
    train_clean.to_csv(train_out, index=False, encoding="utf-8")
    test_clean.to_csv(test_out, index=False, encoding="utf-8")

    summary: dict[str, Any] = {
        "task": "comments_clean",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"train": str(train_path), "test": str(test_path)},
        "outputs": {"train_clean": str(train_out), "test_clean": str(test_out)},
        "checks": checks,
        "missing": {"train": _missing_stats(train_clean), "test": _missing_stats(test_clean)},
        "meta": {"train": train_meta, "test": test_meta},
    }
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"train_clean": str(train_out), "test_clean": str(test_out), "summary": str(summary_out)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--train_filename", type=str, default="train.csv")
    parser.add_argument("--test_filename", type=str, default="test_file.csv")
    args = parser.parse_args()

    paths = make_clean_dataset(
        team_dir=args.team_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_filename=args.train_filename,
        test_filename=args.test_filename,
    )
    print(json.dumps(paths, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


