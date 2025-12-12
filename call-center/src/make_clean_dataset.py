"""
Clean dataset generator for the Call Center challenge.

This module is designed to run in Google Colab (with Drive mounted) or locally.

Source of truth
---------------
This file does NOT implement its own cleaning/feature logic.
It calls:
- `preprocess.preprocess_callcenter_df()`  (raw -> cleaned canonical df)
- `features.engineer_features()`          (adds engineered columns)

Outputs
-------
Writes to Drive (when available):
  <team_dir>/outputs/callcenter/clean/train_clean.csv
  <team_dir>/outputs/callcenter/clean/test_clean.csv
  <team_dir>/outputs/callcenter/clean/run_summary.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import features
import preprocess
import schema
from config import (
    CallCenterConfig,
    default_config,
    guess_local_data_dir,
    guess_repo_root,
    guess_team_dir,
    resolve_callcenter_data_dir,
    resolve_callcenter_output_dir,
)


def _apply_rare_grouping(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    cols: list[str],
    min_count: int,
    rare_token: str,
    unk_token: str = preprocess.UNK,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Frequency bucketing for high-cardinality categoricals:
      categories with train frequency < min_count -> rare_token

    Returns (train_df, test_df, meta).
    """
    if min_count <= 1:
        return df_train, df_test, {"enabled": False, "reason": "min_count<=1"}

    train_out = df_train.copy()
    test_out = df_test.copy()

    meta: dict[str, Any] = {
        "enabled": True,
        "min_count": int(min_count),
        "rare_token": str(rare_token),
        "cols": [],
    }

    for c in cols:
        if c not in train_out.columns or c not in test_out.columns:
            continue

        tr = train_out[c].astype("string").fillna(unk_token)
        te = test_out[c].astype("string").fillna(unk_token)

        vc = tr.value_counts(dropna=False)
        keep = set(vc[vc >= min_count].index.astype(str).tolist())
        keep.add(unk_token)

        train_out[c] = tr.map(lambda x: x if str(x) in keep else rare_token)
        test_out[c] = te.map(lambda x: x if str(x) in keep else rare_token)

        meta["cols"].append(
            {
                "col": c,
                "train_unique_before": int(tr.nunique(dropna=False)),
                "train_unique_after": int(train_out[c].astype("string").nunique(dropna=False)),
                "n_rare_train": int((train_out[c] == rare_token).sum()),
                "n_rare_test": int((test_out[c] == rare_token).sum()),
            }
        )

    return train_out, test_out, meta


def _prepare_clean_df(df_raw: pd.DataFrame, cfg: CallCenterConfig, *, is_train: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Raw -> canonical cleaned -> engineered features -> final ordered output columns.
    """
    # Preprocessing (single source of truth)
    df = preprocess.preprocess_callcenter_df(df_raw, cfg)
    # DoD sanity check: after cleaning, no raw phone-like numbers should remain.
    preprocess.assert_no_unmasked_phones(df, cfg)

    # Standardize id/target types (model-friendly)
    if cfg.id_col in df.columns:
        df[cfg.id_col] = pd.to_numeric(df[cfg.id_col], errors="coerce").astype("Int64")
    if is_train and cfg.target_col in df.columns:
        df[cfg.target_col] = pd.to_numeric(df[cfg.target_col], errors="coerce").astype("Int64")

    # Datetime parse success (post parse step)
    dt = pd.to_datetime(df[cfg.datetime_col], errors="coerce")
    parse_rate = float(dt.notna().mean()) if len(df) else 0.0

    # Feature engineering (single source of truth)
    df = features.engineer_features(df, cfg)
    spec = features.build_feature_spec(cfg)
    engineered_num = list(spec["num_cols"])

    # Final column order: id/target + contract columns + engineered numeric features
    base_cols = list(cfg.keep_cols)
    if not cfg.keep_sn_in_outputs and "sn" in base_cols:
        base_cols = [c for c in base_cols if c != "sn"]
    base_cols = [c for c in base_cols if c in df.columns]

    ordered: list[str] = [cfg.id_col]
    if is_train:
        ordered.append(cfg.target_col)
    ordered.extend([c for c in base_cols if c not in ordered])
    ordered.extend([c for c in engineered_num if c in df.columns and c not in ordered])

    df = df[ordered].copy()

    meta: dict[str, Any] = {
        "engineered_num_features": engineered_num,
        "datetime_parse_success_rate": parse_rate,
        "datetime_parse_success_pct": 100.0 * parse_rate,
        "columns": df.columns.tolist(),
        "feature_spec": spec,
    }
    return df, meta


def _missing_stats(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    miss_count = df.isna().sum()
    miss_rate = df.isna().mean()
    return {
        "count": {k: int(v) for k, v in miss_count.to_dict().items()},
        "rate": {k: float(v) for k, v in miss_rate.to_dict().items()},
    }


def make_clean_dataset(
    *,
    team_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    train_filename: str = "train.csv",
    test_filename: str = "test.csv",
    submission_filename: str = "sample_submission.csv",
    cfg: CallCenterConfig | None = None,
) -> dict[str, str]:
    """
    Generate cleaned datasets and write them to an output folder.

    Returns a dict with output paths (as strings).
    """
    cfg = cfg or default_config()

    # ----- Resolve input/output paths -----
    if team_dir is None:
        team_dir_resolved = guess_team_dir()
    else:
        team_dir_resolved = Path(team_dir)

    if data_dir is not None:
        data_dir_path = Path(data_dir)
    elif team_dir_resolved is not None:
        data_dir_path = resolve_callcenter_data_dir(team_dir_resolved)
    else:
        data_dir_path = guess_local_data_dir()

    if output_dir is not None:
        output_dir_path = Path(output_dir)
    elif team_dir_resolved is not None:
        output_dir_path = resolve_callcenter_output_dir(team_dir_resolved)
    else:
        output_dir_path = guess_repo_root() / "outputs" / "callcenter" / "clean"

    train_path = data_dir_path / train_filename
    test_path = data_dir_path / test_filename
    sub_path = data_dir_path / submission_filename

    schema.validate_file_exists(train_path)
    schema.validate_file_exists(test_path)
    schema.validate_file_exists(sub_path)

    output_dir_path.mkdir(parents=True, exist_ok=True)
    train_out = output_dir_path / cfg.clean_train_name
    test_out = output_dir_path / cfg.clean_test_name
    summary_out = output_dir_path / cfg.run_summary_name

    # ----- Read raw -----
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_sub = pd.read_csv(sub_path)

    # ----- Validate schema -----
    schema.run_all_checks(df_train=df_train, df_test=df_test, df_sub=df_sub, cfg=cfg)

    # ----- Clean -----
    train_clean, train_meta = _prepare_clean_df(df_train, cfg, is_train=True)
    test_clean, test_meta = _prepare_clean_df(df_test, cfg, is_train=False)

    # ----- High-cardinality strategy (ACTEL/DOT) -----
    train_clean, test_clean, rare_meta = _apply_rare_grouping(
        train_clean,
        test_clean,
        cols=list(cfg.rare_group_cols),
        min_count=int(cfg.rare_min_count),
        rare_token=str(cfg.rare_token),
    )

    # ----- Task C/D: profiling + optional safe imputation for subs_level -----
    def _raw_name_for(canon: str) -> str | None:
        # Find raw column name that maps to this canonical name.
        for raw, c in (cfg.raw_to_canon or {}).items():
            if c == canon:
                return raw
        return None

    cat_profile: dict[str, Any] = {"train": {}, "test": {}}
    for canon in ("customer_level", "dept_traitement", "dept_initial"):
        raw_name = _raw_name_for(canon)
        if raw_name and raw_name in df_train.columns and canon in train_clean.columns:
            cat_profile["train"][canon] = {
                "raw_nunique": int(df_train[raw_name].nunique(dropna=True)),
                "clean_nunique": int(train_clean[canon].nunique(dropna=True)),
                "clean_top_values": train_clean[canon].astype("string").value_counts(dropna=False).head(10).to_dict(),
            }
        if raw_name and raw_name in df_test.columns and canon in test_clean.columns:
            cat_profile["test"][canon] = {
                "raw_nunique": int(df_test[raw_name].nunique(dropna=True)),
                "clean_nunique": int(test_clean[canon].nunique(dropna=True)),
                "clean_top_values": test_clean[canon].astype("string").value_counts(dropna=False).head(10).to_dict(),
            }

    subs_profile: dict[str, Any] = {}
    if {"subs_level", "service_type"}.issubset(train_clean.columns):
        # Missing is represented as UNK by preprocess.clean_categoricals; we ALSO add subs_level_missing feature (Task D).
        subs = train_clean["subs_level"].astype("string")
        if "subs_level_missing" in train_clean.columns:
            missing_mask = train_clean["subs_level_missing"].to_numpy() > 0.5
        else:
            missing_mask = (subs == "UNK").to_numpy()

        known_mask = ~missing_mask
        total_missing = int(missing_mask.sum())
        subs_profile["missing_count"] = total_missing
        subs_profile["missing_rate"] = float(total_missing / len(train_clean)) if len(train_clean) else 0.0

        postpaid_label = "postpaid"
        # Cross-tab style evidence: P(postpaid | service_type) on known rows only
        if known_mask.any():
            known = train_clean.loc[known_mask, ["service_type", "subs_level"]].copy()
            known["is_postpaid"] = (known["subs_level"].astype("string") == postpaid_label).astype(int)
            support = known.groupby("service_type").size()
            rate = known.groupby("service_type")["is_postpaid"].mean()

            subs_profile["postpaid_rate_by_service_type"] = {str(k): float(v) for k, v in rate.to_dict().items()}
            subs_profile["support_by_service_type"] = {str(k): int(v) for k, v in support.to_dict().items()}

            # Decide eligible service_type values for imputing missing -> postpaid
            threshold = 0.95
            min_support = 100
            eligible = rate.index[(support >= min_support) & (rate >= threshold)].tolist()
            subs_profile["decision_threshold"] = threshold
            subs_profile["decision_min_support"] = min_support
            subs_profile["eligible_service_types"] = [str(x) for x in eligible]

            miss_counts = train_clean.loc[missing_mask, "service_type"].value_counts(dropna=False)
            subs_profile["missing_by_service_type"] = {str(k): int(v) for k, v in miss_counts.to_dict().items()}

            if total_missing > 0:
                expected = float((miss_counts * rate.reindex(miss_counts.index).fillna(0)).sum() / miss_counts.sum())
                coverage = float(miss_counts.loc[miss_counts.index.isin(eligible)].sum() / miss_counts.sum())
            else:
                expected = 0.0
                coverage = 0.0

            subs_profile["expected_postpaid_prob_for_missing"] = expected
            subs_profile["eligible_coverage_on_missing"] = coverage
            subs_profile["recommendation"] = (
                "impute_postpaid_for_missing_when_service_type_is_eligible"
                if (eligible and expected >= threshold)
                else "keep_UNK (no strong evidence)"
            )

            # Apply conservative per-service_type imputation if evidence is strong enough
            did_impute = bool(eligible and expected >= threshold)
            subs_profile["did_impute"] = did_impute
            if did_impute:
                # Train
                mask_train_imp = missing_mask & train_clean["service_type"].isin(eligible).to_numpy()
                train_clean.loc[mask_train_imp, "subs_level"] = postpaid_label
                subs_profile["imputed_train_rows"] = int(mask_train_imp.sum())

                # Test (use same eligible service_type list)
                subs_test = test_clean["subs_level"].astype("string") if "subs_level" in test_clean.columns else None
                if subs_test is not None and "service_type" in test_clean.columns:
                    if "subs_level_missing" in test_clean.columns:
                        miss_test = test_clean["subs_level_missing"].to_numpy() > 0.5
                    else:
                        miss_test = (subs_test == "UNK").to_numpy()
                    mask_test_imp = miss_test & test_clean["service_type"].isin(eligible).to_numpy()
                    test_clean.loc[mask_test_imp, "subs_level"] = postpaid_label
                    subs_profile["imputed_test_rows"] = int(mask_test_imp.sum())
                else:
                    subs_profile["imputed_test_rows"] = 0
            else:
                subs_profile["imputed_train_rows"] = 0
                subs_profile["imputed_test_rows"] = 0

            # IMPORTANT: if we updated subs_level, refresh derived missing-indicator feature to avoid stale values.
            # (subs_level_missing is used by training; it must match the final subs_level values).
            if "subs_level" in train_clean.columns:
                train_clean["subs_level_missing"] = (
                    (train_clean["subs_level"].astype("string") == preprocess.UNK).astype("float32")
                )
            if "subs_level" in test_clean.columns:
                test_clean["subs_level_missing"] = (
                    (test_clean["subs_level"].astype("string") == preprocess.UNK).astype("float32")
                )
        else:
            subs_profile["error"] = "No non-missing subs_level rows available for profiling."

        # Final distribution after optional imputation
        subs_profile["final_value_counts_train"] = (
            train_clean["subs_level"].astype("string").value_counts(dropna=False).head(10).to_dict()
        )

    # ----- Write outputs -----
    train_clean.to_csv(train_out, index=False)
    test_clean.to_csv(test_out, index=False)

    # ----- Run summary -----
    model_features_train = [c for c in train_clean.columns if c not in {cfg.id_col, cfg.target_col}]
    model_features_test = [c for c in test_clean.columns if c != cfg.id_col]

    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paths": {
            "train_raw": str(train_path),
            "test_raw": str(test_path),
            "sample_submission_raw": str(sub_path),
            "train_clean": str(train_out),
            "test_clean": str(test_out),
            "run_summary": str(summary_out),
        },
        "rows": {"train": int(len(train_clean)), "test": int(len(test_clean))},
        "missing": {"train": _missing_stats(train_clean), "test": _missing_stats(test_clean)},
        "datetime_parse_success": {
            "rate": {
                "train": float(train_meta["datetime_parse_success_rate"]),
                "test": float(test_meta["datetime_parse_success_rate"]),
            },
            "pct": {
                "train": float(train_meta["datetime_parse_success_pct"]),
                "test": float(test_meta["datetime_parse_success_pct"]),
            },
        },
        "engineered_features": train_meta["engineered_num_features"],
        "feature_spec": train_meta["feature_spec"],
        "high_cardinality": {"rare_grouping": rare_meta},
        "categorical_profile": cat_profile,
        "subs_level_profile": subs_profile,
        "clean_columns": {"train": train_meta["columns"], "test": test_meta["columns"]},
        "model_features": {"train": model_features_train, "test": model_features_test},
    }

    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {"train_clean": str(train_out), "test_clean": str(test_out), "run_summary": str(summary_out)}


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate cleaned Call Center datasets (Drive-friendly).")
    p.add_argument(
        "--team_dir",
        type=str,
        default=None,
        help="Path to team folder in Drive (e.g. /content/drive/MyDrive/FORSA_team). "
        "If omitted, we try to auto-detect it in Colab.",
    )
    p.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Explicit data folder containing train.csv/test.csv/sample_submission.csv. "
        "Overrides team_dir resolution if provided.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Explicit output folder. If omitted, we write to <team_dir>/outputs/callcenter/clean/ "
        "(Drive) when available, else to <repo>/outputs/callcenter/clean/",
    )
    p.add_argument("--train_filename", type=str, default="train.csv")
    p.add_argument("--test_filename", type=str, default="test.csv")
    p.add_argument("--submission_filename", type=str, default="sample_submission.csv")
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    out = make_clean_dataset(
        team_dir=args.team_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_filename=args.train_filename,
        test_filename=args.test_filename,
        submission_filename=args.submission_filename,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()


