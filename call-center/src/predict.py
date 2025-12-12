"""Inference + Kaggle submission writer for the Call Center challenge (Task 8).

What it does:
- Loads cleaned test file (default: FORSA_team/outputs/callcenter/clean/test_clean.csv)
- Loads saved CatBoost model (default: FORSA_team/outputs/callcenter/catboost/model.cbm)
- Rebuilds features exactly (using feature_spec.json when available)
- Predicts `class_int`
- Writes a Kaggle-ready submission CSV with columns exactly like sample_submission.csv:
    id,class_int
- Saves to Drive:
    <team_dir>/submissions/callcenter/callcenter_<run>.csv

Run (Colab example):
  python forca-hack/call-center/src/predict.py --team_dir /content/drive/MyDrive/FORSA_team
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import features
from config import (
    default_config,
    guess_local_data_dir,
    guess_repo_root,
    guess_team_dir,
    resolve_callcenter_data_dir,
    resolve_callcenter_output_dir,
)


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _resolve_feature_spec(cfg, feature_spec: dict[str, Any]) -> dict[str, Any]:
    """
    Support both legacy and current feature spec formats.
    """
    cat_cols = list(feature_spec.get("cat_cols") or list(cfg.cat_cols))

    # Legacy: text_cols=[...], current: text_col="..."
    if "text_col" in feature_spec and feature_spec["text_col"]:
        text_col = str(feature_spec["text_col"])
    else:
        text_cols = feature_spec.get("text_cols") or [cfg.text_col]
        text_col = str(text_cols[0])

    num_cols = list(feature_spec.get("num_cols") or features.build_feature_spec(cfg)["num_cols"])

    # Legacy spec may not have feature_cols.
    feature_cols = list(feature_spec.get("feature_cols") or [*cat_cols, text_col, *num_cols])

    return {
        "cat_cols": cat_cols,
        "text_col": text_col,
        "text_cols": [text_col],
        "num_cols": num_cols,
        "feature_cols": feature_cols,
        "datetime_col": str(feature_spec.get("datetime_col") or cfg.datetime_col),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--team_dir", type=str, default=None, help="Drive root, e.g. /content/drive/MyDrive/FORSA_team")
    p.add_argument("--clean_dir", type=str, default=None, help="Override cleaned data folder (contains test_clean.csv).")
    p.add_argument("--artifacts_dir", type=str, default=None, help="Override artifacts folder (contains model.cbm).")
    p.add_argument("--model_path", type=str, default=None, help="Override model path (model.cbm).")
    p.add_argument("--feature_spec_path", type=str, default=None, help="Override feature_spec.json path.")
    p.add_argument("--sample_submission_path", type=str, default=None, help="Override sample_submission.csv path.")
    p.add_argument("--run_id", type=str, default=None, help="Run id for submission filename.")
    p.add_argument("--out_path", type=str, default=None, help="Explicit submission output path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = default_config()

    team_dir = Path(args.team_dir) if args.team_dir else guess_team_dir()
    if team_dir is not None and not team_dir.exists():
        raise FileNotFoundError(f"--team_dir does not exist: {team_dir}")

    if args.clean_dir:
        clean_dir = Path(args.clean_dir)
    elif team_dir is not None:
        clean_dir = resolve_callcenter_output_dir(team_dir)
    else:
        clean_dir = guess_local_data_dir()

    if args.artifacts_dir:
        artifacts_dir = Path(args.artifacts_dir)
    elif team_dir is not None:
        artifacts_dir = team_dir / "outputs" / "callcenter" / "catboost"
    else:
        artifacts_dir = guess_repo_root() / "call-center" / "artifacts" / "catboost"

    test_path = clean_dir / cfg.clean_test_name
    if not test_path.exists():
        raise FileNotFoundError(f"Cleaned test file not found: {test_path}")

    model_path = Path(args.model_path) if args.model_path else (artifacts_dir / "model.cbm")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    feature_spec_path = (
        Path(args.feature_spec_path) if args.feature_spec_path else (artifacts_dir / "feature_spec.json")
    )
    feature_spec = _load_json(feature_spec_path) if feature_spec_path.exists() else {}

    # Resolve sample submission path
    if args.sample_submission_path:
        sample_path = Path(args.sample_submission_path)
    elif team_dir is not None:
        data_dir = resolve_callcenter_data_dir(team_dir)
        sample_path = data_dir / "sample_submission.csv"
    else:
        sample_path = guess_local_data_dir() / "sample_submission.csv"

    if not sample_path.exists():
        raise FileNotFoundError(f"sample_submission.csv not found: {sample_path}")

    # Import CatBoost lazily
    try:
        from catboost import CatBoostClassifier, Pool  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "catboost is not installed. In Colab, run: pip install -q catboost\n"
            f"Original error: {e}"
        ) from e

    print(f"[predict] test_path={test_path}")
    print(f"[predict] model_path={model_path}")
    print(f"[predict] sample_submission_path={sample_path}")

    df_test = pd.read_csv(test_path)
    if cfg.id_col not in df_test.columns:
        raise ValueError(f"Missing id column '{cfg.id_col}' in cleaned test file.")

    # Feature spec (prefer saved spec to avoid drift; supports legacy formats)
    resolved = _resolve_feature_spec(cfg, feature_spec)
    cat_cols = resolved["cat_cols"]
    text_col = resolved["text_col"]
    feature_cols = resolved["feature_cols"]

    # Ensure engineered columns exist (single source of truth: features.py)
    df_feat = features.engineer_features(df_test.copy(), cfg)

    # Safety fills for model-friendly dtypes
    for c in cat_cols:
        df_feat[c] = df_feat[c].astype("string").fillna("UNK")
    df_feat[text_col] = df_feat[text_col].astype("string").fillna("UNK")

    missing_feats = [c for c in feature_cols if c not in df_feat.columns]
    if missing_feats:
        raise ValueError(f"Missing engineered feature columns: {missing_feats}")

    X = df_feat[feature_cols].copy()

    model = CatBoostClassifier()
    model.load_model(str(model_path))

    pool = Pool(X, cat_features=cat_cols, text_features=[text_col])
    pred = model.predict(pool, prediction_type="Class")
    pred = np.asarray(pred).reshape(-1)
    # CatBoost may return floats; cast to int safely
    pred = pred.astype(int)

    # Build Kaggle submission by copying sample submission and replacing target
    df_sub = pd.read_csv(sample_path)
    expected_cols = [cfg.id_col_raw, cfg.target_col_raw]
    if df_sub.columns.tolist() != expected_cols:
        raise ValueError(f"sample_submission columns mismatch. expected={expected_cols}, got={df_sub.columns.tolist()}")

    pred_df = pd.DataFrame({cfg.id_col_raw: df_test[cfg.id_col], cfg.target_col_raw: pred})
    out = df_sub[[cfg.id_col_raw]].merge(pred_df, on=cfg.id_col_raw, how="left")
    if out[cfg.target_col_raw].isna().any():
        missing = int(out[cfg.target_col_raw].isna().sum())
        raise RuntimeError(f"Failed to align predictions to sample submission ids; missing={missing}")
    out[cfg.target_col_raw] = out[cfg.target_col_raw].astype(int)

    # Output path
    run_id = args.run_id or _utc_run_id()
    if args.out_path:
        out_path = Path(args.out_path)
    else:
        base = team_dir if team_dir is not None else guess_repo_root()
        out_dir = base / "submissions" / "callcenter"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"callcenter_{run_id}.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"[predict] Saved submission: {out_path}")
    print(out.head(3).to_string(index=False))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main()
