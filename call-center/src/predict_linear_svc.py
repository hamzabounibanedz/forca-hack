"""
Inference + Kaggle submission writer for the Call Center challenge using the LinearSVC TF-IDF model.

Loads:
- cleaned test file:    <team_dir>/outputs/callcenter/clean/test_clean.csv
- trained model:        <team_dir>/outputs/callcenter/linear_svc/model.joblib
- feature_spec.json:    <team_dir>/outputs/callcenter/linear_svc/feature_spec.json
- sample_submission.csv <team_dir>/data/callcenter/sample_submission.csv

Writes:
- <team_dir>/submissions/callcenter/callcenter_linear_svc_<run>.csv
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--team_dir", type=str, default=None, help="Drive root, e.g. /content/drive/MyDrive/FORSA_team")
    p.add_argument("--clean_dir", type=str, default=None, help="Override cleaned data folder (contains test_clean.csv).")
    p.add_argument("--artifacts_dir", type=str, default=None, help="Override artifacts folder (contains model.joblib).")
    p.add_argument("--model_path", type=str, default=None, help="Override model path (model.joblib).")
    p.add_argument("--feature_spec_path", type=str, default=None, help="Override feature_spec.json path.")
    p.add_argument("--sample_submission_path", type=str, default=None, help="Override sample_submission.csv path.")
    p.add_argument("--run_id", type=str, default=None, help="Run id for submission filename.")
    p.add_argument("--out_path", type=str, default=None, help="Explicit submission output path.")
    return p.parse_args()


def _prepare_frame(df: pd.DataFrame, cfg, spec: dict[str, Any]) -> pd.DataFrame:
    """
    Build a model-ready DataFrame with stable dtypes, matching training expectations.
    """
    spec2 = features.normalize_feature_spec(spec, cfg)
    df_feat = features.engineer_features(df.copy(), cfg)

    text_col = str(spec2["text_col"])
    cat_cols = list(spec2["cat_cols"])
    num_cols = list(spec2.get("num_cols") or [])

    df_feat[text_col] = df_feat[text_col].astype("string").fillna("")
    for c in cat_cols:
        df_feat[c] = df_feat[c].astype("string").fillna("UNK")
    for c in num_cols:
        df_feat[c] = pd.to_numeric(df_feat[c], errors="coerce").fillna(0.0).astype("float32")

    cols = [text_col, *cat_cols, *num_cols]
    missing = [c for c in cols if c not in df_feat.columns]
    if missing:
        raise ValueError(f"Missing engineered feature columns: {missing}")
    return df_feat[cols].copy()


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
        artifacts_dir = team_dir / "outputs" / "callcenter" / "linear_svc"
    else:
        artifacts_dir = guess_repo_root() / "call-center" / "artifacts" / "linear_svc"

    test_path = clean_dir / cfg.clean_test_name
    if not test_path.exists():
        raise FileNotFoundError(f"Cleaned test file not found: {test_path}")

    model_path = Path(args.model_path) if args.model_path else (artifacts_dir / "model.joblib")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    feature_spec_path = (
        Path(args.feature_spec_path) if args.feature_spec_path else (artifacts_dir / "feature_spec.json")
    )
    if not feature_spec_path.exists():
        raise FileNotFoundError(f"feature_spec.json not found: {feature_spec_path}")
    feature_spec = _load_json(feature_spec_path)

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

    print(f"[predict_linear_svc] test_path={test_path}")
    print(f"[predict_linear_svc] model_path={model_path}")
    print(f"[predict_linear_svc] feature_spec_path={feature_spec_path}")
    print(f"[predict_linear_svc] sample_submission_path={sample_path}")

    # Load model
    try:
        import joblib  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("joblib is required (usually installed with scikit-learn).") from e
    model = joblib.load(model_path)

    df_test = pd.read_csv(test_path)
    if cfg.id_col not in df_test.columns:
        raise ValueError(f"Missing id column '{cfg.id_col}' in cleaned test file.")

    X = _prepare_frame(df_test, cfg=cfg, spec=feature_spec)
    pred = model.predict(X)
    pred = np.asarray(pred).reshape(-1).astype(int)

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
        out_path = out_dir / f"callcenter_linear_svc_{run_id}.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"[predict_linear_svc] Saved submission: {out_path}")
    print(out.head(3).to_string(index=False))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main()

