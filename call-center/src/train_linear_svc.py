"""
Train a strong linear baseline for the Call Center challenge:
TF-IDF (char+word) on cleaned text + OneHot categorical features + engineered numeric features.

Why this exists
---------------
CatBoost's built-in text processing can underperform on this dataset's bottlenecks (class 3/4 vs class 2).
A classic sparse linear model with char n-grams is often much stronger.

Inputs
------
- Cleaned train CSV (default): <team_dir>/outputs/callcenter/clean/train_clean.csv

Artifacts (default)
-------------------
<team_dir>/outputs/callcenter/linear_svc/
- model.joblib
- metrics.json
- feature_spec.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder
from sklearn.svm import LinearSVC

import features
import metrics as metrics_utils
from config import default_config, guess_repo_root, guess_team_dir, resolve_callcenter_output_dir


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")


def _json_dump(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _to_py(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.ndarray,)):
        return v.tolist()
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    if isinstance(v, (Path,)):
        return str(v)
    return v


def _safe_git_commit(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--team_dir", type=str, default=None, help="Drive root, e.g. /content/drive/MyDrive/FORSA_team")
    p.add_argument("--clean_dir", type=str, default=None, help="Override cleaned data folder (contains train_clean.csv).")
    p.add_argument("--artifacts_dir", type=str, default=None, help="Override artifacts output folder.")

    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)

    # Vectorizer params (safe defaults)
    p.add_argument("--char_ngram_min", type=int, default=2)
    p.add_argument("--char_ngram_max", type=int, default=6)
    p.add_argument("--char_max_features", type=int, default=250_000)
    p.add_argument("--char_min_df", type=int, default=2)

    p.add_argument("--word_ngram_min", type=int, default=1)
    p.add_argument("--word_ngram_max", type=int, default=2)
    p.add_argument("--word_max_features", type=int, default=120_000)
    p.add_argument("--word_min_df", type=int, default=2)

    # LinearSVC params
    p.add_argument("--C", type=float, default=0.25)
    p.add_argument("--max_iter", type=int, default=12000)

    return p.parse_args()


def _prepare_frame(
    df: pd.DataFrame,
    *,
    cfg,
    spec: dict[str, Any],
) -> pd.DataFrame:
    """
    Build a model-ready DataFrame with stable dtypes.
    The sklearn Pipeline will select/transform by column name.
    """
    df_feat = features.engineer_features(df.copy(), cfg)
    text_col = str(spec["text_col"])
    cat_cols = list(spec["cat_cols"])
    num_cols = list(spec["num_cols"])

    # Safety fills / stable dtypes
    df_feat[text_col] = df_feat[text_col].astype("string").fillna("")
    for c in cat_cols:
        df_feat[c] = df_feat[c].astype("string").fillna("UNK")
    for c in num_cols:
        df_feat[c] = pd.to_numeric(df_feat[c], errors="coerce").fillna(0.0).astype("float32")

    return df_feat[[text_col, *cat_cols, *num_cols]].copy()


def _build_pipeline(
    *,
    text_col: str,
    cat_cols: list[str],
    num_cols: list[str],
    args: argparse.Namespace,
) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(int(args.char_ngram_min), int(args.char_ngram_max)),
                    min_df=int(args.char_min_df),
                    max_features=int(args.char_max_features),
                ),
                text_col,
            ),
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(int(args.word_ngram_min), int(args.word_ngram_max)),
                    min_df=int(args.word_min_df),
                    max_features=int(args.word_max_features),
                ),
                text_col,
            ),
            ("cats", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", Pipeline([("scale", MaxAbsScaler())]), num_cols),
        ],
        remainder="drop",
    )

    clf = LinearSVC(class_weight="balanced", C=float(args.C), max_iter=int(args.max_iter))
    return Pipeline([("pre", pre), ("clf", clf)])


def main() -> None:
    args = parse_args()
    cfg = default_config()

    # Resolve paths (prefer Drive; fallback to repo local for debugging)
    team_dir = Path(args.team_dir) if args.team_dir else guess_team_dir()
    if team_dir is not None and not team_dir.exists():
        raise FileNotFoundError(f"--team_dir does not exist: {team_dir}")

    if args.clean_dir:
        clean_dir = Path(args.clean_dir)
    elif team_dir is not None:
        clean_dir = resolve_callcenter_output_dir(team_dir)
    else:
        clean_dir = guess_repo_root() / "outputs" / "callcenter" / "clean"

    if args.artifacts_dir:
        artifacts_dir = Path(args.artifacts_dir)
    elif team_dir is not None:
        artifacts_dir = team_dir / "outputs" / "callcenter" / "linear_svc"
    else:
        artifacts_dir = guess_repo_root() / "call-center" / "artifacts" / "linear_svc"

    train_path = clean_dir / cfg.clean_train_name
    if not train_path.exists():
        raise FileNotFoundError(
            "Cleaned train file not found. Expected:\n"
            f"- {train_path}\n"
            "Tip: run your cleaning step to create train_clean.csv in the Drive outputs folder."
        )

    run_id = _utc_run_id()
    repo_root = guess_repo_root()
    commit = _safe_git_commit(repo_root)

    print(f"[train_linear_svc] run_id={run_id}")
    print(f"[train_linear_svc] train_path={train_path}")
    print(f"[train_linear_svc] artifacts_dir={artifacts_dir}")
    if commit:
        print(f"[train_linear_svc] git_commit={commit}")

    df = pd.read_csv(train_path)
    if cfg.target_col not in df.columns:
        raise ValueError(f"Missing target column '{cfg.target_col}' in cleaned train file.")

    y = pd.to_numeric(df[cfg.target_col], errors="raise").astype(int).to_numpy()
    labels = sorted(np.unique(y).tolist())
    print(f"[train_linear_svc] rows={len(df)}  classes={labels}  n_classes={len(labels)}")

    # Feature spec (single source of truth)
    spec = features.build_feature_spec(cfg)
    text_col = str(spec["text_col"])
    cat_cols = list(spec["cat_cols"])
    num_cols = list(spec["num_cols"])

    X_df = _prepare_frame(df, cfg=cfg, spec=spec)
    model = _build_pipeline(text_col=text_col, cat_cols=cat_cols, num_cols=num_cols, args=args)

    # CV
    skf = StratifiedKFold(n_splits=int(args.folds), shuffle=True, random_state=int(args.seed))
    oof_pred = np.full(shape=(len(df),), fill_value=-1, dtype=int)
    fold_rows: list[dict[str, Any]] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_df, y), start=1):
        X_tr, X_va = X_df.iloc[tr_idx], X_df.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model.fit(X_tr, y_tr)
        pred_va = model.predict(X_va).astype(int)
        oof_pred[va_idx] = pred_va

        fold_f1 = metrics_utils.evaluate(y_va, pred_va, labels=labels)
        fold_rows.append(
            {
                "fold": fold,
                "n_train": int(len(tr_idx)),
                "n_valid": int(len(va_idx)),
                "macro_f1": float(fold_f1),
            }
        )
        print(f"[train_linear_svc] fold={fold} macro_f1={fold_f1:.5f}")

    if (oof_pred < 0).any():
        bad = int((oof_pred < 0).sum())
        raise RuntimeError(f"OOF predictions not filled for {bad} rows; CV loop failed unexpectedly.")

    oof_macro_f1 = metrics_utils.evaluate(y, oof_pred, labels=labels)
    report = classification_report(y, oof_pred, labels=labels, output_dict=True, zero_division=0)
    per_class_f1 = {str(k): float(v["f1-score"]) for k, v in report.items() if str(k).isdigit()}
    print(f"[train_linear_svc] OOF Macro F1 = {oof_macro_f1:.5f}")

    # Train final model on full data
    model.fit(X_df, y)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "model.joblib"
    metrics_path = artifacts_dir / "metrics.json"
    feature_spec_path = artifacts_dir / "feature_spec.json"

    # Save model (joblib)
    try:
        import joblib  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("joblib is required (usually installed with scikit-learn).") from e
    joblib.dump(model, model_path)

    metrics: dict[str, Any] = {
        "run_id": run_id,
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "data": {
            "train_path": str(train_path),
            "rows": int(len(df)),
            "labels": labels,
        },
        "cv": {
            "n_splits": int(args.folds),
            "seed": int(args.seed),
            "folds": fold_rows,
            "oof_macro_f1": float(oof_macro_f1),
            "per_class_f1": per_class_f1,
            "classification_report": report,
        },
        "model": {
            "type": "LinearSVC",
            "params": {
                "C": float(args.C),
                "max_iter": int(args.max_iter),
                "class_weight": "balanced",
            },
            "vectorizers": {
                "char": {
                    "analyzer": "char_wb",
                    "ngram_range": [int(args.char_ngram_min), int(args.char_ngram_max)],
                    "min_df": int(args.char_min_df),
                    "max_features": int(args.char_max_features),
                },
                "word": {
                    "analyzer": "word",
                    "ngram_range": [int(args.word_ngram_min), int(args.word_ngram_max)],
                    "min_df": int(args.word_min_df),
                    "max_features": int(args.word_max_features),
                },
            },
        },
        "artifacts": {
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
            "feature_spec_path": str(feature_spec_path),
        },
    }
    _json_dump(metrics_path, metrics)

    feature_spec: dict[str, Any] = {
        "run_id": run_id,
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "clean_dir": str(clean_dir),
        "train_path": str(train_path),
        **spec,
        "config": asdict(cfg),
        "linear_svc": {
            "C": float(args.C),
            "max_iter": int(args.max_iter),
            "char_vectorizer": metrics["model"]["vectorizers"]["char"],
            "word_vectorizer": metrics["model"]["vectorizers"]["word"],
        },
    }
    _json_dump(feature_spec_path, feature_spec)

    print("[train_linear_svc] Saved artifacts:")
    print(f"- model:        {model_path}")
    print(f"- metrics:      {metrics_path}")
    print(f"- feature spec: {feature_spec_path}")


if __name__ == "__main__":
    # Avoid extremely noisy thread usage in some Colab configs (optional)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main()

