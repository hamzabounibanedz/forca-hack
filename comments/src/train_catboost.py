"""
CatBoost model for Social Media Comments.

Why CatBoost here
-----------------
- Handles categorical feature (`platform`) directly
- Has native text features (`comment`) with strong performance on noisy short texts
- CPU-friendly compared to transformers; good complement for ensembling

Usage (local)
-------------
python forca-hack/comments/src/train_catboost.py
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold

import features
import preprocess
import schema
from config import (
    CommentsConfig,
    cfg_to_dict,
    guess_local_data_dir,
    guess_repo_root,
    guess_team_dir,
    resolve_comments_data_dir,
    resolve_comments_output_dir,
)
from metrics import macro_f1_score


def _build_xy(df_train_raw: pd.DataFrame, cfg: CommentsConfig) -> tuple[pd.DataFrame, np.ndarray, dict[int, int], dict[int, int], list[str], list[str], list[str]]:
    df = preprocess.preprocess_comments_df(df_train_raw, cfg, is_train=True)
    df = features.engineer_features(df, cfg)

    label2id, id2label = schema.build_label_mapping(df[cfg.target_col])
    y_raw = df[cfg.target_col].astype(int).to_numpy()
    y = np.asarray([label2id[int(v)] for v in y_raw], dtype=np.int64)

    spec = features.build_feature_spec(cfg)
    num_cols = list(spec["num_cols"])

    # Model input columns (CatBoost supports mixing)
    cat_cols = [cfg.platform_col]
    text_cols = [cfg.text_col]
    used_cols = cat_cols + text_cols + [c for c in num_cols if c in df.columns]

    X = df[used_cols].copy()
    return X, y, label2id, id2label, used_cols, cat_cols, text_cols


def train_cv(
    df_train_raw: pd.DataFrame,
    cfg: CommentsConfig,
    *,
    n_splits: int,
    seed: int,
    iterations: int,
    learning_rate: float,
    depth: int,
    l2_leaf_reg: float,
    early_stopping_rounds: int,
    verbose: int,
) -> dict[str, Any]:
    X, y, label2id, id2label, used_cols, cat_cols, text_cols = _build_xy(df_train_raw, cfg)
    num_labels = len(label2id)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_scores: list[float] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        tr_pool = Pool(X_tr, y_tr, cat_features=cat_cols, text_features=text_cols)
        va_pool = Pool(X_va, y_va, cat_features=cat_cols, text_features=text_cols)

        model = CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="TotalF1:average=Macro",
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            random_seed=seed,
            allow_writing_files=False,
            verbose=verbose,
        )
        model.fit(tr_pool, eval_set=va_pool, use_best_model=True, early_stopping_rounds=early_stopping_rounds)

        proba = model.predict_proba(va_pool)
        pred = np.asarray(proba).argmax(axis=1)
        f1 = macro_f1_score(y_va, pred, labels=list(range(num_labels)))
        fold_scores.append(float(f1))
        print(f"[fold {fold}] macro_f1={f1:.5f}")

    mean_f1 = float(np.mean(fold_scores)) if fold_scores else 0.0
    std_f1 = float(np.std(fold_scores)) if fold_scores else 0.0
    print(f"[cv] macro_f1 mean={mean_f1:.5f} std={std_f1:.5f}")

    return {
        "label2id": label2id,
        "id2label": id2label,
        "used_cols": used_cols,
        "cat_cols": cat_cols,
        "text_cols": text_cols,
        "fold_scores": fold_scores,
        "macro_f1_mean": mean_f1,
        "macro_f1_std": std_f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--train_filename", type=str, default="train.csv")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--l2_leaf_reg", type=float, default=3.0)
    parser.add_argument("--early_stopping_rounds", type=int, default=100)
    parser.add_argument("--verbose", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = CommentsConfig()

    # Resolve data dir
    if args.data_dir:
        data_dir = Path(args.data_dir)
    elif args.team_dir:
        data_dir = resolve_comments_data_dir(Path(args.team_dir))
    else:
        team = guess_team_dir()
        data_dir = resolve_comments_data_dir(team) if team else guess_local_data_dir()

    train_path = data_dir / args.train_filename
    schema.validate_file_exists(train_path)
    df_train_raw = schema.read_csv_robust(train_path)

    # Resolve output dir
    if args.output_dir:
        out_dir = Path(args.output_dir)
    elif args.team_dir:
        out_dir = resolve_comments_output_dir(Path(args.team_dir)) / "models" / "catboost"
    else:
        out_dir = guess_repo_root() / "outputs" / "comments" / "models" / "catboost"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- CV ----
    cv = train_cv(
        df_train_raw,
        cfg,
        n_splits=args.n_splits,
        seed=args.seed,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose=args.verbose,
    )

    # ---- Fit full model ----
    X, y, label2id, id2label, used_cols, cat_cols, text_cols = _build_xy(df_train_raw, cfg)
    train_pool = Pool(X, y, cat_features=cat_cols, text_features=text_cols)
    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        random_seed=args.seed,
        allow_writing_files=False,
        verbose=args.verbose,
    )
    model.fit(train_pool)

    model_path = out_dir / "model.cbm"
    model.save_model(str(model_path))

    (out_dir / "label_mapping.json").write_text(
        json.dumps({"label2id": cv["label2id"], "id2label": cv["id2label"]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "preprocess_config.json").write_text(
        json.dumps(cfg_to_dict(cfg), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "feature_spec.json").write_text(
        json.dumps({"used_cols": used_cols, "cat_cols": cat_cols, "text_cols": text_cols}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "task": "comments_catboost",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "train_path": str(train_path),
        "output_dir": str(out_dir),
        "cv": {k: v for k, v in cv.items() if k not in {"label2id", "id2label"}},
        "macro_f1_mean": cv["macro_f1_mean"],
        "macro_f1_std": cv["macro_f1_std"],
        "artifacts": {
            "model": str(model_path),
            "label_mapping": str(out_dir / "label_mapping.json"),
            "preprocess_config": str(out_dir / "preprocess_config.json"),
            "feature_spec": str(out_dir / "feature_spec.json"),
        },
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()


