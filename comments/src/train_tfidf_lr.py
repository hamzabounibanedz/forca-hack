"""
Baseline model for Social Media Comments: TF-IDF (word + char) + LogisticRegression.

Why this baseline matters
-------------------------
- Extremely fast (CPU-friendly)
- Often surprisingly strong on noisy short texts (char ngrams help with Arabizi + typos)
- Great diagnostic: if TF-IDF is bad, likely a preprocessing/label issue

Usage (local)
-------------
python forca-hack/comments/src/train_tfidf_lr.py

Usage (Colab/Drive)
-------------------
python comments/src/train_tfidf_lr.py --team_dir /content/drive/MyDrive/FORSA_team
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline

import metrics
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


def _build_text_with_meta(df: pd.DataFrame, cfg: CommentsConfig) -> list[str]:
    """
    Inject low-cardinality metadata as tokens (works well for TF-IDF and keeps the pipeline simple).
    """
    plat = df[cfg.platform_col].astype("string").fillna("unk").str.lower()
    txt = df[cfg.text_col].astype("string").fillna("")
    return (("platform=" + plat + " ") + txt).astype(str).tolist()


def build_model(
    *,
    word_max_features: int = 50_000,
    char_max_features: int = 100_000,
    C: float = 4.0,
    class_weight: str | dict | None = "balanced",
    random_state: int = 42,
) -> Pipeline:
    tfidf = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=word_max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=char_max_features,
                    sublinear_tf=True,
                ),
            ),
        ]
    )

    clf = LogisticRegression(
        solver="saga",
        max_iter=4000,
        C=C,
        class_weight=class_weight,
        random_state=random_state,
    )

    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def train_cv(
    df_train_raw: pd.DataFrame,
    cfg: CommentsConfig,
    *,
    n_splits: int = 5,
    seed: int = 42,
    word_max_features: int = 50_000,
    char_max_features: int = 100_000,
    C: float = 4.0,
) -> dict[str, Any]:
    """
    Stratified K-fold CV with Macro-F1 evaluation.
    """
    df = preprocess.preprocess_comments_df(df_train_raw, cfg, is_train=True)

    # Encode labels to 0..K-1 (stable probability ordering)
    label2id, id2label = schema.build_label_mapping(df[cfg.target_col])
    y_raw = df[cfg.target_col].astype(int).to_numpy()
    y = np.asarray([label2id[int(v)] for v in y_raw], dtype=np.int64)

    X = _build_text_with_meta(df, cfg)

    model = build_model(
        word_max_features=word_max_features,
        char_max_features=char_max_features,
        C=C,
        class_weight="balanced",
        random_state=seed,
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_proba = np.zeros((len(df), len(label2id)), dtype=np.float32)
    fold_scores: list[float] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        m = clone(model)
        X_tr = [X[i] for i in tr_idx]
        X_va = [X[i] for i in va_idx]
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        m.fit(X_tr, y_tr)
        proba = m.predict_proba(X_va)
        oof_proba[va_idx] = proba.astype(np.float32)

        pred = proba.argmax(axis=1)
        f1 = metrics.macro_f1_score(y_va, pred, labels=list(range(len(label2id))))
        fold_scores.append(float(f1))
        print(f"[fold {fold}] macro_f1={f1:.5f}")

    mean_f1 = float(np.mean(fold_scores)) if fold_scores else 0.0
    std_f1 = float(np.std(fold_scores)) if fold_scores else 0.0
    print(f"[cv] macro_f1 mean={mean_f1:.5f} std={std_f1:.5f}")

    return {
        "label2id": label2id,
        "id2label": id2label,
        "oof_proba": oof_proba,
        "fold_scores": fold_scores,
        "macro_f1_mean": mean_f1,
        "macro_f1_std": std_f1,
        "n_splits": n_splits,
        "seed": seed,
        "params": {
            "word_max_features": word_max_features,
            "char_max_features": char_max_features,
            "C": C,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--train_filename", type=str, default="train.csv")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--word_max_features", type=int, default=50_000)
    parser.add_argument("--char_max_features", type=int, default=100_000)
    parser.add_argument("--C", type=float, default=4.0)
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
        out_dir = resolve_comments_output_dir(Path(args.team_dir)) / "models" / "tfidf_lr"
    else:
        out_dir = guess_repo_root() / "outputs" / "comments" / "models" / "tfidf_lr"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- CV ----
    cv = train_cv(
        df_train_raw,
        cfg,
        n_splits=args.n_splits,
        seed=args.seed,
        word_max_features=args.word_max_features,
        char_max_features=args.char_max_features,
        C=args.C,
    )

    # ---- Fit full model ----
    df_clean = preprocess.preprocess_comments_df(df_train_raw, cfg, is_train=True)
    X_full = _build_text_with_meta(df_clean, cfg)
    y_raw = df_clean[cfg.target_col].astype(int).to_numpy()
    label2id = cv["label2id"]
    y_full = np.asarray([label2id[int(v)] for v in y_raw], dtype=np.int64)

    model = build_model(
        word_max_features=args.word_max_features,
        char_max_features=args.char_max_features,
        C=args.C,
        class_weight="balanced",
        random_state=args.seed,
    )
    model.fit(X_full, y_full)

    # ---- Save artifacts ----
    model_path = out_dir / "model.joblib"
    joblib.dump(model, model_path)

    np.save(out_dir / "oof_proba.npy", cv["oof_proba"])
    (out_dir / "label_mapping.json").write_text(
        json.dumps({"label2id": cv["label2id"], "id2label": cv["id2label"]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "preprocess_config.json").write_text(
        json.dumps(cfg_to_dict(cfg), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "task": "comments_tfidf_lr",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "train_path": str(train_path),
        "output_dir": str(out_dir),
        "cv": {k: v for k, v in cv.items() if k not in {"oof_proba"}},
        "artifacts": {
            "model": str(model_path),
            "oof_proba": str(out_dir / "oof_proba.npy"),
            "label_mapping": str(out_dir / "label_mapping.json"),
            "preprocess_config": str(out_dir / "preprocess_config.json"),
        },
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()


