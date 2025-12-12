"""Train a CatBoost baseline for the Call Center challenge (Task 7).

What it does:
- Loads the cleaned train file from Drive outputs folder (default: FORSA_team/outputs/callcenter/clean/train_clean.csv)
- Builds time features from `handle_time`
- Runs Stratified K-Fold CV (default: 5 folds)
- Trains CatBoost with:
  - categorical features (from `config.py`)
  - text feature (`service_content`)
  - class imbalance handling (auto_class_weights="Balanced")
- Saves artifacts to Drive:
  - model.cbm
  - metrics.json (OOF Macro F1 + per-class)
  - feature_spec.json (cat/text/num cols + config + paths)

Run (Colab example):
  python forca-hack/call-center/src/train_catboost.py --team_dir /content/drive/MyDrive/FORSA_team
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
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

import features
import metrics as metrics_utils
from config import (
    default_config,
    guess_local_data_dir,
    guess_repo_root,
    guess_team_dir,
    resolve_callcenter_output_dir,
)


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")


def _json_dump(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _to_py(v: Any) -> Any:
    # Make numpy/pandas types JSON-serializable
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


"""
NOTE: feature engineering is centralized in `features.py`.
Do not re-introduce ad-hoc time feature code here.
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--team_dir", type=str, default=None, help="Drive root, e.g. /content/drive/MyDrive/FORSA_team")
    p.add_argument("--clean_dir", type=str, default=None, help="Override cleaned data folder (contains train_clean.csv).")
    p.add_argument("--artifacts_dir", type=str, default=None, help="Override artifacts output folder.")

    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)

    # CatBoost params (baseline defaults)
    p.add_argument("--iterations", type=int, default=3000)
    p.add_argument("--learning_rate", type=float, default=0.08)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--l2_leaf_reg", type=float, default=3.0)
    p.add_argument("--early_stopping_rounds", type=int, default=200)
    p.add_argument("--task_type", type=str, default="CPU", choices=["CPU", "GPU"])
    p.add_argument("--verbose", type=int, default=200)

    return p.parse_args()


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
        artifacts_dir = team_dir / "outputs" / "callcenter" / "catboost"
    else:
        artifacts_dir = guess_repo_root() / "call-center" / "artifacts" / "catboost"

    train_path = clean_dir / cfg.clean_train_name
    if not train_path.exists():
        raise FileNotFoundError(
            "Cleaned train file not found. Expected one of:\n"
            f"- {train_path}\n"
            "Tip: run your cleaning step to create train_clean.csv in the Drive outputs folder."
        )

    run_id = _utc_run_id()
    repo_root = guess_repo_root()
    commit = _safe_git_commit(repo_root)

    # Import CatBoost lazily so this file can still be syntax-checked without catboost installed.
    try:
        from catboost import CatBoostClassifier, Pool  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "catboost is not installed. In Colab, run: pip install -q catboost\n"
            f"Original error: {e}"
        ) from e

    print(f"[train_catboost] run_id={run_id}")
    print(f"[train_catboost] train_path={train_path}")
    print(f"[train_catboost] artifacts_dir={artifacts_dir}")
    if commit:
        print(f"[train_catboost] git_commit={commit}")

    df = pd.read_csv(train_path)
    if cfg.target_col not in df.columns:
        raise ValueError(f"Missing target column '{cfg.target_col}' in cleaned train file.")

    # Target
    y = pd.to_numeric(df[cfg.target_col], errors="raise").astype(int).to_numpy()
    labels = sorted(np.unique(y).tolist())
    n_classes = len(labels)
    print(f"[train_catboost] rows={len(df)}  classes={labels}  n_classes={n_classes}")

    # Features (single source of truth: features.py)
    df_feat = features.engineer_features(df.copy(), cfg)
    spec = features.build_feature_spec(cfg)
    cat_cols = list(spec["cat_cols"])
    text_col = str(spec["text_col"])
    text_cols = [text_col]
    num_cols = list(spec["num_cols"])
    feature_cols = list(spec["feature_cols"])

    # Safety fills for model-friendly dtypes
    for c in cat_cols:
        df_feat[c] = df_feat[c].astype("string").fillna("UNK")
    df_feat[text_col] = df_feat[text_col].astype("string").fillna("UNK")

    X = df_feat[feature_cols].copy()

    # CatBoost params
    task_type = args.task_type
    if task_type.upper() == "GPU":
        # Text processing support on GPU can vary; default to CPU for stability.
        print("[train_catboost] NOTE: task_type=GPU requested; switching to CPU for text feature support/stability.")
        task_type = "CPU"

    cb_params: dict[str, Any] = {
        "loss_function": "MultiClass",
        "auto_class_weights": "Balanced",
        "iterations": int(args.iterations),
        "learning_rate": float(args.learning_rate),
        "depth": int(args.depth),
        "l2_leaf_reg": float(args.l2_leaf_reg),
        "random_seed": int(args.seed),
        "task_type": task_type,
        "verbose": int(args.verbose),
        "allow_writing_files": False,
    }
    if args.early_stopping_rounds and args.early_stopping_rounds > 0:
        cb_params.update({"od_type": "Iter", "od_wait": int(args.early_stopping_rounds)})

    # CV
    skf = StratifiedKFold(n_splits=int(args.folds), shuffle=True, random_state=int(args.seed))
    oof_pred = np.full(shape=(len(df),), fill_value=-1, dtype=int)
    fold_rows: list[dict[str, Any]] = []
    best_iters: list[int] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        train_pool = Pool(X_tr, y_tr, cat_features=cat_cols, text_features=text_cols)
        val_pool = Pool(X_va, y_va, cat_features=cat_cols, text_features=text_cols)

        model = CatBoostClassifier(**cb_params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        pred_va = model.predict(val_pool)
        pred_va = np.asarray(pred_va).reshape(-1).astype(int)
        oof_pred[va_idx] = pred_va

        fold_f1 = metrics_utils.evaluate(y_va, pred_va, labels=labels)
        bi = model.get_best_iteration()
        if bi is not None and bi >= 0:
            best_iters.append(int(bi) + 1)  # convert to iterations count

        fold_rows.append(
            {
                "fold": fold,
                "n_train": int(len(tr_idx)),
                "n_valid": int(len(va_idx)),
                "macro_f1": float(fold_f1),
                "best_iteration": int(bi) + 1 if bi is not None and bi >= 0 else None,
            }
        )
        print(f"[train_catboost] fold={fold} macro_f1={fold_f1:.5f}")

    if (oof_pred < 0).any():
        bad = int((oof_pred < 0).sum())
        raise RuntimeError(f"OOF predictions not filled for {bad} rows; CV loop failed unexpectedly.")

    oof_macro_f1 = metrics_utils.evaluate(y, oof_pred, labels=labels)
    report = classification_report(y, oof_pred, labels=labels, output_dict=True, zero_division=0)
    per_class_f1 = {str(k): float(v["f1-score"]) for k, v in report.items() if str(k).isdigit()}

    print(f"[train_catboost] OOF Macro F1 = {oof_macro_f1:.5f}")

    # Train final model on full data (use mean best iteration if available)
    final_iters = int(round(float(np.mean(best_iters)))) if best_iters else int(args.iterations)
    final_params = dict(cb_params)
    final_params["iterations"] = final_iters

    full_pool = Pool(X, y, cat_features=cat_cols, text_features=text_cols)
    final_model = CatBoostClassifier(**final_params)
    final_model.fit(full_pool)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "model.cbm"
    metrics_path = artifacts_dir / "metrics.json"
    feature_spec_path = artifacts_dir / "feature_spec.json"

    final_model.save_model(model_path)

    metrics: dict[str, Any] = {
        "run_id": run_id,
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "data": {
            "train_path": str(train_path),
            "rows": int(len(df)),
            "n_classes": int(n_classes),
            "labels": labels,
        },
        "cv": {
            "n_splits": int(args.folds),
            "seed": int(args.seed),
            "folds": fold_rows,
            "oof_macro_f1": oof_macro_f1,
            "per_class_f1": per_class_f1,
            "classification_report": report,
        },
        "final_train": {
            "iterations": int(final_iters),
            "params": {k: _to_py(v) for k, v in final_params.items()},
        },
        "artifacts": {
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
            "feature_spec_path": str(feature_spec_path),
        },
    }
    _json_dump(metrics_path, metrics)

    # Save a single source-of-truth feature spec (plus backward-compatible keys).
    feature_spec: dict[str, Any] = {
        "run_id": run_id,
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "clean_dir": str(clean_dir),
        "train_path": str(train_path),
        **spec,
        # Backward-compat with older predict.py versions / older artifacts
        "text_cols": [text_col],
        "dropped_cols": [cfg.id_col, "sn", cfg.target_col],
        "config": asdict(cfg),
    }
    _json_dump(feature_spec_path, feature_spec)

    print("[train_catboost] Saved artifacts:")
    print(f"- model:        {model_path}")
    print(f"- metrics:      {metrics_path}")
    print(f"- feature spec: {feature_spec_path}")


if __name__ == "__main__":
    # Avoid extremely noisy thread usage in some Colab configs (optional)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main()
