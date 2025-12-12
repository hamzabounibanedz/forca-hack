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
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
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

    # ----- Optional deterministic post-processing (time-window gating) -----
    # Learns per-class [min(handle_time), max(handle_time)] from TRAIN only.
    # At inference, if a prediction for some class is outside its observed window,
    # force it to a fallback class (default=2).
    #
    # This is OFF by default at inference; here we only *evaluate* it on OOF so you can see
    # whether it helps and save the needed metadata in artifacts.
    p.add_argument(
        "--time_gate_classes",
        type=str,
        default="0,3,4,5",
        help="Comma-separated class labels to apply time-gating to (e.g. '0,3,4').",
    )
    p.add_argument(
        "--time_gate_fallback_class",
        type=int,
        default=2,
        help="Fallback class used when a prediction violates its class time window.",
    )
    p.add_argument(
        "--time_gate_fallback_map",
        type=str,
        default="5:1",
        help=(
            "Optional per-class fallback overrides in the form 'pred:fb,pred:fb'. "
            "Example: '5:1' means if we predicted class 5 outside its window, force it to class 1 instead of the default fallback."
        ),
    )
    p.add_argument(
        "--disable_time_gating_eval",
        action="store_true",
        help="If set, do not compute/print the time-gated OOF Macro-F1 (still saves time windows in feature_spec).",
    )

    # Optional: save fold models for ensembling at inference time
    p.add_argument(
        "--save_fold_models",
        action="store_true",
        help="If set, saves each CV fold model to <artifacts_dir>/fold_models/ for ensemble inference.",
    )
    p.add_argument(
        "--fold_models_dirname",
        type=str,
        default="fold_models",
        help="Subfolder name under artifacts_dir used to store fold models.",
    )
    p.add_argument(
        "--joblib_compress",
        type=int,
        default=3,
        help="joblib compression level for saved models (0=no compression).",
    )

    return p.parse_args()


def _parse_int_list(s: str) -> list[int]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    out: list[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except Exception:
            continue
    # de-dup while preserving order
    seen = set()
    dedup: list[int] = []
    for v in out:
        if v in seen:
            continue
        seen.add(v)
        dedup.append(v)
    return dedup


def _parse_fallback_map(s: str) -> dict[int, int]:
    """
    Parse 'pred:fb,pred:fb' into {pred: fb}. Ignores malformed parts.
    """
    out: dict[int, int] = {}
    txt = (s or "").strip()
    if not txt:
        return out
    for part in txt.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        a, b = part.split(":", 1)
        a = a.strip()
        b = b.strip()
        try:
            pred = int(a)
            fb = int(b)
        except Exception:
            continue
        out[pred] = fb
    return out


def _compute_time_windows_by_class(df: pd.DataFrame, *, cfg, y: np.ndarray) -> dict[str, dict[str, str]]:
    """
    Compute per-class [min,max] timestamps from the cleaned training dataframe.

    Returns a JSON-friendly dict:
      {"0": {"min": "...", "max": "..."}, ...}
    """
    if cfg.datetime_col not in df.columns:
        return {}
    dt = pd.to_datetime(df[cfg.datetime_col], errors="coerce")
    if dt.isna().all():
        return {}

    out: dict[str, dict[str, str]] = {}
    for cls in sorted(np.unique(y).tolist()):
        mask = y == int(cls)
        if not mask.any():
            continue
        s = dt.loc[mask]
        mn = s.min()
        mx = s.max()
        if pd.isna(mn) or pd.isna(mx):
            continue
        out[str(int(cls))] = {"min": mn.isoformat(), "max": mx.isoformat()}
    return out


def _apply_time_gating(
    pred: np.ndarray,
    *,
    handle_time: pd.Series,
    time_windows_by_class: dict[str, dict[str, str]],
    gate_classes: list[int],
    fallback_class: int,
    fallback_map: dict[int, int] | None = None,
) -> tuple[np.ndarray, int]:
    """
    Apply time-window gating to a predicted label vector.

    If pred[i] in gate_classes AND handle_time[i] is outside the train-observed window for that class,
    then pred[i] -> fallback_class.

    Returns (pred_gated, n_changed).
    """
    if not len(pred):
        return pred, 0
    if not time_windows_by_class:
        return pred, 0

    dt = pd.to_datetime(handle_time, errors="coerce")
    out = np.asarray(pred).copy().astype(int)
    changed = 0
    fb_map = fallback_map or {}

    for cls in gate_classes:
        win = time_windows_by_class.get(str(int(cls)))
        if not win:
            continue
        mn = pd.to_datetime(win.get("min"), errors="coerce")
        mx = pd.to_datetime(win.get("max"), errors="coerce")
        if pd.isna(mn) or pd.isna(mx):
            continue

        mask_cls = out == int(cls)
        if not mask_cls.any():
            continue
        mask_time = dt.notna() & ((dt < mn) | (dt > mx))
        mask = mask_cls & mask_time.to_numpy()
        if mask.any():
            fb = int(fb_map.get(int(cls), fallback_class))
            out[mask] = fb
            changed += int(mask.sum())

    return out, changed


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

    # IMPORTANT: set random_state for deterministic liblinear shuffling.
    clf = LinearSVC(
        class_weight="balanced",
        C=float(args.C),
        max_iter=int(args.max_iter),
        random_state=int(getattr(args, "seed", 42)),
    )
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

    # Stabilize row order to make CV splits reproducible even if CSV row order changes.
    if cfg.id_col in df.columns:
        df = df.sort_values(cfg.id_col, kind="mergesort").reset_index(drop=True)

    # Parse handle_time once (used by feature engineering and optional time-gating metadata)
    if cfg.datetime_col in df.columns:
        df[cfg.datetime_col] = pd.to_datetime(df[cfg.datetime_col], errors="coerce")

    y = pd.to_numeric(df[cfg.target_col], errors="raise").astype(int).to_numpy()
    labels = sorted(np.unique(y).tolist())
    print(f"[train_linear_svc] rows={len(df)}  classes={labels}  n_classes={len(labels)}")

    # Train-only metadata for optional inference postprocessing
    time_windows_by_class = _compute_time_windows_by_class(df, cfg=cfg, y=y)
    gate_classes = _parse_int_list(getattr(args, "time_gate_classes", "0,3,4"))
    fallback_class = int(getattr(args, "time_gate_fallback_class", 2))
    fallback_map = _parse_fallback_map(getattr(args, "time_gate_fallback_map", ""))

    # Feature spec (single source of truth)
    spec = features.build_feature_spec(cfg)
    text_col = str(spec["text_col"])
    cat_cols = list(spec["cat_cols"])
    num_cols = list(spec["num_cols"])

    X_df = _prepare_frame(df, cfg=cfg, spec=spec)
    model = _build_pipeline(text_col=text_col, cat_cols=cat_cols, num_cols=num_cols, args=args)

    # joblib is needed for model saving (final + optional fold models)
    try:
        import joblib  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("joblib is required (usually installed with scikit-learn).") from e

    # CV
    skf = StratifiedKFold(n_splits=int(args.folds), shuffle=True, random_state=int(args.seed))
    oof_pred = np.full(shape=(len(df),), fill_value=-1, dtype=int)
    fold_rows: list[dict[str, Any]] = []
    fold_model_paths: list[str] = []

    fold_models_dir: Path | None = None
    if bool(args.save_fold_models):
        fold_models_dir = artifacts_dir / str(args.fold_models_dirname)
        fold_models_dir.mkdir(parents=True, exist_ok=True)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_df, y), start=1):
        X_tr, X_va = X_df.iloc[tr_idx], X_df.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model.fit(X_tr, y_tr)
        pred_va = model.predict(X_va).astype(int)
        oof_pred[va_idx] = pred_va

        # Optionally persist the fold model (for ensemble inference)
        if fold_models_dir is not None:
            fold_model_path = fold_models_dir / f"model_fold{fold}.joblib"
            joblib.dump(model, fold_model_path, compress=int(args.joblib_compress))
            fold_model_paths.append(str(fold_model_path))

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

    # Optional deterministic postprocess evaluation (does NOT change the trained model)
    time_gating_eval: dict[str, Any] | None = None
    if (
        not bool(getattr(args, "disable_time_gating_eval", False))
        and time_windows_by_class
        and (cfg.datetime_col in df.columns)
    ):
        oof_gated, n_changed = _apply_time_gating(
            oof_pred,
            handle_time=df[cfg.datetime_col],
            time_windows_by_class=time_windows_by_class,
            gate_classes=gate_classes,
            fallback_class=fallback_class,
            fallback_map=fallback_map,
        )
        oof_macro_f1_gated = metrics_utils.evaluate(y, oof_gated, labels=labels)
        report_gated = classification_report(y, oof_gated, labels=labels, output_dict=True, zero_division=0)
        per_class_f1_gated = {str(k): float(v["f1-score"]) for k, v in report_gated.items() if str(k).isdigit()}
        print(
            f"[train_linear_svc] OOF Macro F1 (time-gated classes={gate_classes} -> fallback={fallback_class} map={fallback_map}) = {oof_macro_f1_gated:.5f} "
            f"(changed={n_changed})"
        )
        time_gating_eval = {
            "enabled": True,
            "gate_classes": gate_classes,
            "fallback_class": fallback_class,
            "fallback_map": {str(k): int(v) for k, v in sorted(fallback_map.items())},
            "changed": int(n_changed),
            "oof_macro_f1": float(oof_macro_f1_gated),
            "per_class_f1": per_class_f1_gated,
            "classification_report": report_gated,
        }

    # Save fold ensemble manifest if requested
    if fold_model_paths:
        fold_manifest_path = artifacts_dir / "fold_models.json"
        _json_dump(
            fold_manifest_path,
            {
                "run_id": run_id,
                "utc_time": datetime.now(timezone.utc).isoformat(),
                "n_folds": int(args.folds),
                "seed": int(args.seed),
                "model_paths": fold_model_paths,
                "labels": labels,
            },
        )

    # Train final model on full data
    model.fit(X_df, y)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "model.joblib"
    metrics_path = artifacts_dir / "metrics.json"
    feature_spec_path = artifacts_dir / "feature_spec.json"

    # Save model (joblib)
    joblib.dump(model, model_path, compress=int(args.joblib_compress))

    metrics: dict[str, Any] = {
        "run_id": run_id,
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "env": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
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
            "postprocess": {
                "time_gating": time_gating_eval,
            },
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
            "fold_models": {
                "enabled": bool(fold_model_paths),
                "n_models": int(len(fold_model_paths)),
                "dir": str(fold_models_dir) if fold_models_dir is not None else None,
                "manifest_path": str(artifacts_dir / "fold_models.json") if fold_model_paths else None,
            },
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
        # Train-only metadata for deterministic inference postprocessing
        "time_windows_by_class": time_windows_by_class,
        "time_gating_defaults": {
            "gate_classes": gate_classes,
            "fallback_class": fallback_class,
            "fallback_map": {str(k): int(v) for k, v in sorted(fallback_map.items())},
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


