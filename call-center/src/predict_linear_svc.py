"""
Inference + Kaggle submission writer for the Call Center challenge using the LinearSVC TF-IDF model.

Supports:
- single model inference (model.joblib)
- fold-ensemble inference (fold_models/model_fold*.joblib)
- multi-run artifacts ensembling (average decision scores across multiple artifacts dirs)
- deterministic time-window gating postprocess (optional) using train-derived windows saved in feature_spec.json

Typical (Colab/Drive):
python /content/forca-hack/call-center/src/predict_linear_svc.py \
  --team_dir "/content/drive/MyDrive/FORSA_team" \
  --artifacts_dir "/content/drive/MyDrive/FORSA_team/outputs/callcenter/linear_svc_runs/<run_id>" \
  --out_path "/content/drive/MyDrive/FORSA_team/submissions/callcenter/sub_<run_id>.csv" \
  --use_fold_ensemble \
  --time_gate
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


def _parse_int_list(s: str) -> list[int]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    out: list[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except Exception:
            continue
    # de-dup preserving order
    seen: set[int] = set()
    dedup: list[int] = []
    for v in out:
        if v in seen:
            continue
        seen.add(v)
        dedup.append(v)
    return dedup


def _parse_fallback_map(s: str | None) -> dict[int, int]:
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


def _parse_path_list(s: str | None) -> list[Path]:
    if not s:
        return []
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    out = [Path(p) for p in parts]
    # de-dup preserving order
    seen: set[str] = set()
    dedup: list[Path] = []
    for p in out:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(p)
    return dedup


def _parse_float_list(s: str | None) -> list[float]:
    if not s:
        return []
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    out: list[float] = []
    for p in parts:
        try:
            out.append(float(p))
        except Exception:
            raise ValueError(f"Invalid float in list: '{p}'")
    return out


def _get_model_classes(model: Any) -> list[int] | None:
    clf = getattr(model, "named_steps", {}).get("clf") if hasattr(model, "named_steps") else None
    classes = getattr(clf, "classes_", None) if clf is not None else None
    if classes is None:
        return None
    try:
        return [int(x) for x in list(classes)]
    except Exception:
        return None


def _decision_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    s = np.asarray(model.decision_function(X))
    if s.ndim == 1:
        s = s.reshape(-1, 1)
    return s.astype("float64", copy=False)


def _align_scores_to_ref(scores: np.ndarray, *, classes: list[int] | None, classes_ref: list[int]) -> np.ndarray:
    if classes is None:
        return scores
    if len(classes) != scores.shape[1]:
        raise ValueError(f"Scores/classes length mismatch: scores={scores.shape} classes={classes}")
    if classes == classes_ref:
        return scores
    pos = {c: i for i, c in enumerate(classes)}
    idx = [pos[c] for c in classes_ref]
    return scores[:, idx]


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


def _apply_time_gating(
    pred: np.ndarray,
    *,
    handle_time: pd.Series,
    time_windows_by_class: dict[str, Any],
    gate_classes: list[int],
    fallback_class: int,
    fallback_map: dict[int, int] | None = None,
) -> tuple[np.ndarray, int]:
    if not len(pred) or not time_windows_by_class or not gate_classes:
        return pred, 0

    dt = pd.to_datetime(handle_time, errors="coerce")
    out = np.asarray(pred).copy().astype(int)
    changed = 0
    fb_map = fallback_map or {}

    for cls in gate_classes:
        win = time_windows_by_class.get(str(int(cls)))
        if not isinstance(win, dict):
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


def _load_models_from_artifacts_dir(
    *,
    artifacts_dir: Path,
    use_fold_ensemble: bool,
    fold_models_dirname: str,
) -> list[Any]:
    # joblib import (only when needed)
    try:
        import joblib  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("joblib is required (usually installed with scikit-learn).") from e

    if use_fold_ensemble:
        fold_dir = artifacts_dir / str(fold_models_dirname)
        paths = sorted(fold_dir.glob("model_fold*.joblib"))
        if not paths:
            raise FileNotFoundError(f"No fold models found in: {fold_dir}")
        return [joblib.load(p) for p in paths]

    model_path = artifacts_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return [joblib.load(model_path)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--team_dir", type=str, default=None, help="Drive root, e.g. /content/drive/MyDrive/FORSA_team")
    p.add_argument("--clean_dir", type=str, default=None, help="Override cleaned data folder (contains test_clean.csv).")

    # Single-run artifacts
    p.add_argument("--artifacts_dir", type=str, default=None, help="Artifacts folder (contains model.joblib + feature_spec.json).")
    p.add_argument("--model_path", type=str, default=None, help="Override model path (model.joblib). (single-run only)")
    p.add_argument("--feature_spec_path", type=str, default=None, help="Override feature_spec.json path. (single-run only)")

    # Multi-run artifacts ensemble
    p.add_argument(
        "--ensemble_artifacts_dirs",
        type=str,
        default=None,
        help="Comma-separated list of artifacts dirs to ensemble (average decision scores). Overrides --artifacts_dir.",
    )
    p.add_argument(
        "--ensemble_weights",
        type=str,
        default=None,
        help=(
            "Optional comma-separated weights matching --ensemble_artifacts_dirs. "
            "Weights are applied per artifacts dir (distributed equally across its fold models). "
            "Example: '1,1,0.5'."
        ),
    )

    p.add_argument("--sample_submission_path", type=str, default=None, help="Override sample_submission.csv path.")
    p.add_argument("--run_id", type=str, default=None, help="Run id for submission filename.")
    p.add_argument("--out_path", type=str, default=None, help="Explicit submission output path.")

    p.add_argument(
        "--use_fold_ensemble",
        action="store_true",
        help="If set, load <artifacts_dir>/fold_models/model_fold*.joblib and average decision scores.",
    )
    p.add_argument(
        "--fold_models_dirname",
        type=str,
        default="fold_models",
        help="Subfolder name under artifacts_dir used to store fold models.",
    )

    # Optional deterministic post-processing (time-window gating)
    p.add_argument("--time_gate", action="store_true", help="If set, apply time-window gating using feature_spec.json.")
    p.add_argument(
        "--time_gate_classes",
        type=str,
        default=None,
        help="Comma-separated class labels to gate (overrides feature_spec defaults). Example: '0,3,4,5'.",
    )
    p.add_argument(
        "--time_gate_fallback_class",
        type=int,
        default=None,
        help="Fallback class when a prediction violates its time window (overrides feature_spec defaults).",
    )
    p.add_argument(
        "--time_gate_fallback_map",
        type=str,
        default=None,
        help="Optional per-class fallback overrides 'pred:fb,pred:fb'. Example: '5:1'.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = default_config()

    team_dir = Path(args.team_dir) if args.team_dir else guess_team_dir()
    if team_dir is not None and not team_dir.exists():
        raise FileNotFoundError(f"--team_dir does not exist: {team_dir}")

    # Clean dir
    if args.clean_dir:
        clean_dir = Path(args.clean_dir)
    elif team_dir is not None:
        clean_dir = resolve_callcenter_output_dir(team_dir)
    else:
        clean_dir = guess_local_data_dir()

    test_path = clean_dir / cfg.clean_test_name
    if not test_path.exists():
        raise FileNotFoundError(f"Cleaned test file not found: {test_path}")

    # Sample submission
    if args.sample_submission_path:
        sample_path = Path(args.sample_submission_path)
    elif team_dir is not None:
        data_dir = resolve_callcenter_data_dir(team_dir)
        sample_path = data_dir / "sample_submission.csv"
    else:
        sample_path = guess_local_data_dir() / "sample_submission.csv"
    if not sample_path.exists():
        raise FileNotFoundError(f"sample_submission.csv not found: {sample_path}")

    # Artifacts dirs
    ensemble_dirs = _parse_path_list(args.ensemble_artifacts_dirs)
    ensemble_weights = _parse_float_list(args.ensemble_weights)
    if ensemble_dirs:
        for d in ensemble_dirs:
            if not d.exists():
                raise FileNotFoundError(f"Ensemble artifacts dir does not exist: {d}")
        if ensemble_weights and len(ensemble_weights) != len(ensemble_dirs):
            raise ValueError(
                f"--ensemble_weights length ({len(ensemble_weights)}) must match --ensemble_artifacts_dirs length ({len(ensemble_dirs)})."
            )
        spec_source_dir = ensemble_dirs[0]
    else:
        if args.artifacts_dir:
            spec_source_dir = Path(args.artifacts_dir)
        elif team_dir is not None:
            spec_source_dir = team_dir / "outputs" / "callcenter" / "linear_svc"
        else:
            spec_source_dir = guess_repo_root() / "call-center" / "artifacts" / "linear_svc"

    # Load feature spec
    if ensemble_dirs:
        feature_spec_path = spec_source_dir / "feature_spec.json"
    else:
        feature_spec_path = Path(args.feature_spec_path) if args.feature_spec_path else (spec_source_dir / "feature_spec.json")
    if not feature_spec_path.exists():
        raise FileNotFoundError(f"feature_spec.json not found: {feature_spec_path}")
    feature_spec = _load_json(feature_spec_path)

    print(f"[predict_linear_svc] test_path={test_path}")
    if ensemble_dirs:
        print(f"[predict_linear_svc] ensemble_artifacts_dirs={','.join([str(p) for p in ensemble_dirs])}")
    else:
        print(f"[predict_linear_svc] artifacts_dir={spec_source_dir}")
    print(f"[predict_linear_svc] feature_spec_path={feature_spec_path}")
    print(f"[predict_linear_svc] sample_submission_path={sample_path}")

    # Load models
    models: list[Any] = []
    model_weights: list[float] = []
    if ensemble_dirs:
        for idx, d in enumerate(ensemble_dirs):
            w_dir = float(ensemble_weights[idx]) if ensemble_weights else 1.0
            ms = _load_models_from_artifacts_dir(
                artifacts_dir=d,
                use_fold_ensemble=bool(args.use_fold_ensemble),
                fold_models_dirname=str(args.fold_models_dirname),
            )
            if not ms:
                raise RuntimeError(f"No models loaded from: {d}")
            # Distribute dir weight across its models so each run contributes w_dir total.
            w_each = w_dir / float(len(ms))
            models.extend(ms)
            model_weights.extend([w_each] * len(ms))
        print(f"[predict_linear_svc] Using artifacts ensemble: n_models={len(models)}")
    else:
        if args.model_path and not args.use_fold_ensemble:
            # Load explicit model path
            try:
                import joblib  # type: ignore
            except Exception as e:  # pragma: no cover
                raise RuntimeError("joblib is required (usually installed with scikit-learn).") from e
            mp = Path(args.model_path)
            if not mp.exists():
                raise FileNotFoundError(f"Model file not found: {mp}")
            models = [joblib.load(mp)]
            model_weights = [1.0]
        else:
            models = _load_models_from_artifacts_dir(
                artifacts_dir=spec_source_dir,
                use_fold_ensemble=bool(args.use_fold_ensemble),
                fold_models_dirname=str(args.fold_models_dirname),
            )
            model_weights = [1.0] * len(models)

        if args.use_fold_ensemble:
            print(f"[predict_linear_svc] Using fold ensemble: n_models={len(models)}")

    if not models:
        raise RuntimeError("No models loaded for inference.")

    # Load test + prepare X
    df_test = pd.read_csv(test_path)
    if cfg.id_col not in df_test.columns:
        raise ValueError(f"Missing id column '{cfg.id_col}' in cleaned test file.")
    X = _prepare_frame(df_test, cfg=cfg, spec=feature_spec)

    # Sum decision scores across all models
    scores_sum: np.ndarray | None = None
    classes_ref: list[int] | None = None
    total_w = float(np.sum(model_weights)) if model_weights else 0.0
    if total_w <= 0:
        raise RuntimeError("Invalid ensemble weights: sum(weights) must be > 0.")

    for m, w in zip(models, model_weights, strict=True):
        classes = _get_model_classes(m)
        s = _decision_scores(m, X)
        if scores_sum is None:
            scores_sum = float(w) * s
            classes_ref = classes
        else:
            if classes_ref is None:
                if classes is not None:
                    raise ValueError("Cannot align model classes: reference classes unknown but a model provides classes_.")
                scores_sum += float(w) * s
            else:
                s2 = _align_scores_to_ref(s, classes=classes, classes_ref=classes_ref)
                scores_sum += float(w) * s2
    if scores_sum is None:
        raise RuntimeError("Ensemble produced no scores.")
    if classes_ref is None:
        # Fallback (should not happen for LinearSVC)
        pred = scores_sum.argmax(axis=1)
    else:
        pred = np.asarray(classes_ref)[scores_sum.argmax(axis=1)]
    pred = np.asarray(pred).reshape(-1).astype(int)

    # Optional time gating
    if bool(getattr(args, "time_gate", False)):
        if cfg.datetime_col not in df_test.columns:
            raise ValueError(
                f"--time_gate requires '{cfg.datetime_col}' column in cleaned test file; missing in: {test_path}"
            )
        time_windows = feature_spec.get("time_windows_by_class")
        if not isinstance(time_windows, dict) or not time_windows:
            raise ValueError(
                "--time_gate requires 'time_windows_by_class' in feature_spec.json. "
                "Tip: retrain using train_linear_svc.py that saves time windows."
            )

        defaults = feature_spec.get("time_gating_defaults") if isinstance(feature_spec, dict) else None
        if isinstance(defaults, dict):
            spec_gate_classes = defaults.get("gate_classes")
            spec_fallback = defaults.get("fallback_class")
            spec_fallback_map = defaults.get("fallback_map")
        else:
            spec_gate_classes = None
            spec_fallback = None
            spec_fallback_map = None

        if args.time_gate_classes:
            gate_classes = _parse_int_list(args.time_gate_classes)
        elif isinstance(spec_gate_classes, list) and spec_gate_classes:
            gate_classes = [int(x) for x in spec_gate_classes]
        else:
            gate_classes = [0, 3, 4, 5]

        if args.time_gate_fallback_class is not None:
            fallback_class = int(args.time_gate_fallback_class)
        elif spec_fallback is not None:
            fallback_class = int(spec_fallback)
        else:
            fallback_class = 2

        if args.time_gate_fallback_map is not None:
            fallback_map = _parse_fallback_map(args.time_gate_fallback_map)
        elif isinstance(spec_fallback_map, dict) and spec_fallback_map:
            fallback_map = {int(k): int(v) for k, v in spec_fallback_map.items()}
        else:
            fallback_map = {}

        pred2, changed = _apply_time_gating(
            pred,
            handle_time=df_test[cfg.datetime_col],
            time_windows_by_class=time_windows,
            gate_classes=gate_classes,
            fallback_class=fallback_class,
            fallback_map=fallback_map,
        )
        pred = pred2
        print(
            f"[predict_linear_svc] Applied time gating: classes={gate_classes} -> fallback={fallback_class} map={fallback_map}  changed={changed}"
        )

    # Build Kaggle submission
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


