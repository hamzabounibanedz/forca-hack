"""
Deterministically optimize a weighted ensemble of existing LinearSVC runs using OOF scores.

Goal
----
Given multiple artifacts dirs (each created by `train_linear_svc.py` with `--save_fold_models`),
compute *true* out-of-fold (OOF) decision scores for each run, then greedily build a weighted
ensemble that maximizes OOF Macro-F1 (optionally with time-gating).

Why
---
Kaggle test labels are hidden, so we can't guarantee improvements. But optimizing weights on OOF
is the most evidence-based, non-random way to pick an ensemble without "guessing" or spamming submissions.

Notes
-----
- Requires fold models in each artifacts dir: <dir>/fold_models/model_fold{k}.joblib
- Reconstructs CV splits deterministically using (seed, n_folds) saved in <dir>/metrics.json
- Uses train row order sorted by `id` to match training-time splits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

import features
from config import default_config, guess_repo_root, guess_team_dir, resolve_callcenter_output_dir


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_path_list(s: str) -> list[Path]:
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


def _parse_int_list(s: str) -> list[int]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    out: list[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except Exception:
            continue
    # de-dup
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
        try:
            out[int(a.strip())] = int(b.strip())
        except Exception:
            continue
    return out


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
    fb_map = fallback_map or {}
    changed = 0
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
            out[mask] = int(fb_map.get(int(cls), fallback_class))
            changed += int(mask.sum())
    return out, changed


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


def _align_scores(scores: np.ndarray, *, classes: list[int] | None, classes_ref: list[int]) -> np.ndarray:
    if classes is None:
        return scores
    if classes == classes_ref:
        return scores
    pos = {c: i for i, c in enumerate(classes)}
    idx = [pos[c] for c in classes_ref]
    return scores[:, idx]


def _run_oof_scores(
    *,
    df: pd.DataFrame,
    y: np.ndarray,
    X: pd.DataFrame,
    artifacts_dir: Path,
    fold_models_dirname: str,
) -> tuple[np.ndarray, list[int], dict[str, Any]]:
    """
    Compute true OOF decision scores for ONE run directory using its saved fold models.
    """
    metrics_path = artifacts_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found: {metrics_path}")
    metrics = _load_json(metrics_path)
    n_splits = int(metrics.get("cv", {}).get("n_splits"))
    seed = int(metrics.get("cv", {}).get("seed"))

    fold_dir = artifacts_dir / fold_models_dirname
    if not fold_dir.exists():
        raise FileNotFoundError(f"Fold models dir not found: {fold_dir}")

    try:
        import joblib  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("joblib is required (installed with scikit-learn).") from e

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    classes_ref: list[int] | None = None
    scores_oof: np.ndarray | None = None

    # Need to know class count early; load fold1 model
    first_model_path = fold_dir / "model_fold1.joblib"
    if not first_model_path.exists():
        # fall back to any fold model
        paths = sorted(fold_dir.glob("model_fold*.joblib"))
        if not paths:
            raise FileNotFoundError(f"No fold models in: {fold_dir}")
        first_model_path = paths[0]
    m0 = joblib.load(first_model_path)
    c0 = _get_model_classes(m0)
    if c0 is None:
        raise RuntimeError(f"Failed to read classes_ from fold model: {first_model_path}")
    classes_ref = c0
    scores_oof = np.zeros((len(df), len(classes_ref)), dtype="float64")

    for fold, (_, va_idx) in enumerate(skf.split(X, y), start=1):
        model_path = fold_dir / f"model_fold{fold}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing fold model: {model_path}")
        model = joblib.load(model_path)
        classes = _get_model_classes(model)
        s = _decision_scores(model, X.iloc[va_idx])
        s = _align_scores(s, classes=classes, classes_ref=classes_ref)
        scores_oof[va_idx] = s

    return scores_oof, classes_ref, {"n_splits": n_splits, "seed": seed}


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> float:
    return float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--team_dir", type=str, default=None)
    p.add_argument("--clean_dir", type=str, default=None)
    p.add_argument("--artifacts_dirs", type=str, required=True, help="Comma-separated run directories.")
    p.add_argument("--fold_models_dirname", type=str, default="fold_models")

    p.add_argument("--time_gate", action="store_true")
    p.add_argument("--time_gate_classes", type=str, default="0,3,4,5")
    p.add_argument("--time_gate_fallback_class", type=int, default=2)
    p.add_argument("--time_gate_fallback_map", type=str, default="5:1")

    p.add_argument("--weight_step", type=float, default=0.05, help="Step size for alpha search in greedy mixing.")
    p.add_argument("--max_runs", type=int, default=6, help="Max number of runs to include in the greedy ensemble.")
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
        clean_dir = guess_repo_root() / "outputs" / "callcenter" / "clean"

    train_path = clean_dir / cfg.clean_train_name
    if not train_path.exists():
        raise FileNotFoundError(f"Cleaned train file not found: {train_path}")

    df = pd.read_csv(train_path)
    if cfg.target_col not in df.columns:
        raise ValueError(f"Missing target column '{cfg.target_col}' in cleaned train file.")
    if cfg.id_col in df.columns:
        df = df.sort_values(cfg.id_col, kind="mergesort").reset_index(drop=True)
    if cfg.datetime_col in df.columns:
        df[cfg.datetime_col] = pd.to_datetime(df[cfg.datetime_col], errors="coerce")

    y = pd.to_numeric(df[cfg.target_col], errors="raise").astype(int).to_numpy()
    labels = sorted(np.unique(y).tolist())

    # Use spec from the first run to build X (all runs should share the same feature spec contract)
    run_dirs = _parse_path_list(args.artifacts_dirs)
    if not run_dirs:
        raise ValueError("--artifacts_dirs is empty.")
    for d in run_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Artifacts dir not found: {d}")
    spec0_path = run_dirs[0] / "feature_spec.json"
    if not spec0_path.exists():
        raise FileNotFoundError(f"feature_spec.json not found: {spec0_path}")
    spec0 = _load_json(spec0_path)
    X, spec0n = features.prepare_X_from_spec(df, cfg, spec0, strict=True)

    # Time windows for gating (from first run spec if present, else compute from df)
    time_windows = spec0n.get("time_windows_by_class") if isinstance(spec0n, dict) else None
    if not isinstance(time_windows, dict) or not time_windows:
        time_windows = {}
        dt = df[cfg.datetime_col] if cfg.datetime_col in df.columns else pd.Series([pd.NaT] * len(df))
        for cls in labels:
            s = pd.to_datetime(dt, errors="coerce").loc[y == int(cls)]
            time_windows[str(int(cls))] = {"min": s.min(), "max": s.max()}

    gate_classes = _parse_int_list(args.time_gate_classes)
    fallback_map = _parse_fallback_map(args.time_gate_fallback_map)

    # Compute per-run OOF scores
    run_scores: list[np.ndarray] = []
    run_info: list[dict[str, Any]] = []
    classes_ref: list[int] | None = None

    for d in run_dirs:
        scores_oof, classes, meta = _run_oof_scores(
            df=df,
            y=y,
            X=X,
            artifacts_dir=d,
            fold_models_dirname=str(args.fold_models_dirname),
        )
        if classes_ref is None:
            classes_ref = classes
        else:
            if classes != classes_ref:
                # Align by column order
                pos = {int(c): i for i, c in enumerate(classes)}
                scores_oof = scores_oof[:, [pos[int(c)] for c in classes_ref]]
        run_scores.append(scores_oof)
        run_info.append({"dir": str(d), **meta})

    if classes_ref is None:
        raise RuntimeError("No runs loaded.")

    # Score each run
    print(f"[optimize_linear_ensemble] n_runs={len(run_scores)}  rows={len(df)}  labels={labels}")
    for i, s in enumerate(run_scores):
        pred = np.asarray(classes_ref)[s.argmax(axis=1)].astype(int)
        raw = _macro_f1(y, pred, labels=labels)
        if bool(args.time_gate):
            pred_g, _ = _apply_time_gating(
                pred,
                handle_time=df[cfg.datetime_col],
                time_windows_by_class=time_windows,
                gate_classes=gate_classes,
                fallback_class=int(args.time_gate_fallback_class),
                fallback_map=fallback_map,
            )
            gated = _macro_f1(y, pred_g, labels=labels)
            print(f"- run[{i}] raw={raw:.6f} gated={gated:.6f}  dir={run_info[i]['dir']}")
        else:
            print(f"- run[{i}] raw={raw:.6f}  dir={run_info[i]['dir']}")

    # Greedy mixing: start from best single run, then add runs if it improves OOF
    key_is_gated = bool(args.time_gate)

    def score_scores(scores: np.ndarray) -> float:
        pred = np.asarray(classes_ref)[scores.argmax(axis=1)].astype(int)
        if not key_is_gated:
            return _macro_f1(y, pred, labels=labels)
        pred_g, _ = _apply_time_gating(
            pred,
            handle_time=df[cfg.datetime_col],
            time_windows_by_class=time_windows,
            gate_classes=gate_classes,
            fallback_class=int(args.time_gate_fallback_class),
            fallback_map=fallback_map,
        )
        return _macro_f1(y, pred_g, labels=labels)

    scores_list = run_scores
    single_scores = [score_scores(s) for s in scores_list]
    best_i = int(np.argmax(single_scores))
    ens_scores = scores_list[best_i].copy()
    selected = [best_i]
    weights = {best_i: 1.0}
    best_score = float(single_scores[best_i])
    print(f"\n[optimize_linear_ensemble] start best_run={best_i} score={best_score:.6f}")

    step = float(args.weight_step)
    alphas = np.clip(np.arange(0.0, 1.0 + 1e-9, step), 0.0, 1.0)

    max_runs = int(args.max_runs)
    improved = True
    while improved and len(selected) < min(max_runs, len(scores_list)):
        improved = False
        best_candidate: dict[str, Any] | None = None
        for j in range(len(scores_list)):
            if j in selected:
                continue
            cand = scores_list[j]
            # mix current ensemble with candidate: new = a*ens + (1-a)*cand
            for a in alphas:
                mix = (a * ens_scores) + ((1.0 - a) * cand)
                sc = score_scores(mix)
                if best_candidate is None or sc > float(best_candidate["score"]):
                    best_candidate = {"j": j, "a": float(a), "score": float(sc)}
        if best_candidate is None:
            break
        if float(best_candidate["score"]) > best_score + 1e-9:
            j = int(best_candidate["j"])
            a = float(best_candidate["a"])
            ens_scores = (a * ens_scores) + ((1.0 - a) * scores_list[j])
            # update weights: scale existing by a, add new with (1-a)
            for k in list(weights.keys()):
                weights[k] *= a
            weights[j] = weights.get(j, 0.0) + (1.0 - a)
            selected.append(j)
            best_score = float(best_candidate["score"])
            improved = True
            print(f"[optimize_linear_ensemble] + add run={j}  alpha={a:.2f}  score={best_score:.6f}")

    # Normalize weights
    ssum = float(sum(weights.values()))
    weights_norm = {k: float(v / ssum) for k, v in sorted(weights.items(), key=lambda kv: -kv[1])}

    print("\n[optimize_linear_ensemble] FINAL")
    print("score_oof:", f"{best_score:.6f}", "(time_gated)" if key_is_gated else "(raw)")
    print("selected_runs:", [run_info[i]["dir"] for i in sorted(weights_norm.keys())])
    print("weights (dir -> w):")
    for i, w in weights_norm.items():
        print(f"- {run_info[i]['dir']}  w={w:.6f}")

    # Print CLI-ready strings for predict_linear_svc.py
    dirs_str = ",".join([run_info[i]["dir"] for i in weights_norm.keys()])
    w_str = ",".join([f"{weights_norm[i]:.6f}" for i in weights_norm.keys()])
    print("\n[paste_into_predict_linear_svc]")
    print("ensemble_artifacts_dirs=", dirs_str)
    print("ensemble_weights=", w_str)


if __name__ == "__main__":
    main()


