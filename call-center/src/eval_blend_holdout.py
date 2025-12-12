"""
Evaluate LinearSVC vs Transformer vs blended ensemble on a deterministic holdout split.

Why this exists
---------------
You asked "based on what statistics can we expect to pass 0.98541?"
We cannot guarantee Kaggle (test labels are unknown), but we *can* measure:
- whether transformer predictions are complementary to the strong LinearSVC artifacts
- whether blending improves Macro-F1 on a held-out split (same kind of signal as Kaggle)

This script prints:
- macro F1 (raw and time-gated) for: linear-only, transformer-only, blend weights
- per-class F1 for the best blend
- disagreement rate between linear-only and transformer-only (higher => more complementarity)

IMPORTANT
---------
Holdout is noisy (especially for minority classes). Use it as guidance, not a guarantee.
The only real metric is Kaggle, but this gives the best possible pre-submit signal.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

import predict_blend
from config import default_config, guess_repo_root, guess_team_dir, resolve_callcenter_output_dir


def _parse_float_list(s: str) -> list[float]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    out: list[float] = []
    for p in parts:
        try:
            out.append(float(p))
        except Exception:
            continue
    # de-dup preserving order
    seen: set[float] = set()
    dedup: list[float] = []
    for v in out:
        if v in seen:
            continue
        seen.add(v)
        dedup.append(v)
    return dedup


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--team_dir", type=str, default=None)
    p.add_argument("--clean_dir", type=str, default=None)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_size", type=float, default=0.1)

    p.add_argument(
        "--linear_ensemble_artifacts_dirs",
        type=str,
        required=True,
        help="Comma-separated LinearSVC artifacts dirs (the ones that produced your best Kaggle score).",
    )
    p.add_argument("--linear_use_fold_ensemble", action="store_true")
    p.add_argument("--linear_fold_models_dirname", type=str, default="fold_models")

    p.add_argument("--transformer_artifacts_dir", type=str, required=True)
    p.add_argument("--transformer_batch_size", type=int, default=32)

    p.add_argument("--blend_method", type=str, default="softmax_avg", choices=["softmax_avg", "logit_sum"])
    p.add_argument("--linear_weight", type=float, default=1.0)
    p.add_argument(
        "--transformer_weights",
        type=str,
        default="0.0,0.10,0.15,0.20,0.25,0.35,0.50",
        help="Comma-separated transformer weights to try (linear weight is fixed by --linear_weight).",
    )

    p.add_argument("--time_gate", action="store_true")
    p.add_argument("--time_gate_classes", type=str, default="0,3,4,5")
    p.add_argument("--time_gate_fallback_class", type=int, default=2)
    p.add_argument("--time_gate_fallback_map", type=str, default="5:1")
    return p.parse_args()


def _predict_from_scores(
    *,
    lin_scores: np.ndarray,
    tr_logits: np.ndarray,
    classes: list[int],
    blend_method: str,
    lw: float,
    tw: float,
) -> np.ndarray:
    if lw < 0 or tw < 0 or (lw + tw) <= 0:
        raise ValueError("Weights must be non-negative and sum > 0.")
    if blend_method == "softmax_avg":
        p_lin = predict_blend._softmax(lin_scores)
        p_tr = predict_blend._softmax(tr_logits)
        p = (lw * p_lin + tw * p_tr) / (lw + tw)
        return np.asarray(classes, dtype=int)[p.argmax(axis=1)]
    scores = lw * lin_scores + tw * tr_logits
    return np.asarray(classes, dtype=int)[scores.argmax(axis=1)]


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

    y = pd.to_numeric(df[cfg.target_col], errors="raise").astype(int).to_numpy()

    idx = np.arange(len(df))
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=float(args.val_size),
        random_state=int(args.seed),
        shuffle=True,
        stratify=y,
    )
    df_va = df.iloc[va_idx].reset_index(drop=True)
    y_va = y[va_idx]
    labels = sorted(np.unique(y).tolist())

    # Linear scores on val
    linear_dirs = predict_blend._parse_path_list(args.linear_ensemble_artifacts_dirs)
    if not linear_dirs:
        raise ValueError("--linear_ensemble_artifacts_dirs is empty.")
    linear_spec_path = linear_dirs[0] / "feature_spec.json"
    if not linear_spec_path.exists():
        raise FileNotFoundError(f"Linear feature_spec.json not found: {linear_spec_path}")
    linear_spec = predict_blend._load_json(linear_spec_path)

    lin_scores, lin_classes = predict_blend._sum_linear_scores(
        df_test=df_va,
        cfg=cfg,
        linear_feature_spec=linear_spec,
        linear_artifacts_dirs=linear_dirs,
        use_fold_ensemble=bool(args.linear_use_fold_ensemble),
        fold_models_dirname=str(args.linear_fold_models_dirname),
    )

    # Transformer logits on val
    tr_logits, tr_labels = predict_blend._transformer_logits(
        df_test=df_va,
        cfg=cfg,
        transformer_artifacts_dir=Path(args.transformer_artifacts_dir),
        batch_size=int(args.transformer_batch_size),
    )

    # Align transformer columns to linear class order
    if tr_labels != lin_classes:
        pos = {int(c): i for i, c in enumerate(tr_labels)}
        tr_logits = tr_logits[:, [pos[int(c)] for c in lin_classes]]

    # Base predictions
    pred_lin = _predict_from_scores(lin_scores=lin_scores, tr_logits=tr_logits, classes=lin_classes, blend_method=args.blend_method, lw=1.0, tw=0.0)
    pred_tr = _predict_from_scores(lin_scores=lin_scores, tr_logits=tr_logits, classes=lin_classes, blend_method=args.blend_method, lw=0.0, tw=1.0)

    disagree = float((pred_lin != pred_tr).mean())
    print(f"[eval_blend_holdout] val_size={float(args.val_size):.3f}  seed={int(args.seed)}  n_val={len(df_va)}")
    print(f"[eval_blend_holdout] linear_vs_transformer_disagreement={disagree:.4f}")

    # Optional time-gating
    time_windows = linear_spec.get("time_windows_by_class") if isinstance(linear_spec, dict) else None
    gate_classes = predict_blend._parse_int_list(args.time_gate_classes)
    fallback_map = predict_blend._parse_fallback_map(args.time_gate_fallback_map)

    def score_row(name: str, pred: np.ndarray) -> dict[str, Any]:
        raw = float(f1_score(y_va, pred, average="macro", labels=labels, zero_division=0))
        out = {"name": name, "macro_f1": raw}
        if bool(args.time_gate):
            if not isinstance(time_windows, dict) or not time_windows:
                raise ValueError("Missing time_windows_by_class in linear feature_spec.json.")
            pred_g, changed = predict_blend._apply_time_gating(
                pred,
                handle_time=df_va[cfg.datetime_col],
                time_windows_by_class=time_windows,
                gate_classes=gate_classes,
                fallback_class=int(args.time_gate_fallback_class),
                fallback_map=fallback_map,
            )
            gated = float(f1_score(y_va, pred_g, average="macro", labels=labels, zero_division=0))
            out["macro_f1_time_gated"] = gated
            out["time_gated_changed"] = int(changed)
        return out

    rows: list[dict[str, Any]] = []
    rows.append(score_row("linear_only", pred_lin))
    rows.append(score_row("transformer_only", pred_tr))

    lw = float(args.linear_weight)
    best: dict[str, Any] | None = None
    best_pred: np.ndarray | None = None

    for tw in _parse_float_list(args.transformer_weights):
        pred = _predict_from_scores(
            lin_scores=lin_scores,
            tr_logits=tr_logits,
            classes=lin_classes,
            blend_method=str(args.blend_method),
            lw=lw,
            tw=float(tw),
        )
        r = score_row(f"blend_lw{lw:g}_tw{tw:g}", pred)
        rows.append(r)
        key = "macro_f1_time_gated" if bool(args.time_gate) else "macro_f1"
        if best is None or float(r.get(key, -1.0)) > float(best.get(key, -1.0)):
            best = r
            best_pred = pred

    # Print summary table
    key = "macro_f1_time_gated" if bool(args.time_gate) else "macro_f1"
    rows_sorted = sorted(rows, key=lambda d: float(d.get(key, -1.0)), reverse=True)
    print("\n[eval_blend_holdout] Results (sorted):")
    for r in rows_sorted:
        if bool(args.time_gate):
            print(
                f"- {r['name']}: macro_f1={r['macro_f1']:.6f}  macro_f1_gated={r['macro_f1_time_gated']:.6f}  changed={r['time_gated_changed']}"
            )
        else:
            print(f"- {r['name']}: macro_f1={r['macro_f1']:.6f}")

    # Per-class report for best
    if best is not None and best_pred is not None:
        print(f"\n[eval_blend_holdout] Best by '{key}': {best['name']} -> {float(best.get(key)):.6f}")
        rep = classification_report(y_va, best_pred, labels=labels, output_dict=True, zero_division=0)
        per_class = {str(k): float(v['f1-score']) for k, v in rep.items() if str(k).isdigit()}
        print("[eval_blend_holdout] Per-class F1 (best):", per_class)


if __name__ == "__main__":
    main()


