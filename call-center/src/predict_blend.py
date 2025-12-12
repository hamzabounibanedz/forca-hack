"""
Blend (ensemble) a LinearSVC model ensemble with a Transformer model (score-level),
then optionally apply time-window gating, and write a Kaggle submission.

This is the recommended next step once LinearSVC + time-gating has plateaued:
- LinearSVC captures strong character-level patterns + structured features
- Transformer captures semantic/dialectal patterns (Darija / French mix)
Blending them is usually the easiest way to cross a leaderboard gap.

Blend method (deterministic)
----------------------------
We compute per-class scores from:
- LinearSVC: decision_function scores (summed across fold models and/or runs)
- Transformer: logits (model outputs)

Then we combine scores using either:
- softmax_avg: softmax(scores) per model, then weighted average of probabilities
- logit_sum: weighted sum of raw scores (requires careful weighting due to scale)

Finally, we take argmax and (optionally) apply time-window gating with the
train-derived windows saved in the *LinearSVC* feature_spec.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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


def _parse_extra_text_cols(v: Any) -> list[str]:
    """
    `train_transformer.py` saves extra_text_cols as a JSON list.
    Older/other scripts may pass a comma-separated string.
    """
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        out = [str(x).strip() for x in v]
        return [x for x in out if x]
    if isinstance(v, str):
        return [c.strip() for c in v.split(",") if c.strip()]
    s = str(v).strip()
    return [s] if s else []


def _softmax(x: np.ndarray) -> np.ndarray:
    # stable softmax over last axis
    x = x.astype("float64", copy=False)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


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


def _prepare_linear_X(df_test: pd.DataFrame, *, cfg, linear_feature_spec: dict[str, Any]) -> pd.DataFrame:
    return features.prepare_X_from_spec(df_test, cfg, linear_feature_spec, strict=True)[0]


def _load_linear_models_from_dir(artifacts_dir: Path, *, use_fold_ensemble: bool, fold_models_dirname: str) -> list[Any]:
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

    mp = artifacts_dir / "model.joblib"
    if not mp.exists():
        raise FileNotFoundError(f"Model file not found: {mp}")
    return [joblib.load(mp)]


def _sum_linear_scores(
    *,
    df_test: pd.DataFrame,
    cfg,
    linear_feature_spec: dict[str, Any],
    linear_artifacts_dirs: list[Path],
    use_fold_ensemble: bool,
    fold_models_dirname: str,
) -> tuple[np.ndarray, list[int]]:
    X = _prepare_linear_X(df_test, cfg=cfg, linear_feature_spec=linear_feature_spec)

    models: list[Any] = []
    for d in linear_artifacts_dirs:
        models.extend(_load_linear_models_from_dir(d, use_fold_ensemble=use_fold_ensemble, fold_models_dirname=fold_models_dirname))
    if not models:
        raise RuntimeError("No linear models loaded.")

    scores_sum: np.ndarray | None = None
    classes_ref: list[int] | None = None

    for m in models:
        classes = _get_model_classes(m)
        s = _decision_scores(m, X)
        if scores_sum is None:
            scores_sum = s
            classes_ref = classes
        else:
            if classes_ref is None:
                if classes is not None:
                    raise ValueError("Cannot align linear classes: reference unknown but a model provides classes_.")
                scores_sum += s
            else:
                s2 = _align_scores_to_ref(s, classes=classes, classes_ref=classes_ref)
                scores_sum += s2

    if scores_sum is None or classes_ref is None:
        raise RuntimeError("Failed to compute linear scores with known class ordering.")
    return scores_sum, classes_ref


def _build_transformer_text(df: pd.DataFrame, *, text_col: str, extra_cols: list[str]) -> list[str]:
    s = df[text_col].astype("string").fillna("")
    if not extra_cols:
        return s.astype(str).tolist()
    parts = []
    for c in extra_cols:
        if c not in df.columns:
            continue
        parts.append(c + ":" + df[c].astype("string").fillna("UNK"))
    if parts:
        prefix = parts[0]
        for p in parts[1:]:
            prefix = prefix + " | " + p
        s = prefix + " || " + s
    return s.astype(str).tolist()


class _TextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_length: int):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self) -> int:  # pragma: no cover
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


def _transformer_logits(
    *,
    df_test: pd.DataFrame,
    cfg,
    transformer_artifacts_dir: Path,
    batch_size: int,
) -> tuple[np.ndarray, list[int]]:
    spec = _load_json(transformer_artifacts_dir / "feature_spec.json")
    tok_dir = transformer_artifacts_dir / "tokenizer"
    model_dir = transformer_artifacts_dir / "model"
    if not tok_dir.exists() or not model_dir.exists():
        raise FileNotFoundError(f"Missing tokenizer/ or model/ in: {transformer_artifacts_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    labels = spec.get("labels")
    if not isinstance(labels, list) or not labels:
        # assume 0..num_labels-1
        labels = list(range(int(model.config.num_labels)))
    labels = [int(x) for x in labels]

    extra_cols = _parse_extra_text_cols(spec.get("extra_text_cols"))
    max_length = int(spec.get("max_length") or 128)

    texts = _build_transformer_text(df_test, text_col=cfg.text_col, extra_cols=extra_cols)
    ds = _TextDataset(texts, tokenizer, max_length=max_length)
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=False)

    all_logits: list[np.ndarray] = []
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            all_logits.append(out.logits.detach().cpu().numpy())

    logits = np.concatenate(all_logits, axis=0).astype("float64", copy=False)
    return logits, labels


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--team_dir", type=str, default=None)
    p.add_argument("--clean_dir", type=str, default=None)

    p.add_argument(
        "--linear_ensemble_artifacts_dirs",
        type=str,
        required=True,
        help="Comma-separated LinearSVC artifacts dirs to ensemble (each contains model.joblib or fold_models/).",
    )
    p.add_argument("--linear_use_fold_ensemble", action="store_true")
    p.add_argument("--linear_fold_models_dirname", type=str, default="fold_models")

    p.add_argument("--transformer_artifacts_dir", type=str, required=True)
    p.add_argument("--transformer_batch_size", type=int, default=32)

    p.add_argument("--blend_method", type=str, default="softmax_avg", choices=["softmax_avg", "logit_sum"])
    p.add_argument("--linear_weight", type=float, default=1.0)
    p.add_argument("--transformer_weight", type=float, default=1.0)

    p.add_argument("--time_gate", action="store_true")
    p.add_argument("--time_gate_classes", type=str, default="0,3,4,5")
    p.add_argument("--time_gate_fallback_class", type=int, default=2)
    p.add_argument("--time_gate_fallback_map", type=str, default="5:1")

    p.add_argument("--sample_submission_path", type=str, default=None)
    p.add_argument("--run_id", type=str, default=None)
    p.add_argument("--out_path", type=str, default=None)
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

    df_test = pd.read_csv(test_path)
    if cfg.id_col not in df_test.columns:
        raise ValueError(f"Missing id column '{cfg.id_col}' in cleaned test file.")

    # Linear spec comes from first linear artifacts dir
    linear_dirs = _parse_path_list(args.linear_ensemble_artifacts_dirs)
    if not linear_dirs:
        raise ValueError("--linear_ensemble_artifacts_dirs is empty.")
    for d in linear_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Linear artifacts dir does not exist: {d}")
    linear_spec_path = linear_dirs[0] / "feature_spec.json"
    if not linear_spec_path.exists():
        raise FileNotFoundError(f"Linear feature_spec.json not found: {linear_spec_path}")
    linear_spec = _load_json(linear_spec_path)

    # Compute scores
    lin_scores, lin_classes = _sum_linear_scores(
        df_test=df_test,
        cfg=cfg,
        linear_feature_spec=linear_spec,
        linear_artifacts_dirs=linear_dirs,
        use_fold_ensemble=bool(args.linear_use_fold_ensemble),
        fold_models_dirname=str(args.linear_fold_models_dirname),
    )

    tr_logits, tr_labels = _transformer_logits(
        df_test=df_test,
        cfg=cfg,
        transformer_artifacts_dir=Path(args.transformer_artifacts_dir),
        batch_size=int(args.transformer_batch_size),
    )

    # Align transformer columns to linear class order
    if len(tr_labels) != tr_logits.shape[1]:
        raise ValueError(f"Transformer labels/logits mismatch: labels={tr_labels} logits={tr_logits.shape}")
    if len(lin_classes) != lin_scores.shape[1]:
        raise ValueError(f"Linear classes/scores mismatch: classes={lin_classes} scores={lin_scores.shape}")

    if tr_labels != lin_classes:
        pos = {int(c): i for i, c in enumerate(tr_labels)}
        idx = [pos[int(c)] for c in lin_classes]
        tr_logits = tr_logits[:, idx]

    lw = float(args.linear_weight)
    tw = float(args.transformer_weight)
    if lw < 0 or tw < 0 or (lw + tw) <= 0:
        raise ValueError("Weights must be non-negative and sum > 0.")

    if str(args.blend_method) == "softmax_avg":
        p_lin = _softmax(lin_scores)
        p_tr = _softmax(tr_logits)
        p = (lw * p_lin + tw * p_tr) / (lw + tw)
        pred = np.asarray(lin_classes)[p.argmax(axis=1)]
    else:
        scores = lw * lin_scores + tw * tr_logits
        pred = np.asarray(lin_classes)[scores.argmax(axis=1)]

    pred = np.asarray(pred).reshape(-1).astype(int)

    # Optional time gating (use linear spec windows)
    if bool(args.time_gate):
        time_windows = linear_spec.get("time_windows_by_class")
        if not isinstance(time_windows, dict) or not time_windows:
            raise ValueError("Missing time_windows_by_class in linear feature_spec.json; retrain linear with updated scripts.")

        gate_classes = _parse_int_list(args.time_gate_classes)
        fallback_class = int(args.time_gate_fallback_class)
        fallback_map = _parse_fallback_map(args.time_gate_fallback_map)

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
            f"[predict_blend] Applied time gating: classes={gate_classes} -> fallback={fallback_class} map={fallback_map}  changed={changed}"
        )

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
        out_path = out_dir / f"callcenter_blend_{run_id}.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[predict_blend] Saved submission: {out_path}")
    print(out.head(3).to_string(index=False))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main()



