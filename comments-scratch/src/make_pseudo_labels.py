"""
Create pseudo-labels for the Comments challenge from saved model probabilities (scratch-local).

Why
----
When labeled train is small (~1.8k) and unlabeled test is large (~30k), pseudo-labeling
is one of the best ROI "augmentation" techniques:
  1) Train a strong teacher (your DziriBERT ensemble)
  2) Predict probabilities on test
  3) Keep only high-confidence predictions (optionally: also require margin top1-top2)
  4) Retrain models using (train + pseudo) with lower weight for pseudo samples

Input
-----
- test_file.csv (from local scratch data dir unless overridden)
- proba.npy from a predictor (shape: [n_test, n_classes])
- label mapping from a transformer run dir (label_mapping.json) or a fold dir (config.json fallback)

Output
------
CSV with canonical columns:
  id, platform, comment, label, confidence, margin
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import schema
from config import CommentsConfig, guess_local_data_dir, guess_team_dir, resolve_comments_data_dir


def _load_id2label_from_config(model_dir: Path) -> dict[int, int]:
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.json in: {model_dir}")
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    id2label_raw = data.get("id2label") or {}
    if not id2label_raw:
        raise ValueError(f"config.json has no id2label in: {model_dir}")
    return {int(k): int(v) for k, v in id2label_raw.items()}


def _load_id2label_from_run_or_fold(path: Path) -> dict[int, int]:
    """
    Prefer: <run_dir>/label_mapping.json
    Fallback: <fold_dir>/config.json
    """

    if not path.is_dir():
        raise ValueError(f"--mapping_dir must be a directory: {path}")
    lm = path / "label_mapping.json"
    if lm.exists():
        data = json.loads(lm.read_text(encoding="utf-8"))
        return {int(k): int(v) for k, v in data["id2label"].items()}
    return _load_id2label_from_config(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--team_dir", type=str, default=None)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--test_filename", type=str, default="test_file.csv")

    p.add_argument("--proba_npy", type=str, required=True, help="Path to probabilities .npy (n_test, n_classes).")
    p.add_argument("--mapping_dir", type=str, required=True, help="Transformer run dir (preferred) or fold dir (fallback).")

    p.add_argument("--min_conf", type=float, default=0.98, help="Keep only rows with max prob >= min_conf.")
    p.add_argument("--min_margin", type=float, default=0.0, help="Optional: require (top1 - top2) >= min_margin.")
    p.add_argument("--per_class_max", type=int, default=1500, help="Max pseudo rows per class (0 = unlimited).")
    p.add_argument("--output_csv", type=str, required=True)
    args = p.parse_args()

    cfg = CommentsConfig()

    # Resolve data dir
    if args.data_dir:
        data_dir = Path(args.data_dir)
    elif args.team_dir:
        data_dir = resolve_comments_data_dir(Path(args.team_dir))
    else:
        team = guess_team_dir()
        data_dir = resolve_comments_data_dir(team) if team else guess_local_data_dir()

    test_path = data_dir / args.test_filename
    schema.validate_file_exists(test_path)
    df_raw = schema.read_csv_robust(test_path)
    df = schema.rename_raw_columns(df_raw, cfg, is_train=False)

    proba = np.load(args.proba_npy)
    if proba.ndim != 2:
        raise ValueError(f"Expected 2D proba array, got shape={proba.shape}")
    if len(df) != proba.shape[0]:
        raise ValueError(f"Row mismatch: test rows={len(df)} vs proba rows={proba.shape[0]}")

    id2label = _load_id2label_from_run_or_fold(Path(args.mapping_dir))
    num_labels = len(id2label)
    if proba.shape[1] != num_labels:
        raise ValueError(f"Class mismatch: proba has {proba.shape[1]} cols but mapping has {num_labels} labels")

    # Confidence and margin
    proba = np.asarray(proba, dtype=np.float32)
    pred_id = proba.argmax(axis=1).astype(int)
    conf = proba.max(axis=1).astype(np.float32)
    # margin = top1 - top2
    if proba.shape[1] >= 2:
        part = np.partition(proba, -2, axis=1)
        top2 = part[:, -2].astype(np.float32)
        margin = (conf - top2).astype(np.float32)
    else:
        margin = np.zeros((len(df),), dtype=np.float32)

    pred_label = np.array([id2label[int(i)] for i in pred_id], dtype=int)

    out = pd.DataFrame(
        {
            cfg.id_col: df[cfg.id_col].astype(int),
            cfg.platform_col: df[cfg.platform_col].astype("string"),
            cfg.text_col: df[cfg.text_col].astype("string"),
            cfg.target_col: pred_label,
            "confidence": conf,
            "margin": margin,
        }
    )

    out = out[out["confidence"] >= float(args.min_conf)].copy()
    if float(args.min_margin) > 0:
        out = out[out["margin"] >= float(args.min_margin)].copy()
    if len(out) == 0:
        raise ValueError("No pseudo-labels selected. Lower --min_conf or --min_margin.")

    if int(args.per_class_max) > 0:
        keep_parts: list[pd.DataFrame] = []
        for lbl, sub in out.groupby(cfg.target_col, sort=True):
            sub2 = sub.sort_values(["confidence", "margin"], ascending=False).head(int(args.per_class_max))
            keep_parts.append(sub2)
        out = pd.concat(keep_parts, axis=0, ignore_index=True)

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False, encoding="utf-8")

    counts = out[cfg.target_col].value_counts().sort_index().to_dict()
    print(f"Saved pseudo labels -> {out_csv} (rows={len(out)})")
    print("Selected per-class counts:", counts)
    print("Confidence stats:", out["confidence"].describe().to_dict())
    print("Margin stats:", out["margin"].describe().to_dict())


if __name__ == "__main__":
    main()


