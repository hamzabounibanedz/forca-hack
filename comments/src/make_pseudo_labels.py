"""
Create pseudo-labels for the Comments challenge from saved model probabilities.

Why
----
When the labeled train set is small but the unlabeled test set is large, Kaggle scores
can drop vs CV due to distribution shift. Pseudo-labeling is a standard transductive
technique:
  1) Train a reasonably strong model on labeled train
  2) Predict probabilities on unlabeled test
  3) Keep only high-confidence predictions (optionally balanced per class)
  4) Retrain using (train + pseudo)

This script implements step (3).

Input
-----
- test_file.csv (must contain id, platform, comment)
- proba.npy from `predict.py --save_proba_npy` (shape: [n_test, n_classes])
- label mapping from a transformer run dir (label_mapping.json) or a fold dir (config.json fallback)

Output
------
A CSV with canonical columns:
  id, platform, comment, label, confidence
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


def _load_label_mapping_from_config(model_dir: Path) -> dict[int, int]:
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.json in: {model_dir}")
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    id2label_raw = data.get("id2label") or {}
    if not id2label_raw:
        raise ValueError(f"config.json has no id2label in: {model_dir}")
    id2label: dict[int, int] = {}
    for k, v in id2label_raw.items():
        id2label[int(k)] = int(v)
    return id2label


def _load_id2label_from_run_or_fold(path: Path) -> dict[int, int]:
    """
    Prefer: <run_dir>/label_mapping.json
    Fallback: <fold_dir>/config.json
    """
    if path.is_dir():
        lm = path / "label_mapping.json"
        if lm.exists():
            data = json.loads(lm.read_text(encoding="utf-8"))
            return {int(k): int(v) for k, v in data["id2label"].items()}
        # fallback: try config.json (fold dir)
        return _load_label_mapping_from_config(path)
    raise ValueError(f"--mapping_dir must be a directory: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--test_filename", type=str, default="test_file.csv")
    parser.add_argument("--proba_npy", type=str, required=True, help="Path to probabilities .npy from predict.py")
    parser.add_argument(
        "--mapping_dir",
        type=str,
        required=True,
        help="Transformer run dir (preferred) or a fold dir (fallback) to read label mapping.",
    )
    parser.add_argument("--min_conf", type=float, default=0.98, help="Keep only rows with max prob >= min_conf")
    parser.add_argument(
        "--per_class_max",
        type=int,
        default=1500,
        help="Max pseudo-labeled rows to keep per class (balanced). Use 0 for unlimited.",
    )
    parser.add_argument("--output_csv", type=str, required=True)
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

    pred_id = proba.argmax(axis=1).astype(int)
    conf = proba.max(axis=1).astype(float)
    pred_label = np.array([id2label[int(i)] for i in pred_id], dtype=int)

    out = pd.DataFrame(
        {
            cfg.id_col: df[cfg.id_col].astype(int),
            cfg.platform_col: df[cfg.platform_col].astype("string"),
            cfg.text_col: df[cfg.text_col].astype("string"),
            cfg.target_col: pred_label,
            "confidence": conf,
        }
    )

    out = out[out["confidence"] >= float(args.min_conf)].copy()
    if len(out) == 0:
        raise ValueError("No pseudo-labels selected. Lower --min_conf.")

    if int(args.per_class_max) > 0:
        keep_parts: list[pd.DataFrame] = []
        for lbl, sub in out.groupby(cfg.target_col, sort=True):
            sub2 = sub.sort_values("confidence", ascending=False).head(int(args.per_class_max))
            keep_parts.append(sub2)
        out = pd.concat(keep_parts, axis=0, ignore_index=True)

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False, encoding="utf-8")

    # Summary
    counts = out[cfg.target_col].value_counts().sort_index().to_dict()
    print(f"Saved pseudo labels -> {out_csv} (rows={len(out)})")
    print("Selected per-class counts:", counts)
    print("Confidence stats:", out["confidence"].describe().to_dict())


if __name__ == "__main__":
    main()


