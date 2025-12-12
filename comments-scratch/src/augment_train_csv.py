"""
High-precision augmentation for the comments dataset by pseudo-labeling the 30k test set,
then appending selected rows to the labeled train CSV.

Key properties
--------------
- "Precise labeling": we ONLY keep pseudo-labeled rows that pass strict filters:
  - min confidence (top1 prob)
  - min margin (top1 - top2)
  - optional agreement across multiple probability files (multi-teacher agreement)
  - optional balanced per-class caps

- "Append to train.csv": we write an augmented train CSV containing:
  original train rows + selected pseudo rows (from test_file.csv)

Safety
------
By default we DO NOT overwrite your original train.csv.
Use --inplace to overwrite (a .bak copy will be created).

Usage (local)
-------------
1) First analyze drift (optional):
   python forca-hack/comments-scratch/src/analyze_data.py

2) Generate test probabilities with your best teacher (DziriBERT ensemble):
   python forca-hack/comments/src/predict.py --data_dir forca-hack/comments-scratch/data --run_dirs <RUN_DIR> --transformer_add_flags --save_proba_npy proba_test.npy --output_csv submission_tmp.csv

3) Append pseudo-labeled rows:
   python forca-hack/comments-scratch/src/augment_train_csv.py --proba_npys proba_test.npy --mapping_dir <RUN_DIR>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shutil

import preprocess
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


def _top1_top2(proba: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (pred_id, conf, margin) for each row.
    margin = top1 - top2
    """

    proba = np.asarray(proba, dtype=np.float32)
    pred_id = proba.argmax(axis=1).astype(np.int32)
    conf = proba.max(axis=1).astype(np.float32)
    if proba.shape[1] >= 2:
        part = np.partition(proba, -2, axis=1)
        top2 = part[:, -2].astype(np.float32)
        margin = (conf - top2).astype(np.float32)
    else:
        margin = np.zeros((proba.shape[0],), dtype=np.float32)
    return pred_id, conf, margin


def build_pseudo_df(
    *,
    df_test_raw: pd.DataFrame,
    cfg: CommentsConfig,
    proba_npys: list[Path],
    mapping_dir: Path,
    min_conf: float,
    min_margin: float,
    per_class_max: int,
    target_total: int,
    require_agreement: bool,
) -> pd.DataFrame:
    """
    Return canonical pseudo-labeled DataFrame with:
      id, platform, comment, label, confidence, margin
    """

    df_test = schema.rename_raw_columns(df_test_raw, cfg, is_train=False)

    id2label = _load_id2label_from_run_or_fold(mapping_dir)
    num_classes = int(len(id2label))

    preds: list[np.ndarray] = []
    confs: list[np.ndarray] = []
    margins: list[np.ndarray] = []

    for pth in proba_npys:
        proba = np.load(pth)
        if proba.ndim != 2:
            raise ValueError(f"Expected 2D proba array, got shape={proba.shape} for {pth}")
        if len(df_test) != proba.shape[0]:
            raise ValueError(f"Row mismatch: test rows={len(df_test)} vs proba rows={proba.shape[0]} for {pth}")
        if proba.shape[1] != num_classes:
            raise ValueError(f"Class mismatch: proba has {proba.shape[1]} cols but mapping has {num_classes} labels ({pth})")
        pred_id, conf, margin = _top1_top2(proba)
        preds.append(pred_id)
        confs.append(conf)
        margins.append(margin)

    # Agreement filter
    pred0 = preds[0]
    if bool(require_agreement) and len(preds) > 1:
        agree = np.ones((len(pred0),), dtype=bool)
        for pr in preds[1:]:
            agree &= (pr == pred0)
    else:
        agree = np.ones((len(pred0),), dtype=bool)

    # Conservative: use min across teachers for thresholds; average for reporting
    conf_min = np.min(np.stack(confs, axis=1), axis=1) if len(confs) > 1 else confs[0]
    margin_min = np.min(np.stack(margins, axis=1), axis=1) if len(margins) > 1 else margins[0]
    conf_avg = np.mean(np.stack(confs, axis=1), axis=1) if len(confs) > 1 else confs[0]
    margin_avg = np.mean(np.stack(margins, axis=1), axis=1) if len(margins) > 1 else margins[0]

    keep = agree & (conf_min >= float(min_conf)) & (margin_min >= float(min_margin))
    if not keep.any():
        raise ValueError("No pseudo-labels selected. Lower --min_conf/--min_margin or disable --require_agreement.")

    pred_label = np.array([id2label[int(i)] for i in pred0], dtype=np.int32)

    pseudo = pd.DataFrame(
        {
            cfg.id_col: df_test[cfg.id_col].astype(int),
            cfg.platform_col: df_test[cfg.platform_col].astype("string"),
            cfg.text_col: df_test[cfg.text_col].astype("string"),
            cfg.target_col: pred_label.astype(int),
            "confidence": conf_avg.astype(np.float32),
            "margin": margin_avg.astype(np.float32),
        }
    )
    pseudo = pseudo.loc[keep].copy()

    # Balanced selection (optional)
    if int(per_class_max) > 0:
        parts: list[pd.DataFrame] = []
        for lbl, sub in pseudo.groupby(cfg.target_col, sort=True):
            sub2 = sub.sort_values(["confidence", "margin"], ascending=False).head(int(per_class_max))
            parts.append(sub2)
        pseudo = pd.concat(parts, axis=0, ignore_index=True)

    if int(target_total) > 0 and len(pseudo) > int(target_total):
        # Keep highest-confidence rows overall (after per-class cap)
        pseudo = pseudo.sort_values(["confidence", "margin"], ascending=False).head(int(target_total)).reset_index(drop=True)

    return pseudo


def _raw_cols_for(df_raw: pd.DataFrame, cfg: CommentsConfig, *, is_train: bool) -> dict[str, str]:
    """
    Return canonical -> raw column name mapping for the given raw dataframe.
    """

    raw_to_canon = schema.guess_raw_to_canon(df_raw, cfg, is_train=is_train)
    canon_to_raw = {v: k for k, v in raw_to_canon.items()}
    # Ensure required
    req = [cfg.id_col, cfg.platform_col, cfg.text_col]
    if is_train:
        req.append(cfg.target_col)
    missing = [c for c in req if c not in canon_to_raw]
    if missing:
        raise ValueError(f"Could not resolve required raw columns: {missing}")
    return canon_to_raw


def append_pseudo_to_train(
    *,
    df_train_raw: pd.DataFrame,
    df_test_raw: pd.DataFrame,
    pseudo_canon: pd.DataFrame,
    cfg: CommentsConfig,
    dedup_clean_comment: bool,
    add_meta_cols: bool,
) -> pd.DataFrame:
    """
    Return a new raw-style train dataframe with pseudo rows appended.
    Output columns follow the ORIGINAL train.csv schema (raw French headers),
    optionally with extra metadata columns (source/confidence/margin/orig_test_id).
    """

    train_cols = list(df_train_raw.columns)
    tr_map = _raw_cols_for(df_train_raw, cfg, is_train=True)
    te_map = _raw_cols_for(df_test_raw, cfg, is_train=False)

    # Build pseudo rows with raw column names
    df_test = df_test_raw.copy()
    df_test = df_test.rename(columns=schema.guess_raw_to_canon(df_test_raw, cfg, is_train=False))

    pseudo = pseudo_canon.copy()

    # Dedup vs train on CLEANED comment (optional, recommended)
    if bool(dedup_clean_comment):
        tr_clean = preprocess.preprocess_comments_df(df_train_raw, cfg, is_train=True)
        pseudo_clean = preprocess.preprocess_comments_df(pseudo[[cfg.id_col, cfg.platform_col, cfg.text_col, cfg.target_col]].copy(), cfg, is_train=True)
        train_set = set(tr_clean[cfg.text_col].astype(str).tolist())
        mask = ~pseudo_clean[cfg.text_col].astype(str).isin(train_set)
        pseudo = pseudo.loc[mask.to_numpy()].copy()
        # Also drop duplicates within pseudo
        pseudo = pseudo.drop_duplicates(subset=[cfg.text_col], keep="first").reset_index(drop=True)

    if len(pseudo) == 0:
        raise ValueError("After dedup, no pseudo rows remain. Try lowering thresholds or disable --dedup.")

    # Create new unique ids for pseudo rows to avoid collisions with existing train ids
    tr_id = pd.to_numeric(df_train_raw[tr_map[cfg.id_col]], errors="coerce")
    max_id = int(tr_id.max()) if tr_id.notna().any() else 0
    pseudo_new_id = np.arange(max_id + 1, max_id + 1 + len(pseudo), dtype=np.int64)

    # Use original raw values from test for platform/comment
    # Map pseudo rows back to test rows by original test id
    test_by_id = df_test.set_index(cfg.id_col, drop=False)
    pseudo_ids = pseudo[cfg.id_col].astype(int).to_numpy()
    sub_test = test_by_id.loc[pseudo_ids]

    pseudo_raw = pd.DataFrame(index=range(len(pseudo)))
    pseudo_raw[tr_map[cfg.id_col]] = pseudo_new_id
    pseudo_raw[tr_map[cfg.platform_col]] = sub_test[cfg.platform_col].astype(object).to_numpy()
    pseudo_raw[tr_map[cfg.text_col]] = sub_test[cfg.text_col].astype(object).to_numpy()
    pseudo_raw[tr_map[cfg.target_col]] = pseudo[cfg.target_col].astype(int).to_numpy()

    if bool(add_meta_cols):
        pseudo_raw["source"] = "pseudo"
        pseudo_raw["orig_test_id"] = pseudo[cfg.id_col].astype(int).to_numpy()
        pseudo_raw["confidence"] = pseudo["confidence"].astype(float).to_numpy()
        pseudo_raw["margin"] = pseudo["margin"].astype(float).to_numpy()

        df_train_out = df_train_raw.copy()
        if "source" not in df_train_out.columns:
            df_train_out["source"] = "train"
        if "orig_test_id" not in df_train_out.columns:
            df_train_out["orig_test_id"] = np.nan
        if "confidence" not in df_train_out.columns:
            df_train_out["confidence"] = np.nan
        if "margin" not in df_train_out.columns:
            df_train_out["margin"] = np.nan
        train_cols = list(df_train_out.columns)
        # Align columns: ensure pseudo has all
        for c in train_cols:
            if c not in pseudo_raw.columns:
                pseudo_raw[c] = np.nan
        pseudo_raw = pseudo_raw[train_cols]
        out = pd.concat([df_train_out, pseudo_raw], axis=0, ignore_index=True)
        return out

    # No extra cols: ensure pseudo_raw has exact train columns (missing -> NaN)
    for c in train_cols:
        if c not in pseudo_raw.columns:
            pseudo_raw[c] = np.nan
    pseudo_raw = pseudo_raw[train_cols]
    out = pd.concat([df_train_raw, pseudo_raw], axis=0, ignore_index=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--team_dir", type=str, default=None)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--train_filename", type=str, default="train.csv")
    p.add_argument("--test_filename", type=str, default="test_file.csv")

    p.add_argument("--proba_npys", type=str, nargs="+", required=True, help="One or more .npy probability files (same test row order).")
    p.add_argument("--mapping_dir", type=str, required=True, help="Transformer run dir (preferred) or fold dir (fallback) to read label mapping.")

    p.add_argument("--min_conf", type=float, default=0.99)
    p.add_argument("--min_margin", type=float, default=0.10)
    p.add_argument("--per_class_max", type=int, default=333, help="Balanced cap per class (0 = unlimited).")
    p.add_argument("--target_total", type=int, default=3000, help="Optional overall cap after per-class selection (0 = unlimited).")
    p.add_argument("--require_agreement", action="store_true", help="Require identical top-1 class across all proba files.")

    p.add_argument("--dedup", action="store_true", help="Drop pseudo rows whose cleaned comment already exists in train (recommended).")
    p.add_argument("--add_meta_cols", action="store_true", help="Add source/confidence/margin/orig_test_id columns to output train.")

    p.add_argument("--output_pseudo_csv", type=str, default=None, help="Optional path to save the pseudo-labeled rows as a CSV.")
    p.add_argument("--output_train_csv", type=str, default=None, help="Output augmented train CSV path.")
    p.add_argument("--inplace", action="store_true", help="Overwrite the original train.csv (a .bak copy is created).")
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

    train_path = data_dir / args.train_filename
    test_path = data_dir / args.test_filename
    schema.validate_file_exists(train_path)
    schema.validate_file_exists(test_path)

    df_train_raw = schema.read_csv_robust(train_path)
    df_test_raw = schema.read_csv_robust(test_path)

    pseudo = build_pseudo_df(
        df_test_raw=df_test_raw,
        cfg=cfg,
        proba_npys=[Path(x) for x in args.proba_npys],
        mapping_dir=Path(args.mapping_dir),
        min_conf=float(args.min_conf),
        min_margin=float(args.min_margin),
        per_class_max=int(args.per_class_max),
        target_total=int(args.target_total),
        require_agreement=bool(args.require_agreement),
    )

    # Optionally save pseudo csv (canonical)
    if args.output_pseudo_csv:
        out_p = Path(args.output_pseudo_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        pseudo.to_csv(out_p, index=False, encoding="utf-8")
        print(f"Saved pseudo rows -> {out_p} (rows={len(pseudo)})")

    aug = append_pseudo_to_train(
        df_train_raw=df_train_raw,
        df_test_raw=df_test_raw,
        pseudo_canon=pseudo,
        cfg=cfg,
        dedup_clean_comment=bool(args.dedup),
        add_meta_cols=bool(args.add_meta_cols),
    )

    if args.inplace:
        backup = train_path.with_suffix(train_path.suffix + ".bak")
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(train_path), str(backup))
        out_train = train_path
    else:
        out_train = Path(args.output_train_csv) if args.output_train_csv else (data_dir / "train_augmented.csv")

    out_train.parent.mkdir(parents=True, exist_ok=True)
    aug.to_csv(out_train, index=False, encoding="utf-8")
    print(f"Saved augmented train -> {out_train} (rows={len(aug)}; added_pseudo={len(aug)-len(df_train_raw)})")


if __name__ == "__main__":
    main()


