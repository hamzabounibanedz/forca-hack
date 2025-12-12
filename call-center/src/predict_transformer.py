"""
Inference + Kaggle submission writer for Transformer model (Call Center challenge).

Loads artifacts from train_transformer.py:
- <artifacts_dir>/model/
- <artifacts_dir>/tokenizer/
- <artifacts_dir>/feature_spec.json

Optionally applies time-window gating using train-derived windows saved in feature_spec.json.
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
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
    # fallback
    s = str(v).strip()
    return [s] if s else []


def _build_input_text(df: pd.DataFrame, *, text_col: str, extra_cols: list[str]) -> pd.Series:
    s = df[text_col].astype("string").fillna("")
    if not extra_cols:
        return s
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
    return s


class TextDataset(Dataset):
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
            fb = int(fb_map.get(int(cls), fallback_class))
            out[mask] = fb
            changed += int(mask.sum())
    return out, changed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--team_dir", type=str, default=None)
    p.add_argument("--clean_dir", type=str, default=None)
    p.add_argument("--artifacts_dir", type=str, required=True, help="Transformer artifacts dir (contains model/, tokenizer/, feature_spec.json).")
    p.add_argument("--sample_submission_path", type=str, default=None)
    p.add_argument("--run_id", type=str, default=None)
    p.add_argument("--out_path", type=str, default=None)

    p.add_argument("--batch_size", type=int, default=32)

    p.add_argument("--time_gate", action="store_true")
    p.add_argument("--time_gate_classes", type=str, default=None)
    p.add_argument("--time_gate_fallback_class", type=int, default=None)
    p.add_argument("--time_gate_fallback_map", type=str, default=None)
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
        clean_dir = guess_local_data_dir()

    test_path = clean_dir / cfg.clean_test_name
    if not test_path.exists():
        raise FileNotFoundError(f"Cleaned test file not found: {test_path}")

    artifacts_dir = Path(args.artifacts_dir)
    feature_spec_path = artifacts_dir / "feature_spec.json"
    if not feature_spec_path.exists():
        raise FileNotFoundError(f"feature_spec.json not found: {feature_spec_path}")
    spec = _load_json(feature_spec_path)

    model_dir = artifacts_dir / "model"
    tok_dir = artifacts_dir / "tokenizer"
    if not model_dir.exists() or not tok_dir.exists():
        raise FileNotFoundError(f"Missing model/ or tokenizer/ in artifacts_dir: {artifacts_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    df_test = pd.read_csv(test_path)
    if cfg.id_col not in df_test.columns:
        raise ValueError(f"Missing id column '{cfg.id_col}' in cleaned test file.")

    extra_cols = _parse_extra_text_cols(spec.get("extra_text_cols"))
    texts = _build_input_text(df_test, text_col=cfg.text_col, extra_cols=extra_cols).astype(str).tolist()

    ds = TextDataset(texts, tokenizer, int(spec.get("max_length") or 128))
    dl = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False)

    all_logits: list[np.ndarray] = []
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            logits = out.logits.detach().cpu().numpy()
            all_logits.append(logits)
    logits = np.concatenate(all_logits, axis=0)
    pred = logits.argmax(axis=1).astype(int)

    if args.time_gate:
        if cfg.datetime_col not in df_test.columns:
            raise ValueError(f"--time_gate requires '{cfg.datetime_col}' in cleaned test file: {test_path}")
        time_windows = spec.get("time_windows_by_class")
        if not isinstance(time_windows, dict) or not time_windows:
            raise ValueError("Missing time_windows_by_class in transformer feature_spec.json.")

        defaults = spec.get("time_gating_defaults") if isinstance(spec, dict) else None
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
            f"[predict_transformer] Applied time gating: classes={gate_classes} -> fallback={fallback_class} map={fallback_map}  changed={changed}"
        )

    # Resolve sample submission path
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
        out_path = out_dir / f"callcenter_transformer_{run_id}.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[predict_transformer] Saved submission: {out_path}")
    print(out.head(3).to_string(index=False))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main()



