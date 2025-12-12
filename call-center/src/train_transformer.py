"""
Train a Transformer text classifier for the Call Center challenge (6 classes).

Why
----
Your current best model (LinearSVC + TF-IDF + time-gating) is very strong but has plateaued.
A transformer can add *model diversity*; ensembling transformer + linear often improves Kaggle Macro-F1.

This script is designed for Google Colab (Drive mounted) and is deterministic given a fixed seed,
but note that GPU kernels can still introduce minor nondeterminism depending on your runtime.

Inputs
------
- Cleaned train CSV: <team_dir>/outputs/callcenter/clean/train_clean.csv

Artifacts (default)
-------------------
<team_dir>/outputs/callcenter/transformer/<run_id>/
- model/                  (HF save_pretrained)
- tokenizer/              (HF save_pretrained)
- train_config.json
- metrics.json
- feature_spec.json       (includes time_windows_by_class + defaults for time gating)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

import inspect
import warnings
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Suppress harmless CUDA/TensorFlow registration warnings (they don't affect training)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*tokenizer.*")
warnings.filterwarnings("ignore", message=".*cuFFT.*")
warnings.filterwarnings("ignore", message=".*cuDNN.*")
warnings.filterwarnings("ignore", message=".*cuBLAS.*")
warnings.filterwarnings("ignore", message=".*computation placer.*")

from config import default_config, guess_repo_root, guess_team_dir, resolve_callcenter_output_dir


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")


def _json_dump(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def _compute_time_windows_by_class(df: pd.DataFrame, *, datetime_col: str, y: np.ndarray) -> dict[str, dict[str, str]]:
    if datetime_col not in df.columns:
        return {}
    dt = pd.to_datetime(df[datetime_col], errors="coerce")
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


def _build_input_text(df: pd.DataFrame, *, text_col: str, extra_cols: list[str]) -> pd.Series:
    """
    Build the text fed to the transformer. Keep it simple and robust.
    """
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


class TextClsDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int] | None, tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self) -> int:  # pragma: no cover
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


class WeightedTrainer(Trainer):
    """
    Apply class-weighted cross-entropy to handle severe imbalance.
    """

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get("logits")
        if labels is None:
            loss = outputs.get("loss")
        else:
            # IMPORTANT: HuggingFace moves model + batch tensors to GPU automatically.
            # Keep class weights on the same device as logits to avoid device mismatch errors.
            w = self._class_weights
            if w is not None and hasattr(logits, "device") and w.device != logits.device:
                w = w.to(logits.device)
            if hasattr(labels, "device") and hasattr(logits, "device") and labels.device != logits.device:
                labels = labels.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=w)
            loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--team_dir", type=str, default=None)
    p.add_argument("--clean_dir", type=str, default=None)
    p.add_argument("--artifacts_dir", type=str, default=None)

    p.add_argument("--model_name", type=str, default="xlm-roberta-base")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_size", type=float, default=0.1)

    # optional text prefix cols (helps transformer use structured info)
    p.add_argument(
        "--extra_text_cols",
        type=str,
        default="service_type,tech_comm,subs_level",
        help="Comma-separated canonical columns to prepend into the input text. Example: 'service_type,tech_comm'.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = default_config()

    # Repro settings
    os.environ.setdefault("PYTHONHASHSEED", "0")
    set_seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

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

    if args.artifacts_dir:
        artifacts_dir = Path(args.artifacts_dir)
    elif team_dir is not None:
        run_id = _utc_run_id()
        artifacts_dir = team_dir / "outputs" / "callcenter" / "transformer" / run_id
    else:
        run_id = _utc_run_id()
        artifacts_dir = guess_repo_root() / "outputs" / "callcenter" / "transformer" / run_id

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(train_path)
    if cfg.target_col not in df.columns:
        raise ValueError(f"Missing target column '{cfg.target_col}' in cleaned train file.")

    # stable ordering
    if cfg.id_col in df.columns:
        df = df.sort_values(cfg.id_col, kind="mergesort").reset_index(drop=True)

    y = pd.to_numeric(df[cfg.target_col], errors="raise").astype(int).to_numpy()
    labels = sorted(np.unique(y).tolist())

    extra_cols = [c.strip() for c in str(args.extra_text_cols).split(",") if c.strip()]
    text = _build_input_text(df, text_col=cfg.text_col, extra_cols=extra_cols).astype(str).tolist()

    # Split
    idx = np.arange(len(df))
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=float(args.val_size),
        random_state=int(args.seed),
        shuffle=True,
        stratify=y,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_name))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(args.model_name),
        num_labels=len(labels),
    )

    # Class weights (inverse frequency)
    vc = pd.Series(y).value_counts().to_dict()
    n = len(y)
    k = len(labels)
    # Build weight vector aligned with class indices [0..num_labels-1].
    # NOTE: in this competition labels are 0..5, but keep it defensive.
    num_labels = int(model.config.num_labels)
    w = np.ones((num_labels,), dtype="float32")
    for cls in labels:
        ci = int(cls)
        if 0 <= ci < num_labels and ci in vc:
            w[ci] = float(n / (k * vc[ci]))
    # Keep on CPU; we move it to the right device inside compute_loss.
    w_t = torch.tensor(w, dtype=torch.float32)

    ds_tr = TextClsDataset([text[i] for i in tr_idx], [int(y[i]) for i in tr_idx], tokenizer, int(args.max_length))
    ds_va = TextClsDataset([text[i] for i in va_idx], [int(y[i]) for i in va_idx], tokenizer, int(args.max_length))

    def compute_metrics(eval_pred):
        logits, labels_np = eval_pred
        pred = np.argmax(logits, axis=1)
        macro = f1_score(labels_np, pred, average="macro", zero_division=0)
        return {"macro_f1": float(macro)}

    # transformers>=4.57 renamed `evaluation_strategy` -> `eval_strategy`.
    # Build kwargs defensively to support both.
    ta_sig = inspect.signature(TrainingArguments.__init__)
    ta_params = set(ta_sig.parameters.keys())

    ta_kwargs: dict[str, Any] = {
        "output_dir": str(artifacts_dir / "checkpoints"),
        "num_train_epochs": int(args.epochs),
        "per_device_train_batch_size": int(args.batch_size),
        "per_device_eval_batch_size": int(args.batch_size),
        "learning_rate": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 50,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": 2,
        "report_to": [],
        "seed": int(args.seed),
        "data_seed": int(args.seed),
        "fp16": bool(torch.cuda.is_available()),
    }
    if "evaluation_strategy" in ta_params:
        ta_kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in ta_params:
        ta_kwargs["eval_strategy"] = "epoch"

    # Filter unknown args for maximal compatibility.
    ta_kwargs = {k: v for k, v in ta_kwargs.items() if k in ta_params}
    train_args = TrainingArguments(**ta_kwargs)

    # Use processing_class instead of tokenizer for newer Transformers compatibility
    # (tokenizer still works in older versions but can be deprecated in newer ones)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": train_args,
        "train_dataset": ds_tr,
        "eval_dataset": ds_va,
        "compute_metrics": compute_metrics,
        "class_weights": w_t,
    }
    # NOTE: WeightedTrainer.__init__ uses *args/**kwargs, so inspect Trainer.__init__.
    trainer_init_sig = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_init_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_init_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = WeightedTrainer(**trainer_kwargs)

    trainer.train()

    # Evaluate best model
    out = trainer.predict(ds_va)
    logits = out.predictions
    pred = np.argmax(logits, axis=1)
    y_va = np.array([int(y[i]) for i in va_idx], dtype=int)
    macro = float(f1_score(y_va, pred, average="macro", zero_division=0))
    rep = classification_report(y_va, pred, labels=labels, output_dict=True, zero_division=0)

    # Save model + tokenizer
    model_dir = artifacts_dir / "model"
    tok_dir = artifacts_dir / "tokenizer"
    model_dir.mkdir(parents=True, exist_ok=True)
    tok_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(tok_dir))

    # Save feature spec for inference (time-gating metadata)
    time_windows_by_class = _compute_time_windows_by_class(df, datetime_col=cfg.datetime_col, y=y)
    feature_spec = {
        "run_id": _utc_run_id(),
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "clean_dir": str(clean_dir),
        "train_path": str(train_path),
        "model_name": str(args.model_name),
        "max_length": int(args.max_length),
        "extra_text_cols": extra_cols,
        "config": asdict(cfg),
        "time_windows_by_class": time_windows_by_class,
        "time_gating_defaults": {
            "gate_classes": [0, 3, 4, 5],
            "fallback_class": 2,
            "fallback_map": {"5": 1},
        },
        "labels": labels,
    }
    _json_dump(artifacts_dir / "feature_spec.json", feature_spec)

    metrics = {
        "env": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "data": {"rows": int(len(df)), "labels": labels},
        "val": {
            "val_size": float(args.val_size),
            "macro_f1": macro,
            "classification_report": rep,
        },
        "artifacts": {
            "artifacts_dir": str(artifacts_dir),
            "model_dir": str(model_dir),
            "tokenizer_dir": str(tok_dir),
        },
    }
    _json_dump(artifacts_dir / "metrics.json", metrics)

    print("[train_transformer] Saved artifacts to:", artifacts_dir)
    print("[train_transformer] val macro_f1:", macro)


if __name__ == "__main__":
    main()



