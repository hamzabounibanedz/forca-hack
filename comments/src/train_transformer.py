"""
Transformer fine-tuning for Social Media Comments (multi-class, Macro-F1).

Supports multilingual models like:
- xlm-roberta-base (strong baseline)
- any DziriBERT checkpoint (pass via --model_name)

Usage (Colab recommended)
------------------------
python comments/src/train_transformer.py --team_dir /content/drive/MyDrive/FORSA_team --model_name xlm-roberta-base
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import preprocess
import schema
from config import (
    CommentsConfig,
    cfg_to_dict,
    guess_local_data_dir,
    guess_repo_root,
    guess_team_dir,
    resolve_comments_data_dir,
    resolve_comments_output_dir,
)
from metrics import macro_f1_score


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


_FLAG_SPECS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("FIBRE", re.compile(r"\b(?:fibre|ftth|fttx|idoom_fibre)\b", re.IGNORECASE)),
    ("MODEM", re.compile(r"\b(?:modem|wifi6)\b", re.IGNORECASE)),
    ("PORTAL", re.compile(r"\b(?:espace_client|espace\s*client|application|app|login|mot\s*de\s*passe)\b|(?:فضاء\s*الزبون|فضاء\s*الزبائن|الولوج|حساب|كلمة\s*السر)", re.IGNORECASE)),
    ("PRICES", re.compile(r"\b(?:prix|tarif|cher|promo|offre|offres)\b|(?:سعر|الاسعار|الأسعار|اسعار|العروض)", re.IGNORECASE)),
    ("OUTAGE", re.compile(r"\b(?:panne|coupure|down|hs|marche\s+pas|ne\s+marche\s+pas)\b|(?:مقطوع|انقطاع|ماكاش|مكانش|تقطع|مقطوعة)", re.IGNORECASE)),
    ("SUPPORT", re.compile(r"\b(?:100|12|1500)\b")),
    ("SPEED", re.compile(r"\b(?:mbps|gbps)\b|(?:ميغا|جيغا)", re.IGNORECASE)),
    # Extra disambiguation for the hard cluster {2,4,6}:
    ("WAIT", re.compile(r"\b(?:a\s*quand|quand|depuis|attend|attente)\b|(?:وقتاش|متي|مزال|مازال|نستناو|نستنو|انتظار|راني\s+في|راه|راهم)", re.IGNORECASE)),
    ("LOCATION", re.compile(r"\b(?:wilaya|commune|quartier|centre|ville|rue)\b|(?:ولاية|بلدية|حي|شارع|مدينة|الجزائر|وهران|عنابة|قسنطينة|سطيف|بجاية|الشلف|الجلفة)", re.IGNORECASE)),
)


def _format_text(df: pd.DataFrame, cfg: CommentsConfig, *, add_meta: bool = True, add_flags: bool = False) -> list[str]:
    """
    Build the final input string for transformers.
    We keep the raw cleaned text but optionally prepend:
    - platform token
    - high-signal intent flags (helps separate confusable classes like 8 vs 9, 2 vs 4 vs 6)
    """
    txt = df[cfg.text_col].astype("string").fillna("")
    if not add_meta and not add_flags:
        return txt.astype(str).tolist()

    parts = []
    if add_meta:
        plat = df[cfg.platform_col].astype("string").fillna("unk").str.lower()
        parts.append("[PLATFORM] " + plat)

    if add_flags:
        # Vectorized contains per flag (fast even for ~30k rows).
        flag_cols: list[pd.Series] = []
        for name, pat in _FLAG_SPECS:
            flag_cols.append(txt.str.contains(pat, regex=True, na=False).map(lambda b, _n=name: f"[{_n}]" if b else ""))
        flags_str = pd.Series([""] * len(df), index=df.index, dtype="string")
        for col in flag_cols:
            flags_str = (flags_str + " " + col).astype("string")
        parts.append(flags_str.str.strip())

    parts.append("[TEXT] " + txt)
    out = parts[0]
    for p in parts[1:]:
        out = (out + " " + p).astype("string")
    return out.astype(str).tolist()


def _build_training_args_kwargs(
    *,
    output_dir: str,
    learning_rate: float,
    epochs: float,
    train_batch_size: int,
    eval_batch_size: int,
    gradient_accumulation_steps: int,
    dataloader_num_workers: int,
    weight_decay: float,
    warmup_ratio: float,
    seed: int,
    fp16: bool,
    eval_strategy: str = "epoch",
    save_strategy: str = "epoch",
    save_total_limit: int = 2,
    logging_steps: int = 50,
) -> dict[str, Any]:
    """
    Transformers occasionally renames TrainingArguments fields across versions.
    We build kwargs dynamically based on the installed signature.
    """
    from transformers import TrainingArguments

    sig = inspect.signature(TrainingArguments.__init__)
    params = sig.parameters

    kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "learning_rate": learning_rate,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": int(max(1, gradient_accumulation_steps)),
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "seed": seed,
        "fp16": fp16,
        "save_total_limit": save_total_limit,
        "logging_steps": logging_steps,
        "report_to": "none",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
    }

    # evaluation strategy field name
    if "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = eval_strategy
    elif "eval_strategy" in params:
        kwargs["eval_strategy"] = eval_strategy

    # save strategy field name
    if "save_strategy" in params:
        kwargs["save_strategy"] = save_strategy
    elif "save_steps" in params and save_strategy == "steps":
        kwargs["save_steps"] = 500

    # workers
    if "dataloader_num_workers" in params:
        kwargs["dataloader_num_workers"] = int(max(0, dataloader_num_workers))

    return kwargs


def train_transformer_cv(
    df_train_raw: pd.DataFrame,
    cfg: CommentsConfig,
    *,
    model_name: str,
    output_dir: Path,
    n_splits: int,
    seed: int,
    max_length: int,
    train_batch_size: int,
    eval_batch_size: int,
    gradient_accumulation_steps: int,
    dataloader_num_workers: int,
    epochs: float,
    lr: float,
    weight_decay: float,
    warmup_ratio: float,
    add_meta: bool,
    add_flags: bool,
    early_stopping_patience: int,
    fp16: bool,
    use_class_weights: bool,
    train_folds: list[int] | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """
    Stratified K-fold fine-tuning. Saves each fold checkpoint + metrics.
    """
    _seed_everything(seed)

    try:
        import torch
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
        )
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing transformer dependencies. Install with: pip install transformers torch"
        ) from e

    # --- Prepare cleaned training data ---
    df = preprocess.preprocess_comments_df(df_train_raw, cfg, is_train=True)
    label2id, id2label = schema.build_label_mapping(df[cfg.target_col])

    y_raw = df[cfg.target_col].astype(int).to_numpy()
    y = np.asarray([label2id[int(v)] for v in y_raw], dtype=np.int64)
    texts = _format_text(df, cfg, add_meta=add_meta, add_flags=add_flags)

    num_labels = len(label2id)

    # Write mappings early (so interrupted runs can still be used for prediction)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "label_mapping.json").write_text(
        json.dumps({"label2id": label2id, "id2label": id2label}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "preprocess_config.json").write_text(
        json.dumps(cfg_to_dict(cfg), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # --- Dataset wrapper ---
    class TextDataset(Dataset):
        def __init__(self, _texts: list[str], _labels: np.ndarray | None, _tokenizer, _max_len: int):
            self.texts = _texts
            self.labels = _labels
            self.tok = _tokenizer
            self.max_len = _max_len

        def __len__(self) -> int:
            return len(self.texts)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            # No padding here; we use DataCollatorWithPadding for dynamic padding (faster).
            item = self.tok(
                self.texts[idx],
                truncation=True,
                max_length=self.max_len,
                padding=False,
            )
            if self.labels is not None:
                item["labels"] = int(self.labels[idx])
            return item

    # --- Metrics ---
    def compute_metrics(eval_pred):
        # Compatible with both (predictions, labels) tuple and EvalPrediction object
        logits = getattr(eval_pred, "predictions", None)
        labels = getattr(eval_pred, "label_ids", None)
        if logits is None or labels is None:
            logits, labels = eval_pred
        pred = np.asarray(logits).argmax(axis=1)
        f1 = macro_f1_score(labels, pred, labels=list(range(num_labels)))
        return {"macro_f1": f1}

    # --- Optional weighted loss ---
    class WeightedTrainer(Trainer):
        def __init__(self, *args, class_weights=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            """
            Compatible with newer Transformers Trainer which may pass extra kwargs
            like `num_items_in_batch`.
            """
            labels = inputs.get("labels")
            if labels is None:
                return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

            outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
            logits = outputs.logits

            if self.class_weights is None:
                loss_fct = torch.nn.CrossEntropyLoss()
            else:
                loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))

            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    # --- CV loop ---
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_scores: list[float] = []

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if fp16 else None)

    # Global class weights (from full train) for stability
    class_weights = None
    if use_class_weights:
        counts = np.bincount(y, minlength=num_labels).astype(np.float32)
        w = counts.sum() / np.clip(counts, 1.0, None)
        w = w / w.mean()
        class_weights = torch.tensor(w, dtype=torch.float32)

    folds_set = set(int(x) for x in train_folds) if train_folds else None

    def _has_saved_weights(p: Path) -> bool:
        return any((p / name).exists() for name in ("model.safetensors", "pytorch_model.bin", "pytorch_model.safetensors"))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(texts, y), start=1):
        if folds_set is not None and fold not in folds_set:
            continue
        fold_dir = output_dir / f"{model_name.replace('/', '_')}_fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_texts = [texts[i] for i in tr_idx]
        val_texts = [texts[i] for i in va_idx]
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        ds_tr = TextDataset(train_texts, y_tr, tokenizer, max_length)
        ds_va = TextDataset(val_texts, y_va, tokenizer, max_length)

        # Resume mode: only skip when the fold directory has BOTH config and weights.
        # (If training was interrupted, config.json may exist but weights may be missing.)
        if resume and (fold_dir / "config.json").exists() and _has_saved_weights(fold_dir):
            from transformers import TrainingArguments

            model = AutoModelForSequenceClassification.from_pretrained(str(fold_dir))
            eval_args = TrainingArguments(
                output_dir=str(fold_dir),
                per_device_eval_batch_size=eval_batch_size,
                dataloader_num_workers=int(max(0, dataloader_num_workers)),
                report_to="none",
            )
            trainer = Trainer(
                model=model,
                args=eval_args,
                eval_dataset=ds_va,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )
            eval_metrics = trainer.evaluate()
            f1 = float(eval_metrics.get("eval_macro_f1", eval_metrics.get("macro_f1", 0.0)))
            fold_scores.append(f1)
            print(f"[fold {fold}] resumed (skipped training) macro_f1={f1:.5f} -> {fold_dir}")
            continue

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label={i: str(id2label[i]) for i in range(num_labels)},
            label2id={str(k): int(v) for k, v in label2id.items()},
        )

        training_args_kwargs = _build_training_args_kwargs(
            output_dir=str(fold_dir),
            learning_rate=lr,
            epochs=epochs,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            dataloader_num_workers=dataloader_num_workers,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            seed=seed,
            fp16=fp16,
        )
        from transformers import TrainingArguments

        training_args = TrainingArguments(**training_args_kwargs)

        callbacks = []
        if early_stopping_patience > 0:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

        trainer_cls = WeightedTrainer if use_class_weights else Trainer
        trainer_kwargs: dict[str, Any] = dict(
            model=model,
            args=training_args,
            train_dataset=ds_tr,
            eval_dataset=ds_va,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )
        if use_class_weights:
            trainer_kwargs["class_weights"] = class_weights

        trainer = trainer_cls(**trainer_kwargs)

        trainer.train()
        eval_metrics = trainer.evaluate()
        f1 = float(eval_metrics.get("eval_macro_f1", eval_metrics.get("macro_f1", 0.0)))
        fold_scores.append(f1)
        print(f"[fold {fold}] macro_f1={f1:.5f} -> {fold_dir}")

        # Save best model + tokenizer
        trainer.save_model(str(fold_dir))
        tokenizer.save_pretrained(str(fold_dir))

    mean_f1 = float(np.mean(fold_scores)) if fold_scores else 0.0
    std_f1 = float(np.std(fold_scores)) if fold_scores else 0.0

    return {
        "model_name": model_name,
        "n_splits": n_splits,
        "seed": seed,
        "max_length": max_length,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": int(max(1, gradient_accumulation_steps)),
        "dataloader_num_workers": int(max(0, dataloader_num_workers)),
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "add_meta": add_meta,
        "add_flags": add_flags,
        "early_stopping_patience": early_stopping_patience,
        "use_class_weights": use_class_weights,
        "fold_scores": fold_scores,
        "macro_f1_mean": mean_f1,
        "macro_f1_std": std_f1,
        "output_dir": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--train_filename", type=str, default="train.csv")
    parser.add_argument("--model_name", type=str, default="xlm-roberta-base")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--epochs", type=float, default=4.0)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--add_meta", action="store_true", help="Prepend platform tokens to text.")
    parser.add_argument("--add_flags", action="store_true", help="Prepend high-signal intent flags to text.")
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--run_dir", type=str, default=None, help="Use an existing run dir (resume-friendly).")
    parser.add_argument("--train_folds", type=int, nargs="*", default=None, help="Train only these fold numbers (1..K).")
    parser.add_argument("--resume", action="store_true", help="Skip already-trained folds found in run_dir.")
    parser.add_argument("--output_dir", type=str, default=None)
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

    train_path = data_dir / args.train_filename
    schema.validate_file_exists(train_path)
    df_train_raw = schema.read_csv_robust(train_path)

    # Resolve output dir
    if args.output_dir:
        out_dir = Path(args.output_dir)
    elif args.team_dir:
        out_dir = resolve_comments_output_dir(Path(args.team_dir)) / "models" / "transformers"
    else:
        out_dir = guess_repo_root() / "outputs" / "comments" / "models" / "transformers"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        resume = True  # run_dir implies resume-safe behavior
    else:
        run_dir = out_dir / f"{args.model_name.replace('/', '_')}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)
        resume = bool(args.resume)

    result = train_transformer_cv(
        df_train_raw,
        cfg,
        model_name=args.model_name,
        output_dir=run_dir,
        n_splits=args.n_splits,
        seed=args.seed,
        max_length=args.max_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        add_meta=bool(args.add_meta),
        add_flags=bool(args.add_flags),
        early_stopping_patience=int(args.early_stopping_patience),
        fp16=bool(args.fp16),
        use_class_weights=bool(args.use_class_weights),
        train_folds=args.train_folds,
        resume=resume,
    )

    (run_dir / "run_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


