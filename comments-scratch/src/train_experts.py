"""
Train 4 scratch expert models for the comments challenge.

Experts
-------
- fr      : word-level TextCNN
- ar      : char-level TextCNN
- dz_ar   : char-level TextCNN (dialect markers in Arabic script)
- dz_lat  : char-level TextCNN (Arabizi / Latin-script Darija)

Key idea
--------
Each expert is trained with **sample weights**:
  weight = 1.0 for samples belonging to its detected group
  weight = off_group_weight for all other samples

This keeps all experts seeing all labels, while specializing to their script/language.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import lang
import metrics
import preprocess
import schema
from config import (
    CommentsConfig,
    cfg_to_dict,
    guess_local_data_dir,
    guess_local_output_dir,
    guess_team_dir,
    resolve_comments_data_dir,
    resolve_comments_scratch_output_dir,
)
from models import TextCNNClassifier, TextCNNConfig
from text_tokenizers import Vocab, build_char_vocab, build_word_vocab, encode_chars, encode_words, save_vocab


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
    (
        "PORTAL",
        re.compile(
            r"\b(?:espace_client|espace\s*client|application|app|login|mot\s*de\s*passe)\b|(?:فضاء\s*الزبون|فضاء\s*الزبائن|الولوج|حساب|كلمة\s*السر)",
            re.IGNORECASE,
        ),
    ),
    ("PRICES", re.compile(r"\b(?:prix|tarif|cher|promo|offre|offres)\b|(?:سعر|الاسعار|الأسعار|اسعار|العروض)", re.IGNORECASE)),
    (
        "OUTAGE",
        re.compile(
            r"\b(?:panne|coupure|down|hs|marche\s+pas|ne\s+marche\s+pas)\b|(?:مقطوع|انقطاع|ماكاش|مكانش|تقطع|مقطوعة)",
            re.IGNORECASE,
        ),
    ),
    ("SUPPORT", re.compile(r"\b(?:100|12|1500)\b")),
    ("SPEED", re.compile(r"\b(?:mbps|gbps)\b|(?:ميغا|جيغا)", re.IGNORECASE)),
    ("WAIT", re.compile(r"\b(?:a\s*quand|quand|depuis|attend|attente)\b|(?:وقتاش|متي|مزال|مازال|نستناو|نستنو|انتظار|راني\s+في|راه|راهم)", re.IGNORECASE)),
    ("LOCATION", re.compile(r"\b(?:wilaya|commune|quartier|centre|ville|rue)\b|(?:ولاية|بلدية|حي|شارع|مدينة|الجزائر|وهران|عنابة|قسنطينة|سطيف|بجاية|الشلف|الجلفة)", re.IGNORECASE)),
    ("MOBILE", re.compile(r"\b(?:3g|4g|5g)\b|(?:جيل\s*4|جيل\s*5)", re.IGNORECASE)),
    ("TOWER", re.compile(r"\b(?:antenne|tour|pylone|emetteur|emet(?:teur|trice))\b|(?:برج|أبراج|ابراج)", re.IGNORECASE)),
    ("PRAISE", re.compile(r"\b(?:bravo|merci|felicitations?|félicitations?)\b|(?:مبروك|بالتوفيق|👏|❤️)", re.IGNORECASE)),
    ("LAUGH", re.compile(r"(?:هههه+|lol|mdr|😂|🤣)", re.IGNORECASE)),
)


def _format_text(df: pd.DataFrame, cfg: CommentsConfig, *, add_meta: bool, add_flags: bool) -> list[str]:
    """
    Stable text formatting:
    - optional platform token
    - optional domain flags
    - raw cleaned text
    """

    txt = df[cfg.text_col].astype("string").fillna("")
    parts: list[pd.Series] = []

    if add_meta:
        plat = df[cfg.platform_col].astype("string").fillna("unk").str.lower()
        parts.append(("[PLATFORM] " + plat).astype("string"))

    if add_flags:
        flags_str = pd.Series([""] * len(df), index=df.index, dtype="string")
        for name, pat in _FLAG_SPECS:
            col = txt.str.contains(pat, regex=True, na=False).map(lambda b, _n=name: f"[{_n}]" if b else "")
            flags_str = (flags_str + " " + col).astype("string")
        parts.append(flags_str.str.strip())

    parts.append(("[TEXT] " + txt).astype("string"))
    out = parts[0]
    for p in parts[1:]:
        out = (out + " " + p).astype("string")
    return out.astype(str).tolist()


def _augment_words(text: str, *, drop_prob: float, rng: random.Random) -> str:
    if drop_prob <= 0:
        return text
    toks = str(text or "").split()
    if len(toks) <= 3:
        return text
    out = [t for t in toks if rng.random() > drop_prob]
    if len(out) < 2:
        return text
    return " ".join(out)


def _augment_chars(text: str, *, drop_prob: float, swap_prob: float, rng: random.Random) -> str:
    t = str(text or "")
    if not t:
        return t
    chars = list(t)
    if drop_prob > 0:
        kept = [ch for ch in chars if rng.random() > drop_prob]
        if len(kept) >= 2:
            chars = kept
    if swap_prob > 0 and len(chars) >= 4:
        # a few random adjacent swaps
        n_swaps = max(1, int(len(chars) * swap_prob * 0.1))
        for _ in range(n_swaps):
            i = rng.randrange(0, len(chars) - 1)
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def _make_optimizer(name: str, params, lr: float, weight_decay: float):
    import torch

    n = (name or "adam").lower().strip()
    if n == "adam":
        return torch.optim.Adam(params, lr=float(lr), weight_decay=float(weight_decay))
    if n == "nadam":
        return torch.optim.NAdam(params, lr=float(lr), weight_decay=float(weight_decay))
    raise ValueError(f"Unsupported optimizer: {name!r}. Use 'adam' or 'nadam'.")


def _cross_entropy_per_sample(logits, y, *, label_smoothing: float):
    import torch.nn.functional as F

    try:
        return F.cross_entropy(logits, y, reduction="none", label_smoothing=float(label_smoothing))
    except TypeError:
        # older torch without label_smoothing
        return F.cross_entropy(logits, y, reduction="none")


def _predict_proba_torch(model, dl, device: str, num_classes: int) -> np.ndarray:
    import torch

    model.eval()
    out = np.zeros((len(dl.dataset), int(num_classes)), dtype=np.float32)
    idx0 = 0
    with torch.no_grad():
        for batch in dl:
            x = batch["x"].to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float32)
            b = probs.shape[0]
            out[idx0 : idx0 + b] = probs
            idx0 += b
    return out


def _train_one_fold(
    *,
    X_texts: list[str],
    y: np.ndarray,
    sample_weight: np.ndarray,
    vocab: Vocab,
    tokenization: str,
    model_cfg: TextCNNConfig,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    seed: int,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    optimizer_name: str,
    label_smoothing: float,
    dropout: float,
    grad_clip: float,
    early_stopping_patience: int,
    augment_prob: float,
    word_drop_prob: float,
    char_drop_prob: float,
    char_swap_prob: float,
    pseudo_texts: list[str] | None = None,
    pseudo_y: np.ndarray | None = None,
    pseudo_w: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader, Dataset

    rng = random.Random(int(seed))

    def _encode(txt: str) -> list[int]:
        if tokenization == "word":
            return encode_words(txt, vocab, max_len=int(model_cfg.max_len))
        return encode_chars(txt, vocab, max_len=int(model_cfg.max_len))

    class TextDs(Dataset):
        def __init__(
            self,
            idx: np.ndarray,
            *,
            train: bool,
            pseudo_texts: list[str] | None = None,
            pseudo_y: np.ndarray | None = None,
            pseudo_w: np.ndarray | None = None,
        ):
            self.idx = idx.astype(int)
            self.train = bool(train)
            self.pseudo_texts = list(pseudo_texts or []) if self.train else []
            self.pseudo_y = np.asarray(pseudo_y, dtype=np.int64) if (self.train and pseudo_y is not None) else None
            self.pseudo_w = np.asarray(pseudo_w, dtype=np.float32) if (self.train and pseudo_w is not None) else None
            self.n_labeled = int(len(self.idx))

        def __len__(self) -> int:
            n_pseudo = int(len(self.pseudo_texts)) if self.train else 0
            return int(self.n_labeled + n_pseudo)

        def __getitem__(self, i: int) -> dict[str, Any]:
            if self.train and int(i) >= int(self.n_labeled):
                # pseudo-labeled row
                k = int(i) - int(self.n_labeled)
                if k < 0 or k >= len(self.pseudo_texts):
                    raise IndexError("Pseudo index out of range.")
                txt = self.pseudo_texts[k]
                yj = int(self.pseudo_y[k]) if self.pseudo_y is not None else 0
                wj = float(self.pseudo_w[k]) if self.pseudo_w is not None else 1.0
            else:
                # labeled row (from original train)
                j = int(self.idx[i])
                txt = X_texts[j]
                yj = int(y[j])
                wj = float(sample_weight[j])

            if self.train and float(augment_prob) > 0 and rng.random() < float(augment_prob):
                if tokenization == "word":
                    txt = _augment_words(txt, drop_prob=float(word_drop_prob), rng=rng)
                else:
                    txt = _augment_chars(txt, drop_prob=float(char_drop_prob), swap_prob=float(char_swap_prob), rng=rng)
            ids = _encode(txt)
            return {
                "x": torch.tensor(ids, dtype=torch.long),
                "y": torch.tensor(int(yj), dtype=torch.long),
                "w": torch.tensor(float(wj), dtype=torch.float32),
            }

    dl_tr = DataLoader(
        TextDs(train_idx, train=True, pseudo_texts=pseudo_texts, pseudo_y=pseudo_y, pseudo_w=pseudo_w),
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
    )
    dl_va = DataLoader(TextDs(val_idx, train=False), batch_size=int(batch_size), shuffle=False, num_workers=0)

    # Ensure model dropout matches
    model_cfg2 = TextCNNConfig(**{**asdict(model_cfg), "dropout": float(dropout)})
    model = TextCNNClassifier(model_cfg2).to(device)
    opt = _make_optimizer(optimizer_name, model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    best_state = None
    best_f1 = -1.0
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, int(epochs) + 1):
        model.train()
        losses: list[float] = []
        for batch in dl_tr:
            x = batch["x"].to(device)
            yy = batch["y"].to(device)
            ww = batch["w"].to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            per = _cross_entropy_per_sample(logits, yy, label_smoothing=float(label_smoothing))
            # Weighted mean (normalized) so changing weights doesn't implicitly change the effective LR.
            # This matters because experts use `off_group_weight`, and (optionally) pseudo-labels use `pseudo_weight`.
            denom = ww.sum().clamp_min(1e-9)
            loss = (per * ww).sum() / denom
            loss.backward()
            if float(grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
            opt.step()
            losses.append(float(loss.detach().cpu().item()))

        # Val
        proba = _predict_proba_torch(model, dl_va, device=device, num_classes=int(model_cfg.num_classes))
        pred = proba.argmax(axis=1)
        f1 = metrics.macro_f1_score(y[val_idx], pred, labels=list(range(int(model_cfg.num_classes))))

        if f1 > best_f1 + 1e-9:
            best_f1 = float(f1)
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if int(early_stopping_patience) > 0 and bad_epochs >= int(early_stopping_patience):
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state, strict=True)
    proba_val = _predict_proba_torch(model, dl_va, device=device, num_classes=int(model_cfg.num_classes))

    fold_metrics = {
        "best_f1": float(best_f1),
        "best_epoch": int(best_epoch),
        "epochs_ran": int(best_epoch if best_epoch else epochs),
    }
    save_blob = {
        "model_config": asdict(model_cfg2),
        "state_dict": best_state,
    }
    return fold_metrics, proba_val, save_blob


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--team_dir", type=str, default=None)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--train_filename", type=str, default="train.csv")
    p.add_argument("--test_filename", type=str, default="test_file.csv")
    p.add_argument("--output_dir", type=str, default=None)

    p.add_argument("--experts", type=str, nargs="*", default=["fr", "ar", "dz_ar", "dz_lat"])
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)

    # Formatting (meta + flags are important signals)
    p.add_argument("--add_meta", dest="add_meta", action="store_true")
    p.add_argument("--no_add_meta", dest="add_meta", action="store_false")
    # Default OFF: platform column is sometimes polluted in test (platform drift).
    p.set_defaults(add_meta=False)
    p.add_argument("--add_flags", dest="add_flags", action="store_true")
    p.add_argument("--no_add_flags", dest="add_flags", action="store_false")
    p.set_defaults(add_flags=True)

    # Specialization strength
    p.add_argument("--off_group_weight", type=float, default=0.25, help="Loss weight for samples outside expert group.")

    # Model hyperparams
    p.add_argument("--word_max_len", type=int, default=96)
    p.add_argument("--word_max_vocab", type=int, default=50_000)
    p.add_argument("--word_emb_dim", type=int, default=128)
    p.add_argument("--word_filters", type=int, default=128)
    p.add_argument("--word_kernels", type=str, default="3,4,5")
    p.add_argument("--word_hidden_dim", type=int, default=0)

    p.add_argument("--char_max_len", type=int, default=256)
    p.add_argument("--char_max_vocab", type=int, default=512)
    p.add_argument("--char_emb_dim", type=int, default=64)
    p.add_argument("--char_filters", type=int, default=128)
    p.add_argument("--char_kernels", type=str, default="3,4,5,6")
    p.add_argument("--char_hidden_dim", type=int, default=0)

    # Training hyperparams
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "nadam"])
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--early_stopping_patience", type=int, default=3)
    p.add_argument("--device", type=str, default=None)

    # Augmentation
    p.add_argument("--augment_prob", type=float, default=0.20)
    p.add_argument("--word_drop_prob", type=float, default=0.10)
    p.add_argument("--char_drop_prob", type=float, default=0.05)
    p.add_argument("--char_swap_prob", type=float, default=0.10)

    # Pseudo-labels (optional)
    p.add_argument("--pseudo_csv", type=str, default=None, help="Optional pseudo-labeled CSV (id,platform,comment,label[,confidence]).")
    p.add_argument("--pseudo_weight", type=float, default=0.4, help="Base loss weight for pseudo-labeled samples (0.2-0.6).")
    p.add_argument("--pseudo_min_conf", type=float, default=0.0, help="If pseudo CSV has 'confidence', keep only >= this value.")
    p.add_argument("--pseudo_use_confidence", action="store_true", help="If pseudo CSV has 'confidence', multiply weights by confidence.")
    p.add_argument("--pseudo_per_class_max", type=int, default=0, help="Optional cap per class (0 = unlimited).")

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
    schema.validate_file_exists(train_path)
    df_train_raw = schema.read_csv_robust(train_path)

    # Resolve output dir
    if args.output_dir:
        out_root = Path(args.output_dir)
    elif args.team_dir:
        out_root = resolve_comments_scratch_output_dir(Path(args.team_dir))
    else:
        out_root = guess_local_output_dir()
    out_root.mkdir(parents=True, exist_ok=True)

    run_dir = out_root / "models" / "experts_charcnn" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Cleaning (labeled train)
    df = preprocess.preprocess_comments_df(df_train_raw, cfg, is_train=True)
    if cfg.target_col not in df.columns:
        raise ValueError(f"Missing target column '{cfg.target_col}' after schema normalization.")

    label2id, id2label = schema.build_label_mapping(df[cfg.target_col])
    y_raw = df[cfg.target_col].astype(int).to_numpy()
    y = np.asarray([label2id[int(v)] for v in y_raw], dtype=np.int64)
    num_classes = int(len(label2id))

    add_meta = bool(args.add_meta)
    add_flags = bool(args.add_flags)
    X_texts = _format_text(df, cfg, add_meta=add_meta, add_flags=add_flags)

    # Group detection (for sample weighting) must be computed on the cleaned comment text
    # (NOT on the wrapped string that contains [PLATFORM]/[TEXT]/flags, which would bias ratios).
    base_texts = df[cfg.text_col].astype("string").fillna("").astype(str).tolist()
    groups = [lang.detect_group(t) for t in base_texts]

    # Optional pseudo-labeled data (kept OUT of validation folds; added only to training datasets)
    pseudo_texts: list[str] = []
    pseudo_y: np.ndarray | None = None
    pseudo_conf: np.ndarray | None = None
    pseudo_groups: list[str] = []
    if args.pseudo_csv:
        pth = Path(args.pseudo_csv)
        schema.validate_file_exists(pth)
        df_p_raw = schema.read_csv_robust(pth)
        df_p = schema.rename_raw_columns(df_p_raw, cfg, is_train=True)
        if "confidence" in df_p.columns and float(args.pseudo_min_conf) > 0:
            conf = pd.to_numeric(df_p["confidence"], errors="coerce").fillna(0.0)
            df_p = df_p[conf >= float(args.pseudo_min_conf)].copy()
        if len(df_p) > 0:
            df_p_clean = preprocess.preprocess_comments_df(df_p, cfg, is_train=True)
            # Keep only labels that exist in the labeled mapping (safety) and keep row alignment.
            known = set(int(k) for k in label2id.keys())
            y_ser = pd.to_numeric(df_p_clean[cfg.target_col], errors="coerce").astype("Int64")
            mask = y_ser.notna() & y_ser.astype(int).isin(known)
            df_p_clean = df_p_clean.loc[mask].copy()
            if len(df_p_clean) > 0:
                y_p_lbl = df_p_clean[cfg.target_col].astype(int).to_numpy()
                pseudo_y = np.asarray([label2id[int(v)] for v in y_p_lbl.tolist()], dtype=np.int64)
                pseudo_texts = _format_text(df_p_clean, cfg, add_meta=add_meta, add_flags=add_flags)
                # Optional confidence scaling
                if "confidence" in df_p_clean.columns:
                    pseudo_conf = pd.to_numeric(df_p_clean["confidence"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32)
                    pseudo_conf = np.clip(pseudo_conf, 0.0, 1.0)
                pseudo_groups = [lang.detect_group(t) for t in df_p_clean[cfg.text_col].astype(str).tolist()]

                # Optional per-class cap (balanced pseudo)
                if int(args.pseudo_per_class_max) > 0:
                    cap = int(args.pseudo_per_class_max)
                    keep_idx: list[int] = []
                    # If confidence exists, prefer highest confidence per class
                    conf_used = pseudo_conf if pseudo_conf is not None else np.ones((len(pseudo_texts),), dtype=np.float32)
                    for cls_id in range(num_classes):
                        idx = np.where(pseudo_y == int(cls_id))[0]
                        if len(idx) == 0:
                            continue
                        order = idx[np.argsort(conf_used[idx])[::-1]]
                        keep_idx.extend(order[:cap].tolist())
                    keep_idx = sorted(set(keep_idx))
                    pseudo_texts = [pseudo_texts[i] for i in keep_idx]
                    pseudo_y = pseudo_y[keep_idx]
                    if pseudo_conf is not None:
                        pseudo_conf = pseudo_conf[keep_idx]
                    pseudo_groups = [pseudo_groups[i] for i in keep_idx]

    # Save shared artifacts
    (run_dir / "label_mapping.json").write_text(
        json.dumps({"label2id": label2id, "id2label": id2label}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "preprocess_config.json").write_text(json.dumps(cfg_to_dict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")

    # Device
    if args.device:
        device = str(args.device)
    else:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    # CV splits
    skf = StratifiedKFold(n_splits=int(args.n_splits), shuffle=True, random_state=int(args.seed))

    experts = [e.strip() for e in (args.experts or []) if e and e.strip()]
    valid = {"fr", "ar", "dz_ar", "dz_lat"}
    unknown = [e for e in experts if e not in valid]
    if unknown:
        raise ValueError(f"Unknown experts: {unknown}. Valid: {sorted(valid)}")
    if not experts:
        raise ValueError("No experts selected.")

    result_summary: dict[str, Any] = {
        "task": "comments_scratch_experts_charcnn",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "train_path": str(train_path),
        "run_dir": str(run_dir),
        "n_splits": int(args.n_splits),
        "seed": int(args.seed),
        "formatting": {"add_meta": add_meta, "add_flags": add_flags},
        "experts": {},
    }

    for expert in experts:
        print(f"\n[train_experts] expert={expert}")
        _seed_everything(int(args.seed))

        tokenization = "word" if expert == "fr" else "char"
        expert_dir = run_dir / f"expert_{expert}"
        expert_dir.mkdir(parents=True, exist_ok=True)

        # Build vocab on full cleaned texts (unsupervised; stable across folds)
        if tokenization == "word":
            vocab = build_word_vocab(X_texts, max_vocab=int(args.word_max_vocab), min_freq=2)
            max_len = int(args.word_max_len)
            emb_dim = int(args.word_emb_dim)
            filters = int(args.word_filters)
            kernels = tuple(int(x) for x in str(args.word_kernels).split(",") if x.strip())
            hidden_dim = int(args.word_hidden_dim)
        else:
            vocab = build_char_vocab(X_texts, max_vocab=int(args.char_max_vocab), min_freq=1)
            max_len = int(args.char_max_len)
            emb_dim = int(args.char_emb_dim)
            filters = int(args.char_filters)
            kernels = tuple(int(x) for x in str(args.char_kernels).split(",") if x.strip())
            hidden_dim = int(args.char_hidden_dim)

        save_vocab(vocab, expert_dir / "vocab.json")

        model_cfg = TextCNNConfig(
            vocab_size=int(vocab.size),
            num_classes=int(num_classes),
            max_len=int(max_len),
            embedding_dim=int(emb_dim),
            num_filters=int(filters),
            kernel_sizes=kernels,
            hidden_dim=int(hidden_dim),
            dropout=float(args.dropout),
        )
        (expert_dir / "expert_config.json").write_text(
            json.dumps(
                {
                    "expert": expert,
                    "tokenization": tokenization,
                    "model": asdict(model_cfg),
                    "training": {
                        "epochs": int(args.epochs),
                        "batch_size": int(args.batch_size),
                        "lr": float(args.lr),
                        "weight_decay": float(args.weight_decay),
                        "optimizer": str(args.optimizer),
                        "dropout": float(args.dropout),
                        "label_smoothing": float(args.label_smoothing),
                        "grad_clip": float(args.grad_clip),
                        "early_stopping_patience": int(args.early_stopping_patience),
                        "augment_prob": float(args.augment_prob),
                        "word_drop_prob": float(args.word_drop_prob),
                        "char_drop_prob": float(args.char_drop_prob),
                        "char_swap_prob": float(args.char_swap_prob),
                        "off_group_weight": float(args.off_group_weight),
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        # Sample weights: specialize to detected group
        target_group = expert
        sw = np.full((len(X_texts),), float(args.off_group_weight), dtype=np.float32)
        for i, g in enumerate(groups):
            if g == target_group:
                sw[i] = 1.0

        # Pseudo weights for this expert (optional)
        p_texts = pseudo_texts
        p_y = pseudo_y
        p_w = None
        if p_texts and p_y is not None and len(p_texts) == len(p_y):
            base_w = np.full((len(p_texts),), float(args.pseudo_weight), dtype=np.float32)
            if bool(args.pseudo_use_confidence) and pseudo_conf is not None and len(pseudo_conf) == len(base_w):
                base_w = base_w * pseudo_conf
            # Expert specialization (same idea as labeled): downweight off-group pseudo
            p_w = base_w.copy()
            for i, g in enumerate(pseudo_groups):
                if g != target_group:
                    p_w[i] *= float(args.off_group_weight)

        oof_proba = np.zeros((len(X_texts), int(num_classes)), dtype=np.float32)
        fold_scores: list[float] = []
        fold_details: list[dict[str, Any]] = []

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_texts, y), start=1):
            fold_dir = expert_dir / f"fold{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            fold_metrics, proba_val, save_blob = _train_one_fold(
                X_texts=X_texts,
                y=y,
                sample_weight=sw,
                vocab=vocab,
                tokenization=tokenization,
                model_cfg=model_cfg,
                train_idx=tr_idx,
                val_idx=va_idx,
                seed=int(args.seed) + fold,
                device=device,
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                optimizer_name=str(args.optimizer),
                label_smoothing=float(args.label_smoothing),
                dropout=float(args.dropout),
                grad_clip=float(args.grad_clip),
                early_stopping_patience=int(args.early_stopping_patience),
                augment_prob=float(args.augment_prob),
                word_drop_prob=float(args.word_drop_prob),
                char_drop_prob=float(args.char_drop_prob),
                char_swap_prob=float(args.char_swap_prob),
                pseudo_texts=p_texts,
                pseudo_y=p_y,
                pseudo_w=p_w,
            )

            oof_proba[va_idx] = proba_val
            pred = proba_val.argmax(axis=1)
            f1 = metrics.macro_f1_score(y[va_idx], pred, labels=list(range(int(num_classes))))
            fold_scores.append(float(f1))
            fold_details.append({"fold": int(fold), "macro_f1": float(f1), **fold_metrics})
            print(f"[expert={expert}] fold={fold} macro_f1={float(f1):.5f}")

            # Save fold model
            import torch

            torch.save(save_blob, fold_dir / "model.pt")
            (fold_dir / "metrics.json").write_text(json.dumps(fold_details[-1], ensure_ascii=False, indent=2), encoding="utf-8")

        # OOF score
        oof_pred = oof_proba.argmax(axis=1)
        oof_f1 = metrics.macro_f1_score(y, oof_pred, labels=list(range(int(num_classes))))
        np.save(expert_dir / "oof_proba.npy", oof_proba.astype(np.float32))

        expert_summary = {
            "expert": expert,
            "tokenization": tokenization,
            "device": device,
            "vocab_size": int(vocab.size),
            "oof_macro_f1": float(oof_f1),
            "fold_scores": fold_scores,
            "fold_details": fold_details,
        }
        (expert_dir / "run_summary.json").write_text(json.dumps(expert_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        result_summary["experts"][expert] = expert_summary
        print(f"[expert={expert}] OOF macro_f1={float(oof_f1):.5f}")

    (run_dir / "run_summary.json").write_text(json.dumps(result_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved run -> {run_dir}")


if __name__ == "__main__":
    main()


