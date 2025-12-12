"""
Predict + merge for comments-scratch.

This script:
1) loads a scratch experts run dir (created by train_experts.py)
2) predicts probabilities on test_file.csv using the expert fold ensemble
3) optionally predicts probabilities from one or more transformer run dirs
4) merges (gated scratch + weighted transformer) and writes Kaggle submission (id,Class)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import lang
import preprocess
import schema
from config import CommentsConfig, cfg_from_dict, guess_local_data_dir, guess_team_dir, resolve_comments_data_dir
from models import TextCNNClassifier, TextCNNConfig
from text_tokenizers import load_vocab, encode_chars, encode_words


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------- scratch experts -------------------------


def _format_scratch_text(df: pd.DataFrame, cfg: CommentsConfig, *, add_meta: bool, add_flags: bool) -> list[str]:
    """
    Must match train_experts.py formatting.
    """

    import re as _re

    _FLAG_SPECS: tuple[tuple[str, _re.Pattern[str]], ...] = (
        ("FIBRE", _re.compile(r"\b(?:fibre|ftth|fttx|idoom_fibre)\b", _re.IGNORECASE)),
        ("MODEM", _re.compile(r"\b(?:modem|wifi6)\b", _re.IGNORECASE)),
        (
            "PORTAL",
            _re.compile(
                r"\b(?:espace_client|espace\s*client|application|app|login|mot\s*de\s*passe)\b|(?:فضاء\s*الزبون|فضاء\s*الزبائن|الولوج|حساب|كلمة\s*السر)",
                _re.IGNORECASE,
            ),
        ),
        ("PRICES", _re.compile(r"\b(?:prix|tarif|cher|promo|offre|offres)\b|(?:سعر|الاسعار|الأسعار|اسعار|العروض)", _re.IGNORECASE)),
        (
            "OUTAGE",
            _re.compile(
                r"\b(?:panne|coupure|down|hs|marche\s+pas|ne\s+marche\s+pas)\b|(?:مقطوع|انقطاع|ماكاش|مكانش|تقطع|مقطوعة)",
                _re.IGNORECASE,
            ),
        ),
        ("SUPPORT", _re.compile(r"\b(?:100|12|1500)\b")),
        ("SPEED", _re.compile(r"\b(?:mbps|gbps)\b|(?:ميغا|جيغا)", _re.IGNORECASE)),
        ("WAIT", _re.compile(r"\b(?:a\s*quand|quand|depuis|attend|attente)\b|(?:وقتاش|متي|مزال|مازال|نستناو|نستنو|انتظار|راني\s+في|راه|راهم)", _re.IGNORECASE)),
        ("LOCATION", _re.compile(r"\b(?:wilaya|commune|quartier|centre|ville|rue)\b|(?:ولاية|بلدية|حي|شارع|مدينة|الجزائر|وهران|عنابة|قسنطينة|سطيف|بجاية|الشلف|الجلفة)", _re.IGNORECASE)),
        ("MOBILE", _re.compile(r"\b(?:3g|4g|5g)\b|(?:جيل\s*4|جيل\s*5)", _re.IGNORECASE)),
        ("TOWER", _re.compile(r"\b(?:antenne|tour|pylone|emetteur|emet(?:teur|trice))\b|(?:برج|أبراج|ابراج)", _re.IGNORECASE)),
        ("PRAISE", _re.compile(r"\b(?:bravo|merci|felicitations?|félicitations?)\b|(?:مبروك|بالتوفيق|👏|❤️)", _re.IGNORECASE)),
        ("LAUGH", _re.compile(r"(?:هههه+|lol|mdr|😂|🤣)", _re.IGNORECASE)),
    )

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


def _predict_proba_scratch_expert(
    *,
    expert_dir: Path,
    texts: list[str],
    device: str,
    batch_size: int,
    use_folds: bool,
) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader, Dataset

    cfg_path = expert_dir / "expert_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing expert_config.json: {cfg_path}")
    cfg = _read_json(cfg_path)

    tokenization = str(cfg.get("tokenization", "char"))
    vocab = load_vocab(expert_dir / "vocab.json")

    def _encode(txt: str, max_len: int) -> list[int]:
        if tokenization == "word":
            return encode_words(txt, vocab, max_len=max_len)
        return encode_chars(txt, vocab, max_len=max_len)

    fold_dirs: list[Path] = []
    candidates = sorted([d for d in expert_dir.iterdir() if d.is_dir() and d.name.startswith("fold")])
    if use_folds:
        fold_dirs = candidates
    else:
        # Use the first fold only (fast debug mode)
        if candidates:
            fold_dirs = [candidates[0]]

    if not fold_dirs:
        # fallback: allow using a single model.pt at expert root (if added later)
        if (expert_dir / "model.pt").exists():
            fold_dirs = [expert_dir]
        else:
            raise FileNotFoundError(f"No fold dirs found in expert dir: {expert_dir}")

    proba_sum: np.ndarray | None = None
    n_loaded = 0
    for fd in fold_dirs:
        model_path = fd / "model.pt"
        if not model_path.exists():
            continue
        blob = torch.load(model_path, map_location="cpu")
        mcfg = blob.get("model_config")
        if not isinstance(mcfg, dict):
            raise ValueError(f"Invalid model.pt (missing model_config) at: {model_path}")
        model_cfg = TextCNNConfig(**mcfg)
        model = TextCNNClassifier(model_cfg).to(device)
        model.load_state_dict(blob["state_dict"], strict=True)
        model.eval()

        max_len = int(model_cfg.max_len)
        num_classes = int(model_cfg.num_classes)

        class TextDs(Dataset):
            def __init__(self, _texts: list[str]):
                self.texts = _texts

            def __len__(self) -> int:
                return len(self.texts)

            def __getitem__(self, idx: int) -> dict[str, Any]:
                ids = _encode(self.texts[idx], max_len=max_len)
                return {"x": torch.tensor(ids, dtype=torch.long)}

        dl = DataLoader(TextDs(texts), batch_size=int(batch_size), shuffle=False, num_workers=0)

        out = np.zeros((len(texts), num_classes), dtype=np.float32)
        i0 = 0
        with torch.no_grad():
            for batch in dl:
                x = batch["x"].to(device)
                logits = model(x)
                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float32)
                b = probs.shape[0]
                out[i0 : i0 + b] = probs
                i0 += b

        proba_sum = out if proba_sum is None else (proba_sum + out)
        n_loaded += 1

    if proba_sum is None or n_loaded <= 0:
        raise RuntimeError(f"No models loaded from expert dir: {expert_dir}")

    return proba_sum / float(n_loaded)


# ------------------------- transformers (optional) -------------------------


def _has_saved_weights(model_dir: Path) -> bool:
    return any(
        (model_dir / name).exists() for name in ("model.safetensors", "pytorch_model.bin", "pytorch_model.safetensors")
    )


def _extract_fold_number(dir_name: str) -> int | None:
    m = re.search(r"_fold(\d+)\b", dir_name)
    return int(m.group(1)) if m else None


def _expand_transformer_run_dir(run_dir: Path) -> list[Path]:
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")
    fold_dirs = []
    for d in run_dir.iterdir():
        if not d.is_dir():
            continue
        if _extract_fold_number(d.name) is None:
            continue
        if (d / "config.json").exists() and _has_saved_weights(d):
            fold_dirs.append(d)
    fold_dirs.sort(key=lambda p: (_extract_fold_number(p.name) or 10_000, p.name))
    if not fold_dirs:
        raise FileNotFoundError(f"No complete fold checkpoints found inside run dir: {run_dir}")
    return fold_dirs


def _load_label_mapping(mapping_path: Path) -> tuple[dict[int, int], dict[int, int]]:
    data = json.loads(mapping_path.read_text(encoding="utf-8"))
    label2id = {int(k): int(v) for k, v in data["label2id"].items()}
    id2label = {int(k): int(v) for k, v in data["id2label"].items()}
    return label2id, id2label


def _load_label_mapping_from_transformer_config(model_dir: Path) -> tuple[dict[int, int], dict[int, int]]:
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.json in transformer dir: {model_dir}")
    data = json.loads(cfg_path.read_text(encoding="utf-8"))

    id2label_raw = data.get("id2label") or {}
    label2id_raw = data.get("label2id") or {}

    id2label: dict[int, int] = {}
    for k, v in id2label_raw.items():
        ik = int(k)
        try:
            iv = int(v)
        except Exception:
            raise ValueError(f"Non-numeric id2label value in config.json: {v!r}")
        id2label[ik] = iv

    if label2id_raw:
        label2id = {int(k): int(v) for k, v in label2id_raw.items()}
    else:
        label2id = {lbl: i for i, lbl in id2label.items()}

    if not id2label or not label2id:
        raise ValueError(f"Could not build label mapping from config.json in {model_dir}")
    return label2id, id2label


def _load_label_mapping_for_transformer_dir(model_dir: Path) -> tuple[dict[int, int], dict[int, int]]:
    candidates = [model_dir / "label_mapping.json", model_dir.parent / "label_mapping.json"]
    for c in candidates:
        if c.exists():
            return _load_label_mapping(c)
    return _load_label_mapping_from_transformer_config(model_dir)


def _align_proba_to_labels(proba: np.ndarray, id2label: dict[int, int], target_labels_in_order: list[int]) -> np.ndarray:
    labels_in_this_model = [id2label[i] for i in sorted(id2label.keys())]
    if labels_in_this_model == target_labels_in_order:
        return proba
    idx_map = {lbl: j for j, lbl in enumerate(labels_in_this_model)}
    missing = [lbl for lbl in target_labels_in_order if lbl not in idx_map]
    if missing:
        raise ValueError(f"Model label mapping mismatch. Missing labels: {missing}")
    reordered = np.zeros((proba.shape[0], len(target_labels_in_order)), dtype=np.float32)
    for j, lbl in enumerate(target_labels_in_order):
        reordered[:, j] = proba[:, idx_map[lbl]]
    return reordered


def _try_load_transformer_run_settings(run_dir: Path) -> dict[str, Any] | None:
    p = run_dir / "run_summary.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return {
            "add_meta": bool(data.get("add_meta", False)),
            "add_flags": bool(data.get("add_flags", False)),
            "max_length": int(data.get("max_length", 128)),
        }
    except Exception:
        return None


def _format_transformer_text(
    df: pd.DataFrame,
    cfg: CommentsConfig,
    *,
    add_meta: bool,
    add_flags: bool,
    allowed_special_tokens: set[str] | None,
) -> list[str]:
    """
    Matches `forca-hack/comments/src/predict.py` formatting so we can reuse transformer checkpoints safely.
    """

    import re as _re

    txt = df[cfg.text_col].astype("string").fillna("")
    parts: list[pd.Series] = []

    if add_meta:
        plat = df[cfg.platform_col].astype("string").fillna("unk").str.lower()
        if allowed_special_tokens is None or "[PLATFORM]" in allowed_special_tokens:
            parts.append(("[PLATFORM] " + plat).astype("string"))

    if add_flags:
        flag_specs: tuple[tuple[str, _re.Pattern[str]], ...] = (
            ("FIBRE", _re.compile(r"\b(?:fibre|ftth|fttx|idoom_fibre)\b", _re.IGNORECASE)),
            ("MODEM", _re.compile(r"\b(?:modem|wifi6)\b", _re.IGNORECASE)),
            (
                "PORTAL",
                _re.compile(
                    r"\b(?:espace_client|espace\s*client|application|app|login|mot\s*de\s*passe)\b|(?:فضاء\s*الزبون|فضاء\s*الزبائن|الولوج|حساب|كلمة\s*السر)",
                    _re.IGNORECASE,
                ),
            ),
            ("PRICES", _re.compile(r"\b(?:prix|tarif|cher|promo|offre|offres)\b|(?:سعر|الاسعار|الأسعار|اسعار|العروض)", _re.IGNORECASE)),
            (
                "OUTAGE",
                _re.compile(
                    r"\b(?:panne|coupure|down|hs|marche\s+pas|ne\s+marche\s+pas)\b|(?:مقطوع|انقطاع|ماكاش|مكانش|تقطع|مقطوعة)",
                    _re.IGNORECASE,
                ),
            ),
            ("SUPPORT", _re.compile(r"\b(?:100|12|1500)\b")),
            ("SPEED", _re.compile(r"\b(?:mbps|gbps)\b|(?:ميغا|جيغا)", _re.IGNORECASE)),
            ("WAIT", _re.compile(r"\b(?:a\s*quand|quand|depuis|attend|attente)\b|(?:وقتاش|متي|مزال|مازال|نستناو|نستنو|انتظار|راني\s+في|راه|راهم)", _re.IGNORECASE)),
            ("LOCATION", _re.compile(r"\b(?:wilaya|commune|quartier|centre|ville|rue)\b|(?:ولاية|بلدية|حي|شارع|مدينة|الجزائر|وهران|عنابة|قسنطينة|سطيف|بجاية|الشلف|الجلفة)", _re.IGNORECASE)),
            ("MOBILE", _re.compile(r"\b(?:3g|4g|5g)\b|(?:جيل\s*4|جيل\s*5)", _re.IGNORECASE)),
            ("TOWER", _re.compile(r"\b(?:antenne|tour|pylone|emetteur|emet(?:teur|trice))\b|(?:برج|أبراج|ابراج)", _re.IGNORECASE)),
            ("PRAISE", _re.compile(r"\b(?:bravo|merci|felicitations?|félicitations?)\b|(?:مبروك|بالتوفيق|👏|❤️)", _re.IGNORECASE)),
            ("LAUGH", _re.compile(r"(?:هههه+|lol|mdr|😂|🤣)", _re.IGNORECASE)),
        )

        flags_str = pd.Series([""] * len(df), index=df.index, dtype="string")
        for name, pat in flag_specs:
            tok = f"[{name}]"
            if allowed_special_tokens is not None and tok not in allowed_special_tokens:
                continue
            col = txt.str.contains(pat, regex=True, na=False).map(lambda b, _n=name: f"[{_n}]" if b else "")
            flags_str = (flags_str + " " + col).astype("string")
        parts.append(flags_str.str.strip())

    if allowed_special_tokens is None or "[TEXT]" in allowed_special_tokens:
        parts.append(("[TEXT] " + txt).astype("string"))
    else:
        parts.append(txt.astype("string"))

    out = parts[0] if parts else pd.Series([""] * len(df), index=df.index, dtype="string")
    for p in parts[1:]:
        out = (out + " " + p).astype("string")
    return out.astype(str).tolist()


def _predict_proba_transformer_fold(
    *,
    model_dir: Path,
    df_clean: pd.DataFrame,
    cfg: CommentsConfig,
    max_length: int,
    batch_size: int,
    device: str | None,
    add_meta: bool,
    add_flags: bool,
) -> tuple[np.ndarray, dict[int, int]]:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Missing transformers/torch. Install with: pip install transformers torch") from e

    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model.eval()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    allowed_special = set(getattr(tok, "additional_special_tokens", []) or [])
    allowed_special = allowed_special if len(allowed_special) > 0 else None
    texts = _format_transformer_text(
        df_clean,
        cfg,
        add_meta=bool(add_meta),
        add_flags=bool(add_flags),
        allowed_special_tokens=allowed_special,
    )

    n = len(texts)
    num_labels = int(getattr(model.config, "num_labels", 0)) or 1
    out = np.zeros((n, num_labels), dtype=np.float32)

    for i in range(0, n, int(batch_size)):
        batch = texts[i : i + int(batch_size)]
        enc = tok(batch, truncation=True, max_length=int(max_length), padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float32)
        out[i : i + len(batch)] = probs

    _, id2label = _load_label_mapping_for_transformer_dir(model_dir)
    return out, id2label


def _predict_proba_transformers(
    *,
    run_dirs: list[Path],
    df_clean: pd.DataFrame,
    cfg: CommentsConfig,
    target_labels_in_order: list[int],
    batch_size: int,
    device: str | None,
    override_add_meta: bool | None,
    override_add_flags: bool | None,
    override_max_length: int | None,
    weights: list[float] | None,
) -> np.ndarray:
    model_dirs: list[Path] = []
    run_settings: list[dict[str, Any] | None] = []
    for rd in run_dirs:
        model_dirs.extend(_expand_transformer_run_dir(rd))
        run_settings.append(_try_load_transformer_run_settings(rd))

    if not model_dirs:
        raise ValueError("No transformer fold dirs found.")

    if weights is not None and len(weights) != len(model_dirs):
        raise ValueError("If provided, transformer --weights must match number of expanded fold dirs.")

    w_list = weights if weights is not None else [1.0] * len(model_dirs)
    w_sum = float(sum(w_list))

    proba_sum: np.ndarray | None = None
    # Per-fold inference: use run settings inferred from the parent run dir when possible
    for w, md in zip(w_list, model_dirs, strict=True):
        # Guess settings from the nearest run_dir run_summary.json (md.parent)
        s = _try_load_transformer_run_settings(md.parent) or {}
        add_meta = bool(s.get("add_meta", False)) if override_add_meta is None else bool(override_add_meta)
        add_flags = bool(s.get("add_flags", False)) if override_add_flags is None else bool(override_add_flags)
        max_len = int(s.get("max_length", 128)) if override_max_length is None else int(override_max_length)

        proba, id2label = _predict_proba_transformer_fold(
            model_dir=md,
            df_clean=df_clean,
            cfg=cfg,
            max_length=max_len,
            batch_size=int(batch_size),
            device=device,
            add_meta=add_meta,
            add_flags=add_flags,
        )
        proba = _align_proba_to_labels(proba, id2label, target_labels_in_order)
        proba = proba * float(w)
        proba_sum = proba if proba_sum is None else (proba_sum + proba)

    if proba_sum is None:
        raise RuntimeError("No transformer predictions produced.")
    return proba_sum / max(w_sum, 1e-9)


# ------------------------- merging -------------------------


def _merge_scratch(
    *,
    proba_by_expert: dict[str, np.ndarray],
    base_texts: list[str],
    method: str,
) -> np.ndarray:
    experts = tuple(proba_by_expert.keys())
    n = next(iter(proba_by_expert.values())).shape[0]
    k = next(iter(proba_by_expert.values())).shape[1]

    if method == "avg":
        proba = np.zeros((n, k), dtype=np.float32)
        for e in experts:
            proba += proba_by_expert[e].astype(np.float32)
        return proba / float(len(experts))

    if method != "gated":
        raise ValueError(f"Unknown scratch merge method: {method!r}. Use 'avg' or 'gated'.")

    # Gated mixture-of-experts
    out = np.zeros((n, k), dtype=np.float32)
    for i, txt in enumerate(base_texts):
        w = lang.gating_weights(txt, experts=experts)
        row = np.zeros((k,), dtype=np.float32)
        for e in experts:
            row += float(w[e]) * proba_by_expert[e][i].astype(np.float32)
        out[i] = row
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--team_dir", type=str, default=None)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--test_filename", type=str, default="test_file.csv")

    p.add_argument("--scratch_run_dir", type=str, required=True)
    p.add_argument("--scratch_experts", type=str, nargs="*", default=["fr", "ar", "dz_ar", "dz_lat"])
    p.add_argument("--scratch_use_folds", dest="scratch_use_folds", action="store_true")
    p.add_argument("--no_scratch_use_folds", dest="scratch_use_folds", action="store_false", help="Disable fold ensembling (use first fold only).")
    p.set_defaults(scratch_use_folds=True)
    p.add_argument("--scratch_batch_size", type=int, default=256)
    p.add_argument("--scratch_device", type=str, default=None)
    p.add_argument("--scratch_merge", type=str, default="gated", choices=["avg", "gated"])

    # Optional transformer merge
    p.add_argument("--transformer_run_dirs", type=str, nargs="*", default=None)
    p.add_argument("--transformer_weights", type=float, nargs="*", default=None)
    p.add_argument("--transformer_batch_size", type=int, default=32)
    p.add_argument("--transformer_device", type=str, default=None)
    p.add_argument("--transformer_add_meta", action="store_true", default=None)
    p.add_argument("--transformer_no_add_meta", action="store_true", default=False)
    p.add_argument("--transformer_add_flags", action="store_true", default=None)
    p.add_argument("--transformer_no_add_flags", action="store_true", default=False)
    p.add_argument("--transformer_max_length", type=int, default=None)
    p.add_argument("--w_transformer", type=float, default=0.75)

    p.add_argument("--output_csv", type=str, default="submission.csv")
    p.add_argument("--target_col_name", type=str, default="Class")
    p.add_argument("--save_proba_npy", type=str, default=None)
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
    df_test_raw = schema.read_csv_robust(test_path)

    # Prefer preprocess config from scratch run when present (reproducibility)
    scratch_run = Path(args.scratch_run_dir)
    cfg_path = scratch_run / "preprocess_config.json"
    if cfg_path.exists():
        cfg = cfg_from_dict(json.loads(cfg_path.read_text(encoding="utf-8")))

    df_test_clean = preprocess.preprocess_comments_df(df_test_raw, cfg, is_train=False)
    base_texts = df_test_clean[cfg.text_col].astype("string").fillna("").astype(str).tolist()

    # Load scratch run formatting defaults
    # Default OFF: platform column can be noisy in test.
    add_meta = False
    add_flags = True
    rs_path = scratch_run / "run_summary.json"
    if rs_path.exists():
        try:
            rs = json.loads(rs_path.read_text(encoding="utf-8"))
            fmt = rs.get("formatting") or {}
            add_meta = bool(fmt.get("add_meta", True))
            add_flags = bool(fmt.get("add_flags", True))
        except Exception:
            pass

    scratch_texts = _format_scratch_text(df_test_clean, cfg, add_meta=add_meta, add_flags=add_flags)

    # Device for scratch
    if args.scratch_device:
        scratch_device = str(args.scratch_device)
    else:
        try:
            import torch

            scratch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            scratch_device = "cpu"

    # Determine label order from scratch run mapping (shared by all experts)
    map_path = scratch_run / "label_mapping.json"
    if not map_path.exists():
        raise FileNotFoundError(f"Missing label_mapping.json in scratch run: {map_path}")
    _, id2label = _load_label_mapping(map_path)
    target_labels_in_order = [id2label[i] for i in sorted(id2label.keys())]

    # Predict each expert
    selected = [e.strip() for e in (args.scratch_experts or []) if e and e.strip()]
    proba_by_expert: dict[str, np.ndarray] = {}
    for expert in selected:
        expert_dir = scratch_run / f"expert_{expert}"
        if not expert_dir.exists():
            raise FileNotFoundError(f"Scratch expert dir not found: {expert_dir}")
        use_folds = bool(args.scratch_use_folds)
        proba = _predict_proba_scratch_expert(
            expert_dir=expert_dir,
            texts=scratch_texts,
            device=scratch_device,
            batch_size=int(args.scratch_batch_size),
            use_folds=use_folds,
        )
        proba_by_expert[expert] = proba.astype(np.float32)

    proba_scratch = _merge_scratch(proba_by_expert=proba_by_expert, base_texts=base_texts, method=str(args.scratch_merge))

    # Optional transformers
    proba_final = proba_scratch
    if args.transformer_run_dirs and len(args.transformer_run_dirs) > 0:
        run_dirs = [Path(x) for x in args.transformer_run_dirs]
        weights = list(args.transformer_weights) if args.transformer_weights else None

        proba_tr = _predict_proba_transformers(
            run_dirs=run_dirs,
            df_clean=df_test_clean,
            cfg=cfg,
            target_labels_in_order=target_labels_in_order,
            batch_size=int(args.transformer_batch_size),
            device=args.transformer_device,
            override_add_meta=(False if bool(args.transformer_no_add_meta) else args.transformer_add_meta),
            override_add_flags=(False if bool(args.transformer_no_add_flags) else args.transformer_add_flags),
            override_max_length=args.transformer_max_length,
            weights=weights,
        ).astype(np.float32)

        wt = float(args.w_transformer)
        wt = min(max(wt, 0.0), 1.0)
        proba_final = (wt * proba_tr) + ((1.0 - wt) * proba_scratch)

    pred_id = proba_final.argmax(axis=1)
    pred_label = [target_labels_in_order[int(i)] for i in pred_id]

    sub = pd.DataFrame({cfg.id_col: df_test_clean[cfg.id_col].astype(int), args.target_col_name: pred_label})
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Saved submission -> {out_csv} (rows={len(sub)})")

    if args.save_proba_npy:
        out_npy = Path(args.save_proba_npy)
        out_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_npy, proba_final.astype(np.float32))
        print(f"Saved proba -> {out_npy}")


if __name__ == "__main__":
    main()


