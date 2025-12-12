"""
Optimize merging weights (OOF) between:
- scratch experts (from train_experts.py) and
- one or more transformer runs (from forca-hack/comments/src/train_transformer.py)

What it does
------------
1) Loads scratch experts OOF probabilities from: <scratch_run>/expert_*/oof_proba.npy
2) Recomputes transformer OOF probabilities by running each fold model on its validation fold
   (CV splits reconstructed from run_summary.json: seed + n_splits)
3) Grid-searches w_transformer in [0..1] to maximize OOF Macro-F1:
     proba = w * proba_transformer + (1-w) * proba_scratch

This gives an evidence-based default for `--w_transformer` in `predict.py`.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import lang
import metrics
import preprocess
import schema
from config import CommentsConfig, cfg_from_dict, guess_local_data_dir, guess_team_dir, resolve_comments_data_dir


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_label_mapping(mapping_path: Path) -> tuple[dict[int, int], dict[int, int]]:
    data = _read_json(mapping_path)
    label2id = {int(k): int(v) for k, v in data["label2id"].items()}
    id2label = {int(k): int(v) for k, v in data["id2label"].items()}
    return label2id, id2label


def _align_proba_to_labels(proba: np.ndarray, id2label: dict[int, int], target_labels_in_order: list[int]) -> np.ndarray:
    labels_in_this_model = [id2label[i] for i in sorted(id2label.keys())]
    if labels_in_this_model == target_labels_in_order:
        return proba
    idx_map = {lbl: j for j, lbl in enumerate(labels_in_this_model)}
    reordered = np.zeros((proba.shape[0], len(target_labels_in_order)), dtype=np.float32)
    for j, lbl in enumerate(target_labels_in_order):
        reordered[:, j] = proba[:, idx_map[lbl]]
    return reordered


def _extract_fold_number(dir_name: str) -> int | None:
    m = re.search(r"_fold(\d+)\b", dir_name)
    return int(m.group(1)) if m else None


def _has_saved_weights(model_dir: Path) -> bool:
    return any(
        (model_dir / name).exists() for name in ("model.safetensors", "pytorch_model.bin", "pytorch_model.safetensors")
    )


def _expand_transformer_run_dir(run_dir: Path) -> dict[int, Path]:
    """
    Return {fold_number: fold_dir}
    """

    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")
    out: dict[int, Path] = {}
    for d in run_dir.iterdir():
        if not d.is_dir():
            continue
        fold = _extract_fold_number(d.name)
        if fold is None:
            continue
        if (d / "config.json").exists() and _has_saved_weights(d):
            out[int(fold)] = d
    if not out:
        raise FileNotFoundError(f"No complete fold checkpoints found inside run dir: {run_dir}")
    return out


def _load_label_mapping_from_transformer_config(model_dir: Path) -> tuple[dict[int, int], dict[int, int]]:
    cfg_path = model_dir / "config.json"
    data = _read_json(cfg_path)
    id2label_raw = data.get("id2label") or {}
    label2id_raw = data.get("label2id") or {}

    id2label: dict[int, int] = {}
    for k, v in id2label_raw.items():
        id2label[int(k)] = int(v)

    if label2id_raw:
        label2id = {int(k): int(v) for k, v in label2id_raw.items()}
    else:
        label2id = {lbl: i for i, lbl in id2label.items()}

    return label2id, id2label


def _load_label_mapping_for_transformer_dir(model_dir: Path) -> tuple[dict[int, int], dict[int, int]]:
    candidates = [model_dir / "label_mapping.json", model_dir.parent / "label_mapping.json"]
    for c in candidates:
        if c.exists():
            return _load_label_mapping(c)
    return _load_label_mapping_from_transformer_config(model_dir)


def _format_transformer_text(
    df: pd.DataFrame,
    cfg: CommentsConfig,
    *,
    add_meta: bool,
    add_flags: bool,
    allowed_special_tokens: set[str] | None,
) -> list[str]:
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


def _predict_proba_transformer_texts(*, model_dir: Path, texts: list[str], max_length: int, batch_size: int, device: str | None) -> np.ndarray:
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
    return out


def _load_scratch_oof(*, scratch_run: Path, experts: list[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for e in experts:
        p = scratch_run / f"expert_{e}" / "oof_proba.npy"
        if not p.exists():
            raise FileNotFoundError(f"Missing scratch OOF proba for expert={e}: {p}")
        out[e] = np.load(p).astype(np.float32)
    return out


def _merge_scratch_oof(*, proba_by_expert: dict[str, np.ndarray], base_texts: list[str], method: str) -> np.ndarray:
    experts = tuple(proba_by_expert.keys())
    n, k = next(iter(proba_by_expert.values())).shape

    if method == "avg":
        proba = np.zeros((n, k), dtype=np.float32)
        for e in experts:
            proba += proba_by_expert[e]
        return proba / float(len(experts))

    if method != "gated":
        raise ValueError(f"Unknown scratch merge method: {method!r}. Use 'avg' or 'gated'.")

    out = np.zeros((n, k), dtype=np.float32)
    for i, txt in enumerate(base_texts):
        w = lang.gating_weights(txt, experts=experts)
        row = np.zeros((k,), dtype=np.float32)
        for e in experts:
            row += float(w[e]) * proba_by_expert[e][i]
        out[i] = row
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--team_dir", type=str, default=None)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--train_filename", type=str, default="train.csv")

    p.add_argument("--scratch_run_dir", type=str, required=True)
    p.add_argument("--scratch_experts", type=str, nargs="*", default=["fr", "ar", "dz_ar", "dz_lat"])
    p.add_argument("--scratch_merge", type=str, default="gated", choices=["avg", "gated"])

    p.add_argument("--transformer_run_dirs", type=str, nargs="*", default=None)
    p.add_argument("--transformer_batch_size", type=int, default=32)
    p.add_argument("--transformer_device", type=str, default=None)
    p.add_argument("--w_step", type=float, default=0.05)
    args = p.parse_args()

    scratch_run = Path(args.scratch_run_dir)
    if not scratch_run.exists():
        raise FileNotFoundError(f"scratch_run_dir not found: {scratch_run}")

    # Prefer preprocess config from scratch run
    cfg = CommentsConfig()
    cfg_path = scratch_run / "preprocess_config.json"
    if cfg_path.exists():
        cfg = cfg_from_dict(json.loads(cfg_path.read_text(encoding="utf-8")))

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
    df = preprocess.preprocess_comments_df(df_train_raw, cfg, is_train=True)

    # Labels (use scratch run label mapping for consistency)
    map_path = scratch_run / "label_mapping.json"
    if not map_path.exists():
        raise FileNotFoundError(f"Missing scratch label_mapping.json: {map_path}")
    label2id, id2label = _load_label_mapping(map_path)
    y_lbl = df[cfg.target_col].astype(int).to_numpy()
    y = np.asarray([label2id[int(v)] for v in y_lbl], dtype=np.int64)
    num_classes = int(len(label2id))
    target_labels_in_order = [id2label[i] for i in sorted(id2label.keys())]

    # Base texts for gating (cleaned comment only)
    base_texts = df[cfg.text_col].astype("string").fillna("").astype(str).tolist()

    # Scratch OOF merge
    experts = [e.strip() for e in (args.scratch_experts or []) if e and e.strip()]
    proba_by_expert = _load_scratch_oof(scratch_run=scratch_run, experts=experts)
    proba_scratch = _merge_scratch_oof(proba_by_expert=proba_by_expert, base_texts=base_texts, method=str(args.scratch_merge))

    score_scratch = metrics.macro_f1_score(y, proba_scratch.argmax(axis=1), labels=list(range(num_classes)))
    print(f"[scratch] merge={args.scratch_merge}  oof_macro_f1={score_scratch:.6f}")

    if not args.transformer_run_dirs:
        print("No transformer_run_dirs provided; nothing else to optimize.")
        return

    # Transformer OOF
    proba_tr_sum: np.ndarray | None = None
    n_runs = 0

    for rd_str in args.transformer_run_dirs:
        run_dir = Path(rd_str)
        if not run_dir.exists():
            raise FileNotFoundError(f"Transformer run dir not found: {run_dir}")

        rs_path = run_dir / "run_summary.json"
        if not rs_path.exists():
            raise FileNotFoundError(f"Missing run_summary.json in transformer run dir: {run_dir}")
        rs = _read_json(rs_path)
        n_splits = int(rs.get("n_splits", 5))
        seed = int(rs.get("seed", 42))
        add_meta = bool(rs.get("add_meta", False))
        add_flags = bool(rs.get("add_flags", False))
        max_length = int(rs.get("max_length", 128))

        fold_map = _expand_transformer_run_dir(run_dir)  # {fold: dir}

        # Build texts in the same order as training (preprocess does not reorder rows)
        try:
            import transformers  # noqa: F401
            from transformers import AutoTokenizer
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("Missing transformers. Install with: pip install transformers") from e

        # Allowed special tokens differ per tokenizer; we use fold1 tokenizer as reference.
        fold1_dir = fold_map.get(1) or next(iter(fold_map.values()))
        tok = AutoTokenizer.from_pretrained(str(fold1_dir), use_fast=True)
        allowed_special = set(getattr(tok, "additional_special_tokens", []) or [])
        allowed_special = allowed_special if len(allowed_special) > 0 else None

        texts = _format_transformer_text(df, cfg, add_meta=add_meta, add_flags=add_flags, allowed_special_tokens=allowed_special)

        # CV splits reconstruction
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof = np.zeros((len(df), num_classes), dtype=np.float32)

        for fold, (_, va_idx) in enumerate(skf.split(texts, y), start=1):
            if fold not in fold_map:
                raise FileNotFoundError(f"Missing fold dir for fold={fold} in run={run_dir}")
            fold_dir = fold_map[int(fold)]
            proba_val = _predict_proba_transformer_texts(
                model_dir=fold_dir,
                texts=[texts[i] for i in va_idx],
                max_length=max_length,
                batch_size=int(args.transformer_batch_size),
                device=args.transformer_device,
            )
            _, id2label_tr = _load_label_mapping_for_transformer_dir(fold_dir)
            proba_val = _align_proba_to_labels(proba_val.astype(np.float32), id2label_tr, target_labels_in_order)
            oof[va_idx] = proba_val

        proba_tr_sum = oof if proba_tr_sum is None else (proba_tr_sum + oof)
        n_runs += 1

    proba_tr = proba_tr_sum / max(1, n_runs)
    score_tr = metrics.macro_f1_score(y, proba_tr.argmax(axis=1), labels=list(range(num_classes)))
    print(f"[transformer] runs={n_runs}  oof_macro_f1={score_tr:.6f}")

    # Grid search w
    step = float(args.w_step)
    grid = np.clip(np.arange(0.0, 1.0 + 1e-9, step), 0.0, 1.0)
    best = {"w": None, "score": -1.0}
    for w in grid:
        blend = (float(w) * proba_tr) + ((1.0 - float(w)) * proba_scratch)
        sc = metrics.macro_f1_score(y, blend.argmax(axis=1), labels=list(range(num_classes)))
        if sc > float(best["score"]):
            best = {"w": float(w), "score": float(sc)}

    print("\n[blend] best")
    print("w_transformer:", best["w"])
    print("oof_macro_f1:", f"{best['score']:.6f}")
    print("\n[paste_into_predict]")
    cmd = (
        "python forca-hack/comments-scratch/src/predict.py "
        f"--scratch_run_dir {scratch_run} "
        f"--scratch_merge {args.scratch_merge} "
        f"--transformer_run_dirs {' '.join(args.transformer_run_dirs)} "
        f"--w_transformer {best['w']}"
    )
    print(cmd)


if __name__ == "__main__":
    main()


