"""
Prediction / submission generator for the Social Media Comments challenge.

Supports:
- TF-IDF + LR baseline saved by `train_tfidf_lr.py`
- Transformer checkpoints saved by `train_transformer.py`
- Simple ensembling by averaging probabilities across multiple models

Example (baseline)
------------------
python forca-hack/comments/src/predict.py ^
  --model_paths forca-hack/outputs/comments/models/tfidf_lr/model.joblib ^
  --test_filename test_file.csv ^
  --output_csv submission.csv

Example (ensemble)
------------------
python forca-hack/comments/src/predict.py --model_paths path/to/model1 path/to/model2 --weights 0.6 0.4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import features
import preprocess
import schema
from config import CommentsConfig, cfg_from_dict, guess_local_data_dir, guess_team_dir, resolve_comments_data_dir


def _format_transformer_text(
    df: pd.DataFrame,
    cfg: CommentsConfig,
    *,
    add_meta: bool,
    add_flags: bool,
) -> list[str]:
    """
    Must match the formatting used in `train_transformer.py`.
    """
    import re

    txt = df[cfg.text_col].astype("string").fillna("")
    parts: list[pd.Series] = []

    if add_meta:
        plat = df[cfg.platform_col].astype("string").fillna("unk").str.lower()
        parts.append(("[PLATFORM] " + plat).astype("string"))

    if add_flags:
        flag_specs: tuple[tuple[str, re.Pattern[str]], ...] = (
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
        )

        flags_str = pd.Series([""] * len(df), index=df.index, dtype="string")
        for name, pat in flag_specs:
            col = txt.str.contains(pat, regex=True, na=False).map(lambda b, _n=name: f"[{_n}]" if b else "")
            flags_str = (flags_str + " " + col).astype("string")
        parts.append(flags_str.str.strip())

    parts.append(("[TEXT] " + txt).astype("string"))

    out = parts[0] if parts else pd.Series([""] * len(df), index=df.index, dtype="string")
    for p in parts[1:]:
        out = (out + " " + p).astype("string")
    return out.astype(str).tolist()


def _build_text_with_meta(df: pd.DataFrame, cfg: CommentsConfig) -> list[str]:
    plat = df[cfg.platform_col].astype("string").fillna("unk").str.lower()
    txt = df[cfg.text_col].astype("string").fillna("")
    return (("platform=" + plat + " ") + txt).astype(str).tolist()


def _load_label_mapping(mapping_path: Path) -> tuple[dict[int, int], dict[int, int]]:
    data = json.loads(mapping_path.read_text(encoding="utf-8"))
    label2id = {int(k): int(v) for k, v in data["label2id"].items()}
    id2label = {int(k): int(v) for k, v in data["id2label"].items()}
    return label2id, id2label


def _load_label_mapping_from_transformer_config(model_dir: Path) -> tuple[dict[int, int], dict[int, int]]:
    """
    Fallback when label_mapping.json is missing (e.g., interrupted training run):
    parse Hugging Face `config.json` which usually contains `id2label` and/or `label2id`.
    """
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.json in transformer dir: {model_dir}")

    data = json.loads(cfg_path.read_text(encoding="utf-8"))

    id2label_raw = data.get("id2label") or {}
    label2id_raw = data.get("label2id") or {}

    id2label: dict[int, int] = {}
    for k, v in id2label_raw.items():
        ik = int(k)
        # Most of our runs store labels as numeric strings ("1".."9")
        try:
            iv = int(v)
        except Exception:
            # If it's not numeric, we cannot safely submit to Kaggle (expects numeric class)
            raise ValueError(f"Non-numeric id2label value in config.json: {v!r}")
        id2label[ik] = iv

    label2id: dict[int, int] = {}
    if label2id_raw:
        for k, v in label2id_raw.items():
            try:
                lk = int(k)
            except Exception:
                raise ValueError(f"Non-numeric label2id key in config.json: {k!r}")
            label2id[lk] = int(v)
    else:
        # Invert id2label if label2id not provided
        label2id = {lbl: i for i, lbl in id2label.items()}

    if not id2label or not label2id:
        raise ValueError(f"Could not build label mapping from config.json in {model_dir}")
    return label2id, id2label


def _load_label_mapping_for_model(model_path: Path) -> tuple[dict[int, int], dict[int, int]]:
    """
    Load label mapping for either:
    - classical models (joblib/cbm): requires label_mapping.json next to model file
    - transformers: prefer label_mapping.json at run root; fallback to fold's config.json
    """
    if model_path.is_file():
        mapping = model_path.parent / "label_mapping.json"
        if mapping.exists():
            return _load_label_mapping(mapping)
        raise FileNotFoundError(f"Could not find label_mapping.json near model: {model_path}")

    # transformer directory (fold)
    candidates = [model_path / "label_mapping.json", model_path.parent / "label_mapping.json"]
    for c in candidates:
        if c.exists():
            return _load_label_mapping(c)

    # Fallback: parse HF config.json inside the fold directory
    return _load_label_mapping_from_transformer_config(model_path)


def _find_preprocess_config_for_model(model_path: Path) -> Path | None:
    if model_path.is_file():
        candidates = [model_path.parent / "preprocess_config.json"]
    else:
        candidates = [model_path / "preprocess_config.json", model_path.parent / "preprocess_config.json"]
    for c in candidates:
        if c.exists():
            return c
    return None


def _predict_proba_sklearn(model_file: Path, df_test_clean: pd.DataFrame, cfg: CommentsConfig) -> np.ndarray:
    model = joblib.load(model_file)
    X = _build_text_with_meta(df_test_clean, cfg)
    proba = model.predict_proba(X)
    return np.asarray(proba, dtype=np.float32)


def _predict_proba_catboost(model_file: Path, df_test_clean: pd.DataFrame, cfg: CommentsConfig) -> np.ndarray:
    try:
        from catboost import CatBoostClassifier, Pool
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Missing catboost. Install with: pip install catboost") from e

    df = features.engineer_features(df_test_clean, cfg)

    # Use saved feature spec when available
    spec_path = model_file.parent / "feature_spec.json"
    if spec_path.exists():
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        used_cols = list(spec["used_cols"])
        cat_cols = list(spec.get("cat_cols", [cfg.platform_col]))
        text_cols = list(spec.get("text_cols", [cfg.text_col]))
    else:
        spec = features.build_feature_spec(cfg)
        used_cols = [cfg.platform_col, cfg.text_col] + [c for c in spec["num_cols"] if c in df.columns]
        cat_cols = [cfg.platform_col]
        text_cols = [cfg.text_col]

    X = df[used_cols].copy()
    pool = Pool(X, cat_features=cat_cols, text_features=text_cols)

    model = CatBoostClassifier()
    model.load_model(str(model_file))
    proba = model.predict_proba(pool)
    return np.asarray(proba, dtype=np.float32)


def _predict_proba_transformer(
    model_dir: Path,
    df_test_clean: pd.DataFrame,
    cfg: CommentsConfig,
    *,
    max_length: int,
    batch_size: int,
    device: str | None,
    add_meta: bool,
    add_flags: bool,
) -> np.ndarray:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Missing transformers/torch. Install with: pip install transformers torch") from e

    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model.eval()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    texts = _format_transformer_text(df_test_clean, cfg, add_meta=add_meta, add_flags=add_flags)

    n = len(texts)
    # Infer number of labels from model
    num_labels = int(getattr(model.config, "num_labels", 0)) or 1
    out = np.zeros((n, num_labels), dtype=np.float32)

    for i in range(0, n, batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float32)
        out[i : i + len(batch)] = probs

    return out


def _align_proba_to_labels(
    proba: np.ndarray,
    id2label: dict[int, int],
    target_labels_in_order: list[int],
) -> np.ndarray:
    """
    Reorder probability columns so that column j corresponds to target_labels_in_order[j].
    """
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--test_filename", type=str, default="test_file.csv")
    parser.add_argument("--model_paths", type=str, nargs="+", required=True, help="One or more model paths.")
    parser.add_argument("--weights", type=float, nargs="*", default=None, help="Optional ensemble weights.")
    parser.add_argument("--output_csv", type=str, default="submission.csv")
    parser.add_argument("--target_col_name", type=str, default="Class", help="Submission target column name.")
    parser.add_argument("--save_proba_npy", type=str, default=None)
    # Transformer inference knobs
    parser.add_argument("--transformer_max_length", type=int, default=128)
    parser.add_argument("--transformer_batch_size", type=int, default=32)
    parser.add_argument("--transformer_device", type=str, default=None)
    parser.add_argument("--transformer_add_meta", action="store_true")
    parser.add_argument("--transformer_add_flags", action="store_true")
    args = parser.parse_args()

    # Use the preprocess config saved next to the first model when available (reproducibility).
    model_paths = [Path(p) for p in args.model_paths]
    cfg_path = _find_preprocess_config_for_model(model_paths[0])
    if cfg_path is not None:
        cfg = cfg_from_dict(json.loads(cfg_path.read_text(encoding="utf-8")))
    else:
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
    df_test_clean = preprocess.preprocess_comments_df(df_test_raw, cfg, is_train=False)

    if args.weights is None or len(args.weights) == 0:
        weights = [1.0] * len(model_paths)
    else:
        if len(args.weights) != len(model_paths):
            raise ValueError("If provided, --weights must have the same length as --model_paths.")
        weights = list(args.weights)

    # Determine global label ordering from first model mapping
    _, id2label0 = _load_label_mapping_for_model(model_paths[0])
    target_labels_in_order = [id2label0[i] for i in sorted(id2label0.keys())]

    proba_sum = None
    weight_sum = float(sum(weights))

    for w, mp in zip(weights, model_paths, strict=True):
        mp_path = mp
        _, id2label = _load_label_mapping_for_model(mp_path)

        if mp_path.is_file() and mp_path.suffix.lower() in {".joblib", ".pkl"}:
            proba = _predict_proba_sklearn(mp_path, df_test_clean, cfg)
        elif mp_path.is_file() and mp_path.suffix.lower() in {".cbm"}:
            proba = _predict_proba_catboost(mp_path, df_test_clean, cfg)
        else:
            proba = _predict_proba_transformer(
                mp_path,
                df_test_clean,
                cfg,
                max_length=int(args.transformer_max_length),
                batch_size=int(args.transformer_batch_size),
                device=args.transformer_device,
                add_meta=bool(args.transformer_add_meta),
                add_flags=bool(args.transformer_add_flags),
            )

        proba = _align_proba_to_labels(proba, id2label, target_labels_in_order)
        proba = proba * float(w)
        proba_sum = proba if proba_sum is None else (proba_sum + proba)

    if proba_sum is None:
        raise RuntimeError("No predictions produced.")

    proba_ens = proba_sum / max(weight_sum, 1e-9)
    pred_id = proba_ens.argmax(axis=1)
    pred_label = [target_labels_in_order[int(i)] for i in pred_id]

    sub = pd.DataFrame({cfg.id_col: df_test_clean[cfg.id_col].astype(int), args.target_col_name: pred_label})

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Saved submission -> {out_csv} (rows={len(sub)})")

    if args.save_proba_npy:
        out_npy = Path(args.save_proba_npy)
        out_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_npy, proba_ens.astype(np.float32))
        print(f"Saved proba -> {out_npy}")


if __name__ == "__main__":
    main()


