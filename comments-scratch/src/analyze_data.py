"""
Analyze the comments dataset (train + 30k test) to guide augmentation decisions.

This script focuses on:
- train vs test drift signals (script mix, platform noise, duplicates)
- prevalence of domain patterns (portal/outage/prices/fibre/modem)

Usage (local):
  python forca-hack/comments-scratch/src/analyze_data.py

Usage (Colab/Drive):
  python comments-scratch/src/analyze_data.py --team_dir /content/drive/MyDrive/FORSA_team
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
from config import CommentsConfig, guess_local_data_dir, guess_team_dir, resolve_comments_data_dir


def _length_stats(texts: pd.Series) -> dict[str, Any]:
    s = texts.astype("string").fillna("")
    lengths = s.str.len()
    q = lengths.quantile([0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]).to_dict()
    return {
        "min": int(lengths.min()) if len(lengths) else 0,
        "max": int(lengths.max()) if len(lengths) else 0,
        "mean": float(lengths.mean()) if len(lengths) else 0.0,
        "quantiles": {str(k): float(v) for k, v in q.items()},
    }


def _script_stats(texts: pd.Series) -> dict[str, Any]:
    stats = [lang.script_stats(x) for x in texts.astype("string").fillna("").astype(str).tolist()]
    ar = np.array([s.arabic_letters for s in stats], dtype=np.int32)
    la = np.array([s.latin_letters for s in stats], dtype=np.int32)
    az = np.array([s.arabizi_tokens for s in stats], dtype=np.int32)
    tot = np.maximum(1, ar + la)
    ar_ratio = (ar / tot).astype(np.float32)
    la_ratio = (la / tot).astype(np.float32)
    return {
        "pct_ar_dominant_ge_055": float((ar_ratio >= 0.55).mean()),
        "pct_lat_dominant_ge_055": float((la_ratio >= 0.55).mean()),
        "pct_has_arabizi_token": float((az > 0).mean()),
        "mean_ar_ratio": float(ar_ratio.mean()),
        "mean_lat_ratio": float(la_ratio.mean()),
    }


def _group_distribution(texts: pd.Series) -> dict[str, int]:
    groups = [lang.detect_group(t) for t in texts.astype("string").fillna("").astype(str).tolist()]
    vc = pd.Series(groups).value_counts()
    return {str(k): int(v) for k, v in vc.to_dict().items()}


def _platform_noise(platforms: pd.Series) -> dict[str, Any]:
    s = platforms.astype("string").fillna("").astype(str).str.lower()
    known = {"facebook", "instagram", "linkedin", "twitter", "tiktok", "youtube", "idoom_market", "unk"}
    unknown = s[~s.isin(list(known))]
    return {
        "unknown_rate": float(len(unknown) / max(1, len(s))),
        "unknown_unique": int(unknown.nunique(dropna=False)),
        "top_unknown": {str(k): int(v) for k, v in unknown.value_counts().head(20).to_dict().items()},
    }


def _flag_prevalence(texts: pd.Series) -> dict[str, float]:
    t = texts.astype("string").fillna("")
    specs: dict[str, re.Pattern[str]] = {
        "FIBRE": re.compile(r"\b(?:fibre|ftth|fttx|idoom_fibre)\b", re.IGNORECASE),
        "MODEM": re.compile(r"\b(?:modem|wifi6)\b", re.IGNORECASE),
        "PORTAL": re.compile(
            r"\b(?:espace_client|espace\s*client|application|app|login|mot\s*de\s*passe)\b|(?:فضاء\s*الزبون|فضاء\s*الزبائن|الولوج|حساب|كلمة\s*السر)",
            re.IGNORECASE,
        ),
        "PRICES": re.compile(r"\b(?:prix|tarif|cher|promo|offre|offres)\b|(?:سعر|الاسعار|الأسعار|اسعار|العروض)", re.IGNORECASE),
        "OUTAGE": re.compile(
            r"\b(?:panne|coupure|down|hs|marche\s+pas|ne\s+marche\s+pas)\b|(?:مقطوع|انقطاع|ماكاش|مكانش|تقطع|مقطوعة)",
            re.IGNORECASE,
        ),
        "WAIT": re.compile(r"\b(?:a\s*quand|quand|depuis|attend|attente)\b|(?:وقتاش|متي|مزال|مازال|نستناو|نستنو|انتظار|راني\s+في|راه|راهم)", re.IGNORECASE),
        "LOCATION": re.compile(r"\b(?:wilaya|commune|quartier|centre|ville|rue)\b|(?:ولاية|بلدية|حي|شارع|مدينة)", re.IGNORECASE),
    }
    out: dict[str, float] = {}
    for k, p in specs.items():
        out[k] = float(t.str.contains(p, regex=True, na=False).mean())
    return out


def analyze(*, data_dir: Path, cfg: CommentsConfig) -> dict[str, Any]:
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test_file.csv"
    schema.validate_file_exists(train_path)
    schema.validate_file_exists(test_path)

    df_train_raw = schema.read_csv_robust(train_path)
    df_test_raw = schema.read_csv_robust(test_path)

    tr = preprocess.preprocess_comments_df(df_train_raw, cfg, is_train=True)
    te = preprocess.preprocess_comments_df(df_test_raw, cfg, is_train=False)

    out: dict[str, Any] = {"paths": {"train": str(train_path), "test": str(test_path)}}

    out["train"] = {
        "rows": int(len(tr)),
        "label_counts": {int(k): int(v) for k, v in tr[cfg.target_col].astype("Int64").value_counts().sort_index().to_dict().items()},
        "platform_counts": {str(k): int(v) for k, v in tr[cfg.platform_col].astype("string").value_counts().head(20).to_dict().items()},
        "dup_comments": int(tr[cfg.text_col].duplicated().sum()),
        "length": _length_stats(tr[cfg.text_col]),
        "script": _script_stats(tr[cfg.text_col]),
        "groups": _group_distribution(tr[cfg.text_col]),
        "flag_prevalence": _flag_prevalence(tr[cfg.text_col]),
    }
    out["test"] = {
        "rows": int(len(te)),
        "platform_counts": {str(k): int(v) for k, v in te[cfg.platform_col].astype("string").value_counts().head(20).to_dict().items()},
        "platform_noise": _platform_noise(te[cfg.platform_col]),
        "dup_comments": int(te[cfg.text_col].duplicated().sum()),
        "length": _length_stats(te[cfg.text_col]),
        "script": _script_stats(te[cfg.text_col]),
        "groups": _group_distribution(te[cfg.text_col]),
        "flag_prevalence": _flag_prevalence(te[cfg.text_col]),
    }

    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--team_dir", type=str, default=None)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--save_json", type=str, default=None)
    args = p.parse_args()

    cfg = CommentsConfig()
    if args.data_dir:
        data_dir = Path(args.data_dir)
    elif args.team_dir:
        data_dir = resolve_comments_data_dir(Path(args.team_dir))
    else:
        team = guess_team_dir()
        data_dir = resolve_comments_data_dir(team) if team else guess_local_data_dir()

    rep = analyze(data_dir=data_dir, cfg=cfg)

    print("\n## Dataset summary")
    print("train rows:", rep["train"]["rows"], "| test rows:", rep["test"]["rows"])
    print("train labels:", rep["train"]["label_counts"])

    print("\n## Train vs Test script/groups drift (cleaned)")
    print("train script:", rep["train"]["script"])
    print("test  script:", rep["test"]["script"])
    print("train groups:", rep["train"]["groups"])
    print("test  groups:", rep["test"]["groups"])

    print("\n## Platform noise (test)")
    print("top20 platforms:", rep["test"]["platform_counts"])
    print("platform_noise:", rep["test"]["platform_noise"])

    print("\n## Domain flag prevalence (train vs test)")
    print("train flags:", rep["train"]["flag_prevalence"])
    print("test  flags:", rep["test"]["flag_prevalence"])

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved JSON -> {out}")


if __name__ == "__main__":
    main()


