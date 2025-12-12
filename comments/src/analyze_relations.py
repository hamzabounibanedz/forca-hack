"""
In-depth relationships analysis for the Social Media Comments dataset (9 classes).

This script answers:
- What are the main lexical/keyword signatures of each class? (raw vs cleaned)
- Which classes are most similar / most confusable?
- Does platform influence the class distribution?
- How does text length differ by class?

Outputs:
- Prints a concise report to stdout
- Optionally saves JSON to --out_json

Usage:
python forca-hack/comments/src/analyze_relations.py --after_clean --out_json forca-hack/outputs/comments/analysis/relations.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import preprocess
import schema
from config import CommentsConfig, guess_local_data_dir, guess_team_dir, resolve_comments_data_dir


TOKEN_RE = re.compile(r"<[^>]+>|[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+|[a-zA-Z]+|\d+")

# Script counters (Arabic vs Latin vs digits)
_AR_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
_LA_RE = re.compile(r"[A-Za-z]")
_DIG_RE = re.compile(r"\d")


def _tokenize(text: str) -> list[str]:
    toks = TOKEN_RE.findall(str(text))
    out: list[str] = []
    for t in toks:
        t = t.strip()
        if not t:
            continue
        # Normalize latin case
        if re.fullmatch(r"[A-Za-z]+", t):
            t = t.lower()
        out.append(t)
    return out


def _basic_stats(df: pd.DataFrame, cfg: CommentsConfig, *, is_train: bool) -> dict[str, Any]:
    d: dict[str, Any] = {
        "rows": int(len(df)),
        "cols": list(df.columns),
        "dupe_rows": int(df.duplicated().sum()),
        "dupe_text": int(df[cfg.text_col].astype("string").duplicated().sum()) if cfg.text_col in df.columns else None,
    }
    if cfg.platform_col in df.columns:
        d["platform_counts"] = df[cfg.platform_col].astype("string").value_counts().head(20).to_dict()
    if is_train and cfg.target_col in df.columns:
        d["label_counts"] = df[cfg.target_col].astype("Int64").value_counts().sort_index().to_dict()
    # Length
    if cfg.text_col in df.columns:
        s = df[cfg.text_col].astype("string").fillna("")
        lengths = s.str.len()
        d["len_chars"] = {
            "min": int(lengths.min()) if len(lengths) else 0,
            "max": int(lengths.max()) if len(lengths) else 0,
            "mean": float(lengths.mean()) if len(lengths) else 0.0,
            "q": {str(k): float(v) for k, v in lengths.quantile([0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]).to_dict().items()},
        }
    return d


def _script_mix_by_class(df: pd.DataFrame, cfg: CommentsConfig) -> dict[int, dict[str, float]]:
    """
    Per-class script composition:
    - arabic_frac_mean, latin_frac_mean, digit_frac_mean
    - pct_mostly_arabic (arabic_count > latin_count)
    - pct_has_latin / pct_has_arabic / pct_has_digits
    """
    if cfg.target_col not in df.columns:
        return {}
    s = df[cfg.text_col].astype("string").fillna("")
    y = df[cfg.target_col].astype("Int64")
    out: dict[int, dict[str, float]] = {}
    for cls in sorted([int(x) for x in y.dropna().unique().tolist()]):
        sub = s[y == cls]
        ar = sub.str.count(_AR_RE).astype("float32")
        la = sub.str.count(_LA_RE).astype("float32")
        di = sub.str.count(_DIG_RE).astype("float32")
        ln = sub.str.len().replace(0, np.nan).astype("float32")

        out[cls] = {
            "mean_len": float(sub.str.len().mean()) if len(sub) else 0.0,
            "arabic_frac_mean": float((ar / ln).mean()) if len(sub) else 0.0,
            "latin_frac_mean": float((la / ln).mean()) if len(sub) else 0.0,
            "digit_frac_mean": float((di / ln).mean()) if len(sub) else 0.0,
            "pct_mostly_arabic": float((ar > la).mean()) if len(sub) else 0.0,
            "pct_has_latin": float((la > 0).mean()) if len(sub) else 0.0,
            "pct_has_arabic": float((ar > 0).mean()) if len(sub) else 0.0,
            "pct_has_digits": float((di > 0).mean()) if len(sub) else 0.0,
        }
    return out


def _domain_pattern_rates(df: pd.DataFrame, cfg: CommentsConfig) -> dict[int, dict[str, float]]:
    """
    Per-class prevalence of high-signal domain patterns (as rates in [0,1]).
    Useful for designing metadata tokens/flags for transformers.
    """
    if cfg.target_col not in df.columns:
        return {}
    s = df[cfg.text_col].astype("string").fillna("")
    y = df[cfg.target_col].astype("Int64")

    pats: dict[str, re.Pattern[str]] = {
        "has_fibre": re.compile(r"\b(?:fibre|ftth|fttx|fiber|idoom_fibre)\b|فيبر|الفيبر|فايبر", re.IGNORECASE),
        "has_modem": re.compile(r"\b(?:modem|wifi6|wifi)\b|مودام|مودم", re.IGNORECASE),
        "has_portal": re.compile(
            r"\b(?:espace\s*client|espace|application|app|client|compte|login|mot\s*de\s*passe)\b|فضاء\s*الزبون|فضاء|الولوج|حساب|كلمة\s*السر",
            re.IGNORECASE,
        ),
        "has_prices": re.compile(r"\b(?:prix|tarif|cher|promo|offre|offres)\b|سعر|الاسعار|الأسعار|اسعار|العروض", re.IGNORECASE),
        "has_outage": re.compile(
            r"\b(?:panne|coupure|down|hs|marche\s+pas|ne\s+marche\s+pas)\b|مقطوع|انقطاع|ماكاش|مكانش|تقطع|مقطوعة",
            re.IGNORECASE,
        ),
        "has_support_nums": re.compile(r"\b(?:100|12|1500)\b"),
        "has_speed_units": re.compile(r"\b(?:mbps|gbps|mega|giga)\b|ميغا|جيغا", re.IGNORECASE),
    }

    out: dict[int, dict[str, float]] = {}
    for cls in sorted([int(x) for x in y.dropna().unique().tolist()]):
        sub = s[y == cls]
        row: dict[str, float] = {}
        for name, pat in pats.items():
            row[name] = float(sub.str.contains(pat, regex=True, na=False).mean()) if len(sub) else 0.0
        out[cls] = row
    return out


def _platform_vs_class(df: pd.DataFrame, cfg: CommentsConfig) -> dict[str, Any]:
    if cfg.platform_col not in df.columns or cfg.target_col not in df.columns:
        return {}
    tab = pd.crosstab(df[cfg.platform_col].astype("string"), df[cfg.target_col].astype("Int64"))
    tab_pct = tab.div(tab.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return {
        "counts": tab.astype(int).to_dict(),
        "row_pct": (100.0 * tab_pct).round(2).to_dict(),
    }


def _per_class_length(df: pd.DataFrame, cfg: CommentsConfig) -> dict[int, Any]:
    if cfg.target_col not in df.columns:
        return {}
    s = df[cfg.text_col].astype("string").fillna("")
    y = df[cfg.target_col].astype("Int64")
    out: dict[int, Any] = {}
    for cls in sorted([int(x) for x in y.dropna().unique().tolist()]):
        sub = s[y == cls]
        lens = sub.str.len()
        out[cls] = {
            "n": int(len(sub)),
            "mean": float(lens.mean()) if len(lens) else 0.0,
            "median": float(lens.median()) if len(lens) else 0.0,
            "q90": float(lens.quantile(0.9)) if len(lens) else 0.0,
        }
    return out


def _top_tokens_by_class(df: pd.DataFrame, cfg: CommentsConfig, *, top_k: int = 25) -> dict[int, list[tuple[str, int]]]:
    if cfg.target_col not in df.columns:
        return {}
    y = df[cfg.target_col].astype("Int64")
    text = df[cfg.text_col].astype("string").fillna("")
    out: dict[int, list[tuple[str, int]]] = {}
    for cls in sorted([int(x) for x in y.dropna().unique().tolist()]):
        c = Counter()
        for t in text[y == cls].tolist():
            toks = _tokenize(str(t))
            # Filter: remove 1-char tokens and pure digits that are too small to be informative
            toks = [x for x in toks if len(x) >= 2 and not re.fullmatch(r"\d{1,2}", x)]
            c.update(toks)
        out[cls] = c.most_common(top_k)
    return out


def _log_odds_top_tokens(
    df: pd.DataFrame,
    cfg: CommentsConfig,
    *,
    top_k: int = 20,
    alpha: float = 0.01,
) -> dict[int, list[tuple[str, float]]]:
    """
    Distinctive tokens per class using (smoothed) log-odds ratio:
      log((p_w_in_cls)/(1-p_w_in_cls)) - log((p_w_in_rest)/(1-p_w_in_rest))
    """
    if cfg.target_col not in df.columns:
        return {}
    y = df[cfg.target_col].astype("Int64")
    text = df[cfg.text_col].astype("string").fillna("")

    # Global vocabulary counts
    class_counts: dict[int, Counter] = {}
    total_counts = Counter()
    for cls in sorted([int(x) for x in y.dropna().unique().tolist()]):
        c = Counter()
        for t in text[y == cls].tolist():
            toks = [x for x in _tokenize(str(t)) if len(x) >= 2 and not re.fullmatch(r"\d{1,2}", x)]
            c.update(toks)
        class_counts[cls] = c
        total_counts.update(c)

    vocab = list(total_counts.keys())
    V = len(vocab)
    out: dict[int, list[tuple[str, float]]] = {}

    for cls, c_cls in class_counts.items():
        c_rest = total_counts.copy()
        for w, n in c_cls.items():
            c_rest[w] -= n
            if c_rest[w] <= 0:
                del c_rest[w]

        n_cls = sum(c_cls.values())
        n_rest = max(1, sum(c_rest.values()))

        scored: list[tuple[str, float]] = []
        for w in vocab:
            a = c_cls.get(w, 0)
            b = c_rest.get(w, 0)
            # smoothed probabilities
            p1 = (a + alpha) / (n_cls + alpha * V)
            p2 = (b + alpha) / (n_rest + alpha * V)
            # logit difference
            def logit(p: float) -> float:
                p = min(max(p, 1e-9), 1 - 1e-9)
                return math.log(p / (1 - p))

            score = logit(p1) - logit(p2)
            scored.append((w, float(score)))

        scored.sort(key=lambda x: x[1], reverse=True)
        out[cls] = scored[:top_k]

    return out


def _jaccard_similarity(top_tokens: dict[int, list[tuple[str, Any]]], *, k: int = 20) -> dict[str, Any]:
    classes = sorted(top_tokens.keys())
    sets = {c: set([w for w, _ in top_tokens[c][:k]]) for c in classes}
    sim = {}
    for i in classes:
        sim_i = {}
        for j in classes:
            if i == j:
                sim_i[j] = 1.0
            else:
                a, b = sets[i], sets[j]
                denom = len(a | b) if (a | b) else 1
                sim_i[j] = round(len(a & b) / denom, 3)
        sim[i] = sim_i
    # Top neighbors
    neigh = {}
    for i in classes:
        pairs = [(j, sim[i][j]) for j in classes if j != i]
        pairs.sort(key=lambda x: x[1], reverse=True)
        neigh[i] = pairs[:3]
    return {"jaccard": sim, "top_neighbors": neigh}


def _examples_per_class(df: pd.DataFrame, cfg: CommentsConfig, *, n: int = 3, seed: int = 42) -> dict[int, list[str]]:
    if cfg.target_col not in df.columns:
        return {}
    rng = random.Random(seed)
    y = df[cfg.target_col].astype("Int64")
    text = df[cfg.text_col].astype("string").fillna("")
    out: dict[int, list[str]] = {}
    for cls in sorted([int(x) for x in y.dropna().unique().tolist()]):
        idx = (y == cls).to_numpy().nonzero()[0].tolist()
        rng.shuffle(idx)
        out[cls] = [str(text.iloc[i])[:260].replace("\n", " ") for i in idx[:n]]
    return out


def _to_jsonable(obj: Any) -> Any:
    """
    Recursively convert numpy/pandas scalars and dict keys to JSON-friendly types.
    """
    # numpy scalar -> python scalar
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    # pandas NA/NaT -> None
    if obj is pd.NA:
        return None

    if isinstance(obj, dict):
        out: dict[Any, Any] = {}
        for k, v in obj.items():
            kk: Any = k
            if isinstance(kk, (np.integer,)):
                kk = int(kk)
            elif isinstance(kk, (np.floating,)):
                kk = float(kk)
            elif isinstance(kk, (np.bool_,)):
                kk = bool(kk)
            out[kk] = _to_jsonable(v)
        return out

    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]

    return obj


def main() -> None:
    # Windows terminals often default to cp1252; reconfigure so Arabic text doesn't crash printing.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--team_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--train_filename", type=str, default="train.csv")
    parser.add_argument("--after_clean", action="store_true")
    parser.add_argument("--top_k", type=int, default=25)
    parser.add_argument("--out_json", type=str, default=None)
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
    df_raw = schema.read_csv_robust(train_path)

    df_canon = schema.rename_raw_columns(df_raw, cfg, is_train=True)
    report: dict[str, Any] = {"paths": {"train": str(train_path)}}

    report["raw"] = {
        "basic": _basic_stats(df_canon, cfg, is_train=True),
        "platform_vs_class": _platform_vs_class(df_canon, cfg),
        "length_by_class": _per_class_length(df_canon, cfg),
        "script_mix_by_class": _script_mix_by_class(df_canon, cfg),
        "domain_pattern_rates": _domain_pattern_rates(df_canon, cfg),
        "top_tokens_by_class": _top_tokens_by_class(df_canon, cfg, top_k=int(args.top_k)),
        "distinctive_tokens_logodds": _log_odds_top_tokens(df_canon, cfg, top_k=20),
        "examples_by_class": _examples_per_class(df_canon, cfg, n=3),
    }
    report["raw"]["similarity"] = _jaccard_similarity(report["raw"]["top_tokens_by_class"], k=20)

    if args.after_clean:
        df_clean = preprocess.preprocess_comments_df(df_raw, cfg, is_train=True)
        report["clean"] = {
            "basic": _basic_stats(df_clean, cfg, is_train=True),
            "platform_vs_class": _platform_vs_class(df_clean, cfg),
            "length_by_class": _per_class_length(df_clean, cfg),
            "script_mix_by_class": _script_mix_by_class(df_clean, cfg),
            "domain_pattern_rates": _domain_pattern_rates(df_clean, cfg),
            "top_tokens_by_class": _top_tokens_by_class(df_clean, cfg, top_k=int(args.top_k)),
            "distinctive_tokens_logodds": _log_odds_top_tokens(df_clean, cfg, top_k=20),
            "examples_by_class": _examples_per_class(df_clean, cfg, n=3),
        }
        report["clean"]["similarity"] = _jaccard_similarity(report["clean"]["top_tokens_by_class"], k=20)

    # ---- Print concise high-signal report ----
    print("\n## Basic")
    classes = [int(x) for x in sorted(report["raw"]["basic"]["label_counts"].keys())]
    print("Rows:", report["raw"]["basic"]["rows"], "| Classes:", classes)
    print("Platforms(top):", list(report["raw"]["basic"]["platform_counts"].items())[:5])

    print("\n## Length by class (mean chars)")
    for cls, d in report["raw"]["length_by_class"].items():
        print(f"- class {cls}: mean={d['mean']:.1f} median={d['median']:.1f} q90={d['q90']:.1f}")

    print("\n## Strongest class similarities (Jaccard of top tokens)")
    for cls, neigh in report["raw"]["similarity"]["top_neighbors"].items():
        pretty = ", ".join([f"{j}:{v}" for j, v in neigh])
        print(f"- class {cls} -> {pretty}")

    print("\n## Distinctive tokens (log-odds, raw)")
    for cls in sorted(report["raw"]["distinctive_tokens_logodds"].keys()):
        top = report["raw"]["distinctive_tokens_logodds"][cls][:10]
        toks = ", ".join([w for w, _ in top])
        print(f"- class {cls}: {toks}")

    if args.after_clean and "clean" in report:
        print("\n## Cleaning impact (noise check)")
        # Compare top-level length mean
        raw_mean = report["raw"]["basic"]["len_chars"]["mean"]
        cl_mean = report["clean"]["basic"]["len_chars"]["mean"]
        print(f"mean len chars raw={raw_mean:.2f} -> clean={cl_mean:.2f}")

    print("\n## Script mix by class (raw)")
    sm = report["raw"]["script_mix_by_class"]
    for cls in sorted(sm.keys()):
        r = sm[cls]
        print(
            f"- class {cls}: arabic_frac={r['arabic_frac_mean']:.3f} latin_frac={r['latin_frac_mean']:.3f} "
            f"digit_frac={r['digit_frac_mean']:.3f} mostly_arabic={r['pct_mostly_arabic']:.3f}"
        )

    print("\n## Domain pattern rates by class (raw, %)")
    dr = report["raw"]["domain_pattern_rates"]
    for cls in sorted(dr.keys()):
        row = dr[cls]
        pretty = ", ".join([f"{k}={row[k]*100:.1f}%" for k in sorted(row.keys())])
        print(f"- class {cls}: {pretty}")

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        safe = _to_jsonable(report)
        out_path.write_text(json.dumps(safe, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved JSON -> {out_path}")


if __name__ == "__main__":
    main()


