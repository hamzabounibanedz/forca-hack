"""
Quick EDA / profiling for the Social Media Comments dataset.

Usage (local repo)
------------------
python forca-hack/comments/src/profile_data.py

Usage (Colab/Drive)
-------------------
python comments/src/profile_data.py --team_dir /content/drive/MyDrive/FORSA_team
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import preprocess
import schema
from config import CommentsConfig, guess_local_data_dir, guess_repo_root, guess_team_dir, resolve_comments_data_dir


def _noise_stats(texts: pd.Series) -> dict[str, int]:
    s = texts.astype("string").fillna("")

    pats: dict[str, re.Pattern[str]] = {
        "url_candidates": re.compile(r"(?:https?://\S+|www\.\S+)", flags=re.IGNORECASE),
        "email_candidates": re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", flags=re.IGNORECASE),
        "phone_like_candidates": re.compile(r"(?<!\d)(?:\+?\d[\d\s\-.]{6,}\d)(?!\d)"),
        "long_digit_seq_ge6": re.compile(r"\d{6,}"),
        "repeated_punct_ge3": re.compile(r"([!?.,،؛؟])\1{2,}"),
        "elongated_latin_ge4": re.compile(r"([A-Za-z])\1{3,}"),
    }
    # Avoid pandas' "match groups" warning by using pattern.search in Python.
    return {k: int(s.map(lambda x, _p=p: bool(_p.search(str(x)))).sum()) for k, p in pats.items()}


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


def _print_examples(df: pd.DataFrame, cfg: CommentsConfig, *, n: int = 3) -> None:
    # Print a few examples per class (train only)
    if cfg.target_col not in df.columns:
        return
    y = df[cfg.target_col].astype("Int64")
    for cls in sorted([int(x) for x in y.dropna().unique().tolist()]):
        sub = df[df[cfg.target_col] == cls].head(n)
        if not len(sub):
            continue
        print(f"\n=== class={cls} examples (first {min(n, len(sub))})")
        for _, row in sub.iterrows():
            plat = str(row.get(cfg.platform_col, ""))[:30]
            txt = str(row.get(cfg.text_col, "")).replace("\n", " ")
            print(f"- [{plat}] {txt[:220]}")


def profile(*, train_path: Path, test_path: Path, cfg: CommentsConfig, after_clean: bool) -> dict[str, Any]:
    df_train_raw = schema.read_csv_robust(train_path)
    df_test_raw = schema.read_csv_robust(test_path)

    checks = schema.run_all_checks(df_train=df_train_raw, df_test=df_test_raw, cfg=cfg)

    # Canonical (not cleaned yet)
    df_train = schema.rename_raw_columns(df_train_raw, cfg, is_train=True)
    df_test = schema.rename_raw_columns(df_test_raw, cfg, is_train=False)

    out: dict[str, Any] = {"checks": checks, "paths": {"train": str(train_path), "test": str(test_path)}}

    out["raw"] = {
        "train": {
            "rows": int(len(df_train_raw)),
            "cols": list(df_train_raw.columns),
            "noise": _noise_stats(df_train[cfg.text_col]),
            "length": _length_stats(df_train[cfg.text_col]),
            "platform_counts": df_train[cfg.platform_col].value_counts().head(20).to_dict(),
            "label_counts": df_train[cfg.target_col].value_counts().sort_index().to_dict(),
            "dupe_comments": int(df_train[cfg.text_col].duplicated().sum()),
        },
        "test": {
            "rows": int(len(df_test_raw)),
            "cols": list(df_test_raw.columns),
            "noise": _noise_stats(df_test[cfg.text_col]),
            "length": _length_stats(df_test[cfg.text_col]),
            "platform_counts": df_test[cfg.platform_col].value_counts().head(20).to_dict(),
            "dupe_comments": int(df_test[cfg.text_col].duplicated().sum()),
        },
    }

    if after_clean:
        tr_clean = preprocess.preprocess_comments_df(df_train_raw, cfg, is_train=True)
        te_clean = preprocess.preprocess_comments_df(df_test_raw, cfg, is_train=False)

        # Sanity checks
        preprocess.assert_no_unmasked_phones(tr_clean, cfg)
        preprocess.assert_no_unmasked_phones(te_clean, cfg)

        out["clean"] = {
            "train": {
                "noise": _noise_stats(tr_clean[cfg.text_col]),
                "length": _length_stats(tr_clean[cfg.text_col]),
                "platform_counts": tr_clean[cfg.platform_col].value_counts().head(20).to_dict(),
                "label_counts": tr_clean[cfg.target_col].value_counts().sort_index().to_dict(),
            },
            "test": {
                "noise": _noise_stats(te_clean[cfg.text_col]),
                "length": _length_stats(te_clean[cfg.text_col]),
                "platform_counts": te_clean[cfg.platform_col].value_counts().head(20).to_dict(),
            },
        }

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team_dir", type=str, default=None, help="Colab Drive team dir (optional).")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data directory.")
    parser.add_argument("--train_filename", type=str, default="train.csv")
    parser.add_argument("--test_filename", type=str, default="test_file.csv")
    parser.add_argument("--after_clean", action="store_true", help="Also profile after applying cleaning.")
    parser.add_argument("--save_json", type=str, default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    cfg = CommentsConfig()

    # Resolve paths
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

    summary = profile(train_path=train_path, test_path=test_path, cfg=cfg, after_clean=bool(args.after_clean))

    # ----- Print high-signal summary -----
    print("\n## Dataset summary")
    print("train:", summary["raw"]["train"]["rows"], "rows | cols:", summary["raw"]["train"]["cols"])
    print("test :", summary["raw"]["test"]["rows"], "rows | cols:", summary["raw"]["test"]["cols"])
    print("n_classes:", summary["checks"].get("n_classes"), "| labels:", summary["checks"].get("labels_sorted"))

    print("\n## Platform distribution (train, top20)")
    for k, v in summary["raw"]["train"]["platform_counts"].items():
        print(f"- {k}: {v}")

    print("\n## Label distribution (train)")
    for k, v in summary["raw"]["train"]["label_counts"].items():
        print(f"- {k}: {v}")

    print("\n## Noise patterns (raw train)")
    for k, v in summary["raw"]["train"]["noise"].items():
        print(f"- {k}: {v}")

    print("\n## Length stats (raw train)")
    print(summary["raw"]["train"]["length"])

    if args.after_clean and "clean" in summary:
        print("\n## Noise patterns (clean train)")
        for k, v in summary["clean"]["train"]["noise"].items():
            print(f"- {k}: {v}")

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved JSON -> {out_path}")


if __name__ == "__main__":
    main()


