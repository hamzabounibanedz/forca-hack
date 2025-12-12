"""
Data profiling for the Call Center challenge (raw vs clean).

Why
---
Before changing preprocessing, we need facts:
- missingness
- categorical cardinalities + variants
- top values per category
- class imbalance

This script profiles BOTH raw and clean datasets and prints + saves a report.

Outputs (default)
-----------------
<team_dir>/outputs/callcenter/profile/profile_report.json

Run (Colab example)
-------------------
python forca-hack/call-center/src/profile_data.py --team_dir /content/drive/MyDrive/FORSA_team

Local fallback
--------------
Uses:
- raw:   forca-hack/call-center/data/
- clean: forca-hack/outputs/callcenter/clean/
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config import (
    CallCenterConfig,
    default_config,
    guess_local_data_dir,
    guess_repo_root,
    guess_team_dir,
    resolve_callcenter_data_dir,
    resolve_callcenter_output_dir,
)


def _value_key(v: Any) -> str:
    try:
        if pd.isna(v):
            return "<NA>"
    except Exception:
        pass
    return str(v)


def _missing_table(df: pd.DataFrame) -> list[dict[str, Any]]:
    miss_count = df.isna().sum()
    miss_rate = df.isna().mean()
    out = []
    for col in df.columns:
        out.append(
            {
                "column": col,
                "missing_count": int(miss_count[col]),
                "missing_rate": float(miss_rate[col]),
            }
        )
    out.sort(key=lambda r: (-r["missing_rate"], r["column"]))
    return out


def _top_values(df: pd.DataFrame, col: str, *, k: int = 20) -> list[dict[str, Any]]:
    vc = df[col].value_counts(dropna=False).head(k)
    rows: list[dict[str, Any]] = []
    for v, cnt in vc.items():
        rows.append({"value": _value_key(v), "count": int(cnt)})
    return rows


def _cat_profile(df: pd.DataFrame, cols: list[str], *, top_k: int = 20, unk_token: str | None = None) -> dict[str, Any]:
    unique_counts: dict[str, int] = {}
    top: dict[str, list[dict[str, Any]]] = {}
    unk_rates: dict[str, float] = {}

    for c in cols:
        if c not in df.columns:
            continue
        unique_counts[c] = int(df[c].nunique(dropna=False))
        top[c] = _top_values(df, c, k=top_k)
        if unk_token is not None:
            unk_rates[c] = float((df[c].astype("string") == unk_token).mean())

    # High-cardinality helper (sorted)
    high_card = sorted(unique_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return {
        "unique_count": unique_counts,
        "top_values": top,
        "unk_rate": unk_rates if unk_token is not None else None,
        "high_cardinality_sorted": [{"column": c, "unique": int(u)} for c, u in high_card],
    }


def _class_distribution(df: pd.DataFrame, target_col: str) -> dict[str, Any] | None:
    if target_col not in df.columns:
        return None
    vc = df[target_col].value_counts(dropna=False)
    return {str(_value_key(k)): int(v) for k, v in vc.items()}


def _invert_mapping(raw_to_canon: dict[str, str]) -> dict[str, str]:
    canon_to_raw: dict[str, str] = {}
    for raw, canon in raw_to_canon.items():
        canon_to_raw.setdefault(canon, raw)
    return canon_to_raw


def profile_callcenter_raw_vs_clean(
    *,
    team_dir: str | Path | None = None,
    raw_dir: str | Path | None = None,
    clean_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    cfg: CallCenterConfig | None = None,
    top_k: int = 20,
) -> dict[str, Any]:
    cfg = cfg or default_config()

    # ---- Resolve paths ----
    team = Path(team_dir) if team_dir is not None else guess_team_dir()

    raw_dir_path = (
        Path(raw_dir)
        if raw_dir is not None
        else (resolve_callcenter_data_dir(team) if team is not None else guess_local_data_dir())
    )

    # Clean outputs are produced by make_clean_dataset (Drive) or by local fallback:
    # forca-hack/outputs/callcenter/clean/
    clean_dir_path = (
        Path(clean_dir)
        if clean_dir is not None
        else (
            resolve_callcenter_output_dir(team)
            if team is not None
            else (guess_repo_root() / "outputs" / "callcenter" / "clean")
        )
    )

    out_dir_path = (
        Path(out_dir)
        if out_dir is not None
        else (team / "outputs" / "callcenter" / "profile" if team is not None else (guess_repo_root() / "outputs" / "callcenter" / "profile"))
    )
    out_dir_path.mkdir(parents=True, exist_ok=True)

    raw_train_path = raw_dir_path / "train.csv"
    raw_test_path = raw_dir_path / "test.csv"
    clean_train_path = clean_dir_path / cfg.clean_train_name
    clean_test_path = clean_dir_path / cfg.clean_test_name

    for p in [raw_train_path, raw_test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Raw file not found: {p}")
    for p in [clean_train_path, clean_test_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Clean file not found: {p}\n"
                "Tip: run make_clean_dataset.py first to generate train_clean.csv/test_clean.csv."
            )

    # ---- Load ----
    raw_train = pd.read_csv(raw_train_path)
    raw_test = pd.read_csv(raw_test_path)
    clean_train = pd.read_csv(clean_train_path)
    clean_test = pd.read_csv(clean_test_path)

    canon_to_raw = _invert_mapping(cfg.raw_to_canon)
    raw_cat_cols = [canon_to_raw.get(c, c) for c in cfg.cat_cols]

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paths": {
            "raw_train": str(raw_train_path),
            "raw_test": str(raw_test_path),
            "clean_train": str(clean_train_path),
            "clean_test": str(clean_test_path),
            "out_dir": str(out_dir_path),
        },
        "raw": {
            "train": {
                "rows": int(len(raw_train)),
                "columns": raw_train.columns.tolist(),
                "missing": _missing_table(raw_train),
                "categoricals": _cat_profile(raw_train, raw_cat_cols, top_k=top_k),
                "class_distribution": _class_distribution(raw_train, cfg.target_col_raw),
                "duplicates": {
                    "duplicate_id": int(raw_train[cfg.id_col_raw].duplicated().sum()) if cfg.id_col_raw in raw_train.columns else None,
                    "duplicate_sn": int(raw_train["SN"].duplicated().sum()) if "SN" in raw_train.columns else None,
                },
            },
            "test": {
                "rows": int(len(raw_test)),
                "columns": raw_test.columns.tolist(),
                "missing": _missing_table(raw_test),
                "categoricals": _cat_profile(raw_test, raw_cat_cols, top_k=top_k),
                "duplicates": {
                    "duplicate_id": int(raw_test[cfg.id_col_raw].duplicated().sum()) if cfg.id_col_raw in raw_test.columns else None,
                    "duplicate_sn": int(raw_test["SN"].duplicated().sum()) if "SN" in raw_test.columns else None,
                },
            },
        },
        "clean": {
            "train": {
                "rows": int(len(clean_train)),
                "columns": clean_train.columns.tolist(),
                "missing": _missing_table(clean_train),
                "categoricals": _cat_profile(clean_train, list(cfg.cat_cols), top_k=top_k, unk_token="UNK"),
                "class_distribution": _class_distribution(clean_train, cfg.target_col),
                "duplicates": {
                    "duplicate_id": int(clean_train[cfg.id_col].duplicated().sum()) if cfg.id_col in clean_train.columns else None,
                    "duplicate_sn": int(clean_train["sn"].duplicated().sum()) if "sn" in clean_train.columns else None,
                },
            },
            "test": {
                "rows": int(len(clean_test)),
                "columns": clean_test.columns.tolist(),
                "missing": _missing_table(clean_test),
                "categoricals": _cat_profile(clean_test, list(cfg.cat_cols), top_k=top_k, unk_token="UNK"),
                "duplicates": {
                    "duplicate_id": int(clean_test[cfg.id_col].duplicated().sum()) if cfg.id_col in clean_test.columns else None,
                    "duplicate_sn": int(clean_test["sn"].duplicated().sum()) if "sn" in clean_test.columns else None,
                },
            },
        },
        "notes": {
            "raw_cat_cols_used": raw_cat_cols,
            "clean_cat_cols_used": list(cfg.cat_cols),
            "unk_token_clean": "UNK",
        },
    }

    out_path = out_dir_path / "profile_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- Print compact summary ----
    def _top_high_card(section: dict[str, Any], n: int = 5) -> list[dict[str, Any]]:
        rows = section["categoricals"]["high_cardinality_sorted"]
        return rows[:n]

    print("[profile] RAW high-cardinality categoricals (train):")
    for r in _top_high_card(report["raw"]["train"]):
        print(f"- {r['column']}: unique={r['unique']}")
    print("[profile] CLEAN high-cardinality categoricals (train):")
    for r in _top_high_card(report["clean"]["train"]):
        print(f"- {r['column']}: unique={r['unique']}")

    raw_cls = report["raw"]["train"].get("class_distribution") or {}
    clean_cls = report["clean"]["train"].get("class_distribution") or {}
    print("[profile] Class distribution (raw train):", raw_cls)
    print("[profile] Class distribution (clean train):", clean_cls)
    print(f"[profile] Saved: {out_path}")

    return report


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Profile Call Center data (raw vs clean).")
    p.add_argument("--team_dir", type=str, default=None, help="Drive root, e.g. /content/drive/MyDrive/FORSA_team")
    p.add_argument("--raw_dir", type=str, default=None, help="Override raw data dir (contains train.csv/test.csv).")
    p.add_argument("--clean_dir", type=str, default=None, help="Override clean dir (contains train_clean.csv/test_clean.csv).")
    p.add_argument("--out_dir", type=str, default=None, help="Override output directory for the report.")
    p.add_argument("--top_k", type=int, default=20, help="Top-K values per categorical to save.")
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    profile_callcenter_raw_vs_clean(
        team_dir=args.team_dir,
        raw_dir=args.raw_dir,
        clean_dir=args.clean_dir,
        out_dir=args.out_dir,
        top_k=int(args.top_k),
    )


if __name__ == "__main__":
    main()


