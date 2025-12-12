"""
Call Center challenge configuration.

This module is intentionally "boring": only constants + tiny helpers.
Keeping all paths/column names here avoids confusion and prevents bugs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CallCenterConfig:
    # ----- Columns (raw -> canonical) -----
    id_col_raw: str = "id"
    target_col_raw: str = "class_int"

    # NOTE: These raw column names come from your uploaded CSV header.
    raw_to_canon: dict[str, str] = None  # type: ignore[assignment]

    # Canonical column names used AFTER renaming
    id_col: str = "id"
    target_col: str = "class_int"
    datetime_col: str = "handle_time"
    text_col: str = "service_content"

    # Categorical features (canonical names)
    cat_cols: tuple[str, ...] = (
        "tt_status",
        "dept_initial",
        "service_request_type",
        "subs_level",
        "customer_level",
        "dept_traitement",
        "tech_comm",
        "service_type",
        "dot",
        "actel",
        "actel_code",
    )

    # Columns we keep in cleaned dataset besides engineered features
    keep_cols: tuple[str, ...] = (
        "id",
        "sn",
        "handle_time",
        "service_content",
        "tt_status",
        "dept_initial",
        "service_request_type",
        "subs_level",
        "customer_level",
        "dept_traitement",
        "tech_comm",
        "service_type",
        "dot",
        "actel",
        "actel_code",
    )

    # ----- Preprocessing flags -----
    lower: bool = True
    remove_accents: bool = True

    # If True, we keep SN in cleaned outputs (for traceability) but it should NOT be used as a model feature.
    keep_sn_in_outputs: bool = True

    # ----- High-cardinality category handling -----
    # Categories with frequency < rare_min_count (in TRAIN) are mapped to rare_token for both train and test.
    rare_min_count: int = 20
    rare_token: str = "RARE"
    rare_group_cols: tuple[str, ...] = ("actel", "dot", "actel_code")

    # ----- Output filenames -----
    clean_train_name: str = "train_clean.csv"
    clean_test_name: str = "test_clean.csv"
    run_summary_name: str = "run_summary.json"


def default_config() -> CallCenterConfig:
    """
    Returns a CallCenterConfig with the raw->canonical column mapping filled in.
    """
    raw_to_canon = {
        "id": "id",
        "SN": "sn",
        "Handle time": "handle_time",
        "TT status": "tt_status",
        "Dept. for initial handling": "dept_initial",
        "Service request type": "service_request_type",
        "Subs Level": "subs_level",
        "Customer level": "customer_level",
        "Service content": "service_content",
        "Département de traitement": "dept_traitement",
        "Tech/Comm": "tech_comm",
        "Type": "service_type",
        "DOT": "dot",
        "ACTEL's": "actel",
        "class_int": "class_int",
    }
    return CallCenterConfig(raw_to_canon=raw_to_canon)


def guess_repo_root() -> Path:
    """
    Guess repo root from this file location: forca-hack/call-center/src/config.py
    """
    return Path(__file__).resolve().parents[2]


def guess_local_data_dir() -> Path:
    """
    Local fallback data dir (inside repo):
      forca-hack/call-center/data/
    """
    return guess_repo_root() / "call-center" / "data"


def guess_team_dir() -> Path | None:
    """
    Colab/Drive default:
      /content/drive/MyDrive/FORSA_team
    Returns None if it doesn't exist.
    """
    p = Path("/content/drive/MyDrive/FORSA_team")
    return p if p.exists() else None


def resolve_callcenter_data_dir(team_dir: Path) -> Path:
    """
    We try common variants for the call center data folder name.
    Expected: <team_dir>/data/callcenter/
    """
    candidates = [
        team_dir / "data" / "callcenter",
        team_dir / "data" / "call-center",
        team_dir / "data" / "call_center",
        team_dir / "data" / "call centre",
        team_dir / "data" / "callcenter_data",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Default (even if it doesn't exist) for clearer error messages later
    return team_dir / "data" / "callcenter"


def resolve_callcenter_output_dir(team_dir: Path) -> Path:
    """
    Output dir in Drive:
      <team_dir>/outputs/callcenter/clean/
    """
    return team_dir / "outputs" / "callcenter" / "clean"


