"""
Social Media Comments challenge configuration.

Why this exists
---------------
The comments dataset comes with French column headers (sometimes with encoding quirks),
and the text is multilingual (Arabic script + French + Darija/Arabizi).

This module centralizes:
- canonical column names used by OUR pipeline
- preprocessing switches (so experiments stay reproducible)
- path helpers (local repo vs Google Colab Drive)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class CommentsConfig:
    # ----- Canonical column names (after schema normalization) -----
    id_col: str = "id"
    platform_col: str = "platform"
    text_col: str = "comment"
    target_col: str = "label"

    # ----- Output filenames -----
    clean_train_name: str = "train_clean.csv"
    clean_test_name: str = "test_clean.csv"
    run_summary_name: str = "run_summary.json"

    # ----- Preprocessing flags -----
    # Keep lower=True for classical models; it does not affect Arabic letters.
    lower: bool = True
    # Remove accents on Latin chars (helps TF-IDF sparsity); has no impact on Arabic.
    remove_accents_latin: bool = True
    # Arabic normalization (diacritics/tatweel + a few letter variants).
    normalize_arabic: bool = True

    # PII/system noise
    mask_pii: bool = True
    mask_long_ids: bool = True
    long_id_min_digits: int = 6

    # Text normalization
    reduce_elongations: bool = True
    canonicalize_telecom: bool = True
    collapse_whitespace: bool = True

    # Traceability: keep raw fields as *_raw columns in cleaned outputs
    keep_raw_cols: bool = True

    # ----- Special tokens (kept stable for features + transformers) -----
    phone_token: str = "<PHONE>"
    email_token: str = "<EMAIL>"
    url_token: str = "<URL>"
    id_token: str = "<ID>"


def cfg_to_dict(cfg: CommentsConfig) -> dict:
    """
    Serialize config to a JSON-friendly dict (for reproducibility).
    """
    return asdict(cfg)


def cfg_from_dict(data: dict) -> CommentsConfig:
    """
    Restore config from a dict produced by cfg_to_dict().
    """
    return CommentsConfig(**data)


def guess_repo_root() -> Path:
    """
    Guess repo root from this file location: forca-hack/comments/src/config.py
    """
    return Path(__file__).resolve().parents[2]


def guess_local_data_dir() -> Path:
    """
    Local fallback data dir:
      forca-hack/comments/data/
    """
    return guess_repo_root() / "comments" / "data"


def guess_team_dir() -> Path | None:
    """
    Colab/Drive default:
      /content/drive/MyDrive/FORSA_team
    Returns None if it doesn't exist.
    """
    p = Path("/content/drive/MyDrive/FORSA_team")
    return p if p.exists() else None


def resolve_comments_data_dir(team_dir: Path) -> Path:
    """
    We try common variants for the comments data folder name.
    Expected: <team_dir>/data/comments/
    """
    candidates = [
        team_dir / "data" / "comments",
        team_dir / "data" / "comment",
        team_dir / "data" / "social",
        team_dir / "data" / "social_media",
        team_dir / "data" / "social-media",
        team_dir / "data" / "challenge1",
        team_dir / "data" / "comments_data",
    ]
    for c in candidates:
        if c.exists():
            return c
    return team_dir / "data" / "comments"


def resolve_comments_output_dir(team_dir: Path) -> Path:
    """
    Output dir in Drive:
      <team_dir>/outputs/comments/
    """
    return team_dir / "outputs" / "comments"


