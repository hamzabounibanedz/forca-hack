"""
Language/script heuristics for comments-scratch.

Goal
----
We train 4 "experts" (FR / AR / DZ_arabic / DZ_latin). At inference time, we compute
simple script / marker features to produce per-comment gating weights.

This is intentionally heuristic (fast, dependency-free) because Kaggle test is large (~30k).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Rough script ranges
_AR_LETTER_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
_LAT_LETTER_RE = re.compile(r"[A-Za-z]")

# Arabizi digits typically used as Arabic letters in Latin-script Darija:
# 2 (ء), 3 (ع), 5 (خ), 6 (ط), 7 (ح), 8 (ق/غ), 9 (ق)
_ARABIZI_DIGITS = set("2356789")

# Detect "word-like" tokens containing both Latin letters and an Arabizi digit (e.g., "3lach", "ma3lich", "7abit").
_ARABIZI_TOKEN_RE = re.compile(r"\b[0-9A-Za-z]*[2356789][0-9A-Za-z]+\b", flags=re.IGNORECASE)

# Exclude common telecom digit+g tokens from Arabizi counting.
_EXCLUDE_ARABIZI_TOKENS = {"3g", "4g", "5g", "2g"}

# Darija markers (Arabic script) — not exhaustive, but high-signal.
_DARIJA_AR_MARKERS = (
    "واش",
    "علاش",
    "علاه",
    "علاه؟",
    "وقتاش",
    "وين",
    "مزال",
    "مازال",
    "راني",
    "رانا",
    "راهم",
    "راك",
    "بصح",
    "برك",
    "عندكم",
    "نحب",
    "حبيت",
    "نستناو",
    "نستنو",
    "مكانش",
    "ماكاش",
)

# Darija markers (Latin/Arabizi) — also partial.
_DARIJA_LAT_MARKERS = (
    "wesh",
    "wech",
    "chouia",
    "saha",
    "ma3lich",
    "ma3lish",
    "machi",
    "rak",
    "rani",
    "rana",
    "3lach",
    "3lah",
    "3la",
    "7abit",
    "7ab",
    "9al",
    "9alo",
)


@dataclass(frozen=True)
class ScriptStats:
    arabic_letters: int
    latin_letters: int
    arabizi_tokens: int

    @property
    def letters_total(self) -> int:
        return int(self.arabic_letters + self.latin_letters)

    @property
    def arabic_ratio(self) -> float:
        return float(self.arabic_letters / max(1, self.letters_total))

    @property
    def latin_ratio(self) -> float:
        return float(self.latin_letters / max(1, self.letters_total))


def script_stats(text: str) -> ScriptStats:
    t = text or ""
    ar = len(_AR_LETTER_RE.findall(t))
    lat = len(_LAT_LETTER_RE.findall(t))
    arabizi = 0
    for m in _ARABIZI_TOKEN_RE.finditer(t):
        tok = m.group(0).lower()
        if tok in _EXCLUDE_ARABIZI_TOKENS:
            continue
        if tok.isdigit():
            continue
        arabizi += 1
    return ScriptStats(arabic_letters=ar, latin_letters=lat, arabizi_tokens=arabizi)


def _has_any_marker(text: str, markers: tuple[str, ...]) -> bool:
    t = (text or "").lower()
    return any(m in t for m in markers)


def detect_group(text: str) -> str:
    """
    Return one of: "fr", "ar", "dz_ar", "dz_lat", "mixed".
    """

    t = text or ""
    stats = script_stats(t)
    has_dz_ar = _has_any_marker(t, _DARIJA_AR_MARKERS)
    has_dz_lat = _has_any_marker(t, _DARIJA_LAT_MARKERS) or stats.arabizi_tokens > 0

    # Dominant script
    if stats.arabic_ratio >= 0.55:
        return "dz_ar" if has_dz_ar else "ar"
    if stats.latin_ratio >= 0.55:
        return "dz_lat" if has_dz_lat else "fr"

    # Mixed / short: fall back to markers
    if has_dz_lat:
        return "dz_lat"
    if has_dz_ar:
        return "dz_ar"
    return "mixed"


def gating_weights(text: str, *, experts: tuple[str, ...]) -> dict[str, float]:
    """
    Per-comment weights over provided experts.

    Experts names must be subset of: ("fr","ar","dz_ar","dz_lat").
    """

    group = detect_group(text)
    # Default template (sums to 1.0)
    base = {
        "fr": 0.25,
        "ar": 0.25,
        "dz_ar": 0.25,
        "dz_lat": 0.25,
    }
    if group == "fr":
        base = {"fr": 0.70, "dz_lat": 0.20, "ar": 0.05, "dz_ar": 0.05}
    elif group == "dz_lat":
        base = {"dz_lat": 0.70, "fr": 0.20, "ar": 0.05, "dz_ar": 0.05}
    elif group == "ar":
        base = {"ar": 0.70, "dz_ar": 0.20, "fr": 0.05, "dz_lat": 0.05}
    elif group == "dz_ar":
        base = {"dz_ar": 0.70, "ar": 0.20, "fr": 0.05, "dz_lat": 0.05}

    # Keep only requested experts and renormalize
    out = {k: float(base.get(k, 0.0)) for k in experts}
    s = float(sum(out.values()))
    if s <= 0:
        # fall back: uniform over experts
        return {k: float(1.0 / len(experts)) for k in experts}
    return {k: float(v / s) for k, v in out.items()}


