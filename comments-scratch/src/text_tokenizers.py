"""
Tiny tokenizers/vocab builders for scratch models.

Named `text_tokenizers` to avoid clashing with the external `tokenizers` package.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PAD = "<PAD>"
UNK = "<UNK>"


@dataclass(frozen=True)
class Vocab:
    id2token: list[str]
    token2id: dict[str, int]

    @property
    def pad_id(self) -> int:
        return int(self.token2id[PAD])

    @property
    def unk_id(self) -> int:
        return int(self.token2id[UNK])

    @property
    def size(self) -> int:
        return int(len(self.id2token))


def _dedup_preserve_order(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def save_vocab(vocab: Vocab, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"id2token": vocab.id2token}, ensure_ascii=False, indent=2), encoding="utf-8")


def load_vocab(path: str | Path) -> Vocab:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    id2token = list(data["id2token"])
    token2id = {t: i for i, t in enumerate(id2token)}
    if PAD not in token2id or UNK not in token2id:
        raise ValueError(f"Invalid vocab (missing {PAD}/{UNK}) at: {p}")
    return Vocab(id2token=id2token, token2id=token2id)


def build_word_vocab(
    texts: Iterable[str],
    *,
    max_vocab: int = 50_000,
    min_freq: int = 2,
) -> Vocab:
    """
    Whitespace tokenization. Keeps special tokens like <PHONE> if separated by spaces.
    """

    c = Counter()
    for t in texts:
        toks = str(t or "").split()
        c.update(toks)

    # Reserve PAD/UNK at fixed ids
    id2 = [PAD, UNK]
    for tok, freq in c.most_common():
        if tok in (PAD, UNK):
            continue
        if freq < int(min_freq):
            continue
        id2.append(tok)
        if len(id2) >= int(max_vocab):
            break

    id2 = _dedup_preserve_order(id2)
    token2id = {t: i for i, t in enumerate(id2)}
    return Vocab(id2token=id2, token2id=token2id)


def build_char_vocab(
    texts: Iterable[str],
    *,
    max_vocab: int = 512,
    min_freq: int = 1,
) -> Vocab:
    c = Counter()
    for t in texts:
        c.update(list(str(t or "")))

    id2 = [PAD, UNK]
    for ch, freq in c.most_common():
        if ch in (PAD, UNK):
            continue
        if freq < int(min_freq):
            continue
        id2.append(ch)
        if len(id2) >= int(max_vocab):
            break

    id2 = _dedup_preserve_order(id2)
    token2id = {t: i for i, t in enumerate(id2)}
    return Vocab(id2token=id2, token2id=token2id)


def encode_words(text: str, vocab: Vocab, *, max_len: int) -> list[int]:
    toks = str(text or "").split()
    ids = [vocab.token2id.get(tok, vocab.unk_id) for tok in toks[: int(max_len)]]
    if len(ids) < int(max_len):
        ids = ids + [vocab.pad_id] * (int(max_len) - len(ids))
    return ids


def encode_chars(text: str, vocab: Vocab, *, max_len: int) -> list[int]:
    chars = list(str(text or ""))
    ids = [vocab.token2id.get(ch, vocab.unk_id) for ch in chars[: int(max_len)]]
    if len(ids) < int(max_len):
        ids = ids + [vocab.pad_id] * (int(max_len) - len(ids))
    return ids


