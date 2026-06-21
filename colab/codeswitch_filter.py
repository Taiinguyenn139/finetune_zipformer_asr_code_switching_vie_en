#!/usr/bin/env python3
"""Code-switch (Vietnamese + English) utterance detector.

Decides whether a transcript contains genuine *English* word(s) embedded in
Vietnamese text, e.g.::

    "Ngành Logistic đang rất hot tại Việt Nam."  -> English: ["logistic", "hot"]

Why not a regex?
----------------
Vietnamese is written in the Latin alphabet, so "has a Latin-script token" is
useless - every Vietnamese word matches. Instead we test each ASCII token for
membership in an English wordlist while *excluding* common Vietnamese syllables
that happen to be spelled like English words (e.g. "ban", "con", "tin"). This is
a tunable heuristic with a manual-review gate, not a perfect language model -
per-token statistical language-id is unreliable on 1-2 char code-switch tokens.

The English wordlist is loaded from the ``wordfreq`` package by default, but any
iterable of words can be injected (the unit tests do exactly that, so they run
without ``wordfreq`` installed).
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Set

# Lowercase Vietnamese-specific letters. A token containing any of these is
# unmistakably Vietnamese, so we never count it as English.
_VI_DIACRITIC_CHARS: Set[str] = set(
    "ăâđêôơư"
    "áàảãạ"
    "ắằẳẵặ"
    "ấầẩẫậ"
    "éèẻẽẹ"
    "ếềểễệ"
    "íìỉĩị"
    "óòỏõọ"
    "ốồổỗộ"
    "ớờởỡợ"
    "úùủũụ"
    "ứừửữự"
    "ýỳỷỹỵ"
)

# Common ASCII Vietnamese syllables that are ALSO valid English words. These are
# excluded so a Vietnamese sentence written without diacritics (or using these
# genuinely-Vietnamese syllables) is not mistaken for code-switch. Extend this
# via the ``vi_syllables`` argument / a vi_syllables.txt file when you see
# false positives in samples_selected.txt. Lowercase, ASCII only.
_VI_ASCII_SYLLABLES: Set[str] = {
    "an", "anh", "ba", "ban", "bao", "ben", "bi", "bo", "bon", "but",
    "ca", "cac", "cai", "cam", "can", "cao", "chi", "cho", "chu", "co",
    "con", "cua", "da", "dan", "dao", "di", "do", "don", "du", "em",
    "ga", "gan", "gia", "han", "hat", "hen", "hi", "ho", "hoa", "hon",
    "khi", "la", "lai", "lam", "lan", "le", "len", "lo", "loi", "long",
    "ma", "mai", "man", "me", "mi", "min", "mo", "moi", "mon", "mua",
    "na", "nam", "no", "noi", "non", "oi", "om", "on", "ong", "pho",
    "phi", "qua", "ra", "sa", "sang", "sao", "set", "so", "son", "ta",
    "tai", "tam", "tan", "tao", "ten", "thi", "tin", "to", "toi", "tom",
    "ton", "tra", "tu", "va", "vi", "vo", "xa", "xe", "xin", "xu",
}

# Domain (banking / tech) English terms that wordfreq may rank below the cutoff
# but are exactly the code-switch we want to catch. Lowercase, ASCII.
_DOMAIN_ALLOWLIST: Set[str] = {
    "logistic", "logistics", "app", "online", "offline", "internet",
    "banking", "mobile", "card", "credit", "debit", "transfer", "balance",
    "ok", "deadline", "meeting", "marketing", "sale", "sales", "team",
    "email", "feedback", "review", "manager", "report", "budget", "deal",
    "voucher", "cashback", "checkout", "payment", "wallet", "qr", "otp",
}

# Runs of (unicode) letters only: splits on whitespace, punctuation and digits,
# so "card-2024" -> "card", "app/web" -> "app", "web". Vietnamese diacritics are
# unicode letters and are preserved within a token.
_LETTER_RUN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _load_lines(path: Optional[Path]) -> Optional[Set[str]]:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    words = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        w = line.strip().lower()
        if w and not w.startswith("#"):
            words.add(w)
    return words


@lru_cache(maxsize=1)
def _default_english_words() -> frozenset:
    """Top English words from wordfreq. Raises a clear error if missing."""
    try:
        from wordfreq import top_n_list
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ImportError(
            "wordfreq is required for the default English wordlist. "
            "Install it (`pip install wordfreq`) or pass english_words=... "
            "explicitly."
        ) from e
    return frozenset(w.lower() for w in top_n_list("en", 50_000))


class CodeSwitchFilter:
    """Detects English words embedded in (mostly) Vietnamese transcripts."""

    def __init__(
        self,
        english_words: Optional[Iterable[str]] = None,
        vi_syllables: Optional[Iterable[str]] = None,
        domain_allowlist: Optional[Iterable[str]] = None,
        min_en_tokens: int = 1,
        min_len: int = 2,
        max_en_ratio: Optional[float] = None,
    ) -> None:
        """
        Args:
          english_words: iterable of lowercase English words. Defaults to the
            top 50k from wordfreq.
          vi_syllables: ASCII Vietnamese syllables to exclude from the English
            count (in addition to the built-in set).
          domain_allowlist: extra English terms to always treat as English
            (in addition to the built-in banking/tech set).
          min_en_tokens: minimum number of distinct English tokens for an
            utterance to be selected.
          min_len: minimum token length to be considered an English candidate
            (filters out 1-char noise).
          max_en_ratio: if set, reject utterances whose English-token ratio
            exceeds this (use e.g. 0.6 to keep *true* code-switch and drop
            fully-English utterances). None disables the check.
        """
        if english_words is None:
            english_words = _default_english_words()
        self.english: Set[str] = {w.lower() for w in english_words}

        self.allowlist: Set[str] = set(_DOMAIN_ALLOWLIST)
        if domain_allowlist:
            self.allowlist.update(w.lower() for w in domain_allowlist)
        self.english |= self.allowlist

        self.vi_syllables: Set[str] = set(_VI_ASCII_SYLLABLES)
        if vi_syllables:
            self.vi_syllables.update(w.lower() for w in vi_syllables)
        # Domain terms win over the Vietnamese-syllable exclusion.
        self.vi_syllables -= self.allowlist

        self.min_en_tokens = min_en_tokens
        self.min_len = min_len
        self.max_en_ratio = max_en_ratio

    def english_tokens(self, text: str) -> List[str]:
        """Return the list of tokens classified as English (in order seen)."""
        if not text:
            return []
        found: List[str] = []
        for raw in _LETTER_RUN_RE.findall(text):
            tok = raw.lower()
            if any(c in _VI_DIACRITIC_CHARS for c in tok):
                continue  # has Vietnamese diacritic -> Vietnamese
            if not tok.isascii():
                continue  # non-ascii letters (other scripts) -> not English
            if len(tok) < self.min_len:
                continue
            if tok in self.vi_syllables:
                continue  # ASCII Vietnamese syllable -> not English
            if tok in self.english:
                found.append(tok)
        return found

    def is_codeswitch(self, text: str) -> bool:
        """True if `text` should be selected as a VI+EN code-switch utterance."""
        en = self.english_tokens(text)
        if len(set(en)) < self.min_en_tokens:
            return False
        if self.max_en_ratio is not None:
            total = len(_LETTER_RUN_RE.findall(text)) or 1
            if len(en) / total > self.max_en_ratio:
                return False
        return True


@lru_cache(maxsize=1)
def _default_filter() -> CodeSwitchFilter:
    return CodeSwitchFilter()


def contains_english(text: str) -> bool:
    """Convenience wrapper using a default (wordfreq-backed) filter."""
    return _default_filter().is_codeswitch(text)


def english_tokens(text: str) -> List[str]:
    """Convenience wrapper returning English tokens with the default filter."""
    return _default_filter().english_tokens(text)
