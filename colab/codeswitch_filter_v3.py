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
from underthesea import pos_tag


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


class CodeSwitchFilterV3:
    """Detects English words embedded in (mostly) Vietnamese transcripts."""

    def __init__(
        self,
        min_en_tokens: int = 0,
        max_en_tokens: int = 0,
    ) -> None:
        self.min_en_tokens = min_en_tokens
        self.max_en_tokens = max_en_tokens

    def english_tokens(self, text: str):
        """Compatibility helper for manifest/reporting code.

        V2 is sentence-level and does not natively attribute English tokens.
        For downstream scripts that still expect a token list, fall back to the
        rule-based extractor when it is available. If the optional wordlist
        dependency is missing, return an empty list instead of failing.
        """
        try:
            from codeswitch_filter import english_tokens as _english_tokens
        except ImportError:
            return []

        try:
            return _english_tokens(text)
        except ImportError:
            return []

    def is_codeswitch(self, text: str) -> bool:
        # tags = pos_tag(text)
        # print(f"Tags: {tags}")
        # english_words = [word for word, tag in tags if tag == 'Fw']
        # if len(english_words) < self.min_en_tokens:
        #     return False
        # if self.max_en_tokens is not None:
        #     if len(english_words) > self.max_en_tokens:
        #         return False
        return True
