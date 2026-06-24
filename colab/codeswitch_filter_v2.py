#!/usr/bin/env python3
"""Code-switch (Vietnamese + English) utterance detector v2.

Uses statistical machine learning (`fast-langdetect`) to determine if a
transcript contains a genuine mix of Vietnamese and English, replacing
the traditional manual rule/wordlist heuristic.
"""

from __future__ import annotations

import re
from typing import Optional, Dict
from fast_langdetect import detect

# Still useful for calculating total token counts for ratio checks
_LETTER_RUN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


class CodeSwitchFilterV2:
    """Detects English code-switching in Vietnamese transcripts using ML models."""

    def __init__(
        self,
        min_en_prob: float = 0.15,
        max_en_prob: float = 0.75,
        min_vi_prob: float = 0.10,
    ) -> None:
        """
        Args:
            min_en_prob: Minimum confidence score required to confirm English is present.
            max_en_prob: If English score is higher than this, it's likely pure English (not code-switch).
            min_vi_prob: Minimum confidence score required to ensure a Vietnamese backbone exists.
        """
        self.min_en_prob = min_en_prob
        self.max_en_prob = max_en_prob
        self.min_vi_prob = min_vi_prob

    def is_codeswitch(self, text: str) -> bool:
        """True if `text` is determined to be a VI+EN code-switched utterance."""
        if not text or not text.strip():
            return False

        # fast-langdetect's detect(k=N) returns the top-N candidates ordered by
        # score: [{'lang': 'vi', 'score': 0.78}, {'lang': 'en', 'score': 0.21}, ...].
        # k=5 captures both the VI backbone and the EN code-switch among top langs.
        predictions = detect(text, k=5)
        
        # Parse into a flat dictionary mapping lang -> score
        scores: Dict[str, float] = {p['lang']: p['score'] for p in predictions}
        
        en_score = scores.get('en', 0.0)
        vi_score = scores.get('vi', 0.0)

        # Heuristic 1: The script must contain a solid foundation of Vietnamese
        if vi_score < self.min_vi_prob:
            return False

        # Heuristic 2: English must be strong enough to be distinct, but not completely dominant
        if self.min_en_prob <= en_score <= self.max_en_prob:
            return True

        return False


# --- Global Singleton instance for functional/convenience wrappers ---
_DEFAULT_FILTER: Optional[CodeSwitchFilterV2] = None

def _get_default_filter() -> CodeSwitchFilterV2:
    global _DEFAULT_FILTER
    if _DEFAULT_FILTER is None:
        _DEFAULT_FILTER = CodeSwitchFilterV2()
    return _DEFAULT_FILTER


def contains_english(text: str) -> bool:
    """Convenience wrapper using the default statistical filter model."""
    return _get_default_filter().is_codeswitch(text)


# --- Plain Script Execution & Manual Testing ---
if __name__ == "__main__":
    detector = CodeSwitchFilterV2()
    
    test_cases = {
        "Ngành Logistic đang rất hot tại Việt Nam.": True,       # Mixed code-switch
        "Mở app lên kiểm tra số dư": True,                       # Mixed short sentence
        "Tôi muốn chuyển khoản đến ngân hàng.": False,           # Pure Vietnamese
        "Please review the budget report by tomorrow morning.": False, # Pure English
        "ban con tin nhan": False,                                # ASCII Vietnamese syllables
    }

    print("Running evaluation on v2 model:")
    passed = 0
    for text, expected in test_cases.items():
        result = detector.is_codeswitch(text)
        status = "PASS" if result == expected else "FAIL"
        print(f"  [{status}] Text: '{text}' -> Expected: {expected}, Got: {result}")
        if result == expected:
            passed += 1

    print(f"\nResult: {passed}/{len(test_cases)} cases behaved as expected.")