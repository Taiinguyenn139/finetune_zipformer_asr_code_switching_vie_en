#!/usr/bin/env python3
"""Unit tests for codeswitch_filter.

Run: python -m pytest colab/test_codeswitch_filter.py
or:  python colab/test_codeswitch_filter.py   (plain asserts, no pytest needed)

A small English wordlist is injected so the tests need neither wordfreq nor any
network access.
"""

from codeswitch_filter import CodeSwitchFilter

try:
    from codeswitch_filter_v2 import CodeSwitchFilterV2
except ModuleNotFoundError:
    CodeSwitchFilterV2 = None

# Minimal English wordlist sufficient for these cases. The detector also folds
# in its built-in domain allowlist (logistic, app, ...).
_EN = {"hot", "online", "meeting", "deadline", "card", "good", "morning"}


def make_filter(**kw) -> CodeSwitchFilter:
    return CodeSwitchFilter(english_words=_EN, **kw)


def test_selects_codeswitch_example():
    f = make_filter()
    text = "Ngành Logistic đang rất hot tại Việt Nam."
    assert f.is_codeswitch(text)
    assert set(f.english_tokens(text)) == {"logistic", "hot"}


def test_pure_vietnamese_rejected():
    f = make_filter()
    # Diacritic-bearing Vietnamese only -> no English.
    assert not f.is_codeswitch("Tôi muốn chuyển khoản đến ngân hàng.")
    assert f.english_tokens("Tôi muốn chuyển khoản đến ngân hàng.") == []


def test_ascii_vietnamese_not_flagged():
    f = make_filter()
    # "ban", "con", "tin" are ASCII Vietnamese syllables in the exclusion set,
    # even though some are English words.
    assert not f.is_codeswitch("ban con tin nhan")


def test_domain_term_beats_vi_exclusion():
    f = make_filter()
    # "app" is in the domain allowlist; must be detected even amid Vietnamese.
    text = "Mở app lên kiểm tra số dư"
    assert f.is_codeswitch(text)
    assert "app" in f.english_tokens(text)


def test_min_en_tokens_threshold():
    f = make_filter(min_en_tokens=2)
    assert not f.is_codeswitch("Cái này rất hot")  # only 1 English token
    assert f.is_codeswitch("Cái deadline meeting này")  # 2 English tokens


def test_max_en_ratio_drops_full_english():
    f = make_filter(max_en_ratio=0.5)
    assert not f.is_codeswitch("good morning online meeting")  # all English
    assert f.is_codeswitch("Sáng nay có meeting online không")  # mixed


def test_digits_and_punctuation_ignored():
    f = make_filter()
    toks = f.english_tokens("Số 123, thẻ card-2024!")
    assert toks == ["card"]


def test_v2_exposes_compatibility_english_tokens():
    if CodeSwitchFilterV2 is None:
        return
    f = CodeSwitchFilterV2()
    toks = f.english_tokens("Ngành Logistic đang rất hot tại Việt Nam.")
    assert isinstance(toks, list)


if __name__ == "__main__":
    import sys

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
    sys.exit(0)
