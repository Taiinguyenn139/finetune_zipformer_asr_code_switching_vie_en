#!/usr/bin/env python3
"""Unit tests for hf_sync.latest_epoch.

Run: python -m pytest colab/test_hf_sync.py
or:  python colab/test_hf_sync.py   (plain asserts, no pytest needed)

Only the pure epoch-detection logic is covered; the push/pull commands are thin
huggingface_hub wrappers exercised against the live Hub, not in unit tests.
"""

from hf_sync import latest_epoch


def test_empty_returns_none():
    assert latest_epoch([]) is None


def test_no_epoch_files_returns_none():
    assert latest_epoch(["README.md", "bpe.model", "config.json"]) is None


def test_single_epoch():
    assert latest_epoch(["epoch-1.pt"]) == 1


def test_picks_highest_unordered():
    assert latest_epoch(["epoch-3.pt", "epoch-11.pt", "epoch-2.pt"]) == 11


def test_handles_gaps():
    # Missing epochs in the middle must not break max selection.
    assert latest_epoch(["epoch-1.pt", "epoch-5.pt", "epoch-9.pt"]) == 9


def test_directory_prefixes_are_matched():
    names = ["exp_finetune/epoch-7.pt", "exp_finetune/epoch-12.pt"]
    assert latest_epoch(names) == 12


def test_averaged_checkpoints_ignored():
    # epoch-35-avg-6.pt is an averaged export, not a resume point.
    assert latest_epoch(["epoch-35-avg-6.pt", "epoch-4.pt"]) == 4


def test_ignores_noise_around_epoch_files():
    names = ["tensorboard/events.out", "epoch-8.pt", "bpe.model", "epoch-10.pt"]
    assert latest_epoch(names) == 10


def test_does_not_match_partial_names():
    # Neither a different extension nor a name embedding the pattern should count.
    assert latest_epoch(["epoch-2.pt.tmp", "best-epoch-9.ckpt"]) is None


if __name__ == "__main__":
    import sys

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failures else 0)
