#!/usr/bin/env python3
"""Build a noisy copy of a test CutSet by mixing in MUSAN noise.

Reuses the exact transform training uses (``lhotse.dataset.CutMix``) so the
noisy eval matches what the model was augmented with. Every test cut is mixed
with a random MUSAN cut at a random SNR; the result is a CutSet of MixedCuts
that reference the existing test and MUSAN features, so decoding mixes them at
load time (no new feature files written).

Both inputs must already carry fbank features computed with the same config
(80 mel bins, 16 kHz) - i.e. the test cuts from compute_fbank_huggingface.py and
the MUSAN cuts from local/compute_fbank_musan.py.

Example
-------
./make_noisy_cuts.py \
  --test-cuts data/fbank/vibank_cs_cuts_test.jsonl.gz \
  --musan-cuts data/fbank/musan_cuts.jsonl.gz \
  --snr-low 5 --snr-high 20 \
  --output data/fbank/vibank_cs_cuts_test_noisy.jsonl.gz
"""

import argparse
import logging
from pathlib import Path


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--test-cuts", type=Path, required=True)
    p.add_argument("--musan-cuts", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--snr-low", type=float, default=5.0)
    p.add_argument("--snr-high", type=float, default=20.0)
    p.add_argument(
        "--mix-prob",
        type=float,
        default=1.0,
        help="Probability each cut gets noise. 1.0 = every cut (for eval).",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )
    args = get_args()

    import torch
    from lhotse import CutSet, load_manifest
    from lhotse.dataset import CutMix

    # Deterministic mixing so the noisy eval is reproducible across runs.
    torch.manual_seed(args.seed)

    test = load_manifest(args.test_cuts)
    musan = load_manifest(args.musan_cuts)
    logging.info(f"Loaded {len(test)} test cuts, {len(musan)} MUSAN cuts.")

    mixer = CutMix(
        cuts=musan,
        p=args.mix_prob,
        snr=(args.snr_low, args.snr_high),
        preserve_id=True,
    )
    noisy: CutSet = mixer(test)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    noisy.to_file(args.output)
    logging.info(
        f"Wrote {len(noisy)} noisy cuts -> {args.output} "
        f"(snr={args.snr_low}-{args.snr_high} dB, p={args.mix_prob})."
    )


if __name__ == "__main__":
    main()
