#!/usr/bin/env python3
# Copyright    2024  Xiaomi Corp.        (authors: icefall contributors)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Prepare fine-tuning data for the Zipformer recipe directly from a
HuggingFace dataset, using Lhotse's native HuggingFace bridge
(CutSet.from_huggingface_dataset).

It produces precomputed 80-dim Fbank cuts in the exact format the trainer
expects:

    data/fbank/<prefix>_cuts_<output-name>.jsonl.gz

These cuts can then be loaded by zipformer/finetune.py (see the
"--manifest-dir" option and the data module).

The in-memory audio coming from HuggingFace is dropped after feature
extraction, so the saved cuts only carry features + supervisions and stay
small (suitable for the default PrecomputedFeatures input strategy).

Requirements (training environment):
    pip install "lhotse>=1.20" "datasets>=2.0" soundfile

Examples
--------

# 1) A dataset that already uses 16 kHz audio, text in the "text" column:
./local/compute_fbank_huggingface.py \
  --dataset mozilla-foundation/common_voice_17_0 \
  --name en \
  --split train \
  --output-name train \
  --prefix cv_en \
  --text-key sentence \
  --text-normalization upper-no-punct

# 2) A 48 kHz dataset that must be resampled to 16 kHz for Zipformer:
./local/compute_fbank_huggingface.py \
  --dataset my-org/my-asr-data \
  --split validation \
  --output-name dev \
  --prefix mydata \
  --resample-rate 16000

Run once per split (train / dev / test).
"""

import argparse
import logging
import os
import re
import string
from pathlib import Path
from typing import Optional

import torch

# Torch's multithreaded behavior needs to be disabled or it wastes a lot of
# CPU and slows things down. Do this outside of main() so it also takes effect
# in spawned feature-extraction subprocesses.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Zipformer / LibriSpeech features are computed at 16 kHz. Anything else must
# be resampled with --resample-rate or the features will not match the
# pre-trained checkpoint.
EXPECTED_SAMPLING_RATE = 16000

# Drop these punctuation characters in the "upper-no-punct" / "no-punct"
# normalizations (keeps apostrophes, which are part of LibriSpeech words).
_PUNCT_TO_STRIP = "".join(c for c in string.punctuation if c != "'")
_PUNCT_RE = re.compile(f"[{re.escape(_PUNCT_TO_STRIP)}]")
_WS_RE = re.compile(r"\s+")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )

    # --- HuggingFace dataset selection -------------------------------------
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="HuggingFace dataset repo id or local path, e.g. "
        "'mozilla-foundation/common_voice_17_0'.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional HuggingFace dataset config/subset name (e.g. a "
        "language code like 'en'). Passed as the second positional arg to "
        "datasets.load_dataset.",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="HuggingFace split to load, e.g. 'train', 'validation', 'test'. "
        "Slicing syntax such as 'train[:1%%]' is also accepted.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional cache directory for datasets.load_dataset.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to datasets.load_dataset "
        "(needed by some dataset loading scripts).",
    )
    parser.add_argument(
        "--load-from-disk",
        action="store_true",
        help="Treat --dataset as a local directory written by "
        "datasets.save_to_disk (e.g. the combined dataset produced by "
        "colab/materialize_dataset.py) and load it with "
        "datasets.load_from_disk instead of load_dataset. When the saved "
        "object is a DatasetDict, --split selects which split to use.",
    )

    # --- Column keys -------------------------------------------------------
    parser.add_argument(
        "--audio-key",
        type=str,
        default="audio",
        help="Name of the audio column in the HuggingFace dataset. The column "
        "may either be a decoded HuggingFace Audio feature or a plain string "
        "column holding the path to each audio file (see --audio-dir).",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="Base directory for the audio files when the audio column stores "
        "file paths (a plain string column) instead of decoded audio. Relative "
        "paths are resolved against this directory; absolute paths are used "
        "as-is. Ignored when the audio column is already a decoded Audio "
        "feature.",
    )
    parser.add_argument(
        "--text-key",
        type=str,
        default="text",
        help="Name of the transcript column (e.g. 'text', 'sentence', "
        "'transcription').",
    )

    # --- Output ------------------------------------------------------------
    parser.add_argument(
        "--prefix",
        type=str,
        default="hf",
        help="Filename prefix for the output cuts: "
        "<prefix>_cuts_<output-name>.jsonl.gz.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        required=True,
        help="Partition name used in the output filename, e.g. "
        "'train', 'dev', 'test'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/fbank"),
        help="Directory where cuts and feature files are written. This must "
        "match the trainer's --manifest-dir.",
    )

    # --- Feature / audio options ------------------------------------------
    parser.add_argument(
        "--resample-rate",
        type=int,
        default=None,
        help="If set, resample audio to this rate (Hz) before feature "
        f"extraction. Zipformer expects {EXPECTED_SAMPLING_RATE} Hz; set this "
        "when your dataset uses a different rate. If omitted, the native rate "
        "is used (a warning is printed when it is not "
        f"{EXPECTED_SAMPLING_RATE} Hz).",
    )
    parser.add_argument(
        "--num-mel-bins",
        type=int,
        default=80,
        help="Number of mel bins for Fbank. Must match the trained model "
        "(80 for the standard Zipformer).",
    )
    parser.add_argument(
        "--text-normalization",
        type=str,
        default="none",
        choices=["none", "upper", "lower", "upper-no-punct", "no-punct"],
        help="Transcript normalization. MUST be consistent with how the BPE "
        "model was trained. The shipped LibriSpeech bpe.model is UPPERCASE "
        "without punctuation -> use 'upper-no-punct' to reuse it.",
    )
    parser.add_argument(
        "--perturb-speed",
        action="store_true",
        help="Apply 3x speed perturbation (0.9 / 1.0 / 1.1) on the split. "
        "Usually only useful for training splits.",
    )
    parser.add_argument(
        "--bpe-model",
        type=str,
        default=None,
        help="Optional path to a bpe.model. When given, cuts whose number of "
        "post-subsampling frames is smaller than the number of tokens are "
        "removed (reuses local/filter_cuts.py).",
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=min(15, os.cpu_count() or 1),
        help="Number of parallel jobs for feature extraction.",
    )

    return parser.parse_args()


def normalize_text(text: str, mode: str) -> str:
    """Normalize a single transcript according to `mode`.

    Apostrophes are preserved for the punctuation-stripping modes so that
    LibriSpeech-style contractions (e.g. DON'T) stay intact.
    """
    if text is None:
        return ""
    if mode in ("upper-no-punct", "no-punct"):
        text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    if mode in ("upper", "upper-no-punct"):
        text = text.upper()
    elif mode == "lower":
        text = text.lower()
    return text


def main() -> None:
    args = get_args()
    logging.info(vars(args))

    # Imported here (not at module top) so the script can be inspected without
    # the heavy training-env dependencies installed.
    from datasets import Audio, DatasetDict, load_dataset, load_from_disk
    from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter
    from lhotse.utils import fastcopy

    from icefall.utils import get_executor

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cuts_filename = f"{args.prefix}_cuts_{args.output_name}.jsonl.gz"
    cuts_path = output_dir / cuts_filename
    if cuts_path.is_file():
        logging.info(f"{cuts_path} already exists - skipping.")
        return

    # --- Stage 1: load the dataset -----------------------------------------
    if args.load_from_disk:
        logging.info(
            f"Loading dataset from disk {args.dataset} (split={args.split})"
        )
        ds = load_from_disk(args.dataset)
        if isinstance(ds, DatasetDict):
            if args.split not in ds:
                raise ValueError(
                    f"Split '{args.split}' not found in {args.dataset}. "
                    f"Available splits: {list(ds.keys())}. Use --split."
                )
            ds = ds[args.split]
    else:
        logging.info(
            f"Loading HuggingFace dataset {args.dataset} (split={args.split})"
        )
        load_kwargs = {"split": args.split, "cache_dir": args.cache_dir}
        if args.trust_remote_code:
            load_kwargs["trust_remote_code"] = True
        if args.name is not None:
            ds = load_dataset(args.dataset, args.name, **load_kwargs)
        else:
            ds = load_dataset(args.dataset, **load_kwargs)

    if args.audio_key not in ds.column_names:
        raise ValueError(
            f"Audio column '{args.audio_key}' not found. "
            f"Available columns: {ds.column_names}. Use --audio-key."
        )
    if args.text_key not in ds.column_names:
        raise ValueError(
            f"Text column '{args.text_key}' not found. "
            f"Available columns: {ds.column_names}. Use --text-key."
        )

    # --- Stage 1b: path-only audio columns --------------------------------
    # Some datasets do not ship decoded audio; the audio column is a plain
    # string holding a file path, with the actual audio stored elsewhere.
    # Detect this (the column is not a decoded Audio feature) and turn it into
    # a real Audio feature so the rest of the pipeline is unchanged.
    if not isinstance(ds.features[args.audio_key], Audio):
        logging.info(
            f"Audio column '{args.audio_key}' holds file paths, not decoded "
            "audio; resolving paths and casting to an Audio feature."
        )
        if args.audio_dir is None:
            logging.warning(
                "No --audio-dir given for a path-only audio column; relative "
                "paths will be resolved against the current working directory."
            )

        audio_dir = args.audio_dir

        def _resolve_audio_path(example):
            path = example[args.audio_key]
            if path and audio_dir is not None and not os.path.isabs(path):
                path = str(audio_dir / path)
            example[args.audio_key] = path
            return example

        ds = ds.map(_resolve_audio_path)
        ds = ds.cast_column(args.audio_key, Audio())

    # --- Stage 2: optional resampling (decode-time cast) -------------------
    if args.resample_rate is not None:
        logging.info(f"Resampling audio to {args.resample_rate} Hz")
        ds = ds.cast_column(args.audio_key, Audio(sampling_rate=args.resample_rate))
    elif args.resample_rate is None:
        logging.warning(
            "No --resample-rate given. Zipformer expects "
            f"{EXPECTED_SAMPLING_RATE} Hz audio; if this dataset uses a "
            "different rate the features will NOT match the pre-trained model."
        )

    # --- Stage 3: bridge into a Lhotse CutSet (Option A) -------------------
    logging.info("Converting to Lhotse CutSet via from_huggingface_dataset")
    cut_set = CutSet.from_huggingface_dataset(
        ds,
        audio_key=args.audio_key,
        text_key=args.text_key,
    )

    # --- Stage 4: text normalization (must match the BPE model) -----------
    if args.text_normalization != "none":
        logging.info(f"Normalizing text: {args.text_normalization}")

        def _norm_sup(sup):
            return fastcopy(
                sup, text=normalize_text(sup.text, args.text_normalization)
            )

        cut_set = cut_set.map_supervisions(_norm_sup)

    # --- Stage 5: optional speed perturbation -----------------------------
    if args.perturb_speed:
        logging.info("Applying 3x speed perturbation (0.9 / 1.0 / 1.1)")
        cut_set = (
            cut_set + cut_set.perturb_speed(0.9) + cut_set.perturb_speed(1.1)
        )

    # --- Stage 6: optional bpe-based filtering (T >= S) -------------------
    if args.bpe_model:
        import sentencepiece as spm
        from filter_cuts import filter_cuts

        logging.info(f"Filtering cuts using bpe model {args.bpe_model}")
        sp = spm.SentencePieceProcessor()
        sp.load(args.bpe_model)
        cut_set = filter_cuts(cut_set, sp)

    # --- Stage 7: compute & store Fbank, then drop raw audio --------------
    extractor = Fbank(FbankConfig(num_mel_bins=args.num_mel_bins))
    storage_path = f"{output_dir}/{args.prefix}_feats_{args.output_name}"
    logging.info(f"Computing Fbank features -> {storage_path}")

    with get_executor() as ex:
        cut_set = cut_set.compute_and_store_features(
            extractor=extractor,
            storage_path=storage_path,
            num_jobs=args.num_jobs if ex is None else 80,
            executor=ex,
            storage_type=LilcomChunkyWriter,
        )

    # Drop the in-memory HuggingFace audio so the saved cuts stay small and
    # load via the precomputed-features path used by finetune.py.
    cut_set = cut_set.drop_recordings()

    cut_set.to_file(cuts_path)
    logging.info(f"Saved {cuts_path}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
