#!/usr/bin/env python3
"""Materialize the combined VI+EN dataset from a selection manifest.

Reads `combined_manifest.jsonl` (from select_codeswitch_utterances.py),
re-streams each source keeping only the selected rows (audio + transcript),
concatenates across sources, splits train/dev, and writes the result with
`datasets.save_to_disk`. Feed the output to compute_fbank_huggingface.py with
`--load-from-disk`.

Audio is taken as raw encoded bytes (Audio(decode=False)) - no re-encoding - and
the output Audio feature carries the target sampling rate so fbank resamples at
read time. Streaming + from_generator keeps memory low (no full split in RAM).

Index alignment: the manifest's `index` is the streaming position recorded at
selection time; this script re-streams in the same order. Each row's transcript
is checked against the manifest text and mismatches are warned about (the
manifest text is always the one written).

Example
-------
./materialize_dataset.py \
  --manifest /content/drive/MyDrive/vibank_cs/combined_manifest.jsonl \
  --output-dir /content/drive/MyDrive/vibank_cs/combined_dataset \
  --dev-frac 0.02
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="save_to_disk target (a Google Drive path on Colab).",
    )
    p.add_argument("--sampling-rate", type=int, default=16000)
    p.add_argument(
        "--dev-frac",
        type=float,
        default=0.02,
        help="Fraction held out as the dev split.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to load_dataset.",
    )
    return p.parse_args()


def load_manifest(path: Path):
    """Return (groups, texts).

    groups: {(dataset, config, split, audio_key, text_key): set(indices)}
    texts:  {(dataset, config, split, index): transcript}
    """
    groups: Dict[Tuple, set] = defaultdict(set)
    texts: Dict[Tuple, str] = {}
    n = 0
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (
                r["dataset"],
                r.get("config"),
                r["split"],
                r.get("audio_key", "audio"),
                r.get("text_key", "text"),
            )
            groups[key].add(r["index"])
            texts[(r["dataset"], r.get("config"), r["split"], r["index"])] = r["text"]
            n += 1
    logging.info(f"Manifest: {n} rows across {len(groups)} source group(s).")
    return groups, texts


def source_generator(key, indices, texts, sampling_rate, trust_remote_code):
    """Generator yielding {'audio': {...}, 'text': ...} for selected rows."""
    from datasets import Audio, load_dataset

    dataset, config, split, audio_key, text_key = key

    def _gen():
        kwargs = {"split": split, "streaming": True}
        if trust_remote_code:
            kwargs["trust_remote_code"] = True
        if config:
            ds = load_dataset(dataset, config, **kwargs)
        else:
            ds = load_dataset(dataset, **kwargs)
        # Keep raw bytes; do not decode/resample here.
        ds = ds.cast_column(audio_key, Audio(decode=False))

        wanted = indices
        remaining = len(wanted)
        mismatches = 0
        for index, row in enumerate(ds):
            if index not in wanted:
                continue
            audio = row[audio_key]  # {"bytes": ..., "path": ...} (decode=False)
            if audio is None or (audio.get("bytes") is None and not audio.get("path")):
                logging.warning(f"{dataset}[{index}]: no audio bytes/path; skipped.")
                remaining -= 1
                continue
            mtext = texts[(dataset, config, split, index)]
            stext = row.get(text_key)
            if stext is not None and stext != mtext:
                mismatches += 1
            yield {"audio": audio, "text": mtext}
            remaining -= 1
            if remaining <= 0:
                break
        if mismatches:
            logging.warning(
                f"{dataset}:{split}: {mismatches} transcript mismatch(es) vs "
                "manifest (manifest text used). Check index alignment."
            )

    return _gen


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )
    args = get_args()

    from datasets import (
        Audio,
        Dataset,
        DatasetDict,
        Features,
        Value,
        concatenate_datasets,
    )

    groups, texts = load_manifest(args.manifest)

    features = Features(
        {
            "audio": Audio(sampling_rate=args.sampling_rate),
            "text": Value("string"),
        }
    )

    per_source: List[Dataset] = []
    for key, indices in groups.items():
        logging.info(f"Materializing {key[0]}:{key[2]} ({len(indices)} rows)")
        gen = source_generator(
            key, indices, texts, args.sampling_rate, args.trust_remote_code
        )
        ds = Dataset.from_generator(gen, features=features)
        per_source.append(ds)

    combined = (
        per_source[0]
        if len(per_source) == 1
        else concatenate_datasets(per_source)
    )
    combined = combined.shuffle(seed=args.seed)

    if args.dev_frac > 0 and len(combined) > 1:
        split = combined.train_test_split(test_size=args.dev_frac, seed=args.seed)
        out = DatasetDict({"train": split["train"], "dev": split["test"]})
    else:
        out = DatasetDict({"train": combined})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out.save_to_disk(str(args.output_dir))
    sizes = {k: len(v) for k, v in out.items()}
    logging.info(f"Saved combined dataset -> {args.output_dir}  splits={sizes}")
    logging.info(
        "Next: compute_fbank_huggingface.py --load-from-disk "
        f"--dataset {args.output_dir} --split train (and --split dev)."
    )


if __name__ == "__main__":
    main()
