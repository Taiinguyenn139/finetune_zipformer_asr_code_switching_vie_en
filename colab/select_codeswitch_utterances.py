#!/usr/bin/env python3
"""Scan HuggingFace dataset transcripts and select VI+EN code-switch utterances.

Streams each source dataset reading ONLY the transcript column (audio is never
decoded), applies codeswitch_filter, and writes a combined manifest plus
review artifacts. Designed to run on Google Colab; point --output-dir at a
Google Drive path so the manifest survives runtime restarts.

The manifest (`combined_manifest.jsonl`) is the reproducible selection record;
`materialize_dataset.py` turns it into a dataset for fbank computation.

Sources are described by a JSON file (--sources-json), a list of objects::

    [
      {"dataset": "org/ds1", "config": "vi", "split": "train",
       "text_key": "sentence", "audio_key": "audio"},
      {"dataset": "org/ds2", "split": "train"}
    ]

or a single source via --dataset/--config/--split/--text-key.

Example
-------
./select_codeswitch_utterances.py \
  --sources-json sources.json \
  --output-dir /content/drive/MyDrive/vibank_cs \
  --min-en-tokens 1 --limit 5000
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    # --- sources ---
    p.add_argument(
        "--sources-json",
        type=Path,
        default=None,
        help="JSON file: list of {dataset, config?, split, text_key?, "
        "audio_key?}. Mutually exclusive with the single-source flags.",
    )
    p.add_argument("--dataset", type=str, default=None, help="Single source repo id.")
    p.add_argument("--config", type=str, default=None, help="Single source config/name.")
    p.add_argument("--split", type=str, default="train", help="Single source split.")
    p.add_argument(
        "--text-key", type=str, default="text", help="Single source transcript column."
    )
    p.add_argument(
        "--audio-key",
        type=str,
        default="audio",
        help="Single source audio column (recorded in the manifest, not read).",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to load_dataset.",
    )

    # --- output ---
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for combined_manifest.jsonl + review artifacts "
        "(use a Google Drive path on Colab).",
    )

    # --- filter knobs ---
    p.add_argument("--min-en-tokens", type=int, default=1)
    p.add_argument("--min-len", type=int, default=2)
    p.add_argument(
        "--max-en-ratio",
        type=float,
        default=None,
        help="Drop utterances whose English-token ratio exceeds this "
        "(e.g. 0.6 keeps true code-switch, drops fully-English). Unset = off.",
    )
    p.add_argument(
        "--english-words-file",
        type=Path,
        default=None,
        help="Optional newline-separated English wordlist (overrides wordfreq).",
    )
    p.add_argument(
        "--vi-syllables-file",
        type=Path,
        default=None,
        help="Optional extra ASCII Vietnamese syllables to exclude.",
    )
    p.add_argument(
        "--domain-allowlist-file",
        type=Path,
        default=None,
        help="Optional extra English domain terms to always count as English.",
    )

    # --- run control ---
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max rows to scan per source (for dry runs).",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=40,
        help="How many selected/rejected examples to dump for review.",
    )
    return p.parse_args()


def load_sources(args: argparse.Namespace) -> List[Dict]:
    if args.sources_json is not None:
        sources = json.loads(Path(args.sources_json).read_text(encoding="utf-8"))
        if not isinstance(sources, list):
            raise ValueError("--sources-json must contain a JSON list.")
        return sources
    if args.dataset is None:
        raise ValueError("Provide either --sources-json or --dataset.")
    return [
        {
            "dataset": args.dataset,
            "config": args.config,
            "split": args.split,
            "text_key": args.text_key,
            "audio_key": args.audio_key,
        }
    ]


def build_filter(args: argparse.Namespace):
    from codeswitch_filter import CodeSwitchFilter

    def _read(path: Optional[Path]) -> Optional[List[str]]:
        if path is None:
            return None
        return [
            ln.strip()
            for ln in Path(path).read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")
        ]

    return CodeSwitchFilter(
        english_words=_read(args.english_words_file),
        vi_syllables=_read(args.vi_syllables_file),
        domain_allowlist=_read(args.domain_allowlist_file),
        min_en_tokens=args.min_en_tokens,
        min_len=args.min_len,
        max_en_ratio=args.max_en_ratio,
    )


def stream_text_only(source: Dict, trust_remote_code: bool):
    """Yield (index, text) streaming a source, projecting the text column."""
    from datasets import load_dataset

    text_key = source.get("text_key", "text")
    kwargs = {"split": source["split"], "streaming": True}
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    if source.get("config"):
        ds = load_dataset(source["dataset"], source["config"], **kwargs)
    else:
        ds = load_dataset(source["dataset"], **kwargs)

    if text_key not in ds.column_names:
        raise ValueError(
            f"Text column '{text_key}' not in {source['dataset']}. "
            f"Available: {ds.column_names}. Set 'text_key' in the source."
        )
    # Project to the text column to avoid decoding audio. select_columns is the
    # efficient path (newer datasets); fall back to remove_columns.
    try:
        ds = ds.select_columns([text_key])
    except (AttributeError, ValueError):
        ds = ds.remove_columns([c for c in ds.column_names if c != text_key])

    for index, row in enumerate(ds):
        yield index, row[text_key]


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )
    args = get_args()
    sources = load_sources(args)
    csf = build_filter(args)

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "combined_manifest.jsonl"

    stats: List[Dict] = []
    sel_samples: List[str] = []
    rej_samples: List[str] = []
    total_selected = 0

    with manifest_path.open("w", encoding="utf-8") as mf:
        for source in sources:
            text_key = source.get("text_key", "text")
            tag = f"{source['dataset']}:{source.get('config') or '-'}:{source['split']}"
            logging.info(f"Scanning {tag}")
            scanned = kept = 0
            for index, text in stream_text_only(source, args.trust_remote_code):
                scanned += 1
                if text and csf.is_codeswitch(text):
                    kept += 1
                    total_selected += 1
                    en = csf.english_tokens(text)
                    mf.write(
                        json.dumps(
                            {
                                "dataset": source["dataset"],
                                "config": source.get("config"),
                                "split": source["split"],
                                "index": index,
                                "text_key": text_key,
                                "audio_key": source.get("audio_key", "audio"),
                                "text": text,
                                "english_tokens": en,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    if len(sel_samples) < args.num_samples:
                        sel_samples.append(f"[{en}] {text}")
                elif text and len(rej_samples) < args.num_samples:
                    rej_samples.append(text)
                if args.limit is not None and scanned >= args.limit:
                    break
            logging.info(f"  {tag}: kept {kept}/{scanned}")
            stats.append(
                {"source": tag, "scanned": scanned, "kept": kept}
            )

    (out_dir / "selection_stats.json").write_text(
        json.dumps(
            {"total_selected": total_selected, "per_source": stats},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "samples_selected.txt").write_text(
        "\n".join(sel_samples), encoding="utf-8"
    )
    (out_dir / "samples_rejected.txt").write_text(
        "\n".join(rej_samples), encoding="utf-8"
    )

    logging.info(f"Selected {total_selected} utterances -> {manifest_path}")
    logging.info(
        "Review samples_selected.txt / samples_rejected.txt and tune "
        "--min-en-tokens / wordlists before scaling up."
    )


if __name__ == "__main__":
    main()
