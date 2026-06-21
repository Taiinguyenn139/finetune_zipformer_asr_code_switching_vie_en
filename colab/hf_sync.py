#!/usr/bin/env python3
"""Persist Colab fine-tuning artifacts to the Hugging Face Hub.

Colab's local disk is wiped on every runtime restart. This helper treats two
private Hub repos as durable file stores so the workflow survives restarts:

  - a *dataset* repo  : combined_dataset/ (save_to_disk) + fbank cuts
  - a *model* repo    : bpe.model, the base pretrained ckpt, and epoch-N.pt

It is a thin wrapper over huggingface_hub (upload_folder / upload_file /
snapshot_download / hf_hub_download), so any directory or file shape works the
same way. Pushes are on-demand: run the relevant subcommand between/after
training cells. No coupling to finetune.py.

Subcommands
-----------
push             Upload a local file or directory to a repo (optionally under
                 --path-in-repo).
pull             Download a repo (or one --path-in-repo subtree) to a local dir.
pull-latest-ckpt Find the highest epoch-N.pt in a model repo, download just that
                 file into --local, and print the epoch N. The notebook sets
                 --start-epoch = N + 1 to resume.

Auth: log in once per session (`huggingface-cli login`, `notebook_login()`, or
export HF_TOKEN). Subcommands also accept --token.

Examples
--------
# Push the materialized dataset + fbank cuts (private dataset repo)
./hf_sync.py push --repo-id you/vibank-cs-data --repo-type dataset \
  --local data/combined_dataset --path-in-repo combined_dataset --private
./hf_sync.py push --repo-id you/vibank-cs-data --repo-type dataset \
  --local data/fbank --path-in-repo fbank --private

# Push training inputs once (private model repo)
./hf_sync.py push --repo-id you/vibank-cs-model --repo-type model \
  --local model/bpe.model --private
./hf_sync.py push --repo-id you/vibank-cs-model --repo-type model \
  --local model/epoch-35-avg-6.pt --private

# On-demand: push checkpoints written so far
./hf_sync.py push --repo-id you/vibank-cs-model --repo-type model \
  --local data/exp_finetune --path-in-repo exp_finetune --allow-pattern 'epoch-*.pt'

# After a restart: resume from the latest checkpoint on the Hub
./hf_sync.py pull-latest-ckpt --repo-id you/vibank-cs-model \
  --path-in-repo exp_finetune --local data/exp_finetune
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional

# epoch-12.pt -> 12 ; ignores epoch-30-avg-9.pt averaged exports (not resume points)
_EPOCH_RE = re.compile(r"(?:^|/)epoch-(\d+)\.pt$")


def latest_epoch(filenames: Iterable[str]) -> Optional[int]:
    """Return the highest N among ``epoch-N.pt`` names, or None if there are none.

    Pure (no network) so it is unit-testable. Names may include directory
    prefixes (e.g. ``exp_finetune/epoch-7.pt``). Averaged checkpoints such as
    ``epoch-35-avg-6.pt`` are intentionally not matched: they are not valid
    ``--start-epoch`` resume points.
    """
    epochs: List[int] = []
    for name in filenames:
        m = _EPOCH_RE.search(name)
        if m:
            epochs.append(int(m.group(1)))
    return max(epochs) if epochs else None


def _api(token: Optional[str]):
    from huggingface_hub import HfApi

    return HfApi(token=token)


def cmd_push(args: argparse.Namespace) -> int:
    api = _api(args.token)
    local = Path(args.local)
    if not local.exists():
        logging.error("local path does not exist: %s", local)
        return 1

    api.create_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=True,
    )

    if local.is_dir():
        url = api.upload_folder(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            folder_path=str(local),
            path_in_repo=args.path_in_repo or ".",
            allow_patterns=args.allow_pattern or None,
            commit_message=args.message or f"push {local.name}",
        )
    else:
        # Place the file under path-in-repo if given, else at the repo root.
        dest = f"{args.path_in_repo}/{local.name}" if args.path_in_repo else local.name
        url = api.upload_file(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            path_or_fileobj=str(local),
            path_in_repo=dest,
            commit_message=args.message or f"push {local.name}",
        )
    logging.info("pushed %s -> %s", local, url)
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    from huggingface_hub import snapshot_download

    local = Path(args.local)
    local.mkdir(parents=True, exist_ok=True)
    # A subtree pull keeps only files under path-in-repo; default pulls all.
    allow = list(args.allow_pattern or [])
    if args.path_in_repo:
        allow.append(f"{args.path_in_repo}/*")
    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        local_dir=str(local),
        allow_patterns=allow or None,
        token=args.token,
    )
    logging.info("pulled %s -> %s", args.repo_id, path)
    return 0


def cmd_pull_latest_ckpt(args: argparse.Namespace) -> int:
    from huggingface_hub import hf_hub_download

    api = _api(args.token)
    files = api.list_repo_files(repo_id=args.repo_id, repo_type="model")
    prefix = f"{args.path_in_repo}/" if args.path_in_repo else ""
    scoped = [f for f in files if f.startswith(prefix)]
    n = latest_epoch(scoped)
    if n is None:
        logging.warning("no epoch-N.pt found in %s (%s)", args.repo_id, prefix or "root")
        # Print nothing parseable; caller treats empty stdout as "start fresh".
        return 0

    remote = f"{prefix}epoch-{n}.pt"
    local = Path(args.local)
    local.mkdir(parents=True, exist_ok=True)
    dest = hf_hub_download(
        repo_id=args.repo_id,
        repo_type="model",
        filename=remote,
        local_dir=str(local),
        token=args.token,
    )
    logging.info("downloaded %s -> %s", remote, dest)
    # Sole stdout line is the epoch number, so the notebook can capture it.
    print(n)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--repo-id", required=True, help="e.g. you/vibank-cs-data")
        sp.add_argument("--local", required=True, help="local file or directory")
        sp.add_argument("--path-in-repo", default="", help="subfolder inside the repo")
        sp.add_argument("--token", default=None, help="HF token (else cached login / HF_TOKEN)")
        sp.add_argument(
            "--allow-pattern",
            action="append",
            metavar="GLOB",
            help="restrict to matching files; repeatable",
        )

    sp_push = sub.add_parser("push", help="upload a file or directory")
    add_common(sp_push)
    sp_push.add_argument(
        "--repo-type", choices=["dataset", "model"], required=True
    )
    sp_push.add_argument("--private", action="store_true", help="create repo as private")
    sp_push.add_argument("--message", default=None, help="commit message")
    sp_push.set_defaults(func=cmd_push)

    sp_pull = sub.add_parser("pull", help="download a repo or subtree")
    add_common(sp_pull)
    sp_pull.add_argument(
        "--repo-type", choices=["dataset", "model"], required=True
    )
    sp_pull.set_defaults(func=cmd_pull)

    sp_latest = sub.add_parser(
        "pull-latest-ckpt",
        help="download highest epoch-N.pt from a model repo; print N",
    )
    sp_latest.add_argument("--repo-id", required=True)
    sp_latest.add_argument("--local", required=True, help="EXP_DIR to download into")
    sp_latest.add_argument("--path-in-repo", default="", help="subfolder holding checkpoints")
    sp_latest.add_argument("--token", default=None)
    sp_latest.set_defaults(func=cmd_pull_latest_ckpt)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
