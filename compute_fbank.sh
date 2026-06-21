#!/usr/bin/env bash
#
# Precompute 80-dim Fbank cuts for Zipformer fine-tuning from a HuggingFace
# dataset, by wrapping compute_fbank_huggingface.py. Produces the two cut files
# that finetune.sh expects in data/fbank:
#
#   <PREFIX>_cuts_<TRAIN_OUTPUT_NAME>.jsonl.gz   (speed-perturbed by default)
#   <PREFIX>_cuts_<DEV_OUTPUT_NAME>.jsonl.gz
#
# Run from the repo root:
#   ./compute_fbank.sh
#
# Override any variable on the command line, e.g.:
#   DATASET=mozilla-foundation/common_voice_17_0 NAME=en TEXT_KEY=sentence \
#     ./compute_fbank.sh
#
# For datasets that store an audio file path (not decoded audio) in a column,
# with the audio saved in a separate directory:
#   AUDIO_KEY=audio_path AUDIO_DIR=/data/clips ./compute_fbank.sh

set -eou pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ----------------------------------------------------------------------------
# Dataset selection
# ----------------------------------------------------------------------------
DATASET="${DATASET:-my-org/my-asr-data}"   # HuggingFace repo id or local path
NAME="${NAME:-}"                           # optional config/subset (e.g. 'en')
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
DEV_SPLIT="${DEV_SPLIT:-validation}"
CACHE_DIR="${CACHE_DIR:-}"                  # optional datasets cache dir
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"

# ----------------------------------------------------------------------------
# Column keys (override for datasets using different column names)
# ----------------------------------------------------------------------------
TEXT_KEY="${TEXT_KEY:-text}"
AUDIO_KEY="${AUDIO_KEY:-}"                  # empty -> script default ('audio')
AUDIO_DIR="${AUDIO_DIR:-}"                  # base dir for path-only audio columns

# ----------------------------------------------------------------------------
# Output (must match finetune.sh's MANIFEST_DIR / *_MANIFEST)
# ----------------------------------------------------------------------------
PREFIX="${PREFIX:-mydata}"
OUTPUT_DIR="${OUTPUT_DIR:-data/fbank}"
TRAIN_OUTPUT_NAME="${TRAIN_OUTPUT_NAME:-train}"
DEV_OUTPUT_NAME="${DEV_OUTPUT_NAME:-dev}"

# ----------------------------------------------------------------------------
# Feature / audio options
# ----------------------------------------------------------------------------
RESAMPLE_RATE="${RESAMPLE_RATE:-16000}"    # Zipformer expects 16 kHz
NUM_MEL_BINS="${NUM_MEL_BINS:-80}"
TEXT_NORMALIZATION="${TEXT_NORMALIZATION:-upper-no-punct}"  # reuse LibriSpeech bpe
TRAIN_PERTURB_SPEED="${TRAIN_PERTURB_SPEED:-1}"  # 3x speed perturb on train only
BPE_MODEL="${BPE_MODEL:-}"                  # optional: filter cuts with T < S
NUM_JOBS="${NUM_JOBS:-}"                    # empty -> script default
PYTHON="${PYTHON:-python3}"

PY_SCRIPT="${SCRIPT_DIR}/compute_fbank_huggingface.py"

# ----------------------------------------------------------------------------
# Run compute_fbank_huggingface.py for one split.
#   $1 = split name (HuggingFace)   $2 = output-name   $3 = perturb-speed (0/1)
# ----------------------------------------------------------------------------
compute_split() {
  local split="$1"
  local output_name="$2"
  local perturb="$3"

  local args=(
    --dataset "${DATASET}"
    --split "${split}"
    --output-name "${output_name}"
    --prefix "${PREFIX}"
    --output-dir "${OUTPUT_DIR}"
    --text-key "${TEXT_KEY}"
    --text-normalization "${TEXT_NORMALIZATION}"
    --num-mel-bins "${NUM_MEL_BINS}"
    --resample-rate "${RESAMPLE_RATE}"
  )

  [ -n "${NAME}" ] && args+=(--name "${NAME}")
  [ -n "${CACHE_DIR}" ] && args+=(--cache-dir "${CACHE_DIR}")
  [ "${TRUST_REMOTE_CODE}" = "1" ] && args+=(--trust-remote-code)
  [ -n "${AUDIO_KEY}" ] && args+=(--audio-key "${AUDIO_KEY}")
  [ -n "${AUDIO_DIR}" ] && args+=(--audio-dir "${AUDIO_DIR}")
  [ -n "${BPE_MODEL}" ] && args+=(--bpe-model "${BPE_MODEL}")
  [ -n "${NUM_JOBS}" ] && args+=(--num-jobs "${NUM_JOBS}")
  [ "${perturb}" = "1" ] && args+=(--perturb-speed)

  echo "==> Computing Fbank for split='${split}' -> ${PREFIX}_cuts_${output_name}.jsonl.gz"
  "${PYTHON}" "${PY_SCRIPT}" "${args[@]}"
}

# ----------------------------------------------------------------------------
# Train (speed-perturbed) then dev (no perturbation).
# ----------------------------------------------------------------------------
compute_split "${TRAIN_SPLIT}" "${TRAIN_OUTPUT_NAME}" "${TRAIN_PERTURB_SPEED}"
compute_split "${DEV_SPLIT}" "${DEV_OUTPUT_NAME}" "0"

echo "Done. Cuts written under ${OUTPUT_DIR}/"
