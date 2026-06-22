#!/usr/bin/env bash
#
# Fine-tune a pre-trained Zipformer on a custom dataset (e.g. prepared from a
# HuggingFace dataset via local/compute_fbank_huggingface.py), with optional
# MLflow monitoring on a Databricks workspace.
#
# Run from: icefall/egs/librispeech/ASR
#   ./zipformer/finetune.sh
#
# Override any variable on the command line, e.g.:
#   USE_MLFLOW=1 MAX_DURATION=300 ./zipformer/finetune.sh

set -eou pipefail

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# ----------------------------------------------------------------------------
# Hardware / data
# ----------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
WORLD_SIZE="${WORLD_SIZE:-4}"

MANIFEST_DIR="${MANIFEST_DIR:-data/fbank}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-mydata_cuts_train.jsonl.gz}"
VALID_MANIFEST="${VALID_MANIFEST:-mydata_cuts_dev.jsonl.gz}"
VALID_SET_NAME="${VALID_SET_NAME:-mydata}"
BPE_MODEL="${BPE_MODEL:-data/lang_bpe_500/bpe.model}"

# ----------------------------------------------------------------------------
# Fine-tuning hyperparameters
# ----------------------------------------------------------------------------
FINETUNE_CKPT="${FINETUNE_CKPT:-pretrained/exp/pretrained.pt}"
EXP_DIR="${EXP_DIR:-zipformer/exp_finetune}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
BASE_LR="${BASE_LR:-0.0045}"
MAX_DURATION="${MAX_DURATION:-500}"   # lower this if you hit CUDA OOM
USE_FP16="${USE_FP16:-1}"
USE_MUX="${USE_MUX:-0}"
ENABLE_MUSAN="${ENABLE_MUSAN:-1}"     # set 0 if data/fbank/musan_cuts.jsonl.gz is absent

# ----------------------------------------------------------------------------
# MLflow / Databricks monitoring (optional)
# ----------------------------------------------------------------------------
# Auth comes from the environment; never hardcode the token here.
# Set these in your shell or a non-committed env file before running:
#   export MLFLOW_TRACKING_URI=databricks
#   export DATABRICKS_HOST=https://<workspace>.cloud.databricks.com
#   export DATABRICKS_TOKEN=<personal-access-token>
USE_MLFLOW="${USE_MLFLOW:-0}"
MLFLOW_EXPERIMENT="${MLFLOW_EXPERIMENT:-/Users/taiinguyenn139@gmail.com/zipformer-finetune}"
MLFLOW_RUN_NAME="${MLFLOW_RUN_NAME:-$(basename "${EXP_DIR}")}"

if [ "${USE_MLFLOW}" = "1" ] && [ -z "${MLFLOW_TRACKING_URI:-}" ]; then
  echo "WARNING: USE_MLFLOW=1 but MLFLOW_TRACKING_URI is unset." >&2
  echo "         Export MLFLOW_TRACKING_URI=databricks (and DATABRICKS_HOST/DATABRICKS_TOKEN)." >&2
fi

# ----------------------------------------------------------------------------
# Launch
# ----------------------------------------------------------------------------
./zipformer/finetune.py \
  --world-size "${WORLD_SIZE}" \
  --num-epochs "${NUM_EPOCHS}" \
  --start-epoch 1 \
  --use-fp16 "${USE_FP16}" \
  --do-finetune 1 \
  --finetune-ckpt "${FINETUNE_CKPT}" \
  --base-lr "${BASE_LR}" \
  --use-mux "${USE_MUX}" \
  --bpe-model "${BPE_MODEL}" \
  --exp-dir "${EXP_DIR}" \
  --manifest-dir "${MANIFEST_DIR}" \
  --train-manifest "${TRAIN_MANIFEST}" \
  --valid-manifest "${VALID_MANIFEST}" \
  --valid-set-name "${VALID_SET_NAME}" \
  --enable-musan "${ENABLE_MUSAN}" \
  --max-duration "${MAX_DURATION}" \
  --use-mlflow "${USE_MLFLOW}" \
  --mlflow-experiment "${MLFLOW_EXPERIMENT}" \
  --mlflow-run-name "${MLFLOW_RUN_NAME}"

