#!/usr/bin/env bash
#SBATCH --job-name=pretokenize
#SBATCH --account=dssc
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/orfeo/cephfs/home/dssc/rtittoto/AI-text-detection-models/logs/pretokenize_%j.out
#SBATCH --error=/orfeo/cephfs/home/dssc/rtittoto/AI-text-detection-models/logs/pretokenize_%j.err

set -euo pipefail

REPO_DIR="/orfeo/cephfs/home/dssc/rtittoto/AI-text-detection-models"
ENV_DIR="/u/dssc/rtittoto/scratch/envs/ai-text-detection"

mkdir -p "${REPO_DIR}/logs"

cd "${REPO_DIR}"
source "${ENV_DIR}/bin/activate"

echo "[INFO] Hostname: $(hostname)"
echo "[INFO] Workdir : $(pwd)"
echo "[INFO] Python  : $(which python)"
echo "[INFO] Version : $(python --version)"
echo "[INFO] Date    : $(date)"

python - <<'PY'
from data_utils import load_and_tokenize_dataset_generic

print("[INFO] Pretokenization start")

load_and_tokenize_dataset_generic(
    tokenizer_name="microsoft/deberta-v3-large",
    cache_dir="./tokenized_dataset",
    max_length=256,
    num_proc=8,
)
print("[INFO] DeBERTa cache completed: ./tokenized_dataset")

load_and_tokenize_dataset_generic(
    tokenizer_name="Qwen/Qwen3-0.6B-Base",
    cache_dir="./tokenized_dataset_qwen",
    max_length=256,
    num_proc=8,
)
print("[INFO] Qwen cache completed: ./tokenized_dataset_qwen")
PY

echo "[INFO] Pretokenization finished successfully at $(date)"
