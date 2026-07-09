#!/bin/bash
#SBATCH --job-name=umap-supcon
#SBATCH --account=dssc               # TODO: sostituisci con il tuo account/progetto se diverso
#SBATCH --partition=GPU              # unica partizione GPU su ORFEO (nodo gpu003)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2                 # verifica con `scontrol show node gpu003` che 2 GPU siano
                                      # davvero disponibili; se non lo sono, riduci a gpu:1 -
                                      # lo script si adatta automaticamente al numero di GPU viste
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/umap_supcon_%j.out
#SBATCH --error=logs/umap_supcon_%j.err

set -euo pipefail

REPO_DIR="$HOME/AI-text-detection-models/pseudofinal_analysis"
ENV_DIR="$HOME/scratch/envs/ai-text-detection"
SCRATCH_DIR="$HOME/scratch/ai-text-detection"

mkdir -p "${REPO_DIR}/logs"
cd "${REPO_DIR}"

# venv created beforehand on a compute node with setup_env.sh (see repo README):
#   sbatch setup_env.slurm
source "${ENV_DIR}/bin/activate"

echo "[INFO] Hostname: $(hostname)"
echo "[INFO] Workdir : $(pwd)"
echo "[INFO] Python  : $(which python)"
echo "[INFO] GPUs    : $(nvidia-smi -L 2>/dev/null || echo none)"
echo "[INFO] Date    : $(date)"

# keep the multi-GB HF cache off $HOME (quota!) - must be the SAME path used
# when running --download-only on the login node
export HF_HOME="${SCRATCH_DIR}/hf_cache"

# uncomment if compute nodes have no internet access
# (requires having run --download-only on the login node first):
# export HF_HUB_OFFLINE=1
# export HF_DATASETS_OFFLINE=1

python -u pseudofinal_analysis_orfeo.py \
    --data-dir   "${SCRATCH_DIR}/umap_data" \
    --output-dir "${SCRATCH_DIR}/umap_outputs"

echo "[INFO] Finished successfully at $(date)"
