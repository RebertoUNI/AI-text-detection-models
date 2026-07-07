#!/bin/bash
#SBATCH --job-name=train_papercnn
#SBATCH --output=logs/train_papercnn_%j.out
#SBATCH --error=logs/train_papercnn_%j.err
#SBATCH --partition=GPU              # unica partizione GPU su ORFEO (nodo gpu003)
#SBATCH --account=dssc      # TODO: sostituisci con il tuo account/progetto
#SBATCH --gres=gpu:1                 # PaperCNN è piccola: basta 1 GPU qualsiasi (anche una V100)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail
mkdir -p logs

source "$HOME/envs/ai-text-detection/bin/activate"
cd "$SLURM_SUBMIT_DIR"

python train_papercnn.py \
    --epochs 10 \
    --batch-size 128 \
    --num-workers "$SLURM_CPUS_PER_TASK" \
    --seed 42
