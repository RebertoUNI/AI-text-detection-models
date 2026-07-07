#!/bin/bash
#SBATCH --job-name=train_qwen
#SBATCH --output=logs/train_qwen_%j.out
#SBATCH --error=logs/train_qwen_%j.err
#SBATCH --partition=GPU              # unica partizione GPU su ORFEO (nodo gpu003)
#SBATCH --account=dssc      # TODO: sostituisci con il tuo account/progetto
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail
mkdir -p logs

source "$HOME/envs/ai-text-detection/bin/activate"
cd "$SLURM_SUBMIT_DIR"

python train_qwen.py \
    --epochs 3 \
    --batch-size 4 \
    --eval-batch-size 4 \
    --grad-accum 4 \
    --num-proc 1 \
    --seed 42
