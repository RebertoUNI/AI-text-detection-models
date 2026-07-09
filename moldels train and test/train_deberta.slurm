#!/bin/bash
#SBATCH --job-name=train_deberta
#SBATCH --output=logs/train_deberta_%j.out
#SBATCH --error=logs/train_deberta_%j.err
#SBATCH --partition=GPU              # unica partizione GPU su ORFEO (nodo gpu003)
#SBATCH --account=dssc      # TODO: sostituisci con il tuo account/progetto
#SBATCH --gres=gpu:1                 # se il cluster supporta la sintassi per tipo, es. gpu:a100:1, usala
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail
mkdir -p logs

source "$HOME/envs/ai-text-detection/bin/activate"
cd "$SLURM_SUBMIT_DIR"

python train_deberta.py \
    --epochs 3 \
    --batch-size 16 \
    --eval-batch-size 16 \
    --num-proc "$SLURM_CPUS_PER_TASK" \
    --seed 42
