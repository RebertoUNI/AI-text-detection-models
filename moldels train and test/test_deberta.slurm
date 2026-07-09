#!/bin/bash
#SBATCH --job-name=test_deberta
#SBATCH --output=logs/test_deberta_%j.out
#SBATCH --error=logs/test_deberta_%j.err
#SBATCH --partition=GPU              # unica partizione GPU su ORFEO (nodo gpu003)
#SBATCH --account=dssc      # TODO: sostituisci con il tuo account/progetto
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
mkdir -p logs

source "$HOME/envs/ai-text-detection/bin/activate"
cd "$SLURM_SUBMIT_DIR"

python test_deberta.py --batch-size 32 --num-proc "$SLURM_CPUS_PER_TASK"
