#!/bin/bash
#SBATCH --job-name=test_qwen
#SBATCH --output=logs/test_qwen_%j.out
#SBATCH --error=logs/test_qwen_%j.err
#SBATCH --partition=GPU              # unica partizione GPU su ORFEO (nodo gpu003)
#SBATCH --account=dssc      # TODO: sostituisci con il tuo account/progetto
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail
mkdir -p logs

source "$HOME/envs/ai-text-detection/bin/activate"
cd "$SLURM_SUBMIT_DIR"

python test_qwen.py --batch-size 16 --num-proc 1
