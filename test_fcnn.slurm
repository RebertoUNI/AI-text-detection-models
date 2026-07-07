#!/bin/bash
#SBATCH --job-name=test_fcnn
#SBATCH --output=logs/test_fcnn_%j.out
#SBATCH --error=logs/test_fcnn_%j.err
#SBATCH --partition=GPU              # unica partizione GPU su ORFEO (nodo gpu003)
#SBATCH --account=dssc      # TODO: sostituisci con il tuo account/progetto
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
mkdir -p logs

source "$HOME/envs/ai-text-detection/bin/activate"
cd "$SLURM_SUBMIT_DIR"

python test_fcnn.py --batch-size 128 --num-workers "$SLURM_CPUS_PER_TASK"
