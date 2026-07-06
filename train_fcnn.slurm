#!/bin/bash
#SBATCH --job-name=train_fcnn
#SBATCH --output=logs/train_fcnn_%j.out
#SBATCH --error=logs/train_fcnn_%j.err
#SBATCH --partition=GPU              # unica partizione GPU su ORFEO (nodo gpu003)
#SBATCH --account=<TUO_ACCOUNT>      # TODO: sostituisci con il tuo account/progetto (`sacctmgr show associations`)
#SBATCH --gres=gpu:1                 # FCNN è piccola: basta 1 GPU qualsiasi (anche una V100)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail
mkdir -p logs

# Ambiente creato con setup_env.sh
source "$HOME/envs/ai-text-detection/bin/activate"

cd "$SLURM_SUBMIT_DIR"

python train_fcnn.py \
    --epochs 10 \
    --batch-size 128 \
    --num-workers "$SLURM_CPUS_PER_TASK" \
    --seed 42
