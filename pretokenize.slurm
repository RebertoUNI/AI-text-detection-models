#!/bin/bash
#SBATCH --job-name=pretokenize
#SBATCH --output=logs/pretokenize_%j.out
#SBATCH --error=logs/pretokenize_%j.err
#SBATCH --partition=EPYC             # nodo CPU, 128 core: nessuna GPU necessaria per tokenizzare
#SBATCH --account=dssc
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail
mkdir -p logs

source "$HOME/envs/ai-text-detection/bin/activate"
cd "$SLURM_SUBMIT_DIR"

# Tokenizza per entrambi i tokenizer usati nel progetto (deberta -> FCNN/PaperCNN/daBERTa, qwen -> Qwen)
python pretokenize.py --tokenizer deberta --num-proc "$SLURM_CPUS_PER_TASK"
python pretokenize.py --tokenizer qwen    --num-proc "$SLURM_CPUS_PER_TASK"
