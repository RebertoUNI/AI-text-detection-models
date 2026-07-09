#!/bin/bash
#SBATCH --job-name=setup_env
#SBATCH --output=logs/setup_env_%j.out
#SBATCH --error=logs/setup_env_%j.err
#SBATCH --partition=EPYC             # nodo di calcolo CPU: il login node ha RAM troppo poca per pip install
#SBATCH --account=dssc
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                    # il nodo di login va in OOM con la RAM di default: qui ne diamo di più
#SBATCH --time=01:00:00

set -euo pipefail
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

bash setup_env.sh
