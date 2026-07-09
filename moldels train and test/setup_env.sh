#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# setup_env.sh
# ─────────────────────────────────────────────────────────────────────────
# NON lanciarlo direttamente sul nodo di login (`bash setup_env.sh`): il
# login node ha limiti di RAM troppo stretti e `pip install -r
# requirements.txt` (torch/transformers/bitsandbytes) viene ucciso
# dall'OOM killer ("Killed"). Lancialo invece come job SLURM su un nodo di
# calcolo:
#
#   sbatch setup_env.slurm
#
# ("Remember to do the following procedure on a computational node!" —
# dalla documentazione ORFEO sui virtualenv Python)
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail

# 1. Moduli del cluster
module purge
# Nessun modulo 'python' su ORFEO (confermato con `module avail python`): si usa
# il Python di sistema già nel PATH. Verifica la versione (serve >= 3.9):
python3 --version

module load cuda/12.8          # versione di default su ORFEO (confermato con `module avail cuda`)
                                # In pratica quasi mai indispensabile: i wheel pip di
                                # torch/bitsandbytes includono già il proprio runtime CUDA.
                                # Lo carichiamo comunque per sicurezza (nvcc/driver a disposizione).

# 2. Virtualenv dedicato al progetto (nella tua home o area di progetto)
ENV_DIR="$HOME/scratch/envs/ai-text-detection"
python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

# 3. Dipendenze — installate in due passi e senza cache, per ridurre il
#    picco di memoria usato dal resolver di pip (causa comune di OOM se
#    lanciato per errore su un nodo di login con poca RAM disponibile)
pip install --upgrade pip
pip install --no-cache-dir "torch>=2.1"
pip install --no-cache-dir -r requirements.txt

echo ""
echo "Ambiente creato in: $ENV_DIR"
echo "Negli script SLURM, attivalo con:"
echo "  source $ENV_DIR/bin/activate"
