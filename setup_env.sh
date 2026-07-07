#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# setup_env.sh
# ─────────────────────────────────────────────────────────────────────────
# Da eseguire UNA SOLA VOLTA su ORFEO (su un nodo di login, non serve GPU
# per questo passaggio) per creare l'ambiente Python usato da tutti i job.
#
#   bash setup_env.sh
#
# Moduli confermati su ORFEO (via `module avail`): nessun modulo 'python'
# (si usa il Python di sistema), cuda disponibile in versione 11.8/12.0/
# 12.1/12.6/12.8 (default 12.8, usata qui sotto).
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
ENV_DIR="$HOME/envs/ai-text-detection"
python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

# 3. Dipendenze
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Ambiente creato in: $ENV_DIR"
echo "Negli script SLURM, attivalo con:"
echo "  source $ENV_DIR/bin/activate"
