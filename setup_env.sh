#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# setup_env.sh
# ─────────────────────────────────────────────────────────────────────────
# Da eseguire UNA SOLA VOLTA su ORFEO (su un nodo di login, non serve GPU
# per questo passaggio) per creare l'ambiente Python usato da tutti i job.
#
#   bash setup_env.sh
#
# NOTA: i nomi esatti dei moduli (python/..., cuda/...) vanno verificati con
#       `module avail` sul cluster: quelli sotto sono placeholder plausibili,
#       da adattare. Consulta https://orfeo-doc.areasciencepark.it/HPC/
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail

# 1. Moduli del cluster (ADATTA I NOMI a quelli reali di ORFEO)
module purge
module load python/3.11        # <-- verifica il nome esatto con `module avail python`
module load cuda/12.1          # <-- verifica il nome esatto con `module avail cuda`
                                #     (bitsandbytes/torch vogliono una CUDA compatibile)

# 2. Virtualenv dedicato al progetto (nella tua home o area di progetto)
ENV_DIR="$HOME/envs/ai-text-detection"
python -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

# 3. Dipendenze
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Ambiente creato in: $ENV_DIR"
echo "Negli script SLURM, attivalo con:"
echo "  source $ENV_DIR/bin/activate"
