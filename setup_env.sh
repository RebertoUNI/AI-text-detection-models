#!/usr/bin/env bash
# =============================================================================
# setup_env.sh
# =============================================================================
# Crea il virtualenv Python usato da tutti i job di questo progetto su ORFEO.
#
# IMPORTANTE — leggi prima di lanciare:
#
#   1. Esegui questo script su un NODO COMPUTAZIONALE (non sul login node):
#        srun --partition=GPU --gres=gpu:1 --time=01:00:00 --pty bash
#      oppure, se vuoi solo creare l'env senza GPU:
#        srun --partition=THIN --time=01:00:00 --pty bash
#
#   2. Il virtualenv viene creato sotto $SCRATCH (consigliato dalla doc ORFEO
#      perche' gli env possono occupare molto spazio e saturare la home quota).
#      Se $SCRATCH non e' definito sul tuo account, sostituisci la variabile
#      ENV_DIR con il path che preferisci.
#
#   3. La versione Python del virtualenv e' identica a quella del sistema al
#      momento della creazione: nessun module load e' necessario.
#
# Uso:
#   bash setup_env.sh
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Variabili di configurazione
# ---------------------------------------------------------------------------

# Path dell'ambiente (modifica se vuoi un percorso diverso)
# ORFEO doc raccomanda lo scratch per evitare di saturare la home:
#   https://orfeo-doc.areasciencepark.it/HPC/python-environment/
if [[ -n "${SCRATCH:-}" ]]; then
    ENV_DIR="${SCRATCH}/envs/ai-text-detection"
else
    # fallback: home (attenzione alla quota)
    ENV_DIR="${HOME}/envs/ai-text-detection"
    echo "[WARN] La variabile SCRATCH non e' definita: l'env verra' creato in $ENV_DIR"
    echo "       Controlla la quota con: quota -s"
fi

# ---------------------------------------------------------------------------
# 1. Verifica che virtualenv sia disponibile
# ---------------------------------------------------------------------------
if ! python3 -m virtualenv --version &>/dev/null; then
    echo "[ERROR] 'virtualenv' non trovato."
    echo "        Installalo prima con: pip install --user virtualenv"
    echo "        oppure: python3 -m pip install --user virtualenv"
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Crea il virtualenv (o riusalo se esiste gia')
# ---------------------------------------------------------------------------
if [[ -d "$ENV_DIR" ]]; then
    echo "[INFO] Virtualenv gia' esistente in: $ENV_DIR"
    echo "       Per ricrearlo da zero: rm -rf $ENV_DIR && bash setup_env.sh"
else
    echo "[INFO] Creo il virtualenv in: $ENV_DIR"
    python3 -m virtualenv "$ENV_DIR"
fi

# ---------------------------------------------------------------------------
# 3. Attiva e installa le dipendenze
# ---------------------------------------------------------------------------
# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"

echo "[INFO] Python usato: $(python --version) — $(which python)"

pip install --upgrade pip
pip install -r "$(dirname "$0")/requirements.txt"

deactivate

# ---------------------------------------------------------------------------
# 4. Riepilogo
# ---------------------------------------------------------------------------
echo ""
echo "====================================================================="
echo " Ambiente creato correttamente."
echo "====================================================================="
echo ""
echo " Path:      $ENV_DIR"
echo " Attivalo con:"
echo "   source ${ENV_DIR}/bin/activate"
echo ""
echo " Negli script .slurm aggiungi PRIMA del comando python:"
echo "   source ${ENV_DIR}/bin/activate"
echo ""
echo " Per verificare i pacchetti installati:"
echo "   source ${ENV_DIR}/bin/activate && pip list"
echo "====================================================================="
