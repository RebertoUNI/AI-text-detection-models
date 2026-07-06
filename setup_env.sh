#!/usr/bin/env bash
# =============================================================================
# setup_env.sh
# =============================================================================
# Crea il virtualenv Python usato da tutti i job di questo progetto su ORFEO.
#
# IMPORTANTE — leggi prima di lanciare:
#
#   1. Esegui questo script su un NODO COMPUTAZIONALE (non sul login node).
#      Apri una sessione interattiva con uno di questi comandi:
#
#        # con GPU (consigliato, garantisce compatibilita' binaria con i job):
#        srun --partition=GPU --gres=gpu:1 --time=01:00:00 --pty bash
#
#        # senza GPU (per semplici install CPU-only):
#        srun --partition=THIN --time=01:00:00 --pty bash
#
#   2. Lo scratch su ORFEO non e' una variabile d'ambiente ma un path nel
#      filesystem: /u/<group>/<user>/scratch
#      Lo script lo rileva automaticamente; se non lo trova usa $HOME.
#
#   3. La versione Python del virtualenv e' identica a quella del sistema:
#      non e' necessario caricare moduli aggiuntivi.
#
# Uso:
#   bash setup_env.sh
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Rileva il path dello scratch di ORFEO
#    Su ORFEO la struttura e': /u/<group>/<user>/scratch
#    Viene derivata dalla HOME che ha forma:  /u/<group>/<user>
# ---------------------------------------------------------------------------
detect_scratch() {
    local home_candidate="$HOME"
    # HOME su ORFEO: /u/<group>/<user>  (tre livelli sotto /)
    local scratch_candidate
    scratch_candidate="${home_candidate}/scratch"

    if [[ -d "$scratch_candidate" ]]; then
        echo "$scratch_candidate"
    else
        echo ""   # non trovato
    fi
}

SCRATCH_DIR="$(detect_scratch)"

if [[ -n "$SCRATCH_DIR" ]]; then
    ENV_DIR="${SCRATCH_DIR}/envs/ai-text-detection"
    echo "[INFO] Scratch rilevato: $SCRATCH_DIR"
else
    ENV_DIR="${HOME}/envs/ai-text-detection"
    echo "[WARN] Cartella scratch non trovata (attesa in ${HOME}/scratch)."
    echo "       L'env verra' creato in: $ENV_DIR"
    echo "       Controlla la quota home con: quota -s"
fi

# ---------------------------------------------------------------------------
# 1. Verifica che virtualenv sia disponibile
# ---------------------------------------------------------------------------
if ! python3 -m virtualenv --version &>/dev/null; then
    echo "[ERROR] 'virtualenv' non trovato."
    echo "        Installalo con: python3 -m pip install --user virtualenv"
    echo "        poi riapri la shell e rilancia questo script."
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Crea il virtualenv (o riusalo se esiste gia')
# ---------------------------------------------------------------------------
if [[ -d "$ENV_DIR" ]]; then
    echo "[INFO] Virtualenv gia' esistente in: $ENV_DIR"
    echo "       Per ricrearlo da zero:"
    echo "         rm -rf $ENV_DIR && bash setup_env.sh"
else
    echo "[INFO] Creo il virtualenv in: $ENV_DIR"
    mkdir -p "$(dirname "$ENV_DIR")"
    python3 -m virtualenv "$ENV_DIR"
fi

# ---------------------------------------------------------------------------
# 3. Attiva e installa le dipendenze
# ---------------------------------------------------------------------------
# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"

echo "[INFO] Python usato: $(python --version) | $(which python)"

pip install --upgrade pip
pip install -r "$(dirname "$(realpath "$0")")/requirements.txt"

deactivate

# ---------------------------------------------------------------------------
# 4. Riepilogo
# ---------------------------------------------------------------------------
echo ""
echo "====================================================================="
echo " Ambiente creato correttamente."
echo "====================================================================="
echo ""
echo " Path:    $ENV_DIR"
echo ""
echo " Negli script .slurm, attiva l'env PRIMA del comando python:"
echo "   source ${ENV_DIR}/bin/activate"
echo ""
echo " Per verificare i pacchetti installati:"
echo "   source ${ENV_DIR}/bin/activate && pip list"
echo "====================================================================="
