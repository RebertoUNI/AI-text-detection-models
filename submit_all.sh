#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# submit_all.sh
# ─────────────────────────────────────────────────────────────────────────
# Versione per layout "piatto": tutti i file (.py e .slurm) nella stessa
# cartella di lavoro (nessuna sottocartella slurm/).
#
# Lancia tutti i job SLURM per i 4 modelli:
#   train_X (GPU) -> test_X (GPU, dipendente dal training tramite --dependency=afterok)
# e, quando TUTTI i test sono finiti, lancia compare_results.py in un job
# leggero (CPU-only) per generare la tabella finale.
#
# Uso (dalla cartella dove stanno tutti i file):
#   bash submit_all.sh
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p logs results

declare -A TEST_JOB_IDS

for MODEL in fcnn papercnn deberta qwen; do
    TRAIN_ID=$(sbatch --parsable "train_${MODEL}.slurm")
    echo "Sottomesso train_${MODEL}: job ${TRAIN_ID}"

    TEST_ID=$(sbatch --parsable --dependency=afterok:${TRAIN_ID} "test_${MODEL}.slurm")
    echo "Sottomesso test_${MODEL} (dipendente da ${TRAIN_ID}): job ${TEST_ID}"

    TEST_JOB_IDS[$MODEL]=$TEST_ID
done

# Job di confronto finale: parte solo quando TUTTI i test sono completati con successo
# (sintassi SLURM: afterok:id1:id2:id3:id4 -> aspetta che TUTTI i job listati finiscano OK)
DEP_LIST=$(IFS=:; echo "${TEST_JOB_IDS[*]}")

sbatch --dependency="afterok:${DEP_LIST}" --job-name=compare_results \
       --output=logs/compare_results_%j.out --error=logs/compare_results_%j.err \
       --time=00:10:00 --cpus-per-task=1 --mem=2G \
       --wrap="source \$HOME/envs/ai-text-detection/bin/activate && python compare_results.py"

echo ""
echo "Tutti i job sottomessi. Segui l'avanzamento con: squeue -u \$USER"
