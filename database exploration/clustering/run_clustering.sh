#!/bin/bash
#SBATCH --job-name=hdbscan_cluster          # Nome del job
#SBATCH --account=dssc                      # Account di fatturazione
#SBATCH --partition=GPU                     # Partizione
#SBATCH --gres=gpu:V100:1                   # Richiesta GPU (V100)
#SBATCH --cpus-per-task=8                   # Numero di core CPU (per n_jobs=8 in HDBSCAN)
#SBATCH --mem=64G                           # Memoria RAM (64 GB)
#SBATCH --time=10:00:00                     # Tempo massimo di esecuzione (10 ore)
#SBATCH --output=cluster_logs_%j.out        # File per lo standard output (%j inserisce il Job ID)
#SBATCH --error=cluster_logs_%j.err         # File per gli errori

echo "=== Inizio del job di clustering ==="
date

# 1. (Opzionale ma raccomandato) Carica i moduli necessari o attiva il tuo virtual environment
# Scommenta e modifica le righe seguenti in base alla configurazione del tuo HPC:
# module load python/3.x.x
# source /orfeo/cephfs/home/dssc/rtittoto/tuo_ambiente_virtuale/bin/activate

# 2. Spostati nella cartella dove si trova il tuo script e i file .npy
# Sostituisci il percorso con quello corretto se diverso
cd /orfeo/cephfs/home/dssc/rtittoto/AI-text-detection-models/umap_output/

# 3. Lancia lo script Python
echo "Lancio di cluster_pipeline.py..."
python cluster_pipeline.py  # (oppure cluster_test.py se hai tenuto quel nome)

echo "=== Job completato ==="
date