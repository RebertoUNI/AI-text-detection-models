#!/bin/bash
#SBATCH --job-name=ai_detect_inference
#SBATCH --output=res_%j.log
#SBATCH --error=err_%j.log
#SBATCH --partition=gpu          # Nome della partizione GPU del tuo cluster
#SBATCH --gres=gpu:v100:1        # Richiede esattamente 1 GPU V100
#SBATCH --cpus-per-task=4        # Core CPU dedicati ai num_workers del dataloader
#SBATCH --mem=32G                # Memoria RAM di sistema richiesta

# Carica i moduli del cluster (adatta in base al tuo HPC, es. Anaconda o CUDA)
module load cuda/12.1

# Esegui lo script
python inferenza_deberta_HPC --batch_size 256 --output_file risultati_finali.csv