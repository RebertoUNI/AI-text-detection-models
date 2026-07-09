#!/bin/bash
#SBATCH --job-name=umap_batch
#SBATCH --partition=EPYC
#SBATCH --account=dssc
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

cd /u/dssc/rtittoto/AI-text-detection/"databas exploration"/umap

# Attiva l'ambiente virtuale
source /u/dssc/rtittoto/scratch/envs/ai-text-detection-gpu/bin/activate


pip install umap-learn

# Esegue lo script Python
python umap_execution.py