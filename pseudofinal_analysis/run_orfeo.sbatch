#!/bin/bash
#SBATCH --job-name=umap-supcon
#SBATCH --partition=GPU            # adjust to your access on Orfeo: GPU / DGX
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=2                   # the script auto-uses all visible GPUs (1 also works)
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=umap_supcon_%j.log

# ---- environment -----------------------------------------------------------
# check exact module names with: module avail cuda python
module load cuda

# venv created beforehand on the login node with:
#   python -m venv $HOME/venvs/umap && source $HOME/venvs/umap/bin/activate
#   pip install numpy scipy pandas scikit-learn requests plotly \
#               datasets transformers peft accelerate torch
source source /u/dssc/rtittoto/scratch/envs/ai-text-detection/bin/activate

# keep the multi-GB HF cache off $HOME (quota!) - must be the SAME path used
# when running --download-only on the login node
export HF_HOME=$SCRATCH/hf_cache

# uncomment if compute nodes have no internet access
# (requires having run --download-only on the login node first):
# export HF_HUB_OFFLINE=1
# export HF_DATASETS_OFFLINE=1

# ---- run --------------------------------------------------------------------
python -u pseudofinal_analysis_orfeo.py \
    --data-dir  $SCRATCH/umap_data \
    --output-dir $SCRATCH/umap_outputs
