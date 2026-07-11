"""
Script: scarica train, val e test embeddings da HuggingFace,
li unisce mantenendo l'ordine, applica UMAP su tutto il dataset 
e salva i risultati.

"""

import os
# Ottimizzazione per HPC: diciamo a Numba di usare tutti i 32 core
os.environ["NUMBA_NUM_THREADS"] = "16"
os.environ["OMP_NUM_THREADS"] = "16"

import logging
import sys
import time
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

# ── Logging su stdout ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)



try:
    from cuml.manifold import UMAP
    log.info("Libreria cuML (GPU) caricata con successo!")
    USING_GPU = True
except ImportError:
    log.warning("cuML non trovata. Ripiego su umap-learn (CPU)...")
    import umap
    USING_GPU = False

# ── Configurazione ─────────────────────────────────────────────────────────────
REPO_ID    = "R-obi/ai-text-detection"
REPO_TYPE  = "dataset"
OUT_DIR    = Path("umap_output_full")
SPLITS     = ["train", "val", "test"]

UMAP_PARAMS = dict(
    n_neighbors  = 15,
    min_dist     = 0.1,
    n_components = 2,
    random_state = 42,
    verbose      = True,
    low_memory   = False, # Abbiamo 64GB di RAM, non serve risparmiare memoria
    n_jobs       = -1     # Usa tutti i core disponibili per alcune fasi di calcolo
)

# ── 1. Download e Caricamento ──────────────────────────────────────────────────
all_embeddings = []
all_labels = []
all_splits = [] # Array per tracciare la provenienza di ogni punto

for split_name in SPLITS:
    log.info("Downloading %s split...", split_name)
    
    emb_file = f"{split_name}/{split_name}_embeddings.npy"
    lbl_file = f"{split_name}/{split_name}_labels.npy"
    
    emb_path = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename=emb_file)
    lbl_path = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename=lbl_file)
    
    # Caricamento in RAM
    emb_data = np.load(emb_path)
    lbl_data = np.load(lbl_path)
    
    # Creiamo un array di stringhe della stessa lunghezza per tracciare lo split
    split_tracker = np.full(shape=emb_data.shape[0], fill_value=split_name)
    
    all_embeddings.append(emb_data)
    all_labels.append(lbl_data)
    all_splits.append(split_tracker)
    
    log.info("Loaded %s: %d embeddings.", split_name, emb_data.shape[0])

# ── 2. Concatenazione ──────────────────────────────────────────────────────────
log.info("Concatenating all splits...")
# np.vstack e np.concatenate mantengono rigorosamente l'ordine
embeddings = np.vstack(all_embeddings)
labels     = np.concatenate(all_labels)
splits     = np.concatenate(all_splits)

log.info("Total Embeddings shape : %s  dtype: %s", embeddings.shape, embeddings.dtype)
log.info("Total Labels shape     : %s  dtype: %s", labels.shape,     labels.dtype)
log.info("Total Splits shape     : %s  dtype: %s", splits.shape,     splits.dtype)

# Pulizia memoria (opzionale ma buona pratica)
del all_embeddings, all_labels, all_splits

# ── 3. UMAP ────────────────────────────────────────────────────────────────────
log.info("Starting UMAP...")
if USING_GPU:
    # Parametri per la GPU (cuML non supporta esattamente tutti i kwargs testuali di umap-learn, low_memory non serve)
    reducer = UMAP(
        n_neighbors=UMAP_PARAMS["n_neighbors"],
        min_dist=UMAP_PARAMS["min_dist"],
        n_components=UMAP_PARAMS["n_components"],
        random_state=UMAP_PARAMS["random_state"],
        verbose=UMAP_PARAMS["verbose"]
    )
else:
    reducer = umap.UMAP(**UMAP_PARAMS)
t0 = time.perf_counter()
emb_2d = reducer.fit_transform(embeddings)
elapsed = time.perf_counter() - t0
log.info("UMAP done in %.1f s", elapsed)
log.info("Output shape: %s", emb_2d.shape)
# ── 4. Salvataggio ─────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Salvataggio nel file compresso (più pulito)
npz_out = OUT_DIR / "umap_full_results.npz"

np.savez_compressed(
    npz_out, 
    embeddings_2d=emb_2d, 
    labels=labels,
    splits=splits # Ora hai salvato anche il tracciamento degli split!
)

log.info("Saved: %s  (compressed archive)", npz_out)
log.info("Done.")