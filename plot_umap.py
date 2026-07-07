"""
Script: scarica train_embeddings.npy e train_labels.npy da HuggingFace,
applica UMAP e salva i risultati su disco (no display grafico).

Adatto per esecuzione su HPC / headless.

Requisiti:
    pip install numpy umap-learn huggingface_hub tqdm
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

try:
    import umap
except ImportError:
    try:
        import umap.umap_ as umap
    except ImportError as e:
        raise ImportError(
            "Could not import 'umap'. Install with: pip install umap-learn"
        ) from e

# ── Logging su stdout (visibile nei log SLURM / PBS) ──────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ── Configurazione ─────────────────────────────────────────────────────────────
REPO_ID    = "R-obi/ai-text-detection"
REPO_TYPE  = "dataset"
OUT_DIR    = Path("umap_output")

UMAP_PARAMS = dict(
    n_neighbors  = 15,
    min_dist     = 0.1,
    n_components = 2,
    random_state = 42,
    verbose      = True,   # stampa il progresso epoch-by-epoch
    low_memory   = False,  # True se la RAM è limitata
)

# ── 1. Download ────────────────────────────────────────────────────────────────
log.info("Downloading embeddings from HuggingFace …")
emb_path = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE,
                            filename="train/train_embeddings.npy")
lbl_path = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE,
                            filename="train/train_labels.npy")

# ── 2. Caricamento ─────────────────────────────────────────────────────────────
log.info("Loading data …")
embeddings = np.load(emb_path)
labels     = np.load(lbl_path)
log.info("Embeddings shape : %s  dtype: %s", embeddings.shape, embeddings.dtype)
log.info("Labels shape     : %s  dtype: %s", labels.shape,     labels.dtype)

# ── 3. UMAP ────────────────────────────────────────────────────────────────────
log.info("Starting UMAP with params: %s", UMAP_PARAMS)
reducer = umap.UMAP(**UMAP_PARAMS)

t0 = time.perf_counter()
emb_2d = reducer.fit_transform(embeddings)
elapsed = time.perf_counter() - t0

log.info("UMAP done in %.1f s", elapsed)
log.info("Output shape: %s", emb_2d.shape)

# ── 4. Salvataggio ─────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)

emb_out = OUT_DIR / "umap_embeddings_2d.npy"
lbl_out = OUT_DIR / "umap_labels.npy"
npz_out = OUT_DIR / "umap_results.npz"

np.save(emb_out, emb_2d)
np.save(lbl_out, labels)

# file .npz = tutto in un unico archivio (comodo per caricare altrove)
np.savez_compressed(npz_out, embeddings_2d=emb_2d, labels=labels)

log.info("Saved: %s", emb_out)
log.info("Saved: %s", lbl_out)
log.info("Saved: %s  (compressed archive)", npz_out)
log.info("Done.")