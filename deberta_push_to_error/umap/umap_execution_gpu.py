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

# Percorso locale del nuovo file di embedding richiesto
NEW_EMB_PATH = Path("deberta_push_to_error/qwen embeddings/qwen3_embeddings.npy")

UMAP_PARAMS = dict(
    n_neighbors  = 15,
    min_dist     = 0.1,
    n_components = 2,
    random_state = 42,
    verbose      = True,
    low_memory   = False, # Abbiamo 64GB di RAM, non serve risparmiare memoria
    n_jobs       = -1     # Usa tutti i core disponibili per alcune fases di calcolo
)

# ── 1. Download e Caricamento split originali ──────────────────────────────────
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

# ── 1.b Caricamento Nuovi Embedding (test_2) da locale ──────────────────────────
log.info("Caricamento nuovi embedding da locale: %s", NEW_EMB_PATH)
if not NEW_EMB_PATH.exists():
    log.error("Il file richiesto non esiste al percorso specificato!")
    sys.exit(1)

new_emb_data = np.load(NEW_EMB_PATH)
log.info("Loaded test_2 (nuovi): %d embeddings.", new_emb_data.shape[0])

# Generiamo etichette fittizie (es. -1 o zeri) e il tracker per test_2
# Nota: UMAP in modalità non supervisionata ignora le labels, servono solo per il grafico successivo.
new_lbl_data = np.zeros(new_emb_data.shape[0], dtype=all_labels[0].dtype) 
new_split_tracker = np.full(shape=new_emb_data.shape[0], fill_value="test_2")

# Appendiamo i nuovi dati alle liste globali prima del vstack
all_embeddings.append(new_emb_data)
all_labels.append(new_lbl_data)
all_splits.append(new_split_tracker)


# ── 2. Concatenazione di TUTTI i dati ──────────────────────────────────────────
log.info("Concatenating all splits (compresi i nuovi embedding)...")
embeddings = np.vstack(all_embeddings)
labels     = np.concatenate(all_labels)
splits     = np.concatenate(all_splits)

log.info("Total Embeddings shape : %s  dtype: %s", embeddings.shape, embeddings.dtype)
log.info("Total Labels shape     : %s  dtype: %s", labels.shape,     labels.dtype)
log.info("Total Splits shape     : %s  dtype: %s", splits.shape,     splits.dtype)

# Memorizziamo quanti elementi c'erano nei vecchi split per poter fare lo slicing alla fine
num_vecchi = sum(e.shape[0] for e in all_embeddings[:-1])

# Pulizia memoria
del all_embeddings, all_labels, all_splits

# ── 3. UMAP globale ────────────────────────────────────────────────────────────
log.info("Starting UMAP su tutto il dataset unito...")
if USING_GPU:
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
log.info("Output shape globale: %s", emb_2d.shape)

# ── 4. Separazione dei Risultati per il salvataggio ────────────────────────────
# Estraiamo i vettori 2D corrispondenti ai vecchi split e al nuovo split (test_2)
emb_2d_vecchi = emb_2d[:num_vecchi]
emb_2d_test_2 = emb_2d[num_vecchi:]

log.info("Split output 2D -> Vecchi: %s, test_2: %s", emb_2d_vecchi.shape, emb_2d_test_2.shape)

# ── 5. Salvataggio ─────────────────────────────────────────────────────────────

npz_out = "deberta_push_to_error/umap/umap_full_results.npz"

np.savez_compressed(
    npz_out, 
    embeddings_2d=emb_2d_vecchi, # Mantiene il nome originale per non rompere script successivi
    test_2=emb_2d_test_2,         # Salvato specificamente con il nome "test_2"
    labels=labels,               # Array globale unito delle etichette
    splits=splits                # Array globale unito dei tag degli split ("train", "val", "test", "test_2")
)

log.info("Saved: %s  (compressed archive con chiave 'test_2')", npz_out)
log.info("Done.")