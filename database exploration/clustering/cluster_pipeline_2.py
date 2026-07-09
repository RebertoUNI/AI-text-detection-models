import numpy as np
import hdbscan
import json
import logging
import time
from datasets import load_dataset
from sklearn.metrics import pairwise_distances



# Configurazione logging per HPC
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_data():
    logging.info("Caricamento UMAP embeddings e metadati dal file .npz...")
    # Carichiamo il file compresso
    data = np.load("umap_full_results.npz")
    umap_embeddings = data['embeddings_2d']
    labels = data['labels']
    splits_tracker = data['splits'] # Array con scritto 'train', 'val', o 'test' per ogni punto
    
    logging.info("Caricamento dataset di testo da HuggingFace...")
    # Carichiamo l'intero dataset (tutti gli split)
    dataset = load_dataset("srikanthgali/ai-text-detection-pile-cleaned")
    
    def truncate_text(text, max_words=50):
        if not isinstance(text, str):
            return ""
        words = text.split()
        if len(words) > max_words:
            return " ".join(words[:max_words]) + "..."
        return text

    # È FONDAMENTALE mantenere l'ORDINE ESATTO in cui abbiamo unito gli embeddings
    # nello script HPC: 1. train, 2. val, 3. test
    texts_list = []
    
    # Su HuggingFace 'val' è spesso chiamato 'validation'. Controlliamo per sicurezza.
    val_split_name = "validation" if "validation" in dataset else "val"
    
    # Iteriamo nei 3 split in ordine rigoroso
    for split_name in ["train", val_split_name, "test"]:
        if split_name in dataset:
            logging.info(f"Elaborazione e troncamento testi per lo split: {split_name}...")
            # Estraiamo il testo e lo tronchiamo sequenzialmente
            for t in dataset[split_name]['text']:
                texts_list.append(truncate_text(t))
        else:
            logging.warning(f"Split '{split_name}' non trovato nel dataset HuggingFace!")
            
    texts = np.array(texts_list)
    
    logging.info(f"Trovate {len(texts)} frasi e {umap_embeddings.shape[0]} embeddings.")
    
    # Controllo di allineamento critico
    if len(texts) != umap_embeddings.shape[0]:
        raise ValueError(
            f"ERRORE ALLINEAMENTO: {len(texts)} frasi contro {umap_embeddings.shape[0]} embeddings!\n"
            "Verifica che il dataset testuale abbia esattamente le stesse righe degli embedding scaricati."
        )

    # Nota: Ritorno anche 'labels' e 'splits_tracker' perché ti saranno comodissimi per plottare!
    return texts, umap_embeddings, labels, splits_tracker


def tune_hdbscan(embeddings, n_jobs=8):
    logging.info("Inizio Hyperparameter Tuning per HDBSCAN...")
    
    param_grid = [
        {'min_cluster_size': 100, 'min_samples': 15},
        {'min_cluster_size': 300, 'min_samples': 30},
        {'min_cluster_size': 500, 'min_samples': 50},
        {'min_cluster_size': 1000, 'min_samples': 100},
        {'min_cluster_size': 1500, 'min_samples': 150},
        {'min_cluster_size': 1000, 'min_samples': 30}
    ]
    
    all_results = []   # <-- raccoglie tutto
    best_score = -1.0
    best_idx = 0

    for i, params in enumerate(param_grid):
        logging.info(f"Test configurazione {i}: {params}")
        start_time = time.time()
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=params['min_cluster_size'],
            min_samples=params['min_samples'],
            gen_min_span_tree=True,
            core_dist_n_jobs=n_jobs
        )
        clusterer.fit(embeddings)
        
        score = clusterer.relative_validity_
        elapsed = time.time() - start_time
        logging.info(f"Score ottenuto: {score:.4f} (Tempo: {elapsed:.1f}s)")

        result = {
            'run_id': i,
            'params': params,
            'model': clusterer,
            'score': score,
            'n_clusters': len(set(clusterer.labels_) - {-1}),
            'noise_ratio': float((clusterer.labels_ == -1).mean()),
        }
        all_results.append(result)

        if score > best_score:
            best_score = score
            best_idx = i

    logging.info(f"Miglior configurazione: run_id={best_idx}, score={best_score:.4f}")
    return all_results, best_idx

def extract_prototypes_and_save(all_results, best_idx, embeddings, texts, top_k=15):
    logging.info("Estrazione prototipi per tutte le configurazioni...")

    # Salva il riepilogo di tutti i run in un unico JSON
    summary = []
    for r in all_results:
        summary.append({
            'run_id': r['run_id'],
            'params': r['params'],
            'dbcv_score': r['score'],
            'n_clusters': r['n_clusters'],
            'noise_ratio': r['noise_ratio'],
            'is_best': r['run_id'] == best_idx
        })
    with open("hdbscan_all_params.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    for r in all_results:
        run_id = r['run_id']
        model  = r['model']
        params = r['params']
        labels = model.labels_

        # Labels per ogni run
        np.save(f"hdbscan_labels_run{run_id}.npy", labels)

        # File prototipi per ogni run
        unique_clusters = set(labels) - {-1}
        out = []
        out.append(f"Run {run_id} {'[BEST]' if run_id == best_idx else ''}")
        out.append(f"Parametri: {params}")
        out.append(f"DBCV Score: {r['score']:.4f} | Cluster: {len(unique_clusters)} | Noise: {r['noise_ratio']:.2%}")
        out.append("=" * 50)

        for cluster_id in sorted(unique_clusters):
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_points  = embeddings[cluster_indices]
            centroid        = cluster_points.mean(axis=0).reshape(1, -1)
            distances       = pairwise_distances(cluster_points, centroid, metric='euclidean').flatten()
            closest_local   = np.argsort(distances)[:top_k]
            closest_global  = cluster_indices[closest_local]

            out.append(f"\n[CLUSTER {cluster_id}] — {len(cluster_indices)} frasi")
            out.append("Frasi rappresentative:")
            for i, idx in enumerate(closest_global):
                clean = str(texts[idx]).replace('\n', ' ').replace('\r', ' ')
                out.append(f"  {i+1}. {clean}")
            out.append("-" * 30)

        fname = f"cluster_prototypes_run{run_id}.txt"
        with open(fname, "w", encoding="utf-8") as f:
            f.write("\n".join(out))
        logging.info(f"Salvato: {fname}")

    logging.info("Completato! File generati per tutti i run.")



if __name__ == "__main__":
    texts, umap_embeddings,labels, splits_tracker = load_data()
    all_results, best_idx = tune_hdbscan(umap_embeddings, n_jobs=32)
    extract_prototypes_and_save(all_results, best_idx, umap_embeddings, texts)