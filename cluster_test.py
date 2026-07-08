import numpy as np
import hdbscan
import json
import logging
import time
import os
from datasets import load_dataset
from sklearn.metrics import pairwise_distances

# Configurazione logging per HPC
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_and_sample_data(sample_fraction=0.05, random_seed=42):
    logging.info("Caricamento dataset da HuggingFace...")
    dataset = load_dataset("srikanthgali/ai-text-detection-pile-cleaned", split="train")
    texts = np.array(dataset['text']) # Convertito in numpy array per indicizzazione facilitata
    
    logging.info("Caricamento UMAP embeddings (2D)...")
    umap_embeddings = np.load("umap_embeddings_2d.npy")
    
    total_size = len(texts)
    logging.info(f"Trovate {total_size} frasi e {umap_embeddings.shape[0]} embeddings.")
    
    if total_size != umap_embeddings.shape[0]:
        raise ValueError("ERRORE: Il numero di frasi non corrisponde al numero di embeddings!")
        
    # --- LOGICA DI CAMPIONAMENTO ---
    sample_size = int(total_size * sample_fraction)
    logging.info(f"Campionamento del {sample_fraction*100}%: seleziono {sample_size} elementi.")
    
    # Fissiamo il seed per avere risultati riproducibili
    np.random.seed(random_seed)
    
    # Generiamo indici casuali univoci
    sampled_indices = np.random.choice(total_size, sample_size, replace=False)
    
    # Applichiamo gli stessi indici a entrambi per mantenere l'allineamento
    sampled_texts = texts[sampled_indices]
    sampled_embeddings = umap_embeddings[sampled_indices]
    
    return sampled_texts, sampled_embeddings

def tune_hdbscan(embeddings, n_jobs=8):
    logging.info("Inizio Hyperparameter Tuning per HDBSCAN (Versione Sample 5%)...")
    
    # Griglia di parametri ADATTATA per il 5% del dataset.
    # Con ~28.8k punti, un cluster di 15-30 frasi è già un segnale forte.
    param_grid = [
        {'min_cluster_size': 15, 'min_samples': 5},
        {'min_cluster_size': 30, 'min_samples': 10},
        {'min_cluster_size': 50, 'min_samples': 15}
    ]
    
    best_score = -1.0
    best_model = None
    best_params = {}
    
    for params in param_grid:
        logging.info(f"Test configurazione: {params}")
        start_time = time.time()
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=params['min_cluster_size'],
            min_samples=params['min_samples'],
            gen_min_span_tree=True,
            core_dist_n_jobs=-1 
        )
        
        clusterer.fit(embeddings)
        
        # Validazione DBCV
        score = clusterer.relative_validity_
        
        elapsed = time.time() - start_time
        logging.info(f"Score ottenuto: {score:.4f} (Tempo: {elapsed:.1f}s)")
        
        if score > best_score:
            best_score = score
            best_model = clusterer
            best_params = params
            best_params['dbcv_score'] = score
            
    logging.info(f"Miglior configurazione trovata: {best_params}")
    return best_model, best_params

def extract_prototypes_and_save(best_model, best_params, embeddings, texts, top_k=15):
    logging.info("Estrazione dei prototipi e salvataggio risultati...")
    
    with open("hdbscan_sample_best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=4)
        
    labels = best_model.labels_
    np.save("hdbscan_sample_labels.npy", labels)
    
    unique_clusters = set(labels) - {-1}
    noise_points = np.sum(labels == -1)
    
    logging.info(f"Trovati {len(unique_clusters)} cluster validi.")
    logging.info(f"Punti classificati come rumore (outliers): {noise_points} su {len(labels)}")
    
    prototypes_output = []
    prototypes_output.append(f"Report Prototipi Generato Automaticamente (RUN SUL 5%)\n")
    prototypes_output.append(f"Parametri Modello: {best_params}\n")
    prototypes_output.append("="*50 + "\n")
    
    for cluster_id in unique_clusters:
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_points = embeddings[cluster_indices]
        
        centroid = cluster_points.mean(axis=0).reshape(1, -1)
        distances = pairwise_distances(cluster_points, centroid, metric='euclidean').flatten()
        
        # Preveniamo errori se un cluster ha meno di top_k elementi
        actual_top_k = min(top_k, len(cluster_indices))
        closest_local_indices = np.argsort(distances)[:actual_top_k]
        closest_global_indices = cluster_indices[closest_local_indices]
        
        prototypes_output.append(f"\n[CLUSTER {cluster_id}]")
        prototypes_output.append(f"Dimensione: {len(cluster_indices)} frasi")
        prototypes_output.append("Frasi rappresentative:")
        
        for i, idx in enumerate(closest_global_indices):
            clean_text = str(texts[idx]).replace('\n', ' ').replace('\r', ' ')
            prototypes_output.append(f"{i+1}. {clean_text}")
            
        prototypes_output.append("-" * 30)
        
    with open("cluster_sample_prototypes_for_llm.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(prototypes_output))
        
    logging.info("Processo sul campione completato!")

if __name__ == "__main__":
    # Avvia la pipeline campionando il 5%
    texts, umap_embeddings = load_and_sample_data(sample_fraction=0.05)
    best_model, best_params = tune_hdbscan(umap_embeddings, n_jobs=8)
    extract_prototypes_and_save(best_model, best_params, umap_embeddings, texts)