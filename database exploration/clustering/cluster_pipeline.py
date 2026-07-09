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

def load_data():
    logging.info("Caricamento dataset da HuggingFace...")
    # Carichiamo solo lo split 'train'
    dataset = load_dataset("srikanthgali/ai-text-detection-pile-cleaned", split="train")
    
    def truncate_text(text, max_words=50):
        if not isinstance(text, str):
            return ""
        words = text.split()
        if len(words) > max_words:
            return " ".join(words[:max_words]) + "..."
        return text
    
    # ASSUNZIONE: La colonna contenente le frasi si chiama 'text'
    texts = np.array([truncate_text(t) for t in dataset['text']])
    
    logging.info("Caricamento UMAP embeddings (2D)...")
    umap_embeddings = np.load("umap_embeddings_2d.npy")
    
    logging.info(f"Trovate {len(texts)} frasi e {umap_embeddings.shape[0]} embeddings.")
    
    # Controllo di allineamento critico
    if len(texts) != umap_embeddings.shape[0]:
        raise ValueError("ERRORE: Il numero di frasi non corrisponde al numero di embeddings! "
                         "Verifica se durante l'inferenza sono stati scartati dei dati.")

    return texts, umap_embeddings

def tune_hdbscan(embeddings, n_jobs=8):
    logging.info("Inizio Hyperparameter Tuning per HDBSCAN...")
    
    # Griglia di parametri. Puoi espanderla, ma attento ai tempi di calcolo.
    # min_cluster_size: quanti punti minimi formano un argomento valido? (es. 100 frasi)
    param_grid = [
        {'min_cluster_size': 100, 'min_samples': 15},
        {'min_cluster_size': 300, 'min_samples': 30},
        {'min_cluster_size': 500, 'min_samples': 50}
    ]
    
    best_score = -1.0
    best_model = None
    best_params = {}
    
    for params in param_grid:
        logging.info(f"Test configurazione: {params}")
        start_time = time.time()
        
        # Inizializza HDBSCAN
        # core_dist_n_jobs=n_jobs sfrutta gli 8 processori dell'HPC
        # gen_min_span_tree=True è fondamentale per calcolare il punteggio di validità
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=params['min_cluster_size'],
            min_samples=params['min_samples'],
            gen_min_span_tree=True,
            core_dist_n_jobs=-1 
        )
        
        clusterer.fit(embeddings)
        
        # Usa il DBCV (Density-Based Clustering Validation) integrato in HDBSCAN
        # Valori più alti (vicini a 1) indicano cluster più densi e ben separati
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
    
    # 1. Salva i parametri ottimali
    with open("hdbscan_best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=4)
        
    # 2. Salva le etichette per ogni punto (700k punti)
    labels = best_model.labels_
    np.save("hdbscan_best_labels.npy", labels)
    
    # Identifica i cluster escludendo -1 (il rumore/noise in HDBSCAN)
    unique_clusters = set(labels) - {-1}
    logging.info(f"Trovati {len(unique_clusters)} cluster semantici validi.")
    
    prototypes_output = []
    prototypes_output.append(f"Report Prototipi Generato Automaticamente\n")
    prototypes_output.append(f"Parametri Modello: {best_params}\n")
    prototypes_output.append("="*50 + "\n")
    
    for cluster_id in unique_clusters:
        # Trova indici e coordinate dei punti appartenenti a questo cluster
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_points = embeddings[cluster_indices]
        
        # Calcola il centroide geometrico (media delle coordinate 2D)
        centroid = cluster_points.mean(axis=0).reshape(1, -1)
        
        # Calcola la distanza Euclidea dal centroide per tutti i punti del cluster
        distances = pairwise_distances(cluster_points, centroid, metric='euclidean').flatten()
        
        # Prendi gli indici dei 'top_k' punti più vicini al centroide
        closest_local_indices = np.argsort(distances)[:top_k]
        closest_global_indices = cluster_indices[closest_local_indices]
        
        # Formatta il testo per l'LLM
        prototypes_output.append(f"\n[CLUSTER {cluster_id}]")
        prototypes_output.append(f"Dimensione: {len(cluster_indices)} frasi")
        prototypes_output.append("Frasi rappresentative (ordinate per vicinanza al centroide):")
        
        for i, idx in enumerate(closest_global_indices):
            # Puliamo eventuali a capo indesiderati per mantenere un file di testo pulito
            clean_text = str(texts[idx]).replace('\n', ' ').replace('\r', ' ')
            prototypes_output.append(f"{i+1}. {clean_text}")
            
        prototypes_output.append("-" * 30)
        
    # 3. Salva il file TXT per l'analisi LLM manuale
    with open("cluster_prototypes_for_llm.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(prototypes_output))
        
    logging.info("Processo completato! File generati:")
    logging.info("- hdbscan_best_params.json")
    logging.info("- hdbscan_best_labels.npy")
    logging.info("- cluster_prototypes_for_llm.txt")

if __name__ == "__main__":
    texts, umap_embeddings = load_data()
    best_model, best_params = tune_hdbscan(umap_embeddings, n_jobs=8)
    extract_prototypes_and_save(best_model, best_params, umap_embeddings, texts)