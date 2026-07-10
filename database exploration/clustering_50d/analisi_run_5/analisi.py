import os
import numpy as np
import pandas as pd

# 1. Definizione dei percorsi dei file (come da te indicati)
hdbscan_path = 'database exploration/clustering_50d/out/hdbscan_labels_run5.npy'
umap_npz_path = 'database exploration/umap/50d/full_dataset/umap_full_dataset_50d.npz'

# 2. Caricamento dei dati
print("Caricamento dei file in corso...")
cluster_labels = np.load(hdbscan_path)

with np.load(umap_npz_path) as data:
    labels = data['labels']
    splits = data['splits']

# Verifica di consistenza (controllo che tutti i vettori abbiano la stessa lunghezza)
assert len(cluster_labels) == len(labels) == len(splits), \
    f"Errore di consistenza: lunghezze diverse! Cluster: {len(cluster_labels)}, Labels: {len(labels)}, Splits: {len(splits)}"

# 3. Creazione del DataFrame di supporto
df = pd.DataFrame({
    'cluster': cluster_labels,
    'label': labels,
    'split': splits
})

# Nota su HDBSCAN: Il valore -1 indica i punti di rumore (noise) non assegnati a nessun cluster concreto.

# =====================================================================
# ANALISI GLOBAL (Tutti gli split assieme)
# =====================================================================
print("\n=== STATISTICHE GLOBALI PER CLUSTER ===")

# Utilizziamo crosstab per contare le occorrenze di 0 e 1 per ciascun cluster
overall_stats = pd.crosstab(df['cluster'], df['label']).rename(columns={0: 'human_0', 1: 'ai_1'})

# Assicuriamoci che entrambe le colonne esistano (gestione casi limite)
if 'human_0' not in overall_stats.columns: overall_stats['human_0'] = 0
if 'ai_1' not in overall_stats.columns: overall_stats['ai_1'] = 0

# Calcolo del totale, dei rapporti e della percentuale di AI
overall_stats['total'] = overall_stats['human_0'] + overall_stats['ai_1']
overall_stats['ratio_human_ai'] = overall_stats['human_0'] / overall_stats['ai_1']
overall_stats['ratio_ai_human'] = overall_stats['ai_1'] / overall_stats['human_0']
overall_stats['pct_ai'] = (overall_stats['ai_1'] / overall_stats['total']) * 100

# Mostra i risultati a video
print(overall_stats.to_string())

# Esporta in CSV
overall_stats.to_csv('cluster_stats_global.csv')
print("-> File 'cluster_stats_global.csv' salvato con successo.")


# =====================================================================
# ANALISI DISAGGREGATA PER SPLIT
# =====================================================================
print("\n=== STATISTICHE PER OGNI SPLIT ===")

# Crosstab nidificata inserendo sia 'split' che 'cluster' nell'indice delle righe
split_stats = pd.crosstab([df['split'], df['cluster']], df['label']).rename(columns={0: 'human_0', 1: 'ai_1'})

if 'human_0' not in split_stats.columns: split_stats['human_0'] = 0
if 'ai_1' not in split_stats.columns: split_stats['ai_1'] = 0

# Calcolo metriche per singolo split/cluster
split_stats['total'] = split_stats['human_0'] + split_stats['ai_1']
split_stats['ratio_human_ai'] = split_stats['human_0'] / split_stats['ai_1']
split_stats['ratio_ai_human'] = split_stats['ai_1'] / split_stats['human_0']
split_stats['pct_ai'] = (split_stats['ai_1'] / split_stats['total']) * 100

# Mostra i risultati a video
print(split_stats.to_string())

# Esporta in CSV
split_stats.to_csv('cluster_stats_by_split.csv')
print("-> File 'cluster_stats_by_split.csv' salvato con successo.")