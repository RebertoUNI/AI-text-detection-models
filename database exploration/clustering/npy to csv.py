import numpy as np

# 1. Carica il file .npy
cluster_id = np.load('umap_output/hdbscan_best_labels.npy')

# 2. Salva l'array in formato CSV
np.savetxt("hdbscan_readable_labels.csv", cluster_id, fmt="%d")