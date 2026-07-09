import numpy as np

# 1. Carica il file .npy
cluster_id = np.load('/Users/roberto/Università/Deep learning/AI-text-detection-models/database exploration/clustering/full_dataset/hdbscan_labels_run3.npy')

# 2. Salva l'array in formato CSV
np.savetxt("hdbscan_readable_labels.csv", cluster_id, fmt="%d")