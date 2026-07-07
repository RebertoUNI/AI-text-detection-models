"""
Script: scarica train_embeddings.npy e train_label.npy da HuggingFace,
poi plotta con UMAP.

Requisiti:
    pip install numpy umap-learn matplotlib huggingface_hub
"""

import numpy as np
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
# Some editors/linters may not resolve the top-level "umap" package.
# Try the public API location used by umap-learn as a fallback.
try:
    import umap
except Exception:
    try:
        # fallback to the submodule where the UMAP class lives
        import umap.umap_ as umap
    except Exception as e:
        raise ImportError(
            "Could not import 'umap'. Install the 'umap-learn' package: pip install umap-learn"
        ) from e

REPO_ID = "R-obi/ai-text-detection"
REPO_TYPE = "dataset"

# 1. Scarica file
emb_path = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE,
                            filename="train/train_embeddings.npy")
lbl_path = hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE,
                            filename="train/train_labels.npy")

# 2. Carica dati
embeddings = np.load(emb_path)
labels = np.load(lbl_path)

print("Embeddings shape:", embeddings.shape)
print("Labels shape:", labels.shape)

# 3. Riduzione UMAP a 2D
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
emb_2d = reducer.fit_transform(embeddings)

# 4. Plot
plt.figure(figsize=(9, 7))
scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap="coolwarm",
                       s=5, alpha=0.7)
plt.colorbar(scatter, label="Label")
plt.title("UMAP projection - AI Text Detection embeddings")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.tight_layout()
plt.savefig("umap_plot.png", dpi=200)
plt.show()
