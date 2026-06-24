# ============================================
# Model download e caching per SentenceTransformer
# ============================================
from sentence_transformers import SentenceTransformer
import os

# Write your own path to store the models
MODELS_DIR = "/Users/roberto/Università/Deep learning" 

os.makedirs(MODELS_DIR, exist_ok=True)

model_name = "Qwen/Qwen3-Embedding-0.6B"

from huggingface_hub import snapshot_download

model_path = snapshot_download(
    repo_id=model_name,
    cache_dir=MODELS_DIR, #download only if not already cached
)
print(f"Model saved in: {model_path}")