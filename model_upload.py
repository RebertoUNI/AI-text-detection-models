import os
from sentence_transformers import SentenceTransformer
def load_model_from_cache(model_name: str, cache_dir: str) -> SentenceTransformer:
    """
    Load a SentenceTransformer model from a specified cache directory.

    Args:
        model_name (str): The name of the model to load.
        cache_dir (str): The directory where the model is cached.

    Returns:
        SentenceTransformer: The loaded model.
    """

    # 1. Imposta la directory cache PERSISTENTE
    os.environ['HF_HOME'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = cache_dir

    # 2. Carica il modello (NON scaricherà mai se già presente)
    model = SentenceTransformer(
        model_name,
        cache_folder=cache_dir,
        local_files_only=True,
        model_kwargs={"attn_implementation": "eager"} # Aggiunto per evitare errori di compatibilità con alcune versioni di PyTorch su M1/M2 Mac
    )

    print("Model uploaded!")

    return model

# In fondo a model_upload.py cambia così:
if __name__ == "__main__":
    load_model_from_cache("Qwen/Qwen3-Embedding-0.6B", "/Users/roberto/Università/Deep learning")