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
        local_files_only=True  # Forza uso cache locale, errore se non trovato
    )

    print("Model uploaded!")

    return model

load_model_from_cache("Qwen/Qwen3-Embedding-0.6B", "/Users/roberto/Università/Deep learning")
