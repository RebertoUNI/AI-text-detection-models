import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# 1. Carica il modello Qwen
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
model.max_seq_length = 256

# 2. Carica il file CSV (sostituisci il path con quello corretto di Kaggle)
# Es. '/kaggle/input/tuo-dataset/file.csv'
csv_path = "deberta_push_to_error/out_gpt2.csv"
df = pd.read_csv(csv_path)

# Estrae le frasi dalla colonna 'text_in' mantenendo l'ordine originale
# Gestiamo eventuali valori nulli convertendoli in stringhe vuote
sentences = df["text_in"].fillna("").tolist()

print(f"Numero di frasi da elaborare: {len(sentences)}")

# 3. Genera gli embeddings
# Se su Kaggle hai attivato la GPU (T4 o P100), verrà usata automaticamente.
# Impostiamo show_progress_bar=True così vedi l'avanzamento.
embeddings = model.encode(
    sentences, batch_size=32, show_progress_bar=True, convert_to_numpy=True
)

# 4. Salva gli embeddings in formato .npy
output_path = "qwen3_embeddings.npy"
np.save(output_path, embeddings)

print(f"Embedding salvati con successo in '{output_path}'!")
print(f"Dimensioni della matrice: {embeddings.shape}")