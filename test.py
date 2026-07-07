import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import pandas as pd

from model_upload import load_model_from_cache


model = load_model_from_cache("Qwen/Qwen3-Embedding-0.6B", "/Users/roberto/Università/Deep learning")


# ── IMPOSTAZIONI DI CONTROLLO ──────────────────────────────────────────────
MODALITA_TEST = True  

NUM_SAMPLES = 10 if MODALITA_TEST else 20000  

# File di output degli embedding
TRAIN_FILE = 'test_train_embeddings.pt' if MODALITA_TEST else 'train_embeddings.pt'
VAL_FILE   = 'test_val_embeddings.pt'   if MODALITA_TEST else 'val_embeddings.pt'
TEST_FILE  = 'test_test_embeddings.pt'  if MODALITA_TEST else 'test_embeddings.pt'

# Percorsi dei file parquet scaricati sul tuo Mac
PATH_TRAIN = "data/train-00000-of-00003.parquet"
PATH_VAL   = "data/validation-00000-of-00001.parquet"
PATH_TEST  = "data/test-00000-of-00001.parquet"

# ── FUNZIONE DI ELABORAZIONE LOCALE CORRETTA ───────────────────────────────
def process_local_parquet(file_path, num_samples, output_filename, split_name):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Non trovo il file {file_path}. Assicurati di averlo scaricato.")
        
    print(f"\n--- Elaborazione split locale: {split_name} ---")
    
    # Legge le righe dal file parquet locale
    df = pd.read_parquet(file_path, columns=['text', 'generated'], engine='pyarrow').head(num_samples)

    # --- INSERISCI QUESTO CODICE DA QUI ---
    print("\n[DEBUG] Contenuto della prima riga estratta:")
    print(f"Testo: {df['text'].iloc[0]}")
    print(f"Label (generated): {df['generated'].iloc[0]}")
    # --------------------------------------

    sentences = df['text'].tolist()
    labels = df['generated'].tolist()
    
    print(f"Campioni caricati: {len(sentences)}")
    print(f"Generazione embedding con Qwen per {split_name}...")
    
    # Calcolo effettivo degli embedding con Qwen
    X_embeddings = model.encode(
        sentences, 
        batch_size=1,  # <--- FORZA L'ELABORAZIONE DI 1 FRASE ALLA VOLTA
        convert_to_tensor=True, 
        show_progress_bar=True
    )
    y_labels = torch.tensor(labels, dtype=torch.long)
    
    # Salvataggio permanente (manteniamo la chiave 'labels' così il codice della tua rete non cambia)
    torch.save({
        'embeddings': X_embeddings,
        'labels': y_labels
    }, output_filename)
    print(f"Salvato con successo in '{output_filename}' | Shape: {X_embeddings.shape}")

# ── ESECUZIONE ─────────────────────────────────────────────────────────────
process_local_parquet(PATH_TRAIN, NUM_SAMPLES, TRAIN_FILE, "Train")
process_local_parquet(PATH_VAL, NUM_SAMPLES, VAL_FILE, "Validation")
process_local_parquet(PATH_TEST, NUM_SAMPLES, TEST_FILE, "Test")

print("\n[FINISH] Errore risolto! Tutti i file .pt di test sono pronti.")