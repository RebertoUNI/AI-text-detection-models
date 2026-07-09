import numpy as np
from collections import Counter

def stampa_frequenze(file_path):
    """
    Legge un file .npy e stampa il numero di volte che compare ogni numero
    
    Args:
        file_path (str): Percorso del file .npy
    """
    try:
        # Carica il file .npy
        dati = np.load(file_path)
        
        # Converte in lista se necessario (per array multidimensionali)
        if dati.ndim > 1:
            dati = dati.flatten()
        
        # Conta le occorrenze
        conteggio = Counter(dati)
        
        # Stampa i risultati in ordine crescente
        print("Numero : Occorrenze")
        print("-" * 20)
        for numero in sorted(conteggio.keys()):
            print(f"{numero:6} : {conteggio[numero]}")
            
        # Opzionale: stampa anche il totale
        print("-" * 20)
        print(f"Totale elementi: {len(dati)}")
        print(f"Numeri unici: {len(conteggio)}")
        
    except FileNotFoundError:
        print(f"Errore: Il file '{file_path}' non è stato trovato.")
    except Exception as e:
        print(f"Errore durante la lettura del file: {e}")

# Esempio di utilizzo
if __name__ == "__main__":
    # Sostituisci con il percorso del tuo file
    stampa_frequenze("/Users/roberto/Università/Deep learning/AI-text-detection-models/database exploration/clustering/full_dataset/hdbscan_labels_run3.npy")