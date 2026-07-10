import pandas as pd

# 1. Carica il file CSV
df = pd.read_csv('deberta_push_to_error/results_2_1.csv')

# 2. CORREZIONE: Trasforma la colonna 'confidence' da stringa (es. "79.88%") a numero (es. 79.88)
# Usiamo .str.rstrip('%') per togliere il simbolo alla fine e .astype(float) per convertirlo
df['confidence'] = df['confidence'].astype(str).str.rstrip('%').astype(float)

# Se preferisci avere la media espressa in una scala da 0 a 1 (es. 0.7988 invece di 79.88), 
# puoi scommentare la riga successiva:
# df['confidence'] = df['confidence'] / 100

# 3. Calcola il conteggio di LABEL_0 e LABEL_1 per ogni categoria
conteggio_labels = (
    df.groupby(['category ID', 'category name', 'predicted_label'])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

# 4. Calcola la media della colonna 'confidence' (ora che è un numero funzionerà senza errori)
media_confidenza = (
    df.groupby(['category ID', 'category name'])['confidence']
    .mean()
    .reset_index(name='media_confidence')
)

# 5. Unisci i due risultati
risultato_finale = pd.merge(conteggio_labels, media_confidenza, on=['category ID', 'category name'])

# Mostra il risultato
print(risultato_finale)

# (Opzionale) Salva il risultato in un nuovo file CSV
risultato_finale.to_csv('risultato_categorie.csv', index=False)