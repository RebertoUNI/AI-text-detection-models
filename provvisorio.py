import pandas as pd

# 1. Carica i due file CSV
# Sostituisci 'file1.csv' e 'file2.csv' con i nomi effettivi dei tuoi file
df1 = pd.read_csv('results_with_uncertainty_1.csv')  # Struttura: text_out, tone, predicted_label, confidence
df2 = pd.read_csv('deberta_push_to_error/out_gpt2.csv')  # Struttura: category ID, category name, tone, text_in

# 2. Seleziona le colonne dal primo file che vuoi aggiungere (escludendo 'tone' che è già presente nel secondo)
colonne_da_aggiungere = df1[['text_out', 'predicted_label', 'confidence']]

# 3. Unisci i due DataFrame affiancandoli (axis=1)
# Questo manterrà l'ordine esatto: category ID, category name, tone, text_in, text_out, predicted_label, confidence
df_unito = pd.concat([df2, colonne_da_aggiungere], axis=1)

# 4. Salva il risultato in un nuovo file CSV senza includere l'indice di riga
df_unito.to_csv('file_unito.csv', index=False)

print("I file sono stati uniti con successo in 'file_unito.csv'!")