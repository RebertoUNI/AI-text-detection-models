import pandas as pd

# Leggi il CSV
df = pd.read_csv('out_gpt2.csv')

# Elimina la colonna specificata
df = df.drop('Output GPT-2 XL', axis=1)

# Salva il CSV senza la colonna
df.to_csv('out_gpt2.csv', index=False)