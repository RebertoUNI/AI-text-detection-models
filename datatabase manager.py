import pandas as pd
from datasets import load_dataset


dataset = load_dataset('srikanthgali/ai-text-detection-pile-cleaned')

df_train = pd.DataFrame(dataset['train'])
print(df_train.head())
print(f"Colonne: {df_train.columns.tolist()}")
print(f"Totale campioni: {sum(len(dataset[s]) for s in dataset):,}")

# Controlla la distribuzione delle classi
label_col = 'label' if 'label' in df_train.columns else 'generated'
print(f"\nDistribuzione '{label_col}':")
print(df_train[label_col].value_counts(normalize=True))