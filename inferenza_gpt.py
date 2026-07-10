import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd

# 1. Configurazione del dispositivo (GPU se disponibile, altrimenti CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Uso il dispositivo: {device}")

MODEL = "openai-community/gpt2-xl"

# 2. Caricamento di Tokenizer e Modello
tokenizer = GPT2Tokenizer.from_pretrained(MODEL)
model = GPT2LMHeadModel.from_pretrained(MODEL)

# --- CRUCIALE PER IL BATCHING ---
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" 

# Spostiamo il modello sulla GPU e attiviamo la modalità valutazione
model.to(device)
model.eval()

# 3. Configurazione File I/O e Batch Size
input_csv = "gpt2_prompts_2.csv"      # Metti qui il nome del tuo file CSV d'ingresso
output_csv = "out_gpt2_2.csv"  # Il file CSV che verrà creato
BATCH_SIZE = 4                      # Puoi aumentare o diminuire in base alla memoria della tua GPU

# Carichiamo i dati dal CSV
print(f"Caricamento dati da {input_csv}...")
df = pd.read_csv(input_csv)

# Lista temporanea in cui salveremo tutti i testi generati
all_generated_texts = []

# 4. Elaborazione e Generazione in Batch
print("Inizio generazione in corso...")
for i in range(0, len(df), BATCH_SIZE):
    # Estraiamo il batch corrente di righe
    batch_df = df.iloc[i : i + BATCH_SIZE]
    prompts = batch_df["Prompt for GPT-2 XL"].tolist()
    
    # Tokenizzazione in Batch
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            top_k=40,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decodifica dei risultati del batch
    for j, out in enumerate(outputs):
        # OPZIONE A: Se vuoi il testo COMPLETO (Prompt + Continuazione), usa questa riga:
        generated_text = tokenizer.decode(out, skip_special_tokens=True)
        
        # OPZIONE B: Se vuoi SOLO la continuazione eliminando il prompt iniziale, 
        # scommenta la riga qui sotto e commenta quella sopra:
        # generated_text = tokenizer.decode(out[inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        all_generated_texts.append(generated_text)
        
    print(f"Processati {min(i + BATCH_SIZE, len(df))}/{len(df)} prompt...")

# 5. Sostituzione della colonna e Salvataggio
# Sostituiamo il vecchio testo dei prompt con il testo generato
df["Prompt for GPT-2 XL"] = all_generated_texts

# Rinominiamo la colonna per correttezza, mantenendo lo stesso identico formato del CSV
df = df.rename(columns={"Prompt for GPT-2 XL": "Output GPT-2 XL"})

# Salvataggio su file CSV (senza l'indice di pandas per mantenere il formato pulito)
df.to_csv(output_csv, index=False, encoding="utf-8")

print(f"\nSalvataggio completato! Trovi il file modificato in: '{output_csv}'")