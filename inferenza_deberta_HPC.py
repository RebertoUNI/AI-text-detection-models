import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from datasets import load_dataset
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description="Inference ottimizzata su V100 HPC")
    parser.add_argument("--batch_size", type=int, default=256, help="Dimensione del batch (ottimale per V100: 128-256)")
    parser.add_argument("--output_file", type=str, default="risultati_hpc.csv", help="Nome del file di output")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Configurazione Device (Singola GPU V100)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo in uso: {device}")
    if torch.cuda.is_available():
        print(f"GPU Rilevata: {torch.cuda.get_device_name(0)}")
        # Ottimizzazione PyTorch per allocazione memoria interna
        torch.backends.cudnn.benchmark = True

    # 2. Caricamento Modello e Tokenizer
    print("Caricamento modello e tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("srikanthgali/paradetect-deberta-v3-lora")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-large", 
        num_labels=2
    )
    model = PeftModel.from_pretrained(base_model, "srikanthgali/paradetect-deberta-v3-lora")
    
    # IMPORTANTE: Portiamo il modello in FP16 per sfruttare i Tensor Cores della V100
    model = model.half().to(device)
    model.eval()

    # 3. Caricamento Dataset tramite Hugging Face
    print("Caricamento dataset...")
    dataset = load_dataset("srikanthgali/ai-text-detection-pile-cleaned", split="test")
    
    # 4. DataLoader ottimizzato per HPC
    # Usiamo il DataLoader nativo di PyTorch per gestire il multi-processing della CPU
    def collate_fn(batch):
        # Estraiamo i testi e le etichette dal batch del dataset
        texts = [item["text"] for item in batch]
        generated = [item["generated"] for item in batch]
        sources = [item["source"] for item in batch]
        return texts, generated, sources

    # Calcoliamo quanti core CPU assegnare al caricamento dati (ottimale in HPC)
    num_workers = min(4, os.cpu_count() or 1)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,  # Velocizza il trasferimento RAM -> VRAM
        collate_fn=collate_fn
    )

    # 5. Loop di Inferenza con FP16 (Autocast)
    results = []
    print(f"Inizio inferenza (Batch Size: {args.batch_size}, Workers: {num_workers})...")
    
    with torch.no_grad():
        for texts, labels, sources in tqdm(dataloader, desc="Inference Progress"):
            
            # Tokenizzazione dinamica per ogni batch
            inputs = tokenizer(
                texts, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512, 
                padding=True
            ).to(device)
            
            # Attiviamo l'autocast FP16 per i Tensor Cores
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predictions = torch.argmax(probabilities, dim=-1)
            
            # Spostiamo i risultati su CPU in blocco
            probabilities = probabilities.cpu()
            predictions = predictions.cpu()
            
            # Salviamo i record
            for j in range(len(texts)):
                human_prob = probabilities[j][0].item()
                ai_prob = probabilities[j][1].item()
                
                results.append({
                    "text": texts[j],
                    "ground_truth_num": labels[j],
                    "ground_truth_str": sources[j],
                    "label_prodotta": predictions[j].item(),
                    "confidence": max(human_prob, ai_prob),
                    "ai_probability": ai_prob
                })

    # 6. Salvataggio dei Risultati
    print(f"Salvataggio in corso su {args.output_file}...")
    df_results = pd.DataFrame(results)
    df_results.to_csv(args.output_file, index=False, encoding="utf-8")
    print("Processo completato con successo!")

if __name__ == "__main__":
    main()