import os
import csv
import logging
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
from tqdm import tqdm

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Inference paradetect-deberta-v3-lora con Incertezza")
    parser.add_argument("--model_id",   default="srikanthgali/paradetect-deberta-v3-lora")
    parser.add_argument("--csv_path",   default="deberta_push_to_error/out_gpt2_3.csv", help="Percorso del tuo file CSV")
    parser.add_argument("--label_col",  default="tone", help="Colonna da usare come ground_truth (es. tone o category ID)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output",     default="results_with_uncertainty_1.csv")
    parser.add_argument("--fp16",       action="store_true", default=True)
    parser.add_argument("--no_fp16",    dest="fp16", action="store_false")
    return parser.parse_args()


def make_collate_fn(tokenizer, max_length, label_col):
    def collate_fn(batch):
        texts  = [str(item["text_in"]) if item["text_in"] is not None else "" for item in batch]
        # Se la colonna label non esiste o è vuota, mettiamo un valore fittizio "N/A"
        labels = [item.get(label_col, "N/A") for item in batch]
        
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return enc, labels, texts
    return collate_fn


def main():
    args = parse_args()

    # ── Device ──────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if device.type == "cuda":
        log.info("GPU: %s  |  VRAM: %.1f GB",
                 torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9)
    use_fp16 = args.fp16 and device.type == "cuda"

    # ── Carica configurazione LoRA e Tokenizer ──
    log.info("Caricamento configurazione LoRA...")
    peft_config = PeftConfig.from_pretrained(args.model_id)
    base_model_id = peft_config.base_model_name_or_path
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    # ── Modello ─────────────────────────────────
    log.info("Caricamento modello base (%s)...", base_model_id)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16 if use_fp16 else torch.float32,
    )
    model = PeftModel.from_pretrained(base_model, args.model_id)
    model = model.to(device)
    model.eval()

    id2label = model.config.id2label if hasattr(model.config, "id2label") else None
    log.info("Mapping id→label: %s", id2label)

    # ── Caricamento Dataset CSV ─────────────────
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"Non ho trovato il file CSV in: {args.csv_path}")
        
    log.info("Caricamento dataset da: %s", args.csv_path)
    dataset = Dataset.from_csv(args.csv_path)
    
    # Controllo presenza colonna text
    if "text" not in dataset.column_names:
        raise KeyError("Nel CSV deve essere presente la colonna 'text'.")

    # Filtro righe vuote
    n_before = len(dataset)
    dataset = dataset.filter(lambda x: x["text"] is not None and str(x["text"]).strip() != "")
    log.info("Campioni validi: %d/%d", len(dataset), n_before)

    # ── DataLoader ───────────────────────────────
    collate_fn = make_collate_fn(tokenizer, args.max_length, args.label_col)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    # ── Inferenza con Calcolo Incertezza ─────────
    out_path = Path(args.output)
    rows_written = 0

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Aggiunta la colonna 'confidence (%)'
        writer.writerow(["text", "ground_truth", "predicted_label", "confidence (%)"])

        with torch.inference_mode():
            for batch_idx, (enc, labels, texts) in enumerate(
                tqdm(loader, desc="Inferenza in corso", unit="batch")
            ):
                enc = {k: v.to(device) for k, v in enc.items()}

                if use_fp16:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(**enc).logits
                else:
                    logits = model(**enc).logits

                # Applichiamo Softmax per trasformare i logit in probabilità (0-1)
                probs = F.softmax(logits, dim=-1)
                
                # Estraiamo la probabilità massima (confidenza) e l'indice della classe predetta
                confidences, preds = torch.max(probs, dim=-1)

                preds = preds.cpu().tolist()
                confidences = confidences.cpu().tolist()

                for text, gt, pred_id, conf in zip(texts, labels, preds, confidences):
                    pred_label = id2label[pred_id] if id2label else pred_id
                    
                    # Gestione della label reale (se intero mappa, altrimenti stringa/testo)
                    gt_label = (id2label[gt] if id2label and isinstance(gt, int) else gt)
                    
                    # Formattiamo la confidenza come percentuale (es. 94.5%)
                    conf_percentage = f"{conf * 100:.2f}%"
                    
                    writer.writerow([text, gt_label, pred_label, conf_percentage])
                    rows_written += 1

                if (batch_idx + 1) % 50 == 0:
                    log.info("Batch %d/%d — Righe elaborate: %d", batch_idx + 1, len(loader), rows_written)

    log.info("✓ Completato. Risultati salvati in '%s' (%d righe)", out_path, rows_written)


if __name__ == "__main__":
    main()