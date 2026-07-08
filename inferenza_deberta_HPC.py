"""
Inference script: srikanthgali/paradetect-deberta-v3-lora
Dataset:         srikanthgali/ai-text-detection-pile-cleaned  (split=test)
Output:          results.csv  (text, ground_truth, predicted_label)
GPU:             CUDA (ottimizzato per V100 – FP16 + batch dinamico)
"""

import os
import csv
import logging
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
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


# ──────────────────────────────────────────────
# Argomenti CLI (tutti opzionali, ci sono default)
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Inference paradetect-deberta-v3-lora")
    parser.add_argument("--model_id",   default="srikanthgali/paradetect-deberta-v3-lora")
    parser.add_argument("--dataset_id", default="srikanthgali/ai-text-detection-pile-cleaned")
    parser.add_argument("--split",      default="test")
    parser.add_argument("--text_col",   default="text",  help="Nome colonna testo nel dataset")
    parser.add_argument("--label_col",  default="gerated", help="Nome colonna label nel dataset")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output",     default="results.csv")
    parser.add_argument("--fp16",       action="store_true", default=True,
                        help="Usa FP16 su GPU (default: True)")
    parser.add_argument("--no_fp16",    dest="fp16", action="store_false")
    return parser.parse_args()


# ──────────────────────────────────────────────
# Collate fn per DataLoader
# ──────────────────────────────────────────────
def make_collate_fn(tokenizer, max_length):
    def collate_fn(batch):
        texts  = [str(item["text"]) if item["text"] is not None else "" for item in batch]
        labels = [item["label"] for item in batch]
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return enc, labels, texts
    return collate_fn


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    args = parse_args()

    # ── Device ──────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if device.type == "cuda":
        log.info("GPU: %s  |  VRAM totale: %.1f GB",
                 torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9)
    use_fp16 = args.fp16 and device.type == "cuda"
    log.info("FP16: %s", use_fp16)

    # ── Carica configurazione LoRA per ricavare il base model ───
    log.info("Caricamento configurazione LoRA da %s …", args.model_id)
    peft_config = PeftConfig.from_pretrained(args.model_id)
    base_model_id = peft_config.base_model_name_or_path
    log.info("Base model: %s", base_model_id)

    # ── Tokenizer (dal base model) ───────────────
    log.info("Caricamento tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    # ── Modello base + adattatori LoRA ──────────
    log.info("Caricamento base model …")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16 if use_fp16 else torch.float32,
    )
    log.info("Applicazione adattatori LoRA …")
    model = PeftModel.from_pretrained(base_model, args.model_id)
    model = model.to(device)
    model.eval()
    log.info("Parametri totali: %s  |  trainabili: %s",
             f"{sum(p.numel() for p in model.parameters()):,}",
             f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Mappa id→label
    id2label = model.config.id2label if hasattr(model.config, "id2label") else None
    log.info("Mapping id→label: %s", id2label)

    # ── Dataset ──────────────────────────────────
    log.info("Caricamento dataset %s (split=%s) …", args.dataset_id, args.split)
    dataset = load_dataset(args.dataset_id, split=args.split)
    log.info("Campioni nel test set: %d", len(dataset))

    # Filtra righe senza testo
    n_before = len(dataset)
    dataset = dataset.filter(lambda x: x["text"] is not None and str(x["text"]).strip() != "")
    n_skipped = n_before - len(dataset)
    if n_skipped:
        log.warning("Righe scartate per testo mancante/vuoto: %d", n_skipped)
    log.info("Campioni validi: %d", len(dataset))

    # Rinomina colonne se necessario
    rename_map = {}
    if args.text_col != "text" and args.text_col in dataset.column_names:
        rename_map[args.text_col] = "text"
    if args.label_col != "label" and args.label_col in dataset.column_names:
        rename_map[args.label_col] = "label"
    if rename_map:
        dataset = dataset.rename_columns(rename_map)

    # ── DataLoader ───────────────────────────────
    collate_fn = make_collate_fn(tokenizer, args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )
    log.info("Batch size: %d  |  Batch totali: %d", args.batch_size, len(loader))

    # ── Inferenza ────────────────────────────────
    out_path = Path(args.output)
    rows_written = 0

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "ground_truth", "predicted_label"])

        with torch.inference_mode():
            for batch_idx, (enc, labels, texts) in enumerate(
                tqdm(loader, desc="Inferenza", unit="batch")
            ):
                enc = {k: v.to(device) for k, v in enc.items()}

                if use_fp16:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(**enc).logits
                else:
                    logits = model(**enc).logits

                preds = logits.argmax(dim=-1).cpu().tolist()

                for text, gt, pred_id in zip(texts, labels, preds):
                    pred_label = id2label[pred_id] if id2label else pred_id
                    gt_label   = (id2label[gt] if id2label and isinstance(gt, int)
                                  else gt)
                    writer.writerow([text, gt_label, pred_label])
                    rows_written += 1

                # Log ogni 50 batch
                if (batch_idx + 1) % 50 == 0:
                    log.info("Batch %d/%d — righe scritte: %d",
                             batch_idx + 1, len(loader), rows_written)

    log.info("✓ Completato. Risultati salvati in '%s'  (%d righe)", out_path, rows_written)


if __name__ == "__main__":
    main()