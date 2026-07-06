"""
test_deberta.py
─────────────────────────────────────────────────────────────────────────────
Valuta l'adapter LoRA migliore di DeBERTa-v3-large sull'INTERO test set, con
lo stesso protocollo di valutazione usato per gli altri 3 modelli (vedi
eval_utils.py), per un confronto oggettivo.

Uso:
    python test_deberta.py

Richiede che 'checkpoint daBERTa/lora_adapter_best/' esista già (creato da
train_deberta.py).

Output:
    results/daBERTa_test_metrics.json
    results/daBERTa_test_predictions.npz
"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

from data_utils import get_hf_datasets
from eval_utils import compute_metrics_from_probs, save_test_results
from train_utils import setup_logging
from train_deberta import BASE_MODEL, CHECKPOINT_DIR, MODEL_NAME, TOKENIZED_CACHE_DIR


def parse_args():
    p = argparse.ArgumentParser(description="Test DeBERTa-v3-large + LoRA sull'intero test set")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-proc", type=int, default=4)
    p.add_argument("--cache-dir", type=str, default=TOKENIZED_CACHE_DIR)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--adapter-dir", type=str, default=None,
                    help="Path dell'adapter LoRA da valutare (default: checkpoint daBERTa/lora_adapter_best)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs("results", exist_ok=True)
    setup_logging(os.path.join("results", f"{MODEL_NAME}_test.log"))
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    adapter_dir = args.adapter_dir or os.path.join(CHECKPOINT_DIR, "lora_adapter_best")
    logger.info(f"Carico l'adapter da {adapter_dir}")

    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2, ignore_mismatched_sizes=True
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.to(device)
    model.eval()

    _, _, test_ds = get_hf_datasets(
        tokenizer_name=BASE_MODEL, cache_dir=args.cache_dir, num_proc=args.num_proc,
        columns=("input_ids", "attention_mask", "label"),
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs  = F.softmax(logits, dim=-1)[:, 1]   # probabilità della classe "AI-generated"
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch["label"].numpy())

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)

    metrics = compute_metrics_from_probs(y_true, y_prob, threshold=args.threshold)
    save_test_results(MODEL_NAME, metrics, y_true, y_prob)
    logger.info("Fatto.")


if __name__ == "__main__":
    main()
