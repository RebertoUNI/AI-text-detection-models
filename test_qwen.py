"""
test_qwen.py
─────────────────────────────────────────────────────────────────────────────
Valuta l'adapter LoRA migliore di Qwen3-0.6B-Base sull'INTERO test set, con
lo stesso protocollo di valutazione usato per gli altri 3 modelli (vedi
eval_utils.py), per un confronto oggettivo.

Uso:
    python test_qwen.py

Richiede che 'checkpoint Qwen/lora_adapter_best/' esista già (creato da
train_qwen.py). Richiede una GPU (bitsandbytes 4bit non funziona su CPU).

Output:
    results/Qwen_test_metrics.json
    results/Qwen_test_predictions.npz
"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

from data_utils import get_hf_datasets
from eval_utils import compute_metrics_from_probs, save_test_results
from train_utils import setup_logging
from train_qwen import BASE_MODEL, CHECKPOINT_DIR, MODEL_NAME, TOKENIZED_CACHE_DIR, MAX_LENGTH


def parse_args():
    p = argparse.ArgumentParser(description="Test Qwen3-0.6B-Base + LoRA (4bit) sull'intero test set")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-proc", type=int, default=1)
    p.add_argument("--cache-dir", type=str, default=TOKENIZED_CACHE_DIR)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--adapter-dir", type=str, default=None,
                    help="Path dell'adapter LoRA da valutare (default: checkpoint Qwen/lora_adapter_best)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs("results", exist_ok=True)
    setup_logging(os.path.join("results", f"{MODEL_NAME}_test.log"))
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("Nessuna GPU trovata: bitsandbytes 4bit richiede CUDA, lo script probabilmente fallirà.")
    logger.info(f"Device: {device}")

    adapter_dir = args.adapter_dir or os.path.join(CHECKPOINT_DIR, "lora_adapter_best")
    logger.info(f"Carico l'adapter da {adapter_dir}")

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2, ignore_mismatched_sizes=True,
        quantization_config=bnb_config, device_map={"": 0},
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    _, _, test_ds = get_hf_datasets(
        tokenizer_name=BASE_MODEL, cache_dir=args.cache_dir, max_length=MAX_LENGTH,
        num_proc=args.num_proc, columns=("input_ids", "attention_mask", "label"),
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
