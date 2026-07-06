"""
train_qwen.py
─────────────────────────────────────────────────────────────────────────────
Script standalone per il fine-tuning LoRA (con base quantizzata 4bit) di
'Qwen/Qwen3-0.6B-Base' su tutto lo split di training del dataset (nessun
sotto-campionamento a 20000 / 5000 come nel notebook).

Uso tipico su HPC (es. dentro uno script SLURM):

    python train_qwen.py --epochs 3 --batch-size 4 --grad-accum 4

Il resume è gestito automaticamente dal Trainer di HuggingFace tramite
`get_last_checkpoint`: se il job viene interrotto/prerilasciato, rilanciando
lo stesso comando riprende dall'ultimo checkpoint salvato in
'checkpoint Qwen/'.

Output prodotti (nella working directory, accanto allo script):
    checkpoint Qwen/
        checkpoint-<step>/      <- checkpoint HF Trainer standard (usati per il resume)
        lora_adapter_best/      <- adapter LoRA del miglior modello (per f1 su validation)
        train.log
    embeddings Qwen/
        train_shard_000.npz, ...
        validation_shard_000.npz, ...
        test_shard_000.npz, ...
        {split}_meta.json

NOTA: richiede una GPU (bitsandbytes 4bit non funziona su CPU).
"""

import argparse
import logging
import os

import numpy as np
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from data_utils import get_hf_datasets
from train_utils import extract_and_save_embeddings_hf, setup_logging, set_seed

MODEL_NAME  = "Qwen"
BASE_MODEL  = "Qwen/Qwen3-0.6B-Base"
CHECKPOINT_DIR = f"checkpoint {MODEL_NAME}"
EMBEDDINGS_DIR = f"embeddings {MODEL_NAME}"
TOKENIZED_CACHE_DIR = "./tokenized_dataset_qwen"   # cache separata: tokenizer diverso da DeBERTa
MAX_LENGTH = 256


def get_qwen_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def build_lora_model(pad_token_id):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        ignore_mismatched_sizes=True,
        quantization_config=bnb_config,
        device_map={"": 0},
    )
    model.config.pad_token_id = pad_token_id

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    return get_peft_model(model, lora_config)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary"),
    }


@torch.no_grad()
def qwen_embed_fn(model, device, pad_token_id):
    """
    Ritorna una funzione batch -> embedding: per un modello causale (Qwen)
    la rappresentazione della frase è l'hidden state dell'ULTIMO token reale
    (non di padding), la stessa posizione usata internamente dalla testa di
    classificazione (stessa logica di Qwen2ForSequenceClassification).
    """
    def _fn(batch):
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden = out.hidden_states[-1]   # (batch, seq, hidden)

        batch_size = input_ids.shape[0]
        seq_len    = input_ids.shape[-1]
        seq_lengths = torch.eq(input_ids, pad_token_id).int().argmax(-1) - 1
        seq_lengths = seq_lengths % seq_len   # gestisce anche il caso "nessun padding"

        embedding = last_hidden[torch.arange(batch_size, device=device), seq_lengths]
        return embedding.cpu().numpy()
    return _fn


def parse_args():
    p = argparse.ArgumentParser(description="Training Qwen3-0.6B-Base + LoRA (4bit) su HPC")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument("--num-proc", type=int, default=1)
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("--skip-embeddings", action="store_true")
    p.add_argument("--embedding-batch-size", type=int, default=16)
    p.add_argument("--embedding-shard-size", type=int, default=50_000)
    p.add_argument("--cache-dir", type=str, default=TOKENIZED_CACHE_DIR)
    p.add_argument("--seed", type=int, default=42,
                    help="Seed per riproducibilità (usato uguale su tutti i modelli per un confronto equo)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    setup_logging(os.path.join(CHECKPOINT_DIR, "train.log"))
    logger = logging.getLogger(__name__)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("Nessuna GPU trovata: bitsandbytes 4bit richiede CUDA, lo script probabilmente fallirà.")
    logger.info(f"Device: {device}")

    tokenizer = get_qwen_tokenizer()

    # ── Dati (interi split, nessun sotto-campionamento) ────────────────────
    train_ds, val_ds, test_ds = get_hf_datasets(
        tokenizer_name=BASE_MODEL, cache_dir=args.cache_dir, max_length=MAX_LENGTH,
        num_proc=args.num_proc, columns=("input_ids", "attention_mask", "label"),
    )

    # ── Modello + LoRA (4bit) ────────────────────────────────────────────────
    model = build_lora_model(pad_token_id=tokenizer.pad_token_id)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Parametri addestrabili: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    logger.info(f"bf16: {use_bf16} | fp16: {use_fp16}")

    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        seed=args.seed,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_steps=100,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    if not args.skip_training:
        last_ckpt = get_last_checkpoint(CHECKPOINT_DIR) if os.path.isdir(CHECKPOINT_DIR) else None
        if last_ckpt:
            logger.info(f"Checkpoint trovato: riprendo il training da {last_ckpt}")
        trainer.train(resume_from_checkpoint=last_ckpt)

        best_adapter_dir = os.path.join(CHECKPOINT_DIR, "lora_adapter_best")
        trainer.model.save_pretrained(best_adapter_dir)
        tokenizer.save_pretrained(best_adapter_dir)
        logger.info(f"Adapter LoRA (best model, per f1) salvato in {best_adapter_dir}")
    else:
        best_adapter_dir = os.path.join(CHECKPOINT_DIR, "lora_adapter_best")
        logger.info(f"--skip-training attivo: carico l'adapter da {best_adapter_dir}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
        )
        base_model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL, num_labels=2, ignore_mismatched_sizes=True,
            quantization_config=bnb_config, device_map={"": 0},
        )
        base_model.config.pad_token_id = tokenizer.pad_token_id
        model = PeftModel.from_pretrained(base_model, best_adapter_dir)

    # ── Estrazione embeddings su tutti gli split (intero dataset) ─────────
    if not args.skip_embeddings:
        model.eval()
        embed_fn = qwen_embed_fn(model, device, pad_token_id=tokenizer.pad_token_id)

        for split_name, ds in [("train", train_ds), ("validation", val_ds), ("test", test_ds)]:
            logger.info(f"Estrazione embeddings per lo split '{split_name}'...")
            loader = DataLoader(ds, batch_size=args.embedding_batch_size, shuffle=False)
            extract_and_save_embeddings_hf(
                embed_fn, loader, split_name=split_name,
                output_dir=EMBEDDINGS_DIR, shard_size=args.embedding_shard_size
            )

    logger.info("Fatto.")


if __name__ == "__main__":
    main()
