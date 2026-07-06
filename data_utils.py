"""
data_utils.py
─────────────────────────────────────────────────────────────────────────────
Modulo condiviso per il caricamento, la tokenizzazione e la creazione dei
DataLoader del dataset 'srikanthgali/ai-text-detection-pile-cleaned'.

Viene importato da tutti gli script di training (train_fcnn.py,
train_papercnn.py, ...) così la logica di preparazione dati è scritta una
sola volta e resta identica tra i modelli.

La tokenizzazione dell'intero dataset viene fatta una sola volta e salvata su
disco con `save_to_disk`, così se lo stesso nodo HPC lancia più job (uno per
modello) non si ritokenizza da zero ogni volta.
"""

import os
import logging

from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# ── Costanti condivise ────────────────────────────────────────────────────
# MODIFICA: Inserisci il path della cartella in cui hai scaricato il dataset grezzo
RAW_DATASET_PATH = "./ai_text_detection_dataset" 

TOKENIZER_NAME = "microsoft/deberta-v3-large"   # stesso tokenizer usato nel notebook originale
MAX_LENGTH     = 256
DEFAULT_CACHE_DIR = "./tokenized_dataset"       # cartella dove viene cachato il dataset tokenizzato


def get_tokenizer():
    """Restituisce il tokenizer condiviso da FCNN e PaperCNN (embedding layer costruito su questo vocab)."""
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def load_and_tokenize_dataset_generic(tokenizer_name: str, cache_dir: str,
                                      max_length: int = MAX_LENGTH, num_proc: int = 4):
    """
    Versione generica: tokenizza l'intero dataset con QUALSIASI tokenizer
    (usata sia per il vocab custom di FCNN/PaperCNN, sia per i tokenizer di
    DeBERTa e Qwen, che sono diversi tra loro e quindi vanno cachati in
    cartelle separate).
    """
    if os.path.exists(cache_dir):
        logger.info(f"Cache trovata, carico il dataset tokenizzato da: {cache_dir}")
        return load_from_disk(cache_dir)

    # MODIFICA: Il log ora riflette il caricamento locale
    logger.info(f"Nessuna cache trovata in '{cache_dir}': carico il dataset grezzo locale da "
                f"'{RAW_DATASET_PATH}' e lo tokenizzo con '{tokenizer_name}' (può richiedere tempo)...")
    
    # MODIFICA FONDAMENTALE: Usiamo load_from_disk invece di load_dataset
    if not os.path.exists(RAW_DATASET_PATH):
        raise FileNotFoundError(f"Dataset grezzo non trovato in {RAW_DATASET_PATH}. Esegui prima lo script di download!")
        
    dataset = load_from_disk(RAW_DATASET_PATH)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        texts = [t if isinstance(t, str) else "" for t in examples["text"]]
        return tokenizer(texts, truncation=True, padding="max_length", max_length=max_length)

    tokenized = dataset.map(tokenize_function, batched=True, num_proc=num_proc)

    # Uniforma il nome della colonna target
    if "generated" in tokenized["train"].column_names and "label" not in tokenized["train"].column_names:
        tokenized = tokenized.rename_column("generated", "label")

    os.makedirs(os.path.dirname(cache_dir) or ".", exist_ok=True)
    tokenized.save_to_disk(cache_dir)
    logger.info(f"Dataset tokenizzato salvato in cache: {cache_dir}")
    return tokenized


def load_and_tokenize_dataset(cache_dir: str = DEFAULT_CACHE_DIR, num_proc: int = 4):
    """
    Carica il dataset completo (tutti gli split, tutte le righe) tokenizzato
    con il tokenizer di DeBERTa-v3-large (usato anche da FCNN/PaperCNN per
    costruire il proprio vocabolario). Se esiste già una cache su disco, la
    riusa invece di ritokenizzare.
    """
    return load_and_tokenize_dataset_generic(TOKENIZER_NAME, cache_dir, MAX_LENGTH, num_proc)


def get_hf_datasets(tokenizer_name: str, cache_dir: str, max_length: int = MAX_LENGTH,
                    num_proc: int = 4, columns=("input_ids", "attention_mask", "label")):
    """
    Restituisce (train_ds, val_ds, test_ds) come oggetti datasets.Dataset
    formattati in torch, pronti per essere passati direttamente a un
    HuggingFace Trainer (usato dagli script train_deberta.py / train_qwen.py).
    Nessun sotto-campionamento: vengono restituiti gli split completi.
    """
    tokenized = load_and_tokenize_dataset_generic(tokenizer_name, cache_dir, max_length, num_proc)
    train_ds = tokenized["train"]
    val_ds   = tokenized["validation"]
    test_ds  = tokenized["test"]

    cols = list(columns)
    train_ds.set_format(type="torch", columns=cols)
    val_ds.set_format(type="torch", columns=cols)
    test_ds.set_format(type="torch", columns=cols)

    logger.info(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")
    return train_ds, val_ds, test_ds


def get_dataloaders(batch_size: int = 64, num_workers: int = 4, cache_dir: str = DEFAULT_CACHE_DIR):
    """
    Restituisce (train_loader, val_loader, test_loader, vocab_size) usando
    l'INTERO split di training (nessun campionamento a 20000 righe).
    """
    tokenized = load_and_tokenize_dataset(cache_dir=cache_dir)

    train_ds = tokenized["train"]
    val_ds   = tokenized["validation"]
    test_ds  = tokenized["test"]

    cols = ["input_ids", "label"]
    train_ds.set_format(type="torch", columns=cols)
    val_ds.set_format(type="torch", columns=cols)
    test_ds.set_format(type="torch", columns=cols)

    logger.info(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    vocab_size = get_tokenizer().vocab_size
    return train_loader, val_loader, test_loader, vocab_size


def get_full_split_loader(split: str, batch_size: int = 64, num_workers: int = 4,
                          cache_dir: str = DEFAULT_CACHE_DIR, shuffle: bool = False):
    """
    Utile per l'estrazione degli embedding: restituisce il DataLoader di un
    singolo split ('train', 'validation' o 'test') senza sotto-campionamento.
    """
    tokenized = load_and_tokenize_dataset(cache_dir=cache_dir)
    ds = tokenized[split]
    ds.set_format(type="torch", columns=["input_ids", "label"])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)
