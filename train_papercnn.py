"""
train_papercnn.py
─────────────────────────────────────────────────────────────────────────────
Script standalone per l'addestramento del modello PaperCNN (architettura
CLEF 2019) su tutto lo split di training del dataset (nessun
sotto-campionamento a 20000).

Uso tipico su HPC (es. dentro uno script SLURM):

    python train_papercnn.py --epochs 10 --batch-size 128 --num-workers 8

Output prodotti (nella working directory, accanto allo script):
    checkpoint PaperCNN/
        checkpoint_last.pt      <- per riprendere il training se il job viene interrotto
        checkpoint_best.pt      <- pesi con la miglior val accuracy
        train.log
    embeddings PaperCNN/
        train_shard_000.npz, train_shard_001.npz, ...
        validation_shard_000.npz, ...
        test_shard_000.npz, ...
        {split}_meta.json

NOTA sul LazyLinear: come nel notebook originale, il layer fc1 è un
nn.LazyLinear che calcola la dimensione di input al primo forward pass. Per
questo motivo il modello viene "inizializzato" con un batch di prova prima
di costruire l'optimizer / caricare un checkpoint.
"""

import argparse
import logging
import os

import torch
import torch.nn as nn

from data_utils import get_dataloaders, get_full_split_loader
from train_utils import train_one_model, extract_and_save_embeddings, setup_logging, set_seed

MODEL_NAME = "PaperCNN"
CHECKPOINT_DIR = f"checkpoint {MODEL_NAME}"
EMBEDDINGS_DIR = f"embeddings {MODEL_NAME}"


class PaperCNN(nn.Module):
    """Stessa architettura del notebook originale (Cella 7)."""

    def __init__(self, vocab_size, embedding_dim=300):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.conv1    = nn.Conv1d(embedding_dim, 64, kernel_size=4)
        self.dropout1 = nn.Dropout(0.5)
        self.pool1    = nn.MaxPool1d(4)

        self.conv2    = nn.Conv1d(64, 64, kernel_size=4)
        self.dropout2 = nn.Dropout(0.5)
        self.pool2    = nn.MaxPool1d(4)

        self.flatten  = nn.Flatten()
        self.fc1      = nn.LazyLinear(256)
        self.dropout3 = nn.Dropout(0.5)
        self.fc_out   = nn.Linear(256, 1)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, text, return_embedding: bool = False):
        x = self.embedding(text)            # (batch, seq, emb)
        x = x.transpose(1, 2)               # (batch, emb, seq) per Conv1d

        x = self.pool1(self.dropout1(torch.relu(self.conv1(x))))
        x = self.pool2(self.dropout2(torch.relu(self.conv2(x))))

        x = self.flatten(x)
        x = self.dropout3(torch.relu(self.fc1(x)))   # (batch, 256) <- "embedding di frase"

        if return_embedding:
            return x

        return self.sigmoid(self.fc_out(x))


def parse_args():
    p = argparse.ArgumentParser(description="Training PaperCNN su HPC")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--embedding-dim", type=int, default=300)
    p.add_argument("--no-resume", action="store_true",
                    help="Ignora eventuali checkpoint esistenti e riparte da zero")
    p.add_argument("--skip-training", action="store_true",
                    help="Salta il training e usa solo checkpoint_best.pt per estrarre gli embeddings")
    p.add_argument("--skip-embeddings", action="store_true",
                    help="Non calcolare gli embeddings dopo il training")
    p.add_argument("--embedding-shard-size", type=int, default=50_000)
    p.add_argument("--cache-dir", type=str, default="./tokenized_dataset")
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
    logger.info(f"Device: {device}")

    # ── Dati (intero train set, non più il subset da 20000) ────────────────
    train_loader, val_loader, test_loader, vocab_size = get_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers, cache_dir=args.cache_dir
    )
    logger.info(f"Vocab size: {vocab_size}")

    # ── Modello ──────────────────────────────────────────────────────────
    model = PaperCNN(vocab_size=vocab_size, embedding_dim=args.embedding_dim)

    # Il LazyLinear (fc1) deve vedere un batch reale prima di poter essere
    # usato da un optimizer o da un caricamento di checkpoint.
    dummy_batch = next(iter(train_loader))["input_ids"][:2]
    _ = model(dummy_batch)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parametri addestrabili PaperCNN: {n_params:,}")

    # ── Training ─────────────────────────────────────────────────────────
    if not args.skip_training:
        model = train_one_model(
            model, train_loader, val_loader, device,
            checkpoint_dir=CHECKPOINT_DIR,
            epochs=args.epochs, lr=args.lr,
            resume=not args.no_resume,
        )
    else:
        best_path = os.path.join(CHECKPOINT_DIR, "checkpoint_best.pt")
        logger.info(f"--skip-training attivo: carico i pesi da {best_path}")
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)

    # ── Estrazione embeddings su tutti gli split (intero dataset) ─────────
    if not args.skip_embeddings:
        best_path = os.path.join(CHECKPOINT_DIR, "checkpoint_best.pt")
        if os.path.exists(best_path):
            logger.info(f"Carico il best model da {best_path} per l'estrazione degli embeddings")
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)

        for split, split_key in [("train", "train"), ("validation", "validation"), ("test", "test")]:
            logger.info(f"Estrazione embeddings per lo split '{split}'...")
            loader = get_full_split_loader(
                split_key, batch_size=args.batch_size, num_workers=args.num_workers,
                cache_dir=args.cache_dir, shuffle=False
            )
            extract_and_save_embeddings(
                model, loader, split_name=split, device=device,
                output_dir=EMBEDDINGS_DIR, shard_size=args.embedding_shard_size
            )

    logger.info("Fatto.")


if __name__ == "__main__":
    main()
