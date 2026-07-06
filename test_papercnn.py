"""
test_papercnn.py
─────────────────────────────────────────────────────────────────────────────
Valuta il best checkpoint di PaperCNN sull'INTERO test set, con lo stesso
protocollo di valutazione usato per gli altri 3 modelli (vedi eval_utils.py),
per un confronto oggettivo.

Uso:
    python test_papercnn.py

Richiede che 'checkpoint PaperCNN/checkpoint_best.pt' esista già (creato da
train_papercnn.py).

Output:
    results/PaperCNN_test_metrics.json
    results/PaperCNN_test_predictions.npz
"""

import argparse
import logging
import os

import numpy as np
import torch

from data_utils import get_dataloaders, get_full_split_loader
from eval_utils import compute_metrics_from_probs, save_test_results
from train_utils import setup_logging
from train_papercnn import PaperCNN, CHECKPOINT_DIR, MODEL_NAME


def parse_args():
    p = argparse.ArgumentParser(description="Test PaperCNN sull'intero test set")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--embedding-dim", type=int, default=300)
    p.add_argument("--cache-dir", type=str, default="./tokenized_dataset")
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs("results", exist_ok=True)
    setup_logging(os.path.join("results", f"{MODEL_NAME}_test.log"))
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    train_loader, _, _, vocab_size = get_dataloaders(
        batch_size=args.batch_size, num_workers=0, cache_dir=args.cache_dir
    )

    model = PaperCNN(vocab_size=vocab_size, embedding_dim=args.embedding_dim)

    # Inizializza il LazyLinear con un batch di prova prima di caricare i pesi
    dummy_batch = next(iter(train_loader))["input_ids"][:2]
    _ = model(dummy_batch)

    best_path = os.path.join(CHECKPOINT_DIR, "checkpoint_best.pt")
    logger.info(f"Carico i pesi da {best_path}")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    test_loader = get_full_split_loader(
        "test", batch_size=args.batch_size, num_workers=args.num_workers,
        cache_dir=args.cache_dir, shuffle=False
    )

    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["input_ids"].to(device, non_blocking=True)
            y = batch["label"]
            probs = model(x).squeeze(1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.numpy())

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)

    metrics = compute_metrics_from_probs(y_true, y_prob, threshold=args.threshold)
    save_test_results(MODEL_NAME, metrics, y_true, y_prob)
    logger.info("Fatto.")


if __name__ == "__main__":
    main()
