"""
train_utils.py
─────────────────────────────────────────────────────────────────────────────
Funzioni condivise da tutti gli script di training per HPC:

  - train_one_model(...)      loop di training con checkpoint ad ogni epoca
                               e possibilità di resume se il job HPC viene
                               interrotto/prerilasciato prima della fine.
  - extract_and_save_embeddings(...)
                               calcola l'embedding di ogni frase (penultimo
                               layer del modello, prima del classificatore
                               finale) e lo salva su disco a "shard" per non
                               saturare la RAM quando si processa l'intero
                               dataset.
"""

import os
import json
import logging
import time

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# TRAINING con checkpoint + resume
# ─────────────────────────────────────────────────────────────────────────
def save_checkpoint(path, model, optimizer, scheduler, epoch, best_acc):
    torch.save({
        "epoch":              epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_acc":             best_acc,
    }, path)


def load_checkpoint_if_exists(path, model, optimizer, scheduler, device):
    """Se esiste un checkpoint 'last', lo ricarica e restituisce (start_epoch, best_acc)."""
    if not os.path.exists(path):
        return 1, 0.0

    logger.info(f"Checkpoint trovato in '{path}', riprendo il training da lì.")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    best_acc = ckpt["best_acc"]
    logger.info(f"Riparto dall'epoca {start_epoch} (best_acc finora: {best_acc:.4f})")
    return start_epoch, best_acc


def train_one_model(model, train_loader, val_loader, device, checkpoint_dir,
                     epochs=10, lr=1e-3, resume=True, log_every=200):
    """
    Loop di training generico per FCNN / PaperCNN.

    Ad ogni epoca:
      - salva SEMPRE 'checkpoint_last.pt' (per il resume)
      - salva 'checkpoint_best.pt' quando la val accuracy migliora
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    last_ckpt_path = os.path.join(checkpoint_dir, "checkpoint_last.pt")
    best_ckpt_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")

    model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )

    start_epoch, best_acc = (1, 0.0)
    if resume:
        start_epoch, best_acc = load_checkpoint_if_exists(
            last_ckpt_path, model, optimizer, scheduler, device
        )

    if start_epoch > epochs:
        logger.info("Il checkpoint salvato ha già completato tutte le epoche richieste. Skip training.")
        return model

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        # ── Train
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for step, batch in enumerate(train_loader):
            x = batch["input_ids"].to(device, non_blocking=True)
            y = batch["label"].float().to(device, non_blocking=True)

            out  = model(x).squeeze(1)
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tr_loss    += loss.item() * len(y)
            tr_correct += ((out > 0.5) == y.bool()).sum().item()
            tr_total   += len(y)

            if step % log_every == 0:
                logger.info(f"Epoch {epoch}/{epochs} - step {step}/{len(train_loader)} "
                            f"- loss corrente: {loss.item():.4f}")

        # ── Validation
        model.eval()
        vl_loss, vl_correct, vl_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["input_ids"].to(device, non_blocking=True)
                y = batch["label"].float().to(device, non_blocking=True)
                out  = model(x).squeeze(1)
                loss = criterion(out, y)
                vl_loss    += loss.item() * len(y)
                vl_correct += ((out > 0.5) == y.bool()).sum().item()
                vl_total   += len(y)

        tr_acc = tr_correct / tr_total
        vl_acc = vl_correct / vl_total
        scheduler.step(vl_acc)

        dt = time.time() - t0
        logger.info(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train loss: {tr_loss/tr_total:.4f} acc: {tr_acc:.4f} | "
            f"Val loss: {vl_loss/vl_total:.4f} acc: {vl_acc:.4f} | "
            f"tempo epoca: {dt/60:.1f} min"
        )

        # Salva sempre l'ultimo checkpoint (per il resume in caso di crash/timeout HPC)
        save_checkpoint(last_ckpt_path, model, optimizer, scheduler, epoch, best_acc)

        # Salva il best separatamente
        if vl_acc > best_acc:
            best_acc = vl_acc
            save_checkpoint(best_ckpt_path, model, optimizer, scheduler, epoch, best_acc)
            logger.info(f"Nuovo best model salvato (val acc: {best_acc:.4f}) in {best_ckpt_path}")

    logger.info(f"Training completato. Best val accuracy: {best_acc:.4f}")
    return model


# ─────────────────────────────────────────────────────────────────────────
# ESTRAZIONE E SALVATAGGIO EMBEDDINGS (a shard, per dataset grandi)
# ─────────────────────────────────────────────────────────────────────────
def extract_and_save_embeddings(model, loader, split_name, device, output_dir,
                                 shard_size=50_000):
    """
    Passa l'intero split nel modello in eval mode e salva l'embedding di ogni
    frase (penultimo layer, richiesto tramite return_embedding=True nel
    forward del modello) su disco a shard, per non saturare la RAM quando lo
    split è molto grande (es. l'intero train set).

    Genera dentro `output_dir`:
      {split_name}_shard_000.npz  (contiene 'embeddings' e 'labels')
      {split_name}_shard_001.npz
      ...
      {split_name}_meta.json      (num shard, num totale campioni, dim embedding)
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    buf_embeddings, buf_labels = [], []
    shard_idx = 0
    total_samples = 0
    embed_dim = None

    def flush():
        nonlocal buf_embeddings, buf_labels, shard_idx, total_samples
        if not buf_embeddings:
            return
        emb = np.concatenate(buf_embeddings, axis=0)
        lab = np.concatenate(buf_labels, axis=0)
        shard_path = os.path.join(output_dir, f"{split_name}_shard_{shard_idx:03d}.npz")
        np.savez_compressed(shard_path, embeddings=emb, labels=lab)
        logger.info(f"Salvato shard {shard_path} ({emb.shape[0]} campioni)")
        total_samples += emb.shape[0]
        shard_idx += 1
        buf_embeddings, buf_labels = [], []

    running_count = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["input_ids"].to(device, non_blocking=True)
            y = batch["label"]

            emb = model(x, return_embedding=True).cpu().numpy()
            if embed_dim is None:
                embed_dim = emb.shape[1]

            buf_embeddings.append(emb)
            buf_labels.append(y.numpy())
            running_count += emb.shape[0]

            if running_count >= shard_size:
                flush()
                running_count = 0

    flush()  # ultimo shard parziale

    meta = {
        "split": split_name,
        "num_shards": shard_idx,
        "total_samples": total_samples,
        "embedding_dim": embed_dim,
    }
    meta_path = os.path.join(output_dir, f"{split_name}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Embeddings '{split_name}' completati: {total_samples} campioni, "
                f"{shard_idx} shard, dim={embed_dim}. Metadata in {meta_path}")


def setup_logging(log_path):
    """Logging su file + stdout, comodo per job HPC non interattivi."""
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
