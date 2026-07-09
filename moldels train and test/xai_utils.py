"""
xai_utils.py
─────────────────────────────────────────────────────────────────────────────
Explainable AI: quali parole spingono il classificatore verso "AI" o "Human".

Tre tecniche, scelte in base all'architettura (vedi suggerimenti del
professore):

  - Saliency (Gradiente × Input) per FCNN/PaperCNN: modelli strutturalmente
    semplici (nn.Embedding + pooling/conv), il contributo di ogni parola è
    matematicamente diretto da calcolare via backward pass.
  - Integrated Gradients (via Captum, opzionale — richiede `pip install
    captum`) per FCNN/PaperCNN: versione più robusta della saliency, integra
    il gradiente lungo un cammino dal baseline (embedding nulla/pad) fino
    all'input reale, invece di guardare solo il gradiente locale.
  - Attention Rollout (ultimo layer) per daBERTa/Qwen: quanto il token
    "riassuntivo" (CLS per daBERTa, ultimo token valido per Qwen, che è
    causale) presta attenzione a ciascuna parola della frase.

Tutte le funzioni restituiscono liste di (parola, punteggio), compatibili
con analysis_utils.plot_heatmap_importanza, così il grafico finale è
identico indipendentemente dalla tecnica usata.
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# 1. Saliency (Gradiente × Input) — FCNN / PaperCNN
# ─────────────────────────────────────────────────────────────────────────
def saliency_fcnn_papercnn(model, tokenizer, text, device, max_length=256, max_words=60):
    """
    Calcola il gradiente dello score P(AI) rispetto all'embedding di ogni
    token, poi lo riduce a un punteggio per parola con Gradiente · Embedding
    (Grad × Input), lo standard per le saliency map su reti con embedding.
    """
    parole = text.split()[:max_words]
    if not parole:
        return []
    testo_troncato = " ".join(parole)

    enc = tokenizer([testo_troncato], truncation=True, padding="max_length",
                     max_length=max_length, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    model.zero_grad(set_to_none=True)
    model.eval()

    embedded = model.embedding(input_ids)   # (1, seq, dim) — richiede già grad (embedding.weight lo richiede)
    embedded.retain_grad()

    # Ricostruisce il forward pass A PARTIRE dall'embedding già calcolato,
    # per poter propagare il gradiente fino a `embedded`.
    if hasattr(model, "conv1"):
        # PaperCNN
        x = embedded.transpose(1, 2)
        x = model.pool1(model.dropout1(torch.relu(model.conv1(x))))
        x = model.pool2(model.dropout2(torch.relu(model.conv2(x))))
        x = model.flatten(x)
        x = model.dropout3(torch.relu(model.fc1(x)))
        out = model.sigmoid(model.fc_out(x))
    else:
        # FCNN
        pooled = embedded.mean(dim=1)
        x = model.dropout(model.relu(model.fc1(pooled)))
        out = model.sigmoid(model.fc2(x))

    score = out.squeeze()
    score.backward()

    if embedded.grad is None:
        logger.warning("Nessun gradiente calcolato (embedded.grad è None): restituisco punteggi nulli.")
        return list(zip(parole, np.zeros(len(parole))))

    grad = embedded.grad[0]           # (seq, dim)
    emb_vals = embedded[0].detach()   # (seq, dim)
    saliency_per_token = (grad * emb_vals).sum(dim=-1).cpu().numpy()   # Grad × Input

    n_tok = len(parole)
    return list(zip(parole, saliency_per_token[:n_tok]))


# ─────────────────────────────────────────────────────────────────────────
# 2. Integrated Gradients (Captum, opzionale) — FCNN / PaperCNN
# ─────────────────────────────────────────────────────────────────────────
def integrated_gradients_fcnn_papercnn(model, tokenizer, text, device, max_length=256,
                                        max_words=60, n_steps=32):
    """
    Versione più robusta della saliency semplice: richiede `pip install
    captum`. Integra il gradiente lungo un cammino dall'embedding nulla
    (baseline = token id 0, cioè il padding) fino all'embedding reale, invece
    di guardare solo il gradiente nel punto esatto dell'input.
    """
    try:
        from captum.attr import LayerIntegratedGradients
    except ImportError as e:
        raise ImportError(
            "Integrated Gradients richiede 'captum': installalo con `pip install captum`."
        ) from e

    parole = text.split()[:max_words]
    if not parole:
        return []
    testo_troncato = " ".join(parole)

    enc = tokenizer([testo_troncato], truncation=True, padding="max_length",
                     max_length=max_length, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    model.eval()

    def forward_fn(ids):
        return model(ids)   # (batch, 1) = P(AI)

    lig = LayerIntegratedGradients(forward_fn, model.embedding)
    baseline = torch.zeros_like(input_ids)   # baseline: tutta la sequenza a token id 0 (pad)

    attributions = lig.attribute(input_ids, baselines=baseline, n_steps=n_steps)
    scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()

    n_tok = len(parole)
    return list(zip(parole, scores[:n_tok]))


# ─────────────────────────────────────────────────────────────────────────
# 3. Attention Rollout (ultimo layer) — daBERTa / Qwen
# ─────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def attention_last_layer(model, tokenizer, text, device, is_causal=False,
                          max_length=256, max_words=60):
    """
    Estrae la matrice di attenzione dell'ultimo layer e restituisce, per ogni
    parola, quanta attenzione le ha dedicato il token "riassuntivo":
      - daBERTa (bidirezionale): il token [CLS] (posizione 0)
      - Qwen (causale): l'ultimo token valido della sequenza
    Media sulle attention head.

    NOTA: il modello va caricato con attn_implementation="eager"
    (predict_fn_loaders.load_model_deberta/load_model_qwen lo fanno già),
    altrimenti con SDPA/flash-attention le attention weights non vengono
    restituite (out.attentions sarebbe None).
    """
    parole = text.split()[:max_words]
    if not parole:
        return []
    testo_troncato = " ".join(parole)

    enc = tokenizer([testo_troncato], truncation=True, max_length=max_length,
                     return_tensors="pt").to(device)
    model.eval()
    out = model(**enc, output_attentions=True)

    if out.attentions is None:
        raise RuntimeError(
            "Il modello non ha restituito le attention weights. Ricaricalo con "
            "attn_implementation='eager' (vedi predict_fn_loaders.load_model_deberta/load_model_qwen)."
        )

    last_layer_attn = out.attentions[-1][0]          # (n_head, seq, seq)
    attn_media_heads = last_layer_attn.mean(dim=0)   # (seq, seq)

    seq_len = enc["input_ids"].shape[-1]
    if is_causal:
        pad_id = tokenizer.pad_token_id
        ids = enc["input_ids"][0]
        non_pad = (ids != pad_id).nonzero(as_tuple=True)[0]
        query_pos = int(non_pad[-1]) if len(non_pad) else seq_len - 1
    else:
        query_pos = 0   # [CLS]

    riga_attenzione = attn_media_heads[query_pos].cpu().numpy()   # (seq,)

    # Riallinea sub-token -> parole: somma l'attenzione dei sub-token che
    # compongono ciascuna parola (via ri-tokenizzazione parola per parola,
    # un'approssimazione ragionevole per un tokenizer SentencePiece/BPE).
    scores = []
    offset = 1 if not is_causal else 0   # salta [CLS] per daBERTa
    for parola in parole:
        n_subtok = max(1, len(tokenizer.tokenize(parola)))
        fine = min(offset + n_subtok, len(riga_attenzione))
        peso = float(riga_attenzione[offset:fine].sum()) if offset < len(riga_attenzione) else 0.0
        scores.append(peso)
        offset += n_subtok

    return list(zip(parole, scores))


# ─────────────────────────────────────────────────────────────────────────
# Dispatcher unico: sceglie la tecnica giusta in base al modello
# ─────────────────────────────────────────────────────────────────────────
def spiega_frase(model_name, text, model, tokenizer, device, metodo="auto"):
    """
    metodo:
      - "auto"                 -> saliency per FCNN/PaperCNN, attention per daBERTa/Qwen
      - "saliency"              (solo FCNN/PaperCNN)
      - "integrated_gradients"  (solo FCNN/PaperCNN, richiede captum)
      - "attention"             (solo daBERTa/Qwen)
    """
    if model_name in ("FCNN", "PaperCNN"):
        metodo = "saliency" if metodo == "auto" else metodo
        if metodo == "saliency":
            return saliency_fcnn_papercnn(model, tokenizer, text, device)
        elif metodo == "integrated_gradients":
            return integrated_gradients_fcnn_papercnn(model, tokenizer, text, device)
        raise ValueError(f"Metodo '{metodo}' non supportato per {model_name}")

    elif model_name in ("daBERTa", "Qwen"):
        metodo = "attention" if metodo == "auto" else metodo
        if metodo == "attention":
            is_causal = (model_name == "Qwen")
            return attention_last_layer(model, tokenizer, text, device, is_causal=is_causal)
        raise ValueError(f"Metodo '{metodo}' non supportato per {model_name} (usa 'attention')")

    raise ValueError(f"Modello sconosciuto: {model_name}")
