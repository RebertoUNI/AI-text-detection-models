"""
predict_fn_loaders.py
─────────────────────────────────────────────────────────────────────────────
Costruisce, per ciascuno dei 4 modelli, una funzione

    predict_fn(list_of_str) -> np.ndarray di P(AI)

usata da analysis_utils.occlusion_importance / analizza_vocabolario_globale
(model-agnostic: a loro non importa CHI fa la predizione).

Ogni loader riusa la classe modello / le costanti già definite negli script
di training, per non duplicare l'architettura in due posti diversi.
"""

import os

import numpy as np
import torch


def _pick_device(device=None):
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def build_predict_fn_fcnn(checkpoint_dir=None, device=None, batch_size=64):
    from data_utils import get_tokenizer, MAX_LENGTH
    from train_fcnn import EmbeddingMLP, CHECKPOINT_DIR

    checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR
    device = _pick_device(device)

    tok = get_tokenizer()
    model = EmbeddingMLP(vocab_size=tok.vocab_size)
    ckpt = torch.load(os.path.join(checkpoint_dir, "checkpoint_best.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    @torch.no_grad()
    def predict_fn(texts):
        texts = list(texts)
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            enc = tok(chunk, truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")
            x = enc["input_ids"].to(device)
            p = model(x).squeeze(1).cpu().numpy()
            out.append(p)
        return np.concatenate(out)

    return predict_fn


def build_predict_fn_papercnn(checkpoint_dir=None, device=None, batch_size=64):
    from data_utils import get_tokenizer, MAX_LENGTH
    from train_papercnn import PaperCNN, CHECKPOINT_DIR

    checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR
    device = _pick_device(device)

    tok = get_tokenizer()
    model = PaperCNN(vocab_size=tok.vocab_size)
    # Il LazyLinear interno deve vedere un batch reale prima di poter caricare i pesi
    dummy = torch.zeros((2, MAX_LENGTH), dtype=torch.long)
    _ = model(dummy)

    ckpt = torch.load(os.path.join(checkpoint_dir, "checkpoint_best.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    @torch.no_grad()
    def predict_fn(texts):
        texts = list(texts)
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            enc = tok(chunk, truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")
            x = enc["input_ids"].to(device)
            p = model(x).squeeze(1).cpu().numpy()
            out.append(p)
        return np.concatenate(out)

    return predict_fn


def build_predict_fn_deberta(adapter_dir=None, device=None, batch_size=32, max_length=256):
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from train_deberta import BASE_MODEL, CHECKPOINT_DIR

    adapter_dir = adapter_dir or os.path.join(CHECKPOINT_DIR, "lora_adapter_best")
    device = _pick_device(device)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2, ignore_mismatched_sizes=True
    )
    model = PeftModel.from_pretrained(base, adapter_dir).to(device)
    model.eval()

    @torch.no_grad()
    def predict_fn(texts):
        texts = list(texts)
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            enc = tok(chunk, truncation=True, padding=True, max_length=max_length, return_tensors="pt").to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]
            out.append(probs.cpu().numpy())
        return np.concatenate(out)

    return predict_fn


def build_predict_fn_qwen(adapter_dir=None, device=None, batch_size=16, max_length=256):
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
    from train_qwen import BASE_MODEL, CHECKPOINT_DIR

    adapter_dir = adapter_dir or os.path.join(CHECKPOINT_DIR, "lora_adapter_best")
    device = _pick_device(device)
    if device != "cuda":
        raise RuntimeError("Qwen richiede una GPU (bitsandbytes 4bit non gira su CPU).")

    tok = AutoTokenizer.from_pretrained(adapter_dir)   # train_qwen.py salva il tokenizer nell'adapter dir
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2, ignore_mismatched_sizes=True,
        quantization_config=bnb_config, device_map={"": 0},
    )
    base.config.pad_token_id = tok.pad_token_id
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    @torch.no_grad()
    def predict_fn(texts):
        texts = list(texts)
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            enc = tok(chunk, truncation=True, padding=True, max_length=max_length, return_tensors="pt").to(device)
            logits = model(**enc).logits.float()
            probs = torch.softmax(logits, dim=-1)[:, 1]
            out.append(probs.cpu().numpy())
        return np.concatenate(out)

    return predict_fn


PREDICT_FN_BUILDERS = {
    "FCNN":      build_predict_fn_fcnn,
    "PaperCNN":  build_predict_fn_papercnn,
    "daBERTa":   build_predict_fn_deberta,
    "Qwen":      build_predict_fn_qwen,
}
