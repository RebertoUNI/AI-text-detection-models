"""
eval_utils.py
─────────────────────────────────────────────────────────────────────────────
Funzioni condivise dai 4 script di test (test_fcnn.py, test_papercnn.py,
test_deberta.py, test_qwen.py). Vengono usate IDENTICHE per tutti i modelli,
in modo che il confronto finale sia oggettivo: stesse metriche, calcolate
nello stesso modo, sullo stesso split di test, con la stessa soglia di
decisione.
"""

import json
import logging
import os

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = "results"   # cartella comune a tutti i modelli, per il confronto finale


def compute_metrics_from_probs(y_true, y_prob, threshold: float = 0.5):
    """
    y_true : array di label vere (0/1)
    y_prob : array di probabilità stimate della classe 1 (AI-generated)

    Calcola le stesse metriche per qualunque modello, così il confronto tra
    FCNN / PaperCNN / DeBERTa / Qwen è fatto a parità di criterio.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()  # [[TN, FP], [FN, TP]]

    metrics = {
        "n_samples":  int(len(y_true)),
        "threshold":  threshold,
        "accuracy":   float(accuracy_score(y_true, y_pred)),
        "precision":  float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":     float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":         float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm,
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        # capita solo se y_true ha una sola classe (non dovrebbe con un test set bilanciato)
        metrics["roc_auc"] = None

    return metrics


def save_test_results(model_name: str, metrics: dict, y_true, y_prob, output_dir: str = RESULTS_DIR):
    """
    Salva:
      results/{model_name}_test_metrics.json       <- metriche aggregate
      results/{model_name}_test_predictions.npz    <- predizioni grezze (per ROC curve, ecc.)
    """
    os.makedirs(output_dir, exist_ok=True)

    metrics_path = os.path.join(output_dir, f"{model_name}_test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"model": model_name, **metrics}, f, indent=2)

    preds_path = os.path.join(output_dir, f"{model_name}_test_predictions.npz")
    np.savez_compressed(preds_path, y_true=np.asarray(y_true), y_prob=np.asarray(y_prob))

    logger.info(f"[{model_name}] accuracy={metrics['accuracy']:.4f} "
                f"f1={metrics['f1']:.4f} roc_auc={metrics['roc_auc']}")
    logger.info(f"Salvati: {metrics_path} , {preds_path}")
    return metrics_path, preds_path
