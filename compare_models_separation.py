"""
compare_models_separation.py
─────────────────────────────────────────────────────────────────────────────
Legge 'analysis_plots/{MODEL}/separation_metrics.json' prodotti da
analyze_model.py per ciascun modello e produce:

  1. Una tabella di confronto (stampata + CSV + Markdown) con AUC-ROC,
     KS statistic/p-value, Overlap coefficient e Bhattacharyya distance.
  2. Un grafico HTML con le densità P(Human) vs P(AI) di TUTTI i modelli
     affiancate (griglia 2x2), per un confronto visivo diretto.

Diverso da compare_results.py: quello confronta le metriche di
CLASSIFICAZIONE (accuracy/precision/recall/f1) sul test set; questo
confronta quanto bene ogni modello SEPARA le due distribuzioni di
probabilità (utile per capire quale modello è più "sicuro"/discriminante,
non solo quale sbaglia di meno).

Uso (dopo aver lanciato analyze_model.py per ciascun modello):
    python compare_models_separation.py
"""

import argparse
import csv
import json
import logging
import os

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import analysis_utils as au
from train_utils import setup_logging

MODELS = ["FCNN", "PaperCNN", "daBERTa", "Qwen"]

# Chiavi esatte così come salvate da analysis_utils.quantifica_separazione()
METRIC_KEYS = [
    "AUC-ROC",
    "KS statistic",
    "KS p-value",
    "Overlap coefficient (0=separate, 1=identiche)",
    "Bhattacharyya distance (alto=separate)",
]
SHORT_LABELS = {
    "AUC-ROC": "AUC-ROC",
    "KS statistic": "KS stat",
    "KS p-value": "KS p-val",
    "Overlap coefficient (0=separate, 1=identiche)": "Overlap",
    "Bhattacharyya distance (alto=separate)": "Bhattacharyya",
}


def parse_args():
    p = argparse.ArgumentParser(description="Confronta le metriche di separazione dei 4 modelli")
    p.add_argument("--analysis-dir", type=str, default="analysis_plots")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--sort-by", type=str, default="AUC-ROC", choices=METRIC_KEYS)
    p.add_argument("--ascending", action="store_true",
                    help="Ordina crescente invece che decrescente (utile per 'Overlap', dove più basso = meglio)")
    return p.parse_args()


def load_all_metrics(analysis_dir):
    rows = []
    missing = []
    for model in MODELS:
        path = os.path.join(analysis_dir, model, "separation_metrics.json")
        if not os.path.exists(path):
            missing.append(model)
            continue
        with open(path) as f:
            m = json.load(f)
        rows.append(m)
    return rows, missing


def print_and_save_table(rows, sort_by, ascending, analysis_dir):
    rows = sorted(rows, key=lambda r: r.get(sort_by, 0), reverse=not ascending)

    headers = ["model"] + METRIC_KEYS
    short_headers = ["Modello"] + [SHORT_LABELS[k] for k in METRIC_KEYS]

    col_w = 14
    print(" ".join(h.ljust(col_w) for h in short_headers))
    print("-" * (col_w * len(short_headers)))
    for r in rows:
        vals = [str(r.get("model", "?"))] + [f"{r.get(k, float('nan')):.4f}" for k in METRIC_KEYS]
        print(" ".join(v.ljust(col_w) for v in vals))

    os.makedirs(analysis_dir, exist_ok=True)

    csv_path = os.path.join(analysis_dir, "comparison_separation.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow({h: r.get(h) for h in headers})
    print(f"\nTabella salvata in: {csv_path}")

    md_path = os.path.join(analysis_dir, "comparison_separation.md")
    with open(md_path, "w") as f:
        f.write("| " + " | ".join(short_headers) + " |\n")
        f.write("|" + "---|" * len(short_headers) + "\n")
        for r in rows:
            vals = [str(r.get("model", "?"))] + [f"{r.get(k, float('nan')):.4f}" for k in METRIC_KEYS]
            f.write("| " + " | ".join(vals) + " |\n")
    print(f"Tabella Markdown salvata in: {md_path}")

    return rows


def build_combined_density_plot(model_names, results_dir, analysis_dir):
    """Griglia 2x2 con le densità P(Human)/P(AI) di tutti i modelli, per confronto visivo diretto."""
    n = len(model_names)
    ncols = 2
    nrows = (n + 1) // ncols
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=model_names)

    for i, model in enumerate(model_names):
        try:
            y_true, y_prob = au.load_test_predictions(model, results_dir=results_dir)
        except FileNotFoundError as e:
            logging.getLogger(__name__).warning(f"Salto il plot per {model}: {e}")
            continue

        _, grid, kde_h, kde_a = au.quantifica_separazione(y_true, y_prob)
        row, col = (i // ncols) + 1, (i % ncols) + 1

        fig.add_trace(go.Scatter(x=grid, y=kde_h, fill='tozeroy', name='Human',
                                  line=dict(color='royalblue'),
                                  showlegend=(i == 0)), row=row, col=col)
        fig.add_trace(go.Scatter(x=grid, y=kde_a, fill='tozeroy', name='AI',
                                  line=dict(color='indianred'), opacity=0.6,
                                  showlegend=(i == 0)), row=row, col=col)

    fig.update_layout(title="Confronto densità delle predizioni tra i modelli",
                       height=350 * nrows)
    out_path = os.path.join(analysis_dir, "comparison_density_overlay.html")
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"Grafico di confronto densità salvato in: {out_path}")


def main():
    args = parse_args()
    setup_logging(os.path.join(args.analysis_dir, "compare_models_separation.log"))
    logger = logging.getLogger(__name__)

    rows, missing = load_all_metrics(args.analysis_dir)
    if missing:
        logger.warning(f"Nessun 'separation_metrics.json' trovato per: {', '.join(missing)} "
                        f"(esegui prima: python analyze_model.py --model {missing[0]})")
    if not rows:
        print("Nessun modello analizzato ancora. Esegui prima analyze_model.py per almeno un modello.")
        return

    print_and_save_table(rows, args.sort_by, args.ascending, args.analysis_dir)

    available_models = [r["model"] for r in rows]
    build_combined_density_plot(available_models, args.results_dir, args.analysis_dir)


if __name__ == "__main__":
    main()
