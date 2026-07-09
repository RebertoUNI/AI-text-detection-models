"""
analyze_model.py
─────────────────────────────────────────────────────────────────────────────
Esegue l'intera pipeline di analisi (densità, predizioni vs ground truth,
atomografia, cluster embedding, occlusione/vocabolario) per UN modello, e
salva tutti i plot come HTML interattivi + le metriche di separazione come
JSON, in modo che compare_models_separation.py possa poi confrontarli tutti.

Uso:
    python analyze_model.py --model FCNN
    python analyze_model.py --model PaperCNN
    python analyze_model.py --model daBERTa
    python analyze_model.py --model Qwen

Output (per modello) in analysis_plots/{MODEL}/:
    separation_metrics.json
    densita_sovrapposte.html
    predizioni_vs_ground_truth.html
    atomografia_dataset.html
    cluster_embeddings_pca.html          (se embeddings {MODEL}/ esiste)
    occlusion_heatmap_example_0.html     (se richiesto --with-occlusion)
    vocabolario_globale.csv              (se richiesto --with-occlusion)

NOTA: l'occlusione manda al modello UNA sequenza per ogni parola del testo,
quindi è lenta ed è disattivata di default. Attivala con --with-occlusion
(consigliato con GPU, soprattutto per DeBERTa/Qwen).
"""

import argparse
import logging
import os

from train_utils import setup_logging
import analysis_utils as au

MODELS = {
    "FCNN":     {"checkpoint_dir": "checkpoint FCNN",     "embeddings_dir": "embeddings FCNN"},
    "PaperCNN": {"checkpoint_dir": "checkpoint PaperCNN", "embeddings_dir": "embeddings PaperCNN"},
    "daBERTa":  {"checkpoint_dir": "checkpoint daBERTa",  "embeddings_dir": "embeddings daBERTa"},
    "Qwen":     {"checkpoint_dir": "checkpoint Qwen",     "embeddings_dir": "embeddings Qwen"},
}


def parse_args():
    p = argparse.ArgumentParser(description="Analisi di separazione AI vs Human per un modello")
    p.add_argument("--model", required=True, choices=list(MODELS.keys()))
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--output-dir", type=str, default="analysis_plots")
    p.add_argument("--max-points", type=int, default=3000, help="Punti mostrati negli scatter plot")
    p.add_argument("--umap", action="store_true", help="Aggiunge anche la proiezione UMAP (richiede umap-learn)")
    p.add_argument("--with-occlusion", action="store_true",
                    help="Attiva l'analisi di occlusione/vocabolario (lenta, consigliata con GPU)")
    p.add_argument("--n-occlusion-examples", type=int, default=3)
    p.add_argument("--n-vocab-sentences", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = MODELS[args.model]
    out_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(out_dir, exist_ok=True)
    setup_logging(os.path.join(out_dir, "analysis.log"))
    logger = logging.getLogger(__name__)

    # ── 1. Predizioni + testi allineati ────────────────────────────────────
    y_true, y_prob = au.load_test_predictions(args.model, results_dir=args.results_dir)
    texts = au.load_test_texts(split="test")
    logger.info(f"[{args.model}] {len(y_true)} predizioni di test caricate")

    # ── 2. Separazione quantitativa + densità ──────────────────────────────
    metrics, grid, kde_h, kde_a = au.quantifica_separazione(y_true, y_prob)
    import json
    with open(os.path.join(out_dir, "separation_metrics.json"), "w") as f:
        json.dump({"model": args.model, **metrics}, f, indent=2)

    fig = au.plot_densita_sovrapposte(grid, kde_h, kde_a, model_name=args.model)
    au.save_and_maybe_show(fig, os.path.join(out_dir, "densita_sovrapposte.html"))

    # ── 3. Predizioni vs ground truth + atomografia ────────────────────────
    fig, _ = au.plot_predizioni_vs_ground_truth(y_true, y_prob, texts, model_name=args.model,
                                                 max_points=args.max_points)
    au.save_and_maybe_show(fig, os.path.join(out_dir, "predizioni_vs_ground_truth.html"))

    fig, _ = au.atomografia_dataset(y_true, y_prob, texts, model_name=args.model,
                                     max_points=args.max_points)
    au.save_and_maybe_show(fig, os.path.join(out_dir, "atomografia_dataset.html"))

    # ── 4. Cluster degli embedding (se presenti) ───────────────────────────
    emb_dir = cfg["embeddings_dir"]
    if os.path.isdir(emb_dir):
        try:
            X, y = au.load_embeddings_shards(emb_dir, split="test")
            fig, _ = au.plot_cluster_embeddings(X, y, metodo="pca", model_name=args.model)
            au.save_and_maybe_show(fig, os.path.join(out_dir, "cluster_embeddings_pca.html"))
            if args.umap:
                fig, _ = au.plot_cluster_embeddings(X, y, metodo="umap", model_name=args.model)
                au.save_and_maybe_show(fig, os.path.join(out_dir, "cluster_embeddings_umap.html"))
        except FileNotFoundError as e:
            logger.warning(f"Embeddings non trovati per {args.model}: {e}")
    else:
        logger.warning(f"Cartella embeddings '{emb_dir}' non trovata, salto il cluster plot.")

    # ── 5. Occlusione + vocabolario globale (opzionale, lenta) ─────────────
    if args.with_occlusion:
        from predict_fn_loaders import PREDICT_FN_BUILDERS
        logger.info(f"Costruisco predict_fn per {args.model} (carica il modello in memoria)...")
        predict_fn = PREDICT_FN_BUILDERS[args.model](cfg["checkpoint_dir"])

        for i in range(min(args.n_occlusion_examples, len(texts))):
            imp = au.occlusion_importance(texts[i], predict_fn, max_words=60)
            fig = au.plot_heatmap_importanza(imp, model_name=args.model)
            if fig is not None:
                au.save_and_maybe_show(fig, os.path.join(out_dir, f"occlusion_heatmap_example_{i}.html"))

        df_vocab = au.analizza_vocabolario_globale(
            texts, predict_fn, n_frasi=args.n_vocab_sentences, max_words=60, top_n=15
        )
        if df_vocab is not None:
            csv_path = os.path.join(out_dir, "vocabolario_globale.csv")
            df_vocab.to_csv(csv_path, index=False)
            logger.info(f"Vocabolario globale salvato in {csv_path}")

    logger.info(f"Analisi di {args.model} completata. Output in {out_dir}/")


if __name__ == "__main__":
    main()
