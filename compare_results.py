"""
compare_results.py
─────────────────────────────────────────────────────────────────────────────
Legge i file 'results/{MODEL}_test_metrics.json' prodotti dai 4 script di
test e produce un'unica tabella di confronto (stampata a schermo + salvata
in CSV/Markdown), ordinata per F1 decrescente.

Uso (dopo aver lanciato tutti i test_*.py):
    python compare_results.py
"""

import argparse
import csv
import glob
import json
import os


def parse_args():
    p = argparse.ArgumentParser(description="Confronta i risultati di test dei 4 modelli")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--sort-by", type=str, default="f1",
                    choices=["accuracy", "precision", "recall", "f1", "roc_auc"])
    return p.parse_args()


def main():
    args = parse_args()
    pattern = os.path.join(args.results_dir, "*_test_metrics.json")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"Nessun file '*_test_metrics.json' trovato in '{args.results_dir}/'. "
              f"Esegui prima i vari test_*.py.")
        return

    rows = []
    for path in files:
        with open(path) as f:
            m = json.load(f)
        rows.append({
            "model":     m.get("model", os.path.basename(path)),
            "n_samples": m.get("n_samples"),
            "accuracy":  m.get("accuracy"),
            "precision": m.get("precision"),
            "recall":    m.get("recall"),
            "f1":        m.get("f1"),
            "roc_auc":   m.get("roc_auc"),
        })

    rows.sort(key=lambda r: (r[args.sort_by] is None, -(r[args.sort_by] or 0)))

    # ── Stampa a schermo ────────────────────────────────────────────────────
    header = f"{'Modello':<12} {'N':>8} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        def fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else "n/a"
        print(f"{r['model']:<12} {r['n_samples']:>8} {fmt(r['accuracy']):>9} "
              f"{fmt(r['precision']):>10} {fmt(r['recall']):>8} {fmt(r['f1']):>8} {fmt(r['roc_auc']):>8}")

    # ── Salvataggio CSV ──────────────────────────────────────────────────────
    csv_path = os.path.join(args.results_dir, "comparison_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "n_samples", "accuracy",
                                                "precision", "recall", "f1", "roc_auc"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nTabella salvata in: {csv_path}")

    # ── Salvataggio Markdown ─────────────────────────────────────────────────
    md_path = os.path.join(args.results_dir, "comparison_table.md")
    with open(md_path, "w") as f:
        f.write("| Modello | N | Accuracy | Precision | Recall | F1 | ROC-AUC |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            def fmt(v):
                return f"{v:.4f}" if isinstance(v, float) else "n/a"
            f.write(f"| {r['model']} | {r['n_samples']} | {fmt(r['accuracy'])} | "
                    f"{fmt(r['precision'])} | {fmt(r['recall'])} | {fmt(r['f1'])} | {fmt(r['roc_auc'])} |\n")
    print(f"Tabella Markdown salvata in: {md_path}")


if __name__ == "__main__":
    main()
