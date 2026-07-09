"""
Analisi del file results.csv prodotto dall'inferenza.
Output: metriche testuali nel terminale + report HTML interattivo.

Uso:
    python analyze_results.py                        # legge results.csv
    python analyze_results.py --input mio_file.csv  # file custom
    python analyze_results.py --examples 10         # più esempi per classe
"""

import argparse
import csv
import random
import textwrap
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",    default="results.csv")
    p.add_argument("--examples", type=int, default=5,
                   help="Esempi casuali per classe da mostrare")
    p.add_argument("--max_chars", type=int, default=200,
                   help="Caratteri max per testo negli esempi")
    p.add_argument("--report",   default="report.html",
                   help="File HTML di output (usa --no_html per saltarlo)")
    p.add_argument("--no_html",  action="store_true")
    p.add_argument("--seed",     type=int, default=42)
    return p.parse_args()


# ──────────────────────────────────────────────
# Caricamento
# ──────────────────────────────────────────────
def load_results(path: str, truncate_words: int = 40):
    # Aumenta il limite di campo CSV (default 131072 troppo basso per testi lunghi)
    csv.field_size_limit(10 * 1024 * 1024)  # 10 MB
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Tronca il testo a N parole per display/memoria
            words = r["text"].split()
            if len(words) > truncate_words:
                r["text"] = " ".join(words[:truncate_words]) + " …"
            rows.append(r)
    print(f"[INFO] Righe caricate: {len(rows)}  (testi troncati a {truncate_words} parole)")
    return rows


# ──────────────────────────────────────────────
# Metriche testuali
# ──────────────────────────────────────────────
def print_metrics(y_true, y_pred, labels):
    print("\n" + "═" * 60)
    print("  RISULTATI INFERENZA")
    print("═" * 60)

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"\n  Accuratezza globale  : {acc*100:.2f}%")
    print(f"  F1 (weighted)        : {f1*100:.2f}%")
    print(f"  Campioni totali      : {len(y_true)}")

    print("\n" + "─" * 60)
    print("  CLASSIFICATION REPORT")
    print("─" * 60)
    print(classification_report(y_true, y_pred, zero_division=0))

    print("─" * 60)
    print("  CONFUSION MATRIX")
    print("─" * 60)
    cm     = confusion_matrix(y_true, y_pred, labels=labels)
    header = "           " + "  ".join(f"{l:>10}" for l in labels)
    print(header)
    for i, row in enumerate(cm):
        cells = "  ".join(f"{v:>10}" for v in row)
        print(f"  {labels[i]:>8}  {cells}")

    print("\n  Per riga = ground truth | Per colonna = predetto")
    print("═" * 60 + "\n")


# ──────────────────────────────────────────────
# Esempi casuali
# ──────────────────────────────────────────────
def print_examples(rows, n, max_chars, seed):
    random.seed(seed)

    # Raggruppa per (gt, pred)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["ground_truth"], r["predicted_label"])].append(r["text"])

    labels = sorted({r["ground_truth"] for r in rows})
    print("═" * 60)
    print("  ESEMPI CASUALI")
    print("═" * 60)

    for gt in labels:
        for pred in labels:
            bucket = groups[(gt, pred)]
            if not bucket:
                continue
            tag = "✓ CORRETTI" if gt == pred else "✗ ERRORI"
            print(f"\n  [{tag}]  GT={gt}  →  Predetto={pred}  "
                  f"({len(bucket)} campioni)")
            print("  " + "─" * 56)
            samples = random.sample(bucket, min(n, len(bucket)))
            for i, text in enumerate(samples, 1):
                snippet = textwrap.fill(
                    text[:max_chars] + ("…" if len(text) > max_chars else ""),
                    width=70,
                    initial_indent="    ",
                    subsequent_indent="    ",
                )
                print(f"  [{i}] {snippet}")
    print("\n" + "═" * 60)


# ──────────────────────────────────────────────
# Report HTML
# ──────────────────────────────────────────────
def build_html(rows, y_true, y_pred, labels, examples_per_class, seed, max_chars):
    random.seed(seed)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm  = confusion_matrix(y_true, y_pred, labels=labels)
    report_dict = {}
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    for i, lbl in enumerate(labels):
        report_dict[lbl] = dict(precision=p[i], recall=r[i],
                                f1=f[i], support=int(s[i]))

    groups = defaultdict(list)
    for row in rows:
        groups[(row["ground_truth"], row["predicted_label"])].append(row["text"])

    # ── palette colori per label (fino a 8 classi)
    palette = ["#4f6ef7", "#f7774f", "#4fc97b", "#f7c94f",
               "#a04ff7", "#f74f9e", "#4fc9f7", "#888888"]
    color_map = {lbl: palette[i % len(palette)] for i, lbl in enumerate(labels)}

    # ── confusion matrix SVG
    n = len(labels)
    cell = 72
    pad  = 120
    svg_w = pad + n * cell + 20
    svg_h = pad + n * cell + 20
    max_val = cm.max() if cm.max() > 0 else 1

    cm_cells = []
    for ri, gt in enumerate(labels):
        for ci, pred in enumerate(labels):
            val = int(cm[ri, ci])
            alpha = 0.15 + 0.80 * (val / max_val)
            x = pad + ci * cell
            y = pad + ri * cell
            txt_color = "#fff" if alpha > 0.5 else "#222"
            cm_cells.append(
                f'<rect x="{x}" y="{y}" width="{cell-4}" height="{cell-4}" '
                f'rx="6" fill="{color_map[gt]}" opacity="{alpha:.2f}"/>'
                f'<text x="{x+cell//2-2}" y="{y+cell//2+5}" '
                f'font-size="15" fill="{txt_color}" text-anchor="middle">{val}</text>'
            )

    row_labels = "".join(
        f'<text x="{pad-8}" y="{pad + i*cell + cell//2 + 5}" '
        f'font-size="13" fill="{color_map[lbl]}" '
        f'text-anchor="end" font-weight="600">{lbl}</text>'
        for i, lbl in enumerate(labels)
    )
    col_labels = "".join(
        f'<text x="{pad + i*cell + cell//2 - 2}" y="{pad-12}" '
        f'font-size="13" fill="{color_map[lbl]}" '
        f'text-anchor="middle" font-weight="600">{lbl}</text>'
        for i, lbl in enumerate(labels)
    )
    cm_svg = (
        f'<svg viewBox="0 0 {svg_w} {svg_h}" '
        f'xmlns="http://www.w3.org/2000/svg" style="max-width:100%">'
        f'<text x="{svg_w//2}" y="20" text-anchor="middle" '
        f'font-size="14" fill="#888">Predetto →</text>'
        f'<text x="14" y="{pad + n*cell//2}" '
        f'font-size="14" fill="#888" '
        f'transform="rotate(-90,14,{pad + n*cell//2})">Ground Truth →</text>'
        + col_labels + row_labels + "".join(cm_cells)
        + '</svg>'
    )

    # ── tabella metriche per classe
    rows_table = ""
    for lbl in labels:
        m = report_dict[lbl]
        bar = f'<div style="background:{color_map[lbl]};height:6px;border-radius:3px;width:{m["f1"]*100:.0f}%"></div>'
        rows_table += (
            f"<tr>"
            f'<td><span style="color:{color_map[lbl]};font-weight:700">{lbl}</span></td>'
            f'<td>{m["precision"]*100:.1f}%</td>'
            f'<td>{m["recall"]*100:.1f}%</td>'
            f'<td>{m["f1"]*100:.1f}%{bar}</td>'
            f'<td>{m["support"]}</td>'
            f"</tr>"
        )

    # ── card esempi
    example_cards = ""
    for gt in labels:
        for pred in labels:
            bucket = groups[(gt, pred)]
            if not bucket:
                continue
            is_correct = gt == pred
            border = color_map[gt] if is_correct else "#e74c3c"
            tag    = "✓" if is_correct else "✗"
            tag_bg = "#2ecc71" if is_correct else "#e74c3c"
            samples = random.sample(bucket, min(examples_per_class, len(bucket)))
            cards_inner = ""
            for text in samples:
                snippet = text[:max_chars] + ("…" if len(text) > max_chars else "")
                snippet = snippet.replace("&", "&amp;").replace("<", "&lt;")
                cards_inner += (
                    f'<div style="background:#f8f9fa;border-radius:8px;'
                    f'padding:12px 16px;margin-bottom:10px;font-size:13px;'
                    f'color:#444;line-height:1.6">{snippet}</div>'
                )
            example_cards += (
                f'<div style="border-left:4px solid {border};'
                f'background:#fff;border-radius:10px;padding:18px 22px;'
                f'margin-bottom:18px;box-shadow:0 2px 8px rgba(0,0,0,.06)">'
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">'
                f'<span style="background:{tag_bg};color:#fff;border-radius:50%;'
                f'width:26px;height:26px;display:flex;align-items:center;'
                f'justify-content:center;font-size:14px;font-weight:700">{tag}</span>'
                f'<span style="font-weight:600;color:#333">GT = '
                f'<span style="color:{color_map[gt]}">{gt}</span> → '
                f'Predetto = <span style="color:{color_map[pred]}">{pred}</span></span>'
                f'<span style="margin-left:auto;font-size:12px;color:#999">'
                f'{len(bucket)} campioni totali</span></div>'
                + cards_inner + "</div>"
            )

    html = f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Analisi Inferenza — paradetect-deberta-v3-lora</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Segoe UI',system-ui,sans-serif;background:#f0f2f5;color:#222;padding:32px 24px}}
  h1{{font-size:22px;font-weight:700;color:#1a1a2e;margin-bottom:4px}}
  .subtitle{{color:#666;font-size:14px;margin-bottom:32px}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin-bottom:32px}}
  .card{{background:#fff;border-radius:12px;padding:22px 24px;box-shadow:0 2px 8px rgba(0,0,0,.07)}}
  .card .val{{font-size:36px;font-weight:800;line-height:1}}
  .card .lbl{{font-size:13px;color:#888;margin-top:6px}}
  .section{{background:#fff;border-radius:12px;padding:24px 28px;
            margin-bottom:28px;box-shadow:0 2px 8px rgba(0,0,0,.07)}}
  .section h2{{font-size:16px;font-weight:700;color:#333;margin-bottom:18px;
               border-bottom:2px solid #f0f2f5;padding-bottom:10px}}
  table{{width:100%;border-collapse:collapse;font-size:14px}}
  th{{text-align:left;padding:8px 12px;background:#f8f9fa;color:#555;
      font-weight:600;border-bottom:2px solid #e9ecef}}
  td{{padding:10px 12px;border-bottom:1px solid #f0f2f5;vertical-align:middle}}
  tr:last-child td{{border-bottom:none}}
</style>
</head>
<body>
<h1>Analisi Inferenza</h1>
<p class="subtitle">paradetect-deberta-v3-lora &nbsp;·&nbsp;
  srikanthgali/ai-text-detection-pile-cleaned (test split)</p>

<div class="grid">
  <div class="card">
    <div class="val">{acc*100:.1f}<span style="font-size:20px">%</span></div>
    <div class="lbl">Accuratezza globale</div>
  </div>
  <div class="card">
    <div class="val">{f1*100:.1f}<span style="font-size:20px">%</span></div>
    <div class="lbl">F1 weighted</div>
  </div>
  <div class="card">
    <div class="val" style="font-size:28px">{len(y_true):,}</div>
    <div class="lbl">Campioni totali</div>
  </div>
  <div class="card">
    <div class="val" style="font-size:28px">{len(labels)}</div>
    <div class="lbl">Classi</div>
  </div>
</div>

<div class="section">
  <h2>Metriche per classe</h2>
  <table>
    <thead><tr><th>Classe</th><th>Precision</th><th>Recall</th>
    <th>F1</th><th>Supporto</th></tr></thead>
    <tbody>{rows_table}</tbody>
  </table>
</div>

<div class="section">
  <h2>Confusion Matrix</h2>
  {cm_svg}
</div>

<div class="section">
  <h2>Esempi ({examples_per_class} per combinazione GT/Predetto)</h2>
  {example_cards}
</div>

</body></html>"""
    return html


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    args = parse_args()

    rows   = load_results(args.input)
    y_true = [r["ground_truth"]    for r in rows]
    y_pred = [r["predicted_label"] for r in rows]
    labels = sorted(set(y_true) | set(y_pred))

    print_metrics(y_true, y_pred, labels)
    print_examples(rows, args.examples, args.max_chars, args.seed)

    if not args.no_html:
        html = build_html(rows, y_true, y_pred, labels,
                          args.examples, args.seed, args.max_chars)
        out = Path(args.report)
        out.write_text(html, encoding="utf-8")
        print(f"[INFO] Report HTML salvato in '{out}'")


if __name__ == "__main__":
    main()