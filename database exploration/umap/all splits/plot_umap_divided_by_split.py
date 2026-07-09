"""
Plot densità UMAP con hexbin e contorni — separati per split e label.

Requisiti:
    pip install numpy matplotlib scipy
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D

# ── Configurazione ─────────────────────────────────────────────────────────────
# Assicurati che il path punti al file corretto (quello unito)
NPZ_PATH   = Path("umap_output_full/umap_full_results.npz")
OUT_DIR    = Path("umap_output_full/plots")
DPI        = 200
GRIDSIZE   = 40    # grandezza esagoni: più basso = esagoni più grandi
N_CONTOUR  = 8     # livelli di contorno

COLORS = {0: "Blues", 1: "Reds"}
CONTOUR_COLORS = {0: "#4C72B0", 1: "#CC3311"}
LABELS = {0: "Umano", 1: "AI"}
SPLITS = ["train", "val", "test"]

# ── Caricamento ────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)

data   = np.load(NPZ_PATH)
emb    = data["embeddings_2d"]
labels = data["labels"].astype(int)
splits = data["splits"]  # Carichiamo l'array degli split

# Limiti globali: calcolati su TUTTO il dataset per mantenere la stessa scala
pad  = 0.3
p_low, p_high = 1, 99
x_min, x_max = np.percentile(emb[:, 0], [p_low, p_high])
y_min, y_max = np.percentile(emb[:, 1], [p_low, p_high])

# Aggiungiamo un margine proporzionale (10% dello spazio)
pad_x = (x_max - x_min) * 0.1
pad_y = (y_max - y_min) * 0.1

xlim = (x_min - pad_x, x_max + pad_x)
ylim = (y_min - pad_y, y_max + pad_y)

hex_extent = (xlim[0], xlim[1], ylim[0], ylim[1])


def _hexbin_plot(ax, points, colormap, title_label, extent, gridsize=GRIDSIZE):
    """Hexbin: ogni esagono = numero di campioni in quella cella."""
    n = len(points)
    hb = ax.hexbin(
        points[:, 0], points[:, 1],
        gridsize=gridsize,
        cmap=colormap,
        mincnt=1,
        linewidths=0.1,
        edgecolors="none",
        extent=extent # Forza la griglia globale
    )
    plt.colorbar(hb, ax=ax, label="n. campioni")
    ax.set_title(f"Densità — {title_label} (n={n:,})", fontsize=13, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=11)
    ax.set_ylabel("UMAP 2", fontsize=11)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")


def _kde_grid(points, xlim, ylim, resolution=200):
    """Calcola una griglia KDE per i contorni."""
    xi = np.linspace(xlim[0], xlim[1], resolution)
    yi = np.linspace(ylim[0], ylim[1], resolution)
    xx, yy = np.meshgrid(xi, yi)
    kde = gaussian_kde(points.T, bw_method="scott")
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    return xx, yy, zz


# ── Generazione Grafici per Split ──────────────────────────────────────────────
for split_name in SPLITS:
    print(f"\n--- Generazione plot per lo split: {split_name.upper()} ---")
    
    # 1. Filtriamo i dati per lo split corrente
    split_mask = (splits == split_name)
    if not np.any(split_mask):
        print(f"Attenzione: Nessun dato trovato per {split_name}, salto...")
        continue
        
    emb_s = emb[split_mask]
    labels_s = labels[split_mask]
    
    # Maschere per Umano (0) e AI (1) all'interno dello split
    mask_s = {k: labels_s == k for k in (0, 1)}

    # ── Hexbin label 0 ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    _hexbin_plot(ax, emb_s[mask_s[0]], COLORS[0], f"{LABELS[0]} [{split_name}]", extent=hex_extent)
    fig.tight_layout()
    file_0 = OUT_DIR / f"{split_name}_04_hexbin_label0.png"
    fig.savefig(file_0, dpi=DPI)
    plt.close(fig)
    print(f"Saved: {file_0.name}")

    # ── Hexbin label 1 ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    _hexbin_plot(ax, emb_s[mask_s[1]], COLORS[1], f"{LABELS[1]} [{split_name}]", extent=hex_extent)
    fig.tight_layout()
    file_1 = OUT_DIR / f"{split_name}_05_hexbin_label1.png"
    fig.savefig(file_1, dpi=DPI)
    plt.close(fig)
    print(f"Saved: {file_1.name}")

    # ── Contorni sovrapposti ──────────────────────────────────────────────────
    print(f"Calcolo KDE ({split_name}) per i contorni...")
    fig, ax = plt.subplots(figsize=(7, 6))

    for k in (0, 1):
        if len(emb_s[mask_s[k]]) < 2:
            continue # Salta se non ci sono abbastanza punti per la KDE
            
        xx, yy, zz = _kde_grid(emb_s[mask_s[k]], xlim, ylim, resolution=200)
        ax.contour(xx, yy, zz,
                   levels=N_CONTOUR,
                   colors=CONTOUR_COLORS[k],
                   linewidths=1.2,
                   alpha=0.85,
                   label=LABELS[k])

    legend_handles = [
        Line2D([0], [0], color=CONTOUR_COLORS[k], linewidth=2, label=LABELS[k])
        for k in (0, 1)
    ]
    ax.legend(handles=legend_handles, fontsize=11, framealpha=0.8)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(f"Contorni sovrapposti [{split_name}]", fontsize=13, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=11)
    ax.set_ylabel("UMAP 2", fontsize=11)