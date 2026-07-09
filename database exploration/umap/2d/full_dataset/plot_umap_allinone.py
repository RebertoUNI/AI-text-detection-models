"""
Plot densità UMAP con hexbin e contorni — densità totale per split.

Requisiti:
    pip install numpy matplotlib scipy
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ── Configurazione ─────────────────────────────────────────────────────────────
NPZ_PATH   = Path("/Users/roberto/Università/Deep learning/AI-text-detection-models/database exploration/umap/full_dataset/umap_output/umap_full_results.npz")
OUT_DIR    = Path("/Users/roberto/Università/Deep learning/AI-text-detection-models/database exploration/umap/full_dataset/umap_output/plots")
DPI        = 200
GRIDSIZE   = 40    # grandezza esagoni: più basso = esagoni più grandi
N_CONTOUR  = 8     # livelli di contorno

COLORMAP = "viridis"  # Colormap unica per la densità totale
SPLITS = ["train", "val", "test"]

# ── Caricamento ────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)

data   = np.load(NPZ_PATH)
emb    = data["embeddings_2d"]
labels = data["labels"].astype(int)
splits = data["splits"]

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


def _hexbin_plot(ax, points, title_label, extent, gridsize=GRIDSIZE):
    """Hexbin: ogni esagono = numero di campioni in quella cella (densità totale)."""
    n = len(points)
    hb = ax.hexbin(
        points[:, 0], points[:, 1],
        gridsize=gridsize,
        cmap=COLORMAP,
        mincnt=1,
        linewidths=0.1,
        edgecolors="none",
        extent=extent
    )
    plt.colorbar(hb, ax=ax, label="n. campioni")
    ax.set_title(f"Densità totale — {title_label} (n={n:,})", fontsize=13, fontweight="bold")
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
    
    # Filtriamo i dati per lo split corrente
    mask_split = splits == split_name
    emb_s = emb[mask_split]
    
    if len(emb_s) == 0:
        print(f"  Nessun dato per lo split {split_name}, salto...")
        continue
    
    # ── Hexbin densità totale ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    _hexbin_plot(ax, emb_s, f"Densità Totale [{split_name}]", extent=hex_extent)
    fig.tight_layout()
    file_hex = OUT_DIR / f"{split_name}_allinone_hexbin_total.png"
    fig.savefig(file_hex, dpi=DPI)
    plt.close(fig)
    print(f"Saved: {file_hex.name}")
    
    # ── Contorni densità totale ───────────────────────────────────────────
    if len(emb_s) >= 2:
        print(f"  Calcolo KDE ({split_name}) per i contorni...")
        fig, ax = plt.subplots(figsize=(7, 6))
        
        xx, yy, zz = _kde_grid(emb_s, xlim, ylim, resolution=200)
        contour = ax.contour(xx, yy, zz,
                            levels=N_CONTOUR,
                            colors='black',
                            linewidths=1.2,
                            alpha=0.85)
        
        # Aggiungiamo etichette ai contorni
        ax.clabel(contour, inline=True, fontsize=8, fmt='%.3f')
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"Contorni densità totale [{split_name}]", fontsize=13, fontweight="bold")
        ax.set_xlabel("UMAP 1", fontsize=11)
        ax.set_ylabel("UMAP 2", fontsize=11)
        
        fig.tight_layout()
        file_contour = OUT_DIR / f"{split_name}_contour_total.png"
        fig.savefig(file_contour, dpi=DPI)
        plt.close(fig)
        print(f"Saved: {file_contour.name}")
    else:
        print(f"  Pochi punti per KDE in {split_name}, salto contorni...")

# ── Plot per il dataset completo ───────────────────────────────────────────────
print(f"\n--- Generazione plot per il DATASET COMPLETO ---")

# ── Hexbin densità totale (completo) ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
_hexbin_plot(ax, emb, "Dataset Completo", extent=hex_extent)
fig.tight_layout()
file_hex_full = OUT_DIR / "full_dataset_allinone_hexbin_total.png"
fig.savefig(file_hex_full, dpi=DPI)
plt.close(fig)
print(f"Saved: {file_hex_full.name}")

# ── Contorni densità totale (completo) ───────────────────────────────────
print("Calcolo KDE (completo) per i contorni...")
fig, ax = plt.subplots(figsize=(7, 6))

xx, yy, zz = _kde_grid(emb, xlim, ylim, resolution=200)
contour = ax.contour(xx, yy, zz,
                    levels=N_CONTOUR,
                    colors='black',
                    linewidths=1.2,
                    alpha=0.85)

ax.clabel(contour, inline=True, fontsize=8, fmt='%.3f')

ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_title("Contorni densità totale [Dataset Completo]", fontsize=13, fontweight="bold")
ax.set_xlabel("UMAP 1", fontsize=11)
ax.set_ylabel("UMAP 2", fontsize=11)

fig.tight_layout()
file_contour_full = OUT_DIR / "full_dataset_allinone_contour_total.png"
fig.savefig(file_contour_full, dpi=DPI)
plt.close(fig)
print(f"Saved: {file_contour_full.name}")

print("\n✅ Generazione completata!")