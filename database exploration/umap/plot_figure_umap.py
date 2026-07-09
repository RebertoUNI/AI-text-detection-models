"""
Plot densità UMAP con hexbin e contorni — figure separate.

Requisiti:
    pip install numpy matplotlib scipy
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ── Configurazione ─────────────────────────────────────────────────────────────
NPZ_PATH   = Path("umap_output/umap_results.npz")
OUT_DIR    = Path("umap_output/plots")
DPI        = 200
GRIDSIZE   = 40    # grandezza esagoni: più basso = esagoni più grandi
N_CONTOUR  = 8     # livelli di contorno

COLORS = {0: "Blues", 1: "Reds"}
CONTOUR_COLORS = {0: "#4C72B0", 1: "#CC3311"}
LABELS = {0: "Umano", 1: "AI"}

# ── Caricamento ────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)

data   = np.load(NPZ_PATH)
emb    = data["embeddings_2d"]
labels = data["labels"].astype(int)
mask   = {k: labels == k for k in (0, 1)}

# Limiti globali per il plot dei contorni sovrapposti
pad  = 0.3
xlim = (emb[:, 0].min() - pad, emb[:, 0].max() + pad)
ylim = (emb[:, 1].min() - pad, emb[:, 1].max() + pad)


def _hexbin_plot(ax, points, colormap, label, gridsize=GRIDSIZE):
    """Hexbin: ogni esagono = numero di campioni in quella cella."""
    n = len(points)
    hb = ax.hexbin(
        points[:, 0], points[:, 1],
        gridsize=gridsize,
        cmap=colormap,
        mincnt=1,          # non mostra celle vuote
        linewidths=0.1,
        edgecolors="none",
    )
    plt.colorbar(hb, ax=ax, label="n. campioni")
    ax.set_title(f"Densità — {label} (n={n:,})", fontsize=13, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=11)
    ax.set_ylabel("UMAP 2", fontsize=11)
    ax.set_aspect("equal", adjustable="box")


def _kde_grid(points, xlim, ylim, resolution=200):
    """Calcola una griglia KDE per i contorni."""
    xi = np.linspace(xlim[0], xlim[1], resolution)
    yi = np.linspace(ylim[0], ylim[1], resolution)
    xx, yy = np.meshgrid(xi, yi)
    kde = gaussian_kde(points.T, bw_method="scott")
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    return xx, yy, zz


# ── 1. Hexbin label 0 ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
_hexbin_plot(ax, emb[mask[0]], COLORS[0], LABELS[0])
fig.tight_layout()
fig.savefig(OUT_DIR / "04_hexbin_label0.png", dpi=DPI)
plt.close(fig)
print("Saved: 04_hexbin_label0.png")

# ── 2. Hexbin label 1 ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
_hexbin_plot(ax, emb[mask[1]], COLORS[1], LABELS[1])
fig.tight_layout()
fig.savefig(OUT_DIR / "05_hexbin_label1.png", dpi=DPI)
plt.close(fig)
print("Saved: 05_hexbin_label1.png")

# ── 3. Contorni sovrapposti ────────────────────────────────────────────────────
print("Calcolo KDE per i contorni (può richiedere qualche secondo)…")
fig, ax = plt.subplots(figsize=(7, 6))

for k in (0, 1):
    xx, yy, zz = _kde_grid(emb[mask[k]], xlim, ylim, resolution=200)
    ax.contour(xx, yy, zz,
               levels=N_CONTOUR,
               colors=CONTOUR_COLORS[k],
               linewidths=1.2,
               alpha=0.85,
               label=LABELS[k])

# Legenda manuale (contour non supporta label direttamente)
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], color=CONTOUR_COLORS[k], linewidth=2, label=LABELS[k])
    for k in (0, 1)
]
ax.legend(handles=legend_handles, fontsize=11, framealpha=0.8)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_title("Contorni di densità sovrapposti", fontsize=13, fontweight="bold")
ax.set_xlabel("UMAP 1", fontsize=11)
ax.set_ylabel("UMAP 2", fontsize=11)
ax.set_aspect("equal", adjustable="box")
fig.tight_layout()
fig.savefig(OUT_DIR / "06_contours_overlap.png", dpi=DPI)
plt.close(fig)
print("Saved: 06_contours_overlap.png")

print(f"\nDone — plot salvati in: {OUT_DIR.resolve()}")