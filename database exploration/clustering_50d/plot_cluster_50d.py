"""
cluster_plot.py — Visualizzazione 2D dei cluster HDBSCAN su embedding UMAP

Produce 3 subplot affiancati:
  1. Tutti i punti colorati per cluster
  2. Solo label=0 (Human)
  3. Solo label=1 (AI)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')   # obbligatorio su HPC senza display, va prima di pyplot
import matplotlib.pyplot as plt
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ============================================================
# ▼▼▼  MODIFICA SOLO QUESTA SEZIONE  ▼▼▼
# ============================================================
RUN_ID = 4      # ID del run HDBSCAN da visualizzare
SPLIT  = "all"  # "all" | "train" | "val" | "test"
# ============================================================

# ── Percorsi file ────────────────────────────────────────────
UMAP_2D_FILE        = "database exploration/umap/2d/full_dataset/umap_output/umap_full_dataset_2d.npz"
HDBSCAN_LABELS_FILE = f"database exploration/clustering_50d/out/hdbscan_labels_run{RUN_ID}.npy"
OUTPUT_FILE         = f"database exploration/clustering_50d/plot/run_{RUN_ID}/cluster_plot_run{RUN_ID}_{SPLIT}.png"

# ── Parametri grafici ────────────────────────────────────────
DPI             = 200
FIG_SIZE        = (24, 8)    # 3 subplot affiancati
POINT_SIZE      = 0.3        # marcatore piccolo per 700k punti
ALPHA_CLUSTER   = 0.35       # trasparenza punti cluster
ALPHA_NOISE     = 0.15       # trasparenza punti rumore (più trasparenti)
NOISE_COLOR     = (0.88, 0.88, 0.88)  # grigio chiaro per cluster = -1
SHOW_LABELS     = True       # mostra numero cluster al centroide
LABEL_FONTSIZE  = 6
LABEL_NAMES     = {0: "Human (label=0)", 1: "AI (label=1)"}


# ── Utility: palette colori ──────────────────────────────────

def build_color_palette(n: int) -> list:
    """
    Genera n colori distinti.
    Cicla su tab20 / tab20b / tab20c (fino a 60 colori),
    poi usa HSV per dataset con moltissimi cluster.
    Compatibile con matplotlib >= 3.9 (get_cmap rimosso).
    """
    def get_cmap(name):
        # matplotlib.colormaps disponibile da 3.5; fallback per versioni vecchie
        try:
            return matplotlib.colormaps[name]
        except AttributeError:
            return matplotlib.cm.get_cmap(name)  # < 3.5
 
    base = []
    for cmap_name in ('tab20', 'tab20b', 'tab20c'):
        cmap = get_cmap(cmap_name)
        base.extend([cmap(i / 20) for i in range(20)])
    if n <= len(base):
        return base[:n]
    hsv = get_cmap('hsv')
    return [hsv(i / n) for i in range(n)]


def make_rgba_array(cluster_labels: np.ndarray, palette: list) -> np.ndarray:
    """
    Restituisce un array (N, 4) RGBA:
      - cluster k → palette[k]
      - rumore (-1) → lascito a zeros (gestito separatamente in scatter_panel)
    """
    rgba = np.zeros((len(cluster_labels), 4))
    for k, color in enumerate(palette):
        mask = cluster_labels == k
        rgba[mask] = color
    return rgba


# ── Utility: scatter singolo pannello ───────────────────────
def compute_limits(xy: np.ndarray, percentile: float = 0.5):
    """
    Calcola xlim e ylim escludendo gli outlier estremi.
    percentile=0.5 → taglia lo 0.5% più basso e lo 0.5% più alto su ogni asse.
    Aumenta il valore se ci sono ancora outlier residui.
    """
    x_lo, x_hi = np.percentile(xy[:, 0], [percentile, 100 - percentile])
    y_lo, y_hi = np.percentile(xy[:, 1], [percentile, 100 - percentile])
    pad_x = (x_hi - x_lo) * 0.05   # 5% di margine
    pad_y = (y_hi - y_lo) * 0.05
    return (x_lo - pad_x, x_hi + pad_x), (y_lo - pad_y, y_hi + pad_y)
 

def scatter_panel(
    ax,
    xy: np.ndarray,
    cluster_labels: np.ndarray,
    rgba: np.ndarray,
    title: str,
    xlim: tuple = None,
    ylim: tuple = None,
):
    """
    Disegna un pannello scatter.
    Strategia: rumore in fondo (grigio, molto trasparente),
    cluster sopra (colorati).
    rasterized=True mantiene il file leggero pur avendo 700k punti.
    """
    noise_mask   = cluster_labels == -1
    cluster_mask = ~noise_mask
 
    # 1. Rumore (sotto)
    if noise_mask.any():
        ax.scatter(
            xy[noise_mask, 0], xy[noise_mask, 1],
            c=[NOISE_COLOR], s=POINT_SIZE,
            alpha=ALPHA_NOISE, linewidths=0,
            rasterized=True, zorder=1
        )
 
    # 2. Cluster (sopra)
    if cluster_mask.any():
        ax.scatter(
            xy[cluster_mask, 0], xy[cluster_mask, 1],
            c=rgba[cluster_mask], s=POINT_SIZE,
            alpha=ALPHA_CLUSTER, linewidths=0,
            rasterized=True, zorder=2
        )
 
    # 3. Etichette numeriche al centroide di ogni cluster
    if SHOW_LABELS:
        for k in sorted(set(cluster_labels) - {-1}):
            mask = cluster_labels == k
            cx, cy = xy[mask, 0].mean(), xy[mask, 1].mean()
            ax.text(
                cx, cy, str(k),
                fontsize=LABEL_FONTSIZE,
                ha='center', va='center',
                color='black', zorder=3,
                bbox=dict(
                    boxstyle='round,pad=0.1',
                    fc='white', ec='none', alpha=0.55
                )
            )
 
    n_clusters  = len(set(cluster_labels) - {-1})
    noise_ratio = noise_mask.mean()
    n_shown     = len(xy)
 
    ax.set_title(
        f"{title}\n"
        f"{n_shown:,} punti | {n_clusters} cluster | rumore {noise_ratio:.1%}",
        fontsize=10
    )
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    # adjustable='box' rispetta i limiti impostati senza allargarli
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(labelsize=7)


# ── Main ─────────────────────────────────────────────────────

def main():
    # Caricamento embedding 2D e metadati
    logging.info(f"Caricamento {UMAP_2D_FILE} ...")
    raw       = np.load(UMAP_2D_FILE, allow_pickle=True)
    xy        = raw['embeddings_2d']   # (N, 2)
    ai_labels = raw['labels']          # 0=human / 1=AI
    splits    = raw['splits']          # array di stringhe 'train'|'val'|'test'

    # numpy può salvare stringhe come bytes — decodifichiamo se necessario
    if splits.dtype.kind == 'S':
        splits = np.array([s.decode() for s in splits])

    # Caricamento etichette HDBSCAN per il run scelto
    logging.info(f"Caricamento {HDBSCAN_LABELS_FILE} ...")
    hdb_labels = np.load(HDBSCAN_LABELS_FILE)   # (N,), -1 = rumore

    # Filtro split
    if SPLIT == "all":
        mask_split = np.ones(len(xy), dtype=bool)
        logging.info(f"Split='all': {len(xy):,} punti totali")
    else:
        mask_split = splits == SPLIT
        logging.info(f"Split='{SPLIT}': {mask_split.sum():,} / {len(xy):,} punti")

    xy_f  = xy[mask_split]
    hdb_f = hdb_labels[mask_split]
    ai_f  = ai_labels[mask_split]

    # Rimappa cluster ID → indici contigui [0, n_clusters)
    # necessario perché i cluster ID potrebbero non essere consecutivi
    unique_clusters = sorted(set(hdb_f) - {-1})
    n_clusters      = len(unique_clusters)
    logging.info(f"Run {RUN_ID}: {n_clusters} cluster, "
                 f"rumore {(hdb_f == -1).mean():.1%}")

    remap    = {cid: i for i, cid in enumerate(unique_clusters)}
    hdb_norm = np.array([remap.get(c, -1) for c in hdb_f])

    # Palette e array colori
    palette = build_color_palette(n_clusters)
    rgba    = make_rgba_array(hdb_norm, palette)

    xlim, ylim = compute_limits(xy_f, percentile=0.5)
    logging.info(f"Limiti assi — X: {xlim[0]:.2f} / {xlim[1]:.2f} "
                 f"| Y: {ylim[0]:.2f} / {ylim[1]:.2f}")


    # ── Figura con 3 subplot ──────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=FIG_SIZE, dpi=DPI)
    fig.suptitle(
        f"HDBSCAN Run {RUN_ID}  —  Split: {SPLIT}",
        fontsize=13, fontweight='bold', y=1.01
    )
 
    # Subplot 1 — tutti
    scatter_panel(axes[0], xy_f, hdb_norm, rgba, "Tutti i punti", xlim, ylim)
 
    # Subplot 2 — Human (label=0)
    m0 = ai_f == 0
    scatter_panel(axes[1], xy_f[m0], hdb_norm[m0], rgba[m0],
                  LABEL_NAMES.get(0, "label=0"), xlim, ylim)
 
    # Subplot 3 — AI (label=1)
    m1 = ai_f == 1
    scatter_panel(axes[2], xy_f[m1], hdb_norm[m1], rgba[m1],
                  LABEL_NAMES.get(1, "label=1"), xlim, ylim)
 
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches='tight')
    logging.info(f"Salvato: {OUTPUT_FILE}")
 
 
if __name__ == "__main__":
    main()