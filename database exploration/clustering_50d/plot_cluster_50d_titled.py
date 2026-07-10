# cluster_plot.py — Visualizzazione 2D dei cluster HDBSCAN su embedding UMAP
# Produce 3 subplot affiancati:
#   1. Tutti i punti colorati per cluster
#   2. Solo label=0 (Human)
#   3. Solo label=1 (AI)

import numpy as np
import matplotlib
matplotlib.use('Agg')   # obbligatorio su HPC senza display, va prima di pyplot
import matplotlib.pyplot as plt
import logging

# Importazione opzionale ma raccomandata per evitare sovrapposizioni delle etichette
try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ============================================================
# ▼▼▼  MODIFICA SOLO QUESTA SEZIONE  ▼▼▼
# ============================================================
RUN_ID = 5      # ID del run HDBSCAN da visualizzare
SPLIT  = "all"  # "all" | "train" | "val" | "test"
# ============================================================

# ── Percorsi file ────────────────────────────────────────────
UMAP_2D_FILE        = "database exploration/umap/2d/full_dataset/umap_output/umap_full_dataset_2d.npz"
HDBSCAN_LABELS_FILE = f"database exploration/clustering_50d/out/hdbscan_labels_run{RUN_ID}.npy"
OUTPUT_FILE         = f"database exploration/clustering_50d/plot/run_{RUN_ID}/cluster_plot_run{RUN_ID}_{SPLIT}_titled.png"

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

# --- NUOVO DIZIONARIO ---
CLUSTER_NAMES = {
    0: "Sport",
    1: "Narrativa/fiction (prosa creativa)",
    2: "Auto",
    3: "Clima/energia",
    4: "Fai-da-te/tessuti",
    5: "Nutrizione/salute alimentare",
    6: "Studio/apprendimento",
    7: "Cinema/TV",
    8: "Informatica/software",
    9: "Videogiochi",
    10: "Cronaca/giustizia",
    11: "Politica/economia"
}

# ── Utility: palette colori ──────────────────────────────────

def build_color_palette(n: int) -> list:
    """
    Genera n colori distinti.
    Cicla su tab20 / tab20b / tab20c (fino a 60 colori),
    poi usa HSV per dataset con moltissimi cluster.
    Compatibile con matplotlib >= 3.9 (get_cmap rimosso).
    """
    def get_cmap(name):
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
    Disegna un pannello scatter con labels leggibili e anti-collisione.
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
    texts = []
    if SHOW_LABELS:
        for k in sorted(set(cluster_labels) - {-1}):
            mask = cluster_labels == k
            cx, cy = xy[mask, 0].mean(), xy[mask, 1].mean()
            
            nome_cluster = CLUSTER_NAMES.get(k, "")
            testo_etichetta = f"{k} - {nome_cluster}" if nome_cluster else str(k)
            
            # Recupera il colore del cluster (primo pixel della maschera)
            cluster_color = rgba[mask][0]
            
            # Calcola la luminanza (percorso standard RGB) per scegliere colore testo (Nero/Bianco)
            r, g, b = cluster_color[:3]
            luminance = 0.299*r + 0.587*g + 0.114*b
            text_color = 'white' if luminance < 0.5 else 'black'
            
            txt = ax.text(
                cx, cy, testo_etichetta,
                fontsize=LABEL_FONTSIZE,
                ha='center', va='center',
                color=text_color, zorder=4,
                bbox=dict(
                    boxstyle='round,pad=0.25',
                    fc=cluster_color,   # Sfondo = colore del cluster
                    ec='white',         # Bordo bianco per far risaltare il box
                    lw=0.5,
                    alpha=0.85          # Leggera trasparenza per intravedere i punti
                )
            )
            texts.append(txt)
            
        # 4. Applica adjust_text per evitare la sovrapposizione tra le etichette
        if texts:
            if HAS_ADJUST_TEXT:
                # Evita solo la collisione tra le scritte stesse per essere rapidissimo
                # e traccia una sottile linea fino al centroide se vengono spostate.
                adjust_text(
                    texts,
                    ax=ax,
                    arrowprops=dict(arrowstyle="-", color='gray', lw=0.6, alpha=0.8)
                )
            else:
                logging.warning("adjustText non è installato! Le label potrebbero sovrapporsi. Usa: pip install adjustText")
 
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
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(labelsize=7)

# ── Main ─────────────────────────────────────────────────────

def main():
    logging.info(f"Caricamento {UMAP_2D_FILE} ...")
    raw       = np.load(UMAP_2D_FILE, allow_pickle=True)
    xy        = raw['embeddings_2d']   
    ai_labels = raw['labels']          
    splits    = raw['splits']          

    if splits.dtype.kind == 'S':
        splits = np.array([s.decode() for s in splits])

    logging.info(f"Caricamento {HDBSCAN_LABELS_FILE} ...")
    hdb_labels = np.load(HDBSCAN_LABELS_FILE)  

    if SPLIT == "all":
        mask_split = np.ones(len(xy), dtype=bool)
        logging.info(f"Split='all': {len(xy):,} punti totali")
    else:
        mask_split = splits == SPLIT
        logging.info(f"Split='{SPLIT}': {mask_split.sum():,} / {len(xy):,} punti")

    xy_f  = xy[mask_split]
    hdb_f = hdb_labels[mask_split]
    ai_f  = ai_labels[mask_split]

    unique_clusters = sorted(set(hdb_f) - {-1})
    n_clusters      = len(unique_clusters)
    logging.info(f"Run {RUN_ID}: {n_clusters} cluster, "
                 f"rumore {(hdb_f == -1).mean():.1%}")

    remap    = {cid: i for i, cid in enumerate(unique_clusters)}
    hdb_norm = np.array([remap.get(c, -1) for c in hdb_f])

    palette = build_color_palette(n_clusters)
    rgba    = make_rgba_array(hdb_norm, palette)

    xlim, ylim = compute_limits(xy_f, percentile=0.5)
    logging.info(f"Limiti assi — X: {xlim[0]:.2f} / {xlim[1]:.2f} "
                 f"| Y: {ylim[0]:.2f} / {ylim[1]:.2f}")

    fig, axes = plt.subplots(1, 3, figsize=FIG_SIZE, dpi=DPI)
    fig.suptitle(
        f"HDBSCAN Run {RUN_ID}  —  Split: {SPLIT}",
        fontsize=13, fontweight='bold', y=1.01
    )
 
    scatter_panel(axes[0], xy_f, hdb_norm, rgba, "Tutti i punti", xlim, ylim)
 
    m0 = ai_f == 0
    scatter_panel(axes[1], xy_f[m0], hdb_norm[m0], rgba[m0],
                  LABEL_NAMES.get(0, "label=0"), xlim, ylim)
 
    m1 = ai_f == 1
    scatter_panel(axes[2], xy_f[m1], hdb_norm[m1], rgba[m1],
                  LABEL_NAMES.get(1, "label=1"), xlim, ylim)
 
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches='tight')
    logging.info(f"Salvato: {OUTPUT_FILE}")
 
 
if __name__ == "__main__":
    if not HAS_ADJUST_TEXT:
        logging.warning("Si consiglia di installare 'adjustText' per risolvere le sovrapposizioni delle scritte (pip install adjustText)")
    main()