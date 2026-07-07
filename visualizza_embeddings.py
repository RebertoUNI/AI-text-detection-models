"""
Visualizza gli embedding Qwen (0.6B) del dataset R-obi/ai-text-detection
per capire se le classi "umano" vs "AI" sono già separabili nello spazio
di embedding, prima di allenare una rete neurale sopra.

Installazione:
    pip install huggingface_hub numpy umap-learn matplotlib scikit-learn

Uso:
    python visualizza_embeddings.py --n_chunks 5 --max_points 20000
"""

import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "R-obi/ai-text-detection"
REPO_TYPE = "dataset"


def discover_chunk_ids():
    """Elenca tutti i file nel repo e ricava gli id dei chunk disponibili
    (basandosi sui file train__emb__chunkXXXXX.npy presenti davvero)."""
    files = list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)
    pattern = re.compile(r"train/train__emb__chunk(\d+)\.npy")
    ids = sorted(
        m.group(1) for f in files if (m := pattern.match(f))
    )
    return ids


def load_chunks(n_chunks):
    """Scarica e concatena n_chunks coppie (embedding, label) dal repo HF.
    Se n_chunks è None, scarica TUTTI i chunk disponibili."""
    all_ids = discover_chunk_ids()
    print(f"Trovati {len(all_ids)} chunk totali nel repo "
          f"({len(all_ids) * 1000} embedding totali circa).")

    ids = all_ids if n_chunks is None else all_ids[:n_chunks]

    embs, labs = [], []
    for idx in ids:
        emb_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=f"train/train__emb__chunk{idx}.npy",
        )
        lab_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=f"train/train__lab__chunk{idx}.npy",
        )
        e = np.load(emb_path)
        l = np.load(lab_path)
        print(f"chunk {idx}: emb shape={e.shape}, lab shape={l.shape}, "
              f"unique labels={np.unique(l)}")
        embs.append(e)
        labs.append(l)

    X = np.concatenate(embs, axis=0)
    y = np.concatenate(labs, axis=0).ravel()
    return X, y


def subsample(X, y, max_points: int, seed: int = 42):
    if X.shape[0] <= max_points:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=max_points, replace=False)
    return X[idx], y[idx]


def plot_projection(coords, y, title, out_path):
    plt.figure(figsize=(8, 6))
    for label, name, color in [(0, "Umano", "#1f77b4"), (1, "AI", "#d62728")]:
        mask = y == label
        plt.scatter(
            coords[mask, 0], coords[mask, 1],
            s=4, alpha=0.5, label=name, color=color,
        )
    plt.title(title)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.legend(markerscale=4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Salvato: {out_path}")
    plt.close()


def plot_density(coords, y, title, out_path, gridsize=60):
    """
    Genera una figura a 3 pannelli:
      1) densità (hexbin) dei campioni classe "Umano"
      2) densità (hexbin) dei campioni classe "AI"
      3) contorni di densità delle due classi sovrapposti, per confronto diretto

    Utile con molti punti, dove lo scatter plot classico diventa illeggibile
    per overplotting.
    """
    from scipy.stats import gaussian_kde

    x, yc = coords[:, 0], coords[:, 1]
    xlim = (x.min() - 0.05 * np.ptp(x), x.max() + 0.05 * np.ptp(x))
    ylim = (yc.min() - 0.05 * np.ptp(yc), yc.max() + 0.05 * np.ptp(yc))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # --- Pannelli 1-2: hexbin di densità per singola classe ---
    for ax, label, name, cmap in [
        (axes[0], 0, "Umano", "Blues"),
        (axes[1], 1, "AI", "Reds"),
    ]:
        mask = y == label
        hb = ax.hexbin(
            x[mask], yc[mask], gridsize=gridsize, cmap=cmap,
            mincnt=1, extent=(*xlim, *ylim),
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"Densità - {name} (n={mask.sum()})")
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
        fig.colorbar(hb, ax=ax, label="n. campioni")

    # --- Pannello 3: contorni di densità (KDE) sovrapposti, per confronto ---
    ax = axes[2]
    xx, yy = np.mgrid[xlim[0]:xlim[1]:100j, ylim[0]:ylim[1]:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])

    for label, name, color in [(0, "Umano", "#1f77b4"), (1, "AI", "#d62728")]:
        mask = y == label
        pts = coords[mask]
        # subsample per velocità della KDE se ci sono troppi punti
        if pts.shape[0] > 5000:
            idx = np.random.default_rng(0).choice(pts.shape[0], 5000, replace=False)
            pts = pts[idx]
        kde = gaussian_kde(pts.T)
        density = np.reshape(kde(positions).T, xx.shape)
        ax.contour(xx, yy, density, levels=6, colors=color, linewidths=1.5)
        ax.plot([], [], color=color, label=name)  # per la legenda

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title("Contorni di densità sovrapposti")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.legend()

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Salvato: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_chunks", type=str, default="5",
                         help="quanti chunk (da 1000 embedding l'uno) scaricare, "
                              "oppure 'all' per scaricarli tutti")
    parser.add_argument("--max_points", type=int, default=20000,
                         help="max punti da plottare (per velocità)")
    parser.add_argument("--umap_neighbors", type=int, default=30)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    args = parser.parse_args()

    n_chunks = None if args.n_chunks.lower() == "all" else int(args.n_chunks)

    print("Caricamento chunk...")
    X, y = load_chunks(n_chunks)
    print(f"Totale embedding caricati: {X.shape}, label: {y.shape}")

    X, y = subsample(X, y, args.max_points)
    print(f"Dopo subsampling: {X.shape}")

    # --- PCA: veloce, dà un'idea preliminare di separabilità lineare ---
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    coords_pca = pca.fit_transform(X)
    print(f"Varianza spiegata dalle prime 2 PC: {pca.explained_variance_ratio_.sum():.3f}")
    plot_projection(coords_pca, y, "PCA (2D) - Umano vs AI", "pca_projection.png")
    plot_density(coords_pca, y, "PCA (2D) - Densità per classe", "pca_density.png")

    # --- UMAP: cattura strutture non lineari, di solito più informativo ---
    import umap
    reducer = umap.UMAP(
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        n_components=2,
        metric="cosine",  # gli embedding Qwen si confrontano bene con coseno
        random_state=42,
    )
    coords_umap = reducer.fit_transform(X)
    plot_projection(coords_umap, y, "UMAP (2D) - Umano vs AI", "umap_projection.png")
    plot_density(coords_umap, y, "UMAP (2D) - Densità per classe", "umap_density.png")

    # --- Diagnostica quantitativa extra: quanto sono separabili? ---
    # Un classificatore lineare semplice sugli embedding originali (non ridotti)
    # dà un'idea concreta di quanto sarà facile il task per la rete neurale.
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    pred = clf.predict(Xte)
    print(f"\n[Diagnostica] Logistic Regression su embedding grezzi (768/1024-dim):")
    print(f"  Accuracy: {accuracy_score(yte, pred):.4f}")
    print(f"  AUC:      {roc_auc_score(yte, proba):.4f}")
    print("Se AUC è già alta (es. >0.9), la rete neurale probabilmente")
    print("andrà molto bene. Se è vicina a 0.5, le classi si sovrappongono")
    print("molto nello spazio di embedding e servirà un modello più capace")
    print("o feature/embedding diversi.")


if __name__ == "__main__":
    main()
