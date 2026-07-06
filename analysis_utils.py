"""
analysis_utils.py
─────────────────────────────────────────────────────────────────────────────
Funzioni di analisi model-agnostic (portate dal notebook 'analisi_separazione
_modelli_drive'), riusate identiche per tutti e 4 i modelli così il confronto
è omogeneo. Differenza principale rispetto al notebook originale:

  - Nessun mount Drive / autodetect cartella: si lavora su file locali
    (results/, embeddings {MODELLO}/), prodotti dagli script di test/training.
  - I testi allineati a y_true/y_prob si recuperano direttamente dallo split
    'test' del dataset grezzo (stesso ordine, perché nessuno script di test
    fa shuffle) invece di ricostruire seed/shuffle come nel notebook.
  - Le figure Plotly vengono sempre salvate come HTML (oltre a un tentativo
    di .show()), così funzionano anche in un job HPC headless o da terminale.
"""

import os
import glob
import json
import logging

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, gaussian_kde
from sklearn.metrics import roc_auc_score
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

DATASET_NAME = "srikanthgali/ai-text-detection-pile-cleaned"
_TEXTS_CACHE = {}   # evita di ricaricare il dataset grezzo più volte nello stesso processo


# ─────────────────────────────────────────────────────────────────────────
# I/O: predizioni di test, testi allineati, embeddings
# ─────────────────────────────────────────────────────────────────────────
def load_test_predictions(model_name: str, results_dir: str = "results"):
    """Carica y_true, y_prob salvati da test_{model}.py (results/{model}_test_predictions.npz)."""
    path = os.path.join(results_dir, f"{model_name}_test_predictions.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Non trovo {path}: esegui prima test_{model_name.lower()}.py")
    data = np.load(path)
    return data["y_true"], data["y_prob"]


def load_test_texts(split: str = "test"):
    """
    Ritorna i testi grezzi dello split richiesto, nello stesso ordine con cui
    sono stati valutati dagli script test_*.py (nessuno shuffla il test set).
    Cachato in memoria di processo per non ricaricare il dataset 4 volte.
    """
    if split in _TEXTS_CACHE:
        return _TEXTS_CACHE[split]

    from datasets import load_dataset
    logger.info(f"Carico testi grezzi per lo split '{split}' (una tantum)...")
    dataset = load_dataset(DATASET_NAME)
    texts = list(dataset[split]["text"])
    _TEXTS_CACHE[split] = texts
    return texts


def load_embeddings_shards(embeddings_dir: str, split: str = "test", max_shards=None):
    """
    Ricompone gli embedding salvati a shard da train_utils.extract_and_save_embeddings*
    ({split}_shard_NNN.npz, con dentro 'embeddings' e 'labels').
    """
    shard_files = sorted(glob.glob(os.path.join(embeddings_dir, f"{split}_shard_*.npz")))
    if not shard_files:
        raise FileNotFoundError(f"Nessuno shard '{split}_shard_*.npz' trovato in {embeddings_dir}")
    if max_shards:
        shard_files = shard_files[:max_shards]

    embs, labs = [], []
    for f in shard_files:
        d = np.load(f)
        embs.append(d["embeddings"])
        labs.append(d["labels"])
    X = np.concatenate(embs, axis=0)
    y = np.concatenate(labs, axis=0)
    logger.info(f"Embeddings caricati da {embeddings_dir}: {X.shape}, labels: {y.shape}")
    return X, y


def save_and_maybe_show(fig, path):
    """Salva sempre come HTML interattivo; prova anche .show() se c'è un display disponibile."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn")
    logger.info(f"Salvato: {path}")
    try:
        fig.show()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────
# 1. Separazione quantitativa (AUC, KS, overlap, Bhattacharyya)
# ─────────────────────────────────────────────────────────────────────────
def quantifica_separazione(y_true, y_prob, n_grid=512):
    p_human = np.asarray(y_prob)[np.asarray(y_true) == 0]
    p_ai    = np.asarray(y_prob)[np.asarray(y_true) == 1]

    auc = roc_auc_score(y_true, y_prob)
    ks_stat, ks_pval = ks_2samp(p_human, p_ai)

    grid = np.linspace(0, 1, n_grid)
    kde_h = gaussian_kde(p_human)(grid); kde_h /= kde_h.sum()
    kde_a = gaussian_kde(p_ai)(grid);    kde_a /= kde_a.sum()
    overlap = np.minimum(kde_h, kde_a).sum()
    bc = np.sum(np.sqrt(kde_h * kde_a))
    bhattacharyya_dist = -np.log(bc + 1e-12)

    risultati = {
        "AUC-ROC": float(auc),
        "KS statistic": float(ks_stat),
        "KS p-value": float(ks_pval),
        "Overlap coefficient (0=separate, 1=identiche)": float(overlap),
        "Bhattacharyya distance (alto=separate)": float(bhattacharyya_dist),
    }
    for k, v in risultati.items():
        logger.info(f"{k:55s}: {v:.4f}")
    return risultati, grid, kde_h, kde_a


def plot_densita_sovrapposte(grid, kde_h, kde_a, model_name="Modello"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=kde_h, fill='tozeroy', name='Human',
                              line=dict(color='royalblue')))
    fig.add_trace(go.Scatter(x=grid, y=kde_a, fill='tozeroy', name='AI',
                              line=dict(color='indianred'), opacity=0.6))
    fig.update_layout(title=f"Densità predizioni — {model_name}",
                       xaxis_title="P(AI)", yaxis_title="densità (norm.)")
    return fig


# ─────────────────────────────────────────────────────────────────────────
# 2. Predizioni vs ground truth
# ─────────────────────────────────────────────────────────────────────────
def plot_predizioni_vs_ground_truth(y_true, y_prob, texts=None, model_name="Modello",
                                     max_points=3000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    idx = rng.choice(n, size=min(max_points, n), replace=False)

    if texts is None:
        texts = [f"idx_{i}" for i in range(n)]
    else:
        texts = list(texts)

    df = pd.DataFrame({
        "ground_truth": np.array(y_true)[idx],
        "prob_ai": np.array(y_prob)[idx],
        "text": [str(texts[int(i)])[:200] for i in idx],
    })
    df["confidenza"] = np.abs(df["prob_ai"] - 0.5) * 2
    df["classe"] = df["ground_truth"].map({0: "Human", 1: "AI"})
    df["corretto"] = ((df["prob_ai"] > 0.5).astype(int) == df["ground_truth"])
    df["x_jitter"] = df["ground_truth"] + rng.uniform(-0.15, 0.15, size=len(df))

    fig = px.scatter(
        df, x="x_jitter", y="prob_ai", color="confidenza",
        symbol="corretto", hover_data=["text", "classe", "corretto"],
        color_continuous_scale="RdYlGn",
        title=f"Predizioni vs Ground Truth — {model_name}",
        labels={"x_jitter": "Classe vera (0=Human, 1=AI)", "prob_ai": "P(AI) predetta"},
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
    fig.update_xaxes(tickvals=[0, 1], ticktext=["Human", "AI"])
    return fig, df


# ─────────────────────────────────────────────────────────────────────────
# 3. Atomografia del dataset (prob vs lunghezza testo)
# ─────────────────────────────────────────────────────────────────────────
def atomografia_dataset(y_true, y_prob, texts=None, lunghezze=None, model_name="Modello",
                         max_points=3000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    idx = rng.choice(n, size=min(max_points, n), replace=False)

    if texts is None:
        texts = [f"idx_{i}" for i in range(n)]
    else:
        texts = list(texts)
    if lunghezze is None:
        lunghezze = [len(str(t).split()) for t in texts]

    df = pd.DataFrame({
        "prob_ai": np.array(y_prob)[idx],
        "ground_truth": np.array(y_true)[idx],
        "lunghezza": np.array(lunghezze)[idx],
        "text": [str(texts[int(i)])[:200] for i in idx],
    })
    df["classe"] = df["ground_truth"].map({0: "Human", 1: "AI"})
    df["esito"] = np.where((df["prob_ai"] > 0.5).astype(int) == df["ground_truth"],
                            "corretto", "errore")

    fig = px.scatter(
        df, x="lunghezza", y="prob_ai", color="esito", symbol="classe",
        hover_data=["text"], opacity=0.6,
        title=f"Atomografia dataset — {model_name}",
        labels={"lunghezza": "N° parole", "prob_ai": "P(AI) predetta"},
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
    return fig, df


# ─────────────────────────────────────────────────────────────────────────
# 4. Cluster degli embedding
# ─────────────────────────────────────────────────────────────────────────
def plot_cluster_embeddings(X, y, metodo="pca", model_name="Modello", max_points=5000, seed=42):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.choice(n, size=min(max_points, n), replace=False)
    X_sub, y_sub = X[idx], y[idx]

    if metodo == "pca":
        from sklearn.decomposition import PCA
        proj = PCA(n_components=2, random_state=seed).fit_transform(X_sub)
    elif metodo == "umap":
        import umap  # pip install umap-learn
        proj = umap.UMAP(n_components=2, random_state=seed).fit_transform(X_sub)
    else:
        raise ValueError("metodo deve essere 'pca' o 'umap'")

    df = pd.DataFrame({"x": proj[:, 0], "y": proj[:, 1],
                        "classe": np.where(y_sub == 1, "AI", "Human")})
    fig = px.scatter(df, x="x", y="y", color="classe", opacity=0.5,
                      title=f"{metodo.upper()} degli embedding — {model_name}")
    return fig, proj


# ─────────────────────────────────────────────────────────────────────────
# 5. Occlusione parola per parola (model-agnostic: serve solo predict_fn)
# ─────────────────────────────────────────────────────────────────────────
def occlusion_importance(text, predict_fn, max_words=80):
    parole = text.split()[:max_words]
    if not parole:
        return []
    testo_troncato = " ".join(parole)
    base_prob = predict_fn([testo_troncato])[0]
    varianti = [" ".join(parole[:i] + parole[i + 1:]) for i in range(len(parole))]
    probs_senza_parola = predict_fn(varianti)
    importanze = base_prob - np.array(probs_senza_parola)
    return list(zip(parole, importanze))


def plot_heatmap_importanza(parole_importanze, model_name="Modello"):
    if not parole_importanze:
        return None
    parole, imp = zip(*parole_importanze)
    imp = np.array(imp)
    x_indices = [f"{i}_{p}" for i, p in enumerate(parole)]
    vmax = max(abs(imp).max(), 1e-8)

    fig = go.Figure(data=go.Heatmap(
        z=[imp], x=x_indices, text=[list(parole)], texttemplate="%{text}",
        textfont={"size": 13, "color": "black"},
        colorscale="RdBu", reversescale=True, zmin=-vmax, zmax=vmax, showscale=True,
        colorbar=dict(title="Δ P(AI)", orientation="h", y=-0.6),
    ))
    fig.update_layout(
        title=f"Parole che spingono verso AI (rosso) o Human (blu) — {model_name}",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, fixedrange=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        paper_bgcolor='white', plot_bgcolor='white', height=220,
        margin=dict(l=10, r=10, t=60, b=80),
    )
    return fig


def analizza_vocabolario_globale(testi, predict_fn, n_frasi=100, max_words=50, top_n=10):
    from collections import defaultdict

    impatto_globale = defaultdict(list)
    campione_testi = testi[:min(n_frasi, len(testi))]
    logger.info(f"Avvio analisi globale su {len(campione_testi)} frasi...")

    for idx, testo in enumerate(campione_testi):
        if idx % 10 == 0 and idx > 0:
            logger.info(f"Elaborate {idx}/{len(campione_testi)} frasi...")

        parole = testo.split()[:max_words]
        if not parole:
            continue
        testo_troncato = " ".join(parole)
        try:
            base_prob = predict_fn([testo_troncato])[0]
            varianti = [" ".join(parole[:i] + parole[i + 1:]) for i in range(len(parole))]
            probs_senza_parola = predict_fn(varianti)
            importanze = base_prob - np.array(probs_senza_parola)
            for parola, imp in zip(parole, importanze):
                parola_clean = parola.strip(".,;:!?\"'()").lower()
                if parola_clean:
                    impatto_globale[parola_clean].append(imp)
        except Exception:
            continue

    dati_aggregati = [
        {"Parola": p, "Impatto_Medio": np.mean(v), "Frequenza": len(v)}
        for p, v in impatto_globale.items()
    ]
    df_global = pd.DataFrame(dati_aggregati)
    df_global = df_global[df_global["Frequenza"] > 1]
    if df_global.empty:
        return None

    df_ai = df_global.sort_values(by="Impatto_Medio", ascending=False).head(top_n).copy()
    df_ai["Predilezione Modello"] = "AI"
    df_human = df_global.sort_values(by="Impatto_Medio", ascending=True).head(top_n).copy()
    df_human["Predilezione Modello"] = "Human"
    df_human["Impatto_Medio"] = df_human["Impatto_Medio"].abs()

    tabella_finale = pd.concat([df_ai, df_human]).reset_index(drop=True)
    tabella_finale = tabella_finale[["Parola", "Impatto_Medio", "Frequenza", "Predilezione Modello"]]
    tabella_finale.columns = ["Parola", "Impatto Medio (Δ P)", "Frequenza nel Campione", "Predilezione Modello"]
    return tabella_finale


# ─────────────────────────────────────────────────────────────────────────
# 6. Allineamento embeddings <-> predizioni (utile per colorare per confidenza)
# ─────────────────────────────────────────────────────────────────────────
def load_aligned_embeddings_and_predictions(model_name, embeddings_dir, results_dir="results", split="test"):
    """
    Carica embeddings (X, y) e predizioni (y_true, y_prob) per lo stesso
    split e li allinea per indice (entrambi gli script che li generano non
    fanno shuffle sul test set, quindi l'ordine dovrebbe già combaciare).
    Se le lunghezze non combaciano, tronca al minimo e avvisa.
    """
    X, y_emb = load_embeddings_shards(embeddings_dir, split=split)
    y_true, y_prob = load_test_predictions(model_name, results_dir=results_dir)

    n = min(len(X), len(y_true))
    if len(X) != len(y_true):
        logger.warning(
            f"Lunghezze diverse tra embeddings ({len(X)}) e predizioni ({len(y_true)}) per "
            f"{model_name}: tronco a {n} elementi. Verifica che train_*.py e test_*.py abbiano "
            f"processato lo stesso split '{split}' senza shuffle."
        )
    return X[:n], y_true[:n], y_prob[:n]


# ─────────────────────────────────────────────────────────────────────────
# 7. Geometria dei cluster: centroidi, diametro, Silhouette, Davies-Bouldin, asse AI
# ─────────────────────────────────────────────────────────────────────────
def analizza_geometria_cluster(X, y):
    """
    Quantifica numericamente quanto le due nuvole (Human/AI) sono separate:
      - distanza tra i centroidi
      - diametro medio di ciascuna nuvola (dev. media dei punti dal proprio centroide)
      - rapporto distanza/diametro (se >> 1, le classi sono ben separate)
      - Silhouette score e Davies-Bouldin index (metriche standard di clustering)
      - l'asse "Human -> AI" nello spazio degli embedding (per proiettare nuove frasi)
    """
    from sklearn.metrics import silhouette_score, davies_bouldin_score

    X = np.asarray(X)
    y = np.asarray(y)
    X_h, X_a = X[y == 0], X[y == 1]

    mu_h = X_h.mean(axis=0)
    mu_a = X_a.mean(axis=0)
    d_centroids = float(np.linalg.norm(mu_h - mu_a))

    diam_h = float(np.mean(np.linalg.norm(X_h - mu_h, axis=1))) if len(X_h) else float("nan")
    diam_a = float(np.mean(np.linalg.norm(X_a - mu_a, axis=1))) if len(X_a) else float("nan")
    diam_medio = np.nanmean([diam_h, diam_a])

    n_classi = len(np.unique(y))
    sil = float(silhouette_score(X, y)) if n_classi > 1 and len(X) > n_classi else float("nan")
    db  = float(davies_bouldin_score(X, y)) if n_classi > 1 else float("nan")

    ai_axis = mu_a - mu_h
    ai_axis_unit = ai_axis / (np.linalg.norm(ai_axis) + 1e-12)

    risultati = {
        "distanza_centroidi": d_centroids,
        "diametro_human": diam_h,
        "diametro_ai": diam_a,
        "diametro_medio": float(diam_medio),
        "rapporto_distanza_diametro": float(d_centroids / (diam_medio + 1e-12)),
        "silhouette_score": sil,
        "davies_bouldin_index": db,
    }
    for k, v in risultati.items():
        logger.info(f"{k:30s}: {v:.4f}" if isinstance(v, float) and not np.isnan(v) else f"{k:30s}: {v}")
    return risultati, mu_h, mu_a, ai_axis_unit


def proietta_su_asse_ai(X, mu_h, ai_axis_unit):
    """
    Proiezione (prodotto scalare) di ogni embedding sull'asse Human->AI:
    un punteggio scalare di "AI-ness" per ogni frase, riusabile su nuovi testi.
    """
    X = np.asarray(X)
    return (X - mu_h) @ ai_axis_unit


def plot_proiezione_asse_ai(scores, y, model_name="Modello"):
    df = pd.DataFrame({"score": np.asarray(scores),
                        "classe": np.where(np.asarray(y) == 1, "AI", "Human")})
    fig = px.histogram(df, x="score", color="classe", barmode="overlay", opacity=0.6,
                        title=f"Proiezione sull'asse Human→AI — {model_name}",
                        labels={"score": "Proiezione (punteggio di AI-ness)"})
    return fig


# ─────────────────────────────────────────────────────────────────────────
# 8. Tomografia semantica: sottogruppi non supervisionati (K-Means/HDBSCAN)
# ─────────────────────────────────────────────────────────────────────────
def analizza_sottogruppi_semantici(X, y, n_clusters=10, metodo="kmeans", seed=42):
    """
    Clusterizza gli embedding SENZA usare le label Human/AI (solo la
    geometria semantica: i cluster si formeranno per contesto/argomento).
    Poi, dentro ciascun sottogruppo, guarda come si comportano le due classi:
    in alcuni contesti saranno ben separate, in altri sovrapposte.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if metodo == "kmeans":
        from sklearn.cluster import KMeans
        cluster_labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(X)
    elif metodo == "hdbscan":
        import hdbscan   # pip install hdbscan
        cluster_labels = hdbscan.HDBSCAN(min_cluster_size=max(10, len(X) // 100)).fit_predict(X)
    else:
        raise ValueError("metodo deve essere 'kmeans' o 'hdbscan'")

    righe = []
    for c in sorted(set(cluster_labels)):
        mask = cluster_labels == c
        n_tot = int(mask.sum())
        n_ai = int((y[mask] == 1).sum())
        n_h  = int((y[mask] == 0).sum())
        prop_ai = n_ai / n_tot if n_tot else float("nan")

        d_locale = float("nan")
        if n_ai > 1 and n_h > 1:
            mu_h_c = X[mask & (y == 0)].mean(axis=0)
            mu_a_c = X[mask & (y == 1)].mean(axis=0)
            d_locale = float(np.linalg.norm(mu_h_c - mu_a_c))

        righe.append({
            "cluster": int(c), "n_totale": n_tot,
            "n_human": n_h, "n_ai": n_ai, "proporzione_ai": prop_ai,
            "distanza_centroidi_locale": d_locale,
        })

    df = pd.DataFrame(righe).sort_values("n_totale", ascending=False).reset_index(drop=True)
    return df, cluster_labels


def plot_sottogruppi_semantici(X, y, cluster_labels, model_name="Modello", metodo_proiezione="pca", seed=42):
    """Proiezione 2D colorata per sottogruppo semantico, con simbolo per Human/AI."""
    if metodo_proiezione == "pca":
        from sklearn.decomposition import PCA
        proj = PCA(n_components=2, random_state=seed).fit_transform(X)
    elif metodo_proiezione == "umap":
        import umap
        proj = umap.UMAP(n_components=2, random_state=seed).fit_transform(X)
    else:
        raise ValueError("metodo_proiezione deve essere 'pca' o 'umap'")

    df = pd.DataFrame({
        "x": proj[:, 0], "y": proj[:, 1],
        "sottogruppo": [str(c) for c in cluster_labels],
        "classe": np.where(np.asarray(y) == 1, "AI", "Human"),
    })
    fig = px.scatter(df, x="x", y="y", color="sottogruppo", symbol="classe", opacity=0.6,
                      title=f"Sottogruppi semantici ({metodo_proiezione.upper()}) — {model_name}")
    return fig


# ─────────────────────────────────────────────────────────────────────────
# 9. Densità 2D (mappa topografica) e scatter colorato per confidenza
# ─────────────────────────────────────────────────────────────────────────
def plot_densita_2d(X, y, metodo="pca", model_name="Modello", seed=42):
    """
    Curve di livello (stile mappa topografica) delle due classi sulla
    proiezione 2D degli embedding: se i picchi di densità di Human e AI non
    coincidono, è la prova visiva+quantitativa che i due profili sono diversi.
    """
    if metodo == "pca":
        from sklearn.decomposition import PCA
        proj = PCA(n_components=2, random_state=seed).fit_transform(X)
    elif metodo == "umap":
        import umap
        proj = umap.UMAP(n_components=2, random_state=seed).fit_transform(X)
    else:
        raise ValueError("metodo deve essere 'pca' o 'umap'")

    y = np.asarray(y)
    fig = go.Figure()
    for label, nome, colore in [(0, "Human", "Blues"), (1, "AI", "Reds")]:
        pts = proj[y == label]
        if len(pts) < 5:
            continue
        fig.add_trace(go.Histogram2dContour(
            x=pts[:, 0], y=pts[:, 1], name=nome, colorscale=colore,
            showscale=False, opacity=0.55, contours=dict(coloring="lines"),
            line=dict(width=2),
        ))
    fig.update_layout(title=f"Densità 2D — {model_name}", xaxis_title="dim 1", yaxis_title="dim 2")
    return fig


def plot_cluster_embeddings_confidence(X, y, y_prob, metodo="pca", model_name="Modello",
                                        max_points=5000, seed=42):
    """
    Come plot_cluster_embeddings, ma l'intensità del colore rappresenta la
    CONFIDENZA del classificatore (|P(AI) - 0.5| * 2): punti intensi = modello
    sicuro, punti sbiaditi = vicino al decision boundary (il modello "esita").
    Richiede X, y, y_prob allineati (vedi load_aligned_embeddings_and_predictions).
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.choice(n, size=min(max_points, n), replace=False)
    X_sub, y_sub, p_sub = X[idx], np.asarray(y)[idx], np.asarray(y_prob)[idx]

    if metodo == "pca":
        from sklearn.decomposition import PCA
        proj = PCA(n_components=2, random_state=seed).fit_transform(X_sub)
    elif metodo == "umap":
        import umap
        proj = umap.UMAP(n_components=2, random_state=seed).fit_transform(X_sub)
    else:
        raise ValueError("metodo deve essere 'pca' o 'umap'")

    confidenza = np.abs(p_sub - 0.5) * 2
    df = pd.DataFrame({"x": proj[:, 0], "y": proj[:, 1],
                        "classe": np.where(y_sub == 1, "AI", "Human"),
                        "confidenza": confidenza, "P(AI)": p_sub})
    fig = px.scatter(df, x="x", y="y", color="confidenza", symbol="classe",
                      color_continuous_scale="Viridis", opacity=0.7, hover_data=["P(AI)"],
                      title=f"{metodo.upper()} colorato per confidenza — {model_name}")
    return fig