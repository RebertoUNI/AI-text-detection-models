import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

# ==========================================
# Impostazioni e Parametri
# ==========================================
# Parametri per la scala degli assi nei plot (come richiesto nello Step 4)
AXIS_MAX_X = 8.0  # Modifica questo valore in base alla distribuzione della tua UMAP
AXIS_MAX_Y = 8.0  # Modifica questo valore in base alla distribuzione della tua UMAP
UMAP_FILE = "database exploration/umap/2d/full_dataset/umap_output/umap_full_dataset_2d.npz" # Assicurati di inserire il nome file corretto

# ==========================================
# Step 1 — Caricamento dati
# ==========================================
print("Caricamento dati in corso...")

# 1. Carica results.csv e tieni traccia delle righe valide
results_raw = pd.read_csv("database exploration/inferenza deberta (test split)/results.csv", usecols=['text', 'ground_truth', 'predicted_label'])

# Trova gli indici delle righe che NON sono vuote o nulle
valid_indices = results_raw[
    results_raw['text'].notna() & (results_raw['text'].astype(str).str.strip() != '')
].index.tolist()

# Applica il filtro al DataFrame e resetta l'indice
results_df = results_raw.loc[valid_indices].reset_index(drop=True)


# 2. NPZ UMAP - Carica e filtra prima per "test" e POI per le righe valide
umap_data = np.load(UMAP_FILE, allow_pickle=True)
embeddings_2d = umap_data['embeddings_2d']
splits = umap_data['splits']

# Estrai solo i punti di test
test_embeddings_all = embeddings_2d[splits == "test"]

# Filtra tenendo solo i punti corrispondenti alle righe valide del CSV
test_embeddings = test_embeddings_all[valid_indices]


# 3. HuggingFace dataset - Carica e filtra usando gli indici validi
hf_dataset = load_dataset("R-obi/ai-text-detection-pile-cleaned", split="test")

# Seleziona solo gli elementi del dataset HF che corrispondono alle righe valide
hf_dataset_filtered = hf_dataset.select(valid_indices)
cluster_ids = hf_dataset_filtered['cluster_id']


# Verifica finale di sicurezza
print(f"Righe totali dopo la pulizia: {len(results_df)}")
assert len(results_df) == len(test_embeddings) == len(cluster_ids), \
    f"Errore: Dimensioni disallineate! CSV: {len(results_df)}, UMAP: {len(test_embeddings)}, HF: {len(cluster_ids)}"

# ==========================================
# Step 2 — Allineamento e calcolo errori
# ==========================================
print("Allineamento dati e calcolo degli errori...")

df = pd.DataFrame({
    'umap_x': test_embeddings[:, 0],
    'umap_y': test_embeddings[:, 1],
    'ground_truth': results_df['ground_truth'],
    'predicted_label': results_df['predicted_label'],
    'cluster_id': cluster_ids,
    'text': results_df['text']
})

# Calcolo is_correct
df['is_correct'] = df['ground_truth'] == df['predicted_label']

# Definizione error_type
def determine_error_type(row):
    if row['is_correct']:
        return None
    # Adatta 'LABEL_0' e 'LABEL_1' se nel tuo CSV sono salvati come interi (0 e 1)
    if row['ground_truth'] == 'LABEL_0' and row['predicted_label'] == 'LABEL_1':
        return 'FP'
    elif row['ground_truth'] == 'LABEL_1' and row['predicted_label'] == 'LABEL_0':
        return 'FN'
    return 'UNKNOWN_ERROR'

df['error_type'] = df.apply(determine_error_type, axis=1)

# ==========================================
# Step 3 — Plot 1: tutti i punti
# ==========================================
print("Generazione Plot 1 (Tutti i punti)...")

plt.figure(figsize=(10, 8))
correct_mask = df['is_correct'] == True
error_mask = df['is_correct'] == False

# Punti corretti (Verde)
plt.scatter(df.loc[correct_mask, 'umap_x'], df.loc[correct_mask, 'umap_y'], 
            c='green', s=5, alpha=0.3, label='Corretto')

# Punti errati (Rosso)
plt.scatter(df.loc[error_mask, 'umap_x'], df.loc[error_mask, 'umap_y'], 
            c='red', s=5, alpha=0.6, label='Errore')

# Impostazione assi
if AXIS_MAX_X and AXIS_MAX_Y:
    plt.xlim(-AXIS_MAX_X, AXIS_MAX_X)
    plt.ylim(-AXIS_MAX_Y, AXIS_MAX_Y)

plt.title('UMAP 2D - Predizioni Corrette vs Errori')
plt.xlabel('UMAP X')
plt.ylabel('UMAP Y')
plt.legend()
plt.tight_layout()
plt.savefig('plot1_all_points.png', dpi=300)
plt.close()

# ==========================================
# Step 4 — Plot 2: solo gli errori per cluster
# ==========================================
print("Generazione Plot 2 (Errori per cluster)...")

errors_df = df[error_mask].copy()

plt.figure(figsize=(12, 10))
unique_clusters = sorted(errors_df['cluster_id'].unique())
cmap = plt.get_cmap('tab20') # Colormap categoriale

for i, cluster in enumerate(unique_clusters):
    cluster_data = errors_df[errors_df['cluster_id'] == cluster]
    
    # Grigio per il noise (cluster == -1), colori dalla colormap per gli altri
    color = 'gray' if cluster == -1 else cmap(i % 20)
    label = f"Noise (-1)" if cluster == -1 else f"Cluster {cluster}"
    
    plt.scatter(cluster_data['umap_x'], cluster_data['umap_y'], 
                c=[color], s=15, alpha=0.8, label=label)

# Impostazione assi (riutilizzo gli stessi limiti per coerenza visiva)
if AXIS_MAX_X and AXIS_MAX_Y:
    plt.xlim(-AXIS_MAX_X, AXIS_MAX_X)
    plt.ylim(-AXIS_MAX_Y, AXIS_MAX_Y)

plt.title('UMAP 2D - Distribuzione Errori per Cluster')
plt.xlabel('UMAP X')
plt.ylabel('UMAP Y')
# Posiziono la legenda fuori dal grafico per evitare che copra i punti se ci sono molti cluster
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2)
plt.tight_layout()
plt.savefig('plot2_errors_by_cluster.png', dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# Step 5 — File di output errori per cluster
# ==========================================
print("Salvataggio dei file CSV di output...")

# Aggiungo l'indice posizionale originale come richiesto
errors_df['sentence_index'] = errors_df.index

# File 1: Dettaglio errori
output_errors = errors_df[['cluster_id', 'error_type', 'sentence_index', 'text']]
output_errors.to_csv("dettaglio_errori_per_cluster.csv", index=False)

# File 2: Summary conteggi per cluster
# Raggruppo e conto i tipi di errore per ogni cluster
summary_df = errors_df.groupby('cluster_id')['error_type'].value_counts().unstack(fill_value=0)

# Gestisco l'eventualità in cui non ci siano affatto FP o FN nell'intero set
if 'FP' not in summary_df.columns:
    summary_df['FP'] = 0
if 'FN' not in summary_df.columns:
    summary_df['FN'] = 0

# Calcolo totale e rinomino colonne come da specifiche
summary_df['n_total_errors'] = summary_df['FP'] + summary_df['FN']
summary_df = summary_df.rename(columns={'FP': 'n_FP', 'FN': 'n_FN'})

# Riordino le colonne e resetto l'indice
summary_df = summary_df[['n_FP', 'n_FN', 'n_total_errors']].reset_index()
summary_df.to_csv("summary_errori_per_cluster.csv", index=False)

print("Elaborazione completata con successo! Trovi le immagini PNG e i file CSV nella directory corrente.")

# ==========================================
# Step 5 — Plot 5: Linee di livello (KDE) per Corretti vs Errori (Versione Migliorata)
# ==========================================
print("Generazione Plot 5 (Linee di livello KDE)...")

from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D

def _kde_grid(points, xlim, ylim, resolution=200):
    """Calcola la griglia KDE per i punti 2D."""
    x_min, x_max = xlim
    y_min, y_max = ylim
    
    # Crea griglia
    x_grid = np.linspace(x_min, x_max, resolution)
    y_grid = np.linspace(y_min, y_max, resolution)
    xx, yy = np.meshgrid(x_grid, y_grid)
    
    # Calcola KDE
    try:
        kde = gaussian_kde(points.T)
        positions = np.vstack([xx.ravel(), yy.ravel()])
        zz = kde(positions).reshape(xx.shape)
        return xx, yy, zz
    except Exception as e:
        print(f"Errore nel calcolo KDE: {e}")
        return None, None, None

# Prepara i dati per i contorni
correct_points = df[df['is_correct'] == True][['umap_x', 'umap_y']].values
error_points = df[df['is_correct'] == False][['umap_x', 'umap_y']].values

print(f"Punti corretti: {len(correct_points)}, Punti errati: {len(error_points)}")

# Crea il plot
fig, ax = plt.subplots(figsize=(10, 8))

# Parametri per i contorni
CONTOUR_COLORS = ['green', 'red']
LABELS = ['Corretto', 'Errore']

# Calcola e disegna i contorni per i punti corretti
if len(correct_points) >= 3:  # Almeno 3 punti per KDE
    xx, yy, zz = _kde_grid(correct_points, (-AXIS_MAX_X, AXIS_MAX_X), (-AXIS_MAX_Y, AXIS_MAX_Y), resolution=200)
    if xx is not None:
        # Usa livelli percentili per una migliore visualizzazione
        z_flat = zz.flatten()
        z_sorted = np.sort(z_flat[z_flat > 0])
        if len(z_sorted) > 0:
            # Livelli al 20%, 40%, 60%, 80% del massimo
            levels = np.linspace(z_sorted.min(), z_sorted.max(), 6)
            contour_correct = ax.contour(xx, yy, zz, 
                                        levels=levels,
                                        colors=CONTOUR_COLORS[0], 
                                        linewidths=1.5, 
                                        alpha=0.7)
            # Aggiungi etichette ai contorni
            ax.clabel(contour_correct, inline=True, fontsize=8, fmt='%1.0e')

# Calcola e disegna i contorni per i punti errati
if len(error_points) >= 3:
    xx, yy, zz = _kde_grid(error_points, (-AXIS_MAX_X, AXIS_MAX_X), (-AXIS_MAX_Y, AXIS_MAX_Y), resolution=200)
    if xx is not None:
        z_flat = zz.flatten()
        z_sorted = np.sort(z_flat[z_flat > 0])
        if len(z_sorted) > 0:
            levels = np.linspace(z_sorted.min(), z_sorted.max(), 6)
            contour_error = ax.contour(xx, yy, zz, 
                                      levels=levels,
                                      colors=CONTOUR_COLORS[1], 
                                      linewidths=1.5, 
                                      alpha=0.7)
            ax.clabel(contour_error, inline=True, fontsize=8, fmt='%1.0e')
else:
    print("Attenzione: Pochi punti errati per calcolare KDE. Mostro solo i punti.")
    # Se non ci sono abbastanza punti errati, mostra i punti come scatter
    if len(error_points) > 0:
        ax.scatter(error_points[:, 0], error_points[:, 1], 
                  c='red', s=20, alpha=0.8, label='Errori (punti)')

# Scatter plot dei punti (semi-trasparenti per contesto)
ax.scatter(df['umap_x'], df['umap_y'], 
          c='gray', s=1, alpha=0.1, label='_nolegend_')

# Legenda manuale
legend_handles = [
    Line2D([0], [0], color='green', linewidth=2, label='Corretto (KDE)'),
    Line2D([0], [0], color='red', linewidth=2, label='Errore (KDE)')
]
if len(error_points) < 3 and len(error_points) > 0:
    # Aggiungi legenda per i punti se non c'è KDE
    legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor='red', markersize=8, label='Errori (punti)'))

ax.legend(handles=legend_handles, fontsize=11, framealpha=0.8, loc='upper right')

# Impostazione assi
ax.set_xlim(-AXIS_MAX_X, AXIS_MAX_X)
ax.set_ylim(-AXIS_MAX_Y, AXIS_MAX_Y)

ax.set_title('Linee di livello KDE - Predizioni Corrette vs Errori', fontsize=13, fontweight="bold")
ax.set_xlabel('UMAP X', fontsize=11)
ax.set_ylabel('UMAP Y', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot5_kde_contours.png', dpi=300)
plt.close()

print("Plot 5 completato!")