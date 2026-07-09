import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 1. Caricamento dei dati
embeddings = np.load('umap_embeddings_2d.npy')
labels = np.load('hdbscan_best_labels.npy')

# 2. Configurazione del grafico
fig, ax = plt.subplots(figsize=(12, 10), dpi=300)

# Separiamo il rumore (-1) dai cluster reali
is_noise = labels == -1
is_cluster = labels != -1

# 3. Disegniamo prima il rumore di fondo (in grigio chiaro)
ax.scatter(
    embeddings[is_noise, 0], 
    embeddings[is_noise, 1], 
    c='lightgray', 
    s=0.05, 
    label='Rumore (-1)',
    alpha=0.2
)

# 4. Estrazione dei cluster unici e calcolo del loro numero
unique_labels = np.unique(labels[is_cluster])
num_clusters = len(unique_labels)

# Creazione della colormap casuale per il massimo contrasto
base_cmap = plt.colormaps['turbo']
color_list = base_cmap(np.linspace(0, 1, num_clusters))
np.random.seed(42)  # Seed fisso per riproducibilità dei colori
np.random.shuffle(color_list)
shuffled_cmap = mcolors.ListedColormap(color_list)

# Mappatura dei label in indici puliti (0, 1, 2...)
_, cluster_indices = np.unique(labels[is_cluster], return_inverse=True)

# 5. Disegniamo i cluster reali
scatter = ax.scatter(
    embeddings[is_cluster, 0], 
    embeddings[is_cluster, 1], 
    c=cluster_indices, 
    cmap=shuffled_cmap, 
    s=0.1, 
    alpha=0.7, 
)

# =====================================================================
# NUOVA SEZIONE: AGGIUNTA DEI NUMERI DEI CLUSTER SULLA MAPPA
# =====================================================================
for cluster_id in unique_labels:
    # Prendiamo solo le coordinate dei punti che appartengono a questo cluster
    points_in_cluster = embeddings[labels == cluster_id]
    
    # Calcoliamo la mediana (più robusta della media contro eventuali punti isolati)
    # per trovare il "centro geometrico" del cluster
    centroid_x = np.median(points_in_cluster[:, 0])
    centroid_y = np.median(points_in_cluster[:, 1])
    
    # Scriviamo il numero del cluster nel centroide calcolato
    ax.text(
        centroid_x, centroid_y, 
        str(cluster_id), 
        fontsize=9, 
        fontweight='bold',
        color='black',
        ha='center',  # Allinea il testo perfettamente al centro in orizzontale
        va='center',  # Allinea il testo perfettamente al centro in verticale
        # Creiamo un "badge" bianco semitrasparente per far risaltare il testo
        bbox=dict(
            boxstyle='round,pad=0.2', 
            facecolor='white', 
            edgecolor='none', 
            alpha=0.75
        )
    )
# =====================================================================

# 6. Dettagli estetici finali e salvataggio
ax.set_title(f'Visualizzazione UMAP con etichette sui Cluster ({num_clusters} Cluster)', fontsize=14, pad=15)
ax.set_xlabel('UMAP Dimensione 1', fontsize=11)
ax.set_ylabel('UMAP Dimensione 2', fontsize=11)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper right', markerscale=10)

output_filename = 'umap_hdbscan_labeled.png'
fig.savefig(output_filename, bbox_inches='tight', dpi=300)

print(f"Grafico con etichette generato e salvato come '{output_filename}'")