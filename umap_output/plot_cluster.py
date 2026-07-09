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
    alpha=0.3
)

# 4. Estrazione dei cluster unici e calcolo del loro numero
unique_labels = np.unique(labels[is_cluster])
num_clusters = len(unique_labels)

# =====================================================================
# CREAZIONE DI UNA COLORMAP CASUALE PER MASSIMIZZARE IL CONTRASTO
# =====================================================================
# Scegliamo una colormap di base molto ricca di sfumature (es. 'turbo' o 'hsv')
base_cmap = plt.colormaps['turbo']

# Campioniamo N colori equispaziati lungo tutta la colormap
color_list = base_cmap(np.linspace(0, 1, num_clusters))

# Mescoliamo i colori in modo casuale.
# Usiamo un seed fisso (es. 42) per fare in modo che il grafico sia riproducibile.
# Se non ti piacciono i colori estratti, basta cambiare questo numero!
np.random.seed(42)  
np.random.shuffle(color_list)

# Creiamo una nuova colormap discreta con i colori rimescolati
shuffled_cmap = mcolors.ListedColormap(color_list)

# Mappiamo i label originali in indici sequenziali da 0 a N-1.
# Questo garantisce una perfetta corrispondenza 1:1 con la colormap,
# anche se nel dataset originale dovessero mancare dei numeri di cluster.
_, cluster_indices = np.unique(labels[is_cluster], return_inverse=True)
# =====================================================================

# 5. Disegniamo i cluster con la mappa di colori rimescolata
scatter = ax.scatter(
    embeddings[is_cluster, 0], 
    embeddings[is_cluster, 1], 
    c=cluster_indices,      # Usiamo gli indici mappati (0, 1, 2...)
    cmap=shuffled_cmap, 
    s=0.1,                  # Dimensione minuscola per gestire 500k punti
    alpha=0.8,              # Trasparenza per evidenziare le sovrapposizioni
)

# 6. Gestione della Legenda / Colorbar
# Se hai pochi cluster (< 30), la colorbar è utile e leggibile.
# Se ne hai centinaia, una barra con blocchi di colore casuali diventa caotica;
# in quel caso è meglio omettere la colorbar poiché l'obiettivo è solo distinguere i confini.
if num_clusters <= 30:
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('ID Cluster', fontsize=12)
    cbar.set_ticks(np.arange(num_clusters))
    cbar.set_ticklabels(unique_labels) # Ripristiniamo i veri ID sulla barra
else:
    print(f"Rilevati {num_clusters} cluster. La barra dei colori è stata nascosta per pulizia visiva.")

# 7. Dettagli estetici e salvataggio
ax.set_title(f'Visualizzazione UMAP con Cluster HDBSCAN casuali ({num_clusters} Cluster)', fontsize=14, pad=15)
ax.set_xlabel('UMAP Dimensione 1', fontsize=11)
ax.set_ylabel('UMAP Dimensione 2', fontsize=11)

# Nascondiamo i bordi superflui per un look più moderno
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Mostra la legenda solo per il rumore di fondo
ax.legend(loc='upper right', markerscale=10)

# Salvataggio ad alta risoluzione
output_filename = 'umap_hdbscan_shuffled.png'
fig.savefig(output_filename, bbox_inches='tight', dpi=300)

print(f"Grafico generato e salvato come '{output_filename}'")