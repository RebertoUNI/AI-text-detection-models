---

## Piano d'azione

### Step 1 — Caricamento dati

- **`results.csv`** → carico `text`, `ground_truth`, `predicted_label`
- **NPZ UMAP** → carico `embeddings_2d`, `labels`, `splits` e filtro per `splits == "test"` per ottenere solo i punti corrispondenti al test set (stesso ordine di `results.csv`)
- **HuggingFace dataset** → carico `R-obi/ai-text-detection-pile-cleaned`, split `test`, estraggo la colonna `cluster_id` (stesso ordine garantito)

### Step 2 — Allineamento e calcolo errori

Unisco tutto in un unico DataFrame per indice posizionale:

| umap_x | umap_y | ground_truth | predicted_label | cluster_id | is_correct | error_type |
|--------|--------|-------------|-----------------|------------|------------|------------|

- `is_correct = (ground_truth == predicted_label)`
- `error_type`:
  - **FP** (Falso Positivo): `ground_truth == LABEL_0` ma `predicted == LABEL_1`
  - **FN** (Falso Negativo): `ground_truth == LABEL_1` ma `predicted == LABEL_0`

> ⚠️ **Assunzione**: considero `LABEL_1` come classe positiva (AI-generated)

### Step 3 — Plot 1: tutti i punti

Scatter plot 2D UMAP:
- 🟢 Verde → predizione corretta
- 🔴 Rosso → errore
- Punto piccolo, alpha basso per gestire la densità

### Step 4 — Plot 2: solo gli errori per cluster

Scatter plot con **solo i punti sbagliati**:
- Colorati per `cluster_id` con una colormap categoriale
- ⬛ Grigio → `cluster_id == -1` (noise/outlier in HDBSCAN)
- Legenda con ID cluster

salvare in formato png in figure separate, aggiungere un parametro che indica la scala degli assi (x e y max)

### Step 5 — File di output errori per cluster

Salvo un CSV con questa struttura:

```
cluster_id | error_type | sentence_index | text (opzionale)
```

E un secondo file di summary:

```
cluster_id | n_FP | n_FN | n_total_errors
```

---
