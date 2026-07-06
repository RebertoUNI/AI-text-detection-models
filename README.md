# AI Text Detection — training, test e confronto su HPC (ORFEO)

## Struttura dei file

### Moduli condivisi

| Modulo | Responsabilità |
|---|---|
| `data_utils.py` | Caricamento e tokenizzazione dell'**intero** dataset (nessun sotto-campionamento). Cache separate per tokenizer diversi: `./tokenized_dataset` (FCNN, PaperCNN, DeBERTa) e `./tokenized_dataset_qwen` (Qwen). |
| `train_utils.py` | Loop di training con checkpoint/resume, estrazione embeddings a shard, `set_seed()` per riproducibilità. |
| `eval_utils.py` | Metriche di test (accuracy, precision, recall, F1, ROC-AUC) calcolate **identicamente** per tutti i modelli, per un confronto oggettivo. |

### Training (uno script per modello)

| Script | Checkpoint | Embeddings |
|---|---|---|
| `train_fcnn.py` | `checkpoint FCNN/` | `embeddings FCNN/` |
| `train_papercnn.py` | `checkpoint PaperCNN/` | `embeddings PaperCNN/` |
| `train_deberta.py` | `checkpoint daBERTa/` | `embeddings daBERTa/` |
| `train_qwen.py` | `checkpoint Qwen/` | `embeddings Qwen/` |

Tutti supportano `--seed` (default `42`, uguale ovunque) e **resume automatico** se il job viene interrotto.

### Test (uno script per modello — stesso protocollo di valutazione)

- `test_fcnn.py`, `test_papercnn.py`, `test_deberta.py`, `test_qwen.py`
- Valutano l'intero test set col best checkpoint/adapter e salvano in `results/`:
  - `{MODEL}_test_metrics.json` — accuracy, precision, recall, F1, ROC-AUC, confusion matrix
  - `{MODEL}_test_predictions.npz` — `y_true` e `y_prob` grezzi (utili per ROC curve, ecc.)

### Confronto e analisi finale

| Script | Output |
|---|---|
| `compare_results.py` | Legge tutti i `results/*_test_metrics.json` → `results/comparison_table.csv` + `.md`, ordinata per F1. |
| `analyze_model.py --model X` | Pipeline di analisi per-modello (densità, embeddings, occlusione) → `analysis_plots/{MODEL}/`. |
| `compare_models_separation.py` | Confronto cross-modello delle metriche di separazione estratte dagli embeddings. |

---

## Perché il confronto è "oggettivo"

- **Stesso split di test** per tutti i modelli (nessun sotto-campionamento).
- **Stessa funzione di metriche** (`eval_utils.py`) e stessa soglia di decisione (0.5 di default).
- **Stesso seed** (42) per inizializzazione e shuffling di ogni training.
- Gli iperparametri architettura-specifici sono intenzionalmente diversi — FCNN/PaperCNN (10 epoche, LR 1e-3) e DeBERTa/Qwen+LoRA (3 epoche, LR 3e-4) — perché architetture troppo eterogenee non possono condividere una configurazione sensata. La comparabilità è garantita dalla **valutazione finale identica**.

---

## Setup su ORFEO (una tantum)

```bash
bash setup_env.sh
```

Crea un virtualenv in `~/envs/ai-text-detection` e installa `requirements.txt`.

> **Attenzione:** verifica prima i nomi esatti dei moduli disponibili sul cluster (`module avail python`, `module avail cuda`) e correggi `setup_env.sh` di conseguenza — i nomi nello script sono placeholder plausibili, non garantiti per ORFEO.

---

## Lancio dei job su SLURM

`submit_all.sh` sottomette tutto in catena con dipendenze SLURM:
`train_X` → `test_X` (parte solo se il training è `OK`) → `compare_results.py` (parte solo quando tutti i test sono completati).

```bash
bash submit_all.sh
squeue -u $USER    # per seguire l'avanzamento
```

### Prima di lanciare — modifica obbligatoria nei file `.slurm`

- `--account=<TUO_ACCOUNT>` → il tuo account/progetto (usa `sacctmgr show associations` per trovarlo)
- `--partition=GPU` è già corretto (unica partizione GPU su ORFEO, nodo `gpu003`, confermato via `sinfo`)
- `--gres=gpu:1` → verifica quante GPU ha `gpu003` con `scontrol show node gpu003` o `sinfo -o "%N %G"`. Se il cluster distingue i tipi nella sintassi GRES (es. `gpu:v100:1`, `gpu:a100:1`), aggiorna la direttiva di conseguenza.

> Con **un solo nodo GPU** nel cluster, `submit_all.sh` lancia 4 training in parallelo che possono mettersi in coda l'uno dietro l'altro. Controlla `squeue` se i job restano in stato `PD` (pending) più a lungo del previsto.

### Risorse GPU di default (modificabili nei `.slurm`)

| Modello | GPU | Tempo | Note |
|---|---|---|---|
| FCNN / PaperCNN | 1 × qualsiasi | 12 h | Modelli leggeri |
| DeBERTa + LoRA | 1 × (A100/H100 consigliata) | 24 h | Modello grande |
| Qwen + LoRA (4-bit) | 1 × CUDA obbligatoria | 24 h | `bitsandbytes` 4-bit non gira su CPU |

---

## Dipendenze

```bash
pip install -r requirements.txt
```

Le dipendenze non sono pre-installate sui nodi di ORFEO: vanno installate nel virtualenv personale (vedi `setup_env.sh`), che ogni job SLURM attiva prima di eseguire lo script Python.

---

## Execution pipeline

### Passi logici

1. **Setup** — `setup_env.sh` crea il virtualenv e installa le dipendenze.
2. **Caching dataset** — al primo training, `data_utils.py` scarica e tokenizza l'intero dataset; le run successive riusano la cache su disco senza ritokenizzare.
3. **Training** — ogni modello ha il proprio script con checkpoint/resume automatico.
4. **Embedding extraction** — durante il training vengono salvati gli embeddings a shard per l'analisi geometrica successiva.
5. **Test** — ogni `test_X.py` valuta il best checkpoint sull'intero test set con `eval_utils.py`.
6. **Confronto metrico** — `compare_results.py` aggrega tutti i JSON di risultato in una tabella ordinata per F1.
7. **Analisi per-modello** — `analyze_model.py --model X` produce plot interattivi (densità, cluster embeddings, occlusione).
8. **Confronto cross-modello** — `compare_models_separation.py` confronta le metriche di separazione tra modelli.

### Comando rapido (esecuzione locale / interattiva)

```bash
# Esegui per ognuno dei 4 modelli: FCNN, PaperCNN, daBERTa, Qwen
python train_X.py
python test_X.py

# Confronto metrico aggregato
python compare_results.py

# Analisi approfondita per-modello
python analyze_model.py --model FCNN
python analyze_model.py --model PaperCNN
python analyze_model.py --model daBERTa
python analyze_model.py --model Qwen

# Confronto cross-modello della separazione degli embeddings
python compare_models_separation.py
```

### Diagramma della pipeline

```
bash setup_env.sh
        │
        ▼
requirements.txt ──► virtual environment
        │
        ▼
data_utils.py ──► dataset download + tokenization cache
        │                (./tokenized_dataset  /  ./tokenized_dataset_qwen)
        │
        ├──────────────┬──────────────┬──────────────┐
        ▼              ▼              ▼              ▼
  train_fcnn.py  train_papercnn.py  train_deberta.py  train_qwen.py
        │              │              │              │
        ▼              ▼              ▼              ▼
  checkpoint/    checkpoint/    checkpoint/    checkpoint/
  embeddings/    embeddings/    embeddings/    embeddings/
        │              │              │              │
        ▼              ▼              ▼              ▼
  test_fcnn.py  test_papercnn.py  test_deberta.py  test_qwen.py
        │              │              │              │
        └──────────────┴──────────────┴──────────────┘
                               │
                               ▼
             results/*_test_metrics.json
             results/*_test_predictions.npz
                               │
                               ▼
                     compare_results.py
                               │
                               ▼
              results/comparison_table.csv / .md
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
    analyze_model.py  analyze_model.py  ...  (×4 modelli)
                │              │
                └──────────────┘
                               │
                               ▼
                 analysis_plots/{MODEL}/
                   separation_metrics.json
                   *.html  (plot interattivi)
                               │
                               ▼
              compare_models_separation.py
```
