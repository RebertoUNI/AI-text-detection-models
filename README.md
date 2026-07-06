# AI Text Detection — training, test e confronto su HPC (ORFEO)

## Struttura dei file

### Moduli condivisi
- **data_utils.py** — caricamento/tokenizzazione dell'INTERO dataset (nessun sotto-campionamento). Cache separate per tokenizer diversi:
  - `./tokenized_dataset` → usata da FCNN, PaperCNN, DeBERTa (tutti col tokenizer `deberta-v3-large`)
  - `./tokenized_dataset_qwen` → usata solo da Qwen (tokenizer diverso)
- **train_utils.py** — loop di training con checkpoint/resume, estrazione embeddings a shard, `set_seed()` per riproducibilità.
- **eval_utils.py** — metriche di test (accuracy, precision, recall, F1, ROC-AUC) calcolate ALLO STESSO MODO per tutti i modelli, per un confronto oggettivo.

### Training (uno script per modello)
| Script | Checkpoint | Embeddings |
|---|---|---|
| `train_fcnn.py` | `checkpoint FCNN/` | `embeddings FCNN/` |
| `train_papercnn.py` | `checkpoint PaperCNN/` | `embeddings PaperCNN/` |
| `train_deberta.py` | `checkpoint daBERTa/` | `embeddings daBERTa/` |
| `train_qwen.py` | `checkpoint Qwen/` | `embeddings Qwen/` |

Tutti supportano `--seed` (default 42, uguale ovunque) e resume automatico se il job viene interrotto.

### Test (uno script per modello — stesso protocollo di valutazione)
- `test_fcnn.py`, `test_papercnn.py`, `test_deberta.py`, `test_qwen.py`
- Valutano l'intero test set col best checkpoint/adapter, salvano in `results/`:
  - `{MODEL}_test_metrics.json` (accuracy, precision, recall, f1, roc_auc, confusion matrix)
  - `{MODEL}_test_predictions.npz` (y_true, y_prob grezzi, utili per ROC curve dopo)

### Confronto finale
- `compare_results.py` — legge tutti i `results/*_test_metrics.json` e produce:
  - `results/comparison_table.csv`
  - `results/comparison_table.md`
  - una tabella anche stampata a schermo, ordinata per F1

## Perché il confronto è "oggettivo"

- Stesso split di test per tutti i modelli (nessun sotto-campionamento).
- Stessa funzione di calcolo metriche (`eval_utils.py`), stessa soglia di decisione (0.5 di default).
- Stesso seed (42) per l'inizializzazione/shuffling di ogni training.
- I singoli iperparametri (epochs, batch size, LR) restano specifici per architettura — FCNN/PaperCNN (10 epoche, LR 1e-3) e DeBERTa/Qwen+LoRA (3 epoche, LR 3e-4) NON sono uguali tra loro perché le architetture sono troppo diverse per condividere un'unica configurazione sensata: quello che li rende comparabili è la valutazione finale, identica per tutti.

## Setup su ORFEO (una tantum)

```bash
bash setup_env.sh
```
Crea un virtualenv in `~/envs/ai-text-detection` e ci installa `requirements.txt`. **Verifica prima i nomi esatti dei moduli** (`module avail python`, `module avail cuda`) e correggi `setup_env.sh` di conseguenza — i nomi nello script sono placeholder plausibili, non garantiti per ORFEO.

## Lancio dei job

Cartella `slurm/`: uno script `.slurm` per ogni training e ogni test, più:
- `submit_all.sh` — sottomette tutto in catena: `train_X` → `test_X` (dipendente, parte solo se il training è OK) → job finale `compare_results.py` (parte solo quando tutti i test sono completati).

```bash
bash slurm/submit_all.sh
squeue -u $USER          # per seguire l'avanzamento
```

**Prima di lanciare, modifica nei file `.slurm`:**
- `--account=<TUO_ACCOUNT>` → il tuo account/progetto (`sacctmgr show associations`)
- `--partition=GPU` è già corretto (unica partizione GPU su ORFEO, nodo `gpu003`, confermato via `sinfo`)
- `--gres=gpu:1` → verifica quante GPU e di che tipo ha `gpu003` con `scontrol show node gpu003` (o `sinfo -o "%N %G"`); se il cluster distingue i tipi (V100/A100/H100) nella sintassi GRES, es. `gpu:v100:1`, aggiorna la direttiva di conseguenza. Con **un solo nodo GPU nel cluster**, se lanci più job insieme (`submit_all.sh` ne lancia 4 in parallelo) potrebbero mettersi in coda l'uno dietro l'altro se il nodo non ha abbastanza GPU libere contemporaneamente — controlla `squeue` per vedere se restano in `PD` (pending) più a lungo del previsto.

Risorse GPU assegnate di default (adattabili):
- FCNN / PaperCNN: 1 GPU qualsiasi, 12h, modelli piccoli
- DeBERTa+LoRA: 1 GPU (idealmente A100/H100 vista la dimensione del modello), 24h
- Qwen+LoRA (4bit): 1 GPU, 24h — bitsandbytes 4bit richiede CUDA, non gira su CPU

## Dipendenze

```bash
pip install -r requirements.txt
```
Non sono pre-installate sui nodi di ORFEO: vanno installate nel tuo virtualenv/conda env personale (vedi `setup_env.sh`), che poi ogni job SLURM attiva prima di eseguire lo script Python.

## Execution pipeline
- Environment setup on ORFEO with `setup_env.sh`, which creates a virtual environment and installs the Python dependencies from `requirements.txt` .
- Full dataset download and tokenizer-specific caching through `data_utils.py`, so repeated jobs do not re-tokenize the dataset from scratch .
- Training of each model through its own script, with automatic resume from checkpoint if a job is interrupted .
- Extraction and sharded storage of intermediate embeddings for later geometric/separation analysis .
- Testing on the full test split using a common metrics implementation in `eval_utils.py`, producing `results/{MODEL}_test_metrics.json` and `results/{MODEL}_test_predictions.npz`.
- Aggregation of all test metrics with `compare_results.py`, which writes `results/comparison_table.csv` and `results/comparison_table.md` sorted by F1 .
- Per-model post-hoc analysis with `analyze_model.py`, followed by cross-model comparison with `compare_models_separation.py` using saved embeddings and analysis outputs .

A compact command-level version of the intended execution flow is:
```bash
python train_X.py
python test_X.py
python compare_results.py
python analyze_model.py --model X
python compare_models_separation.py
```
