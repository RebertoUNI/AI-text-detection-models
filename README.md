- data_utils.py — condiviso: scarica/tokenizza l'intero dataset (non più il subset da 20000) con il tokenizer di deberta-v3-large, e cachizza il risultato su disco (./tokenized_dataset) così se lanci più job HPC non ritokenizzi ogni volta da zero.
- train_utils.py — condiviso: loop di training con checkpoint ad ogni epoca (checkpoint_last.pt) e resume automatico se il job HPC viene interrotto o va in timeout, più checkpoint_best.pt per il modello migliore. Contiene anche extract_and_save_embeddings, che salva gli embedding delle frasi a shard (file .npz progressivi) per non saturare la RAM quando processi l'intero dataset.
- train_fcnn.py — standalone, esegue: training FCNN → salva in checkpoint FCNN/ → estrae embedding (penultimo layer, 128-dim) su train/val/test → salva in embeddings FCNN/.
- train_papercnn.py — stessa logica per PaperCNN → checkpoint PaperCNN/ e embeddings PaperCNN/ (embedding 256-dim, penultimo layer prima del classificatore).


Uso su HPC, per esempio:

---

bashpython train_fcnn.py --epochs 10 --batch-size 128 --num-workers 8 

python train_papercnn.py --epochs 10 --batch-size 128 --num-workers 8

---
Se il job viene killato a metà, rilanciando lo stesso comando riprende da checkpoint_last.pt senza ripartire da zero.

Batch size / num_workers: 128 e 8 come default (più alti di quelli Colab, dato che l'HPC probabilmente ha più CPU/GPU/RAM), ma vanno tarati sul nodo che userai.
--embedding-shard-size (default 50.000): controlla quanti embedding tiene in RAM prima di scrivere uno shard su disco — abbassalo se il nodo ha poca memoria.
