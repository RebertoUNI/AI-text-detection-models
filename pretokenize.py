"""
pretokenize.py
─────────────────────────────────────────────────────────────────────────────
Costruisce UNA VOLTA la cache tokenizzata dell'intero dataset, usando SOLO
CPU (nessuna GPU necessaria). Da lanciare sul nodo EPYC (128 core) invece che
sul nodo GPU: la tokenizzazione è puro lavoro CPU e non ha senso occupare
l'unica GPU del cluster (V100 su gpu003) per farla.

Uso (tipicamente dentro pretokenize.slurm, partizione EPYC):
    python pretokenize.py --tokenizer deberta               # per FCNN/PaperCNN/daBERTa
    python pretokenize.py --tokenizer qwen                  # per Qwen (tokenizer diverso)
    python pretokenize.py --tokenizer deberta --num-proc 32

Dopo aver lanciato questo script (per entrambi i tokenizer), i job di
training su GPU (train_fcnn.slurm, ecc.) troveranno la cache già pronta e
partiranno DIRETTAMENTE con il training, senza perdere tempo GPU sulla
tokenizzazione.
"""

import argparse
import logging
import time

from data_utils import load_and_tokenize_dataset_generic, TOKENIZER_NAME
from train_utils import setup_logging

TOKENIZER_PRESETS = {
    "deberta": {"tokenizer_name": TOKENIZER_NAME, "cache_dir": "./tokenized_dataset", "max_length": 256},
    "qwen":    {"tokenizer_name": "Qwen/Qwen3-0.6B-Base", "cache_dir": "./tokenized_dataset_qwen", "max_length": 256},
}


def parse_args():
    p = argparse.ArgumentParser(description="Pre-tokenizza l'intero dataset (solo CPU, una tantum)")
    p.add_argument("--tokenizer", choices=list(TOKENIZER_PRESETS.keys()), required=True)
    p.add_argument("--num-proc", type=int, default=32,
                    help="Processi paralleli per la tokenizzazione (metti quanti più core hai, es. 32-64 su EPYC)")
    return p.parse_args()


def main():
    args = parse_args()
    preset = TOKENIZER_PRESETS[args.tokenizer]
    setup_logging("logs/pretokenize.log")
    logger = logging.getLogger(__name__)

    logger.info(f"Avvio pre-tokenizzazione con '{preset['tokenizer_name']}' "
                f"({args.num_proc} processi) -> cache in '{preset['cache_dir']}'")
    t0 = time.time()

    load_and_tokenize_dataset_generic(
        tokenizer_name=preset["tokenizer_name"],
        cache_dir=preset["cache_dir"],
        max_length=preset["max_length"],
        num_proc=args.num_proc,
    )

    logger.info(f"Fatto in {(time.time() - t0)/60:.1f} minuti. "
                f"Cache pronta in '{preset['cache_dir']}' per i job di training su GPU.")


if __name__ == "__main__":
    main()
