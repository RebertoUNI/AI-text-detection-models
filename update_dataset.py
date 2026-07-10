"""
Modifica il dataset R-obi/ai-text-detection-pile-cleaned:
- Rinomina 'generated' → 'label'
- Aggiunge 'id' (sequenziale per split)
- Aggiunge 'cluster_id' da hdbscan_labels_run5.npy
- Struttura finale: id, text, label, source, cluster_id

Ordine split per il mapping cluster_id: train → val → test
"""

import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import login

# ─── CONFIGURAZIONE ────────────────────────────────────────────────────────────
HF_TOKEN    = ""   # <-- sostituisci con il tuo token
DATASET_ID  = "R-obi/ai-text-detection-pile-cleaned"
NPY_PATH    = "database exploration/clustering_50d/out/hdbscan_labels_run5.npy"

# Ordine degli split come definito nel progetto
SPLITS = ["train", "val", "test"]

# Mappa tra nome split nel .npy e nome split nel dataset HF
SPLIT_NAME_MAP = {
    "train": "train",
    "val":   "validation",
    "test":  "test",
}
# ───────────────────────────────────────────────────────────────────────────────


def main():
    # 1. Login HuggingFace
    print("🔐 Login HuggingFace...")
    login(token=HF_TOKEN)

    # 2. Carica il dataset originale
    print(f"\n📥 Download dataset '{DATASET_ID}'...")
    raw = load_dataset(DATASET_ID, token=HF_TOKEN)
    print("Split disponibili:", list(raw.keys()))

    # Verifica dimensioni
    total_rows = sum(len(raw[SPLIT_NAME_MAP[s]]) for s in SPLITS)
    print(f"   Righe totali: {total_rows}")

    # 3. Carica le label HDBSCAN
    print(f"\n📂 Caricamento '{NPY_PATH}'...")
    cluster_labels = np.load(NPY_PATH)
    print(f"   Shape: {cluster_labels.shape}")
    print(f"   Cluster unici: {np.unique(cluster_labels).tolist()}")

    assert len(cluster_labels) == total_rows, (
        f"❌ Mismatch! npy ha {len(cluster_labels)} elementi, "
        f"dataset ha {total_rows} righe."
    )
    print("   ✅ Dimensioni coerenti.")

    # 4. Processa ogni split
    print("\n⚙️  Elaborazione split...")
    new_splits = {}
    global_offset = 0

    for split_key in SPLITS:
        hf_split = SPLIT_NAME_MAP[split_key]
        df = raw[hf_split].to_pandas()
        n  = len(df)

        print(f"   [{split_key} → {hf_split}] {n} righe (offset {global_offset})")

        # Estrai i cluster_id corrispondenti a questo split
        split_clusters = cluster_labels[global_offset : global_offset + n]

        # Costruisci il nuovo DataFrame
        new_df = pd.DataFrame({
            "id":         range(global_offset, global_offset + n),
            "text":       df["text"].values,
            "label":      df["generated"].values,
            "source":     df["source"].values,
            "cluster_id": split_clusters.astype(int),
        })

        new_splits[hf_split] = Dataset.from_pandas(new_df, preserve_index=False)
        global_offset += n

    # 5. Assembla il DatasetDict
    updated_dataset = DatasetDict(new_splits)
    print("\n📊 Struttura finale:")
    for name, ds in updated_dataset.items():
        print(f"   {name}: {len(ds)} righe | colonne: {ds.column_names}")

    # 6. Carica su HuggingFace
    print(f"\n🚀 Upload su '{DATASET_ID}'...")
    updated_dataset.push_to_hub(
        DATASET_ID,
        token=HF_TOKEN,
        commit_message="Add id, rename generated→label, add cluster_id from hdbscan_labels_run5",
    )
    print("\n✅ Upload completato!")


if __name__ == "__main__":
    main()
