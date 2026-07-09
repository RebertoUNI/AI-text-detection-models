# -*- coding: utf-8 -*-
"""
UMAP Density Analysis - AI vs Human Text Detection
Batch version of pseudofinal-notebook-fixed.ipynb for HPC (Orfeo / SLURM).

Pipeline (same as the notebook, in dependency-correct order):
  1.  download precomputed UMAP files from GitHub
  2.  density plots (HTML, no display) + per-label density peaks + peak phrases
  3.  load the pile dataset, full row-alignment check vs umap_labels.npy
  4.  fetch 50 phrases per named generator sub-dataset (cached to disk)
  5.  keyword-matched fiction/news phrase sets
  6.  ParaDetect (DeBERTa-v3-large + LoRA) classification of those sets
  7.  frozen-probe test: early Qwen3 layers vs final embedding (v1 + v2 with CIs)
  8.  SupCon LoRA fine-tuning with topic-diverse positives + random-crop augment
  9.  holdout evaluation + unseen-topic (science/sports) generalization test
  10. hand-typed text comparison (OOD demo)
  11. ParaDetect on the same holdout (leakage-caveated comparison)
  12. era-matched generator recall + fresh GPT-2 XL zero-overlap test
  13. final comparison table -> CSV

Requirements (venv):
  pip install numpy scipy pandas scikit-learn requests plotly \
              datasets transformers peft accelerate torch

HPC usage (ORFEO, account dssc):
  # on the login node (has internet) - pre-download everything. Use the SAME
  # HF_HOME / --data-dir that run_orfeo.sbatch exports (SCRATCH_DIR there),
  # e.g.:
  export HF_HOME=/u/dssc/rtittoto/scratch/ai-text-detection/hf_cache
  source /u/dssc/rtittoto/scratch/envs/ai-text-detection/bin/activate
  python pseudofinal_analysis_orfeo.py --download-only \
      --data-dir /u/dssc/rtittoto/scratch/ai-text-detection/umap_data

  # then submit the batch job: sbatch run_orfeo.sbatch
  # if compute nodes have no internet, also export
  # HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 in the job (see run_orfeo.sbatch).

The script uses every GPU that SLURM makes visible (1 or 2+ both work).
"""

import argparse
import json
import os
import random
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------------------------- CLI / config

parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
parser.add_argument("--data-dir", default="data", help="cache dir for downloads")
parser.add_argument("--output-dir", default="outputs", help="dir for plots/CSVs/models")
parser.add_argument("--download-only", action="store_true",
                    help="download/cache all remote assets, then exit (run on login node)")
parser.add_argument("--skip-plots", action="store_true", help="skip the Plotly HTML plots")
parser.add_argument("--n-batches", type=int, default=500, help="SupCon training steps")
parser.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end run (~15 min on 1 V100) to validate the "
                         "pipeline before submitting the full job; results are NOT "
                         "scientifically meaningful")
args = parser.parse_args()

DATA_DIR = args.data_dir
OUTPUT_DIR = args.output_dir
N_VICINI = 50    # neighbors to pull around each density peak, per label
BINS = 60        # 2D histogram resolution used to locate the peak

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------- cached remote-data helpers

_ROWS_CACHE_PATH = os.path.join(DATA_DIR, "hf_rows_cache.json")
_rows_cache = {}
if os.path.exists(_ROWS_CACHE_PATH):
    with open(_ROWS_CACHE_PATH, encoding="utf-8") as f:
        _rows_cache = json.load(f)


def fetch_hf_rows(dataset, config="default", split="train", offset=0, length=50,
                  retries=3, backoff=5):
    """datasets-server rows API, cached to disk so compute nodes can run offline."""
    key = f"{dataset}|{config}|{split}|{offset}|{length}"
    if key in _rows_cache:
        return _rows_cache[key]
    params = {"dataset": dataset, "config": config, "split": split,
              "offset": offset, "length": length}
    for attempt in range(retries):
        r = requests.get("https://datasets-server.huggingface.co/rows",
                         params=params, timeout=60)
        if r.status_code == 200:
            rows = [item["row"] for item in r.json()["rows"]]
            _rows_cache[key] = rows
            with open(_ROWS_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(_rows_cache, f)
            return rows
        if r.status_code in (502, 503, 504) and attempt < retries - 1:
            print(f"{r.status_code} from datasets-server, retrying in {backoff}s "
                  f"({attempt + 1}/{retries})...")
            time.sleep(backoff)
            continue
        r.raise_for_status()


def fetch_gpt2_phrases(n=50, variant="xl-1542M", split="test"):
    """Official OpenAI gpt-2-output-dataset (full 1.5B model), cached to disk."""
    cache_path = os.path.join(DATA_DIR, f"gpt2_{variant}_{split}_{n}.json")
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    url = f"https://openaipublic.azureedge.net/gpt-2/output-dataset/v1/{variant}.{split}.jsonl"
    phrases = []
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            phrases.append(json.loads(line)["text"].strip())
            if len(phrases) >= n:
                break
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(phrases, f)
    return phrases


def download_umap_files():
    github_raw = ("https://raw.githubusercontent.com/RebertoUNI/"
                  "AI-text-detection-models/main/umap_output")
    for fname in ["umap_embeddings_2d.npy", "umap_labels.npy", "umap_results.npz"]:
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            print(f"Already downloaded: {fpath}")
            continue
        url = f"{github_raw}/{fname}"
        print(f"Downloading {url} ...")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        with open(fpath, "wb") as f:
            f.write(r.content)
        print(f"Saved {fpath} ({len(r.content):,} bytes)")


HF_MODEL_REPOS = [
    "srikanthgali/paradetect-deberta-v3-lora",
    "microsoft/deberta-v3-large",
    "Qwen/Qwen3-Embedding-0.6B",
    "gpt2-xl",
]

SUB_DATASET_FETCHES = [
    dict(dataset="AlekseyKorshuk/davinci-pairwise", length=50),
    dict(dataset="Dahoas/instruct-synthetic-prompt-responses", length=50),
    dict(dataset="Dahoas/synthetic-instruct-gptj-pairwise", length=50),
    dict(dataset="Hello-SimpleAI/HC3", config="all", length=100),
    dict(dataset="MohamedRashad/ChatGPT-prompts", length=50),
]


def download_everything():
    """--download-only mode: warm every cache, then exit (run on login node)."""
    from huggingface_hub import snapshot_download
    from datasets import load_dataset

    download_umap_files()
    print("Caching pile dataset ...")
    load_dataset("srikanthgali/ai-text-detection-pile-cleaned", split="train")
    print("Caching datasets-server rows ...")
    for kw in SUB_DATASET_FETCHES:
        fetch_hf_rows(**kw)
    print("Caching GPT-2 output-dataset phrases ...")
    fetch_gpt2_phrases(50)
    for repo in HF_MODEL_REPOS:
        print(f"Caching model repo {repo} ...")
        snapshot_download(repo)
    print("All assets cached. Submit the batch job now "
          "(set HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 if compute nodes lack internet).")


if args.download_only:
    download_everything()
    raise SystemExit(0)

# =============================================================== main pipeline

import torch  # noqa: E402  (import after --download-only exit; heavy)


def main():
    # ------------------------------------------------ 1. precomputed UMAP files
    download_umap_files()

    embeddings_2d = np.load(f"{DATA_DIR}/umap_embeddings_2d.npy")
    labels = np.load(f"{DATA_DIR}/umap_labels.npy")

    npz = np.load(f"{DATA_DIR}/umap_results.npz")
    assert np.array_equal(embeddings_2d, npz["embeddings_2d"])
    assert np.array_equal(labels, npz["labels"])

    x, y = embeddings_2d[:, 0], embeddings_2d[:, 1]
    idx_all = np.arange(len(labels))
    mask0 = labels == 0
    mask1 = labels == 1

    print("Embeddings shape:", embeddings_2d.shape)
    print("Labels shape:", labels.shape)
    print(f"label 0 (Human): {mask0.sum():,}   label 1 (AI): {mask1.sum():,}")

    # ------------------------------------------------ 2. density plots (HTML only)
    if not args.skip_plots:
        import plotly.graph_objects as go

        fig1 = go.Figure()
        fig1.add_trace(go.Histogram2dContour(
            x=x[mask1], y=y[mask1], colorscale="Reds", ncontours=25,
            contours=dict(coloring="heatmap")))
        fig1.update_layout(title="Density - Label 1 (AI)", xaxis_title="UMAP 1",
                           yaxis_title="UMAP 2", width=800, height=650)
        fig1.write_html(os.path.join(OUTPUT_DIR, "umap_density_label1.html"))

        fig0 = go.Figure()
        fig0.add_trace(go.Histogram2dContour(
            x=x[mask0], y=y[mask0], colorscale="Blues", ncontours=25,
            contours=dict(coloring="heatmap")))
        fig0.update_layout(title="Density - Label 0 (Human)", xaxis_title="UMAP 1",
                           yaxis_title="UMAP 2", width=800, height=650)
        fig0.write_html(os.path.join(OUTPUT_DIR, "umap_density_label0.html"))

        fig_lines = go.Figure()
        fig_lines.add_trace(go.Histogram2dContour(
            x=x[mask0], y=y[mask0],
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "blue"]], ncontours=15,
            contours=dict(coloring="lines", showlabels=False),
            line=dict(width=1.5), showscale=False, name="Label 0 (Human)"))
        fig_lines.add_trace(go.Histogram2dContour(
            x=x[mask1], y=y[mask1],
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "red"]], ncontours=15,
            contours=dict(coloring="lines", showlabels=False),
            line=dict(width=1.5), showscale=False, name="Label 1 (AI)"))
        fig_lines.update_layout(title="Contour lines - Label 0 (blue) vs Label 1 (red)",
                                xaxis_title="UMAP 1", yaxis_title="UMAP 2",
                                width=800, height=650)
        fig_lines.write_html(os.path.join(OUTPUT_DIR, "umap_contours_only.html"))
        print(f"Density plots written to {OUTPUT_DIR}/")

    # ------------------------------------------------ 3. density peaks per label
    def find_top_peak_and_neighbors(xs, ys, ids, bins=BINS, n_neighbors=N_VICINI):
        H, xedges, yedges = np.histogram2d(xs, ys, bins=bins)
        row, col = np.unravel_index(np.argmax(H), H.shape)
        val = H[row, col]
        xc = (xedges[row] + xedges[row + 1]) / 2
        yc = (yedges[col] + yedges[col + 1]) / 2
        tree = cKDTree(np.column_stack([xs, ys]))
        _, nearest_idxs = tree.query([[xc, yc]], k=n_neighbors)
        nearest_idxs = nearest_idxs[0]
        neighbor_ids = [int(ids[i]) for i in nearest_idxs]
        return {"peak_x": float(xc), "peak_y": float(yc), "density": float(val),
                "neighbor_ids": neighbor_ids}

    top_label0 = find_top_peak_and_neighbors(x[mask0], y[mask0], idx_all[mask0])
    top_label1 = find_top_peak_and_neighbors(x[mask1], y[mask1], idx_all[mask1])

    print("=== Top peak - Label 0 (Human) ===")
    print({k: v for k, v in top_label0.items() if k != "neighbor_ids"})
    print("\n=== Top peak - Label 1 (AI) ===")
    print({k: v for k, v in top_label1.items() if k != "neighbor_ids"})

    # ------------------------------------------------ 4. text dataset + alignment
    from datasets import load_dataset

    ds = load_dataset("srikanthgali/ai-text-detection-pile-cleaned", split="train")
    print(f"Loaded {len(ds):,} rows")

    labels_full = np.array(ds["generated"])
    assert np.array_equal(labels, labels_full), \
        "Row-order mismatch between text dataset and UMAP labels!"
    print("Alignment check passed: dataset rows match the UMAP arrays one-to-one.")

    def print_phrases(idx_list, label_name):
        print(f"\n=== Phrases near the peak - {label_name} ===")
        for i in idx_list:
            entry = ds[i]
            print(f"\nidx {i}")
            print(f"actual label : {entry['generated']}")
            print(f"text: {entry['text'][:800]}...")

    print_phrases(top_label0["neighbor_ids"], "Label 0 ")
    print_phrases(top_label1["neighbor_ids"], "Label 1 ")

    # ------------------------------------------------ 5. named generator sub-datasets
    gpt2_phrases = fetch_gpt2_phrases(50)
    print(f"GPT-2: {len(gpt2_phrases)} phrases")

    rows = fetch_hf_rows("AlekseyKorshuk/davinci-pairwise", length=50)
    gpt3_pairwise_davinci_phrases = [r["chosen"].strip() for r in rows]
    print(f"GPT-3 (pairwise-davinci): {len(gpt3_pairwise_davinci_phrases)} phrases")

    rows = fetch_hf_rows("Dahoas/instruct-synthetic-prompt-responses", length=50)
    gpt3_synthetic_instruct_phrases = [r["response"].strip() for r in rows]
    print(f"GPT-3 (synthetic-instruct-davinci-pairwise): "
          f"{len(gpt3_synthetic_instruct_phrases)} phrases")

    rows = fetch_hf_rows("Dahoas/synthetic-instruct-gptj-pairwise", length=50)
    gptj_phrases = [r["chosen"].strip() for r in rows]
    print(f"GPT-J (synthetic-instruct-gptj-pairwise): {len(gptj_phrases)} phrases")

    chatgpt_twitter_phrases = []  # source never published, no public mirror
    print(f"ChatGPT (Twitter scrape): {len(chatgpt_twitter_phrases)} phrases "
          f"(source not publicly available)")

    rows = fetch_hf_rows("Hello-SimpleAI/HC3", config="all", length=100)
    chatgpt_hc3_phrases = []
    for r in rows:
        answers = r.get("chatgpt_answers") or []
        if answers:
            chatgpt_hc3_phrases.append(answers[0].strip())
        if len(chatgpt_hc3_phrases) >= 50:
            break
    print(f"ChatGPT (HC3): {len(chatgpt_hc3_phrases)} phrases")

    rows = fetch_hf_rows("MohamedRashad/ChatGPT-prompts", length=50)
    chatgpt_emergentmind_phrases = [r["chatgpt_response"].strip() for r in rows]
    print(f"ChatGPT (emergentmind): {len(chatgpt_emergentmind_phrases)} phrases")

    # ------------------------------------------------ 6. fiction/news keyword sets
    FICTION_KEYWORDS = [
        "once upon a time", "car crash", "car accident", "swerved", "collided",
        "novel", "chapter", "protagonist", "fictional", "story begins",
    ]
    NEWS_KEYWORDS = [
        "robbery", "robbed", "murder", "murdered", "homicide", "police said",
        "authorities", "arrested", "stabbed", "shot dead", "killed",
    ]

    def matches_any(text, keywords):
        t = text.lower()
        return any(kw in t for kw in keywords)

    rng = random.Random(42)
    order = list(range(len(ds)))
    rng.shuffle(order)

    fiction_phrases, news_phrases = [], []
    for i in order:
        text = ds[i]["text"]
        if len(fiction_phrases) < 50 and matches_any(text, FICTION_KEYWORDS):
            fiction_phrases.append(text)
        elif len(news_phrases) < 50 and matches_any(text, NEWS_KEYWORDS):
            news_phrases.append(text)
        if len(fiction_phrases) >= 50 and len(news_phrases) >= 50:
            break

    print(f"Fiction: {len(fiction_phrases)} phrases")
    print(f"News:    {len(news_phrases)} phrases")

    extracted_sets = {
        "GPT-2": gpt2_phrases,
        "GPT-3 (pairwise-davinci)": gpt3_pairwise_davinci_phrases,
        "GPT-3 (synthetic-instruct-davinci-pairwise)": gpt3_synthetic_instruct_phrases,
        "GPT-J (synthetic-instruct-gptj-pairwise)": gptj_phrases,
        "ChatGPT (Twitter)": chatgpt_twitter_phrases,
        "ChatGPT (HC3)": chatgpt_hc3_phrases,
        "ChatGPT (emergentmind)": chatgpt_emergentmind_phrases,
        "Fiction": fiction_phrases,
        "News": news_phrases,
    }
    for name, phrases in extracted_sets.items():
        print(f"{name}: {len(phrases)} phrases")

    # ------------------------------------------------ 7. ParaDetect classifier
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
    from peft import PeftModel

    PARADETECT_REPO = "srikanthgali/paradetect-deberta-v3-lora"

    paradetect_tokenizer = AutoTokenizer.from_pretrained(PARADETECT_REPO)
    _base_model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-large", num_labels=2)
    paradetect_model = PeftModel.from_pretrained(_base_model, PARADETECT_REPO)

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    paradetect_model.to(_device)
    paradetect_model.eval()
    print(f"ParaDetect loaded on {_device}")

    def predict_text_origin(text):
        inputs = paradetect_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding=True
        ).to(_device)
        with torch.no_grad():
            logits = paradetect_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
        return {"prediction": "AI" if pred == 1 else "Human",
                "human_probability": probs[0][0].item(),
                "ai_probability": probs[0][1].item()}

    def classify_phrases(phrases, label_name=""):
        results_ = []
        for text in phrases:
            pred = predict_text_origin(text)
            pred["text"] = text
            results_.append(pred)
        n_ai = sum(1 for r in results_ if r["prediction"] == "AI")
        print(f"{label_name}: {len(results_)} phrases -> "
              f"predicted AI={n_ai}  Human={len(results_) - n_ai}")
        return results_

    @torch.no_grad()
    def classify_paradetect_batch(texts, batch_size=32, max_length=512):
        preds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = paradetect_tokenizer(
                batch, return_tensors="pt", truncation=True,
                max_length=max_length, padding=True).to(_device)
            logits = paradetect_model(**inputs).logits
            preds.append(torch.argmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(preds)

    classify_phrases(fiction_phrases, label_name="Fiction")
    classify_phrases(news_phrases, label_name="News")

    # ------------------------------------------------ 8. Qwen3 layer probes
    QWEN_REPO = "Qwen/Qwen3-Embedding-0.6B"

    _num_gpus = torch.cuda.device_count()
    _qwen_devices = [f"cuda:{i}" for i in range(_num_gpus)] if _num_gpus > 0 else ["cpu"]

    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, padding_side="left")

    qwen_models = {}
    for dev in _qwen_devices:
        m = AutoModel.from_pretrained(
            QWEN_REPO, torch_dtype=torch.float16 if dev != "cpu" else torch.float32)
        m.to(dev)
        m.eval()
        qwen_models[dev] = m
    print(f"Qwen3-Embedding-0.6B loaded on: {list(qwen_models)}")

    def last_token_pool(hidden_states, attention_mask):
        # same method Qwen's own model card uses for the final layer
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = hidden_states.shape[0]
        return hidden_states[torch.arange(batch_size, device=hidden_states.device),
                             sequence_lengths]

    EARLY_LAYER_SETS = {"first_4_layers": 4, "first_6_layers": 6, "first_8_layers": 8}
    FEATURE_NAMES = ["first_4_layers", "first_6_layers", "first_8_layers", "final"]

    @torch.no_grad()
    def extract_layer_features(texts, device, batch_size=32, max_length=256):
        model = qwen_models[device]
        out = {name: [] for name in EARLY_LAYER_SETS}
        out["final"] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = qwen_tokenizer(batch, padding=True, truncation=True,
                                 max_length=max_length, return_tensors="pt").to(device)
            result = model(**enc, output_hidden_states=True)
            hidden_states = result.hidden_states  # (embed_out, layer1, ..., layer28)
            attn_mask = enc["attention_mask"]
            for name, k in EARLY_LAYER_SETS.items():
                stacked = torch.stack(hidden_states[1:k + 1], dim=0)
                averaged = stacked.mean(dim=0)
                pooled = last_token_pool(averaged, attn_mask)
                out[name].append(pooled.float().cpu().numpy())
            pooled_final = last_token_pool(result.last_hidden_state, attn_mask)
            out["final"].append(pooled_final.float().cpu().numpy())
        return {name: np.concatenate(chunks, axis=0) for name, chunks in out.items()}

    def _extract_chunk(job):
        texts_chunk, device = job
        return extract_layer_features(texts_chunk, device=device)

    def extract_parallel(texts):
        chunks = np.array_split(np.arange(len(texts)), len(_qwen_devices))
        jobs = [([texts[i] for i in chunk], _qwen_devices[j])
                for j, chunk in enumerate(chunks)]
        with ThreadPoolExecutor(max_workers=len(_qwen_devices)) as pool:
            chunk_results = list(pool.map(_extract_chunk, jobs))
        return {name: np.concatenate([cr[name] for cr in chunk_results], axis=0)
                for name in FEATURE_NAMES}

    # --- v1: random 20k sample ---
    N_SAMPLE = 2_000 if args.smoke else 20_000
    rng2 = np.random.default_rng(42)
    all_idx = np.arange(len(ds))
    human_idx = rng2.choice(all_idx[labels_full == 0], N_SAMPLE // 2, replace=False)
    ai_idx = rng2.choice(all_idx[labels_full == 1], N_SAMPLE // 2, replace=False)
    sample_idx = np.concatenate([human_idx, ai_idx])
    rng2.shuffle(sample_idx)

    sample_texts = [ds[int(i)]["text"] for i in sample_idx]
    sample_labels = np.array([ds[int(i)]["generated"] for i in sample_idx])

    def tag_topic(text):
        if matches_any(text, FICTION_KEYWORDS):
            return "fiction"
        if matches_any(text, NEWS_KEYWORDS):
            return "news"
        return "other"

    sample_topics = np.array([tag_topic(t) for t in sample_texts])
    print(f"Sampled {len(sample_texts):,} rows")
    print("Label counts:", dict(zip(*np.unique(sample_labels, return_counts=True))))
    print("Topic counts:", dict(zip(*np.unique(sample_topics, return_counts=True))))

    features = extract_parallel(sample_texts)
    np.savez_compressed(os.path.join(DATA_DIR, "qwen_layer_features.npz"),
                        labels=sample_labels, topics=sample_topics, **features)
    print({name: arr.shape for name, arr in features.items()})

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    def bootstrap_ci(y_true, y_pred, n_boot=1000, rng_=None):
        rng_ = rng_ or np.random.default_rng(0)
        n = len(y_true)
        if n == 0:
            return (np.nan, np.nan)
        accs = []
        for _ in range(n_boot):
            idx = rng_.integers(0, n, n)
            accs.append((y_true[idx] == y_pred[idx]).mean())
        return np.percentile(accs, [2.5, 97.5])

    def evaluate_probe(X, y_, topics, name, with_ci=True):
        X_train, X_test, y_train, y_test, _, topics_test = train_test_split(
            X, y_, topics, test_size=0.2, random_state=42, stratify=y_)
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X_train, y_train)
        y_pred_all = clf.predict(X_test)
        row = {"approach": name, "overall": (y_pred_all == y_test).mean()}
        for topic in ["fiction", "news", "other"]:
            mask = topics_test == topic
            if mask.sum() > 0:
                row[topic] = (y_pred_all[mask] == y_test[mask]).mean()
                if with_ci:
                    lo, hi = bootstrap_ci(y_test[mask], y_pred_all[mask])
                    row[f"{topic}_ci"] = f"[{lo:.3f}, {hi:.3f}]"
                row[f"{topic}_n"] = int(mask.sum())
        return row, clf

    results_v1 = []
    for name in FEATURE_NAMES:
        row, _ = evaluate_probe(features[name], sample_labels, sample_topics,
                                name, with_ci=False)
        results_v1.append(row)
    results_v1_df = pd.DataFrame(results_v1)
    print("\n=== Probe results v1 (incidental topic counts - small, see v2) ===")
    print(results_v1_df)
    results_v1_df.to_csv(os.path.join(OUTPUT_DIR, "probe_results_v1.csv"), index=False)

    # --- sanity check: layer sets must differ ---
    print("first_6 vs first_8 identical:",
          np.allclose(features["first_6_layers"], features["first_8_layers"]))
    print("max abs diff:",
          np.abs(features["first_6_layers"] - features["first_8_layers"]).max())

    # --- v2: dedicated topic x label buckets + bootstrap CIs ---
    TARGET_PER_TOPIC_LABEL = 200 if args.smoke else 1000
    TARGET_PER_OTHER_LABEL = 400 if args.smoke else 2000
    SCAN_BUDGET = 200_000  # unchanged in smoke mode: perm[SCAN_BUDGET:] must
    #                        stay disjoint from the training scan either way

    rng3 = np.random.default_rng(42)
    perm = rng3.permutation(len(ds))    # rows beyond SCAN_BUDGET feed the
    scan_order = perm[:SCAN_BUDGET]     # unseen-topic test later

    buckets_text = {("fiction", 0): [], ("fiction", 1): [],
                    ("news", 0): [], ("news", 1): [],
                    ("other", 0): [], ("other", 1): []}
    targets = {("fiction", 0): TARGET_PER_TOPIC_LABEL,
               ("fiction", 1): TARGET_PER_TOPIC_LABEL,
               ("news", 0): TARGET_PER_TOPIC_LABEL,
               ("news", 1): TARGET_PER_TOPIC_LABEL,
               ("other", 0): TARGET_PER_OTHER_LABEL,
               ("other", 1): TARGET_PER_OTHER_LABEL}

    for i in scan_order:
        entry = ds[int(i)]
        text, label = entry["text"], entry["generated"]
        topic = ("fiction" if matches_any(text, FICTION_KEYWORDS)
                 else "news" if matches_any(text, NEWS_KEYWORDS) else "other")
        key = (topic, label)
        if len(buckets_text[key]) < targets[key]:
            buckets_text[key].append(text)
        if all(len(buckets_text[k]) >= targets[k] for k in targets):
            break

    for k, v in buckets_text.items():
        print(k, len(v))

    sample_texts, sample_labels, sample_topics = [], [], []
    for (topic, label), texts in buckets_text.items():
        sample_texts += texts
        sample_labels += [label] * len(texts)
        sample_topics += [topic] * len(texts)
    sample_labels = np.array(sample_labels)
    sample_topics = np.array(sample_topics)
    print(f"Total sample: {len(sample_texts):,}")

    features = extract_parallel(sample_texts)
    np.savez_compressed(os.path.join(DATA_DIR, "qwen_layer_features_v2.npz"),
                        labels=sample_labels, topics=sample_topics, **features)
    print({name: arr.shape for name, arr in features.items()})

    results = []
    for name in FEATURE_NAMES:
        row, _ = evaluate_probe(features[name], sample_labels, sample_topics, name)
        results.append(row)
    results_df = pd.DataFrame(results)
    print("\n=== Probe results v2 (bucketed, with bootstrap CIs) ===")
    print(results_df)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "probe_results_v2.csv"), index=False)

    # ------------------------------------------------ 9. SupCon LoRA fine-tuning
    import torch.nn as nn
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model, TaskType

    NUM_LAYERS_FOR_SUPCON = 6

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])

    # NOTE: get_peft_model injects LoRA layers into the shared qwen_models
    # instances - no frozen-probe extraction may run after this point.
    _supcon_device = _qwen_devices[0]
    lora_model = get_peft_model(qwen_models[_supcon_device], lora_config)
    lora_model.gradient_checkpointing_enable()
    lora_model.enable_input_require_grads()
    lora_model.train()
    lora_model.print_trainable_parameters()

    class ProjectionHead(nn.Module):
        def __init__(self, in_dim=1024, hidden_dim=512, out_dim=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, out_dim))

        def forward(self, x_):
            return F.normalize(self.net(x_), dim=1)

    projection_head = ProjectionHead().to(_supcon_device)

    class SupConLoss(nn.Module):
        def __init__(self, temperature=0.07):
            super().__init__()
            self.temperature = temperature

        def forward(self, z, labels_):
            device = z.device
            labels_ = labels_.view(-1, 1)
            mask = torch.eq(labels_, labels_.T).float().to(device)
            logits = torch.matmul(z, z.T) / self.temperature
            logits = logits - logits.max(dim=1, keepdim=True).values.detach()
            self_mask = torch.eye(z.shape[0], device=device)
            mask = mask * (1 - self_mask)
            exp_logits = torch.exp(logits) * (1 - self_mask)
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
            mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
            return -mean_log_prob_pos.mean()

    def pooled_embedding(texts, model, device, num_layers=NUM_LAYERS_FOR_SUPCON,
                         max_length=256):
        enc = qwen_tokenizer(texts, padding=True, truncation=True,
                             max_length=max_length, return_tensors="pt").to(device)
        out = model(**enc, output_hidden_states=True)
        if num_layers is None:
            hidden = out.last_hidden_state
        else:
            hidden = torch.stack(out.hidden_states[1:num_layers + 1], dim=0).mean(dim=0)
        return last_token_pool(hidden, enc["attention_mask"]).float()

    supcon_loss_fn = SupConLoss(temperature=0.07)

    def random_crop(text, rng_, p=0.5, min_words=15, max_words=60):
        # truncate ~half the training texts so the representation also holds
        # up on short (single-sentence) inputs, not only full-length passages
        if rng_.random() > p:
            return text
        words = text.split()
        if len(words) <= min_words:
            return text
        k = int(rng_.integers(min_words, min(max_words, len(words)) + 1))
        return " ".join(words[:k])

    def stratified_batches(buckets, batch_size=64, n_batches=300, rng_=None,
                           augment=True):
        rng_ = rng_ or np.random.default_rng(0)
        keys = list(buckets.keys())
        per_bucket = max(1, batch_size // len(keys))
        for _ in range(n_batches):
            texts, labels_ = [], []
            for topic, label in keys:
                pool = buckets[(topic, label)]
                idx = rng_.integers(0, len(pool), per_bucket)
                for i in idx:
                    t = pool[i]
                    if augment:
                        t = random_crop(t, rng_)
                    texts.append(t)
                labels_ += [label] * per_bucket
            yield texts, labels_

    # --- train/holdout split BEFORE training ---
    from sklearn.model_selection import train_test_split as _tts

    train_buckets, holdout_texts, holdout_labels, holdout_topics = {}, [], [], []
    for (topic, label), texts in buckets_text.items():
        tr, te = _tts(texts, test_size=0.2, random_state=42)
        train_buckets[(topic, label)] = tr
        holdout_texts += te
        holdout_labels += [label] * len(te)
        holdout_topics += [topic] * len(te)
    holdout_labels = np.array(holdout_labels)
    holdout_topics = np.array(holdout_topics)
    print(f"Train pool (SupCon sees this): "
          f"{sum(len(v) for v in train_buckets.values()):,}")
    print(f"Held-out (SupCon never sees this): {len(holdout_texts):,}")

    # --- training loop ---
    _trainable_params = ([p for p in lora_model.parameters() if p.requires_grad]
                         + list(projection_head.parameters()))
    optimizer = torch.optim.AdamW(_trainable_params, lr=2e-4)

    n_batches = min(args.n_batches, 60) if args.smoke else args.n_batches
    for step, (texts, labels_batch) in enumerate(
            stratified_batches(train_buckets, batch_size=32,
                               n_batches=n_batches)):
        labels_t = torch.tensor(labels_batch, device=_supcon_device)
        pooled = pooled_embedding(texts, lora_model, _supcon_device, max_length=128)
        z = projection_head(pooled)
        loss = supcon_loss_fn(z, labels_t)

        if not torch.isfinite(loss):
            print(f"step {step}: non-finite loss, batch skipped")
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(_trainable_params, 1.0)
        optimizer.step()

        if step % 20 == 0:
            print(f"step {step:4d}  loss {loss.item():.4f}")

    lora_model.save_pretrained(os.path.join(OUTPUT_DIR, "supcon_lora_adapter"))
    torch.save(projection_head.state_dict(),
               os.path.join(OUTPUT_DIR, "supcon_projection_head.pt"))
    print(f"Saved adapter + projection head to {OUTPUT_DIR}/")

    # --- data-parallel feature extraction with the trained adapter ---
    if len(_qwen_devices) > 1:
        lora_model_2 = get_peft_model(qwen_models[_qwen_devices[1]], lora_config)
        lora_model_2.load_state_dict(lora_model.state_dict())
        lora_model_2.eval()
        lora_models_by_device = {_qwen_devices[0]: lora_model,
                                 _qwen_devices[1]: lora_model_2}
    else:
        lora_models_by_device = {_qwen_devices[0]: lora_model}

    @torch.no_grad()
    def _extract_on_device(texts, model, device, batch_size=64):
        if not texts:
            return np.zeros((0, 1024), dtype=np.float32)
        model.eval()
        chunks = []
        for i in range(0, len(texts), batch_size):
            pooled = pooled_embedding(texts[i:i + batch_size], model, device)
            chunks.append(pooled.float().cpu().numpy())
        return np.concatenate(chunks, axis=0)

    def extract_supcon_features(texts, batch_size=64):
        devices = list(lora_models_by_device.keys())
        idx_chunks = np.array_split(np.arange(len(texts)), len(devices))
        jobs = [([texts[i] for i in idxs], lora_models_by_device[dev], dev)
                for idxs, dev in zip(idx_chunks, devices)]
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            results_ = list(pool.map(
                lambda a: _extract_on_device(a[0], a[1], a[2], batch_size), jobs))
        return np.concatenate(results_, axis=0)

    # --- holdout evaluation ---
    train_flat_texts, train_flat_labels = [], []
    for (topic, label), texts in train_buckets.items():
        train_flat_texts += texts
        train_flat_labels += [label] * len(texts)
    train_flat_labels = np.array(train_flat_labels)

    train_features = extract_supcon_features(train_flat_texts)
    holdout_features = extract_supcon_features(holdout_texts)

    supcon_clf = LogisticRegression(max_iter=2000).fit(train_features,
                                                       train_flat_labels)
    y_pred = supcon_clf.predict(holdout_features)

    supcon_row = {"approach": "supcon_lora_holdout",
                  "overall": (y_pred == holdout_labels).mean()}
    for topic in ["fiction", "news", "other"]:
        mask = holdout_topics == topic
        if mask.sum() > 0:
            lo, hi = bootstrap_ci(holdout_labels[mask], y_pred[mask])
            supcon_row[topic] = (y_pred[mask] == holdout_labels[mask]).mean()
            supcon_row[f"{topic}_ci"] = f"[{lo:.3f}, {hi:.3f}]"
            supcon_row[f"{topic}_n"] = int(mask.sum())
    print("\n=== SupCon holdout ===")
    print(supcon_row)

    # ------------------------------------------------ 10. unseen-topic test
    SCIENCE_KEYWORDS = [
        "researchers", "study published", "experiment", "laboratory", "hypothesis",
        "quantum", "molecules", "climate change", "vaccine", "genome",
    ]
    SPORTS_KEYWORDS = [
        "championship", "tournament", "quarterback", "playoffs", "coach",
        "season opener", "league", "scored a goal", "halftime", "home run",
    ]
    UNSEEN_TOPICS = {"science": SCIENCE_KEYWORDS, "sports": SPORTS_KEYWORDS}
    TARGET_UNSEEN = 100 if args.smoke else 400

    unseen_pool = perm[SCAN_BUDGET:]
    unseen_buckets = {(t, l): [] for t in UNSEEN_TOPICS for l in (0, 1)}

    for i in unseen_pool:
        entry = ds[int(i)]
        text, label = entry["text"], entry["generated"]
        if matches_any(text, FICTION_KEYWORDS) or matches_any(text, NEWS_KEYWORDS):
            continue  # keep unseen topics disjoint from the training topics
        for t, kws in UNSEEN_TOPICS.items():
            if matches_any(text, kws):
                key = (t, label)
                if len(unseen_buckets[key]) < TARGET_UNSEEN:
                    unseen_buckets[key].append(text)
                break
        if all(len(v) >= TARGET_UNSEEN for v in unseen_buckets.values()):
            break

    for k, v in unseen_buckets.items():
        print(k, len(v))

    unseen_texts, unseen_labels, unseen_topic_tags = [], [], []
    for (t, l), texts in unseen_buckets.items():
        unseen_texts += texts
        unseen_labels += [l] * len(texts)
        unseen_topic_tags += [t] * len(texts)
    unseen_labels = np.array(unseen_labels)
    unseen_topic_tags = np.array(unseen_topic_tags)

    unseen_features = extract_supcon_features(unseen_texts)
    y_pred_unseen = supcon_clf.predict(unseen_features)

    unseen_row = {"approach": "supcon_lora_unseen_topics",
                  "overall": (y_pred_unseen == unseen_labels).mean()}
    for t in UNSEEN_TOPICS:
        mask = unseen_topic_tags == t
        if mask.sum() > 0:
            lo, hi = bootstrap_ci(unseen_labels[mask], y_pred_unseen[mask])
            unseen_row[t] = (y_pred_unseen[mask] == unseen_labels[mask]).mean()
            unseen_row[f"{t}_ci"] = f"[{lo:.3f}, {hi:.3f}]"
            unseen_row[f"{t}_n"] = int(mask.sum())
    print("\n=== SupCon on never-seen topics (science/sports) ===")
    print(unseen_row)

    # ------------------------------------------------ 11. hand-typed comparison
    # OOD demo: single sentences, modern-generator AI prose, translated human
    # news - poor results here measure out-of-distribution robustness, they do
    # not contradict the holdout numbers.
    @torch.no_grad()
    def predict_with_supcon(text):
        lora_model.eval()
        pooled = pooled_embedding([text], lora_model, _supcon_device)
        feat = pooled.float().cpu().numpy()
        proba = supcon_clf.predict_proba(feat)[0]
        pred = supcon_clf.predict(feat)[0]
        return {"prediction": "AI" if pred == 1 else "Human",
                "human_probability": proba[0], "ai_probability": proba[1]}

    def compare_on_text(text, expected_label):
        para = predict_text_origin(text)
        qwen = predict_with_supcon(text)
        print(f"Atteso: {expected_label}")
        print(f"  ParaDetect (DeBERTa): {para['prediction']}  "
              f"(AI prob={para['ai_probability']:.3f})")
        print(f"  Qwen+SupCon:          {qwen['prediction']}  "
              f"(AI prob={qwen['ai_probability']:.3f})")
        print()

    human_news_examples = [
        "BREAKING: Labour says it won't stand against Nigel Farage in the Clacton by-election. As it stands, only Count Binface will run against Farage.",
        "Francesco Renga was forced to disembark from a Ryanair flight following some discussions with the cabin crew. The tensions had already begun at the gate due to the carry-on baggage.",
        "A court in eastern China has sentenced a former city official to death for taking more than 2.2bn yuan ($325m; £243m) in bribes over 30 years. Yang Youlin, who served in various positions in Nanjing city from 1993 to 2023, was also convicted of embezzlement, abuse of power and money laundering, with his ill-gotten gains amounting to one of the highest in recent years. The 69-year-old exploited his roles to help others secure engineering contracts, land transfers and financing, in exchange for money and valuables, said state media.",
        'Volodymyr Zelensky: "I met with the Italian Prime Minister, Giorgia Meloni, for an important discussion. I informed her about the situation in Ukraine. Russia has not stopped attacking our cities and communities. Even today, there have been new attacks with ballistic missiles, and attack drones have also been launched. Unfortunately, there are victims in Kyiv, Kharkiv, and Kherson. Russia has taken the lives of five people. My condolences to their families and loved ones."',
        "Russia unleashed waves of missiles and drones at Ukraine early Monday, killing at least 22 people in attacks that exposed widening gaps in the country's air defenses more than four years into Moscow's full-scale invasion, authorities said.",
        'The U.S. Central Command announces that it has launched "a series of powerful strikes against Iran" in response to Iranian attacks on three commercial ships that were transiting through the Strait of Hormuz.',
        'The IOC Executive Board has provisionally lifted the suspension of the Russian Olympic Committee, in effect since October 12, 2023: "The decision—as stated in a press release—was taken following an in-depth analysis by the IOC Legal Affairs Commission, taking into account that the Russian Olympic Committee no longer includes among its members any regional sports organizations in the territories that fall under the jurisdiction of the National Olympic Committee of Ukraine. Furthermore, the Russian Olympic Committee has confirmed that it does not conduct, nor will it conduct in the future, any activities in those territories."',
        "The crux remains political even before legal: in recent months, Le Pen had deemed it \"impossible\" to run a presidential campaign while wearing an electronic ankle bracelet. An announcement on a decision regarding her potential candidacy, or on a possible handover to Bardella, could come as early as this evening during TF1's 8 p.m. news broadcast. The Pas-de-Calais deputy can still appeal to the Court of Cassation.",
        "Marine Le Pen could, on a legal level, remain in the running for the 2027 French presidential election, but her political future remains up in the air. The Paris Court of Appeal sentenced her today to three years in prison - two of which are suspended and one to be served with an electronic ankle bracelet - and to 45 months of ineligibility, 30 of which are suspended. The 15 months of effective ineligibility have already been considered served, having elapsed since the first-degree conviction on March 31, 2025.",
        "The outbreak of the deadly Bundibugyo species of Ebola in the eastern Democratic Republic of the Congo (DRC) is expanding, while the push to accelerate testing and identify effective treatment options continues, the UN World Health Organization (WHO) said on Tuesday.",
    ]
    for t in human_news_examples:
        compare_on_text(t, expected_label="Human")

    ai_fiction_examples = [
        "The rain fell heavily against the cracked asphalt as Elena pulled her coat tighter, staring at the flashing red lights of the two collided vehicles at the intersection.",
        "A thick cloud of steam rose from the shattered radiator of the black sedan, blending with the cold night air and the pale glow of the streetlamps.",
        "She approached cautiously, the crunch of broken glass beneath her boots breaking the eerie silence that had settled over the quiet suburban neighborhood.",
        "Inside the smaller car, the driver sat motionlessly, their hands still gripping the steering wheel as if waiting for the traffic light to turn green.",
        "Sirens echoed in the distance, a faint and rising wail that seemed to stretch the agonizing minutes into eternity for the gathering crowd of onlookers.",
        "An off-duty paramedic rushed past the yellow police tape, kneeling beside the crumpled driver-side door to assess the victim's pulse.",
        "Despite the violent impact, the passenger compartment had remained largely intact, offering a small glimmer of hope amidst the tangled metal.",
        "Shattered glass littered the pavement like fallen ice, reflecting the harsh blue strobes of the approaching emergency vehicles.",
        "The officer on the scene immediately began taking statements, trying to piece together the sequence of events from the conflicting accounts of shaken witnesses.",
        "By the time the tow trucks arrived to clear the wreckage, the storm had finally begun to subside, leaving behind an uneasy quiet on the wet avenue.",
    ]
    for t in ai_fiction_examples:
        compare_on_text(t, expected_label="AI")

    print("=" * 60)
    print("Same AI sentences aggregated into a single paragraph:")
    print("=" * 60)
    compare_on_text(" ".join(ai_fiction_examples), expected_label="AI (paragraph)")
    compare_on_text(" ".join(ai_fiction_examples[:5]), expected_label="AI (half paragraph)")

    # ------------------------------------------------ 12. ParaDetect on holdout
    # CAVEAT: ParaDetect and the pile dataset share an author - it was very
    # likely trained on these texts, so this number is an inflated upper bound.
    y_pred_para = classify_paradetect_batch(holdout_texts)
    paradetect_row = {"approach": "paradetect_deberta",
                      "overall": (y_pred_para == holdout_labels).mean()}
    for topic in ["fiction", "news", "other"]:
        mask = holdout_topics == topic
        if mask.sum() > 0:
            lo, hi = bootstrap_ci(holdout_labels[mask], y_pred_para[mask])
            paradetect_row[topic] = (y_pred_para[mask] == holdout_labels[mask]).mean()
            paradetect_row[f"{topic}_ci"] = f"[{lo:.3f}, {hi:.3f}]"
            paradetect_row[f"{topic}_n"] = int(mask.sum())
    print("\n=== ParaDetect on the same holdout (leakage caveat applies) ===")
    print(paradetect_row)

    # ------------------------------------------------ 13. era-matched recall
    print("\n=== Era-matched generator recall (all-AI sets, higher = better) ===")
    era_rows = []
    for name in ["GPT-2", "GPT-3 (pairwise-davinci)",
                 "GPT-3 (synthetic-instruct-davinci-pairwise)",
                 "GPT-J (synthetic-instruct-gptj-pairwise)",
                 "ChatGPT (HC3)", "ChatGPT (emergentmind)"]:
        phrases = extracted_sets.get(name) or []
        if not phrases:
            continue
        supcon_preds = supcon_clf.predict(extract_supcon_features(phrases))
        para_preds = classify_paradetect_batch(phrases)
        era_rows.append({"source": name, "n": len(phrases),
                         "supcon_ai_recall": supcon_preds.mean(),
                         "paradetect_ai_recall": para_preds.mean()})
        print(f"{name:48s}  Qwen+SupCon: {supcon_preds.mean():.0%} AI   "
              f"ParaDetect: {para_preds.mean():.0%} AI   (n={len(phrases)})")
    era_df = pd.DataFrame(era_rows)
    era_df.to_csv(os.path.join(OUTPUT_DIR, "era_matched_recall.csv"), index=False)

    # ------------------------------------------------ 14. fresh GPT-2 XL test
    from transformers import AutoModelForCausalLM

    GPT2_GEN_REPO = "gpt2-xl"   # same size as the pile's xl-1542M source
    gen_tok = AutoTokenizer.from_pretrained(GPT2_GEN_REPO)
    gen_tok.pad_token = gen_tok.eos_token
    gen_model = AutoModelForCausalLM.from_pretrained(
        GPT2_GEN_REPO, torch_dtype=torch.float16).to(_supcon_device)
    gen_model.eval()

    GEN_PROMPTS = [
        "The story begins on a rainy night, when",
        "Once upon a time, in a small village by the sea,",
        "The protagonist of the novel walked into the courtroom and",
        "The car crash on the highway left",
        "Police said the suspect was arrested after",
        "Authorities confirmed that the robbery took place",
        "The murder investigation took an unexpected turn when",
        "Witnesses told reporters that the two cars collided",
        "In the third chapter of the novel, the detective",
        "Breaking news: a man was shot dead in",
    ]

    @torch.no_grad()
    def generate_gpt2(prompt, n=4, max_new_tokens=120):
        enc = gen_tok(prompt, return_tensors="pt").to(_supcon_device)
        out = gen_model.generate(**enc, do_sample=True, top_k=40,
                                 max_new_tokens=max_new_tokens,
                                 num_return_sequences=n,
                                 pad_token_id=gen_tok.eos_token_id)
        return [gen_tok.decode(o, skip_special_tokens=True) for o in out]

    torch.manual_seed(42)
    fresh_gpt2_texts = []
    for p in GEN_PROMPTS:
        fresh_gpt2_texts += generate_gpt2(p, n=1 if args.smoke else 4)
    print(f"\nGenerated {len(fresh_gpt2_texts)} fresh GPT-2 XL texts")
    print("--- sample ---")
    print(fresh_gpt2_texts[0][:400])

    supcon_preds = supcon_clf.predict(extract_supcon_features(fresh_gpt2_texts))
    para_preds = classify_paradetect_batch(fresh_gpt2_texts)
    print(f"\nFresh GPT-2 XL text (never seen by any model, n={len(fresh_gpt2_texts)}):")
    print(f"  Qwen+SupCon AI recall: {supcon_preds.mean():.1%}")
    print(f"  ParaDetect  AI recall: {para_preds.mean():.1%}")

    del gen_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------ 15. final table
    all_results = pd.DataFrame(results + [supcon_row, unseen_row, paradetect_row])
    print("\n=== Final comparison table ===")
    print(all_results)
    all_results.to_csv(os.path.join(OUTPUT_DIR, "all_results.csv"), index=False)
    print(f"\nAll CSVs, plots and the trained adapter are in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
