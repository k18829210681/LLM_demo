#!/usr/bin/env python3
"""
Helper script: Embed chunks with a single embedding model in a separate process.
This allows full GPU memory cleanup when the process exits.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from vllm import LLM

from utils import load_config, read_jsonl, setup_output_dirs

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


def main():
    if len(sys.argv) != 2:
        print("Usage: step5_sub_embed_chunks.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]

    config = load_config()
    dirs = setup_output_dirs(config)

    BATCH_SIZE = config["batch_size"]
    MAX_MODEL_LEN = config["max_model_len"]
    CHUNKS_FILE = dirs["chunks"] / "chunks_dedu.jsonl"

    print(f"Embedding chunks with {model_name}...")

    # Load chunks
    print(f"Loading chunks from {CHUNKS_FILE}...")
    chunks = read_jsonl(CHUNKS_FILE)
    print(f"Loaded {len(chunks)} chunks")

    chunk_ids = [chunk["id"] for chunk in chunks]
    texts = [chunk["text"] for chunk in chunks]

    # Load model
    model_id = config["embedding_models"][model_name]
    llm = LLM(
        model=model_id,
        task="embed",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.9,
    )

    # Embed in batches
    all_embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"Embedding chunks ({model_name})"):
        batch_texts = texts[i : i + BATCH_SIZE]
        outputs = llm.embed(batch_texts)
        embeddings = [o.outputs.embedding for o in outputs]
        all_embeddings.extend(embeddings)

    embeddings = np.array(all_embeddings, dtype=np.float32)

    # L2 normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    print(f"Embeddings shape: {embeddings.shape}")

    # Save embeddings
    emb_file = dirs["embeddings"] / f"{model_name}_chunks.npy"
    print(f"Saving embeddings to {emb_file}...")
    np.save(emb_file, embeddings)

    # Save IDs only once (same for all models)
    ids_file = dirs["embeddings"] / "chunks_ids.json"
    if not ids_file.exists():
        print(f"Saving IDs to {ids_file}...")
        with open(ids_file, "w") as f:
            json.dump(chunk_ids, f)
    else:
        print(f"IDs file already exists: {ids_file}")

    print(f"✓ Saved chunk embeddings for {model_name}")
    print("Process will exit and free all GPU memory...")


if __name__ == "__main__":
    main()
