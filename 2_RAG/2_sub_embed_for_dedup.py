#!/usr/bin/env python3
"""
Helper script: Embed chunks for deduplication in a separate process.
This allows full GPU memory cleanup when the process exits.
"""

import gc
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
    config = load_config()
    dirs = setup_output_dirs(config)

    BATCH_SIZE = config["batch_size"]
    EMB_MODEL = config["deduplication"]["embedding_model"]
    EMB_MODEL_ID = config["embedding_models"][EMB_MODEL]
    MAX_MODEL_LEN = config["max_model_len"]

    # Input/output paths
    chunks_file = dirs["chunks"] / "chunks.jsonl"
    embeddings_file = dirs["deduplication"] / "temp_embeddings.npy"

    # Load chunks
    print(f"Loading chunks from {chunks_file}...")
    chunks = read_jsonl(chunks_file)

    # Testing mode
    if len(chunks) > 2000 and os.getenv("TESTING_MODE", "0") == "1":
        print(f"⚠ TESTING MODE: Using only first 2000 chunks (out of {len(chunks)})")
        chunks = chunks[:2000]

    print(f"Loaded {len(chunks)} chunks")

    # Embed
    print(f"\nEmbedding with {EMB_MODEL}...")
    llm = LLM(
        model=EMB_MODEL_ID,
        task="embed",
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.9,  # Can use full GPU since this is only process
    )

    texts = [chunk["text"] for chunk in chunks]

    all_embeddings = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding batches"):
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
    print(f"Saving embeddings to {embeddings_file}...")
    np.save(embeddings_file, embeddings)

    print("✓ Embedding completed successfully!")
    print("Process will exit and free all GPU memory...")

if __name__ == "__main__":
    main()
