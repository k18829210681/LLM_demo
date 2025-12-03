#!/usr/bin/env python3
"""
Step 2: Deduplication

1. Embed all chunks using the configured embedding model
2. Use FAISS to find similar chunks
3. For very similar chunks (>= similarity_max): auto-mark as duplicates
4. For moderately similar chunks (similarity_min to similarity_max): use LLM to judge
5. Apply decision logic ("and" or "or") to combine judgments from multiple LLMs
6. Save deduplicated chunks and sample from them

Outputs:
  - outputs/deduplication/{model}_dedu.jsonl - Deduplication judgments for each LLM
  - outputs/chunks/chunks_dedu.jsonl - Final deduplicated chunks
  - outputs/chunks/chunks_dedu_sampled.jsonl - Sampled subset (if enabled)
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

import faiss
import numpy as np
import torch
from tqdm import tqdm

from utils import (
    load_config,
    load_prompts,
    read_jsonl,
    setup_output_dirs,
    write_jsonl_incremental,
)

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

# ====================================================================
# Configuration
# ====================================================================

BASE_DIR = Path(__file__).resolve().parent
config = load_config()
prompts = load_prompts()

RANDOM_SEED = config["random_seed"]
BATCH_SIZE = config["batch_size"]

# Deduplication settings
LLM_MODELS = config["deduplication"]["llm_models"]
MAX_MODEL_LEN = config["max_model_len"]
MAX_NEW_TOKENS = config["max_new_tokens"]

EMB_MODEL = config["deduplication"]["embedding_model"]
EMB_MODEL_ID = config["embedding_models"][EMB_MODEL]

TOP_K = config["deduplication"]["top_k"]
SIM_MIN = config["deduplication"]["similarity_min"]
SIM_MAX = config["deduplication"]["similarity_max"]
DECISION = config["deduplication"]["decision"]
SAMPLE_SIZE = config["deduplication"]["sample_size"]

# Setup output directories
dirs = setup_output_dirs(config)
CHUNKS_FILE = dirs["chunks"] / "chunks.jsonl"
CHUNKS_DEDU_FILE = dirs["chunks"] / "chunks_dedu.jsonl"
CHUNKS_DEDU_SAMPLED_FILE = dirs["chunks"] / "chunks_dedu_sampled.jsonl"

random.seed(RANDOM_SEED)


# ====================================================================
# Step 1: Embed Chunks (in separate process for full GPU cleanup)
# ====================================================================


def embed_chunks_subprocess() -> np.ndarray:
    """
    Run embedding in a separate subprocess to ensure full GPU memory cleanup.
    This allows the OS to reclaim all GPU memory when the subprocess exits.
    """
    print("\n" + "=" * 70)
    print("SUBSTEP 1: Embedding chunks in separate process")
    print("=" * 70)

    # Run embedding script in subprocess
    embed_script = BASE_DIR / "2_sub_embed_for_dedup.py"
    env = os.environ.copy()

    result = subprocess.run(
        [sys.executable, str(embed_script)],
        env=env,
        capture_output=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Embedding subprocess failed with code {result.returncode}")

    # Load embeddings from file
    embeddings_file = dirs["deduplication"] / "temp_embeddings.npy"
    print(f"\nLoading embeddings from {embeddings_file}...")
    embeddings = np.load(embeddings_file)
    print(f"Loaded embeddings shape: {embeddings.shape}")

    # Wait for GPU memory to be fully released by OS
    print("Waiting 5 seconds for OS to reclaim GPU memory from subprocess...")
    time.sleep(5)

    return embeddings


# ====================================================================
# Step 2: Find Similar Pairs with FAISS
# ====================================================================


def find_similar_pairs(
    embeddings: np.ndarray, chunk_ids: List[int]
) -> List[tuple[int, int, float]]:
    """Find pairs of similar chunks using FAISS."""
    print(f"\nBuilding FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine for normalized vectors)
    index.add(embeddings)

    print(f"Searching for top-{TOP_K} neighbors...")
    similarities, indices = index.search(embeddings, TOP_K + 1)  # +1 to skip self

    # Collect pairs
    pairs = []
    seen = set()

    for i in range(len(chunk_ids)):
        for j in range(1, TOP_K + 1):  # Skip first (self)
            neighbor_idx = indices[i][j]
            similarity = float(similarities[i][j])

            if similarity < SIM_MIN:
                continue

            # Ensure we don't duplicate pairs (i,j) and (j,i)
            id_i, id_j = chunk_ids[i], chunk_ids[neighbor_idx]
            pair_key = tuple(sorted([id_i, id_j]))

            if pair_key not in seen:
                seen.add(pair_key)
                pairs.append((id_i, id_j, similarity))

    print(f"Found {len(pairs)} similar pairs (similarity >= {SIM_MIN})")
    return pairs


# ====================================================================
# Step 3: Judge Duplicates with LLM (in separate process)
# ====================================================================


def judge_with_llm_subprocess(
    model_name: str, pairs: List[tuple[int, int, float]]
) -> Dict[tuple[int, int], Dict]:
    """
    Run LLM judging in a separate subprocess to ensure full GPU memory cleanup.
    This allows the OS to reclaim all GPU memory when the subprocess exits.
    """
    print("\n" + "=" * 70)
    print(f"SUBSTEP 3: Judging pairs with {model_name} in separate process")
    print("=" * 70)

    # Prepare pairs that need LLM judging
    pairs_to_judge = [(id_i, id_j, sim) for id_i, id_j, sim in pairs if sim < SIM_MAX]

    if not pairs_to_judge:
        print("No pairs need LLM judging (all auto-marked)")
        return {}

    print(f"Judging {len(pairs_to_judge)} pairs (similarity {SIM_MIN} to {SIM_MAX})")

    # Save temp file for subprocess
    pairs_file = dirs["deduplication"] / "temp_pairs_to_judge.json"

    print(f"Saving pairs to {pairs_file}...")
    with open(pairs_file, "w") as f:
        json.dump(pairs_to_judge, f)

    # Run judge script in subprocess
    judge_script = BASE_DIR / "2_sub_judge_for_dedup.py"
    env = os.environ.copy()

    result = subprocess.run(
        [sys.executable, str(judge_script), model_name],
        env=env,
        capture_output=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Judging subprocess for {model_name} failed with code {result.returncode}"
        )

    # Load results from output file
    output_file = dirs["deduplication"] / f"{model_name}_dedu.jsonl"
    results = {}

    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    pair_key = (result["id_i"], result["id_j"])
                    results[pair_key] = result

    print(f"Loaded {len(results)} judgments from {output_file}")

    # Wait for GPU memory to be fully released by OS
    print("Waiting 5 seconds for OS to reclaim GPU memory from subprocess...")
    time.sleep(5)

    return results


# ====================================================================
# Step 4: Combine Judgments and Apply Decision Logic
# ====================================================================


def determine_duplicates(
    pairs: List[tuple[int, int, float]],
    llm_judgments: Dict[str, Dict[tuple[int, int], Dict]],
) -> Set[int]:
    """Determine which chunks to remove based on decision logic."""
    print(f"\nApplying decision logic: {DECISION}")

    to_remove = set()

    for id_i, id_j, similarity in pairs:
        # Auto-mark as duplicate if similarity >= SIM_MAX
        if similarity >= SIM_MAX:
            # Remove higher ID
            to_remove.add(max(id_i, id_j))
            continue

        # Check LLM judgments
        pair_key = (id_i, id_j)
        judgments = [
            model_results.get(pair_key, {}).get("decision") == "yes"
            for model_results in llm_judgments.values()
        ]

        # Apply decision logic
        if DECISION == "and":
            # Remove only if ALL models agree it's a duplicate
            if all(judgments):
                to_remove.add(max(id_i, id_j))
        elif DECISION == "or":
            # Remove if ANY model thinks it's a duplicate
            if any(judgments):
                to_remove.add(max(id_i, id_j))

    print(f"Chunks to remove: {len(to_remove)}")
    return to_remove


# ====================================================================
# Step 5: Save Deduplicated Chunks and Sample
# ====================================================================


def save_deduplicated_and_sample(chunks: List[Dict], to_remove: Set[int]):
    """Save deduplicated chunks and create sample."""
    print(f"\nSaving deduplicated chunks...")

    # Filter out duplicates
    chunks_dedu = [chunk for chunk in chunks if chunk["id"] not in to_remove]

    print(f"Original chunks: {len(chunks)}")
    print(f"After deduplication: {len(chunks_dedu)} (removed {len(to_remove)})")

    # Save deduplicated chunks
    write_jsonl_incremental(chunks_dedu, CHUNKS_DEDU_FILE, batch_size=BATCH_SIZE)
    print(f"Saved to {CHUNKS_DEDU_FILE}")

    # Sample if enabled
    if SAMPLE_SIZE is not None and SAMPLE_SIZE < len(chunks_dedu):
        print(f"\nSampling {SAMPLE_SIZE} chunks from deduplicated set...")
        sampled = random.sample(chunks_dedu, SAMPLE_SIZE)
        write_jsonl_incremental(sampled, CHUNKS_DEDU_SAMPLED_FILE, batch_size=BATCH_SIZE)
        print(f"Saved to {CHUNKS_DEDU_SAMPLED_FILE}")
    else:
        print(f"\nSampling disabled (sample_size={SAMPLE_SIZE}, total={len(chunks_dedu)})")


# ====================================================================
# Main
# ====================================================================


def main():
    print("=" * 70)
    print("STEP 2: DEDUPLICATION")
    print("=" * 70)
    print(f"Embedding model: {EMB_MODEL}")
    print(f"LLM models: {LLM_MODELS}")
    print(f"Similarity range: [{SIM_MIN}, {SIM_MAX}]")
    print(f"Decision logic: {DECISION}")
    print(f"Sample size: {SAMPLE_SIZE}")
    print("=" * 70)

    # Load chunks
    print(f"\nLoading chunks from {CHUNKS_FILE}...")
    chunks = read_jsonl(CHUNKS_FILE)
    print(f"Loaded {len(chunks)} chunks")

    # Testing mode: use only first 2000 chunks if file is being tested
    if len(chunks) > 2000 and os.getenv("TESTING_MODE", "0") == "1":
        print(f"\n⚠ TESTING MODE: Using only first 2000 chunks (out of {len(chunks)})")
        chunks = chunks[:2000]

    chunk_ids = [chunk["id"] for chunk in chunks]

    # Step 1: Embed chunks in separate process (ensures full GPU memory cleanup)
    embeddings = embed_chunks_subprocess()

    # Step 2: Find similar pairs
    pairs = find_similar_pairs(embeddings, chunk_ids)

    # Step 3: Judge with LLMs (each in separate process for full GPU cleanup)
    llm_judgments = {}
    for model_name in LLM_MODELS:
        judgments = judge_with_llm_subprocess(model_name, pairs)
        llm_judgments[model_name] = judgments

    # Step 4: Determine duplicates
    to_remove = determine_duplicates(pairs, llm_judgments)

    # Step 5: Save results
    save_deduplicated_and_sample(chunks, to_remove)

    print(f"\n✓ Step 2 completed successfully!")


if __name__ == "__main__":
    main()
