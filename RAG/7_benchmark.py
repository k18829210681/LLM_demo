#!/usr/bin/env python3
"""
Step 7: Benchmark

Benchmark all combinations of:
  - Embedding models
  - Query instructions
  - Embedding dimensions (truncated)

Metrics:
  - Top-k accuracy (k = 1, 10, 20, 50)
  - MRR (Mean Reciprocal Rank)
  - MAP (Mean Average Precision)

Also benchmark reranking results:
  - Top-k accuracy (k = 1, 3, 5)

Outputs:
  - outputs/benchmarks/benchmark_results.json - Detailed JSON results
  - outputs/benchmarks/benchmark_results.csv - CSV table for spreadsheet viewing
  - outputs/benchmarks/benchmark_reranking_results.json - Reranking results
  - outputs/benchmarks/benchmark_reranking_results.csv - Reranking CSV table
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from utils import load_config, read_jsonl, setup_output_dirs

# ====================================================================
# Configuration
# ====================================================================

BASE_DIR = Path(__file__).resolve().parent
config = load_config()

# Benchmark settings
EMB_MODELS = config["embedding"]["embedding_models"]
RERANK_MODELS = config["reranking"]["reranking_models"]
QUERY_INSTRUCTIONS = config["embedding"]["query_instructions"]
DIMENSIONS = config["benchmark"]["embedding_dimensions"]
TOP_K_VALUES = config["benchmark"]["top_k_values"]
RERANK_TOP_K_VALUES = config["benchmark"]["reranking_top_k_values"]
COMPUTE_MRR = config["benchmark"]["compute_mrr"]
COMPUTE_MAP = config["benchmark"]["compute_map"]

# Setup output directories
dirs = setup_output_dirs(config)


# ====================================================================
# Load Embeddings
# ====================================================================


def load_embeddings(model: str, instruction: str = None) -> tuple[np.ndarray, List[int]]:
    """Load embeddings and IDs from file."""
    if instruction is None:
        # Load chunk embeddings
        emb_file = dirs["embeddings"] / f"{model}_chunks.npy"
        ids_file = dirs["embeddings"] / "chunks_ids.json"  # Shared across all models
    else:
        # Load query embeddings
        emb_file = dirs["embeddings"] / f"{model}_{instruction}_query.npy"
        ids_file = dirs["embeddings"] / "query_ids.json"  # Shared across all models

    embeddings = np.load(emb_file)

    with open(ids_file, "r") as f:
        ids = json.load(f)

    return embeddings, ids


# ====================================================================
# Metrics
# ====================================================================


def compute_top_k_accuracy(
    similarities: np.ndarray, query_ids: List[int], chunk_ids: List[int], k: int
) -> float:
    """
    Compute top-k accuracy.

    For each query, check if the ground-truth chunk appears in top-k results.
    """
    n_queries = len(query_ids)
    correct = 0

    for i in range(n_queries):
        # Get top-k chunk indices
        top_k_indices = np.argsort(-similarities[i])[:k]
        top_k_chunk_ids = [chunk_ids[idx] for idx in top_k_indices]

        # Check if ground truth is in top-k
        if query_ids[i] in top_k_chunk_ids:
            correct += 1

    return correct / n_queries


def compute_mrr(
    similarities: np.ndarray, query_ids: List[int], chunk_ids: List[int]
) -> float:
    """
    Compute Mean Reciprocal Rank.

    MRR = average of 1/rank for ground-truth chunk.
    """
    n_queries = len(query_ids)
    reciprocal_ranks = []

    for i in range(n_queries):
        # Get sorted chunk indices (highest similarity first)
        sorted_indices = np.argsort(-similarities[i])
        sorted_chunk_ids = [chunk_ids[idx] for idx in sorted_indices]

        # Find rank of ground truth (1-indexed)
        try:
            rank = sorted_chunk_ids.index(query_ids[i]) + 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            # Ground truth not found
            reciprocal_ranks.append(0.0)

    return np.mean(reciprocal_ranks)


def compute_map(
    similarities: np.ndarray, query_ids: List[int], chunk_ids: List[int]
) -> float:
    """
    Compute Mean Average Precision.

    For single relevant document per query, MAP = MRR.
    """
    return compute_mrr(similarities, query_ids, chunk_ids)


# ====================================================================
# Benchmark
# ====================================================================


def benchmark_combination(
    emb_model: str, instruction: str, dim: int
) -> Dict:
    """Benchmark a specific combination of model, instruction, and dimension."""
    # Load chunk embeddings
    chunk_embs, chunk_ids = load_embeddings(emb_model)

    # Load query embeddings
    query_embs, query_ids = load_embeddings(emb_model, instruction)

    # Get native dimension from embedding shape
    native_dim = chunk_embs.shape[1]

    # Skip if requested dimension > native dimension
    if dim > native_dim:
        return None

    # Truncate embeddings to requested dimension
    chunk_embs_trunc = chunk_embs[:, :dim]
    query_embs_trunc = query_embs[:, :dim]

    # Re-normalize after truncation
    chunk_embs_trunc = chunk_embs_trunc / np.linalg.norm(
        chunk_embs_trunc, axis=1, keepdims=True
    )
    query_embs_trunc = query_embs_trunc / np.linalg.norm(
        query_embs_trunc, axis=1, keepdims=True
    )

    # Compute similarity matrix (cosine similarity via dot product)
    similarities = np.dot(query_embs_trunc, chunk_embs_trunc.T)

    # Compute metrics
    results = {
        "embedding_model": emb_model,
        "query_instruction": instruction,
        "dimension": dim,
        "native_dimension": native_dim,
        "n_queries": len(query_ids),
        "n_chunks": len(chunk_ids),
    }

    # Top-k accuracy
    for k in TOP_K_VALUES:
        acc = compute_top_k_accuracy(similarities, query_ids, chunk_ids, k)
        results[f"top_{k}_accuracy"] = round(acc, 4)

    # MRR
    if COMPUTE_MRR:
        mrr = compute_mrr(similarities, query_ids, chunk_ids)
        results["mrr"] = round(mrr, 4)

    # MAP
    if COMPUTE_MAP:
        map_score = compute_map(similarities, query_ids, chunk_ids)
        results["map"] = round(map_score, 4)

    return results


# ====================================================================
# Reranking Benchmark
# ====================================================================


def compute_reranking_top_k_accuracy(
    reranking_results: List[Dict], k: int
) -> float:
    """
    Compute top-k accuracy for reranking results.

    Args:
        reranking_results: List of dicts with 'query_id', 'chunk_ids', 'scores'
        k: Top-k to check

    Returns:
        Top-k accuracy
    """
    correct = 0
    total = len(reranking_results)

    for result in reranking_results:
        query_id = result["query_id"]
        chunk_ids = result["chunk_ids"]
        scores = result["scores"]

        # Sort by scores descending
        sorted_indices = np.argsort(-np.array(scores))
        top_k_chunk_ids = [chunk_ids[idx] for idx in sorted_indices[:k]]

        # Check if ground truth is in top-k
        if query_id in top_k_chunk_ids:
            correct += 1

    return correct / total if total > 0 else 0.0


def benchmark_reranking_combination(
    emb_model: str, rerank_model: str, instruction: str
) -> Dict:
    """Benchmark a specific reranking combination."""
    # Load reranking results
    rerank_file = dirs["reranking"] / f"{emb_model}_{rerank_model}_{instruction}_reranking.jsonl"

    if not rerank_file.exists():
        return None

    reranking_results = read_jsonl(rerank_file)

    if not reranking_results:
        return None

    # Compute metrics
    results = {
        "embedding_model": emb_model,
        "reranker_model": rerank_model,
        "query_instruction": instruction,
        "n_queries": len(reranking_results),
    }

    # Top-k accuracy
    for k in RERANK_TOP_K_VALUES:
        acc = compute_reranking_top_k_accuracy(reranking_results, k)
        results[f"top_{k}_accuracy"] = round(acc, 4)

    return results


# ====================================================================
# Main
# ====================================================================


def main():
    print("=" * 70)
    print("STEP 7: BENCHMARK")
    print("=" * 70)
    print(f"Embedding models: {EMB_MODELS}")
    print(f"Query instructions: {QUERY_INSTRUCTIONS}")
    print(f"Dimensions: {DIMENSIONS}")
    print(f"Top-k values: {TOP_K_VALUES}")
    print("=" * 70)

    # ====================================================================
    # Part 1: Benchmark Embeddings
    # ====================================================================
    print("\n" + "=" * 70)
    print("PART 1: BENCHMARKING EMBEDDINGS")
    print("=" * 70)

    # Benchmark all combinations
    all_results = []

    total_combinations = len(EMB_MODELS) * len(QUERY_INSTRUCTIONS) * len(DIMENSIONS)

    with tqdm(total=total_combinations, desc="Benchmarking embeddings") as pbar:
        for emb_model in EMB_MODELS:
            for instruction in QUERY_INSTRUCTIONS:
                for dim in DIMENSIONS:
                    result = benchmark_combination(emb_model, instruction, dim)

                    if result is not None:
                        all_results.append(result)

                    pbar.update(1)

    # Save JSON results
    json_file = dirs["benchmarks"] / "benchmark_results.json"
    with open(json_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved JSON results to {json_file}")

    # Generate CSV table
    csv_file = dirs["benchmarks"] / "benchmark_results.csv"
    generate_csv_table(all_results, csv_file)
    print(f"Saved CSV table to {csv_file}")

    print(f"\n✓ Embedding benchmarks completed!")
    print(f"  Total benchmarks: {len(all_results)}")

    # ====================================================================
    # Part 2: Benchmark Reranking
    # ====================================================================
    print("\n" + "=" * 70)
    print("PART 2: BENCHMARKING RERANKING")
    print("=" * 70)
    print(f"Reranker models: {RERANK_MODELS}")
    print(f"Reranking top-k values: {RERANK_TOP_K_VALUES}")
    print("=" * 70)

    rerank_results = []

    total_rerank_combinations = len(EMB_MODELS) * len(RERANK_MODELS) * len(QUERY_INSTRUCTIONS)

    with tqdm(total=total_rerank_combinations, desc="Benchmarking reranking") as pbar:
        for instruction in QUERY_INSTRUCTIONS:
            for emb_model in EMB_MODELS:
                for rerank_model in RERANK_MODELS:
                    result = benchmark_reranking_combination(emb_model, rerank_model, instruction)

                    if result is not None:
                        rerank_results.append(result)

                    pbar.update(1)

    # Save reranking JSON results
    rerank_json_file = dirs["benchmarks"] / "benchmark_reranking_results.json"
    with open(rerank_json_file, "w") as f:
        json.dump(rerank_results, f, indent=2)

    print(f"\nSaved reranking JSON results to {rerank_json_file}")

    # Generate reranking CSV table
    rerank_csv_file = dirs["benchmarks"] / "benchmark_reranking_results.csv"
    generate_reranking_csv_table(rerank_results, rerank_csv_file)
    print(f"Saved reranking CSV table to {rerank_csv_file}")

    print(f"\n✓ Reranking benchmarks completed!")
    print(f"  Total benchmarks: {len(rerank_results)}")

    print(f"\n✓ Step 7 completed successfully!")
    print(f"\nTotal embedding benchmarks: {len(all_results)}")
    print(f"Total reranking benchmarks: {len(rerank_results)}")


# ====================================================================
# CSV Table
# ====================================================================


def generate_csv_table(results: List[Dict], output_file: Path):
    """Generate a CSV table for spreadsheet viewing."""
    if not results:
        return

    # Build headers
    headers = ["embedding_model", "query_instruction", "dimension", "native_dimension", "n_queries", "n_chunks"]
    for k in TOP_K_VALUES:
        headers.append(f"top_{k}_accuracy")
    if COMPUTE_MRR:
        headers.append("mrr")
    if COMPUTE_MAP:
        headers.append("map")

    # Write CSV
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for result in results:
            # Filter to only include the headers we want
            row = {k: result[k] for k in headers if k in result}
            writer.writerow(row)


def generate_reranking_csv_table(results: List[Dict], output_file: Path):
    """Generate a CSV table for reranking results."""
    if not results:
        return

    # Build headers
    headers = ["embedding_model", "reranker_model", "query_instruction", "n_queries"]
    for k in RERANK_TOP_K_VALUES:
        headers.append(f"top_{k}_accuracy")

    # Write CSV
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for result in results:
            # Filter to only include the headers we want
            row = {k: result[k] for k in headers if k in result}
            writer.writerow(row)


if __name__ == "__main__":
    main()
