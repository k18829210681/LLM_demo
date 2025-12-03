#!/usr/bin/env python3
"""
Step 6: Reranking

1. Load chunk and query embeddings for each embedding model
2. Retrieve initial top-K chunks using embedding similarity
3. Rerank the top-K chunks using reranker models
4. Save reranking results for benchmarking

Each reranker model runs in a separate subprocess to ensure complete GPU memory cleanup.

Outputs:
  - outputs/reranking/{emb_model}_{rerank_model}_{instruction}_reranking.jsonl
    Format: {"query_id": int, "chunk_ids": [int], "scores": [float]}
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from utils import load_config, load_prompts, setup_output_dirs

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

# ====================================================================
# Configuration
# ====================================================================

BASE_DIR = Path(__file__).resolve().parent
config = load_config()
prompts = load_prompts()

# Reranking settings
EMB_MODELS = config["embedding"]["embedding_models"]
RERANK_MODELS = config["reranking"]["reranking_models"]
QUERY_INSTRUCTIONS = config["embedding"]["query_instructions"]  # Use same instructions as embedding

# Setup output directories
dirs = setup_output_dirs(config)


# ====================================================================
# Rerank (in separate process)
# ====================================================================


def rerank_subprocess(emb_model: str, rerank_model: str, instruction: str):
    """
    Run reranking in a separate subprocess to ensure full GPU memory cleanup.
    This allows the OS to reclaim all GPU memory when the subprocess exits.
    """
    print("\n" + "=" * 70)
    print(f"Reranking {emb_model}/{instruction} with {rerank_model} in separate process")
    print("=" * 70)

    # Run reranking script in subprocess
    rerank_script = BASE_DIR / "6_sub_rerank.py"
    env = os.environ.copy()

    result = subprocess.run(
        [sys.executable, str(rerank_script), emb_model, rerank_model, instruction],
        env=env,
        capture_output=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Reranking subprocess for {emb_model}/{rerank_model}/{instruction} failed with code {result.returncode}"
        )

    # Wait for GPU memory to be fully released by OS
    print("Waiting 5 seconds for OS to reclaim GPU memory from subprocess...")
    time.sleep(5)


# ====================================================================
# Main
# ====================================================================


def main():
    print("=" * 70)
    print("STEP 6: RERANKING")
    print("=" * 70)
    print(f"Embedding models: {EMB_MODELS}")
    print(f"Reranker models: {RERANK_MODELS}")
    print(f"Query instructions: {QUERY_INSTRUCTIONS}")
    print("=" * 70)

    # Process each combination in separate subprocesses for full GPU cleanup
    # Loop order: instruction -> emb_model -> rerank_model
    # This ensures each (emb_model, instruction) pair is tested with all rerankers
    for instruction in QUERY_INSTRUCTIONS:
        for emb_model in EMB_MODELS:
            for rerank_model in RERANK_MODELS:
                rerank_subprocess(emb_model, rerank_model, instruction)

    print(f"\n✓ Step 6 completed successfully!")


if __name__ == "__main__":
    main()
