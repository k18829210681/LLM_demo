#!/usr/bin/env python3
"""
Step 5: Embedding

1. Embed all deduplicated chunks with each embedding model
2. Embed all selected queries with each embedding model and instruction variant

Each embedding model runs in a separate subprocess to ensure complete GPU memory cleanup.

Outputs:
  - outputs/embeddings/{model}_chunks.npy - Chunk embeddings per model
  - outputs/embeddings/chunks_ids.json - Chunk ID mapping (saved once)
  - outputs/embeddings/{model}_{instruction}_query.npy - Query embeddings per model/instruction
  - outputs/embeddings/query_ids.json - Query ID mapping (saved once)
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

# Embedding settings
EMB_MODELS = config["embedding"]["embedding_models"]
QUERY_INSTRUCTIONS = config["embedding"]["query_instructions"]

# Setup output directories
dirs = setup_output_dirs(config)


# ====================================================================
# Embed Chunks (in separate process)
# ====================================================================


def embed_chunks_subprocess(model_name: str):
    """
    Run chunk embedding in a separate subprocess to ensure full GPU memory cleanup.
    This allows the OS to reclaim all GPU memory when the subprocess exits.
    """
    print("\n" + "=" * 70)
    print(f"Embedding chunks with {model_name} in separate process")
    print("=" * 70)

    # Run embedding script in subprocess
    embed_script = BASE_DIR / "5_sub_embed_chunks.py"
    env = os.environ.copy()

    result = subprocess.run(
        [sys.executable, str(embed_script), model_name],
        env=env,
        capture_output=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Chunk embedding subprocess for {model_name} failed with code {result.returncode}"
        )

    # Wait for GPU memory to be fully released by OS
    print("Waiting 5 seconds for OS to reclaim GPU memory from subprocess...")
    time.sleep(5)


# ====================================================================
# Embed Queries (in separate process)
# ====================================================================


def embed_queries_subprocess(model_name: str):
    """
    Run query embedding in a separate subprocess to ensure full GPU memory cleanup.
    This allows the OS to reclaim all GPU memory when the subprocess exits.
    """
    print("\n" + "=" * 70)
    print(f"Embedding queries with {model_name} in separate process")
    print("=" * 70)

    # Run embedding script in subprocess
    embed_script = BASE_DIR / "5_sub_embed_queries.py"
    env = os.environ.copy()

    result = subprocess.run(
        [sys.executable, str(embed_script), model_name],
        env=env,
        capture_output=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Query embedding subprocess for {model_name} failed with code {result.returncode}"
        )

    # Wait for GPU memory to be fully released by OS
    print("Waiting 5 seconds for OS to reclaim GPU memory from subprocess...")
    time.sleep(5)


# ====================================================================
# Main
# ====================================================================


def main():
    print("=" * 70)
    print("STEP 5: EMBEDDING")
    print("=" * 70)
    print(f"Embedding models: {EMB_MODELS}")
    print(f"Query instructions: {QUERY_INSTRUCTIONS}")
    print("=" * 70)

    # Process each embedding model in separate subprocesses for full GPU cleanup
    for model_name in EMB_MODELS:
        print(f"\n{'=' * 70}")
        print(f"Processing with {model_name}")
        print(f"{'=' * 70}")

        embed_chunks_subprocess(model_name)
        embed_queries_subprocess(model_name)

    print(f"\n✓ Step 5 completed successfully!")


if __name__ == "__main__":
    main()
