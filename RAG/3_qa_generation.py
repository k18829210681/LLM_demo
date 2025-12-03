#!/usr/bin/env python3
"""
Step 3: QA Generation

Generate question-answer pairs from deduplicated chunks using multiple LLM models.
Each model runs in a separate subprocess to ensure complete GPU memory cleanup.

Outputs:
  - outputs/qa/{model}_qa.jsonl - QA pairs for each model (includes chunk ID for tracking)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from utils import load_config, load_prompts, read_jsonl, setup_output_dirs

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

# ====================================================================
# Configuration
# ====================================================================

BASE_DIR = Path(__file__).resolve().parent
config = load_config()
prompts = load_prompts()

BATCH_SIZE = config["batch_size"]

# QA generation settings
LLM_MODELS = config["qa_generation"]["llm_models"]
MAX_MODEL_LEN = config["max_model_len"]
MAX_NEW_TOKENS = config["max_new_tokens"]
USE_SAMPLED = config["qa_generation"]["use_sampled_chunks"]

# Setup output directories
dirs = setup_output_dirs(config)

# Input file
if USE_SAMPLED:
    CHUNKS_FILE = dirs["chunks"] / "chunks_dedu_sampled.jsonl"
else:
    CHUNKS_FILE = dirs["chunks"] / "chunks_dedu.jsonl"


# ====================================================================
# QA Generation (in separate process for full GPU cleanup)
# ====================================================================


def generate_qa_subprocess(model_name: str):
    """
    Run QA generation in a separate subprocess to ensure full GPU memory cleanup.
    This allows the OS to reclaim all GPU memory when the subprocess exits.
    """
    print("\n" + "=" * 70)
    print(f"Generating QA with {model_name} in separate process")
    print("=" * 70)

    # Run generation script in subprocess
    generate_script = BASE_DIR / "3_sub_generate_qa.py"
    env = os.environ.copy()

    result = subprocess.run(
        [sys.executable, str(generate_script), model_name],
        env=env,
        capture_output=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"QA generation subprocess for {model_name} failed with code {result.returncode}"
        )

    # Wait for GPU memory to be fully released by OS
    print("Waiting 5 seconds for OS to reclaim GPU memory from subprocess...")
    time.sleep(5)


# ====================================================================
# Main
# ====================================================================


def main():
    print("=" * 70)
    print("STEP 3: QA GENERATION")
    print("=" * 70)
    print(f"Input: {CHUNKS_FILE}")
    print(f"LLM models: {LLM_MODELS}")
    print(f"Use sampled chunks: {USE_SAMPLED}")
    if os.getenv("TESTING_MODE", "0") == "1":
        print(f"⚠ TESTING MODE: Will generate only 10 QA pairs per model")
    print("=" * 70)

    # Load chunks to verify file exists
    print(f"\nLoading chunks from {CHUNKS_FILE}...")
    chunks = read_jsonl(CHUNKS_FILE)
    print(f"Loaded {len(chunks)} chunks")

    # Generate QA with each model (each in separate process for full GPU cleanup)
    for model_name in LLM_MODELS:
        generate_qa_subprocess(model_name)

    print(f"\n✓ Step 3 completed successfully!")


if __name__ == "__main__":
    main()
