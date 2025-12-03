#!/usr/bin/env python3
"""
Step 4: QA Quality Judging

1. Load all QA files from step 3
2. Judge each QA file with each judge model (3 dimensions each)
3. Select best QA pairs that pass all judges' thresholds

Each judge model runs in a separate subprocess to ensure complete GPU memory cleanup.

Outputs:
  - outputs/judgments/{judge_model}_judge_{qa_model}_qa.jsonl - Judgments for each combination
  - outputs/qa/qa_selected.jsonl - Final selected QA pairs (best per chunk)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

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

BATCH_SIZE = config["batch_size"]

# QA judging settings
QA_MODELS = config["qa_generation"]["llm_models"]
JUDGE_MODELS = config["qa_judging"]["llm_models"]
MAX_MODEL_LEN = config["max_model_len"]
MAX_NEW_TOKENS = config["max_new_tokens"]

MIN_GROUNDEDNESS = config["qa_judging"]["min_groundedness"]
MIN_RELEVANCE = config["qa_judging"]["min_relevance"]
MIN_STANDALONE = config["qa_judging"]["min_standalone"]

# Setup output directories
dirs = setup_output_dirs(config)


# ====================================================================
# Judge QA with LLM (in separate process)
# ====================================================================


def judge_qa_subprocess(judge_model: str, qa_model: str):
    """
    Run QA judging in a separate subprocess to ensure full GPU memory cleanup.
    This allows the OS to reclaim all GPU memory when the subprocess exits.
    """
    print("\n" + "=" * 70)
    print(f"Judging {qa_model}_qa.jsonl with {judge_model} in separate process")
    print("=" * 70)

    # Run judge script in subprocess
    judge_script = BASE_DIR / "4_sub_judge_qa.py"
    env = os.environ.copy()

    result = subprocess.run(
        [sys.executable, str(judge_script), judge_model, qa_model],
        env=env,
        capture_output=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Judging subprocess for {judge_model} judge {qa_model} failed with code {result.returncode}"
        )

    # Wait for GPU memory to be fully released by OS
    print("Waiting 5 seconds for OS to reclaim GPU memory from subprocess...")
    time.sleep(5)


# ====================================================================
# Select Best QA
# ====================================================================


def select_best_qa(qa_models: List[str], judge_models: List[str]) -> List[Dict]:
    """
    Select best QA pairs based on judgments.

    Rules:
    1. QA must pass ALL judges (all 3 dimensions >= threshold)
    2. If multiple QA for same chunk: pick highest average rating
    3. If tie: pick from last model in llm_models list
    """
    print("\nSelecting best QA pairs...")

    # Load all judgments
    all_judgments = defaultdict(lambda: defaultdict(dict))

    for judge_model in judge_models:
        for qa_model in qa_models:
            judgment_file = dirs["judgments"] / f"{judge_model}_judge_{qa_model}_qa.jsonl"
            if not judgment_file.exists():
                continue

            judgments = read_jsonl(judgment_file)
            for j in judgments:
                chunk_id = j["id"]
                dimension = j["dimension"]
                # Store: all_judgments[chunk_id][(qa_model, judge_model)][dimension] = rating
                key = (qa_model, judge_model)
                if key not in all_judgments[chunk_id]:
                    all_judgments[chunk_id][key] = {
                        "question": j["question"],
                        "answer": j["answer"],
                        "ratings": {},
                    }
                all_judgments[chunk_id][key]["ratings"][dimension] = j["rating"]

    # Select best QA for each chunk
    selected = []

    for chunk_id, qa_options in all_judgments.items():
        valid_qa = []

        for (qa_model, judge_model), qa_data in qa_options.items():
            ratings = qa_data["ratings"]

            # Check if this QA passes this judge
            if (
                ratings.get("groundedness", 0) >= MIN_GROUNDEDNESS
                and ratings.get("relevance", 0) >= MIN_RELEVANCE
                and ratings.get("standalone", 0) >= MIN_STANDALONE
            ):
                # This QA passes this judge
                qa_data["qa_model"] = qa_model
                qa_data["judge_model"] = judge_model

                # Check if we've already added this (qa_model, question, answer) combo
                found = False
                for existing in valid_qa:
                    if (
                        existing["qa_model"] == qa_model
                        and existing["question"] == qa_data["question"]
                        and existing["answer"] == qa_data["answer"]
                    ):
                        # Same QA, add this judge's ratings
                        existing["all_ratings"].append(ratings)
                        found = True
                        break

                if not found:
                    valid_qa.append(
                        {
                            "qa_model": qa_model,
                            "question": qa_data["question"],
                            "answer": qa_data["answer"],
                            "all_ratings": [ratings],
                        }
                    )

        # Filter: keep only QA that passed ALL judges
        fully_valid = []
        for qa in valid_qa:
            if len(qa["all_ratings"]) == len(judge_models):
                # Passed all judges
                # Compute average rating across all judges and dimensions
                all_scores = []
                for ratings in qa["all_ratings"]:
                    all_scores.extend(ratings.values())
                avg_rating = sum(all_scores) / len(all_scores)

                fully_valid.append(
                    {
                        "id": chunk_id,
                        "question": qa["question"],
                        "answer": qa["answer"],
                        "qa_model": qa["qa_model"],
                        "avg_rating": avg_rating,
                    }
                )

        if not fully_valid:
            continue

        # Pick best: highest avg rating, then last model in list
        fully_valid.sort(key=lambda x: (x["avg_rating"], qa_models.index(x["qa_model"])))
        best = fully_valid[-1]

        selected.append(
            {
                "id": best["id"],
                "question": best["question"],
                "answer": best["answer"],
            }
        )

    print(f"Selected {len(selected)} QA pairs from {len(all_judgments)} chunks")
    return selected


# ====================================================================
# Main
# ====================================================================


def main():
    print("=" * 70)
    print("STEP 4: QA QUALITY JUDGING")
    print("=" * 70)
    print(f"QA models: {QA_MODELS}")
    print(f"Judge models: {JUDGE_MODELS}")
    print(f"Thresholds: groundedness>={MIN_GROUNDEDNESS}, relevance>={MIN_RELEVANCE}, standalone>={MIN_STANDALONE}")
    print("=" * 70)

    # Judge all combinations (each in separate process for full GPU cleanup)
    for qa_model in QA_MODELS:
        qa_file = dirs["qa"] / f"{qa_model}_qa.jsonl"
        if not qa_file.exists():
            print(f"\nWarning: {qa_file} not found, skipping...")
            continue

        qa_list = read_jsonl(qa_file)
        print(f"\nLoaded {len(qa_list)} QA pairs from {qa_model}_qa.jsonl")

        for judge_model in JUDGE_MODELS:
            judge_qa_subprocess(judge_model, qa_model)

    # Select best QA
    selected_qa = select_best_qa(QA_MODELS, JUDGE_MODELS)

    # Save selected QA
    output_file = dirs["qa"] / "qa_selected.jsonl"
    write_jsonl_incremental(selected_qa, output_file, batch_size=BATCH_SIZE)
    print(f"\nSaved selected QA to {output_file}")

    print(f"\n✓ Step 4 completed successfully!")


if __name__ == "__main__":
    main()
