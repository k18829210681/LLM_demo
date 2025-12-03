#!/usr/bin/env python3
"""
Helper script: Embed queries with a single embedding model in a separate process.
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

from utils import load_config, load_prompts, read_jsonl, setup_output_dirs

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


def main():
    if len(sys.argv) != 2:
        print("Usage: step5_sub_embed_queries.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]

    config = load_config()
    prompts = load_prompts()
    dirs = setup_output_dirs(config)

    BATCH_SIZE = config["batch_size"]
    MAX_MODEL_LEN = config["max_model_len"]
    QUERY_INSTRUCTIONS = config["embedding"]["query_instructions"]
    QA_FILE = dirs["qa"] / "qa_selected.jsonl"

    # Instruction prompt mapping
    INSTRUCTION_PROMPTS = {
        "base": prompts["query_retrieve_prompt_base"],
        "ml": prompts["query_retrieve_prompt_ML"],
        "ml_hf": prompts["query_retrieve_prompt_ML_HF"],
    }

    print(f"Embedding queries with {model_name}...")

    # Load QA
    print(f"Loading QA pairs from {QA_FILE}...")
    qa_list = read_jsonl(QA_FILE)
    print(f"Loaded {len(qa_list)} QA pairs")

    query_ids = [qa["id"] for qa in qa_list]
    questions = [qa["question"] for qa in qa_list]

    # Load model
    model_id = config["embedding_models"][model_name]
    llm = LLM(
        model=model_id,
        task="embed",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.9,
    )

    # Embed with each instruction
    for instruction in QUERY_INSTRUCTIONS:
        print(f"  Instruction: {instruction}")

        task_desc = INSTRUCTION_PROMPTS[instruction]

        # Format queries with instruction
        formatted_queries = [f"Instruct: {task_desc}\nQuery: {q}" for q in questions]

        # Embed in batches
        all_embeddings = []
        for i in tqdm(
            range(0, len(formatted_queries), BATCH_SIZE),
            desc=f"  Embedding queries ({model_name}/{instruction})",
        ):
            batch_texts = formatted_queries[i : i + BATCH_SIZE]
            outputs = llm.embed(batch_texts)
            embeddings = [o.outputs.embedding for o in outputs]
            all_embeddings.extend(embeddings)

        embeddings = np.array(all_embeddings, dtype=np.float32)

        # L2 normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        print(f"    Embeddings shape: {embeddings.shape}")

        # Save embeddings
        emb_file = dirs["embeddings"] / f"{model_name}_{instruction}_query.npy"
        print(f"    Saving embeddings to {emb_file}...")
        np.save(emb_file, embeddings)

        print(f"    ✓ Saved query embeddings for {model_name}/{instruction}")

    # Save IDs only once (same for all models and instructions)
    ids_file = dirs["embeddings"] / "query_ids.json"
    if not ids_file.exists():
        print(f"Saving query IDs to {ids_file}...")
        with open(ids_file, "w") as f:
            json.dump(query_ids, f)
    else:
        print(f"Query IDs file already exists: {ids_file}")

    print(f"✓ Completed all query embeddings for {model_name}")
    print("Process will exit and free all GPU memory...")


if __name__ == "__main__":
    main()
