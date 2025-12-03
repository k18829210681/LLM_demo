#!/usr/bin/env python3
"""
Helper script: Rerank top-K chunks with a single reranker model in a separate process.
This allows full GPU memory cleanup when the process exits.
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt

from utils import append_jsonl, load_config, load_prompts, read_jsonl, setup_output_dirs

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


def format_instruction(instruction, query, doc):
    """Format the reranker prompt."""
    text = [
        {
            "role": "system",
            "content": 'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".',
        },
        {
            "role": "user",
            "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}",
        },
    ]
    return text


def process_inputs(tokenizer, pairs, instruction, max_length, suffix_tokens):
    """Process input pairs into tokenized prompts."""
    messages = [format_instruction(instruction, query, doc) for query, doc in pairs]
    messages = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
    )
    messages = [ele[:max_length] + suffix_tokens for ele in messages]
    messages = [TokensPrompt(prompt_token_ids=ele) for ele in messages]
    return messages


def compute_logits(model, messages, sampling_params, true_token, false_token):
    """Compute reranking scores from logits."""
    outputs = model.generate(messages, sampling_params, use_tqdm=False)
    scores = []
    for i in range(len(outputs)):
        final_logits = outputs[i].outputs[0].logprobs[-1]

        if true_token not in final_logits:
            true_logit = -10
        else:
            true_logit = final_logits[true_token].logprob

        if false_token not in final_logits:
            false_logit = -10
        else:
            false_logit = final_logits[false_token].logprob

        true_score = math.exp(true_logit)
        false_score = math.exp(false_logit)
        score = true_score / (true_score + false_score)
        scores.append(score)
    return scores


def main():
    if len(sys.argv) != 4:
        print("Usage: step6_sub_rerank.py <emb_model> <rerank_model> <instruction>")
        sys.exit(1)

    emb_model = sys.argv[1]
    rerank_model = sys.argv[2]
    instruction = sys.argv[3]

    config = load_config()
    prompts = load_prompts()
    dirs = setup_output_dirs(config)

    BATCH_SIZE = config["batch_size"]
    MAX_MODEL_LEN = config["max_model_len"]
    INITIAL_TOP_K = config["reranking"]["initial_top_k"]

    # Instruction prompt mapping
    INSTRUCTION_PROMPTS = {
        "base": prompts["query_retrieve_prompt_base"],
        "ml": prompts["query_retrieve_prompt_ML"],
        "ml_hf": prompts["query_retrieve_prompt_ML_HF"],
    }

    task_instruction = INSTRUCTION_PROMPTS[instruction]

    print(f"Reranking {emb_model}/{instruction} with {rerank_model}...")

    # Load embeddings
    chunk_embs_file = dirs["embeddings"] / f"{emb_model}_chunks.npy"
    query_embs_file = dirs["embeddings"] / f"{emb_model}_{instruction}_query.npy"
    chunk_ids_file = dirs["embeddings"] / "chunks_ids.json"
    query_ids_file = dirs["embeddings"] / "query_ids.json"

    print(f"Loading embeddings...")
    chunk_embs = np.load(chunk_embs_file)
    query_embs = np.load(query_embs_file)

    with open(chunk_ids_file) as f:
        chunk_ids = json.load(f)

    with open(query_ids_file) as f:
        query_ids = json.load(f)

    print(f"Loaded {len(chunk_ids)} chunks and {len(query_ids)} queries")

    # Load chunks and queries
    chunks_file = dirs["chunks"] / "chunks_dedu.jsonl"
    qa_file = dirs["qa"] / "qa_selected.jsonl"

    print(f"Loading chunks from {chunks_file}...")
    chunks = read_jsonl(chunks_file)
    chunks_dict = {chunk["id"]: chunk["text"] for chunk in chunks}

    print(f"Loading queries from {qa_file}...")
    qa_list = read_jsonl(qa_file)
    queries_dict = {qa["id"]: qa["question"] for qa in qa_list}

    # Compute initial similarities and get top-K
    print(f"\nComputing top-{INITIAL_TOP_K} chunks for each query...")
    similarities = np.dot(query_embs, chunk_embs.T)

    # Get top-K indices for each query
    top_k_indices = np.argsort(-similarities, axis=1)[:, :INITIAL_TOP_K]

    # Load reranker model
    print(f"\nLoading reranker model {rerank_model}...")
    rerank_model_id = config["reranking_models"][rerank_model]

    llm = LLM(
        model=rerank_model_id,
        max_model_len=MAX_MODEL_LEN,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.9,
    )
    tokenizer = llm.get_tokenizer()
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare suffix and tokens
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    true_token = tokenizer("yes", add_special_tokens=False).input_ids[0]
    false_token = tokenizer("no", add_special_tokens=False).input_ids[0]

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        logprobs=20,
        allowed_token_ids=[true_token, false_token],
    )

    # Rerank for each query
    output_file = dirs["reranking"] / f"{emb_model}_{rerank_model}_{instruction}_reranking.jsonl"

    # Clear output file if it exists
    if output_file.exists():
        output_file.unlink()

    print(f"\nReranking queries...")
    for query_idx in tqdm(range(len(query_ids)), desc="Reranking"):
        query_id = query_ids[query_idx]
        query_text = queries_dict[query_id]

        # Get top-K chunk IDs and texts
        top_k_chunk_indices = top_k_indices[query_idx]
        top_k_chunk_ids = [chunk_ids[idx] for idx in top_k_chunk_indices]
        top_k_chunk_texts = [chunks_dict[cid] for cid in top_k_chunk_ids]

        # Prepare pairs for reranking
        pairs = [(query_text, chunk_text) for chunk_text in top_k_chunk_texts]

        # Process in batches
        all_scores = []
        for i in range(0, len(pairs), BATCH_SIZE):
            batch_pairs = pairs[i : i + BATCH_SIZE]
            messages = process_inputs(
                tokenizer, batch_pairs, task_instruction, MAX_MODEL_LEN - len(suffix_tokens), suffix_tokens
            )
            scores = compute_logits(llm, messages, sampling_params, true_token, false_token)
            all_scores.extend(scores)

        # Save results
        result = {
            "query_id": query_id,
            "chunk_ids": top_k_chunk_ids,
            "scores": all_scores,
        }
        append_jsonl(result, output_file)

    print(f"✓ Saved reranking results to {output_file}")
    print("Process will exit and free all GPU memory...")


if __name__ == "__main__":
    main()
