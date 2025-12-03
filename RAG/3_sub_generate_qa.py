#!/usr/bin/env python3
"""
Helper script: Generate QA pairs with a single LLM in a separate process.
This allows full GPU memory cleanup when the process exits.
"""

import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams

from utils import (
    append_jsonl,
    build_chat_prompt,
    load_config,
    load_prompts,
    parse_llm_output,
    read_jsonl,
    setup_output_dirs,
)

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


def main():
    if len(sys.argv) != 2:
        print("Usage: step3_generate_for_qa.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]

    config = load_config()
    prompts = load_prompts()

    BATCH_SIZE = config["batch_size"]
    MAX_MODEL_LEN = config["max_model_len"]
    MAX_NEW_TOKENS = config["max_new_tokens"]
    USE_SAMPLED = config["qa_generation"]["use_sampled_chunks"]

    dirs = setup_output_dirs(config)

    # Input file
    if USE_SAMPLED:
        chunks_file = dirs["chunks"] / "chunks_dedu_sampled.jsonl"
    else:
        chunks_file = dirs["chunks"] / "chunks_dedu.jsonl"

    output_file = dirs["qa"] / f"{model_name}_qa.jsonl"

    print(f"Loading chunks from {chunks_file}...")
    chunks = read_jsonl(chunks_file)

    # Testing mode: use only first 10 chunks if enabled
    if os.getenv("TESTING_MODE", "0") == "1":
        print(f"⚠ TESTING MODE: Using only first 100 chunks (out of {len(chunks)})")
        chunks = chunks[:100]

    print(f"Loaded {len(chunks)} chunks")

    if not chunks:
        print("No chunks to process")
        # Create empty output file
        with open(output_file, "w") as f:
            pass
        return

    print(f"Generating QA with {model_name}...")

    # Load model
    model_id = config["llm_models"][model_name]
    llm = LLM(
        model=model_id,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.9,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_NEW_TOKENS,
        top_p=1.0,
        top_k=-1,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
        seed=42,
    )

    system_msg = prompts["model_identity_QA_generation"]
    prompt_template = prompts["QA_generation_prompt"]

    # Prepare prompts
    prompts_list = []
    for chunk in chunks:
        context = chunk["text"]
        user_msg = prompt_template.format(context=context)
        chat_text = build_chat_prompt(tokenizer, model_id, system_msg, user_msg)
        prompts_list.append(chat_text)

    # Clear output file if it exists
    if output_file.exists():
        output_file.unlink()

    # Process in batches
    for i in tqdm(
        range(0, len(prompts_list), BATCH_SIZE), desc=f"Generating QA with {model_name}"
    ):
        batch_prompts = prompts_list[i : i + BATCH_SIZE]
        batch_chunks = chunks[i : i + BATCH_SIZE]

        outputs = llm.generate(batch_prompts, sampling_params)

        for chunk, output in zip(batch_chunks, outputs):
            text = output.outputs[0].text

            parsed = parse_llm_output(text, model_name, ["Factoid question", "Answer"])

            question = parsed.get("Factoid question", "").strip()
            answer = parsed.get("Answer", "").strip()

            # Save QA with chunk ID for ground truth tracking
            qa_record = {
                "id": chunk["id"],  # Link back to chunk
                "question": question,
                "answer": answer,
            }

            append_jsonl(qa_record, output_file)

    print(f"✓ Saved QA pairs to {output_file}")
    print("Process will exit and free all GPU memory...")


if __name__ == "__main__":
    main()
