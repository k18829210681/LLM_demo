#!/usr/bin/env python3
"""
Helper script: Judge QA quality with a single judge model in a separate process.
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
    to_int,
)

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


def main():
    if len(sys.argv) != 3:
        print("Usage: step4_sub_judge_qa.py <judge_model> <qa_model>")
        sys.exit(1)

    judge_model = sys.argv[1]
    qa_model = sys.argv[2]

    config = load_config()
    prompts = load_prompts()
    dirs = setup_output_dirs(config)

    BATCH_SIZE = config["batch_size"]
    MAX_MODEL_LEN = config["max_model_len"]
    MAX_NEW_TOKENS = config["max_new_tokens"]

    # Judgment dimensions
    JUDGMENTS = {
        "groundedness": "question_groundedness_critique_prompt",
        "relevance": "question_relevance_critique_prompt",
        "standalone": "question_standalone_critique_prompt",
    }

    # Load QA pairs
    qa_file = dirs["qa"] / f"{qa_model}_qa.jsonl"
    print(f"Loading QA pairs from {qa_file}...")
    qa_list = read_jsonl(qa_file)
    print(f"Loaded {len(qa_list)} QA pairs")

    if not qa_list:
        print("No QA pairs to judge")
        return

    # Load chunk context
    chunks_file = dirs["chunks"] / "chunks_dedu.jsonl"
    print(f"Loading chunk context from {chunks_file}...")
    chunks = read_jsonl(chunks_file)
    context_map = {chunk["id"]: chunk["text"] for chunk in chunks}
    print(f"Loaded context for {len(context_map)} chunks")

    # Load model
    print(f"\nJudging {qa_model}_qa.jsonl with {judge_model}...")
    model_id = config["llm_models"][judge_model]
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

    system_msg = prompts["model_identity_critique"]
    output_file = dirs["judgments"] / f"{judge_model}_judge_{qa_model}_qa.jsonl"

    # Clear output file if it exists
    if output_file.exists():
        output_file.unlink()

    # Judge each dimension separately
    for dimension, prompt_key in JUDGMENTS.items():
        print(f"  Judging {dimension}...")

        # Prepare prompts for this dimension
        prompts_list = []
        for qa in qa_list:
            question = qa["question"]
            answer = qa["answer"]
            chunk_id = qa["id"]
            context = context_map.get(chunk_id, "")

            user_msg = prompts[prompt_key].format(
                question=question , context=context
            )
            chat_text = build_chat_prompt(tokenizer, model_id, system_msg, user_msg)
            prompts_list.append(chat_text)

        # Process in batches
        ratings = []
        for i in tqdm(
            range(0, len(prompts_list), BATCH_SIZE),
            desc=f"  {dimension}",
            leave=False,
        ):
            batch_prompts = prompts_list[i : i + BATCH_SIZE]
            outputs = llm.generate(batch_prompts, sampling_params)

            for output in outputs:
                text = output.outputs[0].text
                parsed = parse_llm_output(
                    text, judge_model, ["Evaluation", "Total rating"]
                )
                rating = to_int(parsed.get("Total rating", "0")) or 0
                ratings.append((parsed.get("Evaluation", ""), rating))

        # Save judgments for this dimension
        for qa, (evaluation, rating) in zip(qa_list, ratings):
            judgment = {
                "id": qa["id"],
                "question": qa["question"],
                "answer": qa["answer"],
                "qa_model": qa_model,
                "judge_model": judge_model,
                "dimension": dimension,
                "evaluation": evaluation,
                "rating": rating,
            }
            append_jsonl(judgment, output_file)

    print(f"✓ Saved judgments to {output_file}")
    print("Process will exit and free all GPU memory...")


if __name__ == "__main__":
    main()
