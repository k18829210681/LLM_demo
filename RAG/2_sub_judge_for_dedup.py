#!/usr/bin/env python3
"""
Helper script: Judge duplicate pairs with a single LLM in a separate process.
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
    to_int,
)

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


def main():
    if len(sys.argv) != 2:
        print("Usage: step2_judge_for_dedup.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]

    config = load_config()
    prompts = load_prompts()

    BATCH_SIZE = config["batch_size"]
    MAX_MODEL_LEN = config["max_model_len"]
    MAX_NEW_TOKENS = config["max_new_tokens"]

    from utils import setup_output_dirs

    dirs = setup_output_dirs(config)

    # Load pairs to judge from temp file
    import json

    pairs_file = dirs["deduplication"] / "temp_pairs_to_judge.json"
    chunks_file = dirs["chunks"] / "chunks.jsonl"
    output_file = dirs["deduplication"] / f"{model_name}_dedu.jsonl"

    print(f"Loading pairs from {pairs_file}...")
    with open(pairs_file) as f:
        pairs_data = json.load(f)
        # Convert IDs back to int (JSON converts them to strings)
        pairs_to_judge = [(int(p[0]), int(p[1]), p[2]) for p in pairs_data]

    print(f"Loading chunks from {chunks_file}...")
    chunks = read_jsonl(chunks_file)

    # Testing mode
    if len(chunks) > 2000 and os.getenv("TESTING_MODE", "0") == "1":
        chunks = chunks[:2000]

    chunks_dict = {chunk["id"]: chunk["text"] for chunk in chunks}

    if not pairs_to_judge:
        print("No pairs to judge")
        # Create empty output file
        with open(output_file, "w") as f:
            pass
        return

    print(f"Judging {len(pairs_to_judge)} pairs with {model_name}...")

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

    system_msg = prompts["model_identity_duplication"]
    prompt_template = prompts["document_duplication_critique_prompt"]

    # Prepare prompts
    prompts_list = []
    for id_i, id_j, _ in pairs_to_judge:
        text_a = chunks_dict[id_i]
        text_b = chunks_dict[id_j]
        user_msg = prompt_template.format(doc_a=text_a, doc_b=text_b)
        chat_text = build_chat_prompt(tokenizer, model_id, system_msg, user_msg)
        prompts_list.append(chat_text)

    # Clear output file if it exists
    if output_file.exists():
        output_file.unlink()

    # Process in batches
    for i in tqdm(
        range(0, len(prompts_list), BATCH_SIZE), desc=f"Judging with {model_name}"
    ):
        batch_prompts = prompts_list[i : i + BATCH_SIZE]
        batch_pairs = pairs_to_judge[i : i + BATCH_SIZE]

        outputs = llm.generate(batch_prompts, sampling_params)

        for pair, output in zip(batch_pairs, outputs):
            id_i, id_j, similarity = pair
            text = output.outputs[0].text
            parsed = parse_llm_output(
                text, model_name, ["Evaluation", "Duplication rating", "Decision"]
            )

            evaluation = parsed.get("Evaluation", "")
            rating_str = parsed.get("Duplication rating", "")
            decision = parsed.get("Decision", "").lower()

            rating = to_int(rating_str)

            result = {
                "id_i": id_i,
                "id_j": id_j,
                "similarity": similarity,
                "evaluation": evaluation,
                "rating": rating,
                "decision": decision,
            }

            append_jsonl(result, output_file)

    print(f"✓ Saved judgments to {output_file}")
    print("Process will exit and free all GPU memory...")


if __name__ == "__main__":
    main()
