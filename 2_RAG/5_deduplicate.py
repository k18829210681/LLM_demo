# Requires: vllm>=0.8.5, faiss (cpu or gpu), torch, transformers, pyyaml, tqdm
import os
import json
import yaml
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

import faiss
import numpy as np
import torch
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from vllm.inputs import TokensPrompt

# --------------------------------
# Config
# --------------------------------
EMB_NPY_PATH   = Path("/mnt/g/G/RAG/outputs/faiss_store/06B_full_embeddings.npy")
ID_JSON_PATH   = Path("/mnt/g/G/RAG/outputs/faiss_store/06B_doc_ids.json")
FAISS_INDEX    = Path("/mnt/g/G/RAG/outputs/faiss_store/06B_d1024.faiss")
CHUNKS_PATH    = Path("/mnt/g/G/RAG/outputs/chunks.jsonl")


TOP_K          = 20
SIM_MIN_THRESHOLD = 0.97  # cosine similarity threshold
SIM_MAX_THRESHOLD = 0.995

PROMPTS_PATH   = Path("/mnt/g/G/RAG/prompts.yaml")
PROMPT_SECTION = "duplication_judge"  # YAML section name for this task

MODEL_ID_OPTIONS = {
    "qwen": "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
    "gpt":  "openai/gpt-oss-20b",
}
MODEL_CHOICE = "gpt"  # "qwen" or "gpt"

MAX_MODEL_LEN  = 20_000
MAX_NEW_TOKENS = 10_000

OUT_DEDUP_PATH = Path(f"/mnt/g/G/RAG/outputs/dedup_{MODEL_CHOICE}.ndjson")

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


# --------------------------------
# Helpers
# --------------------------------
def safe_max_new_tokens(input_len: int, max_ctx: int, requested: int) -> int:
    room = max_ctx - input_len
    return max(0, min(requested, room))

def load_chunks(path: Path) -> Dict[int, Dict[str, Any]]:
    id2chunk: Dict[int, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            idx = rec["index"]
            id2chunk[idx] = rec
    return id2chunk


def build_chat_text(tokenizer, model_id: str, system_msg: str, user_msg: str) -> str:
    if "Qwen" in model_id:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:  # gpt-oss-20b
        messages = [{"role": "user", "content": user_msg}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort="medium",
            model_identity=system_msg,
        )

def parse_output(text: str, model_choice) -> Tuple[str, str]:
    think_split = "final<|message|>" if model_choice == 'gpt' else "</think>"
    end_split = "<|return|>" if model_choice == 'gpt' else "<|im_end|>"

    evaluation, rating, decision= "", "", ""

    if not text:
        return evaluation, rating
    if think_split in text:
        final = text.split(think_split)[-1]
        if "Evaluation:::" in final and "Duplication rating:::" in final and "Decision:::" in final:
            evaluation = final.split("Evaluation:::")[-1].split("Duplication rating:::")[0].strip()
            rating = final.split("Duplication rating:::")[-1].split("Decision:::")[0].strip()
            decision = final.split("Decision:::")[-1].split(end_split)[0].strip()
    return evaluation, rating, decision


# --------------------------------
# Main
# --------------------------------
def main():
    prompts = yaml.safe_load(Path(PROMPTS_PATH).read_text(encoding="utf-8"))
    # ---- Load embeddings, ids, FAISS ----
    emb_all = np.load(EMB_NPY_PATH).astype(np.float32)
    with ID_JSON_PATH.open("r", encoding="utf-8") as f:
        id_all = json.load(f)

    faiss_index = faiss.read_index(str(FAISS_INDEX))
    assert len(id_all) == emb_all.shape[0], "id_all and emb_all must align"

    # Normalize embeddings for cosine similarity (dot product)
    norms = np.linalg.norm(emb_all, ord=2, axis=1, keepdims=True)
    emb_all_norm = emb_all / norms

    id2chunk = load_chunks(CHUNKS_PATH)

    model_id = MODEL_ID_OPTIONS[MODEL_CHOICE]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm = LLM(model=model_id,  dtype="auto", max_model_len=MAX_MODEL_LEN)

    # Base sampling (we’ll clamp max_tokens per prompt)
    base_sampling = dict(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
        seed=42,
    )

    OUT_DEDUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_f = OUT_DEDUP_PATH.open("w", encoding="utf-8")

    # ---- Loop over anchors (A) ----
    for anchor_idx in tqdm(range(emb_all_norm.shape[0]), desc="Judging potential duplicates"):
        anchor_emb = emb_all_norm[anchor_idx : anchor_idx + 1]  # shape (1, dim)
        anchor_id = id_all[anchor_idx]

        # Retrieve FAISS neighbors
        scores, ids = faiss_index.search(anchor_emb, TOP_K)  # scores (1, k), ids (1, k)
        scores = scores[0]
        ids = ids[0]

        # Filter by similarity threshold and exclude self (assuming id is chunk index)
        for sim, cand_id in zip(scores, ids):
            if cand_id == anchor_id:
                continue
            if sim < SIM_MIN_THRESHOLD:
                continue

            # Map FAISS ids (which should match doc_ids) back to chunk records
            anchor_chunk = id2chunk.get(int(anchor_id))
            cand_chunk   = id2chunk.get(int(cand_id))

            if anchor_chunk is None or cand_chunk is None:
                # If mapping is off, skip this pair
                continue
            if sim > SIM_MAX_THRESHOLD:
                record = {
                "anchor_id": int(anchor_id),
                "candidate_id": int(cand_id),
                "anchor_index": int(anchor_chunk["index"]),
                "candidate_index": int(cand_chunk["index"]),
                "embedding_score": float(sim),
                "rating": 5,
                "decision": "yes",
                "evaluation": "sim score too high",
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()  # so you keep progress even if interrupted
                continue

            chunk_a_text = anchor_chunk["page_content"]
            chunk_b_text = cand_chunk["page_content"]

            # Build prompt for this pair (A, B)
            user_msg = prompts['document_duplication_critique_prompt'].format(doc_a=chunk_a_text, doc_b=chunk_b_text)
            chat_text = build_chat_text(
                tokenizer,
                model_id,
                system_msg=prompts["model_identity_duplication"],
                user_msg=user_msg,
            )
            enc = tokenizer(chat_text, return_tensors="pt", add_special_tokens=False)
            input_ids = enc["input_ids"][0].tolist()
            max_new = safe_max_new_tokens(len(input_ids), MAX_MODEL_LEN, MAX_NEW_TOKENS)
            sampling = SamplingParams(max_tokens=max_new, **base_sampling)
            # Call vLLM
            req = TokensPrompt(prompt_token_ids=input_ids)
            out = llm.generate(prompts=[req], sampling_params=sampling)[0]
            text = ""
            if out.outputs:
                gen_ids = out.outputs[0].token_ids
                text = tokenizer.decode(gen_ids, skip_special_tokens=False)

            evaluation, rating, decision = parse_output(text, model_choice=MODEL_CHOICE)

            # Log result as NDJSON
            record = {
                "anchor_id": int(anchor_id),
                "candidate_id": int(cand_id),
                "anchor_index": int(anchor_chunk["index"]),
                "candidate_index": int(cand_chunk["index"]),
                "embedding_score": float(sim),
                "rating": rating,
                "evaluation": evaluation,
                "decision": decision,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()  # so you keep progress even if interrupted

    out_f.close()
    print(f"Saved LLM dedup judgements to {OUT_DEDUP_PATH}")


if __name__ == "__main__":
    main()
