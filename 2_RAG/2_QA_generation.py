from tqdm.auto import tqdm
from typing import List, Dict, Any
import json, os, yaml, random
from pathlib import Path
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from transformers import AutoTokenizer

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

# --------------------
# Config
# --------------------
MODEL_ID_OPTIONS = {
    "qwen": "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
    "gpt":  "openai/gpt-oss-20b",
}
MODEL_CHOICE = "gpt"  # "qwen" or "gpt"

PROMPTS_PATH = "/mnt/g/G/RAG/prompts.yaml"
CHUNKS_PATH = "/mnt/g/G/RAG/outputs/chunks_sampled.jsonl"

OUT_DIR  = Path("/mnt/g/G/RAG/outputs")
OUT_FILE = OUT_DIR / f"QA_{MODEL_CHOICE}.ndjson"

# Batch sizes tuned conservatively
BATCH_SIZE = 4 if MODEL_CHOICE == "qwen" else 16

MAX_MODEL_LEN = 30000
MAX_NEW_TOKENS = 20000  # will be clipped per input to respect MAX_MODEL_LEN

# --------------------
# Helpers
# --------------------
def safe_max_new_tokens(input_len: int, max_ctx: int, requested: int) -> int:
    """Ensure input_len + max_new <= max_ctx."""
    room = max_ctx - input_len
    return max(0, min(requested, room))

def load_chunks(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def build_chat_text(tokenizer, model_id: str, prompts: Dict[str, Any], context: str) -> str:
    if "Qwen" in model_id:
        messages = [
            {"role": "system", "content": prompts["model_identity_QA_generation"]},
            {"role": "user",   "content": prompts["QA_generation_prompt"].format(context=context)},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:  # gpt-oss-20b
        messages = [
            {"role": "user", "content": prompts["QA_generation_prompt"].format(context=context)}
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort="medium",
            model_identity=prompts["model_identity_QA_generation"],
        )

def parsing_output_gpt(text: str):
    """Extract final factoid QA from GPT-style output."""
    q, a = "", ""
    if text:
        if "final<|message|>" in text:
            final = text.split("final<|message|>")[-1]
            if "Factoid question:::" in final and "Answer:::" in final:
                q = final.split("Answer:::")[0].split("Factoid question:::")[-1].strip()
                a = final.split("Answer:::")[-1].split("<|return|>")[0].strip()
    return q, a


def parsing_output_qwen(text: str):
    """Qwen sometimes outputs slightly different tokens—adapt as needed."""
    q, a = "", ""
    if text:
        if "</think>" in text:
            final = text.split("</think>")[-1]
            if "Factoid question:::" in final and "Answer:::" in final:
                q = final.split("Answer:::")[0].split("Factoid question:::")[-1].strip()
                a = final.split("Answer:::")[-1].split("<|im_end|>")[0].strip()

    return q, a

def chunked(items, n):
    for i in range(0, len(items), n):
        yield items[i:i+n]

# --------------------
# Main
# --------------------
def main():
    # IO
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prompts = yaml.safe_load(Path(PROMPTS_PATH).read_text(encoding="utf-8"))

    model_id = MODEL_ID_OPTIONS[MODEL_CHOICE]
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # vLLM engine
    llm = LLM(model=model_id, dtype="auto",max_model_len=MAX_MODEL_LEN)

    # Prepare prompts (+ keep meta alongside)
    prompt_packets = []  # list of dicts: {tp: TokensPrompt, source: str, context: str, input_len: int}

    for sp in tqdm(load_chunks(CHUNKS_PATH), desc="Build prompts"):
        source  = sp.get("source", "")
        context = sp.get("page_content", "")
        index = sp.get("index", "")

        # Build chat text
        chat_text = build_chat_text(tokenizer, model_id, prompts, context)

        # Tokenize (CPU ok; we only need ids)
        enc = tokenizer(chat_text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"][0].tolist()

        prompt_packets.append({
            "index": index,
            "tp": TokensPrompt(prompt_token_ids=input_ids),
            "source": source,
            "context": context,
            "input_len": len(input_ids),
        })

    # Resume support (append-only)
    existing = 0
    if OUT_FILE.exists():
        with OUT_FILE.open("r", encoding="utf-8") as f:
            for _ in f:
                existing += 1

    # Main generate loop
    with OUT_FILE.open("a", encoding="utf-8") as f_out:
        work = list(enumerate(prompt_packets))[existing:]
        for batch in tqdm(list(chunked(work, BATCH_SIZE)), desc="vLLM generate (batched)"):
            # batch_indices = [i for (i, _) in batch]
            batch_prompts = [pkt["tp"] for (_, pkt) in batch]

            # Per-request max_new (guard context window)
            per_req_max = [
                safe_max_new_tokens(pkt["input_len"], MAX_MODEL_LEN, MAX_NEW_TOKENS)
                for (_, pkt) in batch
            ]
            # Use the smallest across the batch to satisfy all requests
            eff_max_new = max(0, min(per_req_max)) if per_req_max else 0

            sampling = SamplingParams(
                max_tokens=eff_max_new,
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
                skip_special_tokens=False,
                spaces_between_special_tokens=False,
                stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
            )

            gen_results = llm.generate(prompts=batch_prompts, sampling_params=sampling)

            # Write results
            for local_i, req_out in enumerate(gen_results):
                pkt = batch[local_i][1]
                if not req_out.outputs:
                    generated_text = ""
                else:
                    gen_ids = req_out.outputs[0].token_ids
                    generated_text = tokenizer.decode(gen_ids, skip_special_tokens=False)

                if "gpt" in model_id.lower():
                    q, a = parsing_output_gpt(generated_text)
                else:
                    q, a = parsing_output_qwen(generated_text)

                # Only save if both present
                record = {
                    "index": pkt['index'],
                    # "source": pkt["source"],
                    # "context": pkt["context"],
                    "question": q,
                    "answer": a,
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()
                os.fsync(f_out.fileno())
    try:
        llm.shutdown()
    except Exception:
        pass

if __name__ == "__main__":
    main()
