from tqdm.auto import tqdm
from typing import Dict, Any, List, Tuple
from pathlib import Path
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from transformers import AutoTokenizer
import json, os, yaml, shutil, random

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


MODEL_ID_OPTIONS = {
    "qwen": "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
    "gpt":  "openai/gpt-oss-20b",
}
MODEL_CHOICE   = "gpt"  # "qwen" or "gpt"
PROMPTS_PATH   = "/mnt/g/G/RAG/prompts.yaml"
QA_PATH        = Path("/mnt/g/G/RAG/outputs/QA_qwen.ndjson")
CHUNKS_PATH    = Path("/mnt/g/G/RAG/outputs/chunks_sampled.jsonl")

# Conservative context / generation caps for judging
MAX_MODEL_LEN  = 40_000
MAX_NEW_TOKENS = 4_000

# Which YAML keys correspond to each judgment dimension
JUDGMENTS = {
    "groundedness": "question_groundedness_critique_prompt",
    "relevance":    "question_relevance_critique_prompt",
    "standalone":   "question_standalone_critique_prompt",
}

# --------------------
# Helpers
# --------------------
def safe_max_new_tokens(input_len: int, max_ctx: int, requested: int) -> int:
    room = max_ctx - input_len
    return max(0, min(requested, room))

def load_ndjson(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

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

    evaluation, rating = "", ""

    if not text:
        return evaluation, rating
    if think_split in text:
        final = text.split(think_split)[-1]
        if "Evaluation:::" in final and "Total rating:::" in final:
            evaluation = final.split("Total rating:::")[0].split("Evaluation:::")[-1].strip()
            rating = final.split("Total rating:::")[-1].split(end_split)[0].strip()
    return evaluation, rating


# --------------------
# Main
# --------------------
def main():
    # IO
    prompts = yaml.safe_load(Path(PROMPTS_PATH).read_text(encoding="utf-8"))
    rows     = load_ndjson(QA_PATH)
    contexts = load_ndjson(CHUNKS_PATH)
    contexts = [c['page_content'] for c in contexts]

    assert len(contexts) == len(rows), "len of contexts do not match with QA"

    # Model + tokenizer + engine
    model_id  = MODEL_ID_OPTIONS[MODEL_CHOICE]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm = LLM(model=model_id, dtype="auto", max_model_len=MAX_MODEL_LEN)

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

    # Prepare temp output (atomic rewrite)
    tmp_path = QA_PATH.with_suffix(".ndjson.tmp")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)

    with tmp_path.open("w", encoding="utf-8") as fout:
        for i, rec in enumerate(tqdm(rows, desc="Judging QA")):
            q = (rec.get("question") or "").strip()
            a = (rec.get("answer") or "").strip()
            c = contexts[i]

            # Skip empty QA (leave row unchanged)
            if not q or not a:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            # Evaluate each dimension exactly once (idempotent re-runs ok)
            for dim, yaml_key in JUDGMENTS.items():
                rating_key = f"rating_{MODEL_CHOICE}_{dim}"
                eval_key   = f"evaluation_{MODEL_CHOICE}_{dim}"
                if rec.get(rating_key) is not None and rec.get(eval_key) is not None:
                    continue

                # Build prompt
                user_msg = prompts[yaml_key].format(question=q, answer=a, context=c)
                chat_text = build_chat_text(
                    tokenizer,
                    model_id,
                    system_msg=prompts["model_identity_critique"],
                    user_msg=user_msg,
                )
                # read chat text to make sure it is good.
                if i == 0:
                    print(chat_text)
                # Tokenize to ids (CPU ok)
                enc = tokenizer(chat_text, return_tensors="pt", add_special_tokens=False)
                input_ids = enc["input_ids"][0].tolist()

                # Guard max_new by context window
                max_new = safe_max_new_tokens(len(input_ids), MAX_MODEL_LEN, MAX_NEW_TOKENS)

                sampling = SamplingParams(max_tokens=max_new, **base_sampling)
                req = TokensPrompt(prompt_token_ids=input_ids)
                out = llm.generate(prompts=[req], sampling_params=sampling)[0]

                text = ""
                if out.outputs:
                    gen_ids = out.outputs[0].token_ids
                    text = tokenizer.decode(gen_ids, skip_special_tokens=False)

                evaluation, rating = parse_output(text, model_choice=MODEL_CHOICE)

                # Persist on the same record
                rec[rating_key] = rating
                rec[eval_key]   = evaluation

            # Write updated record
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            os.fsync(fout.fileno())

    # Atomic replace original with backup
    backup_path = QA_PATH.with_suffix(".ndjson.bak")
    try:
        if backup_path.exists():
            backup_path.unlink()
        shutil.move(QA_PATH, backup_path)
    except FileNotFoundError:
        pass
    shutil.move(tmp_path, QA_PATH)

    # Clean shutdown (quiet NCCL/etc.)
    try:
        llm.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
