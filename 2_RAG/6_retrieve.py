# Requires: vllm>=0.8.5, torch
import os, json, yaml
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import torch
from vllm import LLM

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

MODEL_ID_OPTIONS = {
    "06B": "Qwen/Qwen3-Embedding-0.6B",
    "4B":  "Qwen/Qwen3-Embedding-4B",
    "8B":  "Qwen/Qwen3-Embedding-8B",
}

MODEL_CHOICE = "06B"

QA_PATH      = Path("/mnt/g/G/RAG/outputs/QA_gpt.ndjson")
OUT_ROOT     = Path("/mnt/g/G/RAG/outputs/")
PROMPTS_PATH = "/mnt/g/G/RAG/prompts.yaml"


def load_ndjson(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_detailed_instruct(task_description: str, query: str) -> str:
    """
    Match Qwen's recommended format:
    Instruct: <one-sentence task description>
    Query:<user query>
    """
    return f"Instruct: {task_description}\nQuery:{query}"


def main():
    # --------------------
    # Load prompts & data
    # --------------------
    prompts_yaml = yaml.safe_load(Path(PROMPTS_PATH).read_text(encoding="utf-8"))
    # Give each prompt a short name for filenames
    prompt_variants = [
        ("base",   prompts_yaml["query_retrieve_prompt_base"]),
        ("ml",     prompts_yaml["query_retrieve_prompt_ML"]),
        ("ml_hf",  prompts_yaml["query_retrieve_prompt_ML_HF"]),
    ]

    rows = load_ndjson(QA_PATH)

    doc_ids: List[int] = []
    questions: List[str] = []

    for r in rows:
        rid  = int(r["index"])
        text = (r.get("question") or "").strip()
        if not text or text == "None":
            continue
        doc_ids.append(rid)
        questions.append(text)

    if not questions:
        raise RuntimeError("No non-empty questions found in QA file.")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # --------------------
    # Save doc_ids + question mapping (this is your id-query pair)
    # --------------------
    meta_path = OUT_ROOT / f"{MODEL_CHOICE}_query_meta.ndjson"
    with meta_path.open("w", encoding="utf-8") as f:
        for rid, q in zip(doc_ids, questions):
            rec = {"doc_id": rid, "question": q}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved id-question mapping → {meta_path}")

    # If you still want pure doc_ids as a list:
    (OUT_ROOT / f"{MODEL_CHOICE}_query_doc_ids.json").write_text(
        json.dumps(doc_ids, ensure_ascii=False),
        encoding="utf-8",
    )

    # --------------------
    # Init model
    # --------------------
    model_id = MODEL_ID_OPTIONS[MODEL_CHOICE]
    model = LLM(model=model_id, task="embed", max_model_len=4096)

    # --------------------
    # Embed queries for each prompt variant
    # --------------------
    for key, pro in prompt_variants:
        print(f"\n[{MODEL_CHOICE}] Embedding queries with prompt variant: {key}")

        # 1) Build instruction-augmented inputs for this task
        input_texts = [get_detailed_instruct(pro, q) for q in questions]

        # 2) Single call (no batching, as requested)
        outputs = model.embed(input_texts)

        # 3) Convert to tensor
        emb = torch.tensor(
            [o.outputs.embedding for o in outputs],
            dtype=torch.float32
        ).cpu()

        print(f"{key}: embeddings shape = {tuple(emb.shape)}")

        # 4) Save embeddings
        out_path = OUT_ROOT / f"{MODEL_CHOICE}_query_embeddings_{key}.npy"
        np.save(out_path, emb.numpy())
        print(f"Saved embeddings for {key} → {out_path}")

    print("\nDone. All query embeddings saved under:", OUT_ROOT.resolve())


if __name__ == "__main__":
    main()
