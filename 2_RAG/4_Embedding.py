# Requires: vllm>=0.8.5, faiss-cpu (or faiss-gpu), torch
import os, json
from pathlib import Path
from typing import List, Dict, Any, Iterable
import numpy as np
import torch
import faiss
from vllm import LLM

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

MODEL_ID_OPTIONS = {
    "06B": "Qwen/Qwen3-Embedding-0.6B",
    "4B":  "Qwen/Qwen3-Embedding-4B",
    "8B":  "Qwen/Qwen3-Embedding-8B",
}

MODEL_CHOICE = "06B"
EMBEDDING_DIMS = [32, 64, 128, 256, 512, 768, 1024]

CHUNKS_PATH = Path("/mnt/g/G/RAG/outputs/chunks.jsonl")
OUT_ROOT    = Path("/mnt/g/G/RAG/outputs/faiss_store")
BATCH_SIZE  = 512

def load_ndjson(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def batched(xs: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(xs), n):
        yield xs[i:i+n]    

def build_and_save_indexes(
    full_emb: torch.Tensor,
    doc_ids: List[int],
    dims: List[int],
    out_root: Path,
):
    base_dim = full_emb.shape[1]
    assert len(doc_ids) == full_emb.shape[0]

    for dim in dims:
        if dim > base_dim:
            continue

        # 1) truncate/project
        emb_dim = full_emb[:, :dim].contiguous()

        # 2) normalize AFTER truncation (cosine via IP)
        emb_dim = torch.nn.functional.normalize(emb_dim, p=2, dim=1)

        # 3) build FAISS (exact IP)
        index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        ids_np = np.array(doc_ids, dtype="int64")
        index.add_with_ids(emb_dim.cpu().numpy(), ids_np)

        # 4) persist FAISS + metadata
        # out_dir = out_root / f"{MODEL_CHOICE}_d{dim}"
        # out_dir.mkdir(parents=True, exist_ok=True)

        # faiss.write_index(index, str(out_dir / "index.faiss"))

        # print(f"[{MODEL_CHOICE} | d={dim}] Saved ntotal={index.ntotal} → {out_dir}")

        out_root.mkdir(parents=True, exist_ok=True)
        faiss_path = out_root / f"{MODEL_CHOICE}_d{dim}.faiss"
        faiss.write_index(index, str(faiss_path))
        print(f"[{MODEL_CHOICE} | d={dim}] Saved ntotal={index.ntotal} → {faiss_path.name}")

def main():
    rows = load_ndjson(CHUNKS_PATH)

    doc_ids, texts = [], []
    for r in rows:
        rid  = int(r["index"])
        text = r['page_content'].strip()
        if not text:
            continue
        doc_ids.append(rid)
        texts.append(text)

    if not texts:
        raise RuntimeError("No non-empty documents found.")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    model = LLM(model=MODEL_ID_OPTIONS[MODEL_CHOICE], task="embed", max_model_len=4096)
    vecs: List[List[float]] = []
    for batch in batched(texts, BATCH_SIZE):
        outs = model.embed(batch)
        vecs.extend([o.outputs.embedding for o in outs])
    emb = torch.tensor(vecs, dtype=torch.float32).cpu()
    print(f"Embeddings shape: {tuple(emb.shape)}")

    # Save the full embeddings and doc_ids directly
    np.save(OUT_ROOT / f"{MODEL_CHOICE}_full_embeddings.npy", emb.numpy())
    (OUT_ROOT / f"{MODEL_CHOICE}_doc_ids.json").write_text(
        json.dumps(doc_ids, ensure_ascii=False)
    )
    print(f"Saved full embeddings and doc IDs in {OUT_ROOT}")

    build_and_save_indexes(
        full_emb=emb,
        doc_ids=doc_ids,
        dims=EMBEDDING_DIMS,
        out_root=OUT_ROOT,
    )

    print("\nAll indexes saved under:", OUT_ROOT.resolve())

if __name__ == "__main__":
    main()
