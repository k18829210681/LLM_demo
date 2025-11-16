from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from tqdm.auto import tqdm
from pathlib import Path
import random, json
import hashlib
# ----------------------------
# Config
# ----------------------------
OUT_DIR = "/mnt/g/G/RAG/outputs/"
OUT_ALL = Path(OUT_DIR + "chunks.jsonl")
OUT_SAMPLED = Path(OUT_DIR + "chunks_sampled.jsonl")

N_SAMPLES = 500
random.seed(42)

def text_hash(t: str): 
    return hashlib.sha256(t.strip().encode()).hexdigest()

print("Loading dataset...")
ds = load_dataset("m-ric/huggingface_doc", split="train")

seen = set()
unique_docs = []
for rec in tqdm(ds, desc="Deduplicating"):
    text = rec.get("text", "").strip()
    if not text: 
        continue
    h = text_hash(text)
    if h in seen:
        continue
    seen.add(h)
    unique_docs.append(Document(page_content=text, metadata={"source": rec.get("source", "")}))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    add_start_index=True,
    separators=["\n\n", "\n", ".", " ", ""],
)

all_chunks = []
for doc in tqdm(unique_docs, desc="Chunking"):
    all_chunks.extend(splitter.split_documents([doc]))

OUT_ALL.parent.mkdir(parents=True, exist_ok=True)
with OUT_ALL.open("w", encoding="utf-8") as f:
    for idx, d in enumerate(tqdm(all_chunks, desc="Saving chunks")):
        record = {
            "index": idx,
            "source": d.metadata['source'],
            "start_index": d.metadata["start_index"],
            "page_content": d.page_content,
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
print(f"Saved {len(all_chunks)} sampled chunks to {OUT_ALL}")

# ----------------------------
# Sample 500 random chunks
# ----------------------------
if len(all_chunks) <= N_SAMPLES:
    sampled = all_chunks
else:
    sampled_indices = random.sample(range(len(all_chunks)), N_SAMPLES)

# ----------------------------
# Save to file
# ----------------------------
OUT_SAMPLED.parent.mkdir(parents=True, exist_ok=True)
with OUT_SAMPLED.open("w", encoding="utf-8") as f:
    for idx in tqdm(sampled_indices, desc="Saving sampled chunks"):
        d = all_chunks[idx]
        record = {
            "index": idx,
            "source": d.metadata['source'],
            "start_index": d.metadata["start_index"],
            "page_content": d.page_content,
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Saved {len(sampled_indices)} sampled chunks to {OUT_SAMPLED}")
