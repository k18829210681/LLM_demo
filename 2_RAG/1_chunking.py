#!/usr/bin/env python3
"""
Step 1: Chunking

Load documents from Hugging Face, deduplicate by hash, and split into chunks.

Outputs:
  - outputs/chunks/chunks.jsonl - All chunks with sequential IDs
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from datasets import load_dataset
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

from utils import load_config, setup_output_dirs, write_jsonl_incremental

# ====================================================================
# Configuration
# ====================================================================

BASE_DIR = Path(__file__).resolve().parent
config = load_config()

RANDOM_SEED = config["random_seed"]
BATCH_SIZE = config["batch_size"]

DATASET_NAME = config["chunking"]["dataset"]
CHUNK_SIZE = config["chunking"]["chunk_size"]
CHUNK_OVERLAP = config["chunking"]["chunk_overlap"]

# Setup output directories
dirs = setup_output_dirs(config)
CHUNKS_FILE = dirs["chunks"] / "chunks.jsonl"


# ====================================================================
# Main Processing
# ====================================================================


def hash_text(text: str) -> str:
    """Generate SHA-256 hash of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    print("=" * 70)
    print("STEP 1: CHUNKING")
    print("=" * 70)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Chunk size: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP}")
    print(f"Output: {CHUNKS_FILE}")
    print("=" * 70)

    # Load dataset
    print("\nLoading dataset...")
    ds = load_dataset(DATASET_NAME, split="train")
    print(f"Loaded {len(ds)} documents")

    # Deduplicate documents by hash
    print("\nDeduplicating documents by hash...")
    seen_hashes = set()
    unique_docs = []

    for doc in tqdm(ds, desc="Hashing documents"):
        text = doc["text"]
        doc_hash = hash_text(text)
        if doc_hash not in seen_hashes:
            seen_hashes.add(doc_hash)
            # Create Document object with metadata
            unique_docs.append(
                Document(
                    page_content=text,
                    metadata={"source": doc.get("source", "")},
                )
            )

    print(f"Unique documents: {len(unique_docs)} (removed {len(ds) - len(unique_docs)} duplicates)")

    # Split into chunks
    print("\nSplitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    all_chunks = []
    chunk_id = 0

    for doc in tqdm(unique_docs, desc="Chunking documents"):
        for chunk in splitter.split_documents([doc]):
            all_chunks.append({
                "id": chunk_id,
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", ""),
                "start_index": chunk.metadata.get("start_index", 0),
            })
            chunk_id += 1

    print(f"Total chunks: {len(all_chunks)}")

    # Write chunks to file
    print(f"\nWriting chunks to {CHUNKS_FILE}...")
    write_jsonl_incremental(all_chunks, CHUNKS_FILE, batch_size=BATCH_SIZE)

    print(f"\n✓ Step 1 completed successfully!")
    print(f"  Chunks saved: {CHUNKS_FILE}")
    print(f"  Total chunks: {len(all_chunks)}")


if __name__ == "__main__":
    main()
