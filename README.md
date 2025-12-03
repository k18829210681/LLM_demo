# GPT-2 Reproduction

Dataset: FineWeb Edu 10B
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

Reference (original implementation):
https://github.com/karpathy/build-nanogpt

Changes and improvements in this repo:

Removed multi-GPU training. torch.compile() now works reliably.

Improved the data loading flow, resulting in smoother training and reduced overfitting.


# RAG Dataset Generation Pipeline

A complete pipeline for generating synthetic question-answer pairs from Hugging Face documentation, designed for evaluating Retrieval-Augmented Generation (RAG) systems.

## Overview

This pipeline automates the creation of RAG training datasets through 7 sequential steps:

1. **Chunking** - Split documentation into manageable chunks
2. **Deduplication** - Remove redundant content using embeddings + LLM judging
3. **QA Generation** - Generate question-answer pairs from chunks
4. **QA Judging** - Evaluate QA quality on multiple dimensions
5. **Embedding** - Create embeddings for chunks and queries
6. **Reranking** - Rerank retrieved results for improved accuracy
7. **Benchmarking** - Measure retrieval performance across configurations

**Dataset Source**: [m-ric/huggingface_doc](https://huggingface.co/datasets/m-ric/huggingface_doc)

Due to the size limit of github, only LLM output samples are kept in the outputs folder.

## Quick Start

### Run the Complete Pipeline

```bash
python run_pipeline.py
```
```bash
python run_pipeline.py --test
```

The pipeline will execute all steps in order and save results to the `outputs/` directory. --test will load part of the chunks and start a shorter test run.

### Run Individual Steps

You can also run any step independently.

## Pipeline Steps Explained

### 1. Chunking
- Loads Hugging Face documentation from dataset
- Removes duplicate documents by hash
- Splits into x-character chunks with x*0.1-character overlap
- **Output**: `outputs/chunks/chunks.jsonl`

### 2. Deduplication
- Embeds all chunks using Qwen3-Embedding models
- Finds similar pairs
- Auto-removes chunks with similarity ≥ 0.995
- Uses LLM to judge pairs with similarity 0.97-0.995
- Samples subset for faster QA generation
- **Outputs**: `outputs/chunks/chunks_dedu.jsonl`, `outputs/deduplication/{model}_dedu.jsonl`

### 3. QA Generation
- Generates question-answer pairs from chunks
- Uses multiple LLM models (Qwen, GPT)
- Each chunk produces one QA pair
- **Outputs**: `outputs/qa/{model}_qa.jsonl`

### 4. QA Judging
- Evaluates QA quality on three dimensions:
  - **Groundedness**: Can the question be answered from context?
  - **Relevance**: Is it useful for ML developers?
  - **Standalone**: Is the question self-contained?
- Filters QA pairs that pass all thresholds
- Selects best QA when multiple models generate for same chunk
- **Outputs**: `outputs/qa/qa_selected.jsonl`, `outputs/judgments/{judge}_{qa}_qa.jsonl`

### 5. Embedding
- Embeds deduplicated chunks with multiple embedding models
- Embeds queries with different instruction variants:
  - `base`: Generic retrieval
  - `ml`: ML-specific retrieval
  - `ml_hf`: Hugging Face-specific retrieval
- **Outputs**: `outputs/embeddings/{model}_chunks.npy`, `outputs/embeddings/{model}_{instruction}_query.npy`

### 6. Reranking
- Retrieves top-k chunks using embedding similarity
- Reranks results using Qwen3-Reranker models
- Tests different embedding and reranking model combinations
- **Outputs**: `outputs/reranking/{embedding_model}_{reranker_model}_{instruction}_reranking.jsonl`

### 7. Benchmarking
- Measures retrieval performance:
  - **Top-k Accuracy**: Is ground truth in top-k results?
  - **MRR**: Mean Reciprocal Rank
  - **MAP**: Mean Average Precision
- Tests multiple embedding dimensions (32 to 4096)
- Compares embedding-only vs. embedding+reranking
- **Outputs**: `outputs/benchmarks/benchmark_results.json`, `outputs/benchmarks/benchmark_results.csv`

## Prompts

All LLM prompts are in [prompts.yaml](prompts.yaml):

- `QA_generation_prompt` - Generate factoid QA pairs
- `question_groundedness_critique_prompt` - Judge groundedness
- `question_relevance_critique_prompt` - Judge relevance
- `question_standalone_critique_prompt` - Judge standalone quality
- `document_duplication_critique_prompt` - Judge chunk duplication

## Technical Notes

### 1. MXFP4 Quantization Warning

If you encounter this warning:
```
warning: MXFP4 quantization requires triton >= 3.4.0 and kernels installed,
we will default to dequantizing the model to bf16
```

**Solution**: Upgrade dependencies
```bash
pip install --upgrade torch
pip install git+https://github.com/huggingface/transformers triton==3.4 kernels
pip uninstall torchvision torchaudio -y
```

**Reference**: [Colab Notebook](https://colab.research.google.com/drive/15DJv6QWgc49MuC7dlNS9ifveXBDjCWO5?usp=sharing#scrollTo=7GW0knW2w3ND)

### 2. vLLM Reproducibility

vLLM does not guarantee deterministic outputs across runs, even with fixed random seeds. For details, see [vLLM Reproducibility Documentation](https://docs.vllm.ai/en/stable/usage/reproducibility.html).

### 3. GPU Memory Management

To ensure complete GPU memory cleanup, this pipeline runs vLLM in separate subprocesses. Each step spawns independent processes that fully release GPU memory upon completion, preventing out-of-memory errors when running the full pipeline.

## License

MIT License - See LICENSE file for details

