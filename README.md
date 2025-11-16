# GPT-2 Reproduction

  Dataset: FineWeb Edu 10B
  https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
  
  Reference (original implementation):
  https://github.com/karpathy/build-nanogpt
  
  Changes and improvements in this repo:
  
  Removed multi-GPU training. torch.compile() now works reliably.
  
  Improved the data loading flow, resulting in smoother training and reduced overfitting.

# Retrieval-Augmented Generation (RAG)

  ### A simple RAG pipeline designed to retrieve information from a book or other long documents.
  
  Basic process: documents → chunking → embedding → retrieval → LLM question/answer
  
  ### RAG Evaluation Components (work in progress)
  
  1. Chunking
  
  2. Deduplication using embeddings and LLM-based similarity judgment
  
  3. LLM generation of question–answer (QA) pairs from chunks
  
  4. LLM judgment of QA quality (whether questions are realistic and meaningful)
  
  5. Embedding model experiments using different models and vector dimensions (also used for deduplication in step 2)

6. Benchmarking embedding models and dimensions against validated QA pairs

7. Benchmarking reranker models using the same QA set
