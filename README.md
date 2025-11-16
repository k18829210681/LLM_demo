Just a simple demo repo to show part of my work on LLM.

1. gpt2 reproduce. Dataset fineweb 10B: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Credit: https://github.com/karpathy/build-nanogpt
Improvements / changes: 
	1. dropped muti-gpu training, torch.compile now works
	2. improved the data loading flow, the training is smoother and less overfit.

2. RAG: simple RAG system to retrieve data from a book. 
	0: simple RAG pipeline: data --> chunking --> embedding --> ask questions.

   	1-6 complete RAG evaluation (work in progress):
   	1. chunking
   	2. Use embedding and LLM to deduplicate.
   	3. Use LLM to generate question - answer pair based on the chunks.
   	4. Use LLM to judge if the questions are valid and close to real questions asked by humans.
   	5. Embedding, testing different different embedding model and dimensions. This is also used for deduplication before 2.
   	6. Benchmark different embedding model and dimensions against the selected QA.
   	7. Benchmark reranker models against the QA.
