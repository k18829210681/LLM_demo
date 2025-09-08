Just a simple demo repo to show part of my work on LLM.

1. gpt2 reproduce. Dataset fineweb 10B: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Credit: https://github.com/karpathy/build-nanogpt
Improvements / changes: 
	1. dropped muti-gpu training, torch.compile now works
	2. improved the data loading flow, the training is smoother and less overfit.

2. RAG: simple RAG system to retrieve data from a book. 
Steps:
	1. load data, segment text into sentences using spacy (better than just using ".")
	2. build page - sentence table. later if need more context, can feed the whole page.
	3. chunking size: 2000 chars, about 500 tokens. This groups sentences but not break them, so chunks have different sizes. Also 50% overlap, so each sentence is used twice.
	4. get embedding from latest Qwen3 0.6B model.
	5. ask questions, get the context.
	6. final LLM part is not included, since it is an independent choice.
