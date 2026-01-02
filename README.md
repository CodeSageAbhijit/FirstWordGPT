FirstWordGPT 🧠
Training a GPT-Style Language Model from Scratch using Project Gutenberg

FirstWordGPT is a complete, end-to-end implementation of a GPT-style language model trained entirely from scratch using PyTorch and SentencePiece, without relying on any pretrained models or high-level frameworks.

The name FirstWordGPT reflects the core idea of this project: learning how a language model predicts the very next word, step by step, until coherent language emerges.

This project is built for educational and experimental purposes, with the explicit goal of understanding how modern large language models are created internally — from raw text to a working autoregressive transformer.

The full pipeline is implemented manually: dataset collection, cleaning, tokenization, model training, checkpointing, and text generation.

If you understand this repository, you understand how GPT-style models actually work.

🚀 Project Overview

Trained a GPT-style Transformer from scratch

Uses Project Gutenberg public-domain books

Custom SentencePiece BPE tokenizer

Pure PyTorch implementation (no pretrained models)

Mixed Precision (AMP) training support

Learning-rate warmup + cosine decay

Checkpoint saving and resume capability

Autoregressive text generation from trained checkpoints

📊 Model & Dataset Details
Model Architecture
Component	Value
Parameters	~22.8M
Layers	8 Transformer blocks
Attention Heads	8
Embedding Size	256
Context Length	256 tokens
Vocabulary Size	8,000 (BPE)
Architecture	Decoder-only GPT
Dataset
Item	Value
Source	Project Gutenberg
Books	~5,000
Clean text size	~1.24 GB
Total tokens	~300 million
Avg chars/token	~4.1
License	Public Domain
🧪 Sample Output

Prompt

Abhijit went to the market to buy some fresh vegetables. On his way, he met


Generated

a large family of his own, and of the most beautiful of them had come to see him.
The chief was the first to get a large piece of bread with his wife, and the rest
of the family gathered around the table...


The model demonstrates:

Sentence-level coherence

Narrative continuation

Long-form structure within context limits

📁 Project Structure
FirstWordGPT/
├── checkpoints/
│   └── latest.pt              # Training checkpoint
│
├── gutenberg_books/           # Raw downloaded books
├── gutenberg_clean/           # Cleaned per-book text
├── gutenberg_clean.txt        # Final merged dataset
├── gutenberg_tokens.pt        # Tokenized dataset (torch tensor)
│
├── train_gpt.py               # Main training script
├── generate.py                # Text generation script
│
├── train_tokenizer.py         # SentencePiece BPE training
├── token_counter.py           # Dataset statistics
│
├── dataset_downloader.py      # Gutenberg downloader
├── data_cleaning.py           # Header/footer removal
├── merge_books.py             # Merge cleaned books
├── prepare_dataset.py         # Full preprocessing pipeline
│
├── gutenberg_bpe.model        # Tokenizer model
├── gutenberg_bpe.vocab        # Tokenizer vocab
└── README.md

🛠️ Installation
Requirements

Python 3.10+

PyTorch 2.0+

CUDA-enabled GPU (recommended)

SentencePiece

Install dependencies
pip install torch sentencepiece tqdm

📥 Step-by-Step: Train FirstWordGPT from Scratch
1️⃣ Download Project Gutenberg Books
python dataset_downloader.py

2️⃣ Clean Gutenberg Headers & Footers
python data_cleaning.py


Removes:

License text

Metadata

Boilerplate

Excess blank lines

3️⃣ Merge All Books into One Dataset
python merge_books.py


Creates:

gutenberg_clean.txt

4️⃣ Train the BPE Tokenizer
python train_tokenizer.py


Outputs:

gutenberg_bpe.model
gutenberg_bpe.vocab

5️⃣ Tokenize Dataset (One-Time)
python prepare_dataset.py


Creates:

gutenberg_tokens.pt

6️⃣ Train the GPT Model
python train_gpt.py


Training features:

Automatic Mixed Precision (AMP)

Learning-rate warmup + cosine decay

Checkpoint saving & resume support

Checkpoints saved to:

checkpoints/latest.pt

7️⃣ Generate Text
python generate.py

⚙️ Training Configuration
batch_size = 32
block_size = 256

n_layer = 8
n_head = 8
n_embd = 256

learning_rate = 3e-4
warmup_iters = 2000
max_iters = 40_000
dropout = 0.1

🧠 What This Project Demonstrates

Most tutorials:

Load pretrained GPT-2

Fine-tune a dataset

Hide complexity

FirstWordGPT:

Builds everything from zero

Exposes real training bottlenecks

Shows why small models repeat

Demonstrates how coherence emerges token by token

This is foundational LLM engineering, not a wrapper script.

📜 License

Dataset: Public Domain (Project Gutenberg)

Code: Educational / Research Use