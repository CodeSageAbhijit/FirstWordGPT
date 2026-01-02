import sentencepiece as spm
import torch
from tqdm import tqdm

sp = spm.SentencePieceProcessor()
sp.load("gutenberg_bpe.model")

input_file = "gutenberg_clean.txt"
output_file = "gutenberg_tokens.pt"

tokens = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Tokenizing"):
        line = line.strip()
        if not line:
            continue
        tokens.extend(sp.encode(line, out_type=int))

tokens = torch.tensor(tokens, dtype=torch.int32)
torch.save(tokens, output_file)

print(f"✅ Saved {tokens.numel():,} tokens to {output_file}")
