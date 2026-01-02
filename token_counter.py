import sentencepiece as spm
from tqdm import tqdm
import os


TOKENIZER_PATH = "gutenberg_bpe.model"
DATA_FILE = "gutenberg_clean.txt"


sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_PATH)


total_tokens = 0
total_chars = 0
total_lines = 0


with open(DATA_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="🔢 Counting tokens"):
        line = line.strip()
        if not line:
            continue

        token_ids = sp.encode(line, out_type=int)
        total_tokens += len(token_ids)
        total_chars += len(line)
        total_lines += 1


# Report

print("\n📊 Gutenberg Token Statistics")
print("=" * 40)
print(f"📄 Lines           : {total_lines:,}")
print(f"🔤 Characters      : {total_chars:,}")
print(f"🧠 Tokens          : {total_tokens:,}")
print(f"📐 Avg tokens/line : {total_tokens / max(1, total_lines):.2f}")
print(f"📐 Chars/token     : {total_chars / max(1, total_tokens):.2f}")


# Training readiness hint

print("\n🚀 Training Guidance")
if total_tokens < 100_000_000:
    print("⚠️ Dataset is small → train fewer steps or smaller model")
elif total_tokens < 300_000_000:
    print("✅ Dataset size is IDEAL for your 40M model")
elif total_tokens < 600_000_000:
    print("⚠️ Large dataset → train partial epochs")
else:
    print("🔥 Huge dataset → sample or train in chunks")
