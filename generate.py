import torch
import sentencepiece as spm
from train_gpt import GPTLanguageModel, block_size, device

# =====================
# Load tokenizer
# =====================
sp = spm.SentencePieceProcessor()
sp.load("gutenberg_bpe.model")   # MUST match training tokenizer

def encode(text: str):
    return sp.encode(text, out_type=int)

def decode(tokens):
    return sp.decode(tokens)

# =====================
# Load model
# =====================
model = GPTLanguageModel().to(device)

ckpt = torch.load(
    "checkpoints/latest.pt",
    map_location=device,
    weights_only=True
)

model.load_state_dict(ckpt["model"])
model.eval()

print("✅ FirstWordGPT loaded")
print("Type 'exit' or 'quit' to stop.\n")

# =====================
# Generation
# =====================
@torch.no_grad()
def generate(
    tokens,
    max_new_tokens=300,
    temperature=0.7,
    top_k=50
):
    for _ in range(max_new_tokens):
        idx_cond = tokens[:, -block_size:]

        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, ix = torch.topk(logits, top_k)
            probs = torch.softmax(v, dim=-1)
            next_token = ix.gather(-1, torch.multinomial(probs, 1))
        else:
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

        tokens = torch.cat([tokens, next_token], dim=1)

    return tokens

# =====================
# Interactive loop
# =====================
if __name__ == "__main__":
    while True:
        prompt = input("🧑 Prompt: ").strip()

        if prompt.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        tokens = torch.tensor(
            [encode(prompt)],
            dtype=torch.long,
            device=device
        )

        out = generate(tokens)

        generated_text = decode(out[0].tolist())

        print("\n📝 Generated text:\n")
        print(generated_text)
        print("\n" + "-" * 60 + "\n")
