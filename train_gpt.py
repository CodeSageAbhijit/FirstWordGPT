import os
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import sentencepiece as spm

from torch.amp import autocast, GradScaler  # torch>=2.x

# Hyperparameters

batch_size = 32
block_size = 256

max_iters = 40_000
eval_interval = 1000
eval_iters = 200

learning_rate = 3e-4
min_lr = 3e-5
warmup_iters = 2000

n_embd = 256
n_head = 8
n_layer = 8
dropout = 0.1

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)

use_amp = (device == "cuda")  # AMP only on CUDA [web:63]


# Tokenizer

sp = spm.SentencePieceProcessor()
sp.load("gutenberg_bpe.model")
vocab_size = sp.get_piece_size()

def encode(text: str):
    
    return sp.encode(text, out_type=int)


# Data

# Best practice: tokenize ONCE -> save tensor -> train from tensor.
TOK_PATH = "gutenberg_tokens.pt"

if os.path.exists(TOK_PATH):
    data = torch.load(TOK_PATH, map_location="cpu", weights_only=True)
else:
    with open("gutenberg_clean.txt", "r", encoding="utf-8") as f:
        ids = encode(f.read())
    data = torch.tensor(ids, dtype=torch.long)
    torch.save(data, TOK_PATH)


data = data.to(torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# Batch loader

def get_batch(split: str):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size - 1, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])

    x = x.to(device=device, dtype=torch.long)
    y = y.to(device=device, dtype=torch.long)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = []
        for _ in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses.append(loss.detach().float().cpu())
        out[split] = torch.stack(losses).mean().item()
    model.train()
    return out


# Model

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return self.dropout(out)

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention()
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_emb(idx)

        
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss


# LR schedule

def get_lr(step):
    if step < warmup_iters:
        return learning_rate * step / warmup_iters
    progress = (step - warmup_iters) / max(1, (max_iters - warmup_iters))
    return min_lr + 0.5 * (learning_rate - min_lr) * (1 + math.cos(math.pi * progress))


# Train

if __name__ == "__main__":
    model = GPTLanguageModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    
    scaler = GradScaler(enabled=use_amp)

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/latest.pt"
    start_iter = 0

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = int(ckpt.get("iter", 0)) + 1
        print(f"🔁 Resumed from step {start_iter - 1}")

    print(f"🧠 Model size: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    print(f"🖥️ Device: {device} | AMP: {use_amp}")

    for step in range(start_iter, max_iters):
        lr = get_lr(step)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if step % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {step} | train {losses['train']:.3f} | val {losses['val']:.3f}")
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iter": step},
                ckpt_path,
            )

        xb, yb = get_batch("train")

        optimizer.zero_grad(set_to_none=True)

        
        with autocast(device_type=device, enabled=use_amp):
            _, loss = model(xb, yb)

        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
