import os
from tqdm import tqdm

CLEAN_DIR = "gutenberg_clean"
OUT_FILE = "gutenberg_clean.txt"

with open(OUT_FILE, "w", encoding="utf-8") as out:
    for fname in tqdm(sorted(os.listdir(CLEAN_DIR))):
        if not fname.endswith(".txt"):
            continue

        with open(os.path.join(CLEAN_DIR, fname), "r", encoding="utf-8") as f:
            text = f.read().strip()

        out.write(text)
        out.write("\n\n")

print(f"✅ Merged dataset saved to {OUT_FILE}")
