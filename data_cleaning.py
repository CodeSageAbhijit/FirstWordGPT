import os
import re
from tqdm import tqdm

INPUT_DIR = "gutenberg_books"
OUTPUT_DIR = "gutenberg_clean"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Enhanced start markers
START_MARKERS = [
    "*** START OF THIS PROJECT GUTENBERG EBOOK",
    "*** START OF THE PROJECT GUTENBERG EBOOK",
    "*** START OF PROJECT GUTENBERG EBOOK",
    "***START OF THE PROJECT GUTENBERG EBOOK",
]

# Enhanced end markers
END_MARKERS = [
    "*** END OF THIS PROJECT GUTENBERG EBOOK",
    "*** END OF THE PROJECT GUTENBERG EBOOK",
    "*** END OF PROJECT GUTENBERG EBOOK",
    "***END OF THE PROJECT GUTENBERG EBOOK",
]

# Comprehensive metadata removal patterns
META_PATTERNS = [
    # License sections
    r"(?is)START: FULL LICENSE.*?END OF THE FULL PROJECT GUTENBERG LICENSE",
    r"(?is)THE FULL PROJECT GUTENBERG LICENSE.*?(?=\n\n[A-Z]|\Z)",
    r"(?is)Section \d+\..*?Project Gutenberg.*?(?=\n\n[A-Z]|\Z)",
    
    # Transcriber/Editor notes
    r"(?is)Transcribers?[''']?\s*Notes?:?.*?(?=\n\n\n|\*\*\*)",
    r"(?is)Editors?[''']?\s*Notes?:?.*?(?=\n\n\n|\*\*\*)",
    r"(?is)\[.*?Editor.*?\]",
    r"(?is)\[.*?Transcriber.*?\]",
    
    # Project Gutenberg metadata
    r"(?is)This ebook is for the use of anyone anywhere.*?before using this eBook\.",
    r"(?is)Updated editions will replace.*?(?=\n\n[A-Z])",
    r"(?is)Title:.*?Credits:.*?(?=\*\*\*)",
    r"(?is)Release date:.*?Language:.*?(?=\*\*\*)",
    r"(?is)Author:.*?Release date:.*?(?=\*\*\*)",
    
    # Copyright/production notes
    r"(?is)This file was never copyrighted.*?(?=\n\n)",
    r"(?is)In my research for creating this transcription.*?(?=\n\n|\*\*\*)",
    r"(?is)Produced by.*?Project Gutenberg.*?(?=\n\n)",
    
    # Image/illustration references
    r"\{[^}]*\.(jpg|png|gif)[^}]*\}",
    r"\[Illustration:.*?\]",
    r"\[Image:.*?\]",
    
    # Footnote markers (but keep the text)
    r"\[\d+\]",
    r"\{\d+\}",
]

def clean_text(text):
    """Clean Project Gutenberg text comprehensively"""
    text = text.replace("\r\n", "\n")
    
    # 1. Find and extract only the main content
    start_idx = -1
    for marker in START_MARKERS:
        idx = text.upper().find(marker.upper())
        if idx != -1:
            # Find the end of this line to start after the marker line
            line_end = text.find("\n", idx)
            start_idx = line_end + 1 if line_end != -1 else idx + len(marker)
            break
    
    end_idx = len(text)
    for marker in END_MARKERS:
        idx = text.upper().find(marker.upper())
        if idx != -1:
            end_idx = idx
            break
    
    # Extract main content
    if start_idx != -1:
        text = text[start_idx:end_idx]
    
    # 2. Remove metadata patterns
    for pattern in META_PATTERNS:
        text = re.sub(pattern, "", text)
    
    # 3. Remove lines with only asterisks or dashes
    text = re.sub(r"^\s*[\*\-_=]{3,}\s*$", "", text, flags=re.MULTILINE)
    
    # 4. Remove page numbers (standalone numbers on lines)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    
    # 5. Remove excessive whitespace and normalize
    text = re.sub(r"[ \t]+", " ", text)  # Multiple spaces to single space
    text = re.sub(r" \n", "\n", text)  # Remove trailing spaces before newlines
    text = re.sub(r"\n ", "\n", text)  # Remove leading spaces after newlines
    
    # 6. Remove ALL blank lines - join paragraphs into continuous text
    # Split by double newlines first to preserve paragraph boundaries
    paragraphs = re.split(r"\n\s*\n", text)
    
    # Clean each paragraph by joining all lines with a space
    cleaned_paragraphs = []
    for para in paragraphs:
        # Remove single newlines within paragraph and join
        para = para.strip()
        if para:
            # Replace single newlines with space to join wrapped lines
            para = re.sub(r"\n", " ", para)
            # Clean up multiple spaces
            para = re.sub(r"\s+", " ", para)
            cleaned_paragraphs.append(para)
    
    # Join paragraphs with double newline to maintain paragraph structure
    text = "\n\n".join(cleaned_paragraphs)
    
    return text.strip()

# Process all files
processed = 0
skipped = 0

for fname in tqdm(os.listdir(INPUT_DIR)):
    if not fname.endswith(".txt"):
        continue
    
    in_path = os.path.join(INPUT_DIR, fname)
    out_path = os.path.join(OUTPUT_DIR, fname)
    
    try:
        with open(in_path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        
        cleaned = clean_text(raw)
        
        # Skip files that are too small after cleaning
        if len(cleaned) < 10_000:
            skipped += 1
            continue
        
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(cleaned)
        
        processed += 1
        
    except Exception as e:
        print(f"\n❌ Error processing {fname}: {e}")
        continue

print(f"\n✅ Gutenberg cleaning COMPLETE!")
print(f"📊 Processed: {processed} files")
print(f"⏭️  Skipped: {skipped} files (too short after cleaning)")
