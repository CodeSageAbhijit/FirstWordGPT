import os
import requests
from tqdm import tqdm


OUTPUT_DIR = "gutenberg_books"
START_ID = 1
END_ID = 5000        # change to 1000 / 2000 / 5000 to download more books and increase dataset size
TIMEOUT = 10

BASE_URL = "https://www.gutenberg.org/cache/epub"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_book(book_id):
    """
    Try common Gutenberg TXT locations
    """
    candidates = [
        f"{BASE_URL}/{book_id}/pg{book_id}.txt",
        f"{BASE_URL}/{book_id}/pg{book_id}.txt.utf8",
        f"{BASE_URL}/{book_id}/pg{book_id}.txt.utf-8",
    ]

    for url in candidates:
        try:
            r = requests.get(url, timeout=TIMEOUT)
            if r.status_code == 200 and len(r.text) > 1000:
                return r.text
        except requests.RequestException:
            pass

    return None


downloaded = 0

for book_id in tqdm(range(START_ID, END_ID + 1)):
    out_path = os.path.join(OUTPUT_DIR, f"book_{book_id}.txt")

    if os.path.exists(out_path):
        continue

    text = download_book(book_id)
    if text is None:
        continue

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    downloaded += 1

print(f"\n✅ Downloaded {downloaded} books into '{OUTPUT_DIR}'")
