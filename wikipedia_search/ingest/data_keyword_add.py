#!/usr/bin/env python3
"""
Parallel keyword extraction for existing article JSON files.
Reads JSON files from input_dir (and its subdirectories), computes Bengali keywords, 
writes updated JSON files back (overwrites).

Usage: python add_keywords.py [--input resources/articles] [--workers 8]
"""
import json
import os
import re
import argparse
import time
from multiprocessing import Pool, cpu_count

# Bengali stopwords
BENGALI_STOPWORDS = {
    "এই", "আমরা", "তা", "তাই", "তবে", "দেখা", "দেওয়া", "দিয়ে", "নয়",
    "নিয়ে", "পরে", "মধ্যে", "বলা", "বলতে", "বলে", "হয়", "হয়ে",
    "হয়নি", "হল", "হচ্ছে", "হতে", "হচ্ছেনা", "আছে", "আর", "অনেক",
    "অন্য", "উচিত", "এখন", "এখানে", "একটি", "একটা", "করা", "করে",
    "করেছি", "করেছেন", "কি", "কিন্তু", "কোনো", "চেয়ে",
    "ছাড়া", "জানা", "জানতে", "তার", "তারা", "তাদের", "না", "নিজে",
    "পর", "প্রতি", "প্রথম", "বহু", "ভালো", "মতো", "মনে",
    "যা", "যাই", "যদি", "যে", "যেই", "যেগুলো", "যে কারণে", "যাঁহার",
    "যাঁকে", "যেমন", "যেখানে", "যেহেতু", "যেন", "র", "রা", "রাখা",
    "লব", "শেষ", "সে", "সেই", "সেইটি", "সেগুলো", "সেই কারণে",
    "সেই যে", "সেখানে", "সব", "সবার", "সবচেয়ে", "সময়", "সবগুলো",
    "সরি", "সবাই", "সেজন্য", "হওয়া", "হয়ে", "হবে", "হলে",
    "হলো", "হচ্ছে", "হল", "হয়েছে", "হয়েছিল", "হয়", "হয়নি",
    "আমি", "আমার", "আমরা", "আমাদের", "তুমি", "তোমার", "তোমরা",
    "তোমাদের", "আপনি", "আপনার", "আপনারা",
    "ও", "ওই", "এ", "এই", "এটা", "একটা", "সেটা", "সেকি",
    "কি", "কী", "কোন", "কোনো", "কোনটি", "কোনটা", "যা", "যাটা",
    "যে", "যেটা", "যেই", "যেখানে", "যেহেতু", "যখন", "যত", "যাই",
    "কী", "কেন", "কোথায়", "কীভাবে", "কভাবে", "কাদের", "কাদেরকে",
    "কাদেরও", "কাকে", "কার", "কাজ", "কাজে", "কে", "কেউ", "কেউও",
    "কেহ", "কে বা", "কেও", "কে কার", "কে বা কারও", "কারও",
}

_RE_BENGALI = re.compile(r"[^\sঀ-৿᧰-᧿]")


def _extract_keywords(text: str) -> str:
    """Extract unique Bengali keywords (stopword-removed, pipe-separated)."""
    cleaned = _RE_BENGALI.sub(" ", text).strip()
    words = cleaned.split()
    seen = set()
    result =[]
    for w in words:
        if len(w) < 2 or w in BENGALI_STOPWORDS or w in seen:
            continue
        seen.add(w)
        result.append(w)
    return "|".join(result)


def _process_file(fpath: str):
    """Add keywords to a single article JSON file."""
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except (json.JSONDecodeError, KeyError, FileNotFoundError):
        return None

    full_text = doc.get("text", "")
    if not full_text:
        return None

    new_keywords = _extract_keywords(full_text)

    # Merge with any existing keywords array
    existing = doc.get("keywords", [])
    if isinstance(existing, list) and existing:
        # existing is a list — combine, deduplicate
        all_words = list(dict.fromkeys(existing + new_keywords.split("|")))
        doc["keywords"] = all_words
    else:
        doc["keywords"] = new_keywords.split("|")

    # Overwrite the file with the new keywords
    try:
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
    except Exception:
        return None

    return fpath


def add_keywords(input_dir: str, num_workers: int = 0):
    if num_workers == 0:
        # Leave one core free
        num_workers = max(1, cpu_count() - 1)

    print(f"Scanning {input_dir} for JSON files...", flush=True)
    
    # Recursively find all JSON files in the new subfolder structure
    json_files = []
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if fname.endswith(".json"):
                json_files.append(os.path.join(root, fname))

    if not json_files:
        print("No JSON files found.", flush=True)
        return

    print(f"Found {len(json_files)} files. Starting pool with {num_workers} workers...", flush=True)

    count = 0
    with Pool(processes=num_workers) as pool:
        for result in pool.imap(_process_file, json_files, chunksize=100):
            if result:
                count += 1
                if count % 1000 == 0:
                    print(f"Processed {count} articles...", flush=True)

    print(f"Done! {count} articles updated with keywords.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add Bengali keywords to extracted Wikipedia articles")
    parser.add_argument(
        "--input",
        default="resources/articles",
        help="Input directory containing article JSON files (default: resources/articles)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes (default: cpu_count - 1)",
    )
    args = parser.parse_args()
    add_keywords(args.input, num_workers=args.workers)