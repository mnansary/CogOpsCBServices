#!/usr/bin/env python3
"""
Extract clean Wikipedia articles from the raw XML dump using mwparserfromhell.
Utilizes multiprocessing for speed and groups outputs into batched subfolders.
Files are saved using an article number as the filename to avoid OS path length limits.

Usage: python extract_dump.py [--input file.bz2][--output data/articles][--year 2024]
"""
import argparse
import bz2
import json
import os
import time
import xml.etree.ElementTree as ET
from multiprocessing import Pool, cpu_count

import mwparserfromhell

import re
import urllib.parse

def fix_links_in_wikitext(text: str) -> str:
    """
    Fixes two specific issues in wikitext URLs:
    1. Removes the web.archive.org prefix from archived links.
    2. Decodes URL-encoded characters (like Bangla text) into a readable format.
    """
    
    # --- ধাপ ১: web.archive.org লিঙ্ক থেকে আর্কাইভ প্রিফিক্স সরানো ---
    # এই প্যাটার্নটি 'https://web.archive.org/web/' এরপর যেকোনো সংখ্যা (টাইমস্ট্যাম্প)
    # এবং তারপর মূল http/https লিঙ্কটি খুঁজে বের করে।
    archive_pattern = r'https://web.archive.org/web/\d+/(https?://.+)'
    
    # re.sub ব্যবহার করে পুরো আর্কাইভ লিঙ্কটিকে শুধুমাত্র মূল লিঙ্ক (captured group 1) দিয়ে প্রতিস্থাপন করা হয়।
    # r'\1' মানে হলো প্যাটার্নের প্রথম ব্র্যাকেটের ভেতরের অংশটি।
    text_no_archive = re.sub(archive_pattern, r'\1', text)
    
    
    # --- ধাপ ২: URL-encoded বাংলা টেক্সট ডিকোড করা ---
    # এই প্যাটার্নটি টেক্সটের মধ্যে থাকা সব http/https লিঙ্ক খুঁজে বের করে।
    # এটি স্পেস, '|' বা ']' এর মতো ক্যারেক্টার পেলে থেমে যায়, যা উইকিটেক্সটে সাধারণ।
    url_pattern = r'https?://[^\s|\]\'"]+'
    
    # re.sub-এর একটি শক্তিশালী ফিচার হলো রিপ্লেসমেন্টের জন্য একটি ফাংশন ব্যবহার করা।
    # এখানে, প্রতিটি লিঙ্ক খুঁজে পাওয়ার পর, lambda ফাংশনটি কল করা হয়।
    # ফাংশনটি urllib.parse.unquote ব্যবহার করে লিঙ্কটিকে ডিকোড করে এবং ডিকোড করা লিঙ্কটি রিটার্ন করে।
    def decode_match(match):
        url = match.group(0) # পুরো ম্যাচ হওয়া লিঙ্কটি পাওয়া যায়
        return urllib.parse.unquote(url) # লিঙ্কটিকে ডিকোড করে রিটার্ন করা হয়
        
    fixed_text = re.sub(url_pattern, decode_match, text_no_archive)
    
    return fixed_text


def strip_namespace(tag: str) -> str:
    """Remove XML namespace prefix."""
    return tag.split("}", 1)[1] if "}" in tag else tag


def is_bengali_title(title: str) -> bool:
    """Return True if the title contains at least one Bengali Unicode letter."""
    if not title:
        return False
    for ch in title:
        if 0x0980 <= ord(ch) <= 0x09FF:
            return True
    return False


def is_raw_redirect(text: str) -> bool:
    """Fast check on raw text to bypass heavy parsing for redirects."""
    if not text:
        return True
    t = text.lstrip().lower()
    return t.startswith("#redirect") or t.startswith("#পুনর্নির্দেশ")


def article_generator(input_file, min_year):
    """Generates raw article data strictly for pages that pass basic filters."""
    cutoff = f"{min_year}-01-01T00:00:00Z"
    
    with bz2.open(input_file, "rt", encoding="utf-8") as fin:
        context = ET.iterparse(fin, events=("start", "end"))
        
        try:
            _, root = next(context)
        except StopIteration:
            return

        ns = title = text = page_id = last_updated = None

        for event, elem in context:
            tag = strip_namespace(elem.tag)

            if event == "start" and tag == "page":
                ns = title = text = page_id = last_updated = None

            elif event == "end":
                if tag == "id" and not page_id:
                    page_id = elem.text
                elif tag == "title":
                    title = elem.text
                elif tag == "ns":
                    ns = elem.text
                elif tag == "text":
                    text = elem.text
                elif tag == "timestamp" and not last_updated:
                    last_updated = elem.text

                elif tag == "page":
                    # Fast Pre-filters
                    if (ns == "0" and text and title and 
                        is_bengali_title(title) and 
                        (not last_updated or last_updated >= cutoff) and 
                        not is_raw_redirect(text)):
                        
                        # Yield raw data to be parsed by the worker pool
                        yield (title, text, page_id, last_updated)

                    # Strictly clear memory of parsed elements
                    elem.clear()
                    root.clear()


def process_article(raw_data):
    """
    Worker function: parses wikitext and builds the final JSON dictionary.
    Runs concurrently in the Multiprocessing Pool.
    """
    title, text, page_id, last_updated = raw_data
    
    try:
        parsed = mwparserfromhell.parse(text)
    except Exception:
        return None

    # 1. Extract summary (lead section)
    sections = parsed.get_sections(include_lead=True, flat=True)
    summary = sections[0].strip_code().strip() if sections else ""

    # Catch any leftover formatted redirects 
    if summary.lower().startswith("পুনর্নির্দেশ") or summary.lower().startswith("#redirect"):
        return None

    # 2. Extract full text
    full_text = parsed.strip_code().strip()
    if not full_text:
        return None
    #### WILL THIS JUST CHNAGE LINK OR THE WHOLE TEXT? ####
    text=fix_links_in_wikitext(text)
    return {
        "id": page_id,
        "title": title,
        "url": f"https://bn.wikipedia.org/wiki/{title.replace(' ', '_')}",
        "text": text,
        "last_updated": last_updated,
    }


def extract_dump(input_file, output_dir, min_year=2024):
    print(f"Reading {input_file}...", flush=True)
    start = time.time()
    os.makedirs(output_dir, exist_ok=True)

    # Leave one core free for system stability / main thread
    workers = max(1, cpu_count() - 1)
    print(f"Starting multiprocessing pool with {workers} workers...", flush=True)

    count = 0
    
    with Pool(processes=workers) as pool:
        raw_gen = article_generator(input_file, min_year)
        
        # pool.imap processes in parallel but yields results sequentially 
        for doc in pool.imap(process_article, raw_gen, chunksize=100):
            if doc:
                count += 1
                
                # Calculate the 1000-file grouping bounds
                folder_index = (count - 1) // 1000
                lower_bound = (folder_index * 1000) + 1
                upper_bound = lower_bound + 999
                subfolder_name = f"{lower_bound}-{upper_bound}"
                
                subfolder_path = os.path.join(output_dir, subfolder_name)
                
                # Create the specific sub-folder only when needed
                if count % 1000 == 1:
                    os.makedirs(subfolder_path, exist_ok=True)

                # Use article count number for filename to avoid OS limits on long Bengali characters
                fname = f"{count}.json"
                
                fpath = os.path.join(subfolder_path, fname)
                with open(fpath, "w", encoding="utf-8") as fout:
                    json.dump(doc, fout, ensure_ascii=False, indent=2)

                if count % 1000 == 0:
                    elapsed = round((time.time() - start) / 60, 2)
                    print(f"Saved {count} valid articles ({elapsed} min) - latest in folder {subfolder_name}", flush=True)

    elapsed = time.time() - start
    print(f"\nDone! {count} total articles extracted in {round(elapsed / 60, 2)} minutes.", flush=True)
    print(f"Output saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract clean Wikipedia articles")
    parser.add_argument(
        "--input",
        default="resources/bnwiki-latest-pages-articles.xml.bz2",
        help="Path to the raw Wikipedia XML dump",
    )
    parser.add_argument(
        "--output",
        default="resources/articles",
        help="Output directory for individual JSON files",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2024,
        help="Minimum year for last_updated filter (default: 2024)",
    )
    args = parser.parse_args()
    
    extract_dump(args.input, args.output, min_year=args.year)