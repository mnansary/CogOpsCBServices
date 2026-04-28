#!/usr/bin/env python3
"""
Ingest Wikipedia article JSON files into Elasticsearch.
Maps: title + text + keywords + url as searchable fields.
Only unique keywords are indexed (they are already deduplicated by data_keyword_add.py).

Usage: python data_ingest.py [--input resources/articles]
"""
import argparse
import json
import os
import time

import yaml
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, BulkIndexError


def load_config(path="config.yml"):
    with open(path) as f:
        return yaml.safe_load(f)


def get_json_files(input_dir):
    files = []
    for root, _, filenames in os.walk(input_dir):
        for fname in filenames:
            if fname.endswith(".json"):
                files.append(os.path.join(root, fname))
    return sorted(files)


def run_ingestion(input_dir, config_path="config.yml"):
    cfg = load_config(config_path)
    es_url = cfg["es"]["url"]
    index_name = cfg["es"]["index"]
    timeout = cfg["es"].get("timeout", 60)

    es = Elasticsearch(es_url, request_timeout=timeout)
    print(f"Connected to Elasticsearch at {es_url}", flush=True)

    # Create index with proper mappings
    if es.indices.exists(index=index_name):
        print(f"[*] Index '{index_name}' already exists. Deleting and recreating...", flush=True)
        es.indices.delete(index=index_name)

    mapping = {
        "mappings": {
            "properties": {
                "title": {
                    "type": "text",
                    "analyzer": "standard",
                },
                "url": {"type": "keyword"},
                "text": {
                    "type": "text",
                    "analyzer": "standard",
                },
                "keywords": {
                    "type": "keyword",
                },
                "last_updated": {"type": "date"},
            }
        }
    }
    es.indices.create(index=index_name, body=mapping)
    print(f"[+] Fresh index '{index_name}' created.", flush=True)

    # Collect JSON files
    json_files = get_json_files(input_dir)
    if not json_files:
        print("No JSON files found.", flush=True)
        return

    print(f"[*] Found {len(json_files)} article files to index.", flush=True)

    BATCH_SIZE = 1000
    total_indexed = 0
    start = time.time()

    for i in range(0, len(json_files), BATCH_SIZE):
        batch_files = json_files[i:i + BATCH_SIZE]
        actions = []

        for fpath in batch_files:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    doc = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                continue

            keywords = doc.get("keywords", [])
            if isinstance(keywords, list):
                keywords = list(dict.fromkeys(keywords))

            actions.append({
                "_op_type": "index",
                "_index": index_name,
                "_id": doc.get("id", os.path.basename(fpath)),
                "_source": {
                    "title": doc.get("title", ""),
                    "url": doc.get("url", ""),
                    "text": doc.get("text", ""),
                    "keywords": keywords,
                    "last_updated": doc.get("last_updated"),
                },
            })

        if not actions:
            continue

        print(f"[*] Sending batch {i // BATCH_SIZE + 1} ({len(actions)} docs)...", flush=True)
        try:
            success, errors = bulk(es, actions)
            total_indexed += success
            elapsed = round(time.time() - start, 2)
            print(f"[+] Batch done: {total_indexed} total docs indexed ({elapsed}s).", flush=True)
            if errors:
                print(f"[!] {len(errors)} errors in this batch:", flush=True)
                for err in errors[:5]:
                    print(f"  {err}", flush=True)
        except BulkIndexError as e:
            print(f"[!] Bulk indexing failed:", flush=True)
            for err in e.errors[:5]:
                print(json.dumps(err, indent=2, ensure_ascii=False), flush=True)
            break

    elapsed = round(time.time() - start, 2)
    print(f"\n[+] Done! {total_indexed} documents indexed in {elapsed}s.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Wikipedia JSON articles into Elasticsearch")
    parser.add_argument(
        "--input",
        default=os.path.join(os.path.dirname(__file__), "..", "..", "resources", "articles"),
        help="Directory containing article JSON files (default: resources/articles)",
    )
    parser.add_argument(
        "--config",
        default="config.yml",
        help="Path to config.yml (default: config.yml)",
    )
    args = parser.parse_args()
    run_ingestion(args.input, config_path=args.config)
