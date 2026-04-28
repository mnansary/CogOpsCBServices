#!/usr/bin/env python3
"""
search/retriver.py — Elasticsearch client + query builder.

Single source of truth for ES connection, config loading, and search.
No LLM, no cutter, no pipeline logic.
"""
from __future__ import annotations

from typing import Any

import asyncio

import yaml
from elasticsearch import Elasticsearch

es: Elasticsearch | None = None
_cfg: dict = {}


def init(config_path: str = "config.yml") -> Elasticsearch:
    """Initialise the ES client and load config. Call once at startup."""
    global es, _cfg
    with open(config_path) as f:
        _cfg = yaml.safe_load(f)
    es_cfg = _cfg["es"]
    es = Elasticsearch(es_cfg["url"], request_timeout=es_cfg.get("timeout", 60))
    return es


def _cfg_section(section: str) -> dict:
    return _cfg.get(section, {})


def build_search_body(query: str, top_k: int | None = None) -> dict[str, Any]:
    """Build a multi_match ES query body from config."""
    search_cfg = _cfg_section("search")
    fields = search_cfg.get("fields", ["keywords^5", "title^3", "text^1"])
    fuzziness = search_cfg.get("fuzziness", "AUTO")
    size = top_k if top_k is not None else search_cfg.get("top_k", 10)

    return {
        "query": {
            "multi_match": {
                "query": query,
                "fields": fields,
                "fuzziness": fuzziness,
                "prefix_length": 2,
                "max_expansions": 50,
                "minimum_should_match": "2<75%",
            }
        },
        "size": size,
        "_source": ["title", "url", "text", "keywords", "last_updated"],
    }


async def search(query: str, index: str | None = None, top_k: int | None = None) -> list[dict]:
    """
    Async ES search. Returns hit dicts with title, url, text, keywords, last_updated.
    """
    if es is None:
        raise RuntimeError("Elasticsearch not initialised — call init() first")

    idx = index or _cfg["es"]["index"]
    body = build_search_body(query, top_k)

    loop = asyncio.get_running_loop()
    resp = await loop.run_in_executor(None, lambda: es.search(index=idx, body=body))

    results = []
    for hit in resp["hits"]["hits"]:
        src = hit["_source"]
        results.append({
            "title": src.get("title", ""),
            "url": src.get("url", ""),
            "text": src.get("text") or "",
            "keywords": src.get("keywords", []),
            "last_updated": src.get("last_updated"),
        })
    return results
