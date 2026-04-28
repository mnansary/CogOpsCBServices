#!/usr/bin/env python3
"""
Elasticsearch retriever for Wikipedia articles.
Config-driven search with boostable fields, fuzziness, and fast keyword matching.

Designed to be imported by an async FastAPI service.
"""
from dataclasses import dataclass

import yaml
from elasticsearch import Elasticsearch

# ---------------------------------------------------------------------------
# Global state ( initialised by init_es )
# ---------------------------------------------------------------------------
es: Elasticsearch | None = None
_cfg = {}


@dataclass
class SearchConfig:
    """All tunable search parameters — read from config.yml under `search`."""
    fields: list[str] = ["keywords^5", "title^3", "text^1"]
    fuzziness: str = "AUTO"
    top_k: int = 10
    min_gram: int = 2
    max_gram: int = 4


def init_es(config_path: str = "config.yml") -> Elasticsearch:
    """Initialise the ES client and parse search config. Call once at app startup."""
    global es, _cfg
    with open(config_path) as f:
        _cfg = yaml.safe_load(f)
    es_cfg = _cfg["es"]
    es = Elasticsearch(es_cfg["url"], request_timeout=es_cfg.get("timeout", 60))
    return es


def _parse_search_cfg() -> SearchConfig:
    raw = _cfg.get("search", {})
    return SearchConfig(
        fields=raw.get("fields", ["keywords^5", "title^3", "text^1"]),
        fuzziness=raw.get("fuzziness", "AUTO"),
        top_k=raw.get("top_k", 10),
    )


def _build_body(query: str) -> dict:
    """Build a multi_match search body — no vector needed, pure text search."""
    sc = _parse_search_cfg()
    return {
        "query": {
            "multi_match": {
                "query": query,
                "fields": sc.fields,
                "fuzziness": sc.fuzziness,
                "prefix_length": 2,
                "max_expansions": 50,
                "minimum_should_match": "2<75%",
            }
        },
        "size": sc.top_k,
        "_source": ["title", "url", "text", "keywords"],
    }


async def search(query: str, top_k: int | None = None) -> list[dict]:
    """
    Async search wrapper for FastAPI.
    Returns a list of hit dicts with title, url, text (truncated), keywords.
    """
    if es is None:
        raise RuntimeError("Elasticsearch not initialised — call init_es() first")

    if top_k is not None:
        body = _build_body(query)
        body["size"] = top_k
    else:
        body = _build_body(query)

    resp = await es.search_async(index=_cfg["es"]["index"], body=body)

    results = []
    for hit in resp["hits"]["hits"]:
        src = hit["_source"]
        results.append({
            "title": src.get("title", ""),
            "url": src.get("url", ""),
            "text": (src.get("text", "") or ""),
            "keywords": src.get("keywords", []),
        })
    return results
