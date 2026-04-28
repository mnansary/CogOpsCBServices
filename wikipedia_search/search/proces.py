#!/usr/bin/env python3
"""
search/proces.py — Core processing pipeline.

Responsibilities:
  - ES retrieval via retriver (delegated)
  - Concurrent LLM context cutting via cutter
  - Result assembly with Bangladesh time awareness

Everything non-pipeline (ES connection, query building) lives in retriver.py.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import yaml
from openai import AsyncOpenAI

import re

from .cutter import ContextCutterAgent
from .retriver import init as init_es
from .retriver import search as es_search


def _clean_wikitext(text: str) -> str:
    """Remove <ref> tags and {{/{| ... }}/|} blocks (templates, infoboxes, tables)."""
    # Remove <ref>...</ref>
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    # Remove {{ ... }} templates (depth-aware)
    text = re.sub(r'\{\{.*?\}\}', '', text, flags=re.DOTALL)
    # Remove {| ... |} tables (depth-aware)
    text = re.sub(r'\{\|.*?\|}', '', text, flags=re.DOTALL)
    return text

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
_cfg: dict = {}
_llm_client: AsyncOpenAI | None = None
_cutter: ContextCutterAgent | None = None
_cutter_concurrency: int = 8
_max_context_chars: int = 4000


def _load_config(path: str = "config.yml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _bd_formatted() -> str:
    bd = datetime.now(timezone.utc) + timedelta(hours=6)
    return bd.strftime("%Y-%m-%d %A %H:%M:%S")


# ---------------------------------------------------------------------------
# Init / shutdown
# ---------------------------------------------------------------------------
async def init_service(config_path: str = "config.yml"):
    global _cfg, _llm_client, _cutter, _cutter_concurrency, _max_context_chars

    _cfg = _load_config(config_path)
    init_es(config_path)

    llm_cfg = _cfg["llm"]
    _llm_client = AsyncOpenAI(base_url=llm_cfg["base_url"], api_key=llm_cfg["api_key"])
    _cutter = ContextCutterAgent(_llm_client, llm_cfg["model"])

    _cutter_concurrency = _cfg["cutter"].get("concurrency", 8)
    _max_context_chars = _cfg["cutter"].get("max_context_chars", 4000)


async def shutdown_service():
    global _cutter, _llm_client
    _cutter = None
    _llm_client = None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
@dataclass
class ProcessInput:
    formal_query: str
    keyword_string: str


async def process(input_data: ProcessInput) -> tuple[list[dict], str]:
    """
    Full pipeline: retrieve → cut → assemble.

    Returns (results, combined_context).
    """
    # 1. ES retrieval
    passages = await es_search(input_data.keyword_string)

    # 2. Concurrent cutting
    cut_tasks = [_cut_passage(input_data.formal_query, p) for p in passages]
    cut_results = await _gather_concurrent(cut_tasks)

    # 3. Assemble results
    results: list[dict] = []
    contexts: list[str] = []

    for passage, ranges, condensed in cut_results:
        if ranges == [(0, 0)]:
            continue  # LLM found nothing relevant — skip entirely

        sed = ranges
        ctx = condensed or passage["text"][:_max_context_chars]

        results.append({
            "title": passage["title"],
            "url": passage["url"],
            "context": ctx,
            "sed": sed,
            "published_at": passage.get("last_updated"),
        })
        contexts.append(f"### {passage['title']}\n\n{ctx}")

    combined_context = "\n\n".join(contexts)

    # Truncate to ~1000 words if over
    _MAX_COMBINED_WORDS = 1000
    words = combined_context.split()
    if len(words) > _MAX_COMBINED_WORDS:
        combined_context = " ".join(words[:_MAX_COMBINED_WORDS]) + "..."

    return results, combined_context


# ---------------------------------------------------------------------------
# Cutter helpers
# ---------------------------------------------------------------------------
async def _cut_passage(query: str, passage: dict) -> tuple:
    raw_text = passage["text"]
    clean_text = _clean_wikitext(raw_text)
    numbered = ContextCutterAgent.format_numbered_lines(clean_text)

    bd_time = _bd_formatted()
    ranges = await _cutter.get_cut_ranges(query, numbered, bd_time=bd_time)
    condensed = ContextCutterAgent.apply_cut(clean_text, ranges)
    return (passage, ranges, condensed)


async def _gather_concurrent(cut_tasks):
    semaphore = asyncio.Semaphore(_cutter_concurrency)

    async def _limited(task):
        async with semaphore:
            return await task
    return await asyncio.gather(*[_limited(t) for t in cut_tasks])
