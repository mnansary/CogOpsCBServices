#!/usr/bin/env python3
"""
wikipedia_search/api.py — FastAPI entry point for the Wikipedia search service.

Endpoints:
  POST /search  — main search endpoint
  GET  /health  — liveness probe

Service: 0.0.0.0:10001
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Resolve package imports — this file lives in wikipedia_search/ at repo root
_pkg_root = Path(__file__).resolve().parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from search.proces import init_service, shutdown_service, ProcessInput, process

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_cfg = {}
import yaml
with open(_pkg_root / "config.yml") as _f:
    _cfg = yaml.safe_load(_f)

api_cfg = _cfg.get("api", {})
HOST = api_cfg.get("host", "0.0.0.0")
PORT = api_cfg.get("port", 10001)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Wikipedia Search (Elastic)", version="0.1.0")

# Pydantic model for the request
class SearchRequest(BaseModel):
    formal_query: str = Field(..., description="Formal Bengali query in natural language")
    keyword_string: str = Field(..., description="Expanded keywords with intent (pipe or space separated)")


class HealthResponse(BaseModel):
    status: str


class SearchResponse(BaseModel):
    results: list[dict]
    combined_context: str


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    config_path = str(_pkg_root / "config.yml")
    await init_service(config_path)
    print(f"[api] initialised — listening on {HOST}:{PORT}", flush=True)


@app.on_event("shutdown")
async def shutdown():
    await shutdown_service()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health():
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    if not req.formal_query.strip() or not req.keyword_string.strip():
        raise HTTPException(status_code=400, detail="formal_query and keyword_string are required")

    try:
        input_data = ProcessInput(formal_query=req.formal_query, keyword_string=req.keyword_string)
        results, combined_context = await process(input_data)
        return {"results": results, "combined_context": combined_context}
    except Exception as e:
        logging.exception("Search error")
        raise HTTPException(status_code=500, detail=str(e))
