#!/usr/bin/env python3
"""
test_queries.py — Run all test queries against the API and produce a Markdown report.

Usage:
    python test_queries.py [--base-url http://localhost:10001]
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

OUTPUT_MD = "test_report.md"


async def run_one(client: httpx.AsyncClient, query: dict, base_url: str) -> dict:
    url = f"{base_url}/search"
    payload = {
        "formal_query": query["formal_query"],
        "keyword_string": query["keyword_string"],
    }
    try:
        resp = await client.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return {"query": query, "response": resp.json(), "error": None}
    except Exception as e:
        return {"query": query, "response": None, "error": str(e)}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:10001", help="API base URL")
    parser.add_argument("--input", default=__file__.rsplit("/", 1)[0] + "/test_queries.json", help="Test queries JSON")
    args = parser.parse_args()

    queries_path = Path(args.input)
    if not queries_path.exists():
        print(f"ERROR: {queries_path} not found", file=sys.stderr)
        sys.exit(1)

    queries = json.loads(queries_path.read_text(encoding="utf-8"))

    async with httpx.AsyncClient(timeout=None) as client:
        tasks = [run_one(client, q, args.base_url) for q in queries]
        results = await asyncio.gather(*tasks)

    # Build Markdown
    lines = [
        "# Wikipedia Search — Test Report",
        "",
        f"**Queries:** {len(queries)}  ",
        f"**Base URL:** {args.base_url}  ",
        "",
        "---",
        "",
    ]

    for i, r in enumerate(results, 1):
        q = r["query"]
        lines.append(f"## {i}. {q['formal_query']}")
        lines.append("")
        lines.append(f"**Keywords:** `{q['keyword_string']}`")
        lines.append("")

        if r["error"]:
            lines.append(f"> **ERROR:** {r['error']}")
            lines.append("")
            continue

        resp = r["response"]
        if resp is None:
            lines.append("> No response")
            lines.append("")
            continue

        # Combined context
        ctx = resp.get("combined_context", "")
        lines.append(ctx)
        lines.append("")

        # Per-result summary
        lines.append("| # | Title | URL | Sed Ranges |")
        lines.append("|---|-------|-----|------------|")
        for j, res in enumerate(resp.get("results", []), 1):
            title = res.get("title", "").replace("|", "\\|")
            url = res.get("url", "")
            sed = res.get("sed", [])
            lines.append(f"| {j} | {title} | {url} | {sed} |")
        lines.append("")
        lines.append("---")
        lines.append("")

    md = "\n".join(lines)
    out = Path(OUTPUT_MD)
    out.write_text(md, encoding="utf-8")
    print(f"[ok] Report written to {out}  ({len(queries)} queries)", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
