"""
cogops/llm/context_cutter.py

SED-like Context Extractor Agent.
Takes an original Bengali query and a line-numbered document.
Outputs exact line ranges [(start, end)] to extract, minimizing the context
passed to the final generation stage.
"""

import logging
import ast
import re
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an advanced Context Extraction Agent (SED-like).
Your task is to identify the exact line ranges in a numbered document that are relevant to answering the user's query.
BE VERY VERY STRICT WHILE SELECTING LINES — ONLY RETURN WHAT IS ESSENTIAL TO ANSWER THE QUERY, NOTHING MORE.

CRITICAL: This is a Bangladesh-specific context. Only consider the Bangladesh context.

RULES:
1. Return ONLY a Python list of tuples containing line numbers: [(start_line, end_line)].
2. Include immediately surrounding lines if they provide necessary context (e.g., table headers, step introductions).
3. If multiple separate sections are relevant, return multiple tuples: [(2, 5), (10, 15)].
4. If NO information is relevant to the query, return: [(0, 0)]
5. If the ENTIRE document is highly relevant and cannot be cut, return: [(1, -1)]
6. DO NOT output any text, explanations, or markdown blocks. ONLY the list.

EXAMPLES:
Query: "পাসপোর্ট করতে কি কি কাগজ লাগে"
Document:
1. পাসপোর্ট করার নিয়ম:
2. প্রয়োজনীয় কাগজপত্র:
3. ১. জাতীয় পরিচয়পত্র
4. ২. নাগরিকত্ব সনদ
5. ফি জমা দেওয়ার নিয়ম:
Output: [(2, 4)]

Query: "মেট্রোরেলের ভাড়া কত"
Document:
1. ট্রেড লাইসেন্স ফি:
2. ১. সাধারণ ব্যবসার জন্য ২০০০ টাকা
Output: [(0, 0)]
"""

USER_PROMPT = """\
CURRENT TIME: {bd_time} — ensure the selected information is still relevant now.

Query: "{query}"

Document:
{numbered_passage}

Output:\
"""


class ContextCutterAgent:
    """LLM agent that determines which line ranges to slice from a document."""

    def __init__(self, client: AsyncOpenAI, model: str):
        self.client = client
        self.model = model

    async def get_cut_ranges(self, query: str, numbered_passage: str, bd_time: str = "") -> list[tuple[int, int]]:
        """
        Asks the LLM for line ranges.
        Returns a list of tuples, e.g., [(3, 7), (12, 15)]
        """
        try:
            user = USER_PROMPT.format(query=query, numbered_passage=numbered_passage, bd_time=bd_time)
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_tokens=30, # Keep very small, we only want a short list of ints
            )

            content = resp.choices[0].message.content.strip()
            return self._parse_tuples(content)

        except Exception as e:
            logger.error(f"Context Cutter LLM error: {e}")
            # Fallback: if LLM fails, return the whole document to avoid data loss
            return [(1, -1)]

    def _parse_tuples(self, content: str) -> list[tuple[int, int]]:
        """Safely parses the LLM output into an actual Python list of tuples."""
        try:
            # Extract everything between the first [ and last ]
            match = re.search(r"\[.*\]", content, re.DOTALL)
            if match:
                list_str = match.group(0)
                # ast.literal_eval safely evaluates strings containing Python literals
                parsed = ast.literal_eval(list_str)
                if isinstance(parsed, list) and all(isinstance(i, tuple) and len(i) == 2 for i in parsed):
                    return parsed

            logger.warning(f"Failed to parse context cutter output cleanly: {content}")
            return [(1, -1)]
        except Exception as e:
            logger.error(f"AST Parsing error on string '{content}': {e}")
            return [(1, -1)]

    @staticmethod
    def format_numbered_lines(text: str) -> str:
        """Helper: Converts raw text into line-numbered text for the LLM."""
        lines = text.strip().split('\n')
        return '\n'.join([f"{i+1}. {line}" for i, line in enumerate(lines)])

    @staticmethod
    def apply_cut(raw_text: str, ranges: list[tuple[int, int]]) -> str:
        """
        Helper: Acts as 'SED', applying the returned ranges to the original text.
        Takes raw text, splits it, slices the requested chunks, and joins them back.
        """
        if not ranges or ranges == [(0, 0)]:
            return ""

        lines = raw_text.strip().split('\n')
        total_lines = len(lines)

        if ranges == [(1, -1)]:
            return raw_text

        extracted_lines = []
        for start, end in ranges:
            # Handle out of bounds safely
            idx_start = max(0, start - 1)
            idx_end = total_lines if end == -1 else min(total_lines, end)

            extracted_lines.extend(lines[idx_start:idx_end])
            extracted_lines.append("...") # Visual separator for disjointed cuts

        # Clean up the trailing "..."
        if extracted_lines and extracted_lines[-1] == "...":
            extracted_lines.pop()

        return '\n'.join(extracted_lines)
