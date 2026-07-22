# © 2026 Plant Operations Brain Team — ET AI Hackathon. All rights reserved.
# Proprietary & confidential. Unauthorised copying, reuse, or redistribution is prohibited (see LICENSE).
"""
deep.py
-------
Advanced feature: Deep Analysis mode — visible agentic multi-step reasoning.

Normal RAG answers one question with one retrieval. Complex operational
questions ("Why does P-7 keep failing and what should we do about it before
the next statutory audit?") span maintenance + incidents + compliance at once.

Deep mode makes the copilot an agent:
  1. PLAN      — decompose the question into 2-3 focused sub-questions (1 call)
  2. INVESTIGATE — retrieve + answer each sub-question separately (n calls)
  3. SYNTHESISE — merge the findings into one final, cited answer (1 call)

The intermediate steps are returned so the UI can show the reasoning trace —
judges see HOW the agent thinks, not just what it concludes.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field

from .rag import PlantBrain, _format_excerpts
from .llm import chat

_PLAN = """Decompose the user's plant-operations question into 2-3 focused \
sub-questions that together fully answer it. Angles to consider: equipment \
history/failures, procedures/safety, compliance/regulatory, costs/planning.
Return STRICT JSON only: {"subquestions": ["...", "..."]}"""

_STEP = """You are investigating ONE sub-question using ONLY the excerpts. \
Answer in 2-4 sentences with [Source N] citations. If the excerpts don't cover \
it, say exactly that."""

_SYNTH = """You are the Plant Operations Brain. Merge the sub-answers into ONE \
final answer to the user's original question. Keep every [Source N] citation \
that supports a claim. Structure with short markdown headings if helpful. \
Be practical and concise — an engineer will act on this."""


@dataclass
class DeepAnswer:
    final_md: str
    steps: list = field(default_factory=list)   # [(subquestion, answer), ...]
    sources: list = field(default_factory=list)


def deep_ask(brain: PlantBrain, question: str,
             language: str = "English") -> DeepAnswer:
    """Plan -> investigate -> synthesise, with a visible trace."""
    # 1. PLAN
    raw = chat(_PLAN, question, max_tokens=200)
    try:
        s, e = raw.find("{"), raw.rfind("}")
        subs = json.loads(raw[s:e + 1]).get("subquestions", [])[:3]
    except Exception:
        subs = []
    if not subs:
        subs = [question]                      # degrade gracefully to plain RAG

    # 2. INVESTIGATE each sub-question with its own retrieval.
    steps, all_sources, seen = [], [], set()
    for sq in subs:
        hits = brain.retrieve(sq, k=4)
        ans = chat(_STEP,
                   f"Sub-question: {sq}\n\nExcerpts:\n{_format_excerpts(hits)}",
                   max_tokens=300)
        steps.append((sq, ans))
        for c, _ in hits:
            if c.chunk_id not in seen:
                all_sources.append(c)
                seen.add(c.chunk_id)

    # 3. SYNTHESISE (in the requested language, keeping tags/citations intact)
    lang_note = "" if language == "English" else (
        f" Write the final answer in {language} (keep equipment tags and "
        f"[Source N] citations unchanged).")
    trace = "\n\n".join(f"Sub-question: {sq}\nFinding: {a}" for sq, a in steps)
    final = chat(_SYNTH,
                 f"Original question: {question}\n\n{trace}\n\n"
                 f"Write the final merged answer.{lang_note}",
                 max_tokens=700)
    return DeepAnswer(final, steps, all_sources)
