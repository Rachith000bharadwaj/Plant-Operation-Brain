# © 2026 Plant Operations Brain Team — ET AI Hackathon. All rights reserved.
# Proprietary & confidential. Unauthorised copying, reuse, or redistribution is prohibited (see LICENSE).
"""
whatif.py
---------
The What-If Safety Checker: a technician describes the work they are ABOUT to do,
and the system cross-checks it against every procedure, incident report, permit
rule and operating limit in the knowledge base -- BEFORE the work starts.

Verdict levels:
  STOP    - documented evidence this exact situation caused/nearly caused harm
  CAUTION - proceed only after specific listed precautions
  SAFE    - no documented objection found (still cites what it checked)

This is proactive safety: the Visakhapatnam pattern ("data present, but unacted
upon") inverted -- the data acts BEFORE the job begins.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field

from .rag import PlantBrain, _format_excerpts
from .llm import chat

_SYSTEM = """You are a plant safety officer reviewing PLANNED work before it \
starts, using ONLY the provided document excerpts (procedures, incident reports, \
permits, operating limits).

Return STRICT JSON only:
{
  "verdict": "STOP" | "CAUTION" | "SAFE",
  "headline": "<one sentence: the single most important thing>",
  "reasons": ["<reason with [Source N] citation>", ...],
  "required_actions": ["<specific action before/during the work>", ...]
}

Rules:
- STOP if an incident/near-miss/violation happened under materially similar
  conditions, or the plan violates a documented procedure or operating limit.
- CAUTION if the work is allowed but documented precautions apply.
- SAFE only if nothing in the excerpts argues against the plan.
- Cite [Source N] in every reason. Never invent facts not in the excerpts.
"""


@dataclass
class SafetyVerdict:
    verdict: str            # STOP / CAUTION / SAFE
    headline: str
    reasons: list = field(default_factory=list)
    required_actions: list = field(default_factory=list)
    sources: list = field(default_factory=list)


def _parse(raw: str) -> dict:
    raw = raw.strip()
    if "```" in raw:
        raw = raw.split("```")[1].lstrip("json").strip()
    s, e = raw.find("{"), raw.rfind("}")
    if s == -1:
        return {}
    try:
        return json.loads(raw[s:e + 1])
    except json.JSONDecodeError:
        return {}


def check_plan(brain: PlantBrain, plan: str,
               language: str = "English") -> SafetyVerdict:
    """Cross-check a described work plan against the whole knowledge base."""
    # Retrieve everything relevant: the plan itself + safety-slanted expansions.
    hits = brain.retrieve(plan, k=6)
    extra = brain.retrieve(f"incident near-miss permit procedure limits {plan}", k=4)
    seen = {c.chunk_id for c, _ in hits}
    for c, s in extra:
        if c.chunk_id not in seen:
            hits.append((c, s))
            seen.add(c.chunk_id)

    if not hits:
        return SafetyVerdict(
            "CAUTION", "No relevant documentation found for this work.",
            ["The knowledge base contains nothing about this activity — "
             "proceed only with a manual safety review."], [], [])

    excerpts = _format_excerpts(hits)
    lang_note = "" if language == "English" else (
        f" Write the 'headline', 'reasons' and 'required_actions' text in "
        f"{language}, but keep the 'verdict' value as one of STOP/CAUTION/SAFE "
        f"in English and keep equipment tags/IDs unchanged.")
    user = (f"PLANNED WORK: {plan}\n\n"
            f"Document excerpts:\n{excerpts}\n\n"
            f"Review the planned work and return the JSON verdict.{lang_note}")
    data = _parse(chat(_SYSTEM, user, max_tokens=800))

    return SafetyVerdict(
        verdict=data.get("verdict", "CAUTION"),
        headline=data.get("headline", "Review the cited documents before starting."),
        reasons=data.get("reasons", []),
        required_actions=data.get("required_actions", []),
        sources=[c for c, _ in hits],
    )
