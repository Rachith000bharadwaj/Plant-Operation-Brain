# © 2026 Plant Operations Brain Team — ET AI Hackathon. All rights reserved.
# Proprietary & confidential. Unauthorised copying, reuse, or redistribution is prohibited (see LICENSE).
"""
interview.py
------------
Phase 4 differentiator: Tribal Knowledge Extractor + Knowledge Risk Map.

THE novel idea: every competitor digitises documents that already exist. But ~25%
of India's senior plant engineers retire within a decade, taking UNDOCUMENTED
knowledge with them. This module:
  1. Interviews a senior engineer about an asset (AI asks smart follow-ups).
  2. Converts the conversation into a structured knowledge document that drops
     into the corpus (so it becomes searchable + linked into the graph).
  3. Shows a "Knowledge Risk Map": which assets have almost no documentation and
     should be captured BEFORE the expert leaves.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from datetime import date

from .rag import PlantBrain
from .llm import chat

_INTERVIEWER = """You are interviewing a soon-to-retire senior plant engineer to \
capture their hard-won, undocumented knowledge about a specific asset before it is \
lost. Ask ONE focused, practical follow-up question at a time -- the kind only \
decades of experience would know (warning signs, quirks, 'tricks', what manuals \
miss, past near-misses). Keep it short and conversational. Do not repeat questions \
already asked. Return ONLY the next question."""

_SUMMARISER = """Convert this interview with a senior engineer into a clean, \
structured knowledge document in markdown. Use clear sections (Overview, Warning \
Signs, Operating Tips, Known Quirks, Past Incidents, Maintenance Advice). Preserve \
every concrete detail (numbers, symptoms, part names). Do not invent anything."""


@dataclass
class RiskItem:
    equipment: str
    doc_chunks: int
    risk: str          # High / Medium / Low


def next_question(equipment: str, transcript: list[tuple[str, str]]) -> str:
    """Generate the interviewer's next question given the conversation so far."""
    convo = "\n".join(f"Q: {q}\nA: {a}" for q, a in transcript) or "(none yet)"
    user = (f"Asset being documented: {equipment}\n\n"
            f"Conversation so far:\n{convo}\n\nAsk the next best question.")
    return chat(_INTERVIEWER, user, max_tokens=120)


def save_knowledge(equipment: str, transcript: list[tuple[str, str]],
                   docs_dir: str = "data/docs") -> str:
    """
    Turn the interview into a structured doc saved in data/docs so the next index
    rebuild absorbs it into the searchable corpus + knowledge graph.
    Returns the path written.
    """
    convo = "\n".join(f"Q: {q}\nA: {a}" for q, a in transcript)
    body = chat(_SUMMARISER, f"Asset: {equipment}\n\nInterview:\n{convo}",
                max_tokens=1200)
    header = (f"# Tribal Knowledge — {equipment}\n"
              f"Captured: {date.today().isoformat()} (expert interview)\n\n")
    safe = equipment.replace("/", "-").replace(" ", "_")
    out = Path(docs_dir) / f"tribal_{safe}.md"
    out.write_text(header + body, encoding="utf-8")
    return str(out)


def knowledge_risk_map(brain: PlantBrain) -> list[RiskItem]:
    """
    For every Equipment entity, measure how RICHLY it is documented — by how
    much CONTEXT surrounds it in the knowledge graph: distinct chunks *plus*
    distinct types of connected knowledge (procedures, incidents, regulations,
    limits, people).

    Why context, not raw chunk count: a bare mention in one chunk is thin
    knowledge; an asset linked to a procedure + an incident + a limit is well
    understood even if it appears in few chunks. Crucially this means **adding
    documents can only lower risk, never raise it** — the metric now rewards
    capturing knowledge instead of punishing a growing graph.
    """
    if brain.graph is None:
        return []
    g = brain.graph.g
    und = g.to_undirected(as_view=True)
    items: list[RiskItem] = []
    for name, data in g.nodes(data=True):
        if data.get("type") != "Equipment":
            continue
        n = len(brain.graph.entity_chunks.get(name, set()))
        # distinct types of connected knowledge (excludes other equipment)
        ctx_types = {g.nodes[nb].get("type") for nb in und.neighbors(name)}
        ctx_types.discard("Equipment")
        ctx_types.discard(None)
        richness = n + 2 * len(ctx_types)     # context counts double
        risk = "Low" if richness >= 5 else ("Medium" if richness >= 2 else "High")
        items.append(RiskItem(name, n, risk))
    order = {"High": 0, "Medium": 1, "Low": 2}
    items.sort(key=lambda x: (order[x.risk], -x.doc_chunks))
    return items
