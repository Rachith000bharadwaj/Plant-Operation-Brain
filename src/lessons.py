# © 2026 Plant Operations Brain Team — ET AI Hackathon. All rights reserved.
# Proprietary & confidential. Unauthorised copying, reuse, or redistribution is prohibited (see LICENSE).
"""
lessons.py
----------
Phase 3 capability: Lessons Learned & Failure Intelligence Engine.

Mines incident reports, near-misses, RCAs and non-conformances across the whole
corpus to surface RECURRING, systemic patterns that no single investigation would
catch -- then turns them into proactive warnings ("similar conditions are
recurring on X"). This is the "learn from history before it repeats" capability.
"""

from __future__ import annotations
from dataclasses import dataclass

from .rag import PlantBrain, _format_excerpts
from .llm import chat

_SYSTEM = """You are a safety & reliability analyst mining an industrial plant's \
history for systemic lessons, using ONLY the provided document excerpts.

Output markdown with:
- **Recurring Patterns** — failure/incident themes that appear more than once or
  share a root cause; cite [Source N] for each.
- **Systemic Risks** — underlying issues these patterns point to (e.g. alignment
  after coupling work, outdated procedures still in circulation).
- **Proactive Warnings** — specific, actionable warnings to push to operations now.

Rules:
- Cite [Source N]. Use ONLY the excerpts; never fabricate incidents.
- If there isn't enough history to find a pattern, say so plainly.
"""


@dataclass
class LessonsReport:
    report_md: str
    sources: list


def mine_lessons(brain: PlantBrain, language: str = "English") -> LessonsReport:
    """Scan the corpus for incidents/near-misses and extract systemic lessons."""
    query = ("incident near-miss failure root cause non-conformance recurring "
             "pattern corrective action lesson learned")
    hits = brain.retrieve(query, k=8)

    # Pull in everything linked to known Incident nodes in the graph.
    if brain.graph is not None:
        seen = {c.chunk_id for c, _ in hits}
        for name, data in brain.graph.g.nodes(data=True):
            if data.get("type") == "Incident":
                for cid in brain.graph.entity_chunks.get(name, set()):
                    if cid not in seen and cid in brain._by_id:
                        hits.append((brain._by_id[cid], 0.0))
                        seen.add(cid)

    if not hits:
        return LessonsReport("No incident or failure history found in the corpus.", [])

    excerpts = _format_excerpts(hits)
    user_msg = f"Document excerpts:\n{excerpts}\n\nExtract the systemic lessons."
    from .i18n import answer_in
    report = chat(_SYSTEM + answer_in(language), user_msg, max_tokens=1000)
    return LessonsReport(report, [c for c, _ in hits])
