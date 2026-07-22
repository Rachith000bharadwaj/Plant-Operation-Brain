# © 2026 Plant Operations Brain Team — ET AI Hackathon. All rights reserved.
# Proprietary & confidential. Unauthorised copying, reuse, or redistribution is prohibited (see LICENSE).
"""
maintenance.py
--------------
Phase 3 capability: Maintenance Intelligence & Root-Cause-Analysis (RCA) Agent.

Given an equipment tag, it gathers everything the corpus knows about that asset
(work orders, failures, OEM limits, inspection findings -- pulled via hybrid
GraphRAG retrieval) and asks Claude to produce a structured RCA + predictive
maintenance recommendation. This is the "connect the dots no single person can"
capability the problem statement calls for.
"""

from __future__ import annotations
from dataclasses import dataclass

from .rag import PlantBrain, _format_excerpts
from .llm import chat

_SYSTEM = """You are a reliability engineer performing Root Cause Analysis on \
industrial equipment, using ONLY the provided document excerpts.

Produce a concise, structured report in markdown with these sections:
- **Failure History** — what has failed, when, and the recurrence pattern.
- **Likely Root Cause** — your best-supported hypothesis, citing [Source N].
- **Predictive Maintenance Recommendation** — specific next actions + timing.
- **Operating Limits to Watch** — relevant alarm/trip thresholds if present.

Rules:
- Cite sources inline as [Source N]. Use ONLY the excerpts; never invent data.
- If the excerpts lack enough information, say so explicitly under each section.
"""


@dataclass
class RCAReport:
    equipment: str
    report_md: str
    sources: list


def analyze_equipment(brain: PlantBrain, equipment_tag: str,
                      language: str = "English") -> RCAReport:
    """Run RCA for a single equipment tag (e.g. 'P-7')."""
    query = (
        f"Maintenance history, failures, root cause, inspection findings and "
        f"operating limits for equipment {equipment_tag}"
    )
    hits = brain.retrieve(query, k=6)
    if not hits:
        return RCAReport(equipment_tag,
                         f"No documents reference {equipment_tag}.", [])

    excerpts = _format_excerpts(hits)
    user_msg = (
        f"Equipment under analysis: {equipment_tag}\n\n"
        f"Document excerpts:\n{excerpts}\n\n"
        f"Produce the RCA report."
    )
    from .i18n import answer_in
    report = chat(_SYSTEM + answer_in(language), user_msg, max_tokens=900)
    return RCAReport(equipment_tag, report, [c for c, _ in hits])
