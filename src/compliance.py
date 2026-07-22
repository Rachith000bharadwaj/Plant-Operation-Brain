# © 2026 Plant Operations Brain Team — ET AI Hackathon. All rights reserved.
# Proprietary & confidential. Unauthorised copying, reuse, or redistribution is prohibited (see LICENSE).
"""
compliance.py
-------------
Phase 3 capability: Quality & Regulatory Compliance Intelligence.

Maps regulatory references found in the corpus (Factory Act, OISD, PESO, etc.)
against the procedures and equipment they govern, then asks Claude to find
compliance GAPS -- regulations with no supporting procedure, procedures that
look outdated, or missing evidence -- and emits an audit-ready evidence package.

This is the QMS-integration angle: catch deviations before the auditor does.
"""

from __future__ import annotations
from dataclasses import dataclass

from .rag import PlantBrain, _format_excerpts
from .llm import chat

_SYSTEM = """You are a compliance auditor for an Indian industrial plant. Using \
ONLY the provided document excerpts, assess regulatory compliance.

Output markdown with:
- **Regulations Referenced** — list each regulation found (Factory Act, OISD, PESO, etc.).
- **Compliance Gaps** — for each, state the gap, severity (High/Medium/Low), and the
  evidence or its absence, citing [Source N].
- **Corrective Actions** — a concrete, prioritised action per gap.
- **Audit Evidence Summary** — a short statement an auditor could file.

Rules:
- Cite [Source N]. Use ONLY the excerpts. If evidence is missing, that itself is a
  gap worth flagging — do not invent compliance that isn't documented.
"""


@dataclass
class ComplianceReport:
    report_md: str
    sources: list


def audit(brain: PlantBrain, focus: str = "",
          language: str = "English") -> ComplianceReport:
    """
    Run a compliance audit. `focus` optionally narrows it (e.g. 'lockout', 'P-7').
    Pulls regulation-related context from the graph + vector store.
    """
    query = (
        f"regulatory compliance Factory Act OISD PESO safety procedure inspection "
        f"requirement {focus}".strip()
    )
    hits = brain.retrieve(query, k=8)

    # Also pull in everything linked to known Regulation nodes in the graph.
    if brain.graph is not None:
        seen = {c.chunk_id for c, _ in hits}
        for name, data in brain.graph.g.nodes(data=True):
            if data.get("type") == "Regulation":
                for cid in brain.graph.entity_chunks.get(name, set()):
                    if cid not in seen and cid in brain._by_id:
                        hits.append((brain._by_id[cid], 0.0))
                        seen.add(cid)

    if not hits:
        return ComplianceReport("No compliance-relevant documents found.", [])

    excerpts = _format_excerpts(hits)
    user_msg = f"Document excerpts:\n{excerpts}\n\nProduce the compliance audit."
    from .i18n import answer_in
    report = chat(_SYSTEM + answer_in(language), user_msg, max_tokens=1100)
    return ComplianceReport(report, [c for c, _ in hits])
