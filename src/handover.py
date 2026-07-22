# © 2026 Plant Operations Brain Team — ET AI Hackathon. All rights reserved.
# Proprietary & confidential. Unauthorised copying, reuse, or redistribution is prohibited (see LICENSE).
"""
handover.py
-----------
Advanced feature: the auto-generated Shift Handover Brief.

Shift changeover is where plants lose information -- the outgoing crew knows
things the incoming crew doesn't, and handover registers are patchy. (The
problem statement's FICCI stat: 60%+ of facilities rely on manual handoffs.)

This module generates a one-page brief for the incoming shift by fusing:
  - live sensor states (from the Plant Watch simulation),
  - the documented knowledge base (open actions, pending submissions,
    known equipment risks, recent incidents/conflicts).

One LLM call -> one readable page a supervisor can skim in 60 seconds.
"""

from __future__ import annotations
from dataclasses import dataclass

from .rag import PlantBrain, _format_excerpts
from .llm import chat
from .sensors import get_registry, status_at

_SYSTEM = """You write the shift-handover brief for an industrial plant's \
incoming shift supervisor, using ONLY the provided live readings and document \
excerpts.

Format (markdown, max ~250 words):
## Shift Handover Brief
**⚠ Watch closely** — live readings trending toward limits + WHY that equipment
matters (documented history), with [Source N] citations.
**📌 Open actions** — pending items found in the documents (submissions awaited,
overdue checks, unresolved corrective actions), each with its owner if named.
**📚 Recent lessons** — 1-2 documented incidents/near-misses the crew must keep
in mind for today's work.

Rules: cite [Source N]; only documented facts; short bullet points, no fluff."""


@dataclass
class Handover:
    brief_md: str
    sources: list


def generate_brief(brain: PlantBrain, hours: int = 48,
                   language: str = "English") -> Handover:
    """Build the incoming-shift brief for the given simulation time."""
    # 1. Live sensor summary (free -- pure python). Includes every sensor the
    #    registry auto-discovered from the documents, not just the curated set.
    registry = get_registry(brain.store.chunks)
    lines = []
    for name in registry:
        s = status_at(name, hours, registry)
        lines.append(f"- {s.name}: {s.value} (alarm {s.alarm}, trip {s.trip}) "
                     f"-> {s.state}")
    live = "\n".join(lines)

    # 2. Documented open items + risks (one broad retrieval).
    hits = brain.retrieve(
        "pending action awaited report overdue corrective action near-miss "
        "incident recurring failure inspection due renewal deadline", k=8)
    excerpts = _format_excerpts(hits)

    from .i18n import answer_in
    brief = chat(_SYSTEM + answer_in(language),
                 f"LIVE SENSOR READINGS (now):\n{live}\n\n"
                 f"Document excerpts:\n{excerpts}\n\n"
                 f"Write the handover brief.",
                 max_tokens=600)
    return Handover(brief, [c for c, _ in hits])
