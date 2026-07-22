# © 2026 Plant Operations Brain Team — ET AI Hackathon. All rights reserved.
# Proprietary & confidential. Unauthorised copying, reuse, or redistribution is prohibited (see LICENSE).
"""
conflict.py
-----------
Phase 4 differentiator: Conflict Detection across documents.

In real plants the SOP, the OEM manual and the latest inspection note often
DISAGREE about the same equipment (e.g. "close the suction valve" vs "close BOTH
valves"). Following the wrong one causes incidents. No mainstream RAG tool flags
this. We scan every entity that appears in 2+ documents and ask the LLM whether
those excerpts contradict each other.
"""

from __future__ import annotations
import json
from dataclasses import dataclass

from .rag import PlantBrain
from .llm import chat

_SYSTEM = """You compare excerpts that all refer to the SAME piece of equipment or \
procedure, taken from different plant documents. Decide if they CONTRADICT each \
other on any safety-relevant instruction, limit, or requirement.

Return STRICT JSON only:
{"conflict": true/false,
 "severity": "High"|"Medium"|"Low",
 "summary": "<one sentence describing the contradiction, or empty if none>"}

A contradiction = the documents give different/incompatible instructions or values
for the same thing. Differing-but-compatible detail is NOT a conflict.
"""


@dataclass
class Conflict:
    entity: str
    severity: str
    summary: str
    sources: list


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


def find_conflicts(brain: PlantBrain, max_entities: int = 15) -> list[Conflict]:
    """
    Scan entities that span multiple chunks and report contradictions.
    Only entities appearing in 2+ chunks can possibly conflict.
    """
    if brain.graph is None:
        return []

    conflicts: list[Conflict] = []
    # Prioritise the entity types where conflicts are dangerous.
    candidates = []
    for name, cids in brain.graph.entity_chunks.items():
        if len(cids) < 2:
            continue
        etype = brain.graph.g.nodes.get(name, {}).get("type", "")
        if etype in ("Equipment", "Procedure", "Regulation", "Parameter"):
            candidates.append((name, cids))

    for name, cids in candidates[:max_entities]:
        chunks = [brain._by_id[c] for c in cids if c in brain._by_id]
        if len(chunks) < 2:
            continue
        excerpts = "\n\n".join(
            f"[Doc: {c.source_file}] {c.text}" for c in chunks
        )
        result = _parse(chat(_SYSTEM, f"Entity: {name}\n\n{excerpts}", max_tokens=300))
        if result.get("conflict"):
            conflicts.append(Conflict(
                entity=name,
                severity=result.get("severity", "Medium"),
                summary=result.get("summary", ""),
                sources=chunks,
            ))
    return conflicts
