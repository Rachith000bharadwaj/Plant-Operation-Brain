# © 2026 Plant Operations Brain Team — ET AI Hackathon. All rights reserved.
# Proprietary & confidential. Unauthorised copying, reuse, or redistribution is prohibited (see LICENSE).
"""
impact.py
---------
Advanced feature: equipment failure Impact Analysis, powered by the knowledge
graph. This is where the graph stops being a picture and starts being a tool:
pick an asset, and the graph traversal answers "what is CONNECTED to this —
which procedures, permits, regulations, incidents and people are touched if it
fails?" Then one LLM call turns the traversal into a readable impact summary.
"""

from __future__ import annotations
from dataclasses import dataclass, field

import networkx as nx

from .rag import PlantBrain, _format_excerpts
from .llm import chat

_SYSTEM = """You are a reliability planner. Using ONLY the excerpts and the \
listed graph connections, write a SHORT failure-impact summary (max 180 words) \
for the named equipment:
- **Immediate impact** — what stops working / what risk arises.
- **Connected obligations** — procedures, permits, regulations touched.
- **History says** — documented failures/incidents that make this credible.
Cite [Source N]. Only documented facts."""


@dataclass
class Impact:
    equipment: str
    neighbours: dict = field(default_factory=dict)   # type -> [names]
    summary_md: str = ""
    sources: list = field(default_factory=list)


def analyze_impact(brain: PlantBrain, equipment: str,
                   with_summary: bool = True, language: str = "English") -> Impact:
    """Traverse the graph around one asset and summarise the blast radius."""
    g = brain.graph.g
    if equipment not in g:
        return Impact(equipment, {}, f"'{equipment}' is not in the knowledge graph.")

    # Collect 2-hop neighbourhood, grouped by entity type.
    near = set(nx.single_source_shortest_path_length(g.to_undirected(as_view=False),
                                                     equipment, cutoff=2))
    near.discard(equipment)
    grouped: dict[str, list[str]] = {}
    for n in sorted(near):
        t = g.nodes[n].get("type", "Other")
        grouped.setdefault(t, []).append(n)

    imp = Impact(equipment, grouped)
    if not with_summary:
        return imp

    # One LLM call: traversal + document context -> readable summary.
    hits = brain.retrieve(f"{equipment} failure impact procedure regulation "
                          f"incident downstream", k=6)
    conn = "\n".join(f"- {t}: {', '.join(names[:8])}"
                     for t, names in grouped.items())
    from .i18n import answer_in
    imp.summary_md = chat(
        _SYSTEM + answer_in(language),
        f"Equipment: {equipment}\n\nGraph connections:\n{conn}\n\n"
        f"Document excerpts:\n{_format_excerpts(hits)}\n\n"
        f"Write the impact summary.",
        max_tokens=450)
    imp.sources = [c for c, _ in hits]
    return imp


def subgraph_html(brain: PlantBrain, equipment: str,
                  out_path: str = "data/index/impact.html") -> str | None:
    """Render just the neighbourhood of one asset as interactive pyvis HTML."""
    from pathlib import Path
    from pyvis.network import Network

    g = brain.graph.g
    if equipment not in g:
        return None
    near = set(nx.single_source_shortest_path_length(g.to_undirected(as_view=False),
                                                     equipment, cutoff=2))
    sub = g.subgraph(near)

    colors = {"Equipment": "#2563eb", "Procedure": "#16a34a",
              "Regulation": "#9333ea", "Incident": "#dc2626",
              "Parameter": "#ea580c", "Person": "#0891b2"}
    net = Network(height="480px", width="100%", directed=True, bgcolor="#ffffff")
    for node, data in sub.nodes(data=True):
        t = data.get("type", "Other")
        size = 28 if node == equipment else 14
        net.add_node(node, label=node, size=size,
                     color="#f59e0b" if node == equipment
                     else colors.get(t, "#64748b"),
                     title=f"{t}: {node}")
    for s, t2, data in sub.edges(data=True):
        net.add_edge(s, t2, title=data.get("label", ""))
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(p))
    return p.read_text(encoding="utf-8")
