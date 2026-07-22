# © 2026 Plant Operations Brain Team — ET AI Hackathon. All rights reserved.
# Proprietary & confidential. Unauthorised copying, reuse, or redistribution is prohibited (see LICENSE).
"""
graph.py
--------
Phase 2: the "brain" -- a knowledge graph over the document corpus.

Plain RAG finds text that *looks* similar to a question. A knowledge graph
understands that "P-7", "the cooling water pump", an inspection note, and a
Factory Act clause are all *connected* -- so we can answer multi-hop questions
that vector search alone misses (e.g. "which regulations apply to the equipment
that keeps failing?").

Pipeline:
  chunks --(Claude structured extraction)--> entities + relations
         --> networkx graph (cached to disk)
         --> graph-neighbour expansion for hybrid retrieval
         --> interactive pyvis visualization for the demo
"""

from __future__ import annotations
import json
import hashlib
from pathlib import Path
from collections import defaultdict

import networkx as nx

from .llm import chat

# The industrial ontology: the entity types our graph understands.
ENTITY_TYPES = [
    "Equipment",     # e.g. P-7, MB-07, SV-7
    "Procedure",     # e.g. LOTO procedure, seal replacement
    "Regulation",    # e.g. Factory Act Section 21, OISD, PESO
    "Incident",      # e.g. NM-2026-03, RCA-2025-44
    "Parameter",     # e.g. max bearing temp 85C
    "Person",        # e.g. S. Rao
]

_EXTRACTION_PROMPT = """You extract a knowledge graph from an industrial plant \
document excerpt. Return STRICT JSON only -- no prose, no markdown fences.

Schema:
{
  "entities": [{"name": "<canonical short name>", "type": "<one of: Equipment, Procedure, Regulation, Incident, Parameter, Person>"}],
  "relations": [{"source": "<entity name>", "target": "<entity name>", "relation": "<short verb phrase>"}]
}

Rules:
- Use canonical names: equipment tags like "P-7", "SV-7"; regulations like "Factory Act Section 21".
- Only include entities actually present in THIS excerpt.
- relation examples: "has procedure", "governed by", "had incident", "has limit", "inspected by".
- If nothing relevant, return {"entities": [], "relations": []}.
"""


def _cache_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


class KnowledgeGraph:
    """A knowledge graph + the map of which chunks each entity appears in."""

    def __init__(self):
        self.g = nx.MultiDiGraph()
        # entity name -> set of chunk_ids where it appears (for hybrid retrieval)
        self.entity_chunks: dict[str, set[int]] = defaultdict(set)

    # ---------- LOCAL extraction (no API, no quota, instant) ----------
    # The LLM is for polishing ANSWERS, not for building the index. This
    # rule-based extractor finds the entity types industrial documents actually
    # use — tags, IDs, regulations, people, limits — with zero API calls, so a
    # rebuild can never die on a rate limit.
    _RX_EQUIP = __import__("re").compile(
        r"\b((?:P|C|B|E|T|V|HX|FD|S)-\d{1,4})\b|"
        r"\b(?:Pump|Compressor|Boiler|Fan|Motor|Valve|Exchanger|Tank)\s+"
        r"([A-Z]{1,3}-?\d{1,4})\b")
    _RX_DOCID = __import__("re").compile(
        r"\b((?:INC|NM|RCA)[- ]?\d[\w-]*)\b|"
        r"\b((?:SOP|WO|HW|AUD|INS|MLOG|DWG)[- ]?[A-Z]*\d[\w-]*)\b")
    _RX_REG = __import__("re").compile(
        r"\b(Factor(?:y|ies) Act(?:,? 1948)?(?: Section \d+)?|"
        r"OISD(?:[- ]STD)?[- ]?\d*|PESO|DGMS|IBR|"
        r"Indian Boiler Regulations|TIA-942|FB-\d{4}-\d+)\b")
    _RX_PERSON = __import__("re").compile(
        r"\b(?:Mr\.|Ms\.|Dr\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})|"
        r"Owner:\s*([A-Z]\.\s?[A-Z][a-z]+)|"
        # Bare "A. Surname" — but NOT an initial right after a number/unit
        # (e.g. "74 C. Oil" is a temperature, not a person).
        r"\b(?<![\d°]\s)([A-Z]\.\s[A-Z][a-z]{2,})\b")
    _RX_PARAM = __import__("re").compile(
        r"([A-Za-z][A-Za-z ]{3,32}?):\s*[\d.]+\s*(?:°?C|mm/s|bar|A)\b")
    # P&ID / drawing topology: "P-7 -> HX-2 : discharges to" (from vision on
    # engineering drawings) becomes a real equipment-to-equipment graph edge.
    _RX_CONN = __import__("re").compile(
        r"\b([A-Z]{1,3}-?\d{1,4})\b\s*(?:->|→|connects? to|feeds? into|"
        r"feeds?|discharges? to|routes? to)\s*\b([A-Z]{1,3}-?\d{1,4})\b"
        r"\s*(?::\s*([^\n]{0,40}))?")

    def _extract_local(self, text: str) -> dict:
        """Regex entity extraction + co-occurrence relations. Zero API calls."""
        ents: dict[str, str] = {}
        for m in self._RX_EQUIP.finditer(text):
            ents[(m.group(1) or m.group(2)).upper()] = "Equipment"
        for m in self._RX_DOCID.finditer(text):
            tag = (m.group(1) or m.group(2)).upper().replace(" ", "-")
            ents[tag] = "Incident" if m.group(1) else "Procedure"
        for m in self._RX_REG.finditer(text):
            ents[m.group(1)] = "Regulation"
        for m in self._RX_PERSON.finditer(text):
            name = next(g for g in m.groups() if g)
            ents[name.strip()] = "Person"
        for m in self._RX_PARAM.finditer(text):
            ents[m.group(1).strip()] = "Parameter"

        # Relations: link each equipment to the other entities in the chunk.
        rel_by_type = {"Incident": "had incident", "Procedure": "has procedure",
                       "Regulation": "governed by", "Person": "handled by",
                       "Parameter": "has limit"}
        relations = []
        equips = [n for n, t in ents.items() if t == "Equipment"]
        for eq in equips[:3]:
            for n, t in ents.items():
                if n != eq and t in rel_by_type:
                    relations.append({"source": eq, "target": n,
                                      "relation": rel_by_type[t]})
        # Explicit P&ID topology edges from "A -> B : relation" lines. These are
        # true equipment-to-equipment connections read off engineering drawings.
        for m in self._RX_CONN.finditer(text):
            a, b = m.group(1).upper(), m.group(2).upper()
            if a == b:
                continue
            ents.setdefault(a, "Equipment")
            ents.setdefault(b, "Equipment")
            relations.append({"source": a, "target": b,
                              "relation": (m.group(3) or "connected to").strip()})
        entities = [{"name": n, "type": t} for n, t in ents.items()]
        return {"entities": entities, "relations": relations}

    # ---------- LLM extraction (optional enrichment when quota allows) ----------
    def _extract(self, text: str) -> dict:
        """Ask the LLM for entities + relations in one chunk. Returns parsed JSON."""
        raw = chat(_EXTRACTION_PROMPT, text, max_tokens=900).strip()
        # Be tolerant of stray fences/prose around the JSON.
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1:
            return {"entities": [], "relations": []}
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return {"entities": [], "relations": []}

    # ---------- building ----------
    def build(self, chunks, cache_dir: str | Path = "data/index",
              max_chunks_per_file: int = 6) -> None:
        """
        Extract a graph from the chunks. Extractions are cached on disk keyed
        by chunk text, so re-running is instant and costs nothing.

        Cost control: semantic search already covers EVERY chunk; the graph
        only needs the entity-dense ones. Small operational documents (logs,
        incidents, SOPs) are extracted fully, but huge reference manuals /
        regulation PDFs are capped at their first `max_chunks_per_file` chunks
        — otherwise a single 90-page OEM manual would burn hundreds of LLM
        calls for very few new relationships.
        """
        import os as _os
        if _os.environ.get("GRAPH_MODE", "local").lower() == "local":
            # API-FREE build (default): extract EVERY chunk with local rules —
            # costs nothing, takes seconds, and can never die on a rate limit.
            # Set GRAPH_MODE=llm in .env to use LLM enrichment when quota allows.
            cache_path = Path("data") / "graph_extractions.json"
            cache = (json.loads(cache_path.read_text(encoding="utf-8"))
                     if cache_path.exists() else {})
            for chunk in chunks:
                self._add(self._extract_local(chunk.text), chunk.chunk_id)
                # Merge any PREVIOUSLY-cached LLM extraction for this chunk —
                # richer entities at zero new API cost.
                hit = cache.get(_cache_key(chunk.text))
                if hit:
                    self._add(hit, chunk.chunk_id)
            self.clean()
            print(f"Knowledge graph (LOCAL extractor + cached enrichment, "
                  f"0 API calls): {self.g.number_of_nodes()} nodes, "
                  f"{self.g.number_of_edges()} edges")
            return

        # The extraction cache is keyed by chunk TEXT, so it stays valid across
        # index rebuilds — keep it OUTSIDE the index dir (which "Rebuild index"
        # deletes), otherwise every rebuild re-pays every LLM extraction call.
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = Path("data") / "graph_extractions.json"
        legacy = cache_dir / "graph_extractions.json"
        if legacy.exists() and not cache_path.exists():
            legacy.rename(cache_path)               # migrate old location
        cache: dict[str, dict] = {}
        if cache_path.exists():
            cache = json.loads(cache_path.read_text(encoding="utf-8"))

        per_file: dict[str, int] = {}
        selected = []
        for chunk in chunks:
            n = per_file.get(chunk.source_file, 0)
            if n < max_chunks_per_file:
                selected.append(chunk)
                per_file[chunk.source_file] = n + 1
        skipped = len(chunks) - len(selected)
        if skipped:
            print(f"  graph extraction: {len(selected)} chunks selected, "
                  f"{skipped} deep-reference chunks covered by search only")

        failures = 0
        for i, chunk in enumerate(selected, start=1):
            key = _cache_key(chunk.text)
            if key not in cache:
                print(f"  graph-extract chunk {i}/{len(selected)}")
                try:
                    cache[key] = self._extract(chunk.text)
                except Exception as e:
                    # Free-tier quota / transient failure: checkpoint progress and
                    # SKIP this chunk (it stays covered by vector search). The graph
                    # still builds and saves; a later rebuild fills the gap.
                    failures += 1
                    cache_path.write_text(json.dumps(cache), encoding="utf-8")
                    if failures >= 5:
                        print(f"  [graph] stopping extraction early: {str(e)[:60]}")
                        break
                    continue
                if i % 10 == 0:                      # periodic checkpoint
                    cache_path.write_text(json.dumps(cache), encoding="utf-8")
            self._add(cache[key], chunk.chunk_id)

        cache_path.write_text(json.dumps(cache), encoding="utf-8")
        if failures:
            print(f"  [graph] {failures} chunk(s) skipped (quota/transient); "
                  f"graph built from the rest.")
        self.clean()
        print(f"Knowledge graph: {self.g.number_of_nodes()} nodes, "
              f"{self.g.number_of_edges()} edges")

    def _add(self, extraction: dict, chunk_id: int) -> None:
        for ent in extraction.get("entities", []):
            name, etype = ent.get("name"), ent.get("type", "Unknown")
            if not name:
                continue
            if not self.g.has_node(name):
                self.g.add_node(name, type=etype)
            self.entity_chunks[name].add(chunk_id)
        for rel in extraction.get("relations", []):
            s, t = rel.get("source"), rel.get("target")
            if s and t:
                self.g.add_edge(s, t, label=rel.get("relation", "related to"))

    # ---------- graph hygiene ----------
    _GENERIC = {"ring", "coupling", "sop", "pump", "valve", "boiler", "fan",
                "motor", "bearing", "grease", "belt", "impeller", "seal",
                "plant", "equipment", "document", "procedure", "email"}

    def clean(self) -> None:
        """
        Normalise the graph so the visualization looks engineered, not scraped:
        - drop quantity fragments ("250 running hours", "10.5 bar") — they are
          facts inside chunks, not entities;
        - drop bare generic words ("ring", "coupling") that carry no identity;
        - merge duplicate names: case-insensitive twins and acronym pairs like
          "Indian Boiler Regulations (IBR)" -> "IBR".
        """
        import re
        g = self.g

        # 1. Junk removal.
        drop = [n for n in list(g.nodes)
                if re.match(r"^\d", str(n)) or str(n).lower() in self._GENERIC]
        g.remove_nodes_from(drop)
        for n in drop:
            self.entity_chunks.pop(n, None)

        # 2. Duplicate merging (build old->canonical mapping).
        mapping: dict[str, str] = {}
        by_lower: dict[str, str] = {}
        for n in list(g.nodes):
            low = str(n).lower()
            if low in by_lower and by_lower[low] != n:
                a, b = by_lower[low], n           # keep the better-connected one
                keep, merge = (a, b) if g.degree(a) >= g.degree(b) else (b, a)
                mapping[merge] = keep
                by_lower[low] = keep
            else:
                by_lower[low] = n
        for n in list(g.nodes):                    # "Full Name (ACR)" -> "ACR"
            m = re.match(r"^(.+?)\s*\(([^)]{2,15})\)$", str(n))
            if m:
                inner = m.group(2).strip()
                if inner in g and inner != n:
                    mapping[n] = mapping.get(inner, inner)

        if mapping:
            self.g = nx.relabel_nodes(g, mapping, copy=True)
            for old, new in mapping.items():
                self.entity_chunks.setdefault(new, set())
                self.entity_chunks[new] |= self.entity_chunks.pop(old, set())

    # ---------- hybrid retrieval support ----------
    def chunks_near(self, query: str, max_chunks: int = 4) -> set[int]:
        """
        Find chunk_ids connected to entities mentioned in the query, plus their
        graph neighbours. This is what lets a P-7 question also pull in the
        inspection note that never repeats the word 'lockout'.
        """
        q = query.lower()
        hit_ids: set[int] = set()
        for name in self.g.nodes:
            if name.lower() in q:
                hit_ids |= self.entity_chunks.get(name, set())
                for nbr in nx.all_neighbors(self.g, name):
                    hit_ids |= self.entity_chunks.get(nbr, set())
        return set(list(hit_ids)[:max_chunks])

    # ---------- persistence ----------
    def save(self, folder: str | Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        nx.write_gml(self.g, folder / "graph.gml")
        with open(folder / "entity_chunks.json", "w", encoding="utf-8") as f:
            json.dump({k: list(v) for k, v in self.entity_chunks.items()}, f)

    def load(self, folder: str | Path) -> bool:
        folder = Path(folder)
        gml, ec = folder / "graph.gml", folder / "entity_chunks.json"
        if not (gml.exists() and ec.exists()):
            return False
        self.g = nx.read_gml(gml)
        with open(ec, encoding="utf-8") as f:
            self.entity_chunks = {k: set(v) for k, v in json.load(f).items()}
        return True

    # ---------- visualization ----------
    def to_pyvis_html(self, out_path: str | Path = "data/index/graph.html") -> str:
        """Render an interactive graph; returns the HTML string for Streamlit."""
        from pyvis.network import Network

        colors = {
            "Equipment": "#2563eb", "Procedure": "#16a34a",
            "Regulation": "#9333ea", "Incident": "#dc2626",
            "Parameter": "#ea580c", "Person": "#0891b2",
        }
        net = Network(height="600px", width="100%", directed=True, bgcolor="#ffffff")
        for node, data in self.g.nodes(data=True):
            t = data.get("type", "Unknown")
            net.add_node(node, label=node, color=colors.get(t, "#64748b"),
                         title=f"{t}: {node}")
        for s, t, data in self.g.edges(data=True):
            net.add_edge(s, t, title=data.get("label", ""))
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        net.save_graph(str(out_path))
        return out_path.read_text(encoding="utf-8")
