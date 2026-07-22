# © 2026 Plant Operations Brain Team — ET AI Hackathon. All rights reserved.
# Proprietary & confidential. Unauthorised copying, reuse, or redistribution is prohibited (see LICENSE).
"""
graph_neo4j.py
--------------
Real Neo4j graph-database backend for the knowledge graph.

Design goal: USE Neo4j for real (entities + relationships stored in the DB,
retrieval done with Cypher graph traversal) -- but NEVER break the demo if Neo4j
isn't running. So `make_graph()` returns a Neo4jGraph when a Neo4j instance is
configured & reachable, otherwise it transparently falls back to the in-memory
NetworkX KnowledgeGraph.

Configure in .env:
    NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io   (or bolt://localhost:7687)
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=your-password

Get a FREE cloud instance in 2 min at https://neo4j.com/product/auradb/ (AuraDB Free).
"""

from __future__ import annotations
import os
from pathlib import Path

from .graph import KnowledgeGraph


class Neo4jGraph:
    """
    Neo4j-backed knowledge graph. Keeps a NetworkX mirror (self.kg) for fast
    visualization + node iteration, but STORES data in Neo4j and answers
    `chunks_near` with a real Cypher traversal.
    """

    def __init__(self, uri: str, user: str, password: str):
        from neo4j import GraphDatabase

        # Try the URI as given first. If the network does TLS interception
        # (corporate/college proxy or antivirus HTTPS scanning), strict cert
        # verification fails -- so fall back to the "+ssc" scheme, which still
        # encrypts but doesn't verify the certificate chain.
        candidates = [uri]
        if "+s://" in uri and "+ssc://" not in uri:
            candidates.append(uri.replace("+s://", "+ssc://"))

        last_err = None
        for candidate in candidates:
            try:
                drv = GraphDatabase.driver(candidate, auth=(user, password),
                                           connection_timeout=15)
                drv.verify_connectivity()
                self.driver = drv
                break
            except Exception as e:
                last_err = e
        else:
            raise last_err                          # none of the schemes worked

        self.kg = KnowledgeGraph()                  # source of truth + viz

    # expose the same attributes app.py / rag.py expect
    @property
    def g(self):
        return self.kg.g

    @property
    def entity_chunks(self):
        return self.kg.entity_chunks

    # ---------- build / sync ----------
    def build(self, chunks, cache_dir="data/index") -> None:
        self.kg.build(chunks, cache_dir=cache_dir)  # extraction (LLM) -> networkx
        self._write_to_neo4j()

    def _write_to_neo4j(self) -> None:
        """
        Push the current networkx graph into Neo4j (idempotent). Uses batched
        UNWIND statements — one round-trip for all nodes, one for all edges —
        instead of a query per node/edge, which took ~1 min against a cloud
        instance once the graph grew to hundreds of entities.
        """
        nodes = [{"name": n,
                  "type": d.get("type", "Unknown"),
                  "chunks": sorted(self.kg.entity_chunks.get(n, []))}
                 for n, d in self.kg.g.nodes(data=True)]
        edges = [{"s": s, "t": t, "l": d.get("label", "related to")}
                 for s, t, d in self.kg.g.edges(data=True)]
        with self.driver.session() as sess:
            sess.run("MATCH (n:Entity) DETACH DELETE n")   # clean slate
            sess.run(
                "UNWIND $rows AS row "
                "MERGE (e:Entity {name:row.name}) "
                "SET e.type=row.type, e.chunks=row.chunks",
                rows=nodes)
            sess.run(
                "UNWIND $rows AS row "
                "MATCH (a:Entity {name:row.s}), (b:Entity {name:row.t}) "
                "MERGE (a)-[r:REL {label:row.l}]->(b)",
                rows=edges)

    # ---------- retrieval (real Cypher traversal, with in-memory fallback) ----
    def chunks_near(self, query: str, max_chunks: int = 4) -> set[int]:
        # If Neo4j died earlier this session (Aura auto-paused, network blip),
        # don't keep timing out on every query — use the in-memory mirror, which
        # is always populated and gives identical results.
        if getattr(self, "_neo4j_dead", False):
            return self.kg.chunks_near(query, max_chunks)
        q = query.lower()
        ids: set[int] = set()
        try:
            with self.driver.session() as sess:
                rows = sess.run("MATCH (e:Entity) RETURN e.name AS name, "
                                "e.chunks AS chunks")
                mentioned = [(r["name"], r["chunks"] or []) for r in rows
                             if r["name"] and r["name"].lower() in q]
                for name, chunks in mentioned:
                    ids.update(chunks)
                    nb = sess.run("MATCH (e:Entity {name:$n})-[]-(m:Entity) "
                                  "RETURN m.chunks AS c", n=name)
                    for row in nb:
                        ids.update(row["c"] or [])
            return set(list(ids)[:max_chunks])
        except Exception as e:
            # Any Neo4j failure -> fall back permanently for this session.
            self._neo4j_dead = True
            print(f"[graph] Neo4j unavailable mid-session ({str(e)[:50]}); "
                  f"using in-memory graph from now on.")
            return self.kg.chunks_near(query, max_chunks)

    # ---------- persistence (delegate to networkx mirror + re-sync) ----------
    def save(self, folder) -> None:
        self.kg.save(folder)

    def load(self, folder) -> bool:
        loaded = self.kg.load(folder)
        if loaded:
            self._write_to_neo4j()   # repopulate Neo4j from cached graph
        return loaded

    def to_pyvis_html(self, out_path="data/index/graph.html") -> str:
        return self.kg.to_pyvis_html(out_path)

    def close(self) -> None:
        try:
            self.driver.close()
        except Exception:
            pass


def _reachable(uri: str, timeout: float = 2.0) -> bool:
    """
    Fast pre-check: can we even open a TCP socket to the Neo4j host? A PAUSED
    Aura instance won't resolve/connect, and without this the driver would sit
    retrying for ~30s at every app startup. This makes the fallback near-instant.
    """
    import socket
    from urllib.parse import urlparse
    host = urlparse(uri).hostname
    if not host:
        return False
    try:
        with socket.create_connection((host, 7687), timeout=timeout):
            return True
    except Exception:
        return False


def make_graph():
    """
    Return a Neo4jGraph if Neo4j is configured & reachable, else the in-memory
    KnowledgeGraph. Returns (graph, backend_name).
    """
    uri = os.environ.get("NEO4J_URI", "").strip()
    if uri and _reachable(uri):
        try:
            g = Neo4jGraph(
                uri,
                os.environ.get("NEO4J_USER", "neo4j"),
                os.environ.get("NEO4J_PASSWORD", ""),
            )
            return g, "Neo4j"
        except Exception as e:
            print(f"[graph] Neo4j configured but unreachable ({str(e)[:80]}); "
                  f"using in-memory graph.")
    elif uri:
        print("[graph] Neo4j is paused/unreachable — using in-memory graph "
              "(resume the Aura instance to reconnect).")
    return KnowledgeGraph(), "in-memory (NetworkX)"
