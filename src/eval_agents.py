# © 2026 Plant Operations Brain Team — ET AI Hackathon. All rights reserved.
# Proprietary & confidential. Unauthorised copying, reuse, or redistribution is prohibited (see LICENSE).
"""
eval_agents.py
--------------
Strict evaluation of the RAG *beyond the copilot* — every agentic feature is a
RAG consumer, and each can fail differently. This checks that each agent
(a) retrieves the right evidence and (b) produces the key fact, with citations.

Each check states a concrete expectation and PASS/FAIL. Uses the LLM (agents
generate text), so it costs a handful of calls — run when quota is fresh.

Run:  python -m src.eval_agents
"""

from __future__ import annotations
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from .embed_store import VectorStore, Reranker
from .graph import KnowledgeGraph
from .rag import PlantBrain
from .maintenance import analyze_equipment
from .compliance import audit
from .lessons import mine_lessons
from .conflict import find_conflicts
from .whatif import check_plan
from .impact import analyze_impact
from .handover import generate_brief
from .deep import deep_ask

INDEX = "data/index"


def _has(text: str, *needles: str) -> bool:
    t = (text or "").lower()
    return all(any(n.lower() in t for n in group.split("|")) for group in needles)


def _cited(text: str) -> bool:
    return "[source" in (text or "").lower()


def run() -> None:
    store = VectorStore()
    if not store.load(INDEX):
        print("No index."); return
    g = KnowledgeGraph(); g.load(INDEX)
    brain = PlantBrain(store, graph=g, reranker=Reranker())
    results = []

    def check(name, ok, detail=""):
        results.append((name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + (f"  — {detail}" if detail and not ok else ""))

    print("\n=== Maintenance / RCA Agent ===")
    r = analyze_equipment(brain, "P-7")
    check("RCA finds P-7 root cause (misalignment)",
          _has(r.report_md, "misalign"), r.report_md[:80])
    check("RCA cites sources", _cited(r.report_md))

    print("\n=== Compliance Agent ===")
    c = audit(brain)
    check("Compliance names a real regulation (Factory Act|OISD|PESO|IBR)",
          _has(c.report_md, "factory act|oisd|peso|ibr"))
    check("Compliance flags the SOP/LEL gap",
          _has(c.report_md, "sop-bh-04|lel|discharge|lubric"), c.report_md[:80])
    check("Compliance cites sources", _cited(c.report_md))

    print("\n=== Lessons Learned Agent ===")
    le = mine_lessons(brain)
    check("Lessons surfaces a recurring pattern (misalignment|lubric|valve)",
          _has(le.report_md, "misalign|lubric|valve|recurring"))
    check("Lessons cites sources", _cited(le.report_md))

    print("\n=== Conflict Detection ===")
    cf = find_conflicts(brain)
    entities = {x.entity for x in cf}
    check("Detects the FD-2 / SOP-BH-04 lube-interval conflict",
          any("fd-2" in e.lower() or "bh-04" in e.lower() for e in entities)
          or any(_has(x.summary, "250|500|lubric") for x in cf),
          f"found: {sorted(entities)}")

    print("\n=== Safety Check (What-If) ===")
    v = check_plan(brain, "Replace P-7 seal with discharge valve DV-7 still open.")
    check("Safety returns STOP for the known near-miss condition",
          v.verdict == "STOP", f"verdict={v.verdict}")
    check("Safety cites the near-miss / valve rule",
          _has(" ".join(v.reasons), "dv-7|discharge|both|water hammer|nm-2026"))

    print("\n=== Impact Analysis ===")
    im = analyze_impact(brain, "FD-2")
    types = im.neighbours
    check("Impact links FD-2 to an incident node",
          "Incident" in types and len(types["Incident"]) >= 1,
          f"types={list(types)}")
    check("Impact summary mentions the B-3 boiler interlock",
          _has(im.summary_md, "b-3|boiler|interlock|steam"))

    print("\n=== Handover Brief ===")
    h = generate_brief(brain, hours=90)
    check("Handover surfaces a live watch item (P-7|FD-2|temperature|vibration)",
          _has(h.brief_md, "p-7|fd-2|temperature|vibration|alarm"))
    check("Handover cites sources", _cited(h.brief_md))

    print("\n=== Deep Analysis (multi-step) ===")
    d = deep_ask(brain, "Why does P-7 keep failing and what must we fix before audit?")
    check("Deep produced >=2 reasoning steps", len(d.steps) >= 2,
          f"{len(d.steps)} steps")
    check("Deep final answer cites sources & names the cause",
          _cited(d.final_md) and _has(d.final_md, "misalign|bearing"))

    # ---- summary ----
    n_pass = sum(1 for _, ok, _ in results if ok)
    fails = [n for n, ok, _ in results if not ok]
    print(f"\n{'='*60}\nAGENT RAG SCORE: {n_pass}/{len(results)} checks passed")
    if fails:
        print("FAILURES:")
        for f in fails:
            print(f"  ✗ {f}")


if __name__ == "__main__":
    run()
