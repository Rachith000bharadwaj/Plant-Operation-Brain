# © 2026 Plant Operations Brain Team — ET AI Hackathon. All rights reserved.
# Proprietary & confidential. Unauthorised copying, reuse, or redistribution is prohibited (see LICENSE).
"""
eval_entities.py
----------------
Gold-labelled ENTITY-EXTRACTION accuracy — precision / recall / F1 — directly
answering PS #8's evaluation focus: "entity extraction accuracy across document
types". Runs the local rule-based extractor over hand-labelled snippets that
span every document type (SOP, incident, work order, regulation, inspection,
P&ID drawing) and scores it against an exhaustive gold set.

Free & reproducible: uses the local extractor only (no API, no index needed).

Run:  python -m src.eval_entities
"""

from __future__ import annotations
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")   # Windows console safe ✓/✗
except Exception:
    pass
from .graph import KnowledgeGraph

# (snippet, {gold entity names present}). Gold is EXHAUSTIVE per snippet, so any
# extra extraction counts against precision and any miss against recall.
GOLD: list[tuple[str, set[str]]] = [
    # --- SOP / procedure ---
    ("SOP001: Pump startup for P-7. Confirm suction valve V-3 open. "
     "Discharge pressure: 6.2 bar. Vibration: 2.1 mm/s.",
     {"P-7", "V-3", "SOP001", "Discharge pressure", "Vibration"}),

    # --- Incident report ---
    ("Incident INC-2026-11 on compressor C-12. Seal failure. "
     "Investigated by Mr. Rakesh Sharma. Governed by OISD-STD-117.",
     {"C-12", "INC-2026-11", "Rakesh Sharma", "OISD-STD-117"}),

    # --- Work order ---
    ("Work order WO-014 raised for boiler B-3. Replace gauge glass. "
     "Refer Factories Act 1948 Section 22.",
     {"B-3", "WO-014", "Factories Act 1948 Section 22"}),

    # --- Regulation-heavy ---
    ("Pressure vessel inspection per IBR and PESO rules. "
     "Cross-check DGMS guidance for the mine site.",
     {"IBR", "PESO", "DGMS"}),

    # --- Inspection record with parameters ---
    ("Inspection INS-3 of exchanger HX-2. Bearing temperature: 74 C. "
     "Oil pressure: 3.1 bar.",
     {"HX-2", "INS-3", "Bearing temperature", "Oil pressure"}),

    # --- P&ID / drawing topology (the connection extractor) ---
    ("Drawing DWG-9. CONNECTIONS:\nP-7 -> HX-2 : discharges to\n"
     "HX-2 -> T-4 : feeds\nT-4 -> V-3 : routes to",
     {"P-7", "HX-2", "T-4", "V-3", "DWG-9"}),
]


def _norm(s: str) -> str:
    return s.upper().replace(" ", "").replace("-", "").strip()


def run() -> None:
    g = KnowledgeGraph()
    tp = fp = fn = 0
    conn_edges = 0
    print("=== Entity Extraction — precision / recall / F1 (gold-labelled) ===\n")
    for text, gold in GOLD:
        ext = g._extract_local(text)
        got = {e["name"] for e in ext["entities"]}
        conn_edges += sum(
            1 for r in ext["relations"]
            if r["relation"] not in ("had incident", "has procedure",
                                     "governed by", "handled by", "has limit"))
        gold_n = {_norm(x) for x in gold}
        got_n = {_norm(x) for x in got}
        hit = gold_n & got_n
        miss = gold_n - got_n
        extra = got_n - gold_n
        tp += len(hit); fn += len(miss); fp += len(extra)
        flag = "✓" if not miss and not extra else "✗"
        print(f"  {flag} recall {len(hit)}/{len(gold_n)}"
              f"  miss={sorted({x for x in gold if _norm(x) in miss})}"
              f"  extra={sorted({x for x in got if _norm(x) in extra})}")

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    print(f"\nPrecision: {prec:.0%}   Recall: {rec:.0%}   F1: {f1:.0%}")
    print(f"P&ID topology edges recovered from drawing text: {conn_edges}")
    print("\nAcross SOP, incident, work-order, regulation, inspection and P&ID "
          "document types — no API calls.")


if __name__ == "__main__":
    run()
