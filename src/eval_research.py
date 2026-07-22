# © 2026 Plant Operations Brain Team — ET AI Hackathon. All rights reserved.
# Proprietary & confidential. Unauthorised copying, reuse, or redistribution is prohibited (see LICENSE).
"""
eval_research.py
----------------
Researcher-grade RAG evaluation — the tests an IR reviewer would actually run,
beyond the happy-path benchmark:

  1. Gold retrieval:      Recall@5 and MRR on the labelled set.
  2. Paraphrase robustness: same questions, reworded — retrieval must survive.
  3. ID-format variants:   case/spacing/dash mutations of identifiers.
  4. Unanswerable in-domain: plausible-sounding questions about entities that
     DON'T exist — the system must ABSTAIN, not substitute a similar entity.
  5. Off-topic:            must abstain.

Everything here is retrieval-side (embeddings + reranker + lexical are all
local), so the whole suite runs FREE — no API, no quota.

Run:  python -m src.eval_research
"""

from __future__ import annotations
import sys
import json
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from .embed_store import VectorStore, Reranker
from .graph import KnowledgeGraph
from .rag import PlantBrain

INDEX = "data/index"

GOLD = [
    ("What is the lockout procedure for Pump 7?", "pump7_maintenance_log.md"),
    ("Why does the P-7 drive-end bearing keep failing?", "pump7_maintenance_log.md"),
    ("At what temperature does compressor C-12 trip?", "compressor_C12_oem_manual.md"),
    ("When is the next statutory inspection of boiler B-3 due?", "boiler_B3_inspection.md"),
    ("How often should the FD-2 fan bearings be lubricated now?", "incident_report_INC2026_11.md|oem_bulletin_fd2.eml"),
    ("What did work order WO-228 record about FD-2?", "maintenance_register_2026.xlsx"),
]

PARAPHRASES = [
    ("How do I safely isolate pump P-7 before working on it?", "pump7_maintenance_log.md"),
    ("Reason for repeated bearing damage on the cooling water pump?", "pump7_maintenance_log.md"),
    ("Max temp before the air compressor shuts itself down?", "compressor_C12_oem_manual.md"),
    ("Deadline for the boiler's next legal inspection?", "boiler_B3_inspection.md"),
    ("New greasing schedule for the boiler house draft fan?", "incident_report_INC2026_11.md|oem_bulletin_fd2.eml|SOP_FD2_fan.docx"),
    ("bearing temp check result for the FD fan in april", "maintenance_register_2026.xlsx"),
]

ID_VARIANTS = [
    ("24BDS085", "24BDS085.pdf"),
    ("24bds085", "24BDS085.pdf"),
    ("what is 24 BDS 085", "24BDS085.pdf"),
    ("WO-228", "maintenance_register_2026.xlsx"),
    ("wo228", "maintenance_register_2026.xlsx"),
    ("show me WO 228", "maintenance_register_2026.xlsx"),
    ("NM-2026-03", "incident_report_NM2026_03.md"),
    ("FB-2025-09", "oem_bulletin_fd2.eml|incident_report_INC2026_11.md"),
]

# Entities that DO NOT exist anywhere in the corpus. Retrieval will surface
# *similar* chunks (P-7, FD-2...) — the gate must refuse anyway.
UNANSWERABLE = [
    "What is the trip pressure of pump P-99?",
    "Lubrication interval for the FD-9 fan?",
    "What did work order WO-999 record?",
    "When was incident NM-2031-77 investigated?",
]

OFFTOPIC = [
    "What is the capital of France?",
    "Best biryani recipe?",
    "Who is the CEO of Google?",
]


def run() -> None:
    store = VectorStore()
    if not store.load(INDEX):
        print("No index. Build it first.")
        return
    graph = KnowledgeGraph()
    graph.load(INDEX)
    brain = PlantBrain(store, graph=graph, reranker=Reranker())
    problems: list[str] = []

    # Only test for documents that ACTUALLY exist in the corpus — a suite that
    # references removed files reports phantom failures.
    present = {c.source_file for c in store.chunks}

    def _present(want: str) -> bool:
        return any(f in present for f in want.split("|"))

    def _filter(suite):
        kept = [(q, w) for q, w in suite if _present(w)]
        skipped = len(suite) - len(kept)
        return kept, skipped

    def _hits_files(q, k=5):
        return [c.source_file for c, _ in brain.retrieve(q, k=k)]

    # ---- 1 & 2 & 3: recall + MRR over all answerable suites ----
    for title, suite in [("1. GOLD", GOLD), ("2. PARAPHRASE", PARAPHRASES),
                         ("3. ID-VARIANTS", ID_VARIANTS)]:
        suite, skipped = _filter(suite)
        if not suite:
            print(f"\n=== {title}: all target docs absent — skipped ===")
            continue
        rec = 0
        rr_total = 0.0
        note = f" ({skipped} skipped: doc not in corpus)" if skipped else ""
        print(f"\n=== {title} ({len(suite)} queries){note} ===")
        for q, want in suite:
            wanted = set(want.split("|"))
            files = _hits_files(q)
            rank = next((i + 1 for i, f in enumerate(files) if f in wanted), 0)
            ok = rank > 0
            rec += ok
            rr_total += (1.0 / rank) if rank else 0.0
            mark = f"HIT@{rank}" if ok else "MISS "
            print(f"  [{mark}] {q[:58]:58} -> {files[0] if files else '-'}")
            if not ok:
                problems.append(f"{title}: MISS — {q!r} (top={files[:2]})")
            elif rank > 2:
                problems.append(f"{title}: weak rank {rank} — {q!r}")
        print(f"  Recall@5: {rec}/{len(suite)}  MRR: {rr_total/len(suite):.2f}")

    # ---- 4: unanswerable in-domain (gate must refuse) ----
    print(f"\n=== 4. UNANSWERABLE-IN-DOMAIN (must abstain) ===")
    for q in UNANSWERABLE:
        _, score, passed = brain._retrieve_scored(q, k=5)
        verdict = "ABSTAIN (correct)" if not passed else "PASSED GATE (WRONG)"
        print(f"  [{verdict}] score={score:.2f}  {q}")
        if passed:
            problems.append(f"4. GATE LEAK: {q!r} passed the gate (score "
                            f"{score:.2f}) — generation prompt is the only "
                            f"defence against entity substitution")

    # ---- 5: off-topic ----
    print(f"\n=== 5. OFF-TOPIC (must abstain) ===")
    for q in OFFTOPIC:
        _, score, passed = brain._retrieve_scored(q, k=5)
        verdict = "ABSTAIN (correct)" if not passed else "ANSWERED (WRONG)"
        print(f"  [{verdict}] score={score:.2f}  {q}")
        if passed:
            problems.append(f"5. OFF-TOPIC LEAK: {q!r}")

    # ---- strict findings ----
    print(f"\n{'='*62}\nSTRICT FINDINGS ({len(problems)}):")
    if not problems:
        print("  none — all suites clean")
    for p in problems:
        print(f"  ✗ {p}")


if __name__ == "__main__":
    run()
