# © 2026 Plant Operations Brain Team — ET AI Hackathon. All rights reserved.
# Proprietary & confidential. Unauthorised copying, reuse, or redistribution is prohibited (see LICENSE).
"""
sop_fix.py
----------
Closes the loop that every document-AI product leaves open.

Detection tools FLAG outdated procedures; a human then spends days rewriting
them (or doesn't — which is exactly how SOP-BH-04 stayed wrong for 4 months and
caused INC-2026-11). This module goes from detected conflict -> DRAFTED FIX:

  conflict -> LLM drafts the corrected procedure with a change log
           -> saved to data/drafts/ as PENDING APPROVAL
           -> one click "approve" moves it into the live corpus, where the next
              index rebuild makes it the searchable source of truth.

The human stays in charge (nothing goes live without approval) — but the
rewriting work drops from days to seconds.
"""

from __future__ import annotations
from datetime import date
from pathlib import Path

from .llm import chat

DRAFTS_DIR = Path("data/drafts")

_SYSTEM = """You are a plant documentation engineer. Two or more documents \
CONTRADICT each other about the same equipment/procedure. Draft the CORRECTED \
controlled document that resolves the contradiction in favour of the most \
recent / most authoritative source (incident corrective actions and OEM \
bulletins override older SOPs).

Output markdown with EXACTLY these sections:
# <document title> (REVISED DRAFT)
## Change Log
- what changed, why, and which source mandates it
## Procedure
- the full corrected procedure steps, ready to use
## Approval
- state: DRAFT — PENDING ENGINEER APPROVAL, drafted by Plant Operations Brain

Rules: use ONLY facts from the excerpts; keep every step that is still valid;
change only what the contradiction requires; be precise with numbers."""


def draft_fix(entity: str, summary: str, sources) -> tuple[str, str]:
    """
    Draft the corrected document for one detected conflict.
    Returns (draft_markdown, saved_path).
    """
    excerpts = "\n\n".join(
        f"[Doc: {c.source_file}]\n{c.text}" for c in sources)
    draft = chat(
        _SYSTEM,
        f"Conflicting entity: {entity}\n"
        f"Detected contradiction: {summary}\n\n"
        f"Document excerpts:\n{excerpts}\n\n"
        f"Draft the corrected controlled document.",
        max_tokens=1100)

    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    safe = entity.replace("/", "-").replace(" ", "_")
    path = DRAFTS_DIR / f"REVISED_{safe}_{date.today().isoformat()}.md"
    path.write_text(draft, encoding="utf-8")
    return draft, str(path)


def approve(draft_path: str, docs_dir: str = "data/docs") -> str:
    """Engineer approval: move the draft into the live corpus."""
    src = Path(draft_path)
    dest = Path(docs_dir) / src.name.replace("REVISED_", "APPROVED_")
    text = src.read_text(encoding="utf-8").replace(
        "DRAFT — PENDING ENGINEER APPROVAL",
        f"APPROVED {date.today().isoformat()}")
    dest.write_text(text, encoding="utf-8")
    src.unlink(missing_ok=True)
    return str(dest)
