# © 2026 Plant Operations Brain Team — ET AI Hackathon. All rights reserved.
# Proprietary & confidential. Unauthorised copying, reuse, or redistribution is prohibited (see LICENSE).
"""
health.py
---------
The Plant Health Score: one 0-100 number a plant manager can glance at,
composed from fully explainable, deterministic sub-scores (no LLM, no cost):

  - Live sensors   (40%) : how many watched readings are OK vs warning/alarm
  - Knowledge      (25%) : how well-documented the equipment fleet is
  - Open actions   (20%) : pending/overdue items found in the documents
  - Conflicts      (15%) : unresolved contradictions between documents

Every sub-score lists exactly what moved it, so the number can be defended
line-by-line — the opposite of a black-box KPI.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field

from .rag import PlantBrain
from .sensors import get_registry, status_at
from .interview import knowledge_risk_map

_PENDING_RE = re.compile(
    r"\b(pending|awaited|overdue|not yet|still pending|must not operate|"
    r"flagged for review|was never updated)\b", re.IGNORECASE)


@dataclass
class HealthReport:
    total: int                     # 0-100
    grade: str                     # A / B / C / D
    subscores: dict = field(default_factory=dict)   # name -> (score0to1, note)


def plant_health(brain: PlantBrain, hours: int = 48,
                 n_conflicts: int | None = None,
                 registry: dict | None = None,
                 risk: list | None = None,
                 pending: int | None = None) -> HealthReport:
    # Callers may pass precomputed registry/risk/pending (they scan the whole
    # corpus and don't change between reruns) to avoid recomputing every click.
    # --- 1. Sensors (40%) ---
    if registry is None:
        registry = get_registry(brain.store.chunks)
    pts, worst = [], []
    for name in registry:
        s = status_at(name, hours, registry)
        pts.append({"OK": 1.0, "WARNING": 0.5, "ALARM": 0.0}[s.state])
        if s.state != "OK":
            worst.append(f"{s.name} {s.state}")
    sensor_score = sum(pts) / len(pts) if pts else 1.0
    sensor_note = "; ".join(worst) if worst else "all readings within limits"

    # --- 2. Knowledge coverage (25%) ---
    if risk is None:
        risk = knowledge_risk_map(brain)
    if risk:
        k_pts = [{"Low": 1.0, "Medium": 0.5, "High": 0.0}[r.risk] for r in risk]
        know_score = sum(k_pts) / len(k_pts)
        n_high = sum(1 for r in risk if r.risk == "High")
        know_note = (f"{n_high} asset(s) barely documented"
                     if n_high else "fleet well documented")
    else:
        know_score, know_note = 1.0, "no equipment tracked yet"

    # --- 3. Open actions (20%) ---
    n_pending = (pending if pending is not None
                 else sum(bool(_PENDING_RE.search(c.text)) for c in brain.store.chunks))
    action_score = max(0.0, 1.0 - n_pending / 8.0)
    action_note = (f"{n_pending} document(s) mention pending/overdue items"
                   if n_pending else "no open items found")

    # --- 4. Conflicts (15%) ---
    if n_conflicts is None:
        conflict_score, conflict_note = 0.7, "conflict scan not run yet"
    else:
        conflict_score = max(0.0, 1.0 - n_conflicts / 4.0)
        conflict_note = (f"{n_conflicts} unresolved contradiction(s)"
                         if n_conflicts else "no contradictions")

    total = round(100 * (0.40 * sensor_score + 0.25 * know_score
                         + 0.20 * action_score + 0.15 * conflict_score))
    grade = ("A" if total >= 85 else "B" if total >= 70
             else "C" if total >= 50 else "D")
    return HealthReport(total, grade, {
        "Live sensors (40%)": (sensor_score, sensor_note),
        "Knowledge coverage (25%)": (know_score, know_note),
        "Open actions (20%)": (action_score, action_note),
        "Conflicts (15%)": (conflict_score, conflict_note),
    })
