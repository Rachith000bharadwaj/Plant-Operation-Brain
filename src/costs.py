# © 2026 Plant Operations Brain Team — ET AI Hackathon. All rights reserved.
# Proprietary & confidential. Unauthorised copying, reuse, or redistribution is prohibited (see LICENSE).
"""
costs.py
--------
The Business-Impact engine: puts rupee figures on what the brain prevents.

Judges (and plant managers) don't buy "AI is useful" — they buy avoided cost.
Every number here is deterministic and comes from a stated, editable assumption
(the ASSUMPTIONS table), so the figures are defensible: change the rate, the
totals recompute. No LLM involved.
"""

from __future__ import annotations
from dataclasses import dataclass

# Editable cost assumptions (₹). Conservative mid-size-plant figures.
ASSUMPTIONS = {
    "steam_downtime_per_hour": 250_000,     # lost production when B-3 steam stops
    "cooling_downtime_per_hour": 180_000,   # cooling water loss (P-7 circuit)
    "compressed_air_downtime_per_hour": 120_000,
    "bearing_replacement": 85_000,          # parts + labour, one DE bearing job
    "emergency_callout_premium": 40_000,    # unplanned vs planned work premium
    "near_miss_investigation": 150_000,     # investigation + retraining cost
    "lti_direct_cost": 2_500_000,           # one lost-time injury (direct only)
    "audit_noncompliance_penalty": 500_000, # typical statutory penalty exposure
}

# --- Industry profiles -------------------------------------------------------
# The engine is industry-agnostic: documents drive the knowledge, and the cost
# model rescales per industry. Multipliers are relative to a mid-size process
# plant; swap for real client rates in ASSUMPTIONS for a deployment.
_BASE = dict(ASSUMPTIONS)
INDUSTRY_PROFILES = {
    "Chemical / Process (default)": 1.0,
    "Oil & Gas / Refinery": 2.5,
    "Power & Utilities": 1.8,
    "Pharma (GMP)": 2.0,
    "Steel / Cement / Heavy Mfg": 1.5,
    "General Manufacturing": 0.8,
}


def set_industry(name: str) -> float:
    """Rescale the whole cost model for the selected industry. Idempotent."""
    m = INDUSTRY_PROFILES.get(name, 1.0)
    for k, v in _BASE.items():
        ASSUMPTIONS[k] = int(v * m)
    return m


# Which downtime rate applies to each known asset (fallback: compressed air).
_EQUIP_RATE = {
    "B-3": "steam_downtime_per_hour",
    "FD-2": "steam_downtime_per_hour",      # FD-2 trip takes B-3 down (interlock)
    "P-7": "cooling_downtime_per_hour",
    "C-12": "compressed_air_downtime_per_hour",
}


def fmt_inr(x: float) -> str:
    """₹12,34,567 style (lakh/crore-friendly grouping)."""
    x = int(round(x))
    s = str(x)
    if len(s) <= 3:
        return f"₹{s}"
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return "₹" + ",".join(parts) + "," + tail


@dataclass
class CostLine:
    label: str
    amount: float
    basis: str          # the stated assumption behind the number


def downtime_cost(equipment: str, hours: float) -> CostLine:
    """Cost of this asset being down for N hours."""
    key = _EQUIP_RATE.get(equipment, "compressed_air_downtime_per_hour")
    rate = ASSUMPTIONS[key]
    return CostLine(
        label=f"{equipment} down for {hours:g} h",
        amount=rate * hours,
        basis=f"{key} = {fmt_inr(rate)}/h",
    )


def incident_cost(equipment: str, downtime_hours: float,
                  parts: bool = True) -> list[CostLine]:
    """Documented-incident cost build-up (e.g. the FD-2 trip, INC-2026-11)."""
    lines = [downtime_cost(equipment, downtime_hours)]
    if parts:
        lines.append(CostLine("Bearing replacement",
                              ASSUMPTIONS["bearing_replacement"],
                              "bearing_replacement"))
    lines.append(CostLine("Emergency callout premium",
                          ASSUMPTIONS["emergency_callout_premium"],
                          "emergency_callout_premium"))
    lines.append(CostLine("Investigation & retraining",
                          ASSUMPTIONS["near_miss_investigation"],
                          "near_miss_investigation"))
    return lines


def prevention_value(chunks=None) -> dict[str, tuple[float, str]]:
    """
    Headline avoided-cost figures — each tied to a documented event, and (when
    `chunks` is passed) shown ONLY if that event's source document is actually
    in the current corpus. Upload a different industry's documents and figures
    for absent incidents disappear — the KPI can never claim events the corpus
    doesn't contain. These are scenario ESTIMATES from the editable ASSUMPTIONS
    table, not measured savings.
    """
    fd2 = sum(l.amount for l in incident_cost("FD-2", 3.5))
    p7 = (ASSUMPTIONS["near_miss_investigation"]
          + ASSUMPTIONS["lti_direct_cost"] * 0.1)   # 10% injury probability
    audit = ASSUMPTIONS["audit_noncompliance_penalty"]
    events = {
        "FD-2 trip prevented (INC-2026-11 replay)":
            (fd2, "3.5 h steam loss + bearing + callout + investigation",
             "INC-2026-11"),
        "P-7 water-hammer near-miss not repeated (NM-2026-03)":
            (p7, "investigation cost + 10% weighted injury exposure",
             "NM-2026-03"),
        "Audit penalty exposure closed (LEL / SOP gaps)":
            (audit, "typical statutory non-compliance penalty",
             "HW-2026-014"),
    }
    if chunks is None:
        return {k: (v, basis) for k, (v, basis, _) in events.items()}
    corpus = " ".join(c.text for c in chunks)
    return {k: (v, basis) for k, (v, basis, marker) in events.items()
            if marker in corpus}
