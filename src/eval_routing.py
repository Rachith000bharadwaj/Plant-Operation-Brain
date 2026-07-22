# © 2026 Plant Operations Brain Team — ET AI Hackathon. All rights reserved.
# Proprietary & confidential. Unauthorised copying, reuse, or redistribution is prohibited (see LICENSE).
"""
eval_routing.py
---------------
Measures the agentic ROUTER's accuracy — the % of questions dispatched to the
correct specialist agent. Runs the deterministic rule layer only (0 API calls),
so it's free and fully reproducible. This is the number behind the "agentic
routing accuracy" claim.

Run:  python -m src.eval_routing
"""

from __future__ import annotations
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from .orchestrator import route

# (question, expected_agent) — a labelled routing benchmark.
CASES = [
    # safety
    ("Is it safe to replace the P-7 seal with the discharge valve open?", "safety"),
    ("Can I do hot work near C-12 right now?", "safety"),
    ("I'm about to start maintenance on FD-2, any risks?", "safety"),
    ("Should I proceed with the boiler startup?", "safety"),
    ("Is it ok to lubricate P-7 while it is running?", "safety"),
    # rca
    ("Why does the P-7 bearing keep failing?", "rca"),
    ("What is the root cause of the FD-2 trips?", "rca"),
    ("Why is compressor C-12 overheating?", "rca"),
    ("Diagnose the recurring failure on P-7.", "rca"),
    ("What's causing the boiler B-3 shutdowns?", "rca"),
    # compliance
    ("Are we compliant with the Factory Act?", "compliance"),
    ("Run a compliance audit for the plant.", "compliance"),
    ("What are our OISD regulatory gaps?", "compliance"),
    ("Do we have audit evidence for PESO?", "compliance"),
    ("Show me non-conformances against IBR.", "compliance"),
    # lessons
    ("What lessons learned come from our incident history?", "lessons"),
    ("Are there recurring patterns across incidents?", "lessons"),
    ("What systemic issues show up in near-miss history?", "lessons"),
    ("What common causes appear across our failures?", "lessons"),
    # conflict
    ("Are there any contradictions between our documents?", "conflict"),
    ("Do the SOP and the OEM bulletin disagree?", "conflict"),
    ("Which source is correct on the lube interval?", "conflict"),
    ("Are any procedures outdated versus newer directives?", "conflict"),
    # impact
    ("What happens if FD-2 stops running?", "impact"),
    ("What is the impact of a P-7 failure?", "impact"),
    ("What breaks downstream if C-12 trips?", "impact"),
    ("Consequences of the boiler going down?", "impact"),
    # handover
    ("Give me the shift handover brief.", "handover"),
    ("Prepare a brief for the incoming shift.", "handover"),
    ("What should the next shift know?", "handover"),
    # deep
    ("Why does P-7 keep failing and what should we fix before the audit?", "deep"),
    ("Why is FD-2 tripping and how do we budget the repair?", "deep"),
    # copilot (plain lookups)
    ("What is the lockout procedure for Pump 7?", "copilot"),
    ("At what temperature does C-12 trip?", "copilot"),
    ("When is the boiler B-3 inspection due?", "copilot"),
    ("How often should FD-2 bearings be lubricated?", "copilot"),
    ("What is a bonafide certificate?", "copilot"),
    ("Who is responsible for updating SOP-BH-04?", "copilot"),

    # --- harder / varied phrasings (generalisation check) ---
    ("Before I open the C-12 valve cover, is there anything I should know?", "safety"),
    ("Talk me through the safety of doing seal work on P-7 now.", "safety"),
    ("what's the root cause behind the failure of the cooling pump", "rca"),
    ("Explain why B-3 keeps tripping.", "rca"),
    ("Are our procedures up to date with regulations?", "compliance"),
    ("Prepare audit evidence for the Factory Act.", "compliance"),
    ("Find recurring themes in our near-miss history.", "lessons"),
    ("Do any of our documents contradict each other?", "conflict"),
    ("If the FD-2 fan trips, what else goes down?", "impact"),
    ("Brief the oncoming shift on today's issues.", "handover"),
    ("Why does P-7 fail and what do we do about it before the shutdown?", "deep"),
    ("What is the maximum working pressure of boiler B-3?", "copilot"),
    ("List the maintenance history of P-7.", "copilot"),
    ("What does work order WO-228 say?", "copilot"),
    ("Define lockout tagout.", "copilot"),
    ("What temperature is the C-12 alarm set at?", "copilot"),
    ("Is it ok to run FD-2 with the old lubrication schedule?", "safety"),
    ("Show me the compliance gaps in our lockout procedures.", "compliance"),
    ("What lessons can we draw from past pump incidents?", "lessons"),
    ("Which SOP is correct, the old one or the updated directive?", "conflict"),
]


def run() -> None:
    ok = 0
    misroutes = []
    for q, want in CASES:
        r = route(q, use_llm_fallback=False)   # rules only = free + reproducible
        good = r.agent == want
        ok += good
        if not good:
            misroutes.append((q, want, r.agent))
    acc = ok / len(CASES)
    print(f"Routing accuracy (rules only): {ok}/{len(CASES)} = {acc:.1%}")
    if misroutes:
        print("\nMISROUTES:")
        for q, want, got in misroutes:
            print(f"  ✗ want {want:10} got {got:10} :: {q}")
    else:
        print("All routes correct.")


if __name__ == "__main__":
    run()
