# © 2026 Plant Operations Brain Team — ET AI Hackathon. All rights reserved.
# Proprietary & confidential. Unauthorised copying, reuse, or redistribution is prohibited (see LICENSE).
"""
sensors.py
----------
Simulated live IoT feed + proactive alerting — with a self-configuring watchlist.

Real plants have SCADA historians; for the demo we simulate realistic trends.
The advanced part: the watchlist is NOT hardcoded. `discover_sensors()` scans
every ingested document for written operating limits ("Bearing temperature:
75°C (alarm), 90°C (trip)") and builds sensors for them automatically. Upload a
new equipment manual -> rebuild -> that equipment appears on the watch, with
its documented limits. The manuals *are* the configuration.

A small curated set is kept for the core demo assets; discovered sensors merge
in around it. The simulation is deterministic (seeded) so demos are repeatable.
"""

from __future__ import annotations
import math
import re
import random
from dataclasses import dataclass

from .rag import PlantBrain, _format_excerpts
from .llm import chat

# Curated sensors for the flagship demo assets (limits exactly as documented).
SENSORS = {
    "P-7 bearing temperature (°C)": {
        "equipment": "P-7", "alarm": 85.0, "trip": 95.0,
        "base": 68.0, "drift": 0.22, "noise": 1.6,   # rising -> will alert
        "source": "curated",
    },
    "P-7 vibration (mm/s)": {
        "equipment": "P-7", "alarm": 4.5, "trip": 7.1,
        "base": 2.4, "drift": 0.018, "noise": 0.25,
        "source": "curated",
    },
    "C-12 discharge temperature (°C)": {
        "equipment": "C-12", "alarm": 110.0, "trip": 130.0,
        "base": 96.0, "drift": 0.05, "noise": 2.0,   # stays healthy
        "source": "curated",
    },
    "B-3 steam pressure (bar)": {
        "equipment": "B-3", "alarm": 10.0, "trip": 10.5,
        "base": 9.1, "drift": 0.0, "noise": 0.15,    # stays healthy
        "source": "curated",
    },
}

# "Bearing temperature: 75°C (alarm), 90°C (trip)"  /  "110°C alarm, 130°C trip"
_LIMIT_RE = re.compile(
    r"([A-Za-z][A-Za-z /-]{2,40}?):\s*"
    r"([\d.]+)\s*(°?C|mm/s|bar|A)\s*\(?alarm\)?\s*[,;]?\s*"
    r"([\d.]+)\s*(?:°?C|mm/s|bar|A)?\s*\(?trip\)?",
    re.IGNORECASE,
)
_TAG_RE = re.compile(r"\b((?:P|C|B|E|T|V|HX|FD)-\d+)\b")


def discover_sensors(chunks) -> dict:
    """
    Scan ingested chunks for documented alarm/trip limit pairs and turn each
    into a simulated sensor. Returns {sensor_name: cfg} for limits that are
    not already covered by the curated set.
    """
    found: dict[str, dict] = {}
    known = {(c["equipment"], c["alarm"], c["trip"]) for c in SENSORS.values()}

    for chunk in chunks:
        tag_m = _TAG_RE.search(chunk.text)
        if not tag_m:
            continue
        tag = tag_m.group(1)
        for m in _LIMIT_RE.finditer(chunk.text):
            metric, alarm_s, unit, trip_s = m.group(1), m.group(2), m.group(3), m.group(4)
            try:
                alarm, trip = float(alarm_s), float(trip_s)
            except ValueError:
                continue
            if trip <= alarm:                       # nonsense pair -> skip
                continue
            if (tag, alarm, trip) in known:          # curated already covers it
                continue
            unit = unit if unit.startswith("°") or unit != "C" else "°C"
            name = f"{tag} {metric.strip().lower()} ({unit})"
            if name in found:
                continue
            base = round(alarm - 0.25 * (trip - alarm), 2)
            # Deterministic per-sensor drift: hits the alarm somewhere in the
            # 60-150h window of the simulation, so demos stay interesting.
            h = hash(name) % 4
            drift = (alarm - base) / (60 + 30 * h)
            found[name] = {
                "equipment": tag, "alarm": alarm, "trip": trip,
                "base": base, "drift": round(drift, 4),
                "noise": round(max(alarm * 0.015, 0.05), 3),
                "source": chunk.source_file,        # provenance: which document
            }
    return found


def get_registry(chunks=None) -> dict:
    """Curated sensors + everything discovered in the documents."""
    reg = dict(SENSORS)
    if chunks:
        reg.update(discover_sensors(chunks))
    return reg


def series(name: str, hours: int, registry: dict | None = None,
           drift_scale: float = 1.0) -> list[float]:
    """
    Deterministic simulated history for the first `hours` hours.
    `drift_scale` scales the upward trend: 1.0 = normal (drifts toward the
    limit), 0.0 = a healthy week (flat, noise-only) — so the demo can show
    BOTH a degrading asset and a stable one, not just a scripted alarm.
    """
    cfg = (registry or SENSORS)[name]
    rng = random.Random(hash(name) & 0xFFFF)
    vals = []
    for t in range(hours + 1):
        wave = math.sin(t / 6.0) * cfg["noise"] * 0.6      # slow daily wave
        noise = rng.uniform(-cfg["noise"], cfg["noise"]) * 0.5
        vals.append(round(cfg["base"] + cfg["drift"] * drift_scale * t
                          + wave + noise, 2))
    return vals


@dataclass
class SensorStatus:
    name: str
    equipment: str
    value: float
    alarm: float
    trip: float
    state: str        # OK / WARNING / ALARM
    source: str = "curated"


def status_at(name: str, hours: int, registry: dict | None = None,
              drift_scale: float = 1.0) -> SensorStatus:
    cfg = (registry or SENSORS)[name]
    value = series(name, hours, registry, drift_scale)[-1]
    margin = max((cfg["alarm"] - cfg["base"]) * 0.25, 1e-9)
    if value >= cfg["alarm"]:
        state = "ALARM"
    elif value >= cfg["alarm"] - margin:
        state = "WARNING"                    # trending toward the limit
    else:
        state = "OK"
    return SensorStatus(name, cfg["equipment"], value,
                        cfg["alarm"], cfg["trip"], state,
                        cfg.get("source", "curated"))


_ALERT_SYSTEM = """You are the plant's proactive safety intelligence. A live \
sensor is trending toward its documented limit. Using ONLY the excerpts, write a \
SHORT alert (max 120 words) for the shift supervisor:
- what is happening and how close to alarm/trip it is,
- WHY this specific equipment is risky (documented failure history, RCAs),
- the ONE most important action to take now.
Cite [Source N]. Be direct — this will be read on a phone during a shift."""


def explain_alert(brain: PlantBrain, st_: SensorStatus) -> tuple[str, list]:
    """Fuse the live reading with the equipment's documented history."""
    hits = brain.retrieve(
        f"{st_.equipment} failure history root cause operating limits "
        f"temperature vibration trip alarm", k=5)
    excerpts = _format_excerpts(hits)
    user = (f"LIVE READING: {st_.name} = {st_.value} "
            f"(alarm {st_.alarm}, trip {st_.trip}, state {st_.state})\n\n"
            f"Document excerpts:\n{excerpts}\n\nWrite the alert.")
    return chat(_ALERT_SYSTEM, user, max_tokens=300), [c for c, _ in hits]
