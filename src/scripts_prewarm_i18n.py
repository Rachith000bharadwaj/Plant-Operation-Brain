"""One-time: collect UI strings from app.py and prewarm every language cache."""
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

src = open("app.py", encoding="utf-8").read()
strings = re.findall(r'T\(\s*"((?:[^"\\]|\\.)+)"', src)
strings += re.findall(r"T\(\s*'((?:[^'\\]|\\.)+)'", src)
strings += [
    "Language", "Continue →", "Answer found in documents",
    "No documented answer — the system refuses to guess.",
    "Searching documents and reasoning...", "Ask a question",
    "OK", "WARNING", "ALARM", "High", "Medium", "Low",
    "Healthy", "Stable", "Needs attention", "Critical",
    "Grade", "scenario estimate", "documented event",
    "Sensor", "Live value", "Alarm at", "Trip at", "State", "Limits from",
    "Equipment", "Hours into the shift simulation", "Scenario",
    "Degrading (default)", "Healthy week", "Trend view",
    "All sensors within documented limits.",
    "🧠 Explain this alert from the documents",
    "Planned work", "Preview / download a file",
    "Entities", "Relationships", "File", "Type", "Size (KB)", "Modified",
    "Indexed", "Asset to document", "Expert's answer", "Save answer",
    "💾 Save as knowledge document", "Retrieval confidence",
    "Routed to", "Reasoning trace — how the agent got here", "Sources",
    "Downtime to price (hours)", "Optional focus (e.g. 'lockout', 'P-7')",
    "Equipment tag", "Try an example (or write your own below)",
    "🔍 Preview / download a file",
    # entity types (Document Distribution chart), risk table, alert card
    "Regulation", "Person", "Procedure", "Parameter", "Incident", "Unknown",
    "Asset", "Risk Level", "hours", "entities", "now",
    "sensors on watch", "auto-discovered from documents", "curated",
    "Open actions", "Document Conflicts", "Pending Checklist Items",
    "items need operator review.", "Tools & options", "New chat",
    # sidebar navigation labels (passed as variables, not literal T("…"))
    "Navigate", "Dashboard", "Chat", "Operations", "Live Sensor Data",
    "Capture Interview", "History", "Change",
    "Chat history", "Recent chats", "No saved chats yet.", "Delete this chat",
    "Simulated live SCADA feed. The watchlist configures itself from the "
    "ingested documents: any alarm/trip limit written in a manual or SOP "
    "becomes a live sensor automatically.",
    "'Healthy week' shows the same assets stable — proof the readings aren't "
    "scripted to fail.",
]
strings = [s for s in dict.fromkeys(strings) if s.strip()]
print(f"{len(strings)} unique UI strings collected")

from src.i18n import prewarm, prewarm_dynamic, LANGUAGES

for lang in LANGUAGES:
    if lang == "English":
        continue
    n = prewarm(lang, strings)
    print(f"  {lang}: +{n} UI strings")

# --- Dynamic DATA values: sensor names + graph asset/entity names ---
print("Collecting dynamic data values from the live index...")
try:
    from src.embed_store import VectorStore
    from src.graph import KnowledgeGraph
    from src.sensors import get_registry
    store = VectorStore(); store.load("data/index")
    g = KnowledgeGraph(); g.load("data/index")
    dyn = list(get_registry(store.chunks).keys())          # sensor names
    dyn += [n for n, d in g.g.nodes(data=True)
            if d.get("type") == "Equipment"][:60]          # asset names
    dyn = list(dict.fromkeys(dyn))
    print(f"  {len(dyn)} dynamic values")
    for lang in LANGUAGES:
        if lang == "English":
            continue
        n = prewarm_dynamic(lang, dyn)
        print(f"  {lang}: +{n} data values")
except Exception as e:
    print(f"  (skipped dynamic prewarm: {e})")
print("done")
