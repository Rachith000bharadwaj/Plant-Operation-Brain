# © 2026 Plant Operations Brain Team — ET AI Hackathon. All rights reserved.
# Proprietary & confidential. Unauthorised copying, reuse, or redistribution is prohibited (see LICENSE).
"""
chat_store.py
-------------
Persistent chat history. Conversations are saved to disk (data/chat_history.json)
so they survive reruns, tab switches and app restarts. Only JSON-safe fields are
stored (role, text, and light metadata) — retrieval Chunk objects, audio bytes and
reasoning traces are dropped on save so the file stays small and always loadable.
"""

from __future__ import annotations
import json
import time
from pathlib import Path

_FILE = Path("data/chat_history.json")
_MAX = 50   # keep the most recent N conversations


def _load_all() -> list[dict]:
    if _FILE.exists():
        try:
            return json.loads(_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _write_all(records: list[dict]) -> None:
    _FILE.parent.mkdir(parents=True, exist_ok=True)
    _FILE.write_text(json.dumps(records, ensure_ascii=False, indent=0),
                     encoding="utf-8")


def _safe_turns(chat: list) -> list:
    """Strip in-memory turns down to JSON-serialisable role/text/metadata."""
    out = []
    for turn in chat:
        role, text = turn[0], turn[1]
        if role == "user":
            out.append(["user", text])
        else:
            extras = turn[3] if len(turn) > 3 else {}
            safe = {k: extras.get(k) for k in ("route", "score", "confident")
                    if extras.get(k) is not None}
            out.append(["assistant", text, [], safe])
    return out


def save_chat(chat_id: str, chat: list) -> None:
    """Insert/replace this conversation at the top of the history file."""
    if not chat:
        return
    title = next((t[1] for t in chat if t[0] == "user"), "Chat")
    title = (title or "Chat").strip()[:60]
    record = {"id": chat_id, "title": title, "ts": time.time(),
              "turns": _safe_turns(chat)}
    records = [c for c in _load_all() if c.get("id") != chat_id]
    records.insert(0, record)
    _write_all(records[:_MAX])


def list_chats() -> list[dict]:
    """All saved conversations, newest first: [{id, title, ts, turns}, …]."""
    return _load_all()


def get_chat(chat_id: str) -> list:
    """Reload one conversation as replayable (role, …) tuples."""
    for c in _load_all():
        if c.get("id") == chat_id:
            return [tuple(t) for t in c.get("turns", [])]
    return []


def delete_chat(chat_id: str) -> None:
    _write_all([c for c in _load_all() if c.get("id") != chat_id])
