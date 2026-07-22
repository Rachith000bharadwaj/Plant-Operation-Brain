# © 2026 Plant Operations Brain Team — ET AI Hackathon. All rights reserved.
# Proprietary & confidential. Unauthorised copying, reuse, or redistribution is prohibited (see LICENSE).
"""
i18n.py
-------
Whole-UI translation. Every visible label passes through `t()`; answers and voice
are handled by the LLM layer. Translations are batch-generated ONCE and cached to
disk (data/i18n/<lang>.json), so at runtime the UI switches language instantly
with ZERO API calls — it can never stall or fail on quota during a demo.

English is the source language (pass-through). Emojis, equipment tags (P-7),
IDs and [Source N] markers are preserved by the translation prompt.
"""

from __future__ import annotations
import json
from pathlib import Path

LANGUAGES = ["English", "Hindi", "Kannada", "Tamil", "Telugu", "Marathi", "Bengali"]
CACHE_DIR = Path("data/i18n")

_mem: dict[str, dict] = {}


def _load(lang: str) -> dict:
    if lang not in _mem:
        f = CACHE_DIR / f"{lang}.json"
        _mem[lang] = (json.loads(f.read_text(encoding="utf-8"))
                      if f.exists() else {})
    return _mem[lang]


def _save(lang: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{lang}.json").write_text(
        json.dumps(_mem[lang], ensure_ascii=False, indent=0), encoding="utf-8")


def t(text: str, lang: str = "English") -> str:
    """Translate a UI string to `lang` using the on-disk cache (no API at runtime)."""
    if lang == "English" or not text or not text.strip():
        return text
    return _load(lang).get(text, text)   # fall back to English if not cached


def td(text: str, lang: str = "English") -> str:
    """
    Translate DYNAMIC data (sensor names, asset names, risk levels…) that isn't
    a fixed UI label. Uses a separate on-disk cache; if a value isn't cached yet
    it translates once (LLM) and stores it, keeping any equipment tag/number/unit
    intact. Cached values are instant and API-free thereafter.
    """
    if lang == "English" or not text or not str(text).strip():
        return text
    text = str(text)
    cache = _load(f"_dyn_{lang}")
    if text in cache:
        return cache[text]
    try:
        from .llm import chat
        out = chat(
            f"Translate this short industrial data label to {lang}. Keep "
            f"equipment tags (P-7, FD-2), numbers, units (°C, bar, mm/s, A) and "
            f"any parenthesised code EXACTLY as-is; translate only the words. "
            f"Output ONLY the translation.",
            text, max_tokens=80).strip()
        cache[text] = out or text
    except Exception:
        cache[text] = text          # graceful English fallback (never blocks UI)
    _save(f"_dyn_{lang}")
    return cache[text]


def prewarm_dynamic(lang: str, values: list[str]) -> int:
    """Pre-translate a batch of dynamic values so the demo needs no runtime API."""
    if lang == "English":
        return 0
    cache = _load(f"_dyn_{lang}")
    missing = [v for v in {str(x) for x in values if x} if v not in cache]
    if not missing:
        return 0
    from .llm import chat
    done = 0
    for i in range(0, len(missing), 25):
        chunk = missing[i:i + 25]
        prompt = (f"Translate each string in this JSON array to {lang}. Keep "
                  f"equipment tags (P-7, FD-2), numbers, units and parenthesised "
                  f"codes EXACTLY as-is; translate only words. Return ONLY a JSON "
                  f"array of the same length, same order.")
        try:
            raw = chat(prompt, json.dumps(chunk, ensure_ascii=False), max_tokens=2000)
            arr = json.loads(raw[raw.find("["):raw.rfind("]") + 1])
            if len(arr) == len(chunk):
                for s, tr in zip(chunk, arr):
                    cache[s] = tr if isinstance(tr, str) and tr.strip() else s
                    done += 1
        except Exception:
            _save(f"_dyn_{lang}")
            break
    _save(f"_dyn_{lang}")
    return done


def answer_in(language: str) -> str:
    """
    A system-prompt suffix that makes ANY agent respond in the chosen language,
    so RCA/compliance/lessons/safety/impact/handover all speak the user's
    language — not just the copilot. Keeps tags/IDs/citations intact.
    """
    if not language or language == "English":
        return ""
    return (f"\n\nIMPORTANT: Write your ENTIRE response in {language}. Keep "
            f"equipment tags (like P-7, FD-2), IDs, numbers, units, and "
            f"[Source N] citation markers exactly as-is. Do not use English "
            f"for the prose.")


def prewarm(lang: str, strings: list[str]) -> int:
    """
    Batch-translate any strings not yet cached for `lang` (uses the LLM; run
    ONCE to build the cache). Returns how many were newly translated.
    """
    if lang == "English":
        return 0
    from .llm import chat
    cache = _load(lang)
    missing = [s for s in dict.fromkeys(strings) if s and s.strip() and s not in cache]
    if not missing:
        return 0
    done = 0
    for i in range(0, len(missing), 30):        # chunk to keep prompts sane
        chunk = missing[i:i + 30]
        prompt = (
            f"Translate each string in this JSON array into {lang} for an "
            f"industrial software UI. Rules: keep it concise; KEEP unchanged all "
            f"emojis, equipment tags (like P-7, FD-2), IDs, numbers, units, and "
            f"[Source N] markers; translate only the words. Return ONLY a JSON "
            f"array of the SAME length in the SAME order.")
        try:
            raw = chat(prompt, json.dumps(chunk, ensure_ascii=False), max_tokens=3000)
            arr = json.loads(raw[raw.find("["):raw.rfind("]") + 1])
            if len(arr) == len(chunk):
                for s, tr in zip(chunk, arr):
                    cache[s] = tr if isinstance(tr, str) and tr.strip() else s
                    done += 1
        except Exception:
            # Do NOT poison the cache with English on failure — leave the
            # strings missing so a later run (fresh quota) retries them.
            _save(lang)
            break
    _save(lang)
    return done


def cached_languages() -> list[str]:
    """Languages that already have a (non-empty) cache on disk — safe to offer."""
    ready = ["English"]
    for lang in LANGUAGES:
        if lang == "English":
            continue
        f = CACHE_DIR / f"{lang}.json"
        if f.exists() and json.loads(f.read_text(encoding="utf-8")):
            ready.append(lang)
    return ready
