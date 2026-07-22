# © 2026 Plant Operations Brain Team — ET AI Hackathon. All rights reserved.
# Proprietary & confidential. Unauthorised copying, reuse, or redistribution is prohibited (see LICENSE).
"""
llm.py
------
One small layer that all our modules call to talk to a Large Language Model.

Why this exists: we want to develop for FREE on Google Gemini now, and switch to
Claude for the final demo later -- without editing any other file. You change the
provider in ONE place: the .env file.

    LLM_PROVIDER=gemini   # free, for development   (needs GEMINI_API_KEY)
    LLM_PROVIDER=claude   # best quality, for demo  (needs ANTHROPIC_API_KEY)

Every module calls `chat(system, user)` and gets back a string. That's it.
"""

from __future__ import annotations
import os
import json
import time
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").lower()

# Gemini free tier gives each model its OWN small daily quota. So we keep a
# fallback chain: when one model hits its daily cap, we automatically try the
# next. "lite" models have the most generous free daily limits, so they go first;
# the stronger gemini-2.5-flash is kept as a higher-quality fallback.
GEMINI_MODELS = [
    m.strip() for m in os.environ.get(
        "GEMINI_MODELS",
        "gemini-2.5-flash-lite,gemini-3.1-flash-lite,gemini-2.5-flash",
    ).split(",") if m.strip()
]
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-4-8")

# Local model via Ollama — unlimited, free, offline, runs on the local GPU.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Groq — free tier with generous limits (OpenAI-compatible API).
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# When the primary provider is exhausted (daily quota), try these in order.
# Example .env:  FALLBACK_PROVIDERS=groq,ollama
FALLBACK_PROVIDERS = [p.strip().lower() for p in
                      os.environ.get("FALLBACK_PROVIDERS", "").split(",")
                      if p.strip()]


# ---------------- Gemini (free) ----------------
@lru_cache(maxsize=1)
def _gemini_client():
    from google import genai
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "LLM_PROVIDER=gemini but GEMINI_API_KEY is missing in .env. "
            "Get a free key at https://aistudio.google.com/"
        )
    return genai.Client(api_key=key)


def _is_daily_exhausted(err: Exception) -> bool:
    """A per-DAY quota cap means this model is done until tomorrow -> switch model."""
    m = str(err).lower().replace(" ", "")
    return "perday" in m or "limit:0" in m


def _gemini_one(model: str, system: str, user: str, max_tokens: int) -> str:
    from google.genai import types
    client = _gemini_client()

    def _call(disable_thinking: bool):
        # temperature 0 -> deterministic: the same question yields the same
        # answer every time (a plant tool must not "sometimes" answer).
        kwargs = dict(system_instruction=system,
                      max_output_tokens=max_tokens, temperature=0.0)
        # 2.5/3.x "flash" are thinking models; disabling thinking saves quota
        # and guarantees the answer fits the token budget.
        if disable_thinking:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        resp = client.models.generate_content(
            model=model, contents=user,
            config=types.GenerateContentConfig(**kwargs),
        )
        return (resp.text or "").strip()

    try:
        return _call(disable_thinking=True)
    except Exception as e:
        if "thinking" in str(e).lower():     # model rejects thinking_config
            return _call(disable_thinking=False)
        raise


def _gemini(system: str, user: str, max_tokens: int) -> str:
    """Try each model in the chain; skip capped OR empty-returning models."""
    last_err = None
    for model in GEMINI_MODELS:
        try:
            out = _gemini_one(model, system, user, max_tokens)
            if out and out.strip():
                return out
            # Empty text (safety block, or a 'thinking' model that spent its
            # whole budget, or a weak model that couldn't render the target
            # language) — treat as a failure and try the next model.
            last_err = RuntimeError(f"{model} returned empty text")
            continue
        except Exception as e:
            last_err = e
            if _is_daily_exhausted(e):
                continue          # this model is capped for today -> next model
            raise                 # transient (503 / per-minute) -> let chat() retry
    raise last_err               # every model exhausted / empty for the day


# ---------------- Claude (best quality) ----------------
@lru_cache(maxsize=1)
def _claude_client():
    from anthropic import Anthropic
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "LLM_PROVIDER=claude but ANTHROPIC_API_KEY is missing in .env."
        )
    return Anthropic(api_key=key)


def _claude(system: str, user: str, max_tokens: int) -> str:
    client = _claude_client()
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text.strip()


# ---------------- Ollama (local, unlimited, offline) ----------------
def _ollama(system: str, user: str, max_tokens: int) -> str:
    """Local model on the GPU via Ollama — no key, no quota, works offline."""
    import urllib.request
    body = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": user,
        "system": system,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": max_tokens},
    }).encode()
    req = urllib.request.Request(f"{OLLAMA_HOST}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read()).get("response", "").strip()


# ---------------- Groq (free tier, very fast) ----------------
@lru_cache(maxsize=1)
def _groq_client():
    from openai import OpenAI          # Groq is OpenAI-API-compatible
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("Groq selected but GROQ_API_KEY missing in .env. "
                           "Get a free key at https://console.groq.com/keys")
    return OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")


def _groq(system: str, user: str, max_tokens: int) -> str:
    resp = _groq_client().chat.completions.create(
        model=GROQ_MODEL, max_tokens=max_tokens, temperature=0.2,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    return (resp.choices[0].message.content or "").strip()


_PROVIDER_FN = {"gemini": _gemini, "claude": _claude,
                "ollama": _ollama, "groq": _groq}


# ---------------- public entry point ----------------
_TRANSIENT = ("503", "unavailable", "overloaded", "high demand",
              "429", "resource_exhausted", "rate limit", "500", "internal")


def _is_transient(err: Exception) -> bool:
    msg = str(err).lower()
    # A hard "limit: 0" quota means retrying is pointless -- treat as permanent.
    if "limit: 0" in msg:
        return False
    return any(t in msg for t in _TRANSIENT)


def _is_quota(err: Exception) -> bool:
    m = str(err).lower()
    return any(x in m for x in ("quota", "resource_exhausted", "insufficient_quota",
                                "limit: 0", "perday", "exhausted"))


def chat(system: str, user: str, max_tokens: int = 800, retries: int = 4) -> str:
    """
    Send a system + user prompt to the configured provider; return text.

    Reliability ladder:
      1. Retry transient errors (503/rate-limit) with exponential backoff.
      2. If the primary provider's quota is exhausted, automatically fail over
         to FALLBACK_PROVIDERS (e.g. groq, ollama) so the app never goes dark.
    """
    providers = [PROVIDER] + [p for p in FALLBACK_PROVIDERS if p != PROVIDER]
    last_err = None

    for prov in providers:
        fn = _PROVIDER_FN.get(prov)
        if fn is None:
            continue
        delay = 2.0
        for attempt in range(retries):
            try:
                out = fn(system, user, max_tokens)
                if out and out.strip():
                    return out                 # a real, non-empty answer
                last_err = RuntimeError(f"{prov} returned empty text")
                break                          # empty -> next provider
            except Exception as e:
                last_err = e
                if _is_quota(e):
                    break                      # quota gone -> next provider
                if attempt == retries - 1 or not _is_transient(e):
                    break                      # give up on this provider
                time.sleep(delay)
                delay = min(delay * 2, 20)
    raise last_err if last_err else RuntimeError("No LLM provider available.")


def vision(prompt: str, image_bytes: bytes, mime: str = "image/jpeg",
           max_tokens: int = 400) -> str:
    """
    Read an image (a gauge, nameplate, or equipment photo) and answer `prompt`
    about it. Lets a floor technician snap a photo instead of typing a tag.
    """
    if PROVIDER == "gemini":
        from google.genai import types
        client = _gemini_client()
        last_err = None
        for model in GEMINI_MODELS:           # same daily-cap fallback as text
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type=mime),
                        prompt,
                    ],
                    config=types.GenerateContentConfig(max_output_tokens=max_tokens),
                )
                return (resp.text or "").strip()
            except Exception as e:
                last_err = e
                if _is_daily_exhausted(e):
                    continue
                raise
        raise last_err
    if PROVIDER == "claude":
        import base64
        client = _claude_client()
        resp = client.messages.create(
            model=CLAUDE_MODEL, max_tokens=max_tokens,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                 "media_type": mime,
                 "data": base64.standard_b64encode(image_bytes).decode()}},
                {"type": "text", "text": prompt},
            ]}],
        )
        return resp.content[0].text.strip()
    raise ValueError(f"Vision not supported for provider {PROVIDER!r}")


def transcribe(audio_bytes: bytes, mime: str = "audio/wav") -> str:
    """
    Convert spoken audio to text. Uses Gemini (which supports audio input) even
    if the chat provider is Claude, since transcription is a Gemini capability.
    Lets a technician speak a question instead of typing.
    """
    from google.genai import types
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("Voice transcription needs GEMINI_API_KEY in .env.")
    client = _gemini_client()
    prompt = ("Transcribe this audio to plain text in its ORIGINAL language "
              "and script (Kannada in Kannada script, Hindi in Devanagari, "
              "etc.). Return ONLY the transcript.")
    last_err = None
    for model in GEMINI_MODELS:                # same daily-cap fallback
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=audio_bytes, mime_type=mime),
                    prompt,
                ],
                config=types.GenerateContentConfig(max_output_tokens=200),
            )
            return (resp.text or "").strip()
        except Exception as e:
            last_err = e
            if _is_daily_exhausted(e):
                continue
            raise
    raise last_err


def provider_status() -> str:
    """Human-readable status for the UI sidebar."""
    if PROVIDER == "gemini":
        ok = bool(os.environ.get("GEMINI_API_KEY"))
        return f"Gemini ({GEMINI_MODELS[0]}) — key {'set ✅' if ok else 'MISSING ❌'}"
    if PROVIDER == "claude":
        ok = bool(os.environ.get("ANTHROPIC_API_KEY"))
        return f"Claude ({CLAUDE_MODEL}) — key {'set ✅' if ok else 'MISSING ❌'}"
    return f"Unknown provider: {PROVIDER}"
