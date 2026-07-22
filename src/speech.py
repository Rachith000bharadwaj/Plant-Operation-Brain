# © 2026 Plant Operations Brain Team — ET AI Hackathon. All rights reserved.
# Proprietary & confidential. Unauthorised copying, reuse, or redistribution is prohibited (see LICENSE).
"""
speech.py
---------
Offline text-to-speech: the copilot can SPEAK its answers.

Uses the OS speech engine via pyttsx3 (Windows SAPI5) — fully offline, zero API
cost, works even if the venue internet dies. Closes the hands-free loop for a
technician wearing gloves: speak a question (mic) -> hear the answer (this).
"""

from __future__ import annotations
import re
import tempfile
from pathlib import Path


def _clean_for_speech(text: str) -> str:
    """Strip markdown + citation brackets so the audio sounds natural."""
    text = re.sub(r"\[Source \d+\]", "", text)      # citations are visual
    text = re.sub(r"[#*_`>|-]+", " ", text)          # markdown noise
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def tts_wav(text: str, max_chars: int = 900) -> bytes | None:
    """
    Render `text` to WAV bytes with the OS voice. Returns None if no speech
    engine is available (the UI simply hides the audio player then).
    """
    try:
        import pyttsx3
    except Exception:
        return None

    speakable = _clean_for_speech(text)[:max_chars]
    if not speakable:
        return None

    try:
        engine = pyttsx3.init()                      # fresh engine per call
        engine.setProperty("rate", 175)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "answer.wav"
            engine.save_to_file(speakable, str(out))
            engine.runAndWait()
            engine.stop()
            return out.read_bytes() if out.exists() else None
    except Exception:
        return None
