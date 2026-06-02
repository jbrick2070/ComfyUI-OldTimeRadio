"""Deterministic per-line delivery (emotion) vector.

The OTR ledger carries no per-line affect today (DramaticState is episode-level
plot; line-composer hints are prompt-time only). An expressive voice engine
(IndexTTS2 Emo-Vector, Chatterbox exaggeration) needs a per-line delivery
signal. This module derives one in pure Python -- keyword cues + scene tension
-> an 8-dim vector in IndexTTS2's emotion order -- with no LLM and no RNG, so it
is C7-byte-deterministic and adds no PD6 LLM-call burden.

It is stamped *additively* onto the ledger after freeze. The default Bark path
ignores the field, so the byte-identical baseline is unaffected.
"""
from __future__ import annotations

import re

DELIVERY_TABLE_VERSION = "v1"

# IndexTTS2 Emo-Vector order. Keep this tuple and its order stable -- it is the
# canonical contract every voice adapter projects from.
EMOTIONS = (
    "happy",
    "angry",
    "sad",
    "afraid",
    "disgusted",
    "melancholic",
    "surprised",
    "calm",
)

# Distinctive whole-word / phrase cues per emotion (SFW, non-violent). Single
# tokens match on word boundaries; entries containing a space match as a
# substring so short function words never false-trigger (e.g. "no" in "know").
_KEYWORDS = {
    "happy": {
        "laugh", "laughing", "wonderful", "glad", "delight", "delighted",
        "joy", "joyful", "hope", "marvelous", "splendid", "cheer", "yes",
    },
    "angry": {
        "fool", "never", "enough", "liar", "betrayed", "furious", "rage",
        "traitor", "coward", "how dare", "get out",
    },
    "sad": {
        "sorry", "lost", "gone", "alone", "tears", "mourn", "grief",
        "farewell", "weep",
    },
    "afraid": {
        "help", "run", "danger", "terrified", "hide", "afraid", "flee",
        "scream", "no escape", "behind you",
    },
    "disgusted": {"sick", "vile", "rot", "foul", "revolting", "stench", "filth"},
    "melancholic": {
        "remember", "once", "faded", "empty", "distant", "silence", "ago",
        "memory",
    },
    "surprised": {
        "what", "impossible", "suddenly", "look", "behold", "who", "how",
        "wait",
    },
}

_AROUSAL = ("angry", "afraid", "surprised", "happy")
_CAP = 3.0  # soft cap: this many cues for a non-calm dim saturates at 1.0


def deterministic_delivery_vector(text: str, scene_tension: float = 0.0) -> dict:
    """Return an 8-dim ``{emotion: 0.0..1.0}`` for one line of dialogue.

    Deterministic in ``(text, scene_tension)``; no RNG, no model.
    """
    t = (text or "").lower()
    words = set(re.findall(r"[a-z']+", t))
    scores = {e: 0.0 for e in EMOTIONS}

    for emo, cues in _KEYWORDS.items():
        for cue in cues:
            if " " in cue:
                if cue in t:
                    scores[emo] += 1.0
            elif cue in words:
                scores[emo] += 1.0

    ex = t.count("!")
    q = t.count("?")
    ell = t.count("...")
    scores["surprised"] += 0.5 * q + 0.3 * ex
    scores["angry"] += 0.4 * ex
    scores["afraid"] += 0.3 * ex
    scores["happy"] += 0.2 * ex
    scores["melancholic"] += 0.5 * ell
    scores["sad"] += 0.3 * ell

    st = max(0.0, min(1.0, float(scene_tension)))
    scores["afraid"] += st
    scores["angry"] += 0.6 * st
    scores["surprised"] += 0.4 * st

    arousal = sum(scores[e] for e in _AROUSAL)
    calm = max(0.0, 1.0 - 0.5 * arousal - st)

    out = {}
    for e in EMOTIONS:
        if e == "calm":
            out[e] = round(min(1.0, calm), 3)
        else:
            out[e] = round(min(1.0, scores[e] / _CAP), 3)
    return out


def _iter_dialogue_lines(ledger: dict):
    """Yield mutable line dicts from the OTR ledger ``lines`` list."""
    for line in ledger.get("lines") or []:
        if isinstance(line, dict):
            yield line


def stamp_delivery_vectors(ledger: dict) -> dict:
    """Stamp ``line['delivery'] = {emotion_vector, version}`` on every line.

    Additive -- the default voice path ignores it, so the byte-identical
    baseline is unaffected. Returns the same ledger object for chaining.
    """
    for line in _iter_dialogue_lines(ledger):
        tension = line.get("scene_tension", line.get("tension", 0.0)) or 0.0
        vec = deterministic_delivery_vector(line.get("text", ""), float(tension))
        delivery = line.setdefault("delivery", {})
        delivery["emotion_vector"] = vec
        delivery["version"] = DELIVERY_TABLE_VERSION
    return ledger
