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

import logging
import re

log = logging.getLogger("OTR")

#: v2 (2026-06-11, whiny-fix plan v3.1 P1.1): derived from PREPARED text
#: (callers pass the shared-normalized line); stop-word interrogative cues
#: demoted; punctuation counted ONCE per line (presence, not count -- kills the
#: G6 keyword+punctuation double-dip); de-bleat rule (terminal-? + afraid/sad
#: demotes surprised); total-mass strategy = primary-keeps-full +
#: cap-secondaries (PROVISIONAL pending the operator's P0 audition matrix);
#: NO unconditional calm floor. v1 stamps in old ledgers are ignored by the
#: version guard in :func:`select_delivery_vector` (pass-3 must-fix).
DELIVERY_TABLE_VERSION = "v2"

#: Secondary non-calm dims are capped at this once the primary is chosen
#: (the provisional P1.1 mass strategy; re-anchor from P0 data when it lands).
_SECONDARY_CAP = 0.5

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
    # v2: stop-word interrogatives DEMOTED (G6: "what"/"who"/"how"/"wait"/
    # "look" fire on ordinary dialogue and biased the vector whiny-adjacent).
    # Only genuinely-startled cues remain.
    "surprised": {
        "impossible", "suddenly", "behold", "astonishing", "incredible",
    },
}

_AROUSAL = ("angry", "afraid", "surprised", "happy")
_CAP = 3.0  # soft cap: this many cues for a non-calm dim saturates at 1.0


def deterministic_delivery_vector(text: str, scene_tension: float = 0.0) -> dict:
    """Return an 8-dim ``{emotion: 0.0..1.0}`` for one line of dialogue.

    Deterministic in ``(text, scene_tension)``; no RNG, no model. v2 contract
    (see DELIVERY_TABLE_VERSION note): callers pass PREPARED text; punctuation
    counts once; de-bleat demotes surprised on fearful/sad terminal questions;
    primary keeps full strength, secondaries capped.
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

    # v2: punctuation contributes ONCE per line (presence, not count) -- the
    # G6 fix for "?"-dense lines saturating surprised + the keyword double-dip.
    ex = 1.0 if "!" in t else 0.0
    q = 1.0 if "?" in t else 0.0
    ell = 1.0 if "..." in t else 0.0
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

    # v2 de-bleat rule: a fearful/sad TERMINAL question reads pleading when
    # surprised rides along -- demote surprised, keep the primary affect.
    if t.rstrip().endswith("?") and (scores["afraid"] > 0.0 or scores["sad"] > 0.0):
        scores["surprised"] *= 0.5

    arousal = sum(scores[e] for e in _AROUSAL)
    calm = max(0.0, 1.0 - 0.5 * arousal - st)

    out = {}
    for e in EMOTIONS:
        if e == "calm":
            out[e] = round(min(1.0, calm), 3)
        else:
            out[e] = round(min(1.0, scores[e] / _CAP), 3)

    # v2 total-mass strategy (PROVISIONAL pending P0): the strongest non-calm
    # dim keeps its full value; every other non-calm dim is capped so keyword
    # soup can never stack three+ strong emotions onto one line.
    noncalm = [e for e in EMOTIONS if e != "calm"]
    if any(out[e] > 0.0 for e in noncalm):
        primary = max(noncalm, key=lambda e: (out[e], e))
        for e in noncalm:
            if e != primary and out[e] > _SECONDARY_CAP:
                out[e] = _SECONDARY_CAP
    return out


def select_delivery_vector(line: dict, prepared_text: str,
                           scene_tension: float = 0.0):
    """The ONE per-line vector chooser (consumer-side; whiny-fix P1.1).

    Returns ``(vector_dict, source_tag)``. Prefers a STAMPED
    ``line['delivery']['emotion_vector']`` ONLY when its stamped ``version``
    matches :data:`DELIVERY_TABLE_VERSION` (the pass-3 must-fix: a stale
    v1-stamped ledger must not silently defeat the v2 retune); otherwise
    re-derives from the PREPARED text + scene tension, LOUDLY when a stale
    stamp was bypassed.
    """
    dl = line.get("delivery") if isinstance(line, dict) else None
    stamped = dl.get("emotion_vector") if isinstance(dl, dict) else None
    if isinstance(stamped, dict) and stamped:
        ver = str((dl or {}).get("version") or "")
        if ver == DELIVERY_TABLE_VERSION:
            return stamped, f"stamped:{ver}"
        log.warning(
            "[OTR delivery] stamped vector version %r != active table %r on "
            "line %r; RE-DERIVING from prepared text (stale stamp ignored).",
            ver, DELIVERY_TABLE_VERSION,
            (line or {}).get("line_id") or (line or {}).get("text", "")[:40],
        )
        tag = f"rederived_stale_stamp:{ver or 'unversioned'}"
        return deterministic_delivery_vector(prepared_text, scene_tension), tag
    return deterministic_delivery_vector(prepared_text, scene_tension), "derived"


def _iter_dialogue_lines(ledger: dict):
    """Yield mutable line dicts from the OTR ledger ``lines`` list."""
    for line in ledger.get("lines") or []:
        if isinstance(line, dict):
            yield line
