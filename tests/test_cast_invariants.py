"""tests/test_cast_invariants.py -- the R1-R9 regression bar for the cast
name<->gender<->voice coherence sprint.

Green here (plus the cast-relevant suite wrapped by test_cast.sh) is the
per-sprint + wave gate. Hermetic: no GPU, no I/O beyond reading the golden
fixture, no ComfyUI imports. Every LLM call goes through a stub.

Sprint ownership of the bar:
  R1 determinism, R2 C7 byte-identity, R4 quota, R5 voice-uniqueness,
  R6 one-rng.choice/slot ......... land in S0 (this file, passing now).
  R3 coherence ................... added by S2 (the name-repair).
  R7 schema / R8 fallback ........ added by S6 / S7 (LLM naming).
  R9 freeze-order ................ added by S8 (two-pass writer).
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NODES_DIR = _REPO_ROOT / "nodes"
for _p in (_REPO_ROOT, _NODES_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _otr_casting as _OTRC  # noqa: E402
from config import cast_pools as _POOLS  # noqa: E402

_GOLDEN_PATH = Path(__file__).resolve().parent / "golden" / "cast_pool_baseline.json"


def _golden() -> dict:
    return json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))


def _stub(messages, *, temperature, max_new_tokens):  # noqa: ARG001
    return json.dumps({
        "character_description":
            "A steady, weathered field operator who keeps the crew grounded.",
    })


def _structural(cast):
    return [
        {"char_id": r["char_id"], "name": r["name"],
         "gender": r["gender"], "voice_preset": r["voice_preset"]}
        for r in cast
    ]


def _lock(seed, *, num=4, force_lemmy=False):
    cast, _meta = _OTRC.lock_cast(
        creative_fn=_stub, num_characters=num,
        news_seed="Scientists report a quiet anomaly in the upper atmosphere "
                  "over the test range.",
        style="1950s science fiction radio drama",
        rng=random.Random(seed), force_lemmy=force_lemmy,
        max_attempts_per_call=1,
    )
    return cast


class _CountingRandom(random.Random):
    """random.Random that counts choice() calls -- for the R6 draw-count
    invariant on python_assign_voice_preset."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.choice_calls = 0

    def choice(self, seq):
        self.choice_calls += 1
        return super().choice(seq)


# ---------------------------------------------------------------------------
# R1 -- determinism: same seed twice -> identical cast structure.
# ---------------------------------------------------------------------------
def test_r1_determinism_same_seed_identical_cast():
    for seed in (1, 7, 1234, 99999):
        a = _structural(_lock(seed))
        b = _structural(_lock(seed))
        assert a == b, f"cast not deterministic for seed {seed}"


# ---------------------------------------------------------------------------
# R2 -- C7 byte-identity: the coherence-verified golden seed reproduces the
# golden cast exactly (pool mode). This is the no-op-for-coherent-seeds proof
# that the S2 repair must never break.
# ---------------------------------------------------------------------------
def test_r2_c7_byte_identity_vs_golden():
    g = _golden()
    got = _structural(_lock(g["seed"], num=g["num_characters"],
                            force_lemmy=g["force_lemmy"]))
    assert got == g["cast"], (
        "cast drifted from golden baseline -- C7 byte-identity broken.\n"
        f"golden={g['cast']}\ngot={got}"
    )


# ---------------------------------------------------------------------------
# R4 -- quota: the 40/40/20 male/female/other split is preserved.
# ---------------------------------------------------------------------------
def test_r4_quota_40_40_20():
    # Exact split at count=10 (largest-remainder of 0.4/0.4/0.2).
    genders = _OTRC._plan_gender_distribution(10, [], random.Random(0))
    assert len(genders) == 10
    c = Counter(genders)
    assert (c["male"], c["female"], c["other"]) == (4, 4, 2), dict(c)
    # Parts always sum to the requested count, across seeds + counts.
    for count in range(1, 7):
        for seed in range(5):
            gs = _OTRC._plan_gender_distribution(count, [], random.Random(seed))
            assert len(gs) == count
            assert set(gs) <= {"male", "female", "other"}


# ---------------------------------------------------------------------------
# R5 -- voice uniqueness: no two Bark voices collide in a locked cast.
# ---------------------------------------------------------------------------
def test_r5_voice_uniqueness():
    for seed in (1, 3, 11, 42):
        cast = _lock(seed, num=5)
        bark = [r["voice_preset"] for r in cast if r.get("tts_model") == "bark"]
        assert bark, "expected at least one Bark voice"
        assert len(bark) == len(set(bark)), f"duplicate Bark voice at seed {seed}: {bark}"


# ---------------------------------------------------------------------------
# R6 -- exactly one rng.choice per slot in python_assign_voice_preset.
# Gender/timbre filtering must consume NO rng; only the tie-break draws.
# ---------------------------------------------------------------------------
def test_r6_one_rng_choice_per_slot():
    slot = _OTRC.EnsembleSlot(
        char_id="c02", name="ALICE STONE",
        gender="female", timbre="warm", role="lead",
    )
    voices = _POOLS.open_voice_pool(set())
    rng = _CountingRandom(0)
    preset = _OTRC.python_assign_voice_preset(
        slot, available_voices=voices, rng=rng)
    assert preset.startswith("v2/")
    assert rng.choice_calls == 1, (
        f"python_assign_voice_preset drew rng.choice {rng.choice_calls} times; "
        "the one-draw-per-slot invariant (C7) requires exactly 1"
    )


# ---------------------------------------------------------------------------
# R3 / R7 / R8 / R9 are appended by S2 / S6 / S7 / S8 as their behavior lands.
# ---------------------------------------------------------------------------
