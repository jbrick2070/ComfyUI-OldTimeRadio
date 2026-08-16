"""tests/test_cast_invariants.py -- the R1-R9 regression bar for the cast
name<->gender<->voice coherence sprint.

Green here plus the focused cast/voice pytest suites is the
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
        rng=random.Random(seed), cast_seed=seed, force_lemmy=force_lemmy,
        max_attempts_per_call=1,
    )
    # Sprint 2 (a): the writer no longer stamps bark voice_preset -- OTR_CastLock
    # replays it post-freeze (byte-identical, pinned by
    # test_cast_voice_replay_parity). These coherence invariants assert on the
    # fully-cast ledger, so apply the same replay the CastLock node applies.
    _voices = _OTRC.replay_voice_assignment(
        cast_seed=seed, num_characters=num, lemmy_hit=_meta["lemmy_hit"])
    for _r in cast:
        if _r.get("char_id") in _voices:
            _r["voice_preset"] = _voices[_r["char_id"]]
    return cast


def _binary_mismatches(cast):
    """Open rows whose binary slot gender disagrees with the name's gender."""
    out = []
    for r in cast:
        if r["name"] in ("ANNOUNCER", "LEMMY"):
            continue
        if r["gender"] in ("male", "female"):
            tag = _POOLS.gender_of_first_name(r["name"])
            if tag in ("male", "female") and tag != r["gender"]:
                out.append((r["char_id"], r["name"], r["gender"], tag))
    return out


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
# R3 -- coherence (S2): every binary-gender slot's name gender matches the slot
# gender in strict mode. This is the bug the sprint exists to fix.
# ---------------------------------------------------------------------------
def test_r3_coherence_no_binary_mismatch():
    g = _golden()
    seeds = {g["known_incoherent_seed"], g["seed"]} | set(range(0, 60))
    for seed in sorted(seeds):
        for num in (3, 4, 5):
            cast = _lock(seed, num=num)
            bad = _binary_mismatches(cast)
            assert not bad, f"seed {seed} num {num}: name/gender mismatch {bad}"


def test_r3_known_bug_seed_repaired():
    """The golden's known-incoherent seed had a male name on a female slot
    (NEMO on a female slot). After repair the slot carries a same-gender name;
    the slot gender is unchanged (only the NAME was ever wrong)."""
    g = _golden()
    seed = g["known_incoherent_seed"]
    cast = _lock(seed, num=g["num_characters"])
    assert g["known_incoherent_detail"], "golden must record the bug case"
    for det in g["known_incoherent_detail"]:
        row = next(r for r in cast if r["char_id"] == det["char_id"])
        assert row["gender"] == det["slot_gender"], "slot gender must not change"
        assert row["name"] != det["name"], "the mismatched name was not repaired"
        assert _POOLS.gender_of_first_name(row["name"]) == det["slot_gender"]


def test_r3_repair_is_byte_identical_for_coherent_seed():
    """Belt-and-braces on R2: the repair must be a pure no-op at the coherent
    golden seed -- the whole point of the isolated-rng design."""
    g = _golden()
    assert _structural(_lock(g["seed"], num=g["num_characters"])) == g["cast"]


def test_cross_gender_rate_controls_repair(monkeypatch):
    """OTR_NAME_CROSS_GENDER_RATE: 0.0 (default) repairs every mismatch; 1.0
    lets the deliberate-cross path leave the original mismatch standing."""
    g = _golden()
    seed = g["known_incoherent_seed"]
    assert not _binary_mismatches(_lock(seed, num=g["num_characters"]))  # strict
    monkeypatch.setenv("OTR_NAME_CROSS_GENDER_RATE", "1.0")
    assert _binary_mismatches(_lock(seed, num=g["num_characters"])), (
        "rate=1.0 should leave the deliberate cross-gender name in place")


# ---------------------------------------------------------------------------
# R7 / R8 / R9 are appended by S6 / S7 / S8 as their behavior lands.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# R10 -- THE UNISEX BUCKET IS A PROMISE (2026-08-15, operator-raised)
#
# `_repair_ensemble_names` fires ONLY when a name's tag is male/female AND
# disagrees with the slot. A "unisex" tag is therefore the one value that
# exempts a name from the coherence check on EITHER binary gender -- it tells
# the guard to stand down. Membership is a claim that the name genuinely works
# with a male or a female voice, and a name that does not belong there has no
# guard behind it at all.
#
# The pool's own policy (config/cast_pools.py) says only "genuinely
# cross-cultural or ambiguous" names may sit there. ADRIAN, HAYES, STONE and
# CARTER did, and none is either -- all four are clearly male-coded English
# given names, ADRIAN with its own female counterpart. ADRIAN on a female slot
# would have shipped unrepaired: the "Miss McFiggins" defect.
#
# The existing coherence assertions cannot catch this class, by construction:
# `tests/test_cast_llm_naming.py:189` reads
# `assert gender_of_first_name(name) in (row["gender"], "unisex", "unknown")`,
# where "unisex" is an unconditional free pass. A mistagged name passes it
# every time. That is why this bar is behavioural rather than another tag check.
# ---------------------------------------------------------------------------


def _slot(name: str, gender: str, *, source_owned: bool = False):
    return _OTRC.EnsembleSlot(
        char_id="c01", name=name, gender=gender,
        timbre=_OTRC._TIMBRE_VOCAB[0], role=_OTRC._ROLE_VOCAB[0],
        source_owned=source_owned,
    )


@pytest.mark.parametrize("first", ["ADRIAN", "HAYES", "STONE", "CARTER"])
def test_R10_a_male_coded_name_on_a_female_slot_is_repaired(first):
    """THE REGRESSION. Each of these returned "unisex" before the retag, so the
    repair stood down and the name shipped against a female voice."""
    out = _OTRC._repair_ensemble_names(
        [_slot(f"{first} HIBBERT", "female")], cast_seed=42,
    )
    assert out[0].name != f"{first} HIBBERT", (
        f"{first} rode a female slot unrepaired -- it is tagged "
        f"{_POOLS.gender_of_first_name(first)!r}, and only male/female tags "
        f"trigger the guard"
    )
    assert _POOLS.gender_of_first_name(out[0].name) in ("female", "unisex",
                                                        "unknown")


def test_R10_the_unisex_bucket_stays_EMPTY():
    """Operator ruling 2026-08-15: "I don't trust it".

    A "unisex" tag was the ONE value that exempted a name from the coherence
    repair on either binary slot gender, and 30 of 153 names -- one draw in
    five -- were riding that exemption unverified. The bucket is retired; every
    name now carries a definite tag so the repair always has an answer.

    Pinned as EMPTY rather than deleted because `names_for_genre` and
    `_verify_name_buckets` both index the key by name.
    """
    assert _POOLS.FIRST_NAMES_BY_GENDER["unisex"] == []
    strays = [n for n in _POOLS.FIRST_NAMES
              if _POOLS.gender_of_first_name(n) == "unisex"]
    assert strays == [], f"these names re-entered the exemption: {strays}"


def test_R10_every_drawable_name_has_a_definite_tag():
    """The point of retiring the bucket: no name in the roll pool may reach a
    slot without the repair being able to judge it. "unknown" counts as a hole
    here too -- it is treated exactly like the old unisex free pass."""
    undecided = [
        n for n in _POOLS.FIRST_NAMES
        if _POOLS.gender_of_first_name(n) not in ("male", "female")
    ]
    assert undecided == [], (
        f"{len(undecided)} drawable name(s) cannot be judged by the coherence "
        f"repair: {undecided}"
    )


def test_R10_the_buckets_stay_a_partition():
    """A name in two buckets makes `gender_of_first_name` order-dependent."""
    seen: dict[str, str] = {}
    for bucket, names in _POOLS.FIRST_NAMES_BY_GENDER.items():
        for n in names:
            key = n.upper()
            assert key not in seen, (
                f"{n!r} is in both {seen.get(key)!r} and {bucket!r}"
            )
            seen[key] = bucket


def test_R10_source_owned_names_are_still_exempt():
    """TOBY is Sir Toby Belch whatever the pool thinks. The retag must not
    start renaming adapted characters -- that rename fired on 133 of 400 seeds
    when it was last let loose."""
    out = _OTRC._repair_ensemble_names(
        [_slot("ADRIAN HIBBERT", "female", source_owned=True)], cast_seed=42,
    )
    assert out[0].name == "ADRIAN HIBBERT"
