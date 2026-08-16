"""tests/test_cast_llm_naming.py -- Phase 2 (llm_slot_fill) regression bar.

Covers the OPTIONAL schema-locked LLM naming layer that rides on top of the
deterministic Phase-1 cast:
  S4 CastPlanner (immutable slots), S7 validator, S5 voice x age (R6 preserved),
  S6 Pass-1 overlay (R7 schema conformance), S8 freeze-order (R9), R8 fallback.

Pool mode (OTR_NAME_MODE unset/pool) must stay byte-identical -- that invariant
lives in tests/test_cast_invariants.py; here we only exercise the llm path.
Hermetic: every LLM call goes through a stub generate_fn.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NODES_DIR = _REPO_ROOT / "nodes"
for _p in (_REPO_ROOT, _NODES_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _otr_casting as _OTRC  # noqa: E402
import _otr_castplanner as _CASTPLAN  # noqa: E402
import _otr_cast_validator as _CASTVAL  # noqa: E402
from config import cast_pools as _POOLS  # noqa: E402


class _CountingRandom(random.Random):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.choice_calls = 0

    def choice(self, seq):
        self.choice_calls += 1
        return super().choice(seq)


def _make_llm(pass1_response):
    """Stub creative_fn: returns the per-slot description for the casting loop,
    and `pass1_response` for the Pass-1 naming call (detected by the prompt)."""
    def gen(messages, *, temperature, max_new_tokens):  # noqa: ARG001
        content = messages[0]["content"] if messages else ""
        if "JSON array" in content:
            return pass1_response
        return json.dumps({
            "character_description":
                "A steady operator who keeps the crew grounded and focused."})
    return gen


def _lock(seed, *, num=4, generate_fn=None):
    cast, meta = _OTRC.lock_cast(
        creative_fn=generate_fn or _make_llm(""), num_characters=num,
        news_seed="An anomaly flickers over the test range at dawn.",
        style="1950s science fiction radio drama",
        rng=random.Random(seed), cast_seed=seed,
        force_lemmy=False, max_attempts_per_call=1,
    )
    # Sprint 2 (a): OTR_CastLock now stamps bark voice_preset post-freeze (the
    # writer no longer does). Apply the byte-identical replay here so tests that
    # assert on the cast see the voices.
    _voices = _OTRC.replay_voice_assignment(
        cast_seed=seed, num_characters=num, lemmy_hit=meta["lemmy_hit"])
    for _r in cast:
        if _r.get("char_id") in _voices:
            _r["voice_preset"] = _voices[_r["char_id"]]
    return cast, meta


def _open(cast):
    return [r for r in cast if r["name"] not in ("ANNOUNCER", "LEMMY")]


# ---------------------------------------------------------------------------
# S4 -- CastPlanner immutable slots
# ---------------------------------------------------------------------------
def test_s4_build_cast_plan_fields_and_age_rotation():
    ens = [
        _OTRC.EnsembleSlot(char_id=f"c0{i+2}", name=f"N{i}", gender=g,
                           timbre="warm", role="lead")
        for i, g in enumerate(["male", "female", "other"])
    ]
    voices = {"c02": "v2/en_speaker_0", "c03": "v2/en_speaker_4",
              "c04": "v2/en_speaker_6"}
    plan = _CASTPLAN.build_cast_plan(ens, voices)
    assert [p.char_id for p in plan] == ["c02", "c03", "c04"]
    assert plan[0].gender == "male" and plan[0].voice_preset == "v2/en_speaker_0"
    assert all(p.age_band in _CASTPLAN.AGE_BANDS for p in plan)
    assert [p.age_band for p in plan] == ["young_adult", "middle_adult", "older_adult"]
    assert all(p.dramatic_role for p in plan)


# ---------------------------------------------------------------------------
# S7 -- validator (R7 schema conformance surface)
# ---------------------------------------------------------------------------
def _plan2():
    return [
        _CASTPLAN.CastPlanSlot("c02", "male", "middle_adult", "v2/a", "lead"),
        _CASTPLAN.CastPlanSlot("c03", "female", "young_adult", "v2/b", "foil"),
    ]


def test_s7_accepts_valid_pass1():
    items = [
        {"char_id": "c02", "name": "Vance Kane",
         "one_line_presence": "steady", "dialogue_style": "clipped"},
        {"char_id": "c03", "name": "Mara Venn",
         "one_line_presence": "precise", "dialogue_style": "warm"},
    ]
    r = _CASTVAL.validate_pass1(items, _plan2())
    assert r.ok
    assert r.names_by_char_id == {"c02": "VANCE KANE", "c03": "MARA VENN"}
    assert r.texture_by_char_id["c02"]["one_line_presence"] == "steady"


@pytest.mark.parametrize("items, why", [
    ([{"char_id": "c02", "name": "X", "gender": "male"},
      {"char_id": "c03", "name": "Y"}], "gender carry-back (extra key)"),
    ([{"char_id": "c99", "name": "X"}, {"char_id": "c03", "name": "Y"}], "unknown char_id"),
    ([{"char_id": "c02", "name": "X"}], "wrong count"),
    ([{"char_id": "c02", "name": "Same"}, {"char_id": "c03", "name": "Same"}], "dup name"),
    ([{"char_id": "c02", "name": "X"}, {"char_id": "c02", "name": "Y"}], "dup char_id / missing c03"),
    ("not a list", "non-list"),
    ([{"char_id": "c02", "name": ""}, {"char_id": "c03", "name": "Y"}], "empty name"),
])
def test_s7_rejects_contract_violations(items, why):
    r = _CASTVAL.validate_pass1(items, _plan2())
    assert not r.ok, f"should reject: {why}"
    assert r.reason


# ---------------------------------------------------------------------------
# S5 -- voice x age (R6: still exactly one rng.choice; pool path unchanged)
# ---------------------------------------------------------------------------
def test_s5_age_keeps_one_rng_choice():
    slot = _OTRC.EnsembleSlot("c02", "ALICE STONE", "female", "warm", "lead")
    voices = _POOLS.open_voice_pool(set())
    rng = _CountingRandom(0)
    preset = _OTRC.python_assign_voice_preset(
        slot, available_voices=voices, rng=rng, age_band="older_adult")
    assert preset.startswith("v2/")
    assert rng.choice_calls == 1


def test_s5_age_none_is_byte_identical():
    slot = _OTRC.EnsembleSlot("c02", "ALICE STONE", "female", "warm", "lead")
    voices = _POOLS.open_voice_pool(set())
    a = _OTRC.python_assign_voice_preset(slot, available_voices=voices, rng=random.Random(0))
    b = _OTRC.python_assign_voice_preset(slot, available_voices=voices, rng=random.Random(0), age_band=None)
    assert a == b


# ---------------------------------------------------------------------------
# S6 -- Pass-1 overlay (R7 applied) + R8 fallback + coherence backstop
# ---------------------------------------------------------------------------
def test_pool_mode_is_default_no_llm_naming(monkeypatch):
    monkeypatch.delenv("OTR_NAME_MODE", raising=False)
    cast, meta = _lock(1)
    assert meta.get("name_mode") != "llm_slot_fill"
    assert "llm_naming_applied" not in meta


def test_r7_llm_naming_applied_when_valid(monkeypatch):
    pool_cast, _ = _lock(1)
    used: list = []
    items = []
    for r in _open(pool_cast):
        # An "other"-gender slot used to draw from the unisex bucket. That
        # bucket was RETIRED on 2026-08-15 (operator: "I don't trust it") and is
        # now empty, so this raised StopIteration. Any name suits an "other"
        # slot anyway -- `_repair_ensemble_names` only judges male/female slots
        # -- so it draws from the full pool.
        bucket = r["gender"] if r["gender"] in ("male", "female") else None
        pool = (_POOLS.FIRST_NAMES_BY_GENDER[bucket] if bucket
                else _POOLS.FIRST_NAMES)
        first = next(n for n in pool if n not in used)
        used.append(first)
        items.append({"char_id": r["char_id"], "name": f"{first} Tester",
                      "one_line_presence": "calm and precise",
                      "dialogue_style": "short clipped lines"})
    monkeypatch.setenv("OTR_NAME_MODE", "llm_slot_fill")
    llm_cast, meta = _lock(1, generate_fn=_make_llm(json.dumps(items)))
    assert meta.get("name_mode") == "llm_slot_fill"
    assert meta.get("llm_naming_applied") is True
    by_id = {r["char_id"]: r for r in _open(llm_cast)}
    for it in items:
        row = by_id[it["char_id"]]
        assert row["name"] == it["name"].upper()
        assert row.get("cast_texture", {}).get("one_line_presence") == "calm and precise"
        if row["gender"] in ("male", "female"):
            assert _POOLS.gender_of_first_name(row["name"]) in (row["gender"], "unisex", "unknown")


def test_llm_names_gender_mismatch_is_repaired(monkeypatch):
    pool_cast, _ = _lock(1)
    # Hand every slot a clearly MALE name (NEMO); female slots must be repaired.
    items = [{"char_id": r["char_id"], "name": f"Nemo Case{idx}",
              "one_line_presence": "x", "dialogue_style": "y"}
             for idx, r in enumerate(_open(pool_cast))]
    monkeypatch.setenv("OTR_NAME_MODE", "llm_slot_fill")
    llm_cast, meta = _lock(1, generate_fn=_make_llm(json.dumps(items)))
    assert meta["llm_naming_applied"] is True
    for r in _open(llm_cast):
        if r["gender"] in ("male", "female"):
            tag = _POOLS.gender_of_first_name(r["name"])
            assert tag in (r["gender"], "unisex", "unknown"), (r["char_id"], r["name"], r["gender"])


def test_r8_invalid_llm_naming_raises(monkeypatch):
    """NO-FALLBACK rip (2026-07-03): the OPT-IN llm_slot_fill naming pass that
    returns un-validatable output now FAILS LOUD -- no silent keep of the
    deterministic RNG-pool names (the :1352 validation path)."""
    monkeypatch.setenv("OTR_NAME_MODE", "llm_slot_fill")
    with pytest.raises(_OTRC.CastValidationLLMError, match="no-fallback rip"):
        _lock(1, generate_fn=_make_llm("not json at all"))


def test_r8_naming_llm_raise_is_loud(monkeypatch):
    """NO-FALLBACK rip (2026-07-03): a naming generate_fn that RAISES also fails
    loud (the :1345 path), not a silent deterministic keep. The per-slot cast
    description call still returns valid JSON so the deterministic cast builds
    first; only the Pass-1 naming call blows up."""
    monkeypatch.setenv("OTR_NAME_MODE", "llm_slot_fill")

    def _boom(messages, *, temperature, max_new_tokens):  # noqa: ARG001
        content = messages[0]["content"] if messages else ""
        if "JSON array" in content:
            raise RuntimeError("naming model unavailable")
        return json.dumps({"character_description": "A steady operator."})

    with pytest.raises(_OTRC.CastValidationLLMError, match="no-fallback rip"):
        _lock(1, generate_fn=_boom)


# ---------------------------------------------------------------------------
# S8 / R9 -- names are FINAL at lock time (frozen before any dialogue pass)
# ---------------------------------------------------------------------------
def test_r9_names_final_and_deterministic_at_lock(monkeypatch):
    pool_cast, _ = _lock(1)
    items = [{"char_id": r["char_id"], "name": f"Mara Venn{idx}",
              "one_line_presence": "p", "dialogue_style": "s"}
             for idx, r in enumerate(_open(pool_cast))]
    resp = json.dumps(items)
    monkeypatch.setenv("OTR_NAME_MODE", "llm_slot_fill")
    c1, _ = _lock(1, generate_fn=_make_llm(resp))
    c2, _ = _lock(1, generate_fn=_make_llm(resp))
    assert [r["name"] for r in c1] == [r["name"] for r in c2]
    # The overlay happened inside lock_cast: returned names are the final ones.
    assert any(r["name"].startswith("MARA VENN") or
               _POOLS.gender_of_first_name(r["name"]) for r in _open(c1))
