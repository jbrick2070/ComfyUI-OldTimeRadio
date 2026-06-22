"""tests/test_hybrid_voice_fit.py -- VC chunk 4 (HYBRID LLM voice-fit).

The LLM PROPOSES a voice_ref_id from the engine's gender-slot cards; Python
VALIDATES it (in-library + engine + gender + no-collision) and falls closed to
the deterministic scorer otherwise. Decision rides meta.voice_cast_decision;
CastLock honours an accepted proposal when its resolved engine matches.

Key invariant: a no-LLM / OTR_HYBRID_VOICE_FIT=0 / failure run is byte-identical
to pre-chunk-4 (no voice_cast_decision -> CastLock falls closed).

Hermetic: no GPU, no network. Helper unit tests use a synthetic bank; the
lock_cast + CastLock tests use a prompt-aware stub generate_fn and the real bank.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NODES_DIR = _REPO_ROOT / "nodes"
for p in (_REPO_ROOT, _NODES_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from nodes import _otr_casting as _OTRC  # noqa: E402
from nodes import _otr_voice_bank as VB  # noqa: E402
from nodes.cast_lock import CastLock  # noqa: E402


def _entry(vrid, engine="indextts2", gender="male", tier="", roles=("char_voice",)):
    return VB.VoiceBankEntry(
        voice_ref_id=vrid, engine=engine, gender=gender,
        timbre=("warm",), roles=tuple(roles), age_band="adult",
        ref_path=f"models/TTS/refs/{vrid}.wav", ref_sha256="deadbeef",
        commercial_clean=True, quality_tier=tier, style_tags=("lead_safe",),
    )


# --------------------------------------------------------------------------- #
# build_voice_cards / default_char_engine / validate / lookup
# --------------------------------------------------------------------------- #
def test_build_voice_cards_shape_order_and_no_path():
    bank = (
        _entry("vz_c", gender="male"),
        _entry("vz_a", gender="male"),
        _entry("cb_x", engine="chatterbox", gender="male"),
        _entry("vz_f", gender="female"),
    )
    cards = VB.build_voice_cards("indextts2", "male", bank=bank)
    assert [c["voice_ref_id"] for c in cards] == ["vz_a", "vz_c"]  # sorted, engine+gender
    c = cards[0]
    assert set(c) == {
        "voice_ref_id", "age_band", "timbre", "roles",
        "style_tags", "commercial_clean", "descriptor",
    }
    assert "ref_path" not in c and "ref_sha256" not in c
    assert c["descriptor"]  # synthesized human-readable


def test_build_voice_cards_caps_and_empty_inputs():
    bank = tuple(_entry(f"vz_{i:02d}") for i in range(20))
    assert len(VB.build_voice_cards("indextts2", "male", bank=bank, max_cards=5)) == 5
    assert VB.build_voice_cards("", "male", bank=bank) == []
    assert VB.build_voice_cards("indextts2", "", bank=bank) == []


def test_default_char_engine_real_bank_is_indextts2():
    assert VB.default_char_engine() == "indextts2"


def test_validate_voice_proposal_rules():
    bank = (
        _entry("vz_ok", gender="male"),
        _entry("vz_fem", gender="female"),
        _entry("vz_rej", gender="male", tier="reject"),
    )
    # valid
    assert VB.validate_voice_proposal("vz_ok", "indextts2", "male", bank=bank) == "vz_ok"
    # wrong gender
    assert VB.validate_voice_proposal("vz_fem", "indextts2", "male", bank=bank) == ""
    # reject tier
    assert VB.validate_voice_proposal("vz_rej", "indextts2", "male", bank=bank) == ""
    # wrong engine
    assert VB.validate_voice_proposal("vz_ok", "chatterbox", "male", bank=bank) == ""
    # collision
    assert VB.validate_voice_proposal(
        "vz_ok", "indextts2", "male", bank=bank, used_ids={"vz_ok"}) == ""
    # unknown id
    assert VB.validate_voice_proposal("nope", "indextts2", "male", bank=bank) == ""


def test_voice_ref_entry_lookup():
    bank = (_entry("vz_ok", gender="male"),)
    assert VB.voice_ref_entry("vz_ok", "indextts2", bank).voice_ref_id == "vz_ok"
    assert VB.voice_ref_entry("vz_ok", "chatterbox", bank) is None
    assert VB.voice_ref_entry("", "indextts2", bank) is None


# --------------------------------------------------------------------------- #
# llm_propose_voice_ref
# --------------------------------------------------------------------------- #
def _slot():
    return _OTRC.EnsembleSlot(
        char_id="c02", name="ALICE", gender="male", timbre="warm", role="lead")


def test_llm_propose_parses_id():
    def fn(messages, *, temperature, max_new_tokens):  # noqa: ARG001
        return 'Sure: {"voice_ref_id": "vz_a"} done'
    cards = [{"voice_ref_id": "vz_a", "descriptor": "adult, warm"}]
    assert _OTRC.llm_propose_voice_ref(fn, slot=_slot(), cards=cards) == "vz_a"


def test_llm_propose_failsoft():
    def boom(messages, *, temperature, max_new_tokens):  # noqa: ARG001
        raise RuntimeError("model down")
    cards = [{"voice_ref_id": "vz_a", "descriptor": "x"}]
    assert _OTRC.llm_propose_voice_ref(boom, slot=_slot(), cards=cards) == ""
    # empty cards -> no call, ''
    assert _OTRC.llm_propose_voice_ref(boom, slot=_slot(), cards=[]) == ""


# --------------------------------------------------------------------------- #
# lock_cast end-to-end hybrid decision
# --------------------------------------------------------------------------- #
def _prompt_aware_fn():
    """Return description JSON for casting prompts, voice-fit JSON (the FIRST
    listed candidate id) for voice-fit prompts."""
    def fn(messages, *, temperature, max_new_tokens):  # noqa: ARG001
        text = messages[0]["content"]
        if "best-fitting voice" in text:
            # pick the first "- <id>:" candidate from the prompt
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("- "):
                    vid = line[2:].split(":", 1)[0].strip()
                    return json.dumps({"voice_ref_id": vid})
            return json.dumps({"voice_ref_id": ""})
        return json.dumps({
            "character_description": "A weathered relay tech who never sleeps.",
            "speech_signature": "clipped and terse",
        })
    return fn


def test_lock_cast_populates_voice_cast_decision(monkeypatch):
    monkeypatch.setenv("OTR_HYBRID_VOICE_FIT", "1")
    cast, meta = _OTRC.lock_cast(
        creative_fn=_prompt_aware_fn(), num_characters=3,
        news_seed="A relay station goes dark.", style="closed_room_suspense",
        rng=random.Random(42), cast_seed=42, force_lemmy=False,
    )
    decisions = meta.get("voice_cast_decision")
    assert isinstance(decisions, dict) and decisions
    # Every open slot got a decision; the stub picks a valid candidate -> accepted.
    for cid, d in decisions.items():
        assert set(d) >= {
            "policy_version", "bank_sha", "engine", "prompt_version",
            "seed", "candidate_ids", "proposed_id", "accepted_id", "fallback_reason",
        }
        assert d["engine"] == "indextts2"
        if d["candidate_ids"]:
            assert d["accepted_id"] == d["proposed_id"]
            assert d["accepted_id"] in d["candidate_ids"]
    # No two characters accepted the same voice (no-collision).
    accepted = [d["accepted_id"] for d in decisions.values() if d["accepted_id"]]
    assert len(accepted) == len(set(accepted))


def test_lock_cast_hybrid_off_is_empty(monkeypatch):
    monkeypatch.setenv("OTR_HYBRID_VOICE_FIT", "0")
    _cast, meta = _OTRC.lock_cast(
        creative_fn=_prompt_aware_fn(), num_characters=2,
        news_seed="x", style="open",
        rng=random.Random(7), cast_seed=7, force_lemmy=False,
    )
    assert meta.get("voice_cast_decision") == {}


# --------------------------------------------------------------------------- #
# CastLock honours / falls closed on the decision
# --------------------------------------------------------------------------- #
def _real_male_indextts2_id():
    bank, _ = VB.load_voice_bank()
    cands = sorted(
        e.voice_ref_id for e in bank
        if e.engine == "indextts2" and e.gender == "male")
    return cands[0]


def test_auto_registry_honours_accepted_proposal():
    vid = _real_male_indextts2_id()
    led = {
        "meta": {
            "episode_seed": 5,
            "voice_cast_decision": {
                "c02": {"engine": "indextts2", "accepted_id": vid,
                        "proposed_id": vid, "fallback_reason": ""},
            },
        },
        "cast": [{"char_id": "c02", "name": "BOB", "gender": "male"}],
        "lines": [],
    }
    report: list = []
    CastLock()._auto_registry(led, led["cast"], "default", False, report)
    assert led["cast"][0].get("voice_ref_id") == vid
    assert any("hybrid LLM voice-fit" in r for r in report)


def test_auto_registry_falls_closed_on_engine_mismatch():
    vid = _real_male_indextts2_id()
    led = {
        "meta": {
            "episode_seed": 5,
            "voice_cast_decision": {
                # proposal was for a DIFFERENT engine -> ignored.
                "c02": {"engine": "chatterbox", "accepted_id": vid,
                        "fallback_reason": ""},
            },
        },
        "cast": [{"char_id": "c02", "name": "BOB", "gender": "male"}],
        "lines": [],
    }
    report: list = []
    CastLock()._auto_registry(led, led["cast"], "default", False, report)
    # Deterministic scorer ran (real indextts2 ref stamped), NOT the hybrid path.
    assert led["cast"][0].get("voice_engine") == "indextts2"
    assert not any("hybrid LLM voice-fit" in r for r in report)
