r"""Tests for the per-beat LTX motion_clause module (spec v2)."""
import importlib
import os

import pytest

mc = importlib.import_module("nodes._otr_motion_clause")


@pytest.fixture
def flag_on():
    prev = os.environ.get(mc.FLAG_ENV)
    os.environ[mc.FLAG_ENV] = "1"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(mc.FLAG_ENV, None)
        else:
            os.environ[mc.FLAG_ENV] = prev


@pytest.fixture
def flag_off():
    prev = os.environ.get(mc.FLAG_ENV)
    os.environ.pop(mc.FLAG_ENV, None)
    try:
        yield
    finally:
        if prev is not None:
            os.environ[mc.FLAG_ENV] = prev


# ---- validate_motion_clause -------------------------------------------------

def test_validate_ok():
    ok, reason = mc.validate_motion_clause("Mara leans in and talks", "Mara")
    assert ok and reason == "ok"


@pytest.mark.parametrize("text,subject", [
    ("Mara stands up and gestures", "Mara"),
    ("a person talks softly", "Mara"),
    ("He leans and talks", "Mara"),
    ("Mara waits in silence", "Mara"),
    ("camera cut to a dancer who spins", "Mara"),
])
def test_validate_preserves_authored_vocabulary(text, subject):
    ok, reason = mc.validate_motion_clause(text, subject)
    assert ok and reason == "ok"


#: DERIVED, never a literal. This used to be `"Mara " + "x" * 80`, which was
#: over budget only because the cap happened to be 70 -- so the legitimate
#: 2026-08-17 cap opening (70 -> 130, to let the clause writer describe kinetic
#: motion) turned a passing test red without any behaviour changing. A test that
#: hardcodes the constant it is testing re-breaks on every honest change to it.
_OVER_BUDGET = "Mara " + "x" * (mc.CLAUSE_MAX_CHARS + 1)


@pytest.mark.parametrize("text,reason", [
    ("", "empty"),
    (_OVER_BUDGET, "over_budget"),
])
def test_validate_structural_rejections(text, reason):
    ok, got = mc.validate_motion_clause(text, "Mara")
    assert not ok and got == reason


# ---- compute_source_hash ----------------------------------------------------

def test_source_hash_deterministic():
    a = mc.compute_source_hash("c1", "b1", "We must hurry!")
    b = mc.compute_source_hash("c1", "b1", "We  must   hurry!")  # ws-normalized
    assert a == b


def test_source_hash_changes_with_dialogue():
    a = mc.compute_source_hash("c1", "b1", "We must hurry!")
    b = mc.compute_source_hash("c1", "b1", "Stay calm.")
    assert a != b


# ---- resolve_motion_clause_text (read-side guard) ---------------------------

def test_resolve_valid():
    shot = {"motion_clause": {"text": "Mara nods"}}
    assert mc.resolve_motion_clause_text(shot) == "Mara nods"


@pytest.mark.parametrize("shot", [
    {}, {"motion_clause": None}, {"motion_clause": {"text": ""}},
    {"motion_clause": {"text": 123}}, {"motion_clause": {"text": "x" * 300}},
    "not-a-dict",
])
def test_resolve_none(shot):
    assert mc.resolve_motion_clause_text(shot) is None


# ---- generate_motion_clauses (batch pass) -----------------------------------

def _ledger():
    return {
        "video": {"shots": [
            {"shot_id": "s1", "char_id": "c1", "source_line_ids": ["b1"]},
            {"shot_id": "s2", "char_id": "", "source_line_ids": ["b2"]},
        ]},
        "lines": [{"line_id": "b1", "text": "We must hurry!"},
                  {"line_id": "b2", "text": ""}],
    }


def _names(cid):
    return {"c1": "Mara"}.get(cid, cid)


def test_generate_flag_off_all_fallback(flag_off):
    led = _ledger()
    counts = mc.generate_motion_clauses(
        led, generate_fn=lambda *a, **k: "Mara talks", name_resolver=_names)
    assert counts["fallback"] == 2 and counts["generated"] == 0
    s0 = led["video"]["shots"][0]
    assert s0["motion_clause"]["fallback"] is True
    # fallback -> read side returns None -> render keeps its existing prompt
    assert mc.resolve_motion_clause_text(s0) is None


def test_generate_flag_on_dialogue_beat(flag_on):
    led = _ledger()
    counts = mc.generate_motion_clauses(
        led, generate_fn=lambda *a, **k: "Mara leans in and talks",
        name_resolver=_names)
    assert counts["generated"] == 1  # s1 has char + dialogue
    assert counts["fallback"] == 1   # s2 has neither
    s0 = led["video"]["shots"][0]
    assert s0["motion_clause"]["text"] == "Mara leans in and talks"
    assert s0["motion_clause"]["fallback"] is False
    assert mc.resolve_motion_clause_text(s0) == "Mara leans in and talks"


def test_generate_authored_motion_is_preserved(flag_on):
    led = _ledger()
    authored = "Everyone stands up and runs away"
    counts = mc.generate_motion_clauses(
        led, generate_fn=lambda *a, **k: authored,
        name_resolver=_names)
    assert counts["generated"] == 1 and counts["invalid"] == 0
    clause = led["video"]["shots"][0]["motion_clause"]
    assert clause["text"] == authored
    assert clause["fallback"] is False


def test_generate_over_provider_budget_falls_back(flag_on):
    led = _ledger()
    counts = mc.generate_motion_clauses(
        led, generate_fn=lambda *a, **k: _OVER_BUDGET,
        name_resolver=_names)
    assert counts["invalid"] == 1 and counts["generated"] == 0
    assert led["video"]["shots"][0]["motion_clause"]["fallback"] is True


def test_generate_llm_error_never_raises(flag_on):
    def _boom(*a, **k):
        raise RuntimeError("llm down")
    led = _ledger()
    counts = mc.generate_motion_clauses(led, generate_fn=_boom, name_resolver=_names)
    assert counts["fallback"] == 2 and counts["invalid"] == 0


def test_generate_reuse_on_matching_hash(flag_on):
    led = _ledger()
    mc.generate_motion_clauses(
        led, generate_fn=lambda *a, **k: "Mara leans in and talks",
        name_resolver=_names)
    # second pass: a generate_fn that would fail validation must NOT be used for s1
    counts = mc.generate_motion_clauses(
        led, generate_fn=lambda *a, **k: "garbage stands up", name_resolver=_names)
    assert counts["reused"] == 1
    assert led["video"]["shots"][0]["motion_clause"]["text"] == "Mara leans in and talks"


def test_resolve_none_for_fallback():
    shot = {"motion_clause": {"text": "Mara nods", "fallback": True}}
    assert mc.resolve_motion_clause_text(shot) is None


def test_make_name_resolver():
    led = {"cast": [{"char_id": "c1", "name": "Mara"}, {"char_id": "c2"}]}
    r = mc.make_name_resolver(led)
    assert r("c1") == "Mara"
    assert r("c2") == "c2"    # no name -> char_id
    assert r("zzz") == "zzz"  # unknown -> char_id


# ---- render_driver hook is OFF by default (byte-identical) -------------------

def test_render_hook_off_by_default(flag_off):
    rd = importlib.import_module("nodes._otr_video_engines.render_driver")
    shot = {"motion_clause": {"text": "Mara nods"}}
    assert rd._motion_clause_override(shot) is None


def test_render_hook_on_returns_text(flag_on):
    rd = importlib.import_module("nodes._otr_video_engines.render_driver")
    shot = {"motion_clause": {"text": "Mara nods"}}
    assert rd._motion_clause_override(shot) == "Mara nods"
