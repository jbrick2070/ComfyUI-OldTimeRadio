r"""Tests for the LLM outfit LOCK (_otr_wardrobe): story+character generated, locked,
NO fallback (fail-loud), opt-in/default-safe, and its _appearance_for_char hook."""
import importlib
import os

import pytest

wd = importlib.import_module("nodes._otr_wardrobe")
mc = importlib.import_module("nodes._otr_motion_clause")

FACE = "Face: oval, deep-set eyes, salt-and-pepper hair, strong jawline"
GOOD_OUTFIT = "a patched canvas duster over a sun-bleached work shirt"


@pytest.fixture
def lock_on():
    prev = os.environ.get(wd.FLAG_ENV)
    os.environ[wd.FLAG_ENV] = "1"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(wd.FLAG_ENV, None)
        else:
            os.environ[wd.FLAG_ENV] = prev


@pytest.fixture
def lock_off():
    prev = os.environ.get(wd.FLAG_ENV)
    os.environ.pop(wd.FLAG_ENV, None)
    try:
        yield
    finally:
        if prev is not None:
            os.environ[wd.FLAG_ENV] = prev


def _stub_writer(text):
    """Patch make_writer_generate_fn to return a gen that always yields *text*
    (or None to simulate the LLM being unavailable)."""
    if text is None:
        return lambda *a, **k: None
    return lambda *a, **k: (lambda *aa, **kk: text)


def _ledger():
    return {"cast": [{"char_id": "c02", "name": "SUNAN",
                      "character_description": FACE}],
            "meta": {"story_brief": "a desolate post-apocalyptic settlement"}}


# ---- pure helpers -----------------------------------------------------------

def test_has_clothing_true_false():
    assert wd.has_clothing("a man wearing a coat")
    assert not wd.has_clothing(FACE)
    assert not wd.has_clothing("relentless in pursuit, suitable for the part")


@pytest.mark.parametrize("text,ok", [
    (GOOD_OUTFIT, True),
    ("", False),
    ("x" * 120, False),
    ("I'm sorry, I cannot help with that", False),
    ("a weary determined expression", False),  # not an outfit
])
def test_validate_outfit(text, ok):
    assert wd._validate_outfit(text)[0] is ok


# ---- default OFF = byte-identical -------------------------------------------

def test_apply_off_byte_identical(lock_off):
    assert wd.apply_wardrobe(FACE, "c02", _ledger()) == FACE


def test_appearance_off_identical(lock_off):
    sl = importlib.import_module("nodes.otr_shot_lock")
    assert sl._appearance_for_char(_ledger(), "c02") == FACE


# ---- ON + good LLM = locked outfit appended --------------------------------

def test_apply_on_appends(lock_on, monkeypatch):
    monkeypatch.setattr(mc, "make_writer_generate_fn", _stub_writer(GOOD_OUTFIT))
    out = wd.apply_wardrobe(FACE, "c02", _ledger())
    assert out.startswith(FACE)
    assert "wearing" in out and GOOD_OUTFIT in out


def test_locked_outfit_caches_on_row(lock_on, monkeypatch):
    calls = {"n": 0}

    def factory():
        def gen(*a, **k):
            calls["n"] += 1
            return GOOD_OUTFIT
        return gen
    monkeypatch.setattr(mc, "make_writer_generate_fn", factory)
    led = _ledger()
    a = wd.locked_outfit("c02", led)
    b = wd.locked_outfit("c02", led)        # second call must reuse the cached row value
    assert a == b == GOOD_OUTFIT
    assert calls["n"] == 1                    # LLM invoked once, then cached
    assert led["cast"][0]["_wardrobe_lock"] == GOOD_OUTFIT


def test_apply_respects_existing_clothing(lock_on, monkeypatch):
    monkeypatch.setattr(mc, "make_writer_generate_fn", _stub_writer(GOOD_OUTFIT))
    s = "a weary man wearing a charcoal jacket"
    assert wd.apply_wardrobe(s, "c02", _ledger()) == s    # already clothed -> untouched


# ---- ON + LLM can't deliver = FAIL LOUD (no fallback) ----------------------

def test_fail_loud_when_llm_unavailable(lock_on, monkeypatch):
    monkeypatch.setattr(mc, "make_writer_generate_fn", _stub_writer(None))
    with pytest.raises(wd.WardrobeError):
        wd.apply_wardrobe(FACE, "c02", _ledger())


def test_fail_loud_on_refusal(lock_on, monkeypatch):
    monkeypatch.setattr(mc, "make_writer_generate_fn",
                        _stub_writer("I'm sorry, I cannot do that"))
    with pytest.raises(wd.WardrobeError):
        wd.apply_wardrobe(FACE, "c02", _ledger())


def test_appearance_fail_loud_propagates(lock_on, monkeypatch):
    monkeypatch.setattr(mc, "make_writer_generate_fn", _stub_writer(None))
    sl = importlib.import_module("nodes.otr_shot_lock")
    with pytest.raises(wd.WardrobeError):
        sl._appearance_for_char(_ledger(), "c02")


def test_no_charid_no_change(lock_on, monkeypatch):
    monkeypatch.setattr(mc, "make_writer_generate_fn", _stub_writer(None))
    assert wd.apply_wardrobe(FACE, "") == FACE   # no char -> no lock, no raise
