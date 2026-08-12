"""LEMMY CHUNK B -- one resolver for a voice-bank ref, not four.

THE DEFECT, AND IT WAS LIVE ON THE QUALIFIED ROUTE. Three cloning adapters
(indextts2, chatterbox, dia) each carried a private `_resolve_ref` that tried
exactly ONE candidate -- `<comfy_base>/models/<ref>` -- and otherwise fell back
to `os.path.abspath(ref)`, a cwd-relative path that does not exist. The voice
node's own `_resolve_ref_to_disk` knew about three more places, including the
`C:\\ComfyUI-Models` root introduced by the Comfy Desktop 1.0.4 model-path
migration.

On this box that is not hypothetical. BOTH Lemmy reference WAVs -- the qualified
Branch-A reference `lemmy_algenib_cockney_v1.wav` and the historic incumbent
`vz_donor_marshal_indian.wav` -- live ONLY under the migrated root. So the
node's existence check confirmed the file and the adapter that had to OPEN it
resolved somewhere else entirely. Preflight green, worker handed a path that is
not there.

The fix is one shared `resolve_voice_ref_path`, and the invariant these tests
hold is agreement: every caller must answer the same question the same way.
"""
from __future__ import annotations

import os

import pytest

from nodes._otr_audio_engines import get_engine
from nodes._otr_audio_engines.base import resolve_voice_ref_path
from nodes._otr_voice_node_common import _resolve_ref_to_disk


#: The engines that clone from a reference WAV and therefore must resolve one.
CLONING_ENGINES = ("indextts2", "chatterbox", "dia")

#: A bank-relative ref in the canonical stored form.
BANK_REF = "models/TTS/refs/indextts2/vz_donor_marshal_indian.wav"


def _old_private_resolver(ref):
    """The copy all three adapters carried before chunk B, verbatim.

    Kept so the tests below compare against the ACTUAL previous behaviour rather
    than against a description of it.
    """
    if not ref or os.path.isabs(ref):
        return ref
    try:
        import folder_paths
        cand = os.path.join(os.path.dirname(folder_paths.models_dir), ref)
        if os.path.exists(cand):
            return cand
    except Exception:  # noqa: BLE001
        pass
    return os.path.abspath(ref)


# ---------------------------------------------------------------------------
# Agreement -- the invariant that was broken
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("engine_name", CLONING_ENGINES)
def test_every_cloning_adapter_agrees_with_the_node_that_checked_the_file(
        engine_name):
    """The check and the open must resolve to the SAME path.

    This is the whole defect stated as an invariant: a preflight that confirms a
    reference the renderer then cannot open is worse than no preflight, because
    it converts a missing file into a mid-render failure.
    """
    try:
        engine = get_engine(engine_name)
    except Exception:  # noqa: BLE001 -- engine not registered on this box
        pytest.skip("%s not registered" % engine_name)
    assert engine._resolve_ref(BANK_REF) == _resolve_ref_to_disk(BANK_REF)


@pytest.mark.parametrize("engine_name", CLONING_ENGINES)
def test_no_adapter_keeps_a_PRIVATE_copy_of_the_resolution_rule(engine_name):
    """Agreement today is not enough -- it has to be structural.

    Three copies that happen to agree drift the moment one of them learns about
    a new location, which is exactly how this bug arrived. Each adapter must
    DELEGATE, so there is one place to teach.
    """
    import inspect

    try:
        engine = get_engine(engine_name)
    except Exception:  # noqa: BLE001
        pytest.skip("%s not registered" % engine_name)
    src = inspect.getsource(type(engine)._resolve_ref)
    assert "resolve_voice_ref_path" in src, (
        "%s still resolves refs itself instead of delegating to the shared "
        "resolver" % engine_name)


def test_the_node_checker_delegates_too():
    """It was the BROADER of the two implementations, which is why it hid the
    bug -- it kept finding the file the adapters could not."""
    import inspect

    assert "resolve_voice_ref_path" in inspect.getsource(_resolve_ref_to_disk)


# ---------------------------------------------------------------------------
# Breadth -- the candidate the adapters were missing
# ---------------------------------------------------------------------------
def test_the_migrated_models_root_is_reachable():
    """The Comfy Desktop 1.0.4 migration root must be a candidate.

    Skips only if this box has no such ref, so the test states its own
    precondition instead of silently passing on a box where it cannot look.
    """
    migrated = os.path.join("C:\\ComfyUI-Models", "TTS", "refs", "indextts2",
                            "vz_donor_marshal_indian.wav")
    if not os.path.exists(migrated):
        pytest.skip("no migrated-root reference on this box")
    resolved = resolve_voice_ref_path(BANK_REF)
    assert os.path.exists(resolved), resolved
    assert os.path.samefile(resolved, migrated)


def test_the_old_private_resolver_would_have_MISSED_it():
    """The regression this closes, proved against the real previous code.

    If this ever starts failing it means the box moved the file back to the
    historical location -- not that the bug was imaginary.
    """
    migrated = os.path.join("C:\\ComfyUI-Models", "TTS", "refs", "indextts2",
                            "vz_donor_marshal_indian.wav")
    if not os.path.exists(migrated):
        pytest.skip("no migrated-root reference on this box")
    old = _old_private_resolver(BANK_REF)
    assert not os.path.exists(old), (
        "the old private resolver found the file, so this box does not "
        "reproduce the condition -- re-derive before trusting the fix")
    assert os.path.exists(resolve_voice_ref_path(BANK_REF))


# ---------------------------------------------------------------------------
# Nothing else moved
# ---------------------------------------------------------------------------
def test_an_absolute_path_passes_through_untouched():
    p = os.path.join("C:\\somewhere", "else.wav")
    assert resolve_voice_ref_path(p) == p


def test_an_empty_ref_passes_through_and_the_node_still_answers_None():
    """The two callers differ ONLY here, and deliberately: the node
    distinguishes 'nothing asked for' from 'asked for and not found'."""
    assert resolve_voice_ref_path("") == ""
    assert resolve_voice_ref_path(None) is None
    assert _resolve_ref_to_disk("") is None
    assert _resolve_ref_to_disk(None) is None


def test_the_historical_candidate_is_still_tried_FIRST(monkeypatch, tmp_path):
    """A box that resolved before must resolve identically now: the shared
    resolver may only turn a MISS into a hit, never MOVE an existing hit.

    BEHAVIOURAL, not lexical. The first draft of this test grepped the source
    for the two candidate expressions and compared their positions -- and failed
    on the fixed code, because the function's own docstring names the migrated
    root before the code reaches it. That is the third time in this session a
    text search read prose as code (see lane 19's L20-inverse note). So the
    order is proved by CONSTRUCTION: a ref that exists in BOTH roots must
    resolve to the historical one.
    """
    import sys
    import types

    # A reference that genuinely exists under the migrated root on this box.
    shared_name = "vz_bill_boerst.wav"
    migrated = os.path.join("C:\\ComfyUI-Models", "TTS", "refs", "indextts2",
                            shared_name)
    if not os.path.exists(migrated):
        pytest.skip("no migrated-root reference on this box")

    # Build a historical root that also has it, and point folder_paths there.
    hist_models = tmp_path / "models"
    hist_dir = hist_models / "TTS" / "refs" / "indextts2"
    hist_dir.mkdir(parents=True)
    (hist_dir / shared_name).write_bytes(b"historical")

    fake = types.ModuleType("folder_paths")
    fake.models_dir = str(hist_models)
    monkeypatch.setitem(sys.modules, "folder_paths", fake)

    resolved = resolve_voice_ref_path(
        "models/TTS/refs/indextts2/%s" % shared_name)
    assert os.path.exists(resolved)
    assert os.path.samefile(resolved, hist_dir / shared_name), (
        "the migrated root won over the historical one, which would MOVE a "
        "reference that already resolved on a shipping box")


def test_a_ref_that_exists_nowhere_returns_a_loud_absolute_path():
    """Not None: the caller must fail on a NAMED path rather than report an
    empty filename to a worker."""
    out = resolve_voice_ref_path("models/TTS/refs/indextts2/__no_such__.wav")
    assert os.path.isabs(out)
    assert not os.path.exists(out)
