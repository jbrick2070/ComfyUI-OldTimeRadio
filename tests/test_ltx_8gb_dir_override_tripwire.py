"""``OTR_LTX_8GB_*_DIR`` is a DEPRECATION TRIPWIRE, not a working knob.

Neither `*_DIR` has ever changed which weights render. `_resolve_model_file`
lets a `*_DIR` win on file EXISTENCE ALONE and never consults `folder_paths`,
while `_build_graph` hands `CheckpointLoaderSimple` / `CLIPLoader` a BARE
TOKEN, which ComfyUI resolves through `folder_paths`. So an operator who sets
`*_DIR` gets a green preflight against one file and a render against another --
the same identity lie C-1 closed for the explicit `OTR_LTX_8GB_CKPT` path, one
level up.

The panel's unanimous pick: the TOKEN is the authority, a disagreeing `*_DIR`
is TERMINAL, and the message confesses the build's bug rather than blaming the
operator's setup -- pointing at `extra_model_paths.yaml`, the channel that does
reach the loader and is already live on this box. Registering the folder from
inside preflight (`folder_paths.add_model_folder_path`) was rejected: ComfyUI
ships no unregister, so a CHECK would mutate global state permanently.

SCOPED TO THIS ADAPTER on purpose. `tests/test_wan_loader_preflight.py` uses
`*_DIR` as its no-ComfyUI mock seam, so `wan_shared._resolve_model_file` had to
stay behaviour-identical under the split; the last two controls pin that.

The junction / case-fold half of the comparison belongs to `_same_file`, which
is unit-tested in `tests/test_ltx_8gb_session_config.py`. What is tested HERE is
which question gets asked, and when it gets asked at all.
"""

from __future__ import annotations

import os
import sys

import pytest

from nodes._otr_video_engines import eng_ltx_8gb as m
from nodes._otr_video_engines.registry import EngineUnusable, EngineUsabilityReason

_ENVS = (
    "OTR_LTX_8GB_CKPT", "OTR_LTX_8GB_CKPT_DIR", "OTR_LTX_8GB_CKPT_NAME",
    "OTR_LTX_8GB_T5_DIR", "OTR_LTX_8GB_T5_NAME", "OTR_LTX_8GB_STEPS",
    "OTR_LTX_8GB_TILED_VAE", "OTR_LTX_8GB_T5_DEVICE",
    "OTR_WAN_I2V_VAE_DIR", "OTR_WAN_I2V_CLIP_DIR",
    # The ceiling is the ONE knob still parsed on this file's `assert_usable`
    # path outside prequalification, so a malformed host value would turn a
    # MISSING_MODEL assertion into MALFORMED_CONFIG. The frozen knobs are
    # harmless here only because the freeze makes them unreadable -- scrub the
    # one that is not, and the consent act that would unfreeze the rest.
    "OTR_LTX_8GB_MAX_FRAMES", m.PREQUALIFICATION_ENV,
)

#: The real floor is 4 GiB; materialising that per test buys no extra proof.
_TEST_FLOOR = 1024


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _ENVS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def eng(tmp_path, monkeypatch):
    """A REAL resolution path -- the resolver IS the subject here.

    Nothing is monkeypatched over `_resolve_model_file`. Instead a fake ComfyUI
    root carries the standard ``models/<category>/`` layout, which is where
    `_resolve_model_file_by_token` lands after the conftest `folder_paths` stub
    (which has no `get_full_path`) misses. That makes "what the loader would
    find" a real answer from real code, not a lambda that agrees by
    construction.

    THE FIXTURE MUST OWN EVERY ROOT, not just one (2026-08-11, video transplant
    lane 1). `_resolve_model_file_by_token` gained a third and last probe --
    the CONFIGURED models root, so a lane can resolve its weights off the
    ComfyUI runtime on a box that keeps them outside the comfy tree. This
    fixture controlled `_comfy_root` and nothing else, so it was silently
    relying on the real box's other roots being invisible; the moment one was
    probed, "I deleted the checkpoint" stopped meaning the checkpoint was gone
    -- the operator's REAL ltx_8gb weight answered instead and two absence
    assertions went green-for-the-wrong-reason. A fake model universe has to be
    the WHOLE universe.
    """
    monkeypatch.setattr(m, "_LTX8_CKPT_MIN_BYTES", _TEST_FLOOR)
    root = tmp_path / "comfy"
    (root / "models" / "checkpoints").mkdir(parents=True)
    (root / "models" / "text_encoders").mkdir(parents=True)
    empty_models_root = tmp_path / "no_other_models"
    empty_models_root.mkdir()
    monkeypatch.setenv("OTR_COMFYUI_MODELS_ROOT", str(empty_models_root))
    monkeypatch.setenv("COMFYUI_MODELS_ROOT", str(empty_models_root))
    (root / "models" / "checkpoints" / m._LTX8_DEFAULT_CKPT).write_bytes(
        b"c" * (_TEST_FLOOR * 2))
    (root / "models" / "text_encoders" / m._LTX8_DEFAULT_T5).write_bytes(
        b"t" * 512)
    e = m.Ltx8gbEngine()
    e._comfy_root = lambda: str(root)        # staticmethod -> instance override
    e._root = root
    return e


def _registered_ckpt(eng):
    return str(eng._root / "models" / "checkpoints" / m._LTX8_DEFAULT_CKPT)


def _elsewhere(tmp_path, name, size=_TEST_FLOOR * 2):
    """A directory holding a SAME-NAMED file the loader will never see."""
    d = tmp_path / "elsewhere"
    d.mkdir(exist_ok=True)
    (d / name).write_bytes(b"d" * size)
    return d


# --- THE TRIPWIRE ---------------------------------------------------------- #
def test_a_CKPT_DIR_naming_a_different_file_is_terminal(
        eng, tmp_path, monkeypatch):
    """Two same-named files, one registered. The old code returned the DIR's
    copy and called preflight green; the render would have used the other."""
    d = _elsewhere(tmp_path, m._LTX8_DEFAULT_CKPT)
    monkeypatch.setenv("OTR_LTX_8GB_CKPT_DIR", str(d))

    with pytest.raises(EngineUnusable) as e:
        eng._ckpt_path()
    assert e.value.reason == EngineUsabilityReason.MALFORMED_CONFIG
    msg = str(e.value)
    assert "OTR_LTX_8GB_CKPT_DIR" in msg
    # Paths go through %r, so backslashes are escaped -- match on the segment
    # names, which survive the escaping and still name BOTH files.
    assert "elsewhere" in msg                     # what the operator set
    assert "checkpoints" in msg                   # what actually loads
    assert m._LTX8_DEFAULT_CKPT in msg            # the token in between
    assert "extra_model_paths.yaml" in msg        # the remedy that works


def test_the_refusal_reaches_the_operator_through_assert_usable(
        eng, tmp_path, monkeypatch):
    """The SINGLE-CLIP gate -- the common case, and the one that renders."""
    monkeypatch.setenv("OTR_LTX_8GB_CKPT_DIR",
                       str(_elsewhere(tmp_path, m._LTX8_DEFAULT_CKPT)))
    with pytest.raises(EngineUnusable) as e:
        eng.assert_usable(host_caps={}, profile={})
    assert e.value.reason == EngineUsabilityReason.MALFORMED_CONFIG


def test_the_refusal_reaches_the_operator_through_resolve_session_config(
        eng, tmp_path, monkeypatch):
    """The multi-segment path must reach the SAME verdict on the same env --
    two authorities over one fact is the defect class this whole arc closes."""
    monkeypatch.setenv("OTR_LTX_8GB_CKPT_DIR",
                       str(_elsewhere(tmp_path, m._LTX8_DEFAULT_CKPT)))
    with pytest.raises(EngineUnusable) as e:
        eng.resolve_session_config()
    assert e.value.reason == EngineUsabilityReason.MALFORMED_CONFIG


def test_a_CKPT_DIR_the_loader_cannot_see_AT_ALL_is_terminal(
        eng, tmp_path, monkeypatch):
    """The purest form of the lie: the weights exist ONLY in the `*_DIR`.

    Preflight used to go green on them while the loader had nothing to load, so
    the failure arrived mid-render as a ComfyUI stack trace about a missing
    checkpoint -- after the operator had already paid for the setup.
    """
    (eng._root / "models" / "checkpoints" / m._LTX8_DEFAULT_CKPT).unlink()
    monkeypatch.setenv("OTR_LTX_8GB_CKPT_DIR",
                       str(_elsewhere(tmp_path, m._LTX8_DEFAULT_CKPT)))
    with pytest.raises(EngineUnusable) as e:
        eng._ckpt_path()
    assert e.value.reason == EngineUsabilityReason.MALFORMED_CONFIG
    assert "nothing at all on this box" in str(e.value)


def test_T5_DIR_carries_the_same_tripwire(eng, tmp_path, monkeypatch):
    """The T5 has NO explicit-path override, so `*_DIR` is its ONLY lie channel
    -- and the 0.9.8 checkpoint carries no text encoder, so a wrong T5 is not a
    cosmetic difference."""
    d = tmp_path / "t5_elsewhere"
    d.mkdir()
    (d / m._LTX8_DEFAULT_T5).write_bytes(b"x" * 512)
    monkeypatch.setenv("OTR_LTX_8GB_T5_DIR", str(d))

    with pytest.raises(EngineUnusable) as e:
        eng._t5_path()
    assert e.value.reason == EngineUsabilityReason.MALFORMED_CONFIG
    assert "OTR_LTX_8GB_T5_DIR" in str(e.value)


def test_the_DIR_verdict_is_reached_BEFORE_the_explicit_path_verdict(
        eng, tmp_path, monkeypatch):
    """THREE distinct files, so both guards would fire independently: the
    operator hears about the DIRECTORY, which is the deprecated channel and the
    one to remove.

    The explicit override deliberately names a THIRD file rather than the DIR's
    copy. Pointing both variables at the same decoy makes the explicit guard's
    condition trivially false -- `explicit` IS `by_token` -- so the test would
    pass under a branch swap as well: decorative, in the one test whose entire
    subject is which branch runs first. A QA lens caught that; the third file is
    the fix, and mutation SWAP_explicit_guard_runs_first is the proof.
    """
    d = _elsewhere(tmp_path, m._LTX8_DEFAULT_CKPT)
    third = tmp_path / "third_decoy.safetensors"
    third.write_bytes(b"3" * (_TEST_FLOOR * 2))
    monkeypatch.setenv("OTR_LTX_8GB_CKPT_DIR", str(d))
    monkeypatch.setenv("OTR_LTX_8GB_CKPT", str(third))

    with pytest.raises(EngineUnusable) as e:
        eng._ckpt_path()
    msg = str(e.value)
    assert "does not change which weights load" in msg          # the DIR message
    assert "names a file the loader will NOT load" not in msg   # not C-1's


def test_a_disagreeing_DIR_does_not_escape_the_installed_predicate(
        eng, tmp_path, monkeypatch):
    """`_installed()` is a BOOL contract -- `load()` gates on
    `if not self._installed():`, and every sibling adapter returns a bare bool.

    The new branch is a SECOND place that can raise out of `_ckpt_path()`, and
    only the explicit-path guard was ever pinned here. Today's code is correct
    by inspection; without this test the next edit could raise something
    `_installed` does not catch and `load()` would crash instead of failing
    closed, with nothing to notice.
    """
    monkeypatch.setenv("OTR_LTX_8GB_CKPT_DIR",
                       str(_elsewhere(tmp_path, m._LTX8_DEFAULT_CKPT)))
    assert eng._installed() is False              # a predicate, not a raise
    with pytest.raises(RuntimeError) as e:
        eng.load()
    assert "not installed" in str(e.value)


# --- CONTROLS: a tightening that refuses honest input is not a fix --------- #
def test_CONTROL_no_DIR_set_asks_no_new_question_at_all(eng, monkeypatch):
    """The ONLY configuration that exists today -- nothing in `config/`,
    `scripts/` or `workflows/` sets a `*_DIR`.

    Proving the tripwire is inert is not enough; prove it is UNREACHED.
    `_same_file` is the sole gate of BOTH override guards, so a single call
    would mean a guard fired on a box that set nothing.
    """
    seen = []
    real = m._same_file
    monkeypatch.setattr(
        m, "_same_file", lambda a, b: (seen.append((a, b)), real(a, b))[1])

    assert eng._ckpt_path() == _registered_ckpt(eng)
    assert eng._t5_path() == str(
        eng._root / "models" / "text_encoders" / m._LTX8_DEFAULT_T5)
    assert eng.assert_usable(host_caps={}, profile={}) == "ltx_8gb"
    assert seen == []


def test_CONTROL_a_DIR_that_agrees_with_the_registered_file_passes(
        eng, monkeypatch):
    """A directory that happens to BE the registered one is honest. Refusing it
    would punish a correct setup -- that is a tightening, not a fix."""
    reg = eng._root / "models" / "checkpoints"
    monkeypatch.setenv("OTR_LTX_8GB_CKPT_DIR", str(reg))
    assert eng._ckpt_path() == str(reg / m._LTX8_DEFAULT_CKPT)
    assert eng.assert_usable(host_caps={}, profile={}) == "ltx_8gb"


@pytest.mark.skipif(os.path.normcase("A") == "A",
                    reason="case-insensitive filesystems only")
def test_CONTROL_a_case_respelled_DIR_still_passes(eng, monkeypatch):
    """NTFS is case-insensitive; `C:\\Models` and `c:\\models` are one folder.
    An `abspath` string compare would refuse this box's own configuration."""
    monkeypatch.setenv("OTR_LTX_8GB_CKPT_DIR",
                       str(eng._root / "models" / "checkpoints").upper())
    assert eng._ckpt_path() is not None


def test_CONTROL_a_DIR_respelled_with_a_dot_segment_still_passes(
        eng, monkeypatch):
    """Stands in for the junction case that motivated `_same_file`: two
    spellings, one file. This build reaches its own repo through a junction, so
    a comparison that only folds separators would refuse real installs."""
    monkeypatch.setenv(
        "OTR_LTX_8GB_CKPT_DIR",
        os.path.join(str(eng._root / "models"), ".", "checkpoints"))
    assert eng._ckpt_path() is not None


def test_CONTROL_a_DIR_agreeing_with_extra_model_paths_passes(
        eng, tmp_path, monkeypatch):
    """The real-world PASS: the folder IS registered (extra_model_paths.yaml
    surfaces as `folder_paths.get_full_path`) and the operator also set the
    now-pointless variable. Also the only test here that exercises the
    `folder_paths` branch rather than the standard-layout fallback."""
    shared = tmp_path / "shared_store"
    shared.mkdir()
    (shared / m._LTX8_DEFAULT_CKPT).write_bytes(b"c" * (_TEST_FLOOR * 2))
    monkeypatch.setattr(
        sys.modules["folder_paths"], "get_full_path",
        lambda category, name: (
            str(shared / name)
            if category == "checkpoints" and (shared / name).exists() else None),
        raising=False)
    monkeypatch.setenv("OTR_LTX_8GB_CKPT_DIR", str(shared))

    assert eng._ckpt_path() == str(shared / m._LTX8_DEFAULT_CKPT)


def test_CONTROL_the_explicit_path_guard_still_fires_with_its_own_message(
        eng, tmp_path, monkeypatch):
    """C-1's guard must survive the new branch sitting above it."""
    decoy = tmp_path / "decoy.safetensors"
    decoy.write_bytes(b"d" * (_TEST_FLOOR * 2))
    monkeypatch.setenv("OTR_LTX_8GB_CKPT", str(decoy))

    with pytest.raises(EngineUnusable) as e:
        eng._ckpt_path()
    assert "names a file the loader will NOT load" in str(e.value)


def test_CONTROL_a_missing_checkpoint_still_says_MISSING_MODEL(eng):
    """No override anywhere: the absent-weights message must not be displaced
    by the new MALFORMED_CONFIG one."""
    (eng._root / "models" / "checkpoints" / m._LTX8_DEFAULT_CKPT).unlink()
    with pytest.raises(EngineUnusable) as e:
        eng.assert_usable(host_caps={}, profile={})
    assert e.value.reason == EngineUsabilityReason.MISSING_MODEL
    assert "download_ltx_0_9_8" in str(e.value)


def test_CONTROL_by_token_IGNORES_the_DIR_by_contract(
        eng, tmp_path, monkeypatch):
    """The split's whole point: "where would the LOADER look" and "where does
    this build's preflight look" are different questions. If the new helper
    honoured `*_DIR` it would agree with itself and the tripwire could never
    fire."""
    monkeypatch.setenv("OTR_LTX_8GB_CKPT_DIR",
                       str(_elsewhere(tmp_path, m._LTX8_DEFAULT_CKPT)))
    assert eng._resolve_model_file_by_token(
        ("checkpoints",), m._LTX8_DEFAULT_CKPT) == _registered_ckpt(eng)


def test_CONTROL_wan_resolve_model_file_still_lets_a_DIR_win(
        tmp_path, monkeypatch):
    """`wan_shared._resolve_model_file` was REFACTORED (its tail split into
    `_resolve_model_file_by_token`) and must stay behaviour-identical: the Wan
    suites use `*_DIR` as their no-ComfyUI mock seam, and a DIR that wins on
    existence alone -- with nothing registered anywhere -- is exactly what
    those fixtures rely on. Fixing Wan's copy of this lie needs its fixtures
    migrated first; that is a separate chunk, and this control is what keeps
    it separate."""
    from nodes._otr_video_engines.eng_wan_i2v import WanI2VEngine

    d = tmp_path / "vae_dir"
    d.mkdir()
    (d / "some_vae.safetensors").write_bytes(b"v")
    monkeypatch.setenv("OTR_WAN_I2V_VAE_DIR", str(d))

    wan = WanI2VEngine()
    wan._comfy_root = lambda: str(tmp_path / "nowhere")
    assert wan._resolve_model_file(
        ("vae",), "some_vae.safetensors",
        "OTR_WAN_I2V_VAE_DIR") == str(d / "some_vae.safetensors")
