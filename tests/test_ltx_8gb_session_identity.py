"""B2b -- ``Ltx8gbEngine.session_identity()`` and the refusal it lifts.

Before this, ``session_identity`` existed in exactly ONE file
(``beat_session.py``) and NO adapter declared it, so ``BeatSession.open()``
refused every multi-segment beat for all 31 engines -- deliberately, with no
fallback. A 169- or 237-frame beat was rejected before the render canvas was
ever consulted. These tests pin both halves: the identity describes the
handles, and the refusal is genuinely lifted for this engine.
"""

from __future__ import annotations

import pytest

from nodes._otr_video_engines import beat_session as bs
from nodes._otr_video_engines import eng_ltx_8gb as m

#: EVERY env var reachable from `session_identity` -> `resolve_session_config`,
#: including the token names `_ckpt_name`/`_t5_name` read and the knobs that
#: come alive once a test sets the consent act. A short list lets the HOST
#: environment decide an outcome (QA finding T-6).
_ENVS = ("OTR_LTX_8GB_CKPT", "OTR_LTX_8GB_CKPT_DIR", "OTR_LTX_8GB_CKPT_NAME",
         "OTR_LTX_8GB_T5_DIR", "OTR_LTX_8GB_T5_NAME",
         "OTR_LTX_8GB_T5_DEVICE", "OTR_LTX_8GB_TILED_VAE",
         "OTR_LTX_8GB_STEPS", "OTR_LTX_8GB_CFG", "OTR_LTX_8GB_SAMPLER",
         "OTR_LTX_8GB_MAX_SHIFT", "OTR_LTX_8GB_BASE_SHIFT",
         "OTR_LTX_8GB_TERMINAL", "OTR_LTX_8GB_NEGATIVE",
         "OTR_LTX_8GB_VAE_TILE", "OTR_LTX_8GB_VAE_OVERLAP",
         "OTR_LTX_8GB_VAE_TEMPORAL", "OTR_LTX_8GB_VAE_TEMPORAL_OVERLAP",
         "OTR_LTX_8GB_MAX_FRAMES", m.PREQUALIFICATION_ENV)


def test_the_env_scrub_list_covers_every_frozen_knob():
    """Held to the claim three lines above: every knob the freeze names must be
    scrubbed, or the host box decides an outcome here."""
    assert set(m._RECIPE_ENV_KEYS.values()) <= set(_ENVS)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _ENVS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def eng(tmp_path, monkeypatch):
    (tmp_path / m._LTX8_DEFAULT_CKPT).write_bytes(b"c" * 2048)
    (tmp_path / m._LTX8_DEFAULT_T5).write_bytes(b"t" * 1024)
    e = m.Ltx8gbEngine()
    monkeypatch.setattr(
        e, "_resolve_model_file",
        lambda categories, name, env_dir: (
            str(tmp_path / name) if (tmp_path / name).exists() else None))
    e._tmp = tmp_path
    return e


# --- the refusal is lifted ------------------------------------------------- #
def test_beat_session_can_now_read_an_identity_for_this_engine(eng):
    ident = bs.session_identity(eng)
    assert ident is not None
    assert isinstance(ident, tuple)
    assert "ltx_8gb" in ident[0]


def test_identity_names_the_engine_the_recipe_and_both_weights(eng):
    ident = bs.session_identity(eng)
    blob = " ".join(ident)
    assert "ltx_8gb" in blob
    assert m.RECIPE_LTX8_I2V in blob
    assert m._LTX8_DEFAULT_CKPT in blob
    assert m._LTX8_DEFAULT_T5 in blob
    assert "cpu" in blob                    # the T5 offload is part of identity


# --- stability: it must not move across the load --------------------------- #
def test_identity_is_stable_across_repeated_reads_on_an_unchanged_box(eng):
    assert bs.session_identity(eng) == bs.session_identity(eng)


def test_identity_excludes_per_segment_state(eng, monkeypatch):
    """Prompt, seed, frame count and canvas change per SEGMENT -- if any of them
    leaked in, the identity would report drift on every segment boundary."""
    before = bs.session_identity(eng)
    monkeypatch.setenv("OTR_LTX_8GB_MAX_FRAMES", "65")
    assert bs.session_identity(eng) == before


def test_identity_excludes_tiled_vae_which_selects_a_node_class_not_a_handle(
        eng, monkeypatch):
    """PREQUALIFICATION IS SET ON PURPOSE. Since B6 the frozen recipe ignores
    OTR_LTX_8GB_TILED_VAE on a production leg, so without the flag this test
    would pass because the env did NOTHING -- proving the freeze, not the
    exclusion it is named for. With the knob genuinely open, an identity that
    still does not move is proving what it claims."""
    monkeypatch.setenv(m.PREQUALIFICATION_ENV, "1")
    before = bs.session_identity(eng)
    monkeypatch.setenv("OTR_LTX_8GB_TILED_VAE", "1")
    assert eng._tiled_vae() is True          # the lever really is live here
    assert bs.session_identity(eng) == before


# --- drift: it must move when the handles would ---------------------------- #
def test_a_swapped_checkpoint_moves_the_identity(eng):
    before = bs.session_identity(eng)
    (eng._tmp / m._LTX8_DEFAULT_CKPT).write_bytes(b"c" * 4096)
    assert bs.session_identity(eng) != before


def test_a_swapped_t5_moves_the_identity(eng):
    before = bs.session_identity(eng)
    (eng._tmp / m._LTX8_DEFAULT_T5).write_bytes(b"t" * 8192)
    assert bs.session_identity(eng) != before


def test_a_model_sampling_shift_moves_the_identity(eng, monkeypatch):
    """The shifts are baked into the hoisted patcher, so they ARE the handle.

    The env var is only this test's LEVER for moving a shift; since B6 it moves
    one under prequalification, which is the mode a sweep runs in. The property
    -- a different shift is a different handle -- is unchanged."""
    monkeypatch.setenv(m.PREQUALIFICATION_ENV, "1")
    before = bs.session_identity(eng)
    monkeypatch.setenv("OTR_LTX_8GB_MAX_SHIFT", "3.5")
    assert bs.session_identity(eng) != before


def test_moving_the_t5_off_cpu_moves_the_identity(eng, monkeypatch):
    """Same shape: prequalification is how the T5 device moves at all now."""
    monkeypatch.setenv(m.PREQUALIFICATION_ENV, "1")
    before = bs.session_identity(eng)
    monkeypatch.setenv("OTR_LTX_8GB_T5_DEVICE", "default")
    assert bs.session_identity(eng) != before


# --- an adapter that cannot describe its handles STOPS --------------------- #
def test_an_identity_read_that_cannot_resolve_its_weights_RAISES(eng, monkeypatch):
    """beat_session deliberately does not wrap this in try/except: an adapter
    that cannot describe its own handles has told us to stop, not to guess."""
    monkeypatch.setattr(eng, "_resolve_model_file",
                        lambda categories, name, env_dir: None)
    with pytest.raises(Exception) as e:
        bs.session_identity(eng)
    assert m._LTX8_DEFAULT_CKPT in str(e.value)


# --- CONTROL: engines that never open a multi-segment session stay silent --- #
def test_CONTROL_the_rollout_is_scoped_to_this_engine_only():
    """A package-wide rollout was explicitly cut: unbounded engines never open a
    multi-segment session, and other capped adapters need their own handle
    contracts. If this starts failing, someone widened the blast radius."""
    from nodes._otr_video_engines import eng_wan_ti2v

    wan = eng_wan_ti2v.WanTi2vEngine()
    assert bs.session_identity(wan) is None
