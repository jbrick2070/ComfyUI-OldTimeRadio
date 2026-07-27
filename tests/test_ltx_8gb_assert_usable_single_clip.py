"""The SINGLE-CLIP path finally gets the loader-token guard.

`resolve_session_config()` is reachable only from `session_identity()`, which
`BeatSession` calls only `if self.is_multi_segment`. So every existing test
proved the MULTI-SEGMENT path was protected, and none proved the single-clip
path -- the common case -- was. A QA lens reproduced the hole live: with a decoy
`OTR_LTX_8GB_CKPT` that also cleared the 4 GiB floor, `assert_usable()` returned
green while `resolve_session_config()` raised, in the same environment.

The fix is delegation, not deletion: `_ckpt_path` / `_t5_path` now route through
`_loader_token_path`, the one authority. `assert_usable`'s body is UNTOUCHED,
which is what keeps the 4 GiB integrity floor and the knob-check-first ordering
alive by construction rather than by remembering to port them -- the panel
warned that a wholesale "assert_usable = resolve_session_config()" rewrite would
have silently dropped the floor.

Every test here calls `assert_usable()` -- the real single-clip gate -- not the
resolver.
"""

from __future__ import annotations

import os

import pytest

from nodes._otr_video_engines import eng_ltx_8gb as m
from nodes._otr_video_engines.registry import EngineUnusable, EngineUsabilityReason

_ENVS = (
    "OTR_LTX_8GB_CKPT", "OTR_LTX_8GB_CKPT_DIR", "OTR_LTX_8GB_CKPT_NAME",
    "OTR_LTX_8GB_T5_DIR", "OTR_LTX_8GB_T5_NAME", "OTR_LTX_8GB_TILED_VAE",
    "OTR_LTX_8GB_T5_DEVICE", "OTR_LTX_8GB_STEPS", "OTR_LTX_8GB_MAX_FRAMES",
    "OTR_LTX_8GB_CFG", "OTR_LTX_8GB_SAMPLER", "OTR_LTX_8GB_MAX_SHIFT",
    "OTR_LTX_8GB_BASE_SHIFT", "OTR_LTX_8GB_TERMINAL", "OTR_LTX_8GB_NEGATIVE",
    "OTR_LTX_8GB_VAE_TILE", "OTR_LTX_8GB_VAE_OVERLAP",
    "OTR_LTX_8GB_VAE_TEMPORAL", "OTR_LTX_8GB_VAE_TEMPORAL_OVERLAP",
    m.PREQUALIFICATION_ENV,
)


def test_the_env_scrub_list_covers_every_frozen_knob():
    """A scrub list that stops growing with `_RECIPE_ENV_KEYS` re-opens the
    T-6 leak: a host var reaching the demotion warning these tests read."""
    assert set(m._RECIPE_ENV_KEYS.values()) <= set(_ENVS)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _ENVS:
        monkeypatch.delenv(name, raising=False)


#: The real floor is 4 GiB. Materialising 4 GiB files per test costs ~45s of
#: real I/O even sparse, for zero extra proof -- the property under test is
#: "the floor is still CHECKED", not the size of the constant. Shrink it.
_TEST_FLOOR = 1024


@pytest.fixture
def eng(tmp_path, monkeypatch):
    """A big-enough checkpoint and a T5, resolved by TOKEN."""
    monkeypatch.setattr(m, "_LTX8_CKPT_MIN_BYTES", _TEST_FLOOR)
    ckpt = tmp_path / m._LTX8_DEFAULT_CKPT
    ckpt.write_bytes(b"c" * (_TEST_FLOOR * 2))       # comfortably over the floor
    (tmp_path / m._LTX8_DEFAULT_T5).write_bytes(b"t" * 1024)
    e = m.Ltx8gbEngine()
    monkeypatch.setattr(
        e, "_resolve_model_file",
        lambda categories, name, env_dir: (
            str(tmp_path / name) if (tmp_path / name).exists() else None))
    e._tmp = tmp_path
    return e


# --- THE DEFECT, on the path that actually renders ------------------------- #
def test_assert_usable_REFUSES_a_decoy_override_on_the_single_clip_path(
        eng, tmp_path, monkeypatch):
    """The hole the QA lens reproduced: green preflight, lying identity.

    The decoy is deliberately ALSO over the 4 GiB floor, so the old size check
    could not have caught it -- the only thing that catches it is the token
    cross-check now reached through `_ckpt_path`.
    """
    decoy = tmp_path / "decoy_weight.safetensors"
    decoy.write_bytes(b"d" * (_TEST_FLOOR * 2))      # ALSO clears the floor
    monkeypatch.setenv("OTR_LTX_8GB_CKPT", str(decoy))

    with pytest.raises(EngineUnusable) as e:
        eng.assert_usable(host_caps={}, profile={})
    assert e.value.reason == EngineUsabilityReason.MALFORMED_CONFIG
    msg = str(e.value)
    assert "decoy_weight.safetensors" in msg
    assert m._LTX8_DEFAULT_CKPT in msg          # names BOTH files


def test_the_two_paths_can_no_longer_disagree(eng, tmp_path, monkeypatch):
    """Preflight and the resolver must reach the SAME verdict on one env.

    This is the actual invariant -- two authorities over one fact was the
    defect, not the individual guard.
    """
    decoy = tmp_path / "decoy_weight.safetensors"
    decoy.write_bytes(b"d" * (_TEST_FLOOR * 2))
    monkeypatch.setenv("OTR_LTX_8GB_CKPT", str(decoy))

    with pytest.raises(EngineUnusable):
        eng.assert_usable(host_caps={}, profile={})
    with pytest.raises(EngineUnusable):
        eng.resolve_session_config()


def test_load_fails_closed_rather_than_raising_from_a_predicate(
        eng, tmp_path, monkeypatch):
    """`_installed()` is a bool contract; the refusal must not escape it."""
    decoy = tmp_path / "decoy_weight.safetensors"
    decoy.write_bytes(b"\0" * 32)
    monkeypatch.setenv("OTR_LTX_8GB_CKPT", str(decoy))

    assert eng._installed() is False             # a predicate, not a raise
    with pytest.raises(RuntimeError) as e:
        eng.load()
    assert "not installed" in str(e.value)


# --- CONTROLS: what must NOT change ---------------------------------------- #
def test_CONTROL_the_4GiB_floor_is_still_enforced_through_assert_usable(
        eng, monkeypatch):
    """The panel's sharpest warning: a wholesale delegation would have DROPPED
    this. `resolve_session_config` has no size check at all, so if
    `assert_usable` had been rewritten to call it, an undersized (wrong or
    truncated) checkpoint would newly PASS. It had zero test coverage before."""
    undersized = _TEST_FLOOR // 2
    (eng._tmp / m._LTX8_DEFAULT_CKPT).write_bytes(b"c" * undersized)
    with pytest.raises(EngineUnusable) as e:
        eng.assert_usable(host_caps={}, profile={})
    assert e.value.reason == EngineUsabilityReason.MISSING_MODEL
    assert "%d bytes" % undersized in str(e.value)


def test_CONTROL_a_plain_box_with_no_override_still_passes(eng):
    """The overwhelmingly common configuration must be untouched."""
    assert eng.assert_usable(host_caps={}, profile={}) == "ltx_8gb"


def test_CONTROL_an_override_that_IS_the_token_resolution_still_passes(
        eng, monkeypatch):
    real = os.path.join(str(eng._tmp), m._LTX8_DEFAULT_CKPT)
    monkeypatch.setenv("OTR_LTX_8GB_CKPT", real)
    assert eng.assert_usable(host_caps={}, profile={}) == "ltx_8gb"


@pytest.mark.skipif(os.path.normcase("A") == "A",
                    reason="case-insensitive filesystems only")
def test_CONTROL_a_case_respelled_override_still_passes_through_assert_usable(
        eng, monkeypatch):
    real = os.path.join(str(eng._tmp), m._LTX8_DEFAULT_CKPT)
    monkeypatch.setenv("OTR_LTX_8GB_CKPT", real.upper())
    assert eng.assert_usable(host_caps={}, profile={}) == "ltx_8gb"


def test_CONTROL_a_malformed_knob_still_fails_FIRST_with_its_own_message(
        eng, monkeypatch):
    """Ordering is preserved by construction -- assert_usable's body was not
    touched -- but pin it, because nothing pinned it for assert_usable before.

    Scoped to prequalification by B6: that is the mode in which a recipe knob
    binds at all, and a knob that cannot bind cannot be malformed."""
    monkeypatch.setenv(m.PREQUALIFICATION_ENV, "1")
    monkeypatch.setenv("OTR_LTX_8GB_STEPS", "not-a-number")
    monkeypatch.setenv("OTR_LTX_8GB_CKPT", str(eng._tmp / "nope.safetensors"))
    with pytest.raises(EngineUnusable) as e:
        eng.assert_usable(host_caps={}, profile={})
    assert "OTR_LTX_8GB_STEPS" in str(e.value)   # the knob, not the path


def test_a_stale_malformed_knob_does_NOT_mask_the_REAL_failure_in_production(
        eng, monkeypatch):
    """B6 -- the inversion, on the preflight path this time.

    Per-segment preflight runs inside `render_driver._render_one`, so before B6
    a stale `OTR_LTX_8GB_STEPS` left in an already-booted server's environment
    would raise MALFORMED_CONFIG and NAME THE WRONG THING: the operator chases a
    knob that has no effect while the actual defect -- a checkpoint the loader
    cannot resolve -- goes unreported. The frozen recipe ignores the knob, so
    the error the operator sees is the one they can act on."""
    monkeypatch.setenv("OTR_LTX_8GB_STEPS", "not-a-number")   # no consent act
    monkeypatch.setenv("OTR_LTX_8GB_CKPT", str(eng._tmp / "nope.safetensors"))
    with pytest.raises(EngineUnusable) as e:
        eng.assert_usable(host_caps={}, profile={})
    assert "OTR_LTX_8GB_STEPS" not in str(e.value)
    assert m._LTX8_DEFAULT_CKPT in str(e.value)  # the path, not the knob


def test_an_honest_production_box_with_no_overrides_stays_usable(eng, caplog):
    """CONTROL on the freeze: the normal case is silent AND green. A warning on
    every leg of every episode would be noise the operator learns to ignore."""
    with caplog.at_level("WARNING", logger="OTR.video.ltx_8gb"):
        assert eng.assert_usable(host_caps={}, profile={}) == "ltx_8gb"
    # SILENCE, not `"FROZEN" not in ...` -- with no knob set nothing can log at
    # all, so a substring check would pass with the freeze deleted, the warning
    # reworded, or the warning removed. Scoped to THIS adapter's logger so an
    # unrelated warning elsewhere cannot make it red for the wrong reason.
    assert [r for r in caplog.records if r.name == "OTR.video.ltx_8gb"] == []


def test_CONTROL_a_missing_checkpoint_still_says_MISSING_MODEL(eng, monkeypatch):
    monkeypatch.setattr(eng, "_resolve_model_file",
                        lambda categories, name, env_dir: None)
    with pytest.raises(EngineUnusable) as e:
        eng.assert_usable(host_caps={}, profile={})
    assert e.value.reason == EngineUsabilityReason.MISSING_MODEL
    assert "download_ltx_0_9_8" in str(e.value)  # remediation text preserved


def test_CONTROL_a_missing_t5_still_says_offline(eng):
    (eng._tmp / m._LTX8_DEFAULT_T5).unlink()
    with pytest.raises(EngineUnusable) as e:
        eng.assert_usable(host_caps={}, profile={})
    assert e.value.reason == EngineUsabilityReason.MISSING_MODEL
    assert "offline" in str(e.value)
