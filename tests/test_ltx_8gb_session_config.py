"""B2a -- ``Ltx8gbEngine.resolve_session_config()``: resolve ONCE, fail CLOSED.

Two defects motivate this:

  * ``BeatSession`` asks for the session identity BEFORE ``prepare()`` installs
    the active profile, so an identity derived from live accessors describes
    different state before and after the load -- the identity would report drift
    against its own pre-load reading. One frozen resolution removes that.
  * ``_build_graph`` hands the loader nodes a BARE BASENAME, but ``_ckpt_path()``
    short-circuits on an explicit ``OTR_LTX_8GB_CKPT``. So a receipt taken from
    the env var can describe a DIFFERENT weight than the one the graph loads:
    preflight green, identity a lie, render from an unrecorded checkpoint.

Controls throughout: the honest configurations must still resolve, or the guard
is a blanket refusal rather than a fix.
"""

from __future__ import annotations

import os

import pytest

from nodes._otr_video_engines import eng_ltx_8gb as m
from nodes._otr_video_engines.registry import EngineUnusable

#: EVERY env var `resolve_session_config` reads, directly or through
#: `_resolve_render_config`. An incomplete list lets the HOST environment leak
#: into the fixture and quietly decide a test's outcome (QA finding T-6).
_ENVS = (
    "OTR_LTX_8GB_CKPT", "OTR_LTX_8GB_CKPT_DIR", "OTR_LTX_8GB_CKPT_NAME",
    "OTR_LTX_8GB_T5_DIR", "OTR_LTX_8GB_T5_NAME", "OTR_LTX_8GB_TILED_VAE",
    "OTR_LTX_8GB_T5_DEVICE", "OTR_LTX_8GB_STEPS", "OTR_LTX_8GB_MAX_FRAMES",
    "OTR_LTX_8GB_CFG", "OTR_LTX_8GB_SAMPLER", "OTR_LTX_8GB_MAX_SHIFT",
    "OTR_LTX_8GB_BASE_SHIFT", "OTR_LTX_8GB_TERMINAL", "OTR_LTX_8GB_NEGATIVE",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _ENVS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def eng(tmp_path, monkeypatch):
    """A real engine whose model files live in tmp_path, resolved by TOKEN."""
    ckpt = tmp_path / m._LTX8_DEFAULT_CKPT
    ckpt.write_bytes(b"c" * 2048)
    t5 = tmp_path / m._LTX8_DEFAULT_T5
    t5.write_bytes(b"t" * 1024)
    e = m.Ltx8gbEngine()

    def _resolve(categories, name, env_dir):
        cand = tmp_path / name
        return str(cand) if cand.exists() else None

    monkeypatch.setattr(e, "_resolve_model_file", _resolve)
    e._tmp = tmp_path
    return e


# --- the happy path is a complete, frozen description ---------------------- #
def test_resolves_every_field_the_handles_depend_on(eng):
    cfg = eng.resolve_session_config()
    assert cfg.engine == "ltx_8gb"
    assert cfg.recipe == m.RECIPE_LTX8_I2V
    assert cfg.ckpt_token == m._LTX8_DEFAULT_CKPT
    assert cfg.t5_token == m._LTX8_DEFAULT_T5
    assert cfg.ckpt_path.endswith(m._LTX8_DEFAULT_CKPT)
    assert cfg.t5_device == "cpu"          # load-bearing on 8GB, not an option
    assert cfg.tiled_vae is False
    assert cfg.steps == 8 and cfg.cfg == 1.0 and cfg.sampler == "euler"
    assert cfg.max_frames == m._LTX8_MAX_FRAMES_DEFAULT


def test_receipt_is_bounded_and_not_a_content_hash(eng):
    cfg = eng.resolve_session_config()
    name, size, mtime_ns = cfg.ckpt_receipt
    assert name == m._LTX8_DEFAULT_CKPT
    assert size == 2048
    assert isinstance(mtime_ns, int) and mtime_ns > 0


def test_the_config_is_immutable(eng):
    cfg = eng.resolve_session_config()
    with pytest.raises(AttributeError):
        cfg.steps = 99


def test_two_resolutions_of_an_unchanged_box_are_equal(eng):
    assert eng.resolve_session_config() == eng.resolve_session_config()


def test_a_swapped_weight_changes_the_receipt(eng):
    before = eng.resolve_session_config()
    ckpt = eng._tmp / m._LTX8_DEFAULT_CKPT
    ckpt.write_bytes(b"c" * 4096)          # same name, different weight
    after = eng.resolve_session_config()
    assert before.ckpt_receipt != after.ckpt_receipt
    assert before.ckpt_path == after.ckpt_path


# --- THE IDENTITY LIE ------------------------------------------------------ #
def test_an_explicit_ckpt_path_the_loader_would_not_load_is_TERMINAL(
        eng, tmp_path, monkeypatch):
    other = tmp_path / "some_other_weight.safetensors"
    other.write_bytes(b"x" * 2048)
    monkeypatch.setenv("OTR_LTX_8GB_CKPT", str(other))
    with pytest.raises(EngineUnusable) as e:
        eng.resolve_session_config()
    msg = str(e.value)
    assert "OTR_LTX_8GB_CKPT" in msg
    assert "some_other_weight.safetensors" in msg
    assert m._LTX8_DEFAULT_CKPT in msg      # names BOTH files, not just the bad one


def test_an_explicit_ckpt_path_that_IS_the_token_resolution_is_accepted(
        eng, monkeypatch):
    """CONTROL: the honest use of the override must still work."""
    real = os.path.join(str(eng._tmp), m._LTX8_DEFAULT_CKPT)
    monkeypatch.setenv("OTR_LTX_8GB_CKPT", real)
    cfg = eng.resolve_session_config()
    assert os.path.abspath(cfg.ckpt_path) == os.path.abspath(real)


def test_CONTROL_the_override_guard_tolerates_a_separator_respelling(
        eng, monkeypatch):
    """CONTROL: the same file named differently must not be called a lie."""
    real = os.path.join(str(eng._tmp), m._LTX8_DEFAULT_CKPT)
    monkeypatch.setenv("OTR_LTX_8GB_CKPT", real.replace(os.sep, "/"))
    cfg = eng.resolve_session_config()
    assert os.path.samefile(cfg.ckpt_path, real)


@pytest.mark.skipif(os.path.normcase("A") == "A",
                    reason="case-insensitive comparison only matters on a "
                           "case-insensitive filesystem (Windows)")
def test_CONTROL_the_override_guard_tolerates_a_CASE_respelling(
        eng, monkeypatch):
    """CONTROL, and the one that would have caught the shipped defect.

    NTFS is case-insensitive but `os.path.abspath` never folds case, so the
    original guard raised MALFORMED_CONFIG for `C:\\Models\\x` vs `c:\\models\\x`
    -- the SAME file -- on every multi-segment beat. The previous version of this
    test was named "case_and_separator_tolerant" and only ever varied the
    SEPARATOR, which abspath does normalise. The name promised exactly the
    coverage that was missing."""
    real = os.path.join(str(eng._tmp), m._LTX8_DEFAULT_CKPT)
    monkeypatch.setenv("OTR_LTX_8GB_CKPT", real.upper())
    cfg = eng.resolve_session_config()
    assert os.path.samefile(cfg.ckpt_path, real)


def test_CONTROL_the_override_guard_tolerates_a_redundant_path_spelling(
        eng, monkeypatch):
    """CONTROL: `dir\\.\\file` and `dir\\sub\\..\\file` name the same file."""
    real = os.path.join(str(eng._tmp), m._LTX8_DEFAULT_CKPT)
    noisy = os.path.join(str(eng._tmp), ".", m._LTX8_DEFAULT_CKPT)
    monkeypatch.setenv("OTR_LTX_8GB_CKPT", noisy)
    cfg = eng.resolve_session_config()
    assert os.path.samefile(cfg.ckpt_path, real)


def test_same_file_falls_back_when_a_side_does_not_exist(tmp_path):
    """`os.path.samefile` RAISES on a missing path, so it cannot stand alone.

    The fallback must still fold case and must still say NO for genuinely
    different names."""
    missing_a = str(tmp_path / "gone.safetensors")
    missing_b = str(tmp_path / "GONE.safetensors")
    other = str(tmp_path / "different.safetensors")
    assert m._same_file(missing_a, missing_a) is True
    assert m._same_file(missing_a, other) is False
    # On a case-insensitive box these are the same file; on a case-sensitive
    # one they are not. Assert the platform's own answer, not a guess.
    expected = os.path.normcase(missing_a) == os.path.normcase(missing_b)
    assert m._same_file(missing_a, missing_b) is expected


def test_a_receipt_for_a_file_that_vanished_is_NAMED_not_a_raw_OSError(
        eng, tmp_path, monkeypatch):
    """QA finding C-4: the stat between resolution and receipt was unguarded.

    Models the real race -- resolution succeeds, then the file is gone before the
    receipt -- by handing back a path that resolves but does not exist. Note this
    deliberately does NOT monkeypatch `os.stat`: that is process-wide and breaks
    pytest's own traceback machinery."""
    ghost = tmp_path / "ghost" / m._LTX8_DEFAULT_CKPT
    monkeypatch.setattr(eng, "_resolve_model_file",
                        lambda categories, name, env_dir: str(tmp_path / "ghost" / name))
    with pytest.raises(EngineUnusable) as e:
        eng.resolve_session_config()
    msg = str(e.value)
    assert "stat" in msg
    assert m._LTX8_DEFAULT_CKPT in msg
    assert "FileNotFoundError" in msg or "Error" in msg
    assert str(ghost.parent) in msg or "ghost" in msg


# --- fail closed on absence ------------------------------------------------ #
def test_a_missing_checkpoint_is_named_not_none(eng, monkeypatch):
    monkeypatch.setattr(eng, "_resolve_model_file",
                        lambda categories, name, env_dir: None)
    with pytest.raises(EngineUnusable) as e:
        eng.resolve_session_config()
    assert m._LTX8_DEFAULT_CKPT in str(e.value)


def test_a_missing_t5_is_named_and_says_offline(eng, monkeypatch):
    def _only_ckpt(categories, name, env_dir):
        if name == m._LTX8_DEFAULT_CKPT:
            return str(eng._tmp / name)
        return None

    monkeypatch.setattr(eng, "_resolve_model_file", _only_ckpt)
    with pytest.raises(EngineUnusable) as e:
        eng.resolve_session_config()
    msg = str(e.value)
    assert m._LTX8_DEFAULT_T5 in msg and "offline" in msg


def test_a_malformed_render_knob_fails_before_any_file_work(eng, monkeypatch):
    monkeypatch.setenv("OTR_LTX_8GB_STEPS", "not-a-number")
    with pytest.raises(EngineUnusable) as e:
        eng.resolve_session_config()
    assert "OTR_LTX_8GB_STEPS" in str(e.value)


# --- the levers are captured, so a later reader cannot disagree ------------ #
def test_tiled_vae_and_max_frames_are_captured_at_resolution_time(
        eng, monkeypatch):
    monkeypatch.setenv("OTR_LTX_8GB_TILED_VAE", "1")
    monkeypatch.setenv("OTR_LTX_8GB_MAX_FRAMES", "65")
    cfg = eng.resolve_session_config()
    assert cfg.tiled_vae is True
    assert cfg.max_frames == 65
    # The whole point: flipping the env AFTER resolution does not move the
    # frozen config, so identity and prepare cannot see different values.
    monkeypatch.setenv("OTR_LTX_8GB_TILED_VAE", "0")
    assert cfg.tiled_vae is True


def test_profile_argument_is_accepted_today_without_changing_the_result(eng):
    """B6 will consult it; the signature must not churn under callers later."""
    assert eng.resolve_session_config(profile={"video": {}}) == \
        eng.resolve_session_config()
