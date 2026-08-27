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


# --- CONTROL: the rollout is a NAMED LIST, never a package-wide sweep ------- #
#: Every adapter that declares a handle contract, and the chunk that gave it
#: one. A package-wide rollout was explicitly cut at B1b: an unbounded engine
#: never opens a multi-segment session, and every capped adapter needs its own
#: identity -- what its handles are, and which of them it hoists -- written and
#: measured for that adapter. So the list grows one ratified chunk at a time.
_ENGINES_WITH_A_SESSION = {
    # Ghost Signal (2026-08-22). The first member that is NOT REQUIRED to be
    # here: it declares `max_frames=0`, so the partitioner never splits it and
    # `test_multiclip_session_identity_roster` would not have caught its
    # absence. It declares one anyway because it DOES hold local handles across
    # prepare -> render_clip, and an identity that exists before it is needed
    # costs nothing, while the reverse was learned at 730s into a live leg.
    # The five non-haunted siblings retired 2026-08-23. The survivor's identity
    # carries the adapter file and the STRENGTH applied, so a sweep across
    # strengths can never let one arm reuse another arm's patched model.
    "animatediff15_v3_haunted_video",
    "ltx_8gb",                 # B1b / B2b, 2026-07-27
    "wan_ti2v",                 # WIRE-W3a, 2026-07-29
    "wan_ti2v",                # WIRE-W3b, 2026-07-29
    "fastwan_8gb",             # 2026-08-01 -- inherits WanTi2vEngine's, and its
                               # LoRA rides it for free: _wan_session_receipts is
                               # built from _loader_names() + _aux_loader_files(),
                               # so a swapped or resized LoRA opens a new session.
    "humo",                    # WIRE-W4a, 2026-07-29
    "humo_1.7B",               # WIRE-W4a -- inherits HuMoEngine's
    "humo_1.7B_169",           # WIRE-W4a -- inherits HuMoEngine's
    "humo_14B_169",            # WIRE-W4a -- inherits HuMoEngine's
    # The last two local splitters, added together 2026-07-29 after the live
    # 45-word campaign found the gap the hard way: leg `ltx_video` wrote a
    # script, minted its stills and assembled its audio, then refused at the
    # render gate 730s in with SessionIdentityUnavailable. `ltx_audio_in` had
    # the same hole one leg behind it. With these two the roster invariant is
    # closed for LOCAL engines, and
    # tests/test_multiclip_session_identity_roster.py now holds it from the
    # partitioner's side so the next capped adapter cannot be missed.
    "ltx_video",               # 2026-07-29, live campaign
    "ltx_audio_in",            # 2026-07-29, same fix (declared on _LtxAvBase)
    # LANE 19, 2026-08-12. The first engine to join this set by ARRIVING rather
    # than by being repaired -- and it did not get the choice. It splits (a
    # discrete menu topping out at ~15 s cannot cover a 30 s beat in one call)
    # and it holds local handles, so test_multiclip_session_identity_roster
    # refused it the moment it registered. That is the roster invariant closing
    # from the partitioner's side exactly as the note above predicted, one lane
    # after it was written. Its identity is engine + frozen recipe + a re-stat
    # of all three weights (never a hash of 21 GB).
    "minimax_h3_video",        # 2026-08-12, lane 19
    # LANE 20, 2026-08-12. Inherits the identity from `_MiniMaxH3Base` -- and
    # it is NOT the same tuple as its sibling's, because the identity is built
    # from `_weight_rows()`: this lane's names a different DiT and one more
    # weight, so the two lanes can never be mistaken for one session.
    "minimax_h3_audio_in",     # 2026-08-12, lane 20
    # LTX 2.5 CHUNK A, 2026-08-19. The SECOND engine to join by ARRIVING rather
    # than by being repaired, and like lane 19 it did not get the choice: its
    # contract declares `discrete_frames=(97,)`, so `can_split` is True, so
    # test_multiclip_session_identity_roster refused it the moment it
    # registered. Worth recording that the refusal came from the roster gate on
    # a lane that had never rendered -- which is the whole point of that gate.
    # The 2026-07-29 note above describes what it cost to learn the same thing
    # from a live leg: 730 seconds in, after the script, the stills and the
    # audio were already paid for.
    #
    # Its identity is engine + a re-stat of all FOUR weights, with no recipe
    # token -- unlike the LTX 2.3 lanes there is exactly one locked recipe
    # here, so the weights ARE the identity. The audio VAE is one of the four
    # even though the lane emits no audio, so a swapped audio VAE correctly
    # opens a new session.
    "ltx25_video",             # 2026-08-19, Chunk A
    # THE FOLEY LANE, 2026-08-26. It joins by INHERITANCE rather than by
    # arriving or being repaired -- the first of those in this set. It
    # subclasses Ltx25VideoEngine and overrides neither `session_identity` nor
    # `_weight_rows`, so it declares the same four-weight identity its parent
    # does, and the roster gate admitted it the moment it registered.
    #
    # AND THE SHARED IDENTITY IS CORRECT HERE, which is worth stating because
    # the H3 pair above is the opposite case. Lanes 19 and 20 must never be
    # mistaken for one session because they load DIFFERENT weights. These two
    # load exactly the same four artifacts and run exactly the same graph; the
    # foley lane only keeps an output its parent discards, after the session's
    # weights are already resolved. Two lanes with identical weights SHOULD
    # share a session -- that is what the identity is for.
    "ltx25_foley_plus",        # 2026-08-26, the foley bed
    # MIME, 2026-08-26. Joins by inheritance for the same reason and with the
    # same justification as its sibling directly above: identical weights,
    # identical graph, identical session. All three LTX 2.5 lanes differ only
    # in what happens to the model's audio AFTER the render.
    "ltx25_mime",              # 2026-08-26, the mime lane
}


def test_CONTROL_the_rollout_is_a_NAMED_LIST_and_this_is_the_list():
    """If this starts failing, someone widened the blast radius -- or narrowed
    it -- without saying so here.

    This used to assert that ``wan_ti2v`` alone stayed silent, which made it a
    control over exactly one engine: ``wan_ti2v`` gained an identity at WIRE-W3a
    and no test in this file noticed. Asserting the whole SET is the control
    that was meant, and it fails in both directions.
    """
    import nodes._otr_video_engines  # noqa: F401  -- populate the registry
    from nodes._otr_video_engines import registry as vreg

    declared = set()
    for name in vreg.all_engine_names():
        engine = vreg.get_engine(name)
        # An identity read that RAISES has still been DECLARED -- the adapter
        # is telling us it cannot describe handles it does offer, which is a
        # different failure and one this file tests elsewhere.
        if getattr(engine, "session_identity", None) is None:
            continue
        try:
            if bs.session_identity(engine) is not None:
                declared.add(name)
        except Exception:                    # noqa: BLE001 -- declared, unreadable
            declared.add(name)
    assert declared == _ENGINES_WITH_A_SESSION, (
        "the set of adapters declaring a beat-session identity is %r, not the "
        "ratified %r -- add the engine and its chunk to _ENGINES_WITH_A_SESSION "
        "in the same commit that gives it one"
        % (sorted(declared), sorted(_ENGINES_WITH_A_SESSION)))
