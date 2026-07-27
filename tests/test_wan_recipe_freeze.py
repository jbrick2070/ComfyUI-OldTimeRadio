"""LANE 1 -- the WAN render-recipe freeze (mirrors B6 for the WAN adapters).

The defect: ``PBUG-20260723-02``. A production episode is submitted to an
ALREADY-BOOTED ComfyUI server, so a knob exported at launch cannot bind the
work -- yet both WAN adapters read their whole render recipe from
``os.environ`` on every leg, and neither emitted a ``recipe`` receipt, so a WAN
clip stamped ``recipe: None`` and there was not even a wrong receipt to catch
the drift with.

What is proven here: the recipe binds from CODE on a production leg; a stale
malformed knob is NAMED and never PARSED (so it cannot kill a leg it does not
influence); the consent act re-opens the knobs range-checked and fail-closed for
a measurement run; and a measurement run MARKS ITS OWN ARTIFACTS so a sweep clip
is not mistakable for a published one in the durable ledger.

THE TEST RULE THIS FILE FOLLOWS, from the B6 panel: every override must OPPOSE
the frozen value and the test must assert what it opposes. Six tests went
decorative on the ltx v2 flip because their overrides happened to AGREE with the
new frozen value, leaving them unable to tell whether the recipe or the
environment had won.

CPU-only. No GPU, no model load, no ComfyUI node registry.
"""

from __future__ import annotations

import logging

import pytest

from nodes._otr_video_engines import wan_recipe as wr
from nodes._otr_video_engines.eng_wan_ti2v import (
    PREQUALIFICATION_ENV, RECIPE_WAN_TI2V, WAN_TI2V_RECIPE, WanTi2vEngine,
)
from nodes._otr_video_engines.registry import (
    EngineUnusable, EngineUsabilityReason,
)

_TI2V_LOGGER = "OTR.video.wan_ti2v"

#: Every env knob the ti2v freeze demotes. Cleared before each test so a test
#: states its own leg rather than inheriting the parent process's environment.
_TI2V_RECIPE_ENVS = (
    "OTR_WAN_TI2V_STEPS", "OTR_WAN_TI2V_CFG", "OTR_WAN_TI2V_SHIFT",
    "OTR_WAN_TI2V_SAMPLER", "OTR_WAN_TI2V_SCHEDULER", "OTR_WAN_TI2V_NEGATIVE",
    "OTR_WAN_TI2V_TILED_VAE", "OTR_WAN_TI2V_VAE_TILE",
    "OTR_WAN_TI2V_VAE_OVERLAP", "OTR_WAN_TI2V_VAE_TEMPORAL",
    "OTR_WAN_TI2V_VAE_TEMPORAL_OVERLAP",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in _TI2V_RECIPE_ENVS + (PREQUALIFICATION_ENV, "OTR_WAN_TI2V_MAX_FRAMES"):
        monkeypatch.delenv(k, raising=False)


def _graph(eng=None):
    """The ti2v graph off a bare request -- no filesystem, no node registry."""
    eng = eng or WanTi2vEngine()
    req = {"text_prompt": "a slow pan", "canvas": {"w": 832, "h": 480}}
    return eng._build_graph(req, "init.png", {"seed": 7}, 81, 832, 480)


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.name == _TI2V_LOGGER]


# --------------------------------------------------------------------------- #
# the recipe's own shape -- pinned against LITERALS, not against itself
# --------------------------------------------------------------------------- #
def test_the_frozen_recipe_carries_exactly_these_fields():
    # Compared to a LITERAL set, not to another expression derived from the same
    # dict: the B6 panel found a key-set test that compared two sets to each
    # other and stayed green when a whole branch was deleted.
    assert set(WAN_TI2V_RECIPE) == {
        "steps", "cfg", "shift", "sampler", "scheduler", "negative",
        "tiled_vae", "vae_tile", "vae_overlap", "vae_temporal",
        "vae_temporal_overlap",
    }


def test_every_recipe_field_has_a_named_env_key_to_demote():
    # The T-6 leak class: a scrub/demotion list whose own comment claims
    # completeness it does not have. If a future v2 adds a field and forgets its
    # env name, the demotion notice would silently stop naming that knob.
    from nodes._otr_video_engines.eng_wan_ti2v import _RECIPE_ENV_KEYS
    assert set(_RECIPE_ENV_KEYS) == set(WAN_TI2V_RECIPE)
    assert set(_RECIPE_ENV_KEYS.values()) == set(_TI2V_RECIPE_ENVS)


def test_max_frames_is_NOT_in_the_recipe():
    """The ceiling keeps its own channel, and for WAN that is load-bearing.

    ``config/profiles/otr_8gb_wan.json`` sets BOTH
    ``launch.env.OTR_WAN_TI2V_MAX_FRAMES`` and ``video.max_render_frames``.
    Folding the ceiling into the frozen recipe would silently retire the 8GB
    tier's shipped launch contract -- the very bug this freeze descends from."""
    assert "max_frames" not in WAN_TI2V_RECIPE
    assert not any("MAX_FRAMES" in e for e in _TI2V_RECIPE_ENVS)


def test_the_ceiling_still_binds_on_a_PRODUCTION_leg(monkeypatch):
    # 49 = 4*12+1, well under the 177 engine max, so it can only come from the
    # env pin. If the freeze ever swallowed this knob the 8GB tier would
    # silently inherit 177 again (the 2026-07-23 wan_8gb failure).
    from nodes._otr_video_engines import motion_common as mc
    monkeypatch.setattr(mc, "free_vram_mb", lambda: None)
    monkeypatch.setenv("OTR_WAN_TI2V_MAX_FRAMES", "49")
    assert WanTi2vEngine()._floor_length(177) == 49


# --------------------------------------------------------------------------- #
# the consent act -- explicit, never ambient
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw", ["1", "true", "yes", "on", "TRUE", " On "])
def test_these_spellings_open_the_knobs(monkeypatch, raw):
    monkeypatch.setenv(PREQUALIFICATION_ENV, raw)
    assert wr.prequalification_active(PREQUALIFICATION_ENV) is True


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", "", "  ", "maybe"])
def test_present_but_falsy_is_a_PRODUCTION_leg(monkeypatch, raw):
    """A signal you can arrive at by accident is one a production leg can
    arrive at by accident. Anything that is not an explicit yes is production."""
    monkeypatch.setenv(PREQUALIFICATION_ENV, raw)
    assert wr.prequalification_active(PREQUALIFICATION_ENV) is False


def test_PREQUALIFICATION_ENV_is_the_var_the_resolver_actually_CONSULTS(
        monkeypatch):
    """Not a literal pinned to itself.

    An earlier draft of this test asserted ``PREQUALIFICATION_ENV`` equalled its
    own source literal, which no implementation change could ever break. What
    matters is that the exported name is the one the adapter reads: an operator
    who sets the documented var must actually get a measurement run. Setting it
    and observing the BEHAVIOUR change is what proves that."""
    monkeypatch.setenv("OTR_WAN_TI2V_STEPS", "11")
    assert WAN_TI2V_RECIPE["steps"] != 11            # the override OPPOSES
    assert WanTi2vEngine()._resolve_render_config()["steps"] != 11
    monkeypatch.setenv(PREQUALIFICATION_ENV, "1")
    assert WanTi2vEngine()._resolve_render_config()["steps"] == 11


def test_both_legs_return_the_SAME_KEY_SET(monkeypatch):
    """A return shape that varied by mode would hand the next reader a KeyError
    that only reproduces under the consent act.

    The production branch is spelled out rather than sliced from the recipe
    dict precisely to keep these equal; this is the trip-wire on anyone
    "simplifying" it back to ``dict(WAN_TI2V_RECIPE)``."""
    production = set(WanTi2vEngine()._resolve_render_config())
    monkeypatch.setenv(PREQUALIFICATION_ENV, "1")
    assert set(WanTi2vEngine()._resolve_render_config()) == production
    # Pinned against a LITERAL too, so deleting a key from BOTH branches cannot
    # keep this green by making two derived sets agree with each other.
    assert production == {"steps", "cfg", "shift", "sampler", "scheduler"}


# --------------------------------------------------------------------------- #
# the demotion notice -- NAMED, never PARSED, and its DIRECTION pinned
# --------------------------------------------------------------------------- #
def test_a_production_leg_NAMES_what_it_is_ignoring(monkeypatch, caplog):
    monkeypatch.setenv("OTR_WAN_TI2V_STEPS", "7")
    monkeypatch.setenv("OTR_WAN_TI2V_CFG", "9.5")
    with caplog.at_level(logging.WARNING, logger=_TI2V_LOGGER):
        WanTi2vEngine()._resolve_render_config()
    text = " ".join(_warnings(caplog))
    assert "OTR_WAN_TI2V_STEPS" in text and "OTR_WAN_TI2V_CFG" in text
    assert RECIPE_WAN_TI2V in text
    # DIRECTION. Both branches name the same knobs, interpolate the same recipe
    # and contain "PREQUALIFICATION" (it is inside the env var's own name), so
    # a test asserting only those parts stays GREEN when the two bodies are
    # SWAPPED. The markers are what tell them apart.
    assert wr.FROZEN_MARKER in text
    assert wr.MEASUREMENT_MARKER not in text


def test_a_measurement_run_says_what_it_HONOURED(monkeypatch, caplog):
    monkeypatch.setenv(PREQUALIFICATION_ENV, "1")
    monkeypatch.setenv("OTR_WAN_TI2V_STEPS", "7")
    with caplog.at_level(logging.WARNING, logger=_TI2V_LOGGER):
        WanTi2vEngine()._resolve_render_config()
    text = " ".join(_warnings(caplog))
    assert "OTR_WAN_TI2V_STEPS" in text
    assert wr.MEASUREMENT_MARKER in text
    assert wr.FROZEN_MARKER not in text


def test_a_clean_environment_says_NOTHING(monkeypatch, caplog):
    # Scoped to this adapter's logger and asserting SILENCE: the B6 panel found
    # an `assert "FROZEN" not in caplog.text` that was vacuous, because with no
    # knob set nothing can log at all -- it passed with the freeze deleted.
    with caplog.at_level(logging.WARNING, logger=_TI2V_LOGGER):
        WanTi2vEngine()._resolve_render_config()
    assert _warnings(caplog) == []


def test_an_exported_but_EMPTY_knob_is_not_an_override(monkeypatch, caplog):
    monkeypatch.setenv("OTR_WAN_TI2V_STEPS", "")
    with caplog.at_level(logging.WARNING, logger=_TI2V_LOGGER):
        WanTi2vEngine()._resolve_render_config()
    assert _warnings(caplog) == []


# --------------------------------------------------------------------------- #
# the recipe REACHES the graph -- proven with values that DIFFER from frozen
# --------------------------------------------------------------------------- #
def test_the_frozen_recipe_reaches_the_nodes_that_consume_it():
    g = _graph()
    ks = g["ksampler"]["inputs"]
    assert ks["steps"] == WAN_TI2V_RECIPE["steps"]
    assert ks["cfg"] == WAN_TI2V_RECIPE["cfg"]
    assert ks["sampler_name"] == WAN_TI2V_RECIPE["sampler"]
    assert ks["scheduler"] == WAN_TI2V_RECIPE["scheduler"]
    assert g["modelsampling"]["inputs"]["shift"] == WAN_TI2V_RECIPE["shift"]
    assert g["neg"]["inputs"]["text"] == WAN_TI2V_RECIPE["negative"]


def test_the_resolved_values_reach_the_graph_and_are_not_hard_coded(monkeypatch):
    """The trap that made the ltx version of this test decorative.

    Post-freeze a clean environment makes the resolver return the frozen
    constants -- so a hard-coded literal in ``_build_graph`` would compare EQUAL
    to the recipe and the test would pass while proving nothing. This runs under
    the consent act with values that DIFFER from every frozen one, and asserts
    they differ BEFORE comparing."""
    monkeypatch.setenv(PREQUALIFICATION_ENV, "1")
    monkeypatch.setenv("OTR_WAN_TI2V_STEPS", "11")
    monkeypatch.setenv("OTR_WAN_TI2V_CFG", "2.25")
    monkeypatch.setenv("OTR_WAN_TI2V_SHIFT", "3.75")
    monkeypatch.setenv("OTR_WAN_TI2V_NEGATIVE", "a distinct negative")
    # Prove the overrides OPPOSE the frozen values -- otherwise this test could
    # not tell the environment from the recipe.
    assert WAN_TI2V_RECIPE["steps"] != 11
    assert WAN_TI2V_RECIPE["cfg"] != 2.25
    assert WAN_TI2V_RECIPE["shift"] != 3.75
    assert WAN_TI2V_RECIPE["negative"] != "a distinct negative"
    g = _graph()
    assert g["ksampler"]["inputs"]["steps"] == 11
    assert g["ksampler"]["inputs"]["cfg"] == 2.25
    assert g["modelsampling"]["inputs"]["shift"] == 3.75
    assert g["neg"]["inputs"]["text"] == "a distinct negative"


def test_the_environment_can_NOT_author_the_negative_on_a_production_leg(
        monkeypatch):
    """This adapter has no per-shot negative channel, so before the freeze the
    server's boot environment was the SOLE author of the negative
    conditioning -- two boxes rendered visibly different clips from the same
    episode and both stamped the same (empty) receipt."""
    monkeypatch.setenv("OTR_WAN_TI2V_NEGATIVE", "a distinct negative")
    assert WAN_TI2V_RECIPE["negative"] != "a distinct negative"
    assert _graph()["neg"]["inputs"]["text"] == WAN_TI2V_RECIPE["negative"]


# --------------------------------------------------------------------------- #
# the tiled-decode geometry -- it used to be the ONE knob that failed OPEN
# --------------------------------------------------------------------------- #
def test_the_tile_geometry_is_frozen_on_a_production_leg(monkeypatch):
    monkeypatch.setenv("OTR_WAN_TI2V_VAE_TILE", "1024")
    monkeypatch.setenv("OTR_WAN_TI2V_VAE_TEMPORAL", "64")
    assert WAN_TI2V_RECIPE["vae_tile"] != 1024      # the override OPPOSES
    assert WAN_TI2V_RECIPE["vae_temporal"] != 64
    vd = _graph()["vaedecode"]["inputs"]
    assert vd["tile_size"] == WAN_TI2V_RECIPE["vae_tile"]
    assert vd["temporal_size"] == WAN_TI2V_RECIPE["vae_temporal"]


def test_the_tile_geometry_binds_under_the_consent_act(monkeypatch):
    monkeypatch.setenv(PREQUALIFICATION_ENV, "1")
    monkeypatch.setenv("OTR_WAN_TI2V_VAE_TILE", "1024")
    assert WAN_TI2V_RECIPE["vae_tile"] != 1024
    assert _graph()["vaedecode"]["inputs"]["tile_size"] == 1024


def test_a_mistyped_tile_value_now_fails_CLOSED(monkeypatch):
    """It used to swallow the error and substitute the default.

    That made these four the only knobs on this adapter that failed OPEN: a
    sweep could mistype the value it was measuring, render at something else,
    and stamp a receipt saying it had measured it."""
    monkeypatch.setenv(PREQUALIFICATION_ENV, "1")
    monkeypatch.setenv("OTR_WAN_TI2V_VAE_TILE", "not-a-number")
    with pytest.raises(EngineUnusable) as exc:
        _graph()
    assert exc.value.reason is EngineUsabilityReason.MALFORMED_CONFIG
    assert "OTR_WAN_TI2V_VAE_TILE" in str(exc.value)


def test_a_tile_value_under_the_NODES_OWN_floor_is_refused_by_name(monkeypatch):
    # VAEDecodeTiled declares tile_size min 64 (live /object_info capture). A
    # value under the node's own floor is a render that dies inside ComfyUI, so
    # it is refused here by name instead of failing late.
    monkeypatch.setenv(PREQUALIFICATION_ENV, "1")
    monkeypatch.setenv("OTR_WAN_TI2V_VAE_TILE", "16")
    with pytest.raises(EngineUnusable) as exc:
        _graph()
    assert exc.value.reason is EngineUsabilityReason.MALFORMED_CONFIG
    assert "out of range" in str(exc.value)


def test_an_unrecognised_tiled_vae_value_fails_CLOSED(monkeypatch):
    # It used to collapse to False, so a sweep that mistyped the knob it was
    # varying would decode untiled and stamp a receipt saying it had measured
    # tiled.
    monkeypatch.setenv(PREQUALIFICATION_ENV, "1")
    monkeypatch.setenv("OTR_WAN_TI2V_TILED_VAE", "maybe")
    with pytest.raises(EngineUnusable) as exc:
        WanTi2vEngine()._tiled_vae()
    assert exc.value.reason is EngineUsabilityReason.MALFORMED_CONFIG


def test_an_exported_empty_tiled_vae_does_not_force_the_knob_off(monkeypatch):
    # `get(name, dflt)` returns "" for an exported-empty var -- not truthy --
    # which would read OFF against a frozen default of ON. Every accessor in
    # this build treats empty as unset.
    monkeypatch.setenv(PREQUALIFICATION_ENV, "1")
    monkeypatch.setenv("OTR_WAN_TI2V_TILED_VAE", "")
    assert WAN_TI2V_RECIPE["tiled_vae"] is True
    assert WanTi2vEngine()._tiled_vae() is True


# --------------------------------------------------------------------------- #
# the receipt -- a measurement run MARKS ITS OWN ARTIFACTS
# --------------------------------------------------------------------------- #
def test_the_receipt_VERSION_cannot_drift_from_the_recipe_it_names():
    """The real failure this guards, not a literal pinned to itself.

    Bumping a recipe means adding ``WAN_TI2V_RECIPE_V2`` and repointing
    ``WAN_TI2V_RECIPE`` -- never editing a versioned dict in place, or receipts
    already stamped on disk stop being interpretable. The way that goes wrong is
    landing the new dict and FORGETTING the version inside the receipt string,
    which leaves two different recipes stamping one name in the durable ledger.
    So: the number of versioned dicts must equal the version in the receipt."""
    from nodes._otr_video_engines import eng_wan_ti2v as m
    versions = sorted(int(n.rsplit("_V", 1)[1]) for n in vars(m)
                      if n.startswith("WAN_TI2V_RECIPE_V")
                      and n.rsplit("_V", 1)[1].isdigit())
    assert versions, "no versioned recipe dict found"
    assert RECIPE_WAN_TI2V.endswith("_v%d" % max(versions))
    # And the active binding must BE the newest versioned dict, not a copy.
    assert m.WAN_TI2V_RECIPE is getattr(m, "WAN_TI2V_RECIPE_V%d" % max(versions))


def test_a_production_clip_stamps_the_frozen_name():
    assert wr.recipe_receipt(RECIPE_WAN_TI2V, PREQUALIFICATION_ENV) \
        == RECIPE_WAN_TI2V


def test_a_measurement_clip_stamps_a_DISTINGUISHABLE_receipt(monkeypatch):
    """Under the consent act the knobs genuinely bind, so a sweep's clip may
    share none of the frozen values -- while the receipt rides into a DURABLE
    ledger a published episode carries. Stamping the frozen name onto a sweep
    artifact would make a measurement indistinguishable from production in the
    one record that outlives the run."""
    monkeypatch.setenv(PREQUALIFICATION_ENV, "1")
    stamped = wr.recipe_receipt(RECIPE_WAN_TI2V, PREQUALIFICATION_ENV)
    assert stamped != RECIPE_WAN_TI2V
    assert stamped == RECIPE_WAN_TI2V + wr.PREQUALIFICATION_RECIPE_SUFFIX


def test_the_receipt_reaches_the_canonical_clip_dict():
    # The hop that matters: render_clip -> raw -> _clip_from_raw -> the manifest
    # row -> stamp_durable(meta.render_engines). A WAN clip stamped None here
    # before the freeze, so there was not even a wrong receipt to catch a drift.
    clip = WanTi2vEngine()._clip_from_raw(
        {"out_path": "/x/y.mp4", "frame_count": 81,
         "recipe": RECIPE_WAN_TI2V}, {"shot_id": "b004"})
    assert clip["recipe"] == RECIPE_WAN_TI2V


def test_the_clip_dict_tolerates_a_raw_without_a_receipt():
    # Every consumer already uses clip.get(...), and an engine that does not
    # stamp one must not KeyError the canonicaliser.
    clip = WanTi2vEngine()._clip_from_raw(
        {"out_path": "/x/y.mp4", "frame_count": 81}, {"shot_id": "b004"})
    assert clip["recipe"] is None


# --------------------------------------------------------------------------- #
# gaps the pre-push QA fan-out found in THIS file (2026-07-27, lens B)
# --------------------------------------------------------------------------- #
def test_the_scheduler_override_reaches_the_graph_under_the_consent_act(
        monkeypatch):
    """The scheduler had no opposing-override test, so a hard-coded "simple" in
    ``_build_graph`` would have compared EQUAL to the frozen value and stayed
    green -- the resolver-against-itself trap, surviving on one field.

    ``beta`` is used because it is in ``_PORTABLE_SCHEDULERS`` yet differs from
    the frozen ``simple``. The SAMPLER cannot get the same test: its whitelist
    has exactly one member, so no legal value opposes the frozen one. That is
    inherent, not an oversight -- if the whitelist ever grows, add the twin."""
    monkeypatch.setenv(PREQUALIFICATION_ENV, "1")
    monkeypatch.setenv("OTR_WAN_TI2V_SCHEDULER", "beta")
    assert WAN_TI2V_RECIPE["scheduler"] != "beta"
    assert "beta" in WanTi2vEngine()._PORTABLE_SCHEDULERS
    assert _graph()["ksampler"]["inputs"]["scheduler"] == "beta"


def test_an_exported_EMPTY_text_knob_is_not_an_override(monkeypatch):
    """``config_flag`` got a dedicated empty-string test; ``config_text`` is a
    DIFFERENT function with the same ``or``-based fallback and had none. A
    regression to ``get(env, frozen)`` would push an EMPTY negative prompt into
    the graph under a measurement run instead of falling back to the recipe."""
    monkeypatch.setenv(PREQUALIFICATION_ENV, "1")
    monkeypatch.setenv("OTR_WAN_TI2V_NEGATIVE", "")
    monkeypatch.setenv("OTR_WAN_TI2V_SCHEDULER", "")
    g = _graph()
    assert g["neg"]["inputs"]["text"] == WAN_TI2V_RECIPE["negative"]
    assert g["ksampler"]["inputs"]["scheduler"] == WAN_TI2V_RECIPE["scheduler"]


#: Each tile knob, the graph input it feeds, and a legal value that OPPOSES the
#: frozen one. All four are listed because the four share one ``_i(key)``
#: closure: with only two covered, swapping two entries in ``_RECIPE_ENV_KEYS``
#: would have gone undetected by the whole suite.
_TILE_CASES = (
    ("vae_tile", "OTR_WAN_TI2V_VAE_TILE", "tile_size", 1024),
    ("vae_overlap", "OTR_WAN_TI2V_VAE_OVERLAP", "overlap", 96),
    ("vae_temporal", "OTR_WAN_TI2V_VAE_TEMPORAL", "temporal_size", 64),
    ("vae_temporal_overlap", "OTR_WAN_TI2V_VAE_TEMPORAL_OVERLAP",
     "temporal_overlap", 12),
)


@pytest.mark.parametrize("key,env,graph_input,opposing", _TILE_CASES)
def test_each_tile_knob_maps_to_its_OWN_env_name(monkeypatch, key, env,
                                                 graph_input, opposing):
    # Proves the key -> env -> graph-input correspondence one knob at a time.
    # A set-vs-set check over _RECIPE_ENV_KEYS cannot see a permutation; this
    # can, because each case names all three ends of the mapping.
    monkeypatch.setenv(PREQUALIFICATION_ENV, "1")
    monkeypatch.setenv(env, str(opposing))
    assert WAN_TI2V_RECIPE[key] != opposing
    vd = _graph()["vaedecode"]["inputs"]
    assert vd[graph_input] == opposing
    # ...and every OTHER tile input stayed on its frozen value, so a knob that
    # writes into the wrong slot is caught rather than averaged out.
    for other_key, _e, other_input, _o in _TILE_CASES:
        if other_input != graph_input:
            assert vd[other_input] == WAN_TI2V_RECIPE[other_key]
