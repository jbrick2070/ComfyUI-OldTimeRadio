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
import re

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


# =========================================================================== #
# wan_i2v (the 14B lane) -- LANE 1b
#
# Same mechanism, its OWN data. Before the freeze this adapter read six knobs
# INLINE in _build_graph with bare int()/float(): no range check, no named
# refusal, so a mistyped value surfaced as a raw ValueError from the middle of a
# graph build with nothing naming the knob that caused it.
# =========================================================================== #
from nodes._otr_video_engines.eng_wan_i2v import (          # noqa: E402
    PREQUALIFICATION_ENV as I2V_PREQUAL_ENV,
    RECIPE_WAN_I2V, WAN_I2V_RECIPE, WanI2VEngine,
)

_I2V_LOGGER = "OTR.video.wan_i2v"
_I2V_RECIPE_ENVS = (
    "OTR_WAN_STEPS", "OTR_WAN_CFG", "OTR_WAN_SHIFT", "OTR_WAN_SAMPLER",
    "OTR_WAN_SCHEDULER", "OTR_WAN_NEGATIVE",
)


@pytest.fixture(autouse=True)
def _clean_i2v_env(monkeypatch):
    for k in _I2V_RECIPE_ENVS + (I2V_PREQUAL_ENV,):
        monkeypatch.delenv(k, raising=False)


def _i2v_graph(eng=None):
    eng = eng or WanI2VEngine()
    req = {"text_prompt": "a slow pan", "canvas": {"w": 832, "h": 480}}
    return eng._build_graph(req, "init.png", {"seed": 7}, 81, 832, 480)


def test_the_two_wan_adapters_do_not_share_one_consent_switch():
    """One switch for both tiers would open the OTHER adapter's knobs and stamp
    ``+prequalification`` on a clip that had rendered with its frozen recipe.
    A receipt that lies in the safer direction still lies."""
    assert PREQUALIFICATION_ENV != I2V_PREQUAL_ENV


def test_opening_one_adapters_consent_does_NOT_open_the_others(monkeypatch):
    # The behavioural half of the test above -- the part a name comparison
    # cannot prove.
    monkeypatch.setenv(I2V_PREQUAL_ENV, "1")
    monkeypatch.setenv("OTR_WAN_TI2V_STEPS", "11")
    assert WAN_TI2V_RECIPE["steps"] != 11
    assert WanTi2vEngine()._resolve_render_config()["steps"] \
        == WAN_TI2V_RECIPE["steps"]
    assert wr.recipe_receipt(RECIPE_WAN_TI2V, PREQUALIFICATION_ENV) \
        == RECIPE_WAN_TI2V


def test_i2v_recipe_shape_and_env_map():
    assert set(WAN_I2V_RECIPE) == {
        "steps", "cfg", "shift", "sampler", "scheduler", "negative"}
    from nodes._otr_video_engines.eng_wan_i2v import _RECIPE_ENV_KEYS
    assert set(_RECIPE_ENV_KEYS) == set(WAN_I2V_RECIPE)
    assert set(_RECIPE_ENV_KEYS.values()) == set(_I2V_RECIPE_ENVS)


def test_i2v_both_legs_return_the_SAME_KEY_SET(monkeypatch):
    production = set(WanI2VEngine()._resolve_render_config())
    monkeypatch.setenv(I2V_PREQUAL_ENV, "1")
    assert set(WanI2VEngine()._resolve_render_config()) == production
    assert production == {"steps", "cfg", "shift", "sampler", "scheduler",
                          "negative"}


def test_i2v_keeps_its_OWN_sampler_which_is_not_the_ti2v_floor():
    """The freeze preserves behaviour; it does not add policy.

    wan_ti2v is the 8GB/Mac/AMD floor and carries a portability whitelist that
    admits only ``euler``. The 14B lane never had one and ships ``uni_pc``.
    Freezing it to the floor's value would have silently changed what this
    engine renders."""
    assert WAN_I2V_RECIPE["sampler"] == "uni_pc"
    assert WAN_I2V_RECIPE["sampler"] != WAN_TI2V_RECIPE["sampler"]
    assert _i2v_graph()["ksampler"]["inputs"]["sampler_name"] == "uni_pc"


def test_i2v_frozen_recipe_reaches_the_graph():
    g = _i2v_graph()
    assert g["ksampler"]["inputs"]["steps"] == WAN_I2V_RECIPE["steps"]
    assert g["ksampler"]["inputs"]["cfg"] == WAN_I2V_RECIPE["cfg"]
    assert g["ksampler"]["inputs"]["scheduler"] == WAN_I2V_RECIPE["scheduler"]
    assert g["modelsampling"]["inputs"]["shift"] == WAN_I2V_RECIPE["shift"]
    assert g["neg"]["inputs"]["text"] == WAN_I2V_RECIPE["negative"]


def test_i2v_overrides_reach_the_graph_under_the_consent_act(monkeypatch):
    # Every value OPPOSES the frozen one, and that is asserted before comparing.
    monkeypatch.setenv(I2V_PREQUAL_ENV, "1")
    monkeypatch.setenv("OTR_WAN_STEPS", "11")
    monkeypatch.setenv("OTR_WAN_CFG", "2.25")
    monkeypatch.setenv("OTR_WAN_SHIFT", "3.75")
    monkeypatch.setenv("OTR_WAN_SAMPLER", "euler")
    monkeypatch.setenv("OTR_WAN_SCHEDULER", "beta")
    monkeypatch.setenv("OTR_WAN_NEGATIVE", "a distinct negative")
    assert WAN_I2V_RECIPE["steps"] != 11
    assert WAN_I2V_RECIPE["cfg"] != 2.25
    assert WAN_I2V_RECIPE["shift"] != 3.75
    assert WAN_I2V_RECIPE["sampler"] != "euler"
    assert WAN_I2V_RECIPE["scheduler"] != "beta"
    assert WAN_I2V_RECIPE["negative"] != "a distinct negative"
    g = _i2v_graph()
    ks = g["ksampler"]["inputs"]
    assert ks["steps"] == 11 and ks["cfg"] == 2.25
    assert ks["sampler_name"] == "euler" and ks["scheduler"] == "beta"
    assert g["modelsampling"]["inputs"]["shift"] == 3.75
    assert g["neg"]["inputs"]["text"] == "a distinct negative"


def test_i2v_environment_can_NOT_author_the_render_on_a_production_leg(
        monkeypatch):
    for env, val in (("OTR_WAN_STEPS", "11"), ("OTR_WAN_CFG", "2.25"),
                     ("OTR_WAN_SAMPLER", "euler"),
                     ("OTR_WAN_NEGATIVE", "a distinct negative")):
        monkeypatch.setenv(env, val)
    g = _i2v_graph()
    assert g["ksampler"]["inputs"]["steps"] == WAN_I2V_RECIPE["steps"]
    assert g["ksampler"]["inputs"]["cfg"] == WAN_I2V_RECIPE["cfg"]
    assert g["ksampler"]["inputs"]["sampler_name"] == WAN_I2V_RECIPE["sampler"]
    assert g["neg"]["inputs"]["text"] == WAN_I2V_RECIPE["negative"]


def test_i2v_a_mistyped_knob_no_longer_dies_as_a_raw_ValueError(monkeypatch):
    """The defect this chunk closes on this adapter.

    ``int(os.environ.get("OTR_WAN_STEPS", "20"))`` inline in ``_build_graph``
    raised a bare ValueError from the middle of a graph build, naming nothing.
    Now it is a NAMED MALFORMED_CONFIG -- and only where the knob can bind."""
    monkeypatch.setenv(I2V_PREQUAL_ENV, "1")
    monkeypatch.setenv("OTR_WAN_STEPS", "twenty")
    with pytest.raises(EngineUnusable) as exc:
        _i2v_graph()
    assert exc.value.reason is EngineUsabilityReason.MALFORMED_CONFIG
    assert "OTR_WAN_STEPS" in str(exc.value)


def test_i2v_a_stale_malformed_knob_can_NOT_kill_a_production_leg(monkeypatch):
    monkeypatch.setenv("OTR_WAN_STEPS", "twenty")
    monkeypatch.setenv("OTR_WAN_CFG", "high")
    g = _i2v_graph()
    assert g["ksampler"]["inputs"]["steps"] == WAN_I2V_RECIPE["steps"]
    assert g["ksampler"]["inputs"]["cfg"] == WAN_I2V_RECIPE["cfg"]


def test_i2v_out_of_range_steps_fails_closed_under_the_consent_act(monkeypatch):
    monkeypatch.setenv(I2V_PREQUAL_ENV, "1")
    monkeypatch.setenv("OTR_WAN_STEPS", "999")
    with pytest.raises(EngineUnusable) as exc:
        WanI2VEngine()._resolve_render_config()
    assert exc.value.reason is EngineUsabilityReason.MALFORMED_CONFIG


def test_i2v_warning_direction_is_pinned(monkeypatch, caplog):
    monkeypatch.setenv("OTR_WAN_SAMPLER", "euler")
    with caplog.at_level(logging.WARNING, logger=_I2V_LOGGER):
        WanI2VEngine()._resolve_render_config()
    text = " ".join(r.getMessage() for r in caplog.records
                    if r.name == _I2V_LOGGER)
    assert "OTR_WAN_SAMPLER" in text and RECIPE_WAN_I2V in text
    assert wr.FROZEN_MARKER in text
    assert wr.MEASUREMENT_MARKER not in text


def test_i2v_receipt_and_the_vram_peak_both_reach_the_clip_dict():
    """``vram_peak_mb`` was MEASURED, logged, and then DISCARDED on this
    adapter -- NEWBUG-1's fix landed for wan_ti2v in 2026-07 and was never
    applied here, so every wan_i2v clip reported None and ``render_shot``
    silently fell back to an instantaneous post-render read that can
    under-report the true peak. Found by the pre-push QA fan-out."""
    clip = WanI2VEngine()._clip_from_raw(
        {"out_path": "/x/y.mp4", "frame_count": 81, "vram_peak_mb": 9123,
         "recipe": RECIPE_WAN_I2V}, {"shot_id": "b007"})
    assert clip["recipe"] == RECIPE_WAN_I2V
    assert clip["vram_peak_mb"] == 9123
    assert clip["engine_id"] == "wan_i2v"


def test_i2v_measurement_clips_are_distinguishable(monkeypatch):
    monkeypatch.setenv(I2V_PREQUAL_ENV, "1")
    stamped = wr.recipe_receipt(RECIPE_WAN_I2V, I2V_PREQUAL_ENV)
    assert stamped == RECIPE_WAN_I2V + wr.PREQUALIFICATION_RECIPE_SUFFIX
    assert stamped != RECIPE_WAN_I2V


# =========================================================================== #
# GAPS THE MUTATION ROUND FOUND (2026-07-27). Four REAL mutants survived the
# first pass; each fix below is the test that now kills one.
# =========================================================================== #
def _mk_node(fn):
    return type("FakeNode", (), {"FUNCTION": "f", "f": fn})


class _FakeModel:
    def detach(self, unpatch_all=False):
        return None


def _wan_fakes(np, n=4):
    """The CORE Wan node classes, faked -- same shape as
    ``tests/test_video_motion_forward.py``'s, kept local so this file's receipt
    tests do not depend on that file's import surface.

    THE DECODER HONOURS THE LENGTH IT WAS ASKED FOR (WIRE-W3b, 2026-07-29).
    It used to emit a fixed ``n`` frames no matter what the latent node was
    given, and ``wan_ti2v``'s ping-pong quietly extended the difference back up
    to the target -- so the fake was exercising the PAD on every render rather
    than the render. ``render_clip`` now refuses a decode that returns a
    different count than the graph asked for, which is a real executor
    invariant, so the fake has to be honest about it. ``n`` remains the
    fallback for a graph that never set a length."""
    asked = {"length": int(n)}

    def _record_length(self, **k):
        asked["length"] = max(1, int(k.get("length") or asked["length"]))
        return (("latent",),)

    def _decode(self, **k):
        return (np.zeros((asked["length"], 8, 8, 3), dtype="float32"),)

    def _wan_latent(self, **k):
        asked["length"] = max(1, int(k.get("length") or asked["length"]))
        return (("p",), ("n",), ("latent",))

    return {
        "unet": _mk_node(lambda self, **k: (_FakeModel(),)),
        "modelsampling": _mk_node(lambda self, **k: (_FakeModel(),)),
        "clip": _mk_node(lambda self, **k: (object(),)),
        "pos": _mk_node(lambda self, **k: (("c",),)),
        "neg": _mk_node(lambda self, **k: (("c",),)),
        "vae": _mk_node(lambda self, **k: (object(),)),
        "loadimage": _mk_node(lambda self, **k: (object(), object())),
        "wan": _mk_node(_wan_latent),
        "latent": _mk_node(_record_length),
        "ksampler": _mk_node(lambda self, **k: (("latent",),)),
        "vaedecode": _mk_node(_decode),
    }


def _render_without_ffmpeg(monkeypatch, engine_module, np, tmp_path):
    """Stub the encode + probe tail so ``render_clip`` runs end to end with no
    ffmpeg. The RECEIPT FIELDS on the returned raw are what these tests read,
    and those are decided before this tail ever runs."""
    from nodes._otr_video_engines import wrapper_bridge as _wb
    out = tmp_path / "clip.mp4"
    out.write_bytes(b"not-a-real-mp4")
    monkeypatch.setattr(_wb, "encode_frames_to_silent_mp4",
                        lambda frames, path, fps: (str(out), len(frames)))
    monkeypatch.setattr(engine_module, "ffprobe_clip_fields", lambda p: {})
    monkeypatch.setattr(engine_module, "validate_silent_clip_contract",
                        lambda fields, fps: None)
    monkeypatch.setattr(engine_module._MC, "VramPeakProbe",
                        lambda **kw: _FakeProbe())


class _FakeProbe:
    """A deterministic stand-in for the NVML peak probe."""
    PEAK_MB = 9123

    def start(self):
        return self

    def stop(self):
        return self.PEAK_MB


def _init_png(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    src = tmp_path / "p.png"
    Image.new("RGB", (832, 480), (10, 20, 30)).save(src)
    return str(src)


@pytest.mark.parametrize("engine_name", ["wan_ti2v", "wan_i2v"])
def test_render_clip_RETURNS_the_receipt_and_the_measured_peak(
        monkeypatch, tmp_path, engine_name):
    """The hop the hand-built ``_clip_from_raw`` tests could not see.

    Those tests construct the raw themselves, so they stay green when
    ``render_clip`` stops putting the fields IN it -- the same shape as the
    chunk-6 defect where a test's own builder agreed with the bug. Two mutants
    (recipe -> None, vram_peak_mb -> None) survived the first mutation round on
    exactly this gap. This drives the real ``render_clip``."""
    np = pytest.importorskip("numpy")
    from nodes._otr_video_engines import wrapper_bridge as _wb
    if engine_name == "wan_ti2v":
        from nodes._otr_video_engines import eng_wan_ti2v as mod
        eng, expected = WanTi2vEngine(), RECIPE_WAN_TI2V
    else:
        from nodes._otr_video_engines import eng_wan_i2v as mod
        eng, expected = WanI2VEngine(), RECIPE_WAN_I2V
    monkeypatch.setattr(_wb, "comfy_input_dir", lambda: str(tmp_path))
    _render_without_ffmpeg(monkeypatch, mod, np, tmp_path)
    eng._classes = _wan_fakes(np, n=4)
    req = {"shot_id": "s1", "asset_refs": {"init_image": _init_png(tmp_path)},
           "text_prompt": "subtle motion", "init_w": 832, "init_h": 480,
           "canvas": {"w": 832, "h": 480, "aspect_policy": "pad"},
           "timing": {"target_frame_count": 33},
           "seed_bundle": {"request_seed": 3}}
    raw = eng.render_clip(req, {"patchers": []})
    assert raw["recipe"] == expected
    assert raw["vram_peak_mb"] == _FakeProbe.PEAK_MB
    # ...and it survives canonicalisation into the dict the manifest reads.
    clip = eng.canonicalize(raw, req, {})
    assert clip["recipe"] == expected
    assert clip["vram_peak_mb"] == _FakeProbe.PEAK_MB


#: The consent env vars as an OPERATOR TYPES THEM. Deliberately literal strings
#: and NOT the imported constants: a test that sets `PREQUALIFICATION_ENV`
#: follows the constant wherever it is renamed, so it can never notice the
#: adapter reading a var no operator will ever set. A mutant that renamed the
#: constant survived the first mutation round on exactly that.
_DOCUMENTED_CONSENT_VARS = (
    ("wan_ti2v", "OTR_WAN_TI2V_PREQUALIFICATION"),
    ("wan_i2v", "OTR_WAN_I2V_PREQUALIFICATION"),
)


@pytest.mark.parametrize("engine_name,documented", _DOCUMENTED_CONSENT_VARS)
def test_the_DOCUMENTED_consent_var_is_the_one_the_adapter_reads(
        monkeypatch, engine_name, documented):
    if engine_name == "wan_ti2v":
        eng, env, opposing = WanTi2vEngine(), "OTR_WAN_TI2V_STEPS", 11
        frozen = WAN_TI2V_RECIPE["steps"]
    else:
        eng, env, opposing = WanI2VEngine(), "OTR_WAN_STEPS", 11
        frozen = WAN_I2V_RECIPE["steps"]
    assert frozen != opposing                        # the override OPPOSES
    monkeypatch.setenv(env, str(opposing))
    assert eng._resolve_render_config()["steps"] == frozen
    # Setting the var the DOCS and the module docstring name must open it.
    monkeypatch.setenv(documented, "1")
    assert eng._resolve_render_config()["steps"] == opposing


def test_i2v_shift_and_scheduler_are_frozen_on_a_production_leg(monkeypatch):
    """`shift` had no production-leg test, so a mutant that put
    `float(os.environ.get("OTR_WAN_SHIFT", "8.0"))` back inline survived: the
    consent-act test agreed with it, and the production test never set it."""
    monkeypatch.setenv("OTR_WAN_SHIFT", "3.75")
    monkeypatch.setenv("OTR_WAN_SCHEDULER", "beta")
    assert WAN_I2V_RECIPE["shift"] != 3.75
    assert WAN_I2V_RECIPE["scheduler"] != "beta"
    g = _i2v_graph()
    assert g["modelsampling"]["inputs"]["shift"] == WAN_I2V_RECIPE["shift"]
    assert g["ksampler"]["inputs"]["scheduler"] == WAN_I2V_RECIPE["scheduler"]


def test_ti2v_shift_and_scheduler_are_frozen_on_a_production_leg(monkeypatch):
    # The twin of the test above -- the same gap existed on both adapters.
    monkeypatch.setenv("OTR_WAN_TI2V_SHIFT", "3.75")
    monkeypatch.setenv("OTR_WAN_TI2V_SCHEDULER", "beta")
    assert WAN_TI2V_RECIPE["shift"] != 3.75
    assert WAN_TI2V_RECIPE["scheduler"] != "beta"
    g = _graph()
    assert g["modelsampling"]["inputs"]["shift"] == WAN_TI2V_RECIPE["shift"]
    assert g["ksampler"]["inputs"]["scheduler"] == WAN_TI2V_RECIPE["scheduler"]


# =========================================================================== #
# LANE 2 -- the WAN receipts name WHICH CELL, not merely that a sweep ran
# =========================================================================== #
from nodes._otr_video_engines import recipe_departures as rdep   # noqa: E402


@pytest.mark.parametrize("engine_name", ["wan_ti2v", "wan_i2v"])
def test_a_production_receipt_is_UNCHANGED_by_lane_2(engine_name):
    # LANE 2 must be invisible on the leg that publishes episodes.
    if engine_name == "wan_ti2v":
        eng, frozen = WanTi2vEngine(), RECIPE_WAN_TI2V
    else:
        eng, frozen = WanI2VEngine(), RECIPE_WAN_I2V
    assert eng._recipe_departures() == {}
    assert eng._recipe_receipt() == frozen


@pytest.mark.parametrize("engine_name,env,opposing", [
    ("wan_ti2v", "OTR_WAN_TI2V_STEPS", "11"),
    ("wan_i2v", "OTR_WAN_STEPS", "11"),
])
def test_the_production_path_does_not_even_REACH_the_resolver(
        monkeypatch, engine_name, env, opposing):
    """The early return is STRUCTURAL. Without it the guarantee would depend on
    every accessor returning the frozen value forever -- which they do today,
    which is exactly why a test that only checks the OUTPUT cannot see the
    guard disappear. Detonating the resolver is what proves it."""
    eng = WanTi2vEngine() if engine_name == "wan_ti2v" else WanI2VEngine()

    def _detonate():
        raise AssertionError(
            "a production leg must not resolve knobs to name departures")
    monkeypatch.setattr(eng, "_resolve_render_config", _detonate)
    monkeypatch.setenv(env, opposing)
    assert eng._recipe_departures() == {}


def test_ti2v_a_sweep_cell_NAMES_the_knobs_it_moved(monkeypatch):
    monkeypatch.setenv(PREQUALIFICATION_ENV, "1")
    monkeypatch.setenv("OTR_WAN_TI2V_TILED_VAE", "0")     # OPPOSES frozen True
    monkeypatch.setenv("OTR_WAN_TI2V_STEPS", "11")
    assert WAN_TI2V_RECIPE["tiled_vae"] is True and WAN_TI2V_RECIPE["steps"] != 11
    eng = WanTi2vEngine()
    assert eng._recipe_departures() == {"steps": 11, "tiled_vae": False}
    assert eng._recipe_receipt() == \
        RECIPE_WAN_TI2V + "+prequalification[steps=11,tiled_vae=off]"


def test_i2v_a_sweep_cell_NAMES_the_knobs_it_moved(monkeypatch):
    monkeypatch.setenv(I2V_PREQUAL_ENV, "1")
    monkeypatch.setenv("OTR_WAN_SAMPLER", "euler")        # OPPOSES frozen uni_pc
    assert WAN_I2V_RECIPE["sampler"] != "euler"
    eng = WanI2VEngine()
    assert eng._recipe_departures() == {"sampler": "euler"}
    assert eng._recipe_receipt() == \
        RECIPE_WAN_I2V + "+prequalification[sampler=euler]"


@pytest.mark.parametrize("engine_name", ["wan_ti2v", "wan_i2v"])
def test_TWO_DIFFERENT_CELLS_STAMP_DIFFERENT_RECEIPTS(monkeypatch, engine_name):
    """The defect stated as a test: before LANE 2 both of these produced the
    same string, so the durable ledger could not tell one cell from another."""
    if engine_name == "wan_ti2v":
        eng, env, consent = WanTi2vEngine(), "OTR_WAN_TI2V_STEPS", PREQUALIFICATION_ENV
    else:
        eng, env, consent = WanI2VEngine(), "OTR_WAN_STEPS", I2V_PREQUAL_ENV
    monkeypatch.setenv(consent, "1")
    monkeypatch.setenv(env, "11")
    cell_a = eng._recipe_receipt()
    monkeypatch.setenv(env, "13")
    cell_b = eng._recipe_receipt()
    assert cell_a != cell_b
    assert "steps=11" in cell_a and "steps=13" in cell_b


@pytest.mark.parametrize("engine_name", ["wan_ti2v", "wan_i2v"])
def test_a_cell_that_moved_NOTHING_still_marks_itself(monkeypatch, engine_name):
    if engine_name == "wan_ti2v":
        eng, frozen, consent = (WanTi2vEngine(), RECIPE_WAN_TI2V,
                                PREQUALIFICATION_ENV)
    else:
        eng, frozen, consent = WanI2VEngine(), RECIPE_WAN_I2V, I2V_PREQUAL_ENV
    monkeypatch.setenv(consent, "1")
    assert eng._recipe_receipt() == frozen + rdep.PREQUALIFICATION_SUFFIX


def test_ti2v_the_tile_geometry_is_reported_only_when_tiled_decode_RAN(
        monkeypatch):
    monkeypatch.setenv(PREQUALIFICATION_ENV, "1")
    monkeypatch.setenv("OTR_WAN_TI2V_VAE_TILE", "1024")
    assert WAN_TI2V_RECIPE["vae_tile"] != 1024
    eng = WanTi2vEngine()
    assert eng._recipe_departures()["vae_tile"] == 1024
    monkeypatch.setenv("OTR_WAN_TI2V_TILED_VAE", "0")
    assert eng._recipe_departures() == {"tiled_vae": False}


@pytest.mark.parametrize("key,env,graph_input,opposing", _TILE_CASES)
def test_ti2v_the_GRAPH_and_the_RECEIPT_read_the_same_tile_value(
        monkeypatch, key, env, graph_input, opposing):
    # All four, because _vaedecode_inputs hand-lists its four calls while
    # _recipe_departures loops -- the graph side is independently regressable.
    monkeypatch.setenv(PREQUALIFICATION_ENV, "1")
    monkeypatch.setenv(env, str(opposing))
    assert WAN_TI2V_RECIPE[key] != opposing
    assert _graph()["vaedecode"]["inputs"][graph_input] == opposing
    assert WanTi2vEngine()._recipe_departures()[key] == opposing


@pytest.mark.parametrize("engine_name,env,consent", [
    ("wan_ti2v", "OTR_WAN_TI2V_NEGATIVE", "OTR_WAN_TI2V_PREQUALIFICATION"),
    ("wan_i2v", "OTR_WAN_NEGATIVE", "OTR_WAN_I2V_PREQUALIFICATION"),
])
def test_a_swept_NEGATIVE_is_named_as_a_digest(monkeypatch, engine_name, env,
                                               consent):
    """BOTH adapters. The mutation round caught the ti2v half missing: dropping
    ``negative`` from that adapter's departure report left the suite green,
    because only the i2v twin of this test existed.

    The shipped default is long and full of commas, so a swept negative always
    takes the digest path -- which is also the only end-to-end exercise the
    digest gets against a real departure rather than a synthetic string."""
    monkeypatch.setenv(consent, "1")
    monkeypatch.setenv(env, "a measured negative, with commas")
    eng = WanTi2vEngine() if engine_name == "wan_ti2v" else WanI2VEngine()
    assert eng._recipe_departures()["negative"] \
        == "a measured negative, with commas"
    assert re.search(r"negative=#[0-9a-f]{8}", eng._recipe_receipt())


@pytest.mark.parametrize("module_name", ["eng_wan_ti2v", "eng_wan_i2v"])
def test_EVERY_stamp_goes_through_the_receipt_helper(module_name):
    """A source-level drift guard, the shape the ltx lane already uses. The
    stamp is what a sweep artifact is identified by; an edit that reached for
    the bare constant would silently un-NAME measurement clips, and no value
    assertion catches it because the constant is still correct in production --
    the only leg most tests run."""
    import importlib
    import inspect
    src = inspect.getsource(
        importlib.import_module("nodes._otr_video_engines." + module_name))
    assert '"recipe": self._recipe_receipt()' in src
    assert '"recipe": _WR.recipe_receipt(' not in src
