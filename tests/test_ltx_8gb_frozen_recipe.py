"""B6 -- the ltx_8gb recipe is FROZEN IN CODE, and the env is a consent act.

THE DEFECT. The 8GB tier's real levers -- T5 device, tiled VAE decode, the
sampling knobs -- have no profile channel: the schema accepts only
``device_policy``, ``dtype_policy`` and ``max_render_frames``. They were read
from ``os.environ``, and per ``PBUG-20260723-02`` a production episode is
submitted to an ALREADY-BOOTED server, so a profile's ``launch.env`` can never
reach it (``otr_8gb_ltx.json``'s is empty anyway). The tier's recipe was
therefore whatever the box happened to boot with -- possibly a sweep from weeks
ago -- and nothing in the published episode said which.

THE FIX, in the shape that Bible rule demands: CODE binds on every leg.
``LTX8_RECIPE_V1`` is the recipe; the env vars bind ONLY under an explicit
``OTR_LTX_8GB_PREQUALIFICATION`` consent act, which is what a measurement sweep
sets and a production box never does.

The inversion these tests exist to hold: outside prequalification a stale knob
is NEVER PARSED, only named. Parsing it would let a value with no effect on the
leg kill the leg -- the precise shape the PBUG forbids.
"""

from __future__ import annotations

import pytest

from nodes._otr_video_engines import eng_ltx_8gb as m
from nodes._otr_video_engines.registry import EngineUnusable

#: Every env var this module's freeze touches, plus the consent act itself. An
#: incomplete list lets the HOST environment decide an outcome (QA finding T-6).
_ENVS = tuple(m._RECIPE_ENV_KEYS.values()) + (
    "OTR_LTX_8GB_MAX_FRAMES", m.PREQUALIFICATION_ENV)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _ENVS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def eng():
    return m.Ltx8gbEngine()


# --- the frozen recipe binds on a production leg --------------------------- #
def test_every_frozen_field_wins_when_no_consent_act_is_present(eng, monkeypatch):
    """The core contract: set EVERY knob to something else, get the recipe."""
    for key, env in m._RECIPE_ENV_KEYS.items():
        monkeypatch.setenv(env, {"sampler": "dpmpp_2m", "t5_device": "default",
                                 "tiled_vae": "1"}.get(key, "7"))

    cfg = eng._resolve_render_config()
    for key in ("steps", "cfg", "max_shift", "base_shift", "terminal",
                "sampler"):
        assert cfg[key] == m.LTX8_RECIPE_V1[key], key
    assert eng._t5_device() == m.LTX8_RECIPE_V1["t5_device"]
    assert eng._tiled_vae() is m.LTX8_RECIPE_V1["tiled_vae"]


def test_the_two_load_bearing_values_are_pinned_BY_VALUE(eng):
    """These two are not taste. ``t5xxl_fp16`` alone is ~9 GB, so the T5 encodes
    on CPU or the 8 GB box does not render at all; tiled decode is OFF because
    core ``VAEDecode`` handles the peak at the smoke canvas. A silent flip of
    either is a tier that stops working, so pin the literals, not just the
    round-trip through the dict."""
    assert m.LTX8_RECIPE_V1["t5_device"] == "cpu"
    assert m.LTX8_RECIPE_V1["tiled_vae"] is False
    assert eng._t5_device() == "cpu"
    assert eng._tiled_vae() is False


def test_the_recipe_string_carries_its_own_version_and_rides_the_receipt(eng):
    """The version lives IN the string, so it reaches the manifest row ->
    ``otr_credits_roll`` -> a published episode. A bare version constant that
    never reached that string would be an unowned ledger field in all but
    name -- and it is ``cfg.recipe``, so a bump moves the session identity."""
    assert m.RECIPE_LTX8_I2V.endswith("_v1")
    assert "ltx098" in m.RECIPE_LTX8_I2V and "distilled" in m.RECIPE_LTX8_I2V


# --- the consent act re-opens them ----------------------------------------- #
def test_the_knobs_bind_again_under_prequalification(eng, monkeypatch):
    monkeypatch.setenv(m.PREQUALIFICATION_ENV, "1")
    monkeypatch.setenv("OTR_LTX_8GB_STEPS", "12")
    monkeypatch.setenv("OTR_LTX_8GB_TERMINAL", "0.25")
    monkeypatch.setenv("OTR_LTX_8GB_T5_DEVICE", "default")
    monkeypatch.setenv("OTR_LTX_8GB_TILED_VAE", "1")

    cfg = eng._resolve_render_config()
    assert cfg["steps"] == 12 and cfg["terminal"] == 0.25
    assert eng._t5_device() == "default"
    assert eng._tiled_vae() is True


@pytest.mark.parametrize("spelling", ("1", "true", "TRUE", "Yes", "on", " 1 "))
def test_the_consent_act_accepts_the_ordinary_truthy_spellings(
        eng, monkeypatch, spelling):
    monkeypatch.setenv(m.PREQUALIFICATION_ENV, spelling)
    monkeypatch.setenv("OTR_LTX_8GB_STEPS", "12")
    assert eng._resolve_render_config()["steps"] == 12


@pytest.mark.parametrize("spelling", ("0", "", "no", "off", "false", "   ",
                                      "2", "yes please"))
def test_a_PRESENT_but_non_consenting_value_still_gets_the_frozen_recipe(
        eng, monkeypatch, spelling):
    """Presence is not consent. A box that exports the var as ``0`` -- or as
    anything that is not an ordinary yes -- is a production box."""
    monkeypatch.setenv(m.PREQUALIFICATION_ENV, spelling)
    monkeypatch.setenv("OTR_LTX_8GB_STEPS", "12")
    assert eng._resolve_render_config()["steps"] == m.LTX8_RECIPE_V1["steps"]


# --- the inversion: outside prequalification, NEVER PARSE ------------------ #
def test_a_malformed_knob_cannot_kill_a_leg_it_cannot_influence(eng, monkeypatch):
    """THE POINT OF THE WHOLE CHUNK. A stale ``STEPS=not-a-number`` in a
    long-booted server's environment has no effect on this leg, so it must not
    be able to fail it. Parsing before ignoring would do exactly that."""
    monkeypatch.setenv("OTR_LTX_8GB_STEPS", "not-a-number")
    monkeypatch.setenv("OTR_LTX_8GB_CFG", "definitely-not-a-float")
    monkeypatch.setenv("OTR_LTX_8GB_SAMPLER", "a_sampler_that_does_not_exist")
    cfg = eng._resolve_render_config()
    assert cfg["steps"] == m.LTX8_RECIPE_V1["steps"]
    assert cfg["sampler"] == m.LTX8_RECIPE_V1["sampler"]


def test_the_same_malformed_knob_IS_terminal_under_prequalification(
        eng, monkeypatch):
    """CONTROL on the test above: the fail-closed parse is scoped, not deleted.
    A sweep that mistypes a value it is actually measuring must still stop."""
    monkeypatch.setenv(m.PREQUALIFICATION_ENV, "1")
    monkeypatch.setenv("OTR_LTX_8GB_STEPS", "not-a-number")
    with pytest.raises(EngineUnusable) as e:
        eng._resolve_render_config()
    assert "OTR_LTX_8GB_STEPS" in str(e.value)


def test_the_ignore_list_names_by_PRESENCE_and_never_parses(monkeypatch):
    monkeypatch.setenv("OTR_LTX_8GB_STEPS", "not-a-number")
    monkeypatch.setenv("OTR_LTX_8GB_TILED_VAE", "1")
    assert m._ignored_override_keys() == ["OTR_LTX_8GB_STEPS",
                                          "OTR_LTX_8GB_TILED_VAE"]


def test_an_EMPTY_env_var_is_not_an_override(monkeypatch):
    """A shell that exports ``OTR_LTX_8GB_STEPS=`` is saying nothing. Counting
    it would warn on every leg of a box that overrides nothing."""
    monkeypatch.setenv("OTR_LTX_8GB_STEPS", "")
    assert m._ignored_override_keys() == []


# --- the ceiling is NOT a recipe knob -------------------------------------- #
def test_max_frames_is_absent_from_the_frozen_recipe(eng):
    """It is a render-length CEILING, not a recipe value: B3 gave it a profile
    -> ledger channel and B4's pre-render refusal reads it to make a
    plan-vs-box disagreement terminal. Folding it in would silence that."""
    assert "max_frames" not in m.LTX8_RECIPE_V1
    assert "OTR_LTX_8GB_MAX_FRAMES" not in m._RECIPE_ENV_KEYS.values()


def test_the_ceiling_still_binds_AND_still_fails_closed_in_production(
        eng, monkeypatch):
    monkeypatch.setenv("OTR_LTX_8GB_MAX_FRAMES", "65")
    assert eng._resolve_render_config()["max_frames"] == 65

    monkeypatch.setenv("OTR_LTX_8GB_MAX_FRAMES", "not-a-number")
    with pytest.raises(EngineUnusable) as e:
        eng._resolve_render_config()
    assert "OTR_LTX_8GB_MAX_FRAMES" in str(e.value)


def test_a_ceiling_below_the_engine_floor_is_still_refused(eng, monkeypatch):
    """The range check is the reason ``_LTX8_MIN_FRAMES`` survived B4."""
    monkeypatch.setenv("OTR_LTX_8GB_MAX_FRAMES", str(m._LTX8_MIN_FRAMES - 1))
    with pytest.raises(EngineUnusable) as e:
        eng._resolve_render_config()
    assert "OTR_LTX_8GB_MAX_FRAMES" in str(e.value)


# --- the two legs must not drift apart ------------------------------------- #
def test_both_legs_return_the_SAME_KEY_SET(eng, monkeypatch):
    """A return shape that varies by mode gives the next reader a KeyError that
    reproduces only under prequalification. ``t5_device``/``tiled_vae`` live in
    the recipe dict but are owned by their own accessors, so a bare
    ``dict(LTX8_RECIPE_V1)`` here would leak them onto one leg only."""
    production = set(eng._resolve_render_config())
    monkeypatch.setenv(m.PREQUALIFICATION_ENV, "1")
    assert set(eng._resolve_render_config()) == production
    # Against a LITERAL too, not just against each other: deleting one branch,
    # or dropping a key from BOTH, would satisfy the equality above alone.
    assert production == {"steps", "cfg", "max_shift", "base_shift",
                          "terminal", "sampler", "max_frames"}


def test_every_frozen_field_has_a_named_env_var_and_vice_versa():
    """Drift guard on the demotion notice. Add a field to the recipe and forget
    its entry here and the operator is never told that knob is being ignored --
    a silent no-op knob is the complaint B3 already collected once."""
    assert set(m._RECIPE_ENV_KEYS) == set(m.LTX8_RECIPE_V1)


# --- the operator is told, exactly once, in the right direction ------------ #
def test_the_production_warning_names_the_knob_the_recipe_and_the_way_out(
        eng, monkeypatch, caplog):
    monkeypatch.setenv("OTR_LTX_8GB_TILED_VAE", "1")
    with caplog.at_level("WARNING"):
        eng._resolve_render_config()
    assert "OTR_LTX_8GB_TILED_VAE" in caplog.text        # what is ignored
    assert m.RECIPE_LTX8_I2V in caplog.text              # what bound instead
    assert m.PREQUALIFICATION_ENV in caplog.text         # how to measure
    # DIRECTION, not just content. The prequalification warning also names the
    # knob, also interpolates the recipe, and also contains the substring
    # "PREQUALIFICATION" (it is inside the env var's own name) -- so without a
    # token unique to this branch, swapping the two bodies stays green.
    assert "FROZEN" in caplog.text
    assert "measurement run" not in caplog.text


def test_the_prequalification_warning_says_MEASUREMENT_not_contract(
        eng, monkeypatch, caplog):
    monkeypatch.setenv(m.PREQUALIFICATION_ENV, "1")
    monkeypatch.setenv("OTR_LTX_8GB_STEPS", "12")
    with caplog.at_level("WARNING"):
        eng._resolve_render_config()
    assert "OTR_LTX_8GB_STEPS" in caplog.text
    # The words this test is NAMED for. "PREQUALIFICATION" alone would not
    # discriminate: it is a substring of the env var name, which the
    # production warning also interpolates.
    assert "measurement run" in caplog.text
    assert "FROZEN" not in caplog.text


def test_a_box_that_overrides_NOTHING_logs_nothing(eng, caplog):
    """CONTROL: the overwhelmingly common configuration is silent. A warning on
    every leg of every episode is a warning the operator stops reading."""
    with caplog.at_level("WARNING"):
        eng._resolve_render_config()
    assert caplog.text == ""


# --- the render INPUTS the first draft of this chunk left behind ----------- #
# All three were found by the pre-push QA panel against the real files: a
# freeze that covers the sampling knobs but leaves other boot-environment
# values deciding what a clip LOOKS like is a half-applied freeze, and the
# receipt is the one thing that outlives the run.
def test_the_negative_prompt_no_longer_comes_from_the_environment(
        eng, monkeypatch):
    """It is a render input: two boxes with different stale values produced
    visibly different clips and both stamped the same recipe receipt."""
    monkeypatch.setenv("OTR_LTX_8GB_NEGATIVE", "a stale sweep's negative")
    assert eng._negative_prompt() == m.LTX8_RECIPE_V1["negative"]

    plan = eng._build_render_request(
        {"asset_refs": {"init_image": "p"},
         "timing": {"target_frame_count": 9}, "seed_bundle": {"request_seed": 1}})
    graph = eng._build_graph({"text_prompt": "x"}, "p.png", plan, 9, 512, 288)
    assert graph["neg"]["inputs"]["text"] == m.LTX8_RECIPE_V1["negative"]


def test_the_SHOT_still_owns_its_own_negative_prompt(eng, monkeypatch):
    """CONTROL, and the reason this is a demotion rather than a removal: the
    director's per-shot channel travels WITH the work, so it still wins. Only
    the server-boot channel is closed."""
    monkeypatch.setenv("OTR_LTX_8GB_NEGATIVE", "a stale sweep's negative")
    plan = eng._build_render_request(
        {"asset_refs": {"init_image": "p"},
         "timing": {"target_frame_count": 9}, "seed_bundle": {"request_seed": 1}})
    graph = eng._build_graph(
        {"text_prompt": "x", "negative_prompt": "the shot's own negative"},
        "p.png", plan, 9, 512, 288)
    assert graph["neg"]["inputs"]["text"] == "the shot's own negative"


def test_the_negative_prompt_reopens_under_prequalification(eng, monkeypatch):
    monkeypatch.setenv(m.PREQUALIFICATION_ENV, "1")
    monkeypatch.setenv("OTR_LTX_8GB_NEGATIVE", "a measured negative")
    assert eng._negative_prompt() == "a measured negative"


def test_the_tiled_decode_GEOMETRY_is_frozen_too(eng, monkeypatch):
    """Latent today (tiled decode is off) and frozen anyway: the day a measured
    v2 turns tiling on is the day these four would otherwise come live from the
    boot environment with nothing naming them."""
    for env in ("OTR_LTX_8GB_VAE_TILE", "OTR_LTX_8GB_VAE_OVERLAP",
                "OTR_LTX_8GB_VAE_TEMPORAL", "OTR_LTX_8GB_VAE_TEMPORAL_OVERLAP"):
        monkeypatch.setenv(env, "99")
    # Force the tiled path WITHOUT the consent act, which is exactly the v2
    # world: the recipe says tile, the environment must not say how.
    monkeypatch.setattr(eng, "_tiled_vae", lambda: True)

    inputs = eng._decode_inputs(lambda node, slot: (node, slot))
    assert inputs["tile_size"] == m.LTX8_RECIPE_V1["vae_tile"]
    assert inputs["overlap"] == m.LTX8_RECIPE_V1["vae_overlap"]
    assert inputs["temporal_size"] == m.LTX8_RECIPE_V1["vae_temporal"]
    assert inputs["temporal_overlap"] == \
        m.LTX8_RECIPE_V1["vae_temporal_overlap"]


def test_the_tiled_decode_geometry_reopens_under_prequalification(
        eng, monkeypatch):
    monkeypatch.setenv(m.PREQUALIFICATION_ENV, "1")
    monkeypatch.setenv("OTR_LTX_8GB_VAE_TILE", "256")
    monkeypatch.setattr(eng, "_tiled_vae", lambda: True)
    assert eng._decode_inputs(lambda node, slot: (node, slot))["tile_size"] == 256


# --- the receipt tells the truth about which recipe rendered --------------- #
def test_a_production_leg_stamps_the_frozen_recipe_name(eng):
    assert m.recipe_receipt() == m.RECIPE_LTX8_I2V


def test_a_MEASUREMENT_leg_stamps_a_receipt_of_its_own(eng, monkeypatch):
    """The clip a sweep writes may share NONE of v1's values, and the receipt
    rides the manifest into `stamp_durable(meta.render_engines)` -- a durable
    ledger a published episode carries. Stamping the frozen name on a sweep
    artifact would make a measurement indistinguishable from production in the
    one record that outlives the run."""
    monkeypatch.setenv(m.PREQUALIFICATION_ENV, "1")
    receipt = m.recipe_receipt()
    assert receipt != m.RECIPE_LTX8_I2V
    assert receipt.startswith(m.RECIPE_LTX8_I2V)     # still greppable as v1
    assert m.PREQUALIFICATION_RECIPE_SUFFIX in receipt


def test_the_marked_receipt_reaches_the_session_config_and_the_identity(
        tmp_path, monkeypatch):
    """Both consumers, not just the constant: `resolve_session_config` stamps
    `cfg.recipe` and `session_identity` carries it as element [1], so a sweep
    segment and a production segment can never be read as the same session."""
    from nodes._otr_video_engines import beat_session as bs

    (tmp_path / m._LTX8_DEFAULT_CKPT).write_bytes(b"c" * 2048)
    (tmp_path / m._LTX8_DEFAULT_T5).write_bytes(b"t" * 1024)
    engine = m.Ltx8gbEngine()
    monkeypatch.setattr(
        engine, "_resolve_model_file",
        lambda categories, name, env_dir: (
            str(tmp_path / name) if (tmp_path / name).exists() else None))

    production = bs.session_identity(engine)
    assert engine.resolve_session_config().recipe == m.RECIPE_LTX8_I2V

    monkeypatch.setenv(m.PREQUALIFICATION_ENV, "1")
    assert engine.resolve_session_config().recipe != m.RECIPE_LTX8_I2V
    assert bs.session_identity(engine) != production


def test_EVERY_stamp_goes_through_the_receipt_helper(eng):
    """A source-level drift guard, in the shape B4 used for the deleted frame
    ladder. The two stamp sites (`resolve_session_config`, `render_clip`) are
    what a sweep artifact is identified by; a future edit that reaches for the
    bare constant at either would silently un-mark measurement clips, and no
    value assertion catches that because the constant is still correct in
    production -- the only leg most tests run."""
    import inspect
    src = inspect.getsource(m)
    assert 'recipe=RECIPE_LTX8_I2V' not in src
    assert '"recipe": RECIPE_LTX8_I2V' not in src
    assert 'recipe=recipe_receipt()' in src
    assert '"recipe": recipe_receipt()' in src


# --- the prequalification DEFAULTS follow the recipe, not a literal -------- #
# Dormant while v1's literals happen to coincide with the frozen values, which
# is exactly why these are pinned: a measured v2 is when they diverge, and a
# sweep that re-exports only SOME knobs must still measure the recipe it is
# validating rather than a third configuration.
def test_the_t5_device_prequalification_default_tracks_the_recipe(
        eng, monkeypatch):
    monkeypatch.setitem(m.LTX8_RECIPE_V1, "t5_device", "default")
    monkeypatch.setenv(m.PREQUALIFICATION_ENV, "1")     # knob open, not set
    assert eng._t5_device() == "default"


def test_the_tiled_vae_prequalification_default_tracks_the_recipe(
        eng, monkeypatch):
    monkeypatch.setitem(m.LTX8_RECIPE_V1, "tiled_vae", True)
    monkeypatch.setenv(m.PREQUALIFICATION_ENV, "1")     # knob open, not set
    assert eng._tiled_vae() is True


def test_an_out_of_whitelist_t5_device_falls_back_to_the_RECIPE(
        eng, monkeypatch):
    """The clamp exists so a typo cannot hand ComfyUI a device string it will
    fail on mid-load; it must clamp to the recipe, not to a literal."""
    monkeypatch.setitem(m.LTX8_RECIPE_V1, "t5_device", "default")
    monkeypatch.setenv(m.PREQUALIFICATION_ENV, "1")
    monkeypatch.setenv("OTR_LTX_8GB_T5_DEVICE", "cuda:7")
    assert eng._t5_device() == "default"


# --- one range-check implementation, not two ------------------------------- #
def test_a_malformed_tile_geometry_STOPS_a_sweep_like_every_other_knob(
        eng, monkeypatch):
    """The tile knobs used to fail OPEN -- a bad value was swallowed and the
    default silently substituted, so a sweep would render at 512 and report it
    had measured whatever it typed. They now share `_config_number` with the
    sampling knobs, which is the point: one rule, one implementation."""
    monkeypatch.setenv(m.PREQUALIFICATION_ENV, "1")
    monkeypatch.setenv("OTR_LTX_8GB_VAE_TILE", "not-a-number")
    monkeypatch.setattr(eng, "_tiled_vae", lambda: True)
    with pytest.raises(EngineUnusable) as e:
        eng._decode_inputs(lambda node, slot: (node, slot))
    assert "OTR_LTX_8GB_VAE_TILE" in str(e.value)


def test_a_tile_geometry_below_the_NODE_S_OWN_floor_is_refused(
        eng, monkeypatch):
    """`VAEDecodeTiled` declares temporal_size min 8 in the live /object_info
    capture. Handing it 3 is a render that dies inside ComfyUI; refuse it at
    the boundary that can still name the knob."""
    monkeypatch.setenv(m.PREQUALIFICATION_ENV, "1")
    monkeypatch.setenv("OTR_LTX_8GB_VAE_TEMPORAL", "3")
    monkeypatch.setattr(eng, "_tiled_vae", lambda: True)
    with pytest.raises(EngineUnusable) as e:
        eng._decode_inputs(lambda node, slot: (node, slot))
    assert "OTR_LTX_8GB_VAE_TEMPORAL" in str(e.value)
    assert "out of range" in str(e.value)


def test_every_geometry_knob_has_bounds_and_every_bound_has_a_knob():
    """Drift guard: add a geometry key and forget its bounds and `_i` raises
    KeyError mid-render instead of refusing by name."""
    assert set(m._VAE_TILE_BOUNDS) <= set(m.LTX8_RECIPE_V1)
    for key, (lo, hi) in m._VAE_TILE_BOUNDS.items():
        assert lo <= int(m.LTX8_RECIPE_V1[key]) <= hi, key


def test_an_exported_but_EMPTY_tiled_vae_still_gets_the_frozen_default(
        eng, monkeypatch):
    """`get(name, dflt)` returns "" for an exported-empty var, which is not
    truthy -- so the knob would read OFF against a frozen default of ON. Every
    other accessor treats empty as unset; this one must too."""
    monkeypatch.setitem(m.LTX8_RECIPE_V1, "tiled_vae", True)
    monkeypatch.setenv(m.PREQUALIFICATION_ENV, "1")
    monkeypatch.setenv("OTR_LTX_8GB_TILED_VAE", "")
    assert eng._tiled_vae() is True
