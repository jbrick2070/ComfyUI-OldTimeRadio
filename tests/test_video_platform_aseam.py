"""A-Seam / A-S1 CPU test suite for the OTR Open Video Model Platform (CW-1).

Covers the foundation deltas built in CW-1:
  AS-4 protocol/registry parity + dep-free base, AS-1 role_compat filter,
  AS-2 resolver-prune, the Pydantic schemas (extra=forbid), the shell nodes
  (VideoProbe / VideoDirector) + the three STRING forceInput gates, and the
  ShotLock audio-derived clip budget + M4 per-beat creative derivation.

All CPU; no GPU, no model load (the M4 LLM is injected as a fake callable, and
the production path is headless-safe via OTR_TEST_MODE). These are the CW-1
EXIT-checkpoint tests.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

from nodes._otr_shared import engine_registry_base as base  # noqa: F401
from nodes._otr_shared import role_compat as rc
from nodes._otr_shared import resolver as rs
from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import schemas as sc
from nodes.otr_video_probe import OTRVideoProbe
from nodes.otr_video_director import OTRVideoDirector, ADD_CUSTOM
from nodes import otr_shot_lock as sl
from nodes.otr_shot_lock import OTRShotLock

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def clean_video_registry():
    """Snapshot + restore the global video registry so a test can register
    stub engines without leaking into other tests."""
    saved = dict(vreg._VIDEO_REGISTRY._registry)
    vreg._VIDEO_REGISTRY._registry.clear()
    try:
        yield vreg._VIDEO_REGISTRY
    finally:
        vreg._VIDEO_REGISTRY._registry.clear()
        vreg._VIDEO_REGISTRY._registry.update(saved)


def _stub_engine(**kw):
    import types
    base_attrs = dict(
        name="stub", roles=("text_to_video",), default_roles=("text_to_video",),
        commercial_clean=True, requires_flag=None, family="text_to_video",
        required_inputs=("text_prompt",), canonicalize=None,
    )
    base_attrs.update(kw)
    return types.SimpleNamespace(**base_attrs)


# ---------------------------------------------------------------------------
# AS-4 -- cold import + protocol parity + registry
# ---------------------------------------------------------------------------


def test_cold_import_no_heavy_libs():
    """Importing the video registry + shared modules pulls NO heavy lib (V-12)."""
    code = (
        "import sys;"
        "import nodes._otr_video_engines.registry;"
        "import nodes._otr_video_engines.schemas;"
        "import nodes._otr_shared.role_compat;"
        "import nodes._otr_shared.resolver;"
        "import nodes._otr_shared.engine_registry_base;"
        "import nodes._otr_shared.gpu_residency;"
        "import nodes._otr_shared.portrait_ledger;"
        "import nodes.otr_video_probe; import nodes.otr_video_director;"
        "import nodes.otr_shot_lock;"
        "heavy=[m for m in ('torch','transformers','diffusers') if m in sys.modules];"
        "print('HEAVY', heavy);"
        "sys.exit(1 if heavy else 0)"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], cwd=str(REPO_ROOT),
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"heavy libs pulled at import:\n{r.stdout}\n{r.stderr}"


def test_protocol_parity():
    """VideoEngine is a structural superset of the SHIPPED AudioEngine core,
    and shares the usability taxonomy (AS-4)."""
    from nodes._otr_audio_engines.registry import (
        AudioEngine, EngineUsabilityReason as AReason,
    )
    a_ann = AudioEngine.__annotations__
    v_ann = vreg.VideoEngine.__annotations__
    for name, typ in a_ann.items():
        assert name in v_ann, f"VideoEngine missing audio core attr {name!r}"
        assert v_ann[name] == typ, f"annotation mismatch on {name!r}"
    for meth in ("load", "unload"):
        assert hasattr(vreg.VideoEngine, meth)
    for meth in ("assert_usable", "prepare", "render_clip", "canonicalize", "teardown"):
        assert hasattr(vreg.VideoEngine, meth), f"VideoEngine missing {meth}"
    assert {r.value for r in vreg.EngineUsabilityReason} == {r.value for r in AReason}


def test_video_registry_register_and_assert_usable(clean_video_registry):
    eng = _stub_engine(name="ltx")
    vreg.register(eng)
    assert vreg.all_engine_names() == ["ltx"]
    assert vreg.engines_for_role("text_to_video") == ["ltx"]
    assert vreg.assert_usable("ltx", "text_to_video") == "ltx"
    # C2 (2026-06-30): eligibility is CAPABILITY. A text-only engine fits EVERY real
    # role (every role supplies text_prompt), so the rejection case must be an engine
    # whose required input the role cannot supply: an audio-driven engine is refused
    # by retired_role_b (which supplies only text_prompt, no audio_ref).
    vreg.register(_stub_engine(name="talker", roles=("character_video",),
                               required_inputs=("audio_ref", "init_image")))
    with pytest.raises(vreg.EngineUnusable):
        vreg.assert_usable("talker", "retired_role_b")  # role can't supply audio_ref
    with pytest.raises(vreg.EngineUnusable):
        vreg.assert_usable("missing", "text_to_video")  # not registered


def test_registry_allows_canonicalize_none(clean_video_registry):
    """A reduced-capability engine (canonicalize=None) registers fine (C's
    reduced set; the registry never calls render-lifecycle methods)."""
    vreg.register(_stub_engine(name="img", canonicalize=None))
    assert "img" in vreg.all_engine_names()


# ---------------------------------------------------------------------------
# AS-1 -- shared role_compat filter
# ---------------------------------------------------------------------------


def test_role_compat_filter_shared():
    descs = [
        {"engine_id": "abstract", "roles": ("music_visual",),
         "required_inputs": ()},
        {"engine_id": "humo", "roles": ("character_video", "announcer_visual"),
         "required_inputs": ("audio_ref", "init_image")},
        {"engine_id": "ltx", "roles": ("music_visual",),
         "required_inputs": ("text_prompt",)},
    ]
    # Capability-only (operator 2026-06-22): the per-engine `roles` list is NOT a
    # gate -- an engine fits every role whose inputs satisfy its required_inputs.
    assert rc.filter_engines_for_role("announcer_visual", descs) == ["abstract", "humo", "ltx"]
    assert rc.filter_engines_for_role("character_video", descs) == ["abstract", "humo", "ltx"]
    assert rc.filter_engines_for_role("music_visual", descs) == ["abstract", "humo", "ltx"]
    with pytest.raises(rc.RoleCompatError):
        rc.filter_engines_for_role("not_a_role", descs)
    # rip-sfx-broll (2026-07-01): the dead roles raise like any unknown token.
    with pytest.raises(rc.RoleCompatError):
        rc.filter_engines_for_role("retired_role_b", descs)


def test_role_compat_fail_closed_on_malformed():
    """A descriptor declaring an unknown token or missing its engine_id is
    EXCLUDED, not raised (fail-closed). A missing `roles` key is NO LONGER a
    failure -- eligibility is capability, not the whitelist (2026-06-22)."""
    descs = [
        {"engine_id": "ok", "roles": ("music_visual",), "required_inputs": ()},
        {"engine_id": "bad_token", "roles": ("music_visual",),
         "required_inputs": ("nonsense",)},  # unknown token -> excluded
        {"engine_id": "no_roles", "required_inputs": ()},  # no `roles` key is fine now
        {"roles": ("music_visual",), "required_inputs": ()},  # no engine_id -> skipped
    ]
    assert rc.filter_engines_for_role("music_visual", descs) == ["ok", "no_roles"]


# ---------------------------------------------------------------------------
# AS-2 -- resolver prune + DAG validation
# ---------------------------------------------------------------------------


def test_conditional_skip_prunes_orphan():
    groups = [
        {"group_id": "bg_ltx", "kind": "provider", "depends_on": [],
         "produces_base_for": ["char3d"]},
        {"group_id": "char3d", "kind": "consumer", "depends_on": ["bg_ltx"]},
        {"group_id": "ann", "kind": "consumer", "depends_on": []},
    ]
    pruned = rs.prune_orphaned_groups(groups, ["char3d"])
    assert [g["group_id"] for g in pruned] == ["ann"]
    validated = rs.validate_execution_groups(
        groups, prune_orphans=True, pruned_group_ids=["char3d"],
    )
    assert [g["group_id"] for g in validated] == ["ann"]


def test_resolver_rejects_cycle_and_dangling():
    with pytest.raises(rs.ResolverError):
        rs.validate_execution_groups(
            [{"group_id": "a", "depends_on": ["b"]},
             {"group_id": "b", "depends_on": ["a"]}]
        )
    with pytest.raises(rs.ResolverError):
        rs.validate_execution_groups([{"group_id": "a", "depends_on": ["ghost"]}])


def test_resolver_provider_before_consumer():
    groups = [
        {"group_id": "consumer", "kind": "consumer", "depends_on": ["provider"]},
        {"group_id": "provider", "kind": "provider", "depends_on": []},
    ]
    out = rs.validate_execution_groups(groups)  # must not raise; topo-orderable
    assert {g["group_id"] for g in out} == {"consumer", "provider"}


# ---------------------------------------------------------------------------
# Schemas -- extra=forbid + family required inputs
# ---------------------------------------------------------------------------


def test_schema_extra_forbid_and_family_rules():
    from pydantic import ValidationError

    ok = sc.VideoRequest(
        request_id="r", shot_id="s", role="retired_role_b",
        family_hint="text_to_video", profile_id="p", text_prompt="hello",
        timing=sc.Timing(), canvas=sc.Canvas(w=832, h=480, fps=25),
    )
    assert ok.family_hint == "text_to_video"

    with pytest.raises(ValidationError):  # unknown key
        sc.VideoRequest(
            request_id="r", shot_id="s", role="x", family_hint="text_to_video",
            profile_id="p", text_prompt="hi", timing=sc.Timing(),
            canvas=sc.Canvas(w=8, h=8, fps=25), bogus_key=1,
        )
    with pytest.raises(ValidationError):  # text_to_video needs text_prompt
        sc.VideoRequest(
            request_id="r", shot_id="s", role="x", family_hint="text_to_video",
            profile_id="p", timing=sc.Timing(), canvas=sc.Canvas(w=8, h=8, fps=25),
        )
    with pytest.raises(ValidationError):  # unknown family
        sc.VideoRequest(
            request_id="r", shot_id="s", role="x", family_hint="not_a_family",
            profile_id="p", text_prompt="hi", timing=sc.Timing(),
            canvas=sc.Canvas(w=8, h=8, fps=25),
        )


# ---------------------------------------------------------------------------
# Shell nodes -- widget golden + forceInput-no-widget + string semantic guard
# ---------------------------------------------------------------------------


def test_widget_vector_exact():
    """Golden pin on the director's required widget order/names (drift guard).
    2026-07-03: three first-class video slots (announcer/music/CHARACTER) --
    the legacy catch-all video slot retired, character promoted to slot 3. The
    11-widget vector below matches node 87's widgets_values[0:11] in the JSON."""
    req = list(OTRVideoDirector.INPUT_TYPES()["required"].keys())
    assert req == [
        "announcer_video_model", "music_video_model", "character_video_model",
        "announcer_image_model", "music_image_model", "character_image_model",
        "fps", "canvas_w", "canvas_h",
        "seed_mode", "request_seed",
    ]
    assert "seed" not in req  # V-7: no widget literally named 'seed'


@pytest.mark.parametrize("cls", [OTRVideoProbe, OTRVideoDirector, OTRShotLock])
def test_forceinput_no_widget(cls):
    """No forceInput input may carry a 'widget' sub-key (BUG-LOCAL-258)."""
    it = cls.INPUT_TYPES()
    for section in ("required", "optional"):
        for name, spec in (it.get(section) or {}).items():
            cfg = spec[1] if len(spec) > 1 else {}
            if isinstance(cfg, dict) and cfg.get("forceInput"):
                assert "widget" not in cfg, f"{cls.__name__}.{name} forceInput has widget"


def test_string_socket_semantic_guard():
    """The three STRING gates (script_json / audio_done / image_done) are typed
    STRING + forceInput, no widget sub-key."""
    it = OTRShotLock.INPUT_TYPES()
    gates = {
        "script_json": it["required"]["script_json"],
        "audio_done": it["required"]["audio_done"],
        "image_done": it["optional"]["image_done"],
    }
    for name, spec in gates.items():
        assert spec[0] == "STRING", name
        assert spec[1].get("forceInput") is True, name
        assert "widget" not in spec[1], name


def test_no_new_model_id_widget():
    """V-11: none of the new nodes introduce a model_id widget."""
    for cls in (OTRVideoProbe, OTRVideoDirector, OTRShotLock):
        it = cls.INPUT_TYPES()
        names = list(it.get("required", {})) + list(it.get("optional", {}))
        assert "model_id" not in names


def test_probe_emits_usable_list(clean_video_registry):
    host, usable, report = OTRVideoProbe().probe()
    hc, us = json.loads(host), json.loads(usable)
    assert "ffmpeg" in hc and "cuda_available" in hc
    assert us["_all_registered"] == []  # empty registry in CW-1
    assert "character_video" in us
    assert "retired_role_b" not in us  # dead role (rip-sfx-broll)
    assert isinstance(report, str) and "OTR_VideoProbe" in report


def test_director_policy_json_and_clamp(clean_video_registry):
    out = OTRVideoDirector().direct(
        announcer_video_model=ADD_CUSTOM, music_video_model=ADD_CUSTOM,
        character_video_model=ADD_CUSTOM, announcer_image_model="Flux (gen 1)",
        music_image_model="Flux (gen 1)", character_image_model="Flux (gen 1)",
        fps=25,
        canvas_w=832, canvas_h=480, seed_mode="request_hash", request_seed=0,
    )
    policy = json.loads(out[0])
    assert policy["canvas"] == {"w": 832, "h": 480, "fps": 25}
    assert policy["video_models"]["announcer_video_model"]["custom"] is True


def test_director_fail_closed_incompatible_pick(clean_video_registry):
    """A registered engine that fits NONE of a slot's roles BY CAPABILITY fails
    closed (no silent swap). Under capability-only routing the genuine
    incompatibility is a required input NO role can supply (an unknown token)."""
    vreg.register(_stub_engine(
        name="needs_unknown", roles=("character_video",), default_roles=(),
        required_inputs=("depth_map",),  # unknown token -> fits NO role
    ))
    with pytest.raises(ValueError):
        OTRVideoDirector().direct(
            announcer_video_model="needs_unknown",  # requires an input no role supplies
            music_video_model="needs_unknown",
            character_video_model=ADD_CUSTOM, announcer_image_model="Flux (gen 1)",
            music_image_model="Flux (gen 1)", character_image_model="Flux (gen 1)",
            fps=25,
            canvas_w=832, canvas_h=480, seed_mode="request_hash", request_seed=0,
        )


# ---------------------------------------------------------------------------
# ShotLock -- clip budget + image_done + M4 derivation
# ---------------------------------------------------------------------------


def test_clip_budget_no_double_count():
    beats = [
        {"beat_id": "b1", "role": "character_video", "char_id": "c1", "text": "hi",
         "samples": 24000, "sample_rate": 24000, "dur_s": None},
        {"beat_id": "b2", "role": "retired_role_b", "char_id": "", "text": "",
         "samples": 12000, "sample_rate": 24000, "dur_s": None},
    ]
    budget = sl.compute_clip_budget(beats, {}, 25)
    assert budget["per_beat"]["b1"] == 25
    assert budget["per_beat"]["b2"] == 12
    assert budget["total_frames"] == 37 == sum(budget["per_beat"].values())


def test_clip_budget_is_pure_per_beat():
    # rip-sfx-broll (2026-07-01): the pooling budget (clip_mode / pool_n /
    # render_count) is GONE -- the budget is pure per-beat frames.
    beats = [
        {"beat_id": f"o{i}", "role": "character_video", "char_id": "c1",
         "text": "", "samples": 1000, "sample_rate": 24000, "dur_s": None}
        for i in range(3)
    ]
    budget = sl.compute_clip_budget(beats, {}, 25)
    assert set(budget) == {"per_beat", "total_frames", "warnings"}
    assert len(budget["per_beat"]) == 3
    assert budget["warnings"] == []


def test_image_done_gate_serializes():
    it = OTRShotLock.INPUT_TYPES()
    assert "image_done" in it["optional"]
    led = json.dumps({"cast": [], "lines": [], "meta": {}})
    a = OTRShotLock().lock(led, audio_done="x", video_policy_json="{}", image_done="")
    b = OTRShotLock().lock(led, audio_done="x", video_policy_json="{}", image_done="img:done")
    # NON-BLOCKING in v1: the image_done value does not change the plan.
    assert json.loads(a[0])["video"] == json.loads(b[0])["video"]
    assert a[3].startswith("shot_lock:done:")


def _char_ledger():
    return {
        "cast": [{"char_id": "c1", "name": "BABA",
                  "portrait_prompt": "a tall weathered spacer with a scar"}],
        "lines": [],
        "meta": {"story_brief_terms": {"setting": ["a derelict orbital station"]}},
    }


def _char_beat(text="we have to leave now"):
    return [{"beat_id": "b1", "role": "character_video", "char_id": "c1",
             "text": text, "samples": None, "sample_rate": None, "dur_s": 1.0}]


def test_shotlock_derives_orphans_from_beat():
    led = _char_ledger()

    def fake_llm(_prompt):
        return json.dumps([{"beat_id": "b1", "expression": "grim resolve",
                            "motion": "steps forward", "camera": "slow push-in"}])

    creative, _ = sl.derive_creative_directives(
        _char_beat(), led["meta"], led, llm_fn=fake_llm,
    )
    c = creative["b1"]
    assert c["expression"] == "grim resolve"
    assert c["motion"] == "steps forward"
    assert c["camera"] == "slow push-in"
    assert c["source"] == "llm"
    assert "grim resolve" in c["text_prompt"] and "spacer" in c["text_prompt"]


def test_passpm_empty_json_reseed_then_templates():
    """An ATTEMPTED derivation LLM that collapses to empty after every reseed
    still spends the reseed budget, then the deterministic collapse guard
    carries the beat instead of aborting the episode."""
    led = _char_ledger()
    calls = {"n": 0}

    def empty_llm(_prompt):
        calls["n"] += 1
        return ""  # always collapses

    creative, warns = sl.derive_creative_directives(
        _char_beat("hello there"), led["meta"], led, llm_fn=empty_llm,
        max_reseed=2,
    )
    assert calls["n"] == 3  # initial + 2 reseeds ran before templating
    c = creative["b1"]
    assert c["source"] == "template_after_llm_miss"
    assert "spacer" in c["text_prompt"]
    assert "station" in c["text_prompt"]
    assert "hello there" in c["text_prompt"]
    assert any("no usable directive for beat b1" in w for w in warns)


def test_passpm_partial_batch_spends_reseeds_then_templates_missing_beat():
    """A PARTIAL batch reply (one beat missing) must spend its FULL reseed
    budget, then template only the missing beat while keeping the valid LLM
    directive."""
    led = _char_ledger()
    beats = [
        {"beat_id": "b1", "role": "character_video", "char_id": "c1",
         "text": "first", "samples": None, "sample_rate": None, "dur_s": 1.0},
        {"beat_id": "b2", "role": "character_video", "char_id": "c1",
         "text": "second", "samples": None, "sample_rate": None, "dur_s": 1.0},
    ]
    calls = {"n": 0}

    def partial_llm(_prompt):
        calls["n"] += 1
        # only ever answers b1 -- b2 is always missing from the batch reply
        return json.dumps([{"beat_id": "b1", "expression": "", "motion": "",
                            "camera": "",
                            "text_prompt": "a tall weathered spacer with a scar, "
                                           "facing the camera, on the station"}])

    creative, warns = sl.derive_creative_directives(
        beats, led["meta"], led, llm_fn=partial_llm, max_reseed=2)
    assert calls["n"] == 3  # the partial reply spent the full reseed budget
    assert creative["b1"]["source"] == "llm"
    assert creative["b2"]["source"] == "template_after_llm_miss"
    assert "second" in creative["b2"]["text_prompt"]
    assert any("no usable directive for beat b2" in w for w in warns)


def test_passpm_no_llm_is_template_lane():
    """llm_fn=None is the legit local template lane (NOT a failure): deterministic
    prompt, anchor-led, finished, never raises."""
    led = _char_ledger()
    creative, _warns = sl.derive_creative_directives(
        _char_beat("hello there"), led["meta"], led, llm_fn=None,
    )
    c = creative["b1"]
    assert c["source"] == "template"
    _template = sl._deterministic_template(
        "a tall weathered spacer with a scar", "a derelict orbital station",
        "hello there",
    )
    assert c["text_prompt"].startswith(
        sl._subject_anchor("a tall weathered spacer with a scar"))
    assert _template in c["text_prompt"]
    from nodes._otr_story_brief_helpers import STYLE_TAIL_DEFAULT
    assert c["text_prompt"].endswith(STYLE_TAIL_DEFAULT)
    assert c["prompt_hash"] == sl._content_hash(c["text_prompt"])


def test_story_consistency_gate_templates_or_warn_only_keeps():
    """A hallucinated (inconsistent/faceless) LLM prompt templates by default;
    consistency_gate_warn_only=True KEEPS the AI prompt (operator toggle)."""
    led = _char_ledger()

    def halluc_llm(_prompt):
        # authors a full prompt that drops BOTH the cast trait and the setting
        return json.dumps([{"beat_id": "b1", "expression": "x",
                            "text_prompt": "bright sunny flower meadow"}])

    # default (warn_only=False): the LLM prompt fails the gate -> template guard
    creative, warns = sl.derive_creative_directives(
        _char_beat(), led["meta"], led, llm_fn=halluc_llm)
    c = creative["b1"]
    assert c["source"] == "template_after_consistency_miss"
    assert "bright sunny flower meadow" not in c["text_prompt"]
    assert "spacer" in c["text_prompt"]
    assert "station" in c["text_prompt"]
    assert any("deterministic template collapse guard" in w for w in warns)

    # warn_only=True keeps the AI prompt (source stays "llm"), logs the miss
    creative, warns = sl.derive_creative_directives(
        _char_beat(), led["meta"], led, llm_fn=halluc_llm,
        consistency_gate_warn_only=True,
    )
    c = creative["b1"]
    assert c["source"] == "llm"
    assert "spacer" in c["text_prompt"]   # via the subject anchor
    assert any("consistency gate" in w for w in warns)


def test_appearance_lookup_by_char_id():
    led = {"cast": [{"char_id": "c1", "name": "BABA",
                     "portrait_prompt": "unique-portrait-token"}],
           "lines": [], "meta": {}}
    assert sl._appearance_for_char(led, "c1") == "unique-portrait-token"
    assert sl._appearance_for_char(led, "BABA") == ""   # NOT by display name
    assert sl._appearance_for_char(led, "missing") == ""


def test_3d_expression_not_in_mesh_key():
    led = _char_ledger()
    beats = _char_beat()
    creative = {"b1": {"expression": "angry-scowl", "motion": "m", "camera": "c",
                       "text_prompt": "p", "source": "llm",
                       "request_hash": "rh", "prompt_hash": "ph"}}
    budget = sl.compute_clip_budget(beats, {}, 25)
    _groups, shots = sl.build_execution_plan(beats, budget, creative, {})
    sh = shots[0]
    cache_blob = json.dumps(sh["cache_keys"])
    assert "angry-scowl" not in cache_blob and "expression" not in cache_blob
    assert sh["creative"]["expression"] == "angry-scowl"  # driver channel only


def test_cheap_family_no_creative_llm_call():
    led = {"cast": [], "lines": [], "meta": {}}
    beats = [{"beat_id": "b1", "role": "retired_role_b", "char_id": "",
              "text": "", "samples": None, "sample_rate": None, "dur_s": 1.0}]
    calls = {"n": 0}

    def llm(_p):
        calls["n"] += 1
        return "[]"

    creative, _ = sl.derive_creative_directives(beats, led["meta"], led, llm_fn=llm)
    assert calls["n"] == 0
    assert creative == {}


def test_batch_beats_no_truncation():
    import re

    cast = [{"char_id": f"c{i}", "name": f"N{i}", "portrait_prompt": f"portrait{i}"}
            for i in range(20)]
    led = {"cast": cast, "lines": [],
           "meta": {"story_brief_terms": {"setting": ["a station"]}}}
    beats = [{"beat_id": f"b{i}", "role": "character_video", "char_id": f"c{i}",
              "text": f"line {i}", "samples": None, "sample_rate": None, "dur_s": 1.0}
             for i in range(20)]
    seen = []

    def llm(prompt):
        ids = re.findall(r'"beat_id": "(b\d+)"', prompt)
        seen.append(len(ids))
        return json.dumps([{"beat_id": i, "expression": "e", "motion": "m",
                            "camera": "cam"} for i in ids])

    creative, _ = sl.derive_creative_directives(
        beats, led["meta"], led, llm_fn=llm, batch_size=15,
    )
    assert len(creative) == 20            # every beat covered (no truncation)
    assert len(seen) == 2                 # 20 beats / 15 -> 2 batches
    assert max(seen) <= 15


def test_creative_derivation_render_twice():
    led = _char_ledger()

    def llm(_p):
        return json.dumps([{"beat_id": "b1", "expression": "e", "motion": "m",
                            "camera": "cam"}])

    c1, _ = sl.derive_creative_directives(_char_beat(), led["meta"], led, llm_fn=llm)
    c2, _ = sl.derive_creative_directives(_char_beat(), led["meta"], led, llm_fn=llm)
    assert c1 == c2  # deterministic in its inputs


def test_shotlock_end_to_end_stamps_video_section():
    led = {
        "cast": [{"char_id": "c1", "name": "BABA", "portrait_prompt": "a spacer"}],
        "lines": [
            {"line_id": "l1", "char_id": "c1", "speaker_role": "char_voice",
             "text": "hello", "samples": 24000, "sample_rate": 24000},
            {"line_id": "l2", "char_id": "ANN", "speaker_role": "announcer",
             "text": "and now", "samples": 12000, "sample_rate": 24000},
        ],
        "meta": {"story_brief_terms": {"setting": ["a station"]}},
    }
    policy = json.dumps({
        "video_models": {"announcer_video_model": {"engine_id": "", "custom": False},
                         "music_video_model": {"engine_id": "", "custom": False},
                         "character_video_model": {"engine_id": "", "custom": False}},
        "canvas": {"w": 832, "h": 480, "fps": 25},
    })
    patched, rev, report, done, episode_id = OTRShotLock().lock(
        json.dumps(led), audio_done="audio:done", video_policy_json=policy,
    )
    assert episode_id == ""          # fixture ledger carries no episode_id
    out = json.loads(patched)
    assert out["video"]["video_revision"] == 1 == rev
    assert len(out["video"]["shots"]) == 2
    assert {g["group_id"] for g in out["video"]["execution_groups"]} == {
        "grp_character_video", "grp_announcer_visual",
    }
    assert done == "shot_lock:done:rev=1"
    assert "shot_lock_revision=1" in report


# ---------------------------------------------------------------------------
# CW-2 -- AS-3 GPU-residency lease (cross-process single heavy engine)
# ---------------------------------------------------------------------------
def test_cross_subproject_single_lease(tmp_path):
    """AS-3: ONE GPU-residency lease, cross-process. A second acquire on a
    live-held lock fails closed (timeout); release frees it; a stale lock left
    by a DEAD pid is reclaimed so a crashed sidecar never wedges the platform.
    The lease is a SHARED module imported identically by C + B sidecars."""
    from nodes._otr_shared import gpu_residency as gr

    lockdir = tmp_path / "lease.lockdir"
    lease = gr.acquire(timeout_s=2.0, lockdir=lockdir)
    assert gr.is_held(lockdir)
    assert gr.read_owner(lockdir)[0] == os.getpid()

    # Held by a LIVE owner (this process) -> a second acquire fails closed.
    with pytest.raises(gr.LeaseTimeout):
        gr.acquire(timeout_s=0.3, poll_s=0.05, lockdir=lockdir)

    # Release frees it for the next consumer; release is idempotent.
    gr.release(lease)
    assert not gr.is_held(lockdir)
    gr.release(lease)  # no-op second release

    lease2 = gr.acquire(timeout_s=2.0, lockdir=lockdir)
    assert gr.is_held(lockdir)
    gr.release(lease2)

    # Stale lock from a DEAD pid is reclaimed, not waited on.
    lockdir.mkdir(parents=True, exist_ok=True)
    dead_pid = 2_000_000_000  # implausibly high -> pid_exists() is False
    (lockdir / "owner").write_text(f"{dead_pid}\nstaletoken\n0\n", encoding="utf-8")
    reclaimed = gr.acquire(timeout_s=2.0, lockdir=lockdir)
    assert reclaimed.pid == os.getpid()
    gr.release(reclaimed)
    assert not gr.is_held(lockdir)


def test_gpu_residency_probe_is_machine_wide_int():
    """AS-3: probe_used_mb returns a machine-wide int (never get_free_memory);
    cold-import clean (no torch pulled). 0 is a valid 'NVML unavailable' value
    on a CPU box -- nvml_available distinguishes it for a fail-closed gate."""
    from nodes._otr_shared import gpu_residency as gr
    used = gr.probe_used_mb()
    assert isinstance(used, int) and used >= 0
    assert isinstance(gr.nvml_available(), bool)
    # When NVML is unavailable, the bounded floor-wait fails closed.
    if not gr.nvml_available():
        assert gr.wait_until_below_mb(1, attempts=1, sleep_s=0.0) is False
    import importlib
    src = importlib.import_module("nodes._otr_shared.gpu_residency")
    assert "torch" not in dir(src)


# ---------------------------------------------------------------------------
# CW-2 -- AS-5 content-addressed portrait-ledger resolution
# ---------------------------------------------------------------------------
def test_portrait_ledger_resolution(tmp_path):
    """AS-5: char_id -> content-addressed portrait path; lookup is by char_id,
    never the display name (BUG-098); unknown / unstamped both fail closed."""
    from nodes._otr_shared import portrait_ledger as pl

    h = "a" * 64
    led = {"cast": [
        {"char_id": "c1", "name": "BABA", "portrait_content_hash": h},
        {"char_id": "c2", "name": "NOON"},  # no portrait stamped
    ]}
    p = pl.resolve_portrait_path(led, "c1", output_dir=str(tmp_path))
    # OH-1 (output-tree contract 2026-06-11): the content-addressed pool
    # moved from top-level otr/stills to the episodes/_shared/cache tier.
    assert p == (tmp_path / "otr" / "episodes" / "_shared" / "cache"
                 / f"{h}.png")

    # Lookup is by char_id ONLY -- the display name does not resolve.
    with pytest.raises(pl.PortraitUnresolved):
        pl.resolve_portrait_path(led, "BABA", output_dir=str(tmp_path))
    # Unstamped char + unknown char both fail closed.
    with pytest.raises(pl.PortraitUnresolved):
        pl.resolve_portrait_path(led, "c2", output_dir=str(tmp_path))
    with pytest.raises(pl.PortraitUnresolved):
        pl.resolve_portrait_path(led, "missing", output_dir=str(tmp_path))

    # Hash is over DECODED pixels: content-addressed + re-encoding-stable.
    import numpy as np
    px = np.zeros((4, 4, 3), dtype=np.uint8)
    h1 = pl.compute_portrait_hash(px)
    px2 = px.copy()
    px2[0, 0, 0] = 1
    assert h1 != pl.compute_portrait_hash(px2)        # diff pixels -> new hash
    assert h1 == pl.compute_portrait_hash(px.copy())  # same pixels -> same hash


def test_portrait_magic_byte(tmp_path):
    """AS-5: stamp_portrait writes a REAL PNG (magic bytes) at the
    content-addressed path, stamps the ledger by char_id, never overwrites
    (idempotent for identical pixels), and a regen yields a NEW hash + path
    while the old file stays intact (mesh-cache stays correct across regen)."""
    from nodes._otr_shared import portrait_ledger as pl
    import numpy as np

    led = {"cast": [{"char_id": "c1", "name": "BABA"}]}
    px = np.full((8, 8, 3), 127, dtype=np.uint8)
    path = pl.stamp_portrait(led, "c1", px, output_dir=str(tmp_path))
    assert path.exists()
    with open(path, "rb") as fh:
        assert fh.read(8) == b"\x89PNG\r\n\x1a\n"

    # Ledger stamped by char_id; resolve round-trips to the same path.
    h = led["cast"][0]["portrait_content_hash"]
    assert path.name == f"{h}.png"
    assert pl.resolve_portrait_path(led, "c1", output_dir=str(tmp_path)) == path

    # Idempotent: identical pixels -> same path, no overwrite error.
    again = pl.stamp_portrait(led, "c1", px.copy(), output_dir=str(tmp_path))
    assert again == path

    # Regen (different pixels) -> NEW hash + path; the prior file is intact.
    px2 = np.full((8, 8, 3), 200, dtype=np.uint8)
    path2 = pl.stamp_portrait(led, "c1", px2, output_dir=str(tmp_path))
    assert path2 != path and path.exists() and path2.exists()
    assert led["cast"][0]["portrait_content_hash"] == pl.compute_portrait_hash(px2)


# ---------------------------------------------------------------------------
# CW-2 -- clip-budget coverage: episode-duration bound + empty-script floor
# (the CW-1 compute_clip_budget is the single source; these extend coverage)
# ---------------------------------------------------------------------------
def test_clip_budget_sum_equals_episode_duration():
    """Clip-budget per-beat frames sum to EXACTLY the episode duration in
    frames -- (total_samples*fps)//sample_rate -- so sum <= episode_duration
    holds by construction (no double-count, no gap)."""
    fps, sr = 25, 24000
    beats = [
        {"beat_id": "b1", "role": "character_video", "char_id": "c1", "text": "a",
         "samples": 30000, "sample_rate": sr, "dur_s": None},
        {"beat_id": "b2", "role": "retired_role_a", "char_id": "", "text": "b",
         "samples": 18000, "sample_rate": sr, "dur_s": None},
        {"beat_id": "b3", "role": "retired_role_b", "char_id": "", "text": "",
         "samples": 6000, "sample_rate": sr, "dur_s": None},
    ]
    budget = sl.compute_clip_budget(beats, {}, fps)
    total_samples = sum(b["samples"] for b in beats)
    episode_frames = (total_samples * fps) // sr
    assert budget["total_frames"] == episode_frames
    assert budget["total_frames"] == sum(budget["per_beat"].values())


def test_clip_budget_empty_script_radio_floor():
    """Empty script -> radio-floor: zero frames, NO abort.
    OTR_ShotLock still stamps a video section for an episode with no lines."""
    budget = sl.compute_clip_budget([], {}, 25)
    assert budget["total_frames"] == 0
    assert budget["per_beat"] == {}

    led = json.dumps({"cast": [], "lines": [], "meta": {}})
    patched, rev, report, done, _epid = OTRShotLock().lock(
        led, audio_done="audio:done", video_policy_json="{}",
    )
    out = json.loads(patched)
    assert out["video"]["shots"] == []
    assert out["video"]["clip_budget"]["total_frames"] == 0
    assert done.startswith("shot_lock:done:")


def test_engine_non_invocable_preflight_check(monkeypatch):
    """Verify that OTRVideoDirector raises EngineNotRunnableError for unrunnable/non-invocable engines."""
    from nodes.otr_video_director import OTRVideoDirector
    from nodes._otr_shared.engine_registry_base import EngineNotRunnableError
    from nodes._otr_video_engines import registry as vreg

    class NonInvocableFixture:
        name = "test_non_invocable_video"
        roles = ()
        default_roles = ()
        commercial_clean = True
        requires_flag = None
        family = "static_image_gen"
        required_inputs = ()
        invocable = False
        invocability_reason = "fixture non-invocable reason"

    monkeypatch.setitem(
        vreg._VIDEO_REGISTRY._registry,
        "test_non_invocable_video",
        NonInvocableFixture(),
    )

    # Mock descriptors
    descriptors = [
        {"engine_id": "test_non_invocable_video", "family": "static_image_gen", "roles": ("music_visual",), "required_inputs": ()},
        {"engine_id": "still_pan", "family": "static_image_gen", "roles": ("music_visual",), "required_inputs": ()}
    ]

    # Pick a deliberately non-invocable fixture.
    with pytest.raises(EngineNotRunnableError) as excinfo:
        OTRVideoDirector._resolve_and_validate(
            slot="music_video_model", picked="test_non_invocable_video", custom={}, descriptors=descriptors, warnings=[]
        )
    assert "not runnable: fixture non-invocable reason" in str(excinfo.value)

    # Pick invocable model still_pan
    res = OTRVideoDirector._resolve_and_validate(
        slot="music_video_model", picked="still_pan", custom={}, descriptors=descriptors, warnings=[]
    )
    assert res["engine_id"] == "still_pan"


def test_preflight_uses_effective_redirected_engine_for_humo_bookend(monkeypatch):
    """A HuMo policy pick for music/announcer bookends is structurally routed
    by build_request_from_shot before render. Cast-time preflight must validate
    that effective ltx_audio_in request, not the raw HuMo candidate, or it
    falsely halts on missing HuMo init_image."""
    monkeypatch.delenv("OTR_ENABLE_HUMO_HOSTS", raising=False)
    beats = [{"beat_id": "b1", "role": "music_visual", "char_id": "",
              "dur_s": 2.0}]
    budget = {"total_frames": 50, "per_beat": {"b1": 50}}
    policy = {"video_models": {"music_video_model": "humo_1.7B_169"}}
    ledger = {
        "lines": [{"line_id": "b1", "char_id": "",
                   "start_s": 0.0, "dur_s": 2.0}],
        "images": {"images": [
            {"object_id": "still_b1", "kind": "scene_open", "beat_id": "b1",
             "path": "X:/episode/stills/b1.png"},
        ]},
        "meta": {"story_brief_terms": {"setting": ["an orbital relay room"]}},
        "video": {"fps": 25, "shots": []},
    }

    groups, shots = sl.build_execution_plan(beats, budget, {}, policy, ledger=ledger)

    assert groups[0]["engine_id"] == "humo_1.7B_169"
    assert shots[0]["engine_id"] == "humo_1.7B_169"


def test_preflight_defers_redirected_announcer_radio_face(monkeypatch):
    """The redirected ltx_audio_in announcer may require the wide radio-face
    still, but ShotLock runs before MetaBrief/dispatcher materialize it."""
    monkeypatch.delenv("OTR_ENABLE_HUMO_HOSTS", raising=False)
    monkeypatch.setenv("OTR_LTX_AV_RECIPE", "ia2v_canonical")
    monkeypatch.setenv("OTR_LTX_AV_UNET", "ltx-2.3-22b-dev-Q3_K_M.gguf")
    beats = [{"beat_id": "b3", "role": "announcer_visual", "char_id": "",
              "dur_s": 2.0}]
    budget = {"total_frames": 50, "per_beat": {"b3": 50}}
    policy = {"video_models": {"announcer_video_model": "humo_1.7B_169"}}
    ledger = {
        "lines": [{"line_id": "b3", "char_id": "",
                   "start_s": 0.0, "dur_s": 2.0}],
        "images": {"images": [
            {"object_id": "still_b3", "kind": "scene_open", "beat_id": "b3",
             "path": "X:/episode/stills/b3.png"},
        ]},
        "meta": {"story_brief_terms": {"setting": ["a transmitter room"]}},
        "video": {"fps": 25, "shots": []},
    }

    _groups, shots = sl.build_execution_plan(beats, budget, {}, policy, ledger=ledger)
    assert shots[0]["engine_id"] == "humo_1.7B_169"


def test_preflight_defers_ltx_ambient_audio_to_render_batch(monkeypatch):
    """ShotLock cannot slice the frozen master mix because RenderBatch owns
    master_audio_path. Ambient-audio lanes must not freeze-halt at cast time
    when line timing is absent; render-time still supplies the bounded slice."""
    monkeypatch.delenv("OTR_ENABLE_HUMO_HOSTS", raising=False)
    beats = [{"beat_id": "b6", "role": "music_visual", "char_id": ""}]
    budget = {"total_frames": 50, "per_beat": {"b6": 50}}
    policy = {"video_models": {"music_video_model": "ltx_audio_in"}}
    ledger = {
        "lines": [{"line_id": "b6", "char_id": ""}],
        "images": {"images": [
            {"object_id": "still_b6", "kind": "scene_open", "beat_id": "b6",
             "path": "X:/episode/stills/b6.png"},
        ]},
        "meta": {"story_brief_terms": {"setting": ["a signal room"]}},
        "video": {
            "fps": 25,
            "shots": [
                {"shot_id": "shot_b1", "beat_id": "b1",
                 "target_frame_count": 100},
                {"shot_id": "shot_b6", "beat_id": "b6",
                 "target_frame_count": 50},
            ],
        },
    }

    _groups, shots = sl.build_execution_plan(beats, budget, {}, policy,
                                             ledger=ledger)

    assert shots[0]["engine_id"] == "ltx_audio_in"


def test_preflight_defers_humo_init_image_to_image_phase(monkeypatch):
    """ShotLock runs before OTR_ImageGenDispatcher, so a character HuMo pick
    must not freeze-halt just because the portrait row has not been written
    into the ledger yet."""
    monkeypatch.delenv("OTR_ENABLE_HUMO_HOSTS", raising=False)
    beats = [{"beat_id": "b2", "role": "character_video", "char_id": "c2",
              "dur_s": 2.0}]
    budget = {"total_frames": 50, "per_beat": {"b2": 50}}
    policy = {"video_models": {"character_video_model": "humo_1.7B_169"}}
    ledger = {
        "lines": [{"line_id": "b2", "char_id": "c2",
                   "start_s": 0.0, "dur_s": 2.0}],
        "images": {"images": []},
        "meta": {},
        "video": {"fps": 25, "shots": []},
    }

    _groups, shots = sl.build_execution_plan(beats, budget, {}, policy, ledger=ledger)
    assert shots[0]["engine_id"] == "humo_1.7B_169"


def test_render_time_humo_init_image_gap_is_terminal():
    """The cast-time deferral is not a fallback: a final HuMo render request
    without init_image still fails LOUD as a RenderError."""
    from nodes._otr_video_engines import render_driver as rd

    shot = {"shot_id": "shot_b2", "role": "character_video",
            "engine_id": "humo_1.7B_169", "family": "audio_driven_face",
            "target_frame_count": 25}
    req = rd.build_request(shot, {"audio_ref": "X:/voice.wav"}, 25,
                           canvas=(832, 480))

    with pytest.raises(rd.RenderError) as excinfo:
        rd.render_shot(shot, req)
    msg = str(excinfo.value)
    assert "humo_1.7B_169" in msg
    assert "init_image" in msg
