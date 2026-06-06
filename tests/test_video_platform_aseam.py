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
    with pytest.raises(vreg.EngineUnusable):
        vreg.assert_usable("ltx", "character_video")  # role not served
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
        {"engine_id": "abstract", "roles": ("background_abstract", "music_visual"),
         "required_inputs": ()},
        {"engine_id": "humo", "roles": ("character_video", "announcer_visual"),
         "required_inputs": ("audio_ref", "init_image")},
        {"engine_id": "ltx", "roles": ("scene_broll", "background_abstract"),
         "required_inputs": ("text_prompt",)},
    ]
    assert rc.filter_engines_for_role("background_abstract", descs) == ["abstract", "ltx"]
    assert rc.filter_engines_for_role("character_video", descs) == ["humo"]
    assert rc.filter_engines_for_role("music_visual", descs) == ["abstract"]
    with pytest.raises(rc.RoleCompatError):
        rc.filter_engines_for_role("not_a_role", descs)


def test_role_compat_fail_closed_on_malformed():
    """A descriptor missing keys / declaring an unknown token is EXCLUDED, not
    raised (fail-closed)."""
    descs = [
        {"engine_id": "ok", "roles": ("background_abstract",), "required_inputs": ()},
        {"engine_id": "bad_token", "roles": ("background_abstract",),
         "required_inputs": ("nonsense",)},
        {"engine_id": "no_roles", "required_inputs": ()},
        {"roles": ("background_abstract",), "required_inputs": ()},  # no engine_id
    ]
    assert rc.filter_engines_for_role("background_abstract", descs) == ["ok"]


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
        request_id="r", shot_id="s", role="background_abstract",
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
    """Golden pin on the director's required widget order/names (drift guard)."""
    req = list(OTRVideoDirector.INPUT_TYPES()["required"].keys())
    assert req == [
        "announcer_video_model", "music_video_model", "other_beats_video_model",
        "announcer_image_model", "music_image_model", "other_beats_image_model",
        "other_beats_clip_mode", "other_beats_n", "fps", "canvas_w", "canvas_h",
        "seed_mode", "request_seed", "allow_auto_fallback",
    ]
    assert "seed" not in req  # V-7: no widget literally named 'seed'
    it = OTRVideoDirector.INPUT_TYPES()
    assert it["required"]["other_beats_clip_mode"][0] == ["unique_per_beat", "pool_n_loop"]


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
    assert "background_abstract" in us
    assert isinstance(report, str) and "OTR_VideoProbe" in report


def test_director_policy_json_and_clamp(clean_video_registry):
    out = OTRVideoDirector().direct(
        announcer_video_model=ADD_CUSTOM, music_video_model=ADD_CUSTOM,
        other_beats_video_model=ADD_CUSTOM, announcer_image_model="Flux (gen 1)",
        music_image_model="Flux (gen 1)", other_beats_image_model="Flux (gen 1)",
        other_beats_clip_mode="pool_n_loop", other_beats_n=8, fps=25,
        canvas_w=832, canvas_h=480, seed_mode="request_hash", request_seed=0,
        allow_auto_fallback=True,
    )
    policy = json.loads(out[0])
    assert policy["other_beats"] == {"clip_mode": "pool_n_loop", "pool_n": 8}
    assert policy["canvas"] == {"w": 832, "h": 480, "fps": 25}
    assert policy["video_models"]["announcer_video_model"]["custom"] is True


def test_director_fail_closed_incompatible_pick(clean_video_registry):
    """A registered engine that fits NONE of a slot's roles fails closed."""
    vreg.register(_stub_engine(
        name="audioface", roles=("character_video",), default_roles=(),
        required_inputs=("audio_ref", "init_image"),
    ))
    with pytest.raises(ValueError):
        OTRVideoDirector().direct(
            announcer_video_model="audioface",  # needs audio_ref+init_image; A has them... use music
            music_video_model="audioface",      # music_visual cannot supply audio_ref -> fail
            other_beats_video_model=ADD_CUSTOM, announcer_image_model="Flux (gen 1)",
            music_image_model="Flux (gen 1)", other_beats_image_model="Flux (gen 1)",
            other_beats_clip_mode="unique_per_beat", other_beats_n=8, fps=25,
            canvas_w=832, canvas_h=480, seed_mode="request_hash", request_seed=0,
            allow_auto_fallback=True,
        )


# ---------------------------------------------------------------------------
# ShotLock -- clip budget + image_done + M4 derivation
# ---------------------------------------------------------------------------


def test_clip_budget_no_double_count():
    beats = [
        {"beat_id": "b1", "role": "character_video", "char_id": "c1", "text": "hi",
         "samples": 24000, "sample_rate": 24000, "dur_s": None},
        {"beat_id": "b2", "role": "background_abstract", "char_id": "", "text": "",
         "samples": 12000, "sample_rate": 24000, "dur_s": None},
    ]
    budget = sl.compute_clip_budget(beats, {}, 25)
    assert budget["per_beat"]["b1"] == 25
    assert budget["per_beat"]["b2"] == 12
    assert budget["total_frames"] == 37 == sum(budget["per_beat"].values())


def test_clip_budget_clamp_warning():
    beats = [
        {"beat_id": f"o{i}", "role": "background_abstract", "char_id": "",
         "text": "", "samples": 1000, "sample_rate": 24000, "dur_s": None}
        for i in range(3)
    ]
    budget = sl.compute_clip_budget(
        beats, {"other_beats": {"clip_mode": "pool_n_loop", "pool_n": 8}}, 25,
    )
    assert budget["other_beats_render_count"] == 3
    assert any("clamped" in w for w in budget["warnings"])


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


def test_passpm_empty_json_reseed_then_template():
    led = _char_ledger()
    calls = {"n": 0}

    def empty_llm(_prompt):
        calls["n"] += 1
        return ""  # always collapses

    creative, warns = sl.derive_creative_directives(
        _char_beat("hello there"), led["meta"], led, llm_fn=empty_llm, max_reseed=2,
    )
    c = creative["b1"]
    assert c["source"] in ("template", "template_consistency")
    assert c["text_prompt"] == sl._deterministic_template(
        "a tall weathered spacer with a scar", "a derelict orbital station",
        "hello there",
    )
    assert calls["n"] == 3  # initial + 2 reseeds
    assert any("reseed" in w for w in warns)


def test_story_consistency_gate_warns_and_falls_back():
    led = _char_ledger()

    def halluc_llm(_prompt):
        # authors a full prompt that drops BOTH the cast trait and the setting
        return json.dumps([{"beat_id": "b1", "expression": "x",
                            "text_prompt": "bright sunny flower meadow"}])

    creative, warns = sl.derive_creative_directives(
        _char_beat(), led["meta"], led, llm_fn=halluc_llm,
        consistency_gate_warn_only=True,
    )
    c = creative["b1"]
    assert c["source"] == "template_consistency"
    assert "spacer" in c["text_prompt"]
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
    beats = [{"beat_id": "b1", "role": "background_abstract", "char_id": "",
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
                         "other_beats_video_model": {"engine_id": "", "custom": False}},
        "canvas": {"w": 832, "h": 480, "fps": 25},
        "other_beats": {"clip_mode": "unique_per_beat", "pool_n": 8},
    })
    patched, rev, report, done = OTRShotLock().lock(
        json.dumps(led), audio_done="audio:done", video_policy_json=policy,
    )
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
    assert p == tmp_path / "otr" / "stills" / f"{h}.png"

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
        {"beat_id": "b2", "role": "scene_broll", "char_id": "", "text": "b",
         "samples": 18000, "sample_rate": sr, "dur_s": None},
        {"beat_id": "b3", "role": "background_abstract", "char_id": "", "text": "",
         "samples": 6000, "sample_rate": sr, "dur_s": None},
    ]
    budget = sl.compute_clip_budget(beats, {}, fps)
    total_samples = sum(b["samples"] for b in beats)
    episode_frames = (total_samples * fps) // sr
    assert budget["total_frames"] == episode_frames
    assert budget["total_frames"] == sum(budget["per_beat"].values())


def test_clip_budget_empty_script_radio_floor():
    """Empty script -> radio-floor: zero frames, zero render count, NO abort.
    OTR_ShotLock still stamps a video section for an episode with no lines."""
    budget = sl.compute_clip_budget([], {}, 25)
    assert budget["total_frames"] == 0
    assert budget["other_beats_render_count"] == 0
    assert budget["per_beat"] == {}

    led = json.dumps({"cast": [], "lines": [], "meta": {}})
    patched, rev, report, done = OTRShotLock().lock(
        led, audio_done="audio:done", video_policy_json="{}",
    )
    out = json.loads(patched)
    assert out["video"]["shots"] == []
    assert out["video"]["clip_budget"]["total_frames"] == 0
    assert done.startswith("shot_lock:done:")
