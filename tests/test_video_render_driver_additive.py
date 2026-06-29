"""Additive CPU coverage for the in-process render driver (A-S7.5).

New, self-contained tests complementing tests/test_video_render_driver.py. The
pure helpers (engine_family / build_full_ledger / build_soak_fixture range guard
/ classify_failure kind map / make_fallback_of overlay) plus a CPU exercise of
the REAL render loop (run_episode) driven by STUB engines registered into a
snapshot/restore registry: it proves a HARD failure RAISES LOUD (no fallbacks --
operator 2026-06-16; no engine swap, no still floor), the frozen audio section is
never touched, the input ledger is not mutated, and a clean run is deterministic.
No GPU, no model load. UTF-8, no BOM, ASCII-only, SFW.
"""
from __future__ import annotations

import pytest

from nodes._otr_shared import retry_taxonomy as rt
from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import render_driver as rd


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def test_engine_family_known_and_unknown():
    assert rd.engine_family("humo") == "audio_driven_face"
    assert rd.engine_family("still_kenburns") == "static_motion"
    assert rd.engine_family("totally_unknown_xyz") == "abstract"
    assert rd.engine_family("totally_unknown_xyz", default="static_motion") == (
        "static_motion")


def test_build_full_ledger_freezes_audio():
    section = {"video_revision": 1, "shots": []}
    led = rd.build_full_ledger(section)
    assert led["audio"]["master_audio_sha256"] == rd.FROZEN_AUDIO_SHA
    assert led["audio"]["ledger_frozen"] is True
    assert led["video"] is section


def test_build_soak_fixture_oom_index_out_of_range():
    with pytest.raises(ValueError):
        rd.build_soak_fixture(n_beats=40, oom_index=40)
    with pytest.raises(ValueError):
        rd.build_soak_fixture(n_beats=40, oom_index=-1)


def test_build_soak_fixture_shape_and_empty_trails():
    section, meta = rd.build_soak_fixture(n_beats=12, oom_index=5)
    assert len(section["shots"]) == 12
    assert meta["oom_shot_id"] == "shot_0005"
    oom = section["shots"][5]
    assert (oom["engine_id"], oom["family"]) == ("soak_oom_3d", "character_3d")
    # every shot starts with an empty degradation trail
    assert all(s["degradation_trail"] == [] for s in section["shots"])
    # shot 0 is the first profile in the rotation
    assert section["shots"][0]["engine_id"] == "humo"


def test_classify_failure_specific_kind_mappings():
    assert rd.classify_failure(rd.OomSignal("x")) is rt.FailureKind.OOM
    for exc in (LookupError(), KeyError("k"), FileNotFoundError()):
        assert rd.classify_failure(exc) is rt.FailureKind.DEPENDENCY_MISSING

    class WrapperNodeMissing(Exception):
        pass

    class GraphExecutionError(Exception):
        pass

    assert rd.classify_failure(WrapperNodeMissing()) is (
        rt.FailureKind.DEPENDENCY_MISSING)
    assert rd.classify_failure(GraphExecutionError()) is rt.FailureKind.INVALID_DAG
    assert rd.classify_failure(RuntimeError("boom")) is (
        rt.FailureKind.CRASH_BEFORE_LOAD)


def test_make_fallback_of_overlay_and_terminus():
    fb = rd.make_fallback_of(synth={"custom_engine": "humo"})
    assert fb("custom_engine") == "humo"
    assert fb("soak_oom_3d") == "humo"                  # default synth soak-stub overlay
    assert fb("still_kenburns") is None                 # floor is terminal
    assert fb("zzz_not_registered_nonfloor") == rd.UNIVERSAL_FLOOR


# --------------------------------------------------------------------------- #
# CPU exercise of the real render loop via STUB engines
# --------------------------------------------------------------------------- #
class _StubBase:
    family = "abstract"
    roles = ("background_abstract",)
    default_roles = ()
    commercial_clean = True
    requires_flag = None
    fallback_engine = None

    def load(self):
        pass

    def unload(self):
        pass

    def assert_usable(self, host_caps=None, profile=None, request_template=None):
        return self.name

    def prepare(self, host_caps=None, profile=None, session_ctx=None):
        return {"engine_id": self.name}

    def canonicalize(self, raw, request, profile=None):
        return {"clip_id": request["shot_id"], "engine_id": self.name,
                "family": self.family, "frame_count": 25, "path": ""}

    def teardown(self, prepared):
        pass


class _StubOK(_StubBase):
    name = "stub_ok"

    def render_clip(self, request, prepared):
        return {"raw": True}


class _StubFail(_StubBase):
    name = "stub_fail"

    def render_clip(self, request, prepared):
        raise RuntimeError("boom")


@pytest.fixture
def stub_registry():
    """Snapshot the global video registry, add the stubs, restore after."""
    saved = dict(vreg._VIDEO_REGISTRY._registry)
    vreg.register(_StubOK())
    vreg.register(_StubFail())
    try:
        yield vreg._VIDEO_REGISTRY
    finally:
        vreg._VIDEO_REGISTRY._registry.clear()
        vreg._VIDEO_REGISTRY._registry.update(saved)


def _two_shot_ledger():
    section = {"video_revision": 1, "fps": 25, "shots": [
        {"shot_id": "shot_0000", "beat_id": "b0", "role": "background_abstract",
         "engine_id": "stub_ok", "family": "abstract", "group_id": "g0",
         "target_frame_count": 25, "degradation_trail": []},
        {"shot_id": "shot_0001", "beat_id": "b1", "role": "background_abstract",
         "engine_id": "stub_fail", "family": "abstract", "group_id": "g1",
         "target_frame_count": 25, "degradation_trail": []},
    ]}
    return rd.build_full_ledger(section)


def _fb(name):
    return {"stub_fail": "stub_ok"}.get(name)


def test_run_episode_fails_loud_no_fallback(stub_registry):
    # No fallbacks (operator 2026-06-16): the failing shot RAISES RenderError;
    # there is NO swap to stub_ok and NO still floor. The frozen input audio is
    # untouched (deep-copy transaction).
    ledger = _two_shot_ledger()
    with pytest.raises(rd.RenderError):
        rd.run_episode(ledger, fallback_of=_fb)
    assert ledger["audio"]["master_audio_sha256"] == rd.FROZEN_AUDIO_SHA


def test_run_episode_does_not_mutate_input_ledger(stub_registry):
    ledger = _two_shot_ledger()
    with pytest.raises(rd.RenderError):
        rd.run_episode(ledger, fallback_of=_fb)
    # the original ledger is deep-copied; its failing shot is untouched
    assert ledger["video"]["shots"][1]["engine_id"] == "stub_fail"
    assert ledger["video"]["shots"][1]["degradation_trail"] == []
    assert "runtime_fallback_decisions" not in ledger["video"]


def test_run_episode_is_deterministic(stub_registry):
    # determinism on a clean (all-success) episode -- a single attempt per shot.
    def _ok():
        led = _two_shot_ledger()
        led["video"]["shots"][1]["engine_id"] = "stub_ok"
        return led
    a = rd.run_episode(_ok(), fallback_of=_fb)
    b = rd.run_episode(_ok(), fallback_of=_fb)
    assert a["trace"] == b["trace"]
    assert a["trace"][1]["attempts"] == ["stub_ok"]     # single attempt, no swap
    assert a["trace"][1]["final_engine"] == "stub_ok"


# --------------------------------------------------------------------------- #
# Per-shot request builder + run_real_episode (the REAL episode wiring path).
# Proves build_request_from_shot resolves the portrait + per-beat voice audio +
# the M4 prompt + the hash-keyed seed from a ShotLock-shaped ledger, and that
# run_real_episode drives per-shot requests while keeping the frozen audio
# section untouched, the input unmutated, and two runs deterministic. CPU only.
# --------------------------------------------------------------------------- #
def _real_ledger():
    """A minimal ShotLock-shaped ledger: lines (char_id + per-engine voice wav +
    timing), an OTR_ImageGenDispatcher images write-back (portrait per char),
    and two shots whose engine is the recording stub."""
    return {
        "audio": {"master_audio_sha256": rd.FROZEN_AUDIO_SHA, "ledger_frozen": True},
        "lines": [
            {"line_id": "b001", "char_id": "announcer", "speaker_role": "announcer",
             "start_s": 0.0, "dur_s": 2.0, "bark_wav_path": "X:/a/seg_b001.wav"},
            {"line_id": "b002", "char_id": "c03", "speaker_role": "character",
             "start_s": 2.0, "dur_s": 1.6, "bark_wav_path": "X:/a/seg_b002.wav"},
        ],
        "images": {"images": [
            {"object_id": "announcer", "path": "X:/img/announcer.png"},
            {"object_id": "c03", "path": "X:/img/c03.png"},
        ]},
        "video": {"video_revision": 1, "fps": 25, "shots": [
            {"shot_id": "shot_b001", "source_line_ids": ["b001"],
             "engine_id": "stub_record", "family": "", "group_id": "grp_a",
             "target_frame_count": 50, "degradation_trail": [],
             "render_request_hash": "hash_b001",
             "cache_keys": {"request_hash": "hash_b001"},
             "creative": {"text_prompt": "announcer at the mic, warm tungsten light"}},
            {"shot_id": "shot_b002", "source_line_ids": ["b002"],
             "engine_id": "stub_record", "family": "", "group_id": "grp_c",
             "target_frame_count": 40, "degradation_trail": [],
             "render_request_hash": "hash_b002",
             "cache_keys": {"request_hash": "hash_b002"},
             "creative": {"text_prompt": "c03 reacts, close-up"}},
        ]},
    }


def test_build_request_from_shot_maps_portrait_audio_prompt_timing():
    led = _real_ledger()
    shot = led["video"]["shots"][1]            # character beat b002 / c03
    req = rd.build_request_from_shot(shot, led)
    assert req["asset_refs"]["init_image"] == "X:/img/c03.png"
    assert req["audio_ref"] == {"path": "X:/a/seg_b002.wav"}
    assert req["text_prompt"] == "c03 reacts, close-up"
    assert req["timing"]["target_frame_count"] == 40
    # W7-pre builder migration: the schema field is target_duration_s (never
    # dur_s); char_id rides in conditioning_refs (top-level = schema extra).
    assert req["timing"]["start_s"] == 2.0
    assert req["timing"]["target_duration_s"] == 1.6
    assert req["conditioning_refs"]["char_id"] == "c03"
    assert "char_id" not in req and "dur_s" not in req["timing"]


def test_build_request_from_shot_no_portrait_or_audio_is_blank():
    led = _real_ledger()
    led["lines"].append({"line_id": "b003", "char_id": "",
                         "speaker_role": "background"})
    shot = {"shot_id": "shot_b003", "source_line_ids": ["b003"],
            "engine_id": "abstract", "target_frame_count": 10, "creative": {}}
    req = rd.build_request_from_shot(shot, led)
    assert req["asset_refs"] == {}             # no portrait -> empty asset_refs
    assert req["audio_ref"] is None            # no per-line wav -> None (LOUD)
    assert req["timing"]["target_frame_count"] == 10


def test_build_request_from_shot_seed_keyed_to_hash_not_index():
    led = _real_ledger()
    s1, s2 = led["video"]["shots"]
    r1 = rd.build_request_from_shot(s1, led)
    r2 = rd.build_request_from_shot(s2, led)
    # build_request's numeric-index seed collapses non-numeric shot_ids to one
    # value; the hash-keyed seed makes the two beats differ AND stay stable.
    assert r1["seed_bundle"]["request_seed"] != r2["seed_bundle"]["request_seed"]
    assert rd.build_request_from_shot(s1, led)["seed_bundle"] == r1["seed_bundle"]


def _flux_still_opener_ledger():
    """A ShotLock-shaped ledger whose OPENER beat (b000_music_open, char_id="")
    picks flux_still for the music slot, with the ST-3 scene-still write-back
    present (kind scene_open, beat_id b000_music_open) -- the real opener shape
    from a live render (signal_lost_..._171037)."""
    return {
        "audio": {"master_audio_sha256": rd.FROZEN_AUDIO_SHA, "ledger_frozen": True},
        "lines": [
            {"line_id": "b000_music_open", "char_id": "",
             "speaker_role": "music_open", "start_s": 0.0, "dur_s": 9.5},
            {"line_id": "b001", "char_id": "announcer", "speaker_role": "announcer",
             "start_s": 9.5, "dur_s": 2.0, "bark_wav_path": "X:/a/seg_b001.wav"},
        ],
        "images": {"images": [
            {"object_id": "still_b000_music_open", "kind": "scene_open",
             "beat_id": "b000_music_open", "char_id": "",
             "path": "X:/img/still_b000_music_open.png"},
            {"object_id": "announcer", "kind": "portrait",
             "path": "X:/img/announcer.png"},
        ]},
        "video": {"video_revision": 1, "fps": 25, "shots": [
            {"shot_id": "shot_b000_music_open",
             "source_line_ids": ["b000_music_open"],
             "engine_id": "flux_still", "family": "static_image_gen",
             "group_id": "grp_music", "target_frame_count": 238,
             "degradation_trail": [], "render_request_hash": "hash_b000",
             "cache_keys": {"request_hash": "hash_b000"},
             "creative": {"text_prompt": "radio console, warm tungsten glow"}},
        ]},
    }


def test_build_request_from_shot_flux_still_opener_conditions_on_scene_still():
    # FIX1 / BUG-LOCAL-403: flux_still (family static_image_gen) must condition
    # the opener on its beat scene still -- without it the cheap family draws a
    # black floor (the all-black opener centre). beat_id b000_music_open keys
    # both _beat_id_for_shot and the scene_open still row.
    led = _flux_still_opener_ledger()
    shot = led["video"]["shots"][0]
    req = rd.build_request_from_shot(shot, led)
    assert req["asset_refs"]["init_image"] == "X:/img/still_b000_music_open.png"
    assert req["observability"]["init_source"] == "scene_still"


def test_build_request_from_shot_flux_still_missing_still_is_loud_not_black():
    # No scene still for the opener -> init stays empty (the cheap family will
    # draw its floor) but the degrade is LOUD (the branch warns), never a silent
    # scene_still claim. init_source falls back to "none".
    led = _flux_still_opener_ledger()
    led["images"]["images"] = [r for r in led["images"]["images"]
                               if r["object_id"] != "still_b000_music_open"]
    shot = led["video"]["shots"][0]
    req = rd.build_request_from_shot(shot, led)
    assert req["asset_refs"] == {}                       # no still -> empty
    assert req["observability"]["init_source"] == "none"


def test_build_request_from_shot_flux_still_fills_landscape_canvas():
    # 2026-06-14 operator catch: flux_still (and the other still/floor families)
    # inherited build_request's 480x832 HuMo PORTRAIT default -> a skinny portrait
    # pillarboxed in the 16:9 frame. Non-face engines must FILL the landscape
    # canvas; only audio_driven_face (HuMo) keeps its portrait pillarbox.
    led = _flux_still_opener_ledger()
    req = rd.build_request_from_shot(led["video"]["shots"][0], led)
    assert (req["canvas"]["w"], req["canvas"]["h"]) == (1472, 832)
    assert req["canvas"]["w"] > req["canvas"]["h"]        # landscape, not portrait


def test_build_request_from_shot_humo_keeps_portrait_canvas():
    # the talking-head face family is the ONE that stays portrait (the accepted
    # pillarbox) -- the landscape fill must NOT touch audio_driven_face.
    led = _real_ledger()
    shot = dict(led["video"]["shots"][1])
    shot["engine_id"] = "humo_1.7B"                        # audio_driven_face
    req = rd.build_request_from_shot(shot, led)
    assert (req["canvas"]["w"], req["canvas"]["h"]) == (480, 832)  # portrait kept


def test_voice_audio_resolver_engine_fields_and_sfx_exclusion():
    assert rd._voice_audio_for_line({"audio_wav_path": "a.wav"}) == "a.wav"
    assert rd._voice_audio_for_line({"indextts2_wav_path": "i.wav"}) == "i.wav"
    assert rd._voice_audio_for_line({"bark_wav_path": "b.wav"}) == "b.wav"
    # sfx is never a face-driving voice; a music beat uses the music clip
    assert rd._voice_audio_for_line({"sfx_wav_path": "s.wav"}) == ""
    assert rd._voice_audio_for_line({"music_wav_path": "m.wav"}) == "m.wav"
    assert rd._voice_audio_for_line({}) == ""


class _StubRecord(_StubBase):
    """A success engine that records each request it renders (so the test can
    assert the per-shot portrait/audio/prompt actually reached the engine)."""
    name = "stub_record"
    family = "audio_driven_face"
    roles = ("announcer_visual", "character_video")
    seen = None

    def render_clip(self, request, prepared):
        self.seen.append(request)
        return {"raw": True}


@pytest.fixture
def record_registry():
    saved = dict(vreg._VIDEO_REGISTRY._registry)
    eng = _StubRecord()
    eng.seen = []
    vreg.register(eng)
    try:
        yield eng
    finally:
        vreg._VIDEO_REGISTRY._registry.clear()
        vreg._VIDEO_REGISTRY._registry.update(saved)


def test_run_real_episode_per_shot_requests_audio_frozen(record_registry):
    led = _real_ledger()
    res = rd.run_real_episode(led)
    assert len(res["clips"]) == 2
    seen = {r["shot_id"]: r for r in record_registry.seen}
    assert seen["shot_b001"]["asset_refs"]["init_image"] == "X:/img/announcer.png"
    assert seen["shot_b002"]["audio_ref"] == {"path": "X:/a/seg_b002.wav"}
    assert seen["shot_b001"]["timing"]["target_frame_count"] == 50
    # frozen audio section byte-identical; input ledger not mutated
    assert res["ledger"]["audio"]["master_audio_sha256"] == rd.FROZEN_AUDIO_SHA
    assert led["video"]["shots"][0]["engine_id"] == "stub_record"
    assert "runtime_fallback_decisions" not in led["video"]


def test_run_real_episode_is_deterministic(record_registry):
    a = rd.run_real_episode(_real_ledger())
    record_registry.seen.clear()
    b = rd.run_real_episode(_real_ledger())
    assert a["trace"] == b["trace"]


def test_run_episode_request_builder_overrides_default(record_registry):
    led = _real_ledger()
    calls = []

    def rb(shot, ledger, *, canvas=None):
        calls.append(shot["shot_id"])
        return rd.build_request_from_shot(shot, ledger, canvas=canvas)

    rd.run_episode(led, fallback_of=rd.make_fallback_of(), request_builder=rb)
    assert calls == ["shot_b001", "shot_b002"]   # the builder drove every shot


# --------------------------------------------------------------------------- #
# build_clip_manifest + the OTR_VideoRenderBatch mode="episode" entry (Chunk B).
# The manifest is the beat-ordered STRING contract OTR_SilentComposite consumes;
# the node episode path runs run_real_episode in-process (CPU here via stubs).
# --------------------------------------------------------------------------- #
def test_build_clip_manifest_beat_order_histogram_and_existence():
    result = {
        "ledger": {"video": {
            "video_revision": 1, "fps": 25,
            "canonical_canvas": {"w": 1472, "h": 832},
            "shots": [
                {"shot_id": "shot_b001", "source_line_ids": ["b001"],
                 "engine_id": "humo", "family": "audio_driven_face",
                 "target_frame_count": 50},
                {"shot_id": "shot_b002", "source_line_ids": ["b002"],
                 "engine_id": "abstract", "family": "abstract",
                 "target_frame_count": 30},
            ]}},
        "clips": {
            # a real on-disk path (this test file) proves the existence probe
            "shot_b001": {"engine_id": "humo", "family": "audio_driven_face",
                          "frame_count": 50, "path": __file__},
            "shot_b002": {"engine_id": "abstract", "family": "abstract",
                          "frame_count": 30, "path": ""},
        },
    }
    m = rd.build_clip_manifest(result, episode_id="ep1")
    assert [c["shot_id"] for c in m["clips"]] == ["shot_b001", "shot_b002"]
    assert [c["order"] for c in m["clips"]] == [0, 1]
    assert m["n_beats"] == 2 and m["total_target_frames"] == 80
    assert m["canvas"] == {"w": 1472, "h": 832} and m["fps"] == 25
    assert m["clip_count"] == 1                  # only the on-disk clip counts
    assert m["engine_histogram"] == {"humo": 1}  # empty-path clip excluded
    assert m["clips"][0]["exists"] is True and m["clips"][1]["exists"] is False


def test_video_render_batch_episode_mode_emits_manifest(record_registry, tmp_path,
                                                        monkeypatch):
    import json as _json
    from nodes.otr_video_render_batch import OTRVideoRenderBatch
    monkeypatch.setenv("OTR_OUTPUT_DIR", str(tmp_path))
    out = OTRVideoRenderBatch().render(
        mode="episode", beats=2, oom_index=0, frame_count=25,
        patched_ledger_json=_json.dumps(_real_ledger()))
    report_json, manifest_json = out["result"]    # 2-tuple: report + manifest
    manifest = _json.loads(manifest_json)
    assert [c["shot_id"] for c in manifest["clips"]] == ["shot_b001", "shot_b002"]
    assert all(c["engine_id"] == "stub_record" for c in manifest["clips"])
    assert manifest["n_beats"] == 2
    assert _json.loads(report_json)["mode"] == "episode"


def test_video_render_batch_episode_mode_bad_ledger_failsoft(monkeypatch, tmp_path):
    import json as _json
    from nodes.otr_video_render_batch import OTRVideoRenderBatch
    monkeypatch.setenv("OTR_OUTPUT_DIR", str(tmp_path))
    out = OTRVideoRenderBatch().render(mode="episode", beats=2, oom_index=0,
                                       frame_count=25, patched_ledger_json="{}")
    report_json, manifest_json = out["result"]
    assert manifest_json == ""                     # empty manifest, no crash
    assert _json.loads(report_json)["ok"] is False


def test_video_render_batch_mode_combo_offers_episode():
    # the mode PICKER must list "episode" (not just the render() branch) or the
    # live ComfyUI /prompt validator rejects mode='episode' as value_not_in_list.
    from nodes.otr_video_render_batch import OTRVideoRenderBatch
    modes = OTRVideoRenderBatch.INPUT_TYPES()["required"]["mode"][0]
    assert "episode" in modes and "soak" in modes and "single" in modes


# --------------------------------------------------------------------------- #
# OTR_FORCE_ENGINE_MAP override + the lipsync base provider (2026-06-09,
# operator experiment knob: the all-LTX / forced-engine episodes)
# --------------------------------------------------------------------------- #
def test_parse_engine_override_grammar():
    m = rd.parse_engine_override("*=ltx_video")
    assert m == {"*": "ltx_video"}
    m = rd.parse_engine_override(
        "character_video=humo, scene_broll=ltx_video")
    assert m == {"character_video": "humo", "scene_broll": "ltx_video"}
    with pytest.raises(ValueError):
        rd.parse_engine_override("character_video")          # no '='
    with pytest.raises(ValueError):
        rd.parse_engine_override("*=not_a_real_engine_xyz")  # unknown engine


def test_apply_engine_override_rewrites_by_role(monkeypatch):
    led = _two_shot_ledger()
    led["video"]["shots"][0]["role"] = "character_video"
    monkeypatch.setenv("OTR_FORCE_ENGINE_MAP", "character_video=ltx_video")
    out = rd.apply_engine_override(led)
    assert out["video"]["shots"][0]["engine_id"] == "ltx_video"
    assert out["video"]["shots"][0]["family"] == "text_to_video"
    # the other role is untouched
    assert out["video"]["shots"][1]["engine_id"] == "stub_fail"


def test_apply_engine_override_star_and_noenv(monkeypatch):
    led = _two_shot_ledger()
    monkeypatch.setenv("OTR_FORCE_ENGINE_MAP", "*=still_kenburns")
    out = rd.apply_engine_override(led)
    assert all(s["engine_id"] == "still_kenburns"
               for s in out["video"]["shots"])
    monkeypatch.delenv("OTR_FORCE_ENGINE_MAP", raising=False)
    led2 = _two_shot_ledger()
    out2 = rd.apply_engine_override(led2)
    assert out2 is led2                       # no env -> no rewrite, same obj


def test_apply_engine_override_bad_spec_failsafe(monkeypatch):
    led = _two_shot_ledger()
    monkeypatch.setenv("OTR_FORCE_ENGINE_MAP", "garbage-without-equals")
    out = rd.apply_engine_override(led)       # LOUD log; plan untouched
    assert out["video"]["shots"][0]["engine_id"] == "stub_ok"


class _StubLipsync(_StubBase):
    name = "stub_lipsync"
    family = "lipsync_overlay"

    def render_clip(self, request, prepared):
        assert request.get("base_clip_ref"), "base must be provided first"
        return {"raw": True, "base": request["base_clip_ref"]["path"]}


class _StubBaseProvider(_StubBase):
    name = "stub_baseprov"
    family = "text_to_video"

    def render_clip(self, request, prepared):
        assert request.get("audio_ref") is None     # the base is SILENT (V-1)
        return {"raw": True}

    def canonicalize(self, raw, request, profile=None):
        return {"clip_id": request["shot_id"], "engine_id": self.name,
                "family": self.family, "frame_count": 25,
                "path": "X:/fake/base_clip.mp4"}


@pytest.fixture
def lipsync_registry():
    saved = dict(vreg._VIDEO_REGISTRY._registry)
    vreg.register(_StubLipsync())
    vreg.register(_StubBaseProvider())
    try:
        yield vreg._VIDEO_REGISTRY
    finally:
        vreg._VIDEO_REGISTRY._registry.clear()
        vreg._VIDEO_REGISTRY._registry.update(saved)


def test_provide_lipsync_base_renders_and_stamps(monkeypatch, lipsync_registry):
    monkeypatch.setenv("OTR_LSYNC_BASE_ENGINE", "stub_baseprov")
    monkeypatch.setattr(rd, "ENGINE_FAMILY",
                        dict(rd.ENGINE_FAMILY, stub_lipsync="lipsync_overlay"))
    req = {"shot_id": "shot_x", "base_clip_ref": None,
           "audio_ref": {"path": "a.wav"}, "text_prompt": "scene",
           "timing": {"target_frame_count": 25},
           "canvas": {"w": 64, "h": 64, "fps": 25, "aspect_policy": "pad"},
           "seed_bundle": {"request_seed": 1}}
    rd._provide_lipsync_base("stub_lipsync", req)
    assert req["base_clip_ref"] == {"path": "X:/fake/base_clip.mp4"}
    # the ORIGINAL request's audio_ref is untouched (only the base is silent)
    assert req["audio_ref"] == {"path": "a.wav"}


def test_provide_lipsync_base_noenv_noop(monkeypatch, lipsync_registry):
    monkeypatch.delenv("OTR_LSYNC_BASE_ENGINE", raising=False)
    monkeypatch.setattr(rd, "ENGINE_FAMILY",
                        dict(rd.ENGINE_FAMILY, stub_lipsync="lipsync_overlay"))
    req = {"shot_id": "shot_x", "base_clip_ref": None}
    rd._provide_lipsync_base("stub_lipsync", req)
    assert req["base_clip_ref"] is None


def test_provide_lipsync_base_failure_is_loud_not_fatal(monkeypatch,
                                                        lipsync_registry):
    monkeypatch.setenv("OTR_LSYNC_BASE_ENGINE", "stub_fail_base_xyz")
    monkeypatch.setattr(rd, "ENGINE_FAMILY",
                        dict(rd.ENGINE_FAMILY, stub_lipsync="lipsync_overlay"))
    req = {"shot_id": "shot_x", "base_clip_ref": None}
    rd._provide_lipsync_base("stub_lipsync", req)   # unknown engine -> LOUD
    assert req["base_clip_ref"] is None


# --------------------------------------------------------------------------- #
# W7-pre builder migration (3D plan 7.0): the builders emit SCHEMA-VALID
# VideoRequest dicts; the chain re-validates family inputs per candidate; the
# AS-2 resolver-prune fires on a family-changing fallback.
# --------------------------------------------------------------------------- #
def _char3d_ledger():
    """A ShotLock-shaped ledger whose single shot is the v1 character_3d
    engine (triposg_talk) with a resolvable portrait + per-line voice wav."""
    return {
        "audio": {"master_audio_sha256": rd.FROZEN_AUDIO_SHA,
                  "ledger_frozen": True},
        "lines": [
            {"line_id": "b010", "char_id": "c07", "speaker_role": "character",
             "start_s": 4.0, "dur_s": 2.4, "bark_wav_path": "X:/a/b010.wav"},
        ],
        "images": {"images": [
            {"object_id": "c07", "path": "X:/img/c07.png"},
        ]},
        "video": {"video_revision": 1, "fps": 25, "shots": [
            {"shot_id": "shot_b010", "source_line_ids": ["b010"],
             "role": "character_video", "engine_id": "triposg_talk",
             "family": "character_3d", "group_id": "grp_char",
             "target_frame_count": 60, "degradation_trail": [],
             "creative": {"text_prompt": "c07 speaks, lantern glow"}},
        ]},
    }


def test_built_character_3d_request_is_schema_valid():
    """The 7.0 code-verified gap, closed: VideoRequest.model_validate accepts
    the builder output verbatim (no init_w/init_h, no timing.dur_s, no
    top-level char_id; role/family_hint/profile_id emitted; observability is
    the schema-real field)."""
    from nodes._otr_video_engines.schemas import VideoRequest
    led = _char3d_ledger()
    req = rd.build_request_from_shot(led["video"]["shots"][0], led)
    model = VideoRequest.model_validate(req)
    assert model.family_hint == "character_3d"
    assert model.role == "character_video"
    assert model.conditioning_refs["char_id"] == "c07"
    assert model.timing.target_duration_s == 2.4
    assert model.timing.source_line_ids == ["b010"]
    assert "init_w" not in req and "init_h" not in req
    # the soak/global-assets builder is schema-valid too
    soak_req = rd.build_request(
        {"shot_id": "shot_0003", "role": "scene_broll",
         "engine_id": "still_kenburns", "family": "static_motion"},
        {"init_image": "p.png", "audio_ref": "a.wav"}, 25)
    VideoRequest.model_validate(soak_req)


def test_character_3d_request_missing_inputs_fails_closed():
    """character_3d REQUIRES audio_ref + init_image: a ledger with neither
    yields a request that model_validate REJECTS (defense in depth for the
    adapter-boundary validation, 3D plan 7.2)."""
    from nodes._otr_video_engines.schemas import VideoRequest
    led = _char3d_ledger()
    led["images"] = {"images": []}
    led["lines"][0].pop("bark_wav_path")
    req = rd.build_request_from_shot(led["video"]["shots"][0], led)
    with pytest.raises(Exception):
        VideoRequest.model_validate(req)


def test_family_input_gap_fails_loud(monkeypatch, stub_registry):
    """p3 down-chain shape: a lipsync_overlay engine whose base_clip_ref the
    request cannot supply now FAILS LOUD (no fallbacks, operator 2026-06-16) --
    the wrong-shaped request raises RenderError instead of degrading. Uses a
    stub lipsync_overlay engine (latentsync removed 2026-06-17); the family-input
    gap is the same regardless of which engine carries the family."""
    class _StubLipsync(_StubBase):
        name = "stub_lipsync"
        family = "lipsync_overlay"
        roles = ("announcer_visual", "character_video")

        def render_clip(self, request, prepared):
            return {"raw": True}

    vreg.register(_StubLipsync())
    monkeypatch.delenv("OTR_LSYNC_BASE_ENGINE", raising=False)
    shot = {"shot_id": "shot_gap", "beat_id": "bg", "role": "character_video",
            "engine_id": "stub_lipsync", "family": "lipsync_overlay",
            "group_id": "g", "target_frame_count": 25,
            "degradation_trail": []}
    req = rd.build_request(shot, {"init_image": "p.png", "audio_ref": "a.wav"},
                           25)
    with pytest.raises(rd.RenderError):
        rd.render_shot(shot, req, video_revision=1)


def test_family_changing_failure_is_loud_no_prune(stub_registry):
    """No fallbacks (operator 2026-06-16): a character_3d engine that fails RAISES
    loud -- there is no family-changing degrade, so no execution-group prune. The
    input fixture is untouched (deep-copy transaction)."""
    class _Stub3DFail(_StubBase):
        name = "stub_3d_fail"
        family = "character_3d"
        roles = ("character_video",)

        def render_clip(self, request, prepared):
            raise RuntimeError("3d forward unavailable")

    vreg.register(_Stub3DFail())
    led = rd.build_full_ledger({
        "video_revision": 1, "fps": 25,
        "execution_groups": [
            {"group_id": "grp_bg", "kind": "provider", "engine_id": "ltx_video",
             "profile_id": "", "depends_on": [],
             "produces_base_for": ["grp_char"]},
            {"group_id": "grp_char", "kind": "consumer",
             "engine_id": "stub_3d_fail", "profile_id": "",
             "depends_on": ["grp_bg"], "produces_base_for": []},
        ],
        "shots": [
            {"shot_id": "shot_3d", "beat_id": "b3d", "role": "character_video",
             "engine_id": "stub_3d_fail", "family": "character_3d",
             "group_id": "grp_char", "target_frame_count": 25,
             "degradation_trail": []},
        ]})
    with pytest.raises(rd.RenderError):
        rd.run_episode(
            led, fallback_of={"stub_3d_fail": "stub_ok"}.get,
            assets={"init_image": "p.png", "audio_ref": "a.wav"})
    # the input fixture is untouched (deep-copy transaction)
    assert [g["group_id"] for g in led["video"]["execution_groups"]] \
        == ["grp_bg", "grp_char"]
    assert led["audio"]["master_audio_sha256"] == rd.FROZEN_AUDIO_SHA


def test_engine_failure_raises_loud(stub_registry):
    """A failing engine RAISES (no within-family fallback either, operator
    2026-06-16) -- the episode fails loud rather than swapping stub_fail ->
    stub_ok, and the execution groups are never touched."""
    led = rd.build_full_ledger({
        "video_revision": 1, "fps": 25,
        "execution_groups": [
            {"group_id": "g0", "kind": "consumer", "engine_id": "stub_fail",
             "profile_id": "", "depends_on": [], "produces_base_for": []},
        ],
        "shots": [
            {"shot_id": "shot_0000", "beat_id": "b0",
             "role": "background_abstract", "engine_id": "stub_fail",
             "family": "abstract", "group_id": "g0",
             "target_frame_count": 25, "degradation_trail": []},
        ]})
    with pytest.raises(rd.RenderError):
        rd.run_episode(led, fallback_of={"stub_fail": "stub_ok"}.get)


def test_provide_lipsync_base_non_overlay_noop(monkeypatch):
    monkeypatch.setenv("OTR_LSYNC_BASE_ENGINE", "stub_baseprov")
    req = {"shot_id": "shot_x", "base_clip_ref": None}
    rd._provide_lipsync_base("stub_ok", req)        # abstract family -> no-op
    assert req["base_clip_ref"] is None
