"""Additive CPU coverage for the in-process render driver (A-S7.5).

New, self-contained tests complementing tests/test_video_render_driver.py. The
pure helpers (engine_family / build_full_ledger / build_soak_fixture range guard
/ classify_failure kind map) plus a CPU exercise of
the REAL render loop (run_episode) driven by STUB engines registered into a
snapshot/restore registry: it proves a HARD failure RAISES LOUD (NO FALLBACKS --
2026-07-02 directive; no engine swap, no still floor), the frozen audio section is
never touched, the input ledger is not mutated, and a clean run is deterministic.
No GPU, no model load. UTF-8, no BOM, ASCII-only, SFW.
"""
from __future__ import annotations

import pathlib

import pytest

from nodes._otr_shared import retry_taxonomy as rt
from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import render_driver as rd


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def test_engine_family_known_and_unknown():
    assert rd.engine_family("humo") == "audio_driven_face"
    assert rd.engine_family("still_motion") == "static_motion"
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
    # Rebased 2026-08-23: stub identity is (soak_oom_heavy, audio_driven_face)
    # so the forced-OOM contract outlives the character_3d retirement.
    assert (oom["engine_id"], oom["family"]) == (
        "soak_oom_heavy", "audio_driven_face")
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


# --------------------------------------------------------------------------- #
# CPU exercise of the real render loop via STUB engines
# --------------------------------------------------------------------------- #
class _StubBase:
    family = "abstract"
    roles = ("retired_role_b",)
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
        {"shot_id": "shot_0000", "beat_id": "b0", "role": "retired_role_b",
         "engine_id": "stub_ok", "family": "abstract", "group_id": "g0",
         "target_frame_count": 25, "degradation_trail": []},
        {"shot_id": "shot_0001", "beat_id": "b1", "role": "retired_role_b",
         "engine_id": "stub_fail", "family": "abstract", "group_id": "g1",
         "target_frame_count": 25, "degradation_trail": []},
    ]}
    return rd.build_full_ledger(section)


def test_run_episode_fails_loud_no_fallback(stub_registry):
    # NO FALLBACKS (2026-07-02 directive): the failing shot RAISES RenderError;
    # there is NO swap to stub_ok and NO still floor. The frozen input audio is
    # untouched (deep-copy transaction).
    ledger = _two_shot_ledger()
    with pytest.raises(rd.RenderError):
        rd.run_episode(ledger)
    assert ledger["audio"]["master_audio_sha256"] == rd.FROZEN_AUDIO_SHA


def test_run_episode_does_not_mutate_input_ledger(stub_registry):
    ledger = _two_shot_ledger()
    with pytest.raises(rd.RenderError):
        rd.run_episode(ledger)
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
    a = rd.run_episode(_ok())
    b = rd.run_episode(_ok())
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


def _still_pan_opener_ledger():
    """A ShotLock-shaped ledger whose OPENER beat (b000_music_open, char_id="")
    picks still_pan for the music slot, with the ST-3 scene-still write-back
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
             "engine_id": "still_pan", "family": "static_image_gen",
             "group_id": "grp_music", "target_frame_count": 238,
             "degradation_trail": [], "render_request_hash": "hash_b000",
             "cache_keys": {"request_hash": "hash_b000"},
             "creative": {"text_prompt": "radio console, warm tungsten glow"}},
        ]},
    }


def test_build_request_from_shot_still_pan_opener_conditions_on_scene_still():
    # FIX1 / BUG-LOCAL-403: still_pan (family static_image_gen) must condition
    # the opener on its beat scene still -- without it the cheap family draws a
    # black floor (the all-black opener centre). beat_id b000_music_open keys
    # both _beat_id_for_shot and the scene_open still row.
    led = _still_pan_opener_ledger()
    shot = led["video"]["shots"][0]
    req = rd.build_request_from_shot(shot, led)
    assert req["asset_refs"]["init_image"] == "X:/img/still_b000_music_open.png"
    assert req["observability"]["init_source"] == "scene_still"


def test_build_request_from_shot_seedance_opener_conditions_on_scene_still():
    # Seedance is audio_conditioned_video, but its concrete adapter requires an
    # init_image. The opener must therefore join the same minted wide scene
    # still as the i2v/still lanes, never reach the cloud adapter empty-handed.
    led = _still_pan_opener_ledger()
    shot = dict(led["video"]["shots"][0])
    shot["engine_id"] = "cloud_seedance_2"
    shot["family"] = "audio_conditioned_video"
    req = rd.build_request_from_shot(shot, led)
    assert req["asset_refs"]["init_image"] == "X:/img/still_b000_music_open.png"
    assert req["observability"]["init_source"] == "scene_still"


def test_seedance_preflight_requires_engine_init_image():
    # The family minimum for audio_conditioned_video is text+audio so LTX can
    # stay flexible, but Seedance's concrete engine contract is stricter.
    req = rd.build_request(
        {"shot_id": "shot_b000_music_open", "role": "music_visual",
         "engine_id": "cloud_seedance_2"},
        {"audio_ref": "X:/audio/music.wav"},
        25,
        canvas=(832, 480),
    )
    with pytest.raises(rd.FamilyInputGap) as excinfo:
        rd._assert_family_inputs_satisfiable("cloud_seedance_2", req)
    assert "init_image" in str(excinfo.value)


def test_build_request_from_shot_still_pan_missing_still_is_loud_not_black(caplog):
    # No scene still for the opener -> init stays empty, but the degrade is LOUD
    # (the branch warns), never a silent scene_still claim. init_source falls
    # back to "none".
    #
    # "(the cheap family will draw its floor)" was in that first line and is no
    # longer true for THIS engine (lane 16, 2026-08-11): still_pan now sets
    # _require_still, so the downstream render_clip RAISES instead of painting a
    # floor. The subject here is one layer up -- what build_request_from_shot
    # puts on the request -- so the assertions are unchanged and still correct;
    # only the aside about what happens next was wrong.
    led = _still_pan_opener_ledger()
    led["images"]["images"] = [r for r in led["images"]["images"]
                               if r["object_id"] != "still_b000_music_open"]
    shot = led["video"]["shots"][0]
    caplog.set_level("INFO")
    planned = rd.build_request_from_shot(shot, led, phase="cast_preflight")
    assert any("STILL DEFERRED" in r.message and r.levelname == "INFO"
               for r in caplog.records)
    assert not any("MISSING-STILL" in r.message for r in caplog.records)
    caplog.clear()
    req = rd.build_request_from_shot(shot, led)
    assert req == planned, "diagnostic phase must not change the render request"
    assert any("MISSING-STILL" in r.message and r.levelname == "WARNING"
               for r in caplog.records)
    assert req["asset_refs"] == {}                       # no still -> empty
    assert req["observability"]["init_source"] == "none"


def test_build_request_from_shot_still_pan_fills_landscape_canvas():
    # 2026-06-14 operator catch: still_pan (and the other still/floor families)
    # inherited build_request's 480x832 HuMo PORTRAIT default -> a skinny portrait
    # pillarboxed in the 16:9 frame. Non-face engines must FILL the landscape
    # canvas; only audio_driven_face (HuMo) keeps its portrait pillarbox.
    led = _still_pan_opener_ledger()
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


def test_voice_audio_resolver_engine_fields_and_music_fallback():
    assert rd._voice_audio_for_line({"audio_wav_path": "a.wav"}) == "a.wav"
    assert rd._voice_audio_for_line({"indextts2_wav_path": "i.wav"}) == "i.wav"
    assert rd._voice_audio_for_line({"bark_wav_path": "b.wav"}) == "b.wav"
    # a music beat's clip is the music fallback, never a face-driving voice
    # in the generic *_wav_path sweep (the sfx_wav_path variant died with the
    # sfx subsystem, rip-sfx-broll 2026-07-01)
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

    rd.run_episode(led, request_builder=rb)
    assert calls == ["shot_b001", "shot_b002"]   # the builder drove every shot


def test_run_episode_ignores_render_plan_for_visible_shots(record_registry):
    led = _real_ledger()
    led["lines"].append(
        {"line_id": "b003", "char_id": "c03", "speaker_role": "character",
         "start_s": 3.6, "dur_s": 1.2, "bark_wav_path": "X:/a/seg_b003.wav"}
    )
    led["video"]["shots"].append(
        {"shot_id": "shot_b003", "source_line_ids": ["b003"],
         "engine_id": "stub_record", "family": "", "group_id": "grp_d",
         "target_frame_count": 30, "degradation_trail": [],
         "render_request_hash": "hash_b003",
         "cache_keys": {"request_hash": "hash_b003"},
         "creative": {"text_prompt": "c03 listens to the static"}}
    )
    led["meta"] = {"render_plan": {
        "line_ids": ["b002"],
        "selection_mode": "all",
        "blocked": True,
        "applied_max_n": 1,
        "excluded_flat_lines": ["b003"],
    }}
    calls = []

    def rb(shot, ledger, *, canvas=None):
        calls.append(shot["shot_id"])
        return rd.build_request_from_shot(shot, ledger, canvas=canvas)

    out = rd.run_episode(led, request_builder=rb)
    assert calls == ["shot_b001", "shot_b002", "shot_b003"]
    assert [s["shot_id"] for s in out["ledger"]["video"]["shots"]] == [
        "shot_b001", "shot_b002", "shot_b003"]
    assert sorted(out["clips"]) == ["shot_b001", "shot_b002", "shot_b003"]


def test_run_episode_legacy_render_plan_log_tolerates_bad_line_ids(record_registry):
    led = _real_ledger()
    led["meta"] = {"render_plan": {
        "line_ids": 3,
        "selection_mode": "all",
        "blocked": False,
        "applied_max_n": 0,
    }}

    out = rd.run_episode(led, request_builder=rd.build_request_from_shot)
    assert sorted(out["clips"]) == ["shot_b001", "shot_b002"]


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
    assert m["timeline_total_frames"] == 80
    assert m["render_target_frames"] == 80
    assert m["timeline_frame_source"] == "render_target_sum"
    assert m["canvas"] == {"w": 1472, "h": 832} and m["fps"] == 25
    assert m["clip_count"] == 1                  # only the on-disk clip counts
    assert m["engine_histogram"] == {"humo": 1}  # empty-path clip excluded
    assert m["clips"][0]["exists"] is True and m["clips"][1]["exists"] is False


def test_build_clip_manifest_counts_budget_floor_as_sanctioned_gap():
    from nodes._otr_shared import still_receipt as _receipt
    result = {
        "ledger": {"video": {
            "video_revision": 1, "fps": 25,
            "canonical_canvas": {"w": 1472, "h": 832},
            "shots": [
                {"shot_id": "shot_b001", "beat_id": "b001",
                 "engine_id": "cloud_ltx25_foley_plus",
                 "target_frame_count": 50},
                rd._stamp_budget_floor_shot({
                    "shot_id": "shot_b002", "beat_id": "b002",
                    "engine_id": "cloud_ltx25_foley_plus",
                    "target_frame_count": 50,
                }),
            ]}},
        "clips": {
            "shot_b001": {"engine_id": "cloud_ltx25_foley_plus",
                          "frame_count": 50, "path": __file__},
        },
    }
    m = rd.build_clip_manifest(result, episode_id="ep_budget")
    assert m["clips"][0]["exists"] is True
    assert m["clips"][0]["status"] == _receipt.STATUS_OK
    assert m["clips"][1]["exists"] is False
    assert m["clips"][1]["status"] == _receipt.STATUS_SANCTIONED_GAP
    assert _receipt.is_sanctioned_gap(m["clips"][1])
    _sanctioned = sum(1 for c in m["clips"] if _receipt.is_sanctioned_gap(c))
    _delivered_n = sum(1 for c in m["clips"] if (c or {}).get("exists"))
    _unaccounted = len(m["clips"]) - _delivered_n - _sanctioned
    assert _unaccounted == 0
    assert m["clip_count"] == 1


def test_build_clip_manifest_positioned_timeline_uses_ledger_boundary():
    result = {
        "ledger": {
            "total_episode_dur_s": 3.1,
            "lines": [
                {"line_id": "music_opening_001", "start_s": 0.0},
                {"line_id": "l001", "start_s": 1.9},
            ],
            "video": {
                "video_revision": 1, "fps": 25,
                "canonical_canvas": {"w": 1472, "h": 832},
                "shots": [
                    {"shot_id": "shot_music_opening_001",
                     "source_line_ids": ["music_opening_001"],
                     "engine_id": "abstract", "target_frame_count": 50},
                    {"shot_id": "shot_l001", "source_line_ids": ["l001"],
                     "engine_id": "humo", "target_frame_count": 30},
                ],
            },
        },
        "clips": {
            "shot_music_opening_001": {
                "engine_id": "abstract", "frame_count": 50, "path": __file__},
            "shot_l001": {
                "engine_id": "humo", "frame_count": 30, "path": __file__},
        },
    }

    manifest = rd.build_clip_manifest(result, episode_id="positioned")

    # The two full render requests are 80 frames of work, but the accepted
    # positioned episode boundary is ceil(3.1 * 25) == 78 CFR frames.
    assert manifest["render_target_frames"] == 80
    assert manifest["total_target_frames"] == 78
    assert manifest["timeline_total_frames"] == 78
    assert manifest["timeline_frame_source"] == "ledger.total_episode_dur_s"
    assert manifest["timeline_duration_s"] == 3.1


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
    m = rd.parse_engine_override("*=ltx25_video")
    assert m == {"*": "ltx25_video"}
    m = rd.parse_engine_override(
        "character_video=humo, retired_role_a=ltx25_video")
    assert m == {"character_video": "humo", "retired_role_a": "ltx25_video"}
    with pytest.raises(ValueError):
        rd.parse_engine_override("character_video")          # no '='
    with pytest.raises(ValueError):
        rd.parse_engine_override("*=not_a_real_engine_xyz")  # unknown engine


def test_apply_engine_override_rewrites_by_role(monkeypatch):
    led = _two_shot_ledger()
    led["video"]["shots"][0]["role"] = "character_video"
    monkeypatch.setenv("OTR_FORCE_ENGINE_MAP", "character_video=ltx25_video")
    out = rd.apply_engine_override(led)
    assert out["video"]["shots"][0]["engine_id"] == "ltx25_video"
    # The family override stamps what the engine class declares.
    assert out["video"]["shots"][0]["family"] == "image_to_video"
    # the other role is untouched
    assert out["video"]["shots"][1]["engine_id"] == "stub_fail"


def test_apply_engine_override_star_and_noenv(monkeypatch):
    led = _two_shot_ledger()
    monkeypatch.setenv("OTR_FORCE_ENGINE_MAP", "*=still_motion")
    out = rd.apply_engine_override(led)
    assert all(s["engine_id"] == "still_motion"
               for s in out["video"]["shots"])
    monkeypatch.delenv("OTR_FORCE_ENGINE_MAP", raising=False)
    led2 = _two_shot_ledger()
    out2 = rd.apply_engine_override(led2)
    assert out2 is led2                       # no env -> no rewrite, same obj


def test_apply_engine_override_bad_spec_fails_closed(monkeypatch):
    """A malformed force map is TERMINAL (2026-07-25 route lock).

    This inverts the old ``_failsafe`` contract deliberately. Logging
    ``IGNORED (parse)`` and rendering the UNFORCED plan is a silent fallback:
    the operator asks for a forced-route episode and gets an ordinary one,
    with only a warning line to say so -- and the still spine is then
    validated against routing nobody asked for.
    """
    led = _two_shot_ledger()
    monkeypatch.setenv("OTR_FORCE_ENGINE_MAP", "garbage-without-equals")
    with pytest.raises(rd.RenderError) as exc:
        rd.apply_engine_override(led)
    assert "OTR_FORCE_ENGINE_MAP is malformed" in str(exc.value)
    # the plan was NOT quietly rewritten on the way out
    assert led["video"]["shots"][0]["engine_id"] == "stub_ok"


def test_resolve_final_shot_engines_runs_both_mutations(monkeypatch):
    """The route lock resolves force map AND radio-host redirect in one pass."""
    led = _two_shot_ledger()
    led["video"]["shots"][0]["role"] = "character_video"
    monkeypatch.setenv("OTR_FORCE_ENGINE_MAP", "character_video=ltx25_video")
    monkeypatch.delenv("OTR_ENABLE_HUMO_HOSTS", raising=False)
    out = rd.resolve_final_shot_engines(led)
    assert out["video"]["shots"][0]["engine_id"] == "ltx25_video"


def test_resolve_final_shot_engines_redirects_humo_bookend(monkeypatch):
    """An announcer_visual beat picked onto local HuMo is redirected BEFORE
    the caller validates the still spine -- the ordering defect this pass
    exists to close (kibitz per-beat-stills r1, both seats)."""
    led = _two_shot_ledger()
    shot = led["video"]["shots"][0]
    shot["role"] = "announcer_visual"
    shot["engine_id"] = "humo"
    shot["family"] = "audio_driven_face"
    monkeypatch.delenv("OTR_FORCE_ENGINE_MAP", raising=False)
    monkeypatch.delenv("OTR_ENABLE_HUMO_HOSTS", raising=False)
    rd.resolve_final_shot_engines(led)
    assert shot["engine_id"] == rd._NEVER_HUMO_REDIRECT_ENGINE
    # and it is IDEMPOTENT -- a second pass must not re-redirect or throw
    rd.resolve_final_shot_engines(led)
    assert shot["engine_id"] == rd._NEVER_HUMO_REDIRECT_ENGINE


def test_resolve_final_shot_engines_humo_hosts_on_keeps_portrait(monkeypatch):
    """With OTR_ENABLE_HUMO_HOSTS=1 the redirect is a no-op, so a portrait
    HuMo bookend keeps its portrait engine through the lock."""
    led = _two_shot_ledger()
    shot = led["video"]["shots"][0]
    shot["role"] = "announcer_visual"
    shot["engine_id"] = "humo"
    shot["family"] = "audio_driven_face"
    monkeypatch.delenv("OTR_FORCE_ENGINE_MAP", raising=False)
    monkeypatch.setenv("OTR_ENABLE_HUMO_HOSTS", "1")
    rd.resolve_final_shot_engines(led)
    assert shot["engine_id"] == "humo"


# --------------------------------------------------------------------------- #
# W7-pre builder migration (3D plan 7.0): the builders emit SCHEMA-VALID
# VideoRequest dicts; the chain re-validates family inputs per candidate; the
# AS-2 resolver-prune fires on a family-changing fallback.
# --------------------------------------------------------------------------- #
def _audio_face_ledger():
    """A ShotLock-shaped ledger whose single shot is a LIVE audio_driven_face
    engine (humo) with a resolvable portrait + per-line voice wav. (Was the
    character_3d/triposg_talk fixture until that family was retired 2026-08-23,
    lean-mean order 4 -- audio_driven_face requires the SAME audio_ref +
    init_image pair, so every assertion keeps its meaning on a family that can
    actually render.)"""
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
             "role": "character_video", "engine_id": "humo",
             "family": "audio_driven_face", "group_id": "grp_char",
             "target_frame_count": 60, "degradation_trail": [],
             "creative": {"text_prompt": "c07 speaks, lantern glow"}},
        ]},
    }


def test_built_audio_face_request_is_schema_valid():
    """The 7.0 code-verified gap, closed: VideoRequest.model_validate accepts
    the builder output verbatim (no init_w/init_h, no timing.dur_s, no
    top-level char_id; role/family_hint/profile_id emitted; observability is
    the schema-real field)."""
    from nodes._otr_video_engines.schemas import VideoRequest
    led = _audio_face_ledger()
    req = rd.build_request_from_shot(led["video"]["shots"][0], led)
    model = VideoRequest.model_validate(req)
    assert model.family_hint == "audio_driven_face"
    assert model.role == "character_video"
    assert model.conditioning_refs["char_id"] == "c07"
    assert model.timing.target_duration_s == 2.4
    assert model.timing.source_line_ids == ["b010"]
    assert "init_w" not in req and "init_h" not in req
    # the soak/global-assets builder is schema-valid too
    soak_req = rd.build_request(
        {"shot_id": "shot_0003", "role": "retired_role_a",
         "engine_id": "still_motion", "family": "static_motion"},
        {"init_image": "p.png", "audio_ref": "a.wav"}, 25)
    VideoRequest.model_validate(soak_req)


def test_audio_face_request_missing_inputs_fails_closed():
    """audio_driven_face REQUIRES audio_ref + init_image: a ledger with neither
    yields a request that model_validate REJECTS (defense in depth for the
    adapter-boundary validation)."""
    from nodes._otr_video_engines.schemas import VideoRequest
    led = _audio_face_ledger()
    led["images"] = {"images": []}
    led["lines"][0].pop("bark_wav_path")
    req = rd.build_request_from_shot(led["video"]["shots"][0], led)
    with pytest.raises(Exception):
        VideoRequest.model_validate(req)


def test_google_veo_request_prunes_audio_but_keeps_scene_still():
    """Direct Google Veo cannot consume OTR audio refs, but it can use Veo's
    documented start-image path. The shared builder must keep the per-beat
    scene still and prune only unsupported audio/video refs."""
    from nodes._otr_video_engines.schemas import VideoRequest

    led = {
        "audio": {"master_audio_sha256": rd.FROZEN_AUDIO_SHA,
                  "ledger_frozen": True},
        "lines": [
            {"line_id": "b010", "char_id": "c07", "speaker_role": "character",
             "start_s": 4.0, "dur_s": 2.4, "audio_wav_path": "X:/a/b010.wav"},
        ],
        "images": {"images": [
            {"object_id": "c07", "kind": "portrait", "path": "X:/img/c07.png"},
            {"object_id": "still_b010", "kind": "scene_character",
             "beat_id": "b010", "path": "X:/img/still_b010.png"},
        ]},
        "video": {"video_revision": 1, "fps": 25, "shots": [
            {"shot_id": "shot_b010", "source_line_ids": ["b010"],
             "role": "character_video", "engine_id": "google_veo_video",
             "family": "text_to_video", "group_id": "grp_char",
             "target_frame_count": 60, "degradation_trail": [],
             "creative": {"text_prompt": "archival reading room, slow pan"}},
        ]},
    }
    req = rd.build_request_from_shot(led["video"]["shots"][0], led)
    VideoRequest.model_validate(req)
    assert req["family_hint"] == "text_to_video"
    assert req["text_prompt"]
    assert req["audio_ref"] is None
    assert req["asset_refs"]["init_image"] == "X:/img/still_b010.png"
    assert req["observability"]["init_source"] == "scene_still"


def test_google_veo_video_art_music_open_uses_scene_still_and_style_prompt():
    """A Google all-cloud music/open beat must not fall back to the generic
    1940s-studio host prompt. It should carry the generated scene still plus the
    selected video_art motion/style language."""
    led = _still_pan_opener_ledger()
    led["meta"] = {
        "visual_style": "video_art",
        "style": "media archive mystery",
        "story_brief_terms": {
            "setting": ["film archive vault", "catalog desk"],
            "atmosphere": ["contemplative signal mystery"],
        },
    }
    shot = dict(led["video"]["shots"][0])
    shot["engine_id"] = "google_veo_video"
    shot["family"] = "text_to_video"
    shot["creative"] = {}
    req = rd.build_request_from_shot(shot, led)
    prompt = req["text_prompt"].lower()
    assert req["audio_ref"] is None
    assert req["asset_refs"]["init_image"] == "X:/img/still_b000_music_open.png"
    assert "a 1940s radio studio" not in prompt
    assert "phosphor" in prompt
    assert "feedback" in prompt or "video-art" in prompt or "video art" in prompt
    assert req["observability"]["visual_style"] == "video_art"
    assert req["observability"]["prompt_source"] == "motion_role"


def test_google_video_is_provider_side_not_local_for_reclaim():
    section = {"shots": [
        {"shot_id": "shot_google", "engine_id": "google_veo_video"},
    ]}
    assert rd._section_has_local_video_engine(section) is False


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
            {"group_id": "grp_bg", "kind": "provider", "engine_id": "ltx25_video",
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
            led, assets={"init_image": "p.png", "audio_ref": "a.wav"})
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
             "role": "retired_role_b", "engine_id": "stub_fail",
             "family": "abstract", "group_id": "g0",
             "target_frame_count": 25, "degradation_trail": []},
        ]})
    with pytest.raises(rd.RenderError):
        rd.run_episode(led)


def test_classify_failure_names_a_policy_verdict_not_a_crash():
    """CRASH_BEFORE_LOAD was the old answer and it was wrong twice over."""
    from nodes._otr_shared import cloud_media_backend as cmb
    refused = cmb.CloudMediaError(
        cmb.CloudErrorCode.CONTENT_REFUSED,
        "cloud_ltx25_i2v: Content filtered due to policy restrictions")
    assert rd.classify_failure(refused) is rt.FailureKind.CONTENT_REFUSED
    wrap = RuntimeError("shot shot_shot_001_b40 failed to render")
    wrap.__cause__ = refused
    assert rd.classify_failure(wrap) is rt.FailureKind.CONTENT_REFUSED
    # Everything else keeps the answer it had.
    assert rd.classify_failure(RuntimeError("boom")) is (
        rt.FailureKind.CRASH_BEFORE_LOAD)
    assert rd.classify_failure(FileNotFoundError("nope")) is (
        rt.FailureKind.DEPENDENCY_MISSING)


def test_cloud_floor_reason_sees_through_the_render_error_wrap():
    """render_shot wraps every failure, so the cause walk is the contract."""
    from nodes._otr_shared import cloud_media_backend as cmb
    refused = cmb.CloudMediaError(
        cmb.CloudErrorCode.CONTENT_REFUSED, "Content filtered")
    wrap = RuntimeError("fallbacks are disabled")
    wrap.__cause__ = refused
    cloud = {"shot_id": "shot_b040", "engine_id": "cloud_ltx25_foley_plus"}
    assert rd._cloud_floor_reason(
        "shot_b040", {"shot_b040": wrap}, cloud) == "content_refused"
    # A beat with no error, and a beat with an ordinary one, still raise.
    assert rd._cloud_floor_reason("shot_b040", {}, cloud) == ""
    assert rd._cloud_floor_reason(
        "shot_b040", {"shot_b040": RuntimeError("engine exploded")},
        cloud) == ""


def test_content_floor_is_gated_to_cloud_engines():
    """Only a PROVIDER issues a policy verdict; a local fault fails loud.

    The predicate sits on the general except around render_beat_coverage --
    the same path a local torch or ffmpeg failure takes -- so without this
    gate a coincidental phrase could floor a beat the card could render.
    """
    from nodes._otr_shared import cloud_media_backend as cmb
    refused = cmb.CloudMediaError(
        cmb.CloudErrorCode.CONTENT_REFUSED, "Content filtered")
    errors = {"shot_b040": refused}
    assert rd._cloud_floor_reason(
        "shot_b040", errors,
        {"engine_id": "cloud_ltx25_foley_plus"}) == "content_refused"
    assert rd._cloud_floor_reason(
        "shot_b040", errors, {"engine_id": "humo"}) == ""
    assert rd._cloud_floor_reason(
        "shot_b040", errors, {"engine_id": "ltx25_video"}) == ""


def test_report_cloud_floors_escalates_when_failures_look_systemic(caplog):
    """One failed job is a job; a quarter of the show is a provider problem."""
    import logging
    floored = [rd._stamp_cloud_floor_shot({"shot_id": "s%d" % i},
                                          "content_refused")
               for i in range(4)]
    clean = [{"shot_id": "ok%d" % i} for i in range(4)]
    with caplog.at_level(logging.WARNING):
        assert rd._report_cloud_floors(clean + floored[:1]) == ["s0"]
    assert "publishes DEGRADED" in caplog.text
    assert "SYSTEMIC" not in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        rd._report_cloud_floors(clean + floored)
    assert "SYSTEMIC CLOUD FAILURE" in caplog.text
    # It reports; it never raises. Aborting here would reinstate the defect.
    assert rd._report_cloud_floors([]) == []


def test_report_cloud_floors_covers_every_reason_not_just_refusals(caplog):
    """A provider OUTAGE must not be quieter than one refused prompt.

    An earlier draft counted only ``content_floor``, so a whole episode
    floored on ``timeout`` got per-beat lines and no aggregate banner --
    backwards, since the wider the failure the louder it should be.
    """
    import logging
    from nodes._otr_shared import cloud_media_backend as cmb
    outage = [rd._stamp_cloud_floor_shot({"shot_id": "s%d" % i}, "timeout")
              for i in range(4)]
    with caplog.at_level(logging.WARNING):
        got = rd._report_cloud_floors(outage + [{"shot_id": "ok0"}])
    assert got == ["s0", "s1", "s2", "s3"]
    assert "SYSTEMIC CLOUD FAILURE" in caplog.text
    assert "timeout: s0, s1, s2, s3" in caplog.text
    # Every job-scoped reason reaches the tally, not just the refusal.
    for code in cmb.JOB_SCOPED_CODES:
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            rd._report_cloud_floors(
                [rd._stamp_cloud_floor_shot({"shot_id": "s0"}, code.value)])
        assert code.value in caplog.text, code


def test_build_clip_manifest_counts_content_floor_as_sanctioned_gap():
    """The refused beat keeps its place; the episode is degraded, not broken."""
    from nodes._otr_shared import still_receipt as _receipt
    result = {
        "ledger": {"video": {
            "video_revision": 1, "fps": 25,
            "canonical_canvas": {"w": 1472, "h": 832},
            "shots": [
                {"shot_id": "shot_b039", "beat_id": "b039",
                 "engine_id": "cloud_ltx25_foley_plus",
                 "target_frame_count": 50},
                rd._stamp_cloud_floor_shot({
                    "shot_id": "shot_b040", "beat_id": "b040",
                    "engine_id": "cloud_ltx25_foley_plus",
                    "target_frame_count": 50,
                }, "content_refused"),
            ]}},
        "clips": {
            "shot_b039": {"engine_id": "cloud_ltx25_foley_plus",
                          "frame_count": 50, "path": __file__},
        },
    }
    m = rd.build_clip_manifest(result, episode_id="ep_content")
    assert m["clips"][0]["status"] == _receipt.STATUS_OK
    assert m["clips"][1]["exists"] is False
    assert m["clips"][1]["status"] == _receipt.STATUS_SANCTIONED_GAP
    assert _receipt.is_sanctioned_gap(m["clips"][1])
    _sanctioned = sum(1 for c in m["clips"] if _receipt.is_sanctioned_gap(c))
    _delivered_n = sum(1 for c in m["clips"] if (c or {}).get("exists"))
    assert len(m["clips"]) - _delivered_n - _sanctioned == 0
    # The floor is its OWN stamp: a content refusal is not a spend cap.
    assert rd._shot_is_content_floor(result["ledger"]["video"]["shots"][1])
    assert not rd._shot_is_budget_floor(result["ledger"]["video"]["shots"][1])


# --------------------------------------------------------------------------- #
# A provider policy refusal floors ONE beat; the episode survives (2026-09-16)
# --------------------------------------------------------------------------- #
def _content_refusal():
    from nodes._otr_shared import cloud_media_backend as cmb
    return cmb.CloudMediaError(
        cmb.CloudErrorCode.CONTENT_REFUSED,
        "cloud_ltx25_i2v: Polling aborted due to error: Task failed: "
        '{"error": {"type": "content_filtered_error", "message": '
        '"Content filtered due to policy restrictions"}}')


class _StubCloudOK(_StubBase):
    name = "cloud_stub_ok"
    provider_side = True

    def render_clip(self, request, prepared):
        return {"raw": True}


class _StubCloudRefused(_StubBase):
    name = "cloud_stub_refused"
    provider_side = True

    def render_clip(self, request, prepared):
        raise _content_refusal()


@pytest.fixture
def cloud_stub_registry():
    saved = dict(vreg._VIDEO_REGISTRY._registry)
    vreg.register(_StubCloudOK())
    vreg.register(_StubCloudRefused())
    try:
        yield vreg._VIDEO_REGISTRY
    finally:
        vreg._VIDEO_REGISTRY._registry.clear()
        vreg._VIDEO_REGISTRY._registry.update(saved)


def _cloud_ledger(refused_index, n=3):
    shots = []
    for i in range(n):
        shots.append({
            "shot_id": "shot_%04d" % i, "beat_id": "b%d" % i,
            "role": "retired_role_b",
            "engine_id": ("cloud_stub_refused" if i == refused_index
                          else "cloud_stub_ok"),
            "family": "abstract", "group_id": "g%d" % i,
            "target_frame_count": 25, "degradation_trail": [],
        })
    return rd.build_full_ledger({"video_revision": 1, "fps": 25, "shots": shots})


def test_content_refusal_floors_one_beat_and_the_episode_survives(
        cloud_stub_registry, monkeypatch):
    """THE DEFECT, END TO END (live 2026-09-16).

    LTX 2.5 filtered beat 40 and the episode raised, discarding 39 beats that
    had already rendered and already been paid for. The refused beat must now
    keep its place as a counted gap while every other beat commits.
    """
    monkeypatch.setenv("OTR_CLOUD_FANOUT", "4")
    led = _cloud_ledger(refused_index=1)
    # Prove WHICH branch this exercises -- both paths stamp the same shot, so
    # without this the serial test and this one could be the same test twice.
    assert rd._should_fanout_cloud_episode(led["video"], set()) is True
    out = rd.run_episode(led)
    shots = out["ledger"]["video"]["shots"]
    assert [s["shot_id"] for s in shots] == [
        "shot_0000", "shot_0001", "shot_0002"]        # order is preserved
    assert rd._shot_is_content_floor(shots[1])
    assert not rd._shot_is_content_floor(shots[0])
    assert not rd._shot_is_content_floor(shots[2])
    # The beats AFTER the refusal still rendered -- that is the whole point.
    assert "shot_0000" in out["clips"] and "shot_0002" in out["clips"]
    assert "shot_0001" not in out["clips"]
    assert out["ledger"]["audio"]["master_audio_sha256"] == rd.FROZEN_AUDIO_SHA


def test_content_refusal_floors_on_the_serial_path_too(
        cloud_stub_registry, monkeypatch):
    """Fan-out off (OTR_CLOUD_FANOUT=1) walks the other branch."""
    monkeypatch.setenv("OTR_CLOUD_FANOUT", "1")
    monkeypatch.setenv("OTR_CLOUD_VIDEO_FANOUT", "1")
    led = _cloud_ledger(refused_index=0)
    assert rd._should_fanout_cloud_episode(led["video"], set()) is False
    out = rd.run_episode(led)
    shots = out["ledger"]["video"]["shots"]
    assert rd._shot_is_content_floor(shots[0])
    # A refusal is a fact about ONE prompt: it must NOT halt the queue the way
    # an empty wallet does, so the beats behind it still get their attempt.
    assert "shot_0001" in out["clips"] and "shot_0002" in out["clips"]
    assert not rd._shot_is_budget_floor(shots[0])


def test_an_ordinary_cloud_crash_still_fails_loud(cloud_stub_registry,
                                                  monkeypatch):
    """The floor must not launder a real fault into a publishable gap."""
    monkeypatch.setenv("OTR_CLOUD_FANOUT", "4")

    class _StubCloudBroken(_StubBase):
        name = "cloud_stub_broken"
        provider_side = True

        def render_clip(self, request, prepared):
            raise RuntimeError("the engine actually exploded")

    vreg.register(_StubCloudBroken())
    led = _cloud_ledger(refused_index=1)
    led["video"]["shots"][1]["engine_id"] = "cloud_stub_broken"
    with pytest.raises(rd.RenderError):
        rd.run_episode(led)


def test_cloud_floor_reason_covers_every_job_scoped_verdict():
    """Operator 2026-09-16: a failed cloud output must not break the system."""
    from nodes._otr_shared import cloud_media_backend as cmb
    cloud = {"shot_id": "s1", "engine_id": "cloud_ltx25_foley_plus"}
    for code in cmb.JOB_SCOPED_CODES:
        err = cmb.CloudMediaError(code, "no clip")
        wrap = rd.RenderError("fallbacks are disabled")
        wrap.__cause__ = err
        assert rd._cloud_floor_reason("s1", {"s1": wrap}, cloud) == code.value
        # every reason gets a sentence an operator can act on
        assert rd._cloud_floor_sentence(code.value).endswith(".")
    # Run-scoped verdicts and unstamped crashes still raise.
    for code in cmb.RUN_SCOPED_CODES | {cmb.CloudErrorCode.INTERRUPTED}:
        assert rd._cloud_floor_reason(
            "s1", {"s1": cmb.CloudMediaError(code, "x")}, cloud) == ""
    assert rd._cloud_floor_reason(
        "s1", {"s1": RuntimeError("engine exploded")}, cloud) == ""
    assert rd._cloud_floor_reason("s1", {}, cloud) == ""
    # LOCAL engines are never floored, whatever the exception says.
    assert rd._cloud_floor_reason(
        "s1", {"s1": cmb.CloudMediaError(cmb.CloudErrorCode.TIMEOUT, "x")},
        {"engine_id": "humo"}) == ""


class _StubCloudTimeout(_StubBase):
    name = "cloud_stub_timeout"
    provider_side = True

    def render_clip(self, request, prepared):
        from nodes._otr_shared import cloud_media_backend as cmb
        raise cmb.CloudMediaError(
            cmb.CloudErrorCode.TIMEOUT, "cloud_stub: no answer in 900s")


class _StubCloudAuth(_StubBase):
    name = "cloud_stub_auth"
    provider_side = True

    def render_clip(self, request, prepared):
        from nodes._otr_shared import cloud_media_backend as cmb
        raise cmb.CloudMediaError(cmb.CloudErrorCode.AUTH, "HTTP 401")


def test_a_cloud_timeout_floors_one_beat_and_the_episode_survives(
        cloud_stub_registry, monkeypatch):
    """Not just refusals: ANY job-scoped cloud verdict keeps the run alive."""
    monkeypatch.setenv("OTR_CLOUD_FANOUT", "4")
    vreg.register(_StubCloudTimeout())
    led = _cloud_ledger(refused_index=1)
    led["video"]["shots"][1]["engine_id"] = "cloud_stub_timeout"
    out = rd.run_episode(led)
    shots = out["ledger"]["video"]["shots"]
    assert rd._shot_is_cloud_floor(shots[1])
    assert shots[1]["cloud_floor"] == "timeout"
    # A timeout is NOT a content refusal, and the ledger must not say it was.
    assert not rd._shot_is_content_floor(shots[1])
    assert "shot_0000" in out["clips"] and "shot_0002" in out["clips"]


def test_a_run_scoped_cloud_verdict_still_fails_loud(cloud_stub_registry,
                                                     monkeypatch):
    """AUTH means nothing will EVER render; flooring would publish a void."""
    monkeypatch.setenv("OTR_CLOUD_FANOUT", "4")
    vreg.register(_StubCloudAuth())
    led = _cloud_ledger(refused_index=1)
    led["video"]["shots"][1]["engine_id"] = "cloud_stub_auth"
    with pytest.raises(rd.RenderError):
        rd.run_episode(led)


def test_every_floor_kind_is_counted_as_a_sanctioned_gap():
    """Budget, content and job floors all ride ONE accounting channel."""
    from nodes._otr_shared import still_receipt as _receipt
    rows = [
        rd._stamp_budget_floor_shot({"shot_id": "s0", "beat_id": "b0"}),
        rd._stamp_cloud_floor_shot({"shot_id": "s1", "beat_id": "b1"},
                                   "content_refused"),
        rd._stamp_cloud_floor_shot({"shot_id": "s2", "beat_id": "b2"},
                                   "timeout"),
    ]
    for row in rows:
        assert row["status"] == _receipt.STATUS_SANCTIONED_GAP
    # The content stamp keeps its own boolean; the others do not claim it.
    assert rd._shot_is_content_floor(rows[1])
    assert not rd._shot_is_content_floor(rows[2])
    assert rd._shot_is_cloud_floor(rows[1]) and rd._shot_is_cloud_floor(rows[2])
    assert rd._shot_is_budget_floor(rows[0])


# --------------------------------------------------------------------------- #
# The CASCADE: a floored beat must not kill the run through its dependent
# --------------------------------------------------------------------------- #
def test_floored_predecessor_is_found_for_a_chain_successor():
    assert rd._floored_predecessor(
        {"shot_id": "s2", "starts_on_last_frame_of": "s1"}, {"s1"}) == "s1"
    # A predecessor that rendered is not a floor.
    assert rd._floored_predecessor(
        {"shot_id": "s2", "starts_on_last_frame_of": "s1"}, set()) == ""
    # A jump beat has no predecessor at all.
    assert rd._floored_predecessor({"shot_id": "s2"}, {"s1"}) == ""
    assert rd._floored_predecessor({"shot_id": "s2"}, None) == ""


class _StubCloudChainRefused(_StubBase):
    name = "cloud_stub_chain_refused"
    provider_side = True

    def render_clip(self, request, prepared):
        from nodes._otr_shared import cloud_media_backend as cmb
        raise cmb.CloudMediaError(
            cmb.CloudErrorCode.CONTENT_REFUSED, "Content filtered")


def test_a_floored_beat_does_not_kill_the_run_through_its_chain_successor(
        cloud_stub_registry, monkeypatch):
    """THE CASCADE THE FIRST DRAFT MISSED.

    run_cloud_fanout only submits a shot once every predecessor is in
    finished_ok. A floored predecessor never lands there, so its CHAIN
    successor is never submitted, comes back in stuck_ids rather than errors,
    and used to sail past every floor branch into "cloud fan-out never
    rendered shot" -- which raises. The floor held for jump beats and quietly
    failed for chained ones, which is the harder half of the episode.
    """
    monkeypatch.setenv("OTR_CLOUD_FANOUT", "4")
    vreg.register(_StubCloudChainRefused())
    led = _cloud_ledger(refused_index=1, n=4)
    shots = led["video"]["shots"]
    shots[1]["engine_id"] = "cloud_stub_chain_refused"
    # shot_0002 can only start on the last frame of the beat that gets refused.
    shots[2]["starts_on_last_frame_of"] = "shot_0001"
    out = rd.run_episode(led)
    done = out["ledger"]["video"]["shots"]
    assert [s["shot_id"] for s in done] == [
        "shot_0000", "shot_0001", "shot_0002", "shot_0003"]
    assert done[1]["cloud_floor"] == "content_refused"
    # The successor is floored FOR ITS OWN STATED REASON, not silently absent.
    assert done[2]["cloud_floor"] == "predecessor_floored"
    assert rd._shot_is_cloud_floor(done[2])
    # It is NOT a content refusal -- nothing refused it; it was never asked.
    assert not rd._shot_is_content_floor(done[2])
    # The independent beats around the chain still rendered and committed.
    assert "shot_0000" in out["clips"] and "shot_0003" in out["clips"]
    assert "shot_0001" not in out["clips"] and "shot_0002" not in out["clips"]


def test_a_missing_shot_with_no_floored_predecessor_still_raises():
    """The cascade branch must not become a blanket excuse for absence."""
    assert rd._floored_predecessor(
        {"shot_id": "s9", "starts_on_last_frame_of": "s8"}, {"s7"}) == ""


def _gap_ledger(gap_beat, n=4):
    """A cloud ledger whose beat ``gap_beat`` has a REFUSED required still."""
    led = _cloud_ledger(refused_index=-1, n=n)
    led.setdefault("images", {})["required_scene_targets"] = [{
        "beat_id": gap_beat, "object_id": "obj_%s" % gap_beat,
        "status": "sanctioned_gap",
    }]
    return led


def test_a_still_gap_does_not_kill_the_run_through_its_chain_successor(
        cloud_stub_registry, monkeypatch):
    """THE HOLE THE FIRST CASCADE FIX LEFT OPEN.

    A sanctioned still-gap beat is filtered out of ``work_shots`` and never
    submitted, so it produces NO error. Its chain successor therefore never
    becomes ready and comes back in ``stuck_ids`` beside an EMPTY error dict
    -- which is precisely what the old ``if outcome.stuck_ids and not errors:
    raise`` short-circuit fired on, before the commit walk could floor either
    of them. A refused STILL killed a fully paid VIDEO run.
    """
    monkeypatch.setenv("OTR_CLOUD_FANOUT", "4")
    led = _gap_ledger("b1", n=4)
    led["video"]["shots"][2]["starts_on_last_frame_of"] = "shot_0001"
    out = rd.run_episode(led)
    done = out["ledger"]["video"]["shots"]
    assert [s["shot_id"] for s in done] == [
        "shot_0000", "shot_0001", "shot_0002", "shot_0003"]
    # The gapped beat keeps its place, and so does the successor it stranded.
    assert done[2]["cloud_floor"] == "predecessor_floored"
    # Every independent beat still rendered and committed.
    assert "shot_0000" in out["clips"] and "shot_0003" in out["clips"]
    assert "shot_0001" not in out["clips"] and "shot_0002" not in out["clips"]


def test_a_still_gap_cascade_is_floored_on_the_serial_walk_too(
        cloud_stub_registry, monkeypatch):
    """The serial branch could submit the successor, so it must ask first.

    Rendering it would hand a chain segment an init frame that is a clip which
    does not exist -- it would die inside render_beat_coverage with no stamp,
    no floor, and the paid beats with it.
    """
    monkeypatch.setenv("OTR_CLOUD_FANOUT", "1")
    monkeypatch.setenv("OTR_CLOUD_VIDEO_FANOUT", "1")
    led = _gap_ledger("b1", n=4)
    led["video"]["shots"][2]["starts_on_last_frame_of"] = "shot_0001"
    assert rd._should_fanout_cloud_episode(led["video"], {"b1"}) is False
    out = rd.run_episode(led)
    done = out["ledger"]["video"]["shots"]
    assert done[2]["cloud_floor"] == "predecessor_floored"
    assert "shot_0000" in out["clips"] and "shot_0003" in out["clips"]
    assert "shot_0002" not in out["clips"]


def test_a_stuck_shot_with_no_floored_predecessor_still_raises_loud(
        cloud_stub_registry, monkeypatch):
    """Removing the short-circuit must not make every leftover survivable.

    A shot that never became ready for a reason nobody floored is an
    unexplained hole, and the walk must still fail LOUD -- now from the branch
    that can name which shot.
    """
    monkeypatch.setenv("OTR_CLOUD_FANOUT", "4")
    led = _cloud_ledger(refused_index=-1, n=3)
    # A predecessor that is not in this episode at all: nothing ever floors
    # it, so shot_0002 can never become ready and nothing explains why.
    led["video"]["shots"][2]["starts_on_last_frame_of"] = "shot_does_not_exist"
    with pytest.raises(rd.RenderError) as ei:
        rd.run_episode(led)
    assert "shot_0002" in str(ei.value)


def test_chain_order_inversion_is_caught_before_a_walk_begins():
    """The invariant both cascades rest on, stated and enforced.

    A successor listed AHEAD of its predecessor consults a floored-set that
    does not contain it yet, so it would not cascade -- and one refusal would
    take a paid episode down again through the door this machinery closed.
    """
    good = [
        {"shot_id": "s0"},
        {"shot_id": "s1", "starts_on_last_frame_of": "s0"},
        {"shot_id": "s2", "starts_on_last_frame_of": "s1"},
    ]
    assert rd.assert_chain_order(good) is good
    bad = [
        {"shot_id": "s1", "starts_on_last_frame_of": "s0"},
        {"shot_id": "s0"},
    ]
    with pytest.raises(rd.RenderError, match="chain order is inverted"):
        rd.assert_chain_order(bad)
    # A predecessor that is simply not in this episode is NOT an ordering
    # fault -- the walk raises about it later, where it can say more.
    rd.assert_chain_order([{"shot_id": "s1", "starts_on_last_frame_of": "gone"}])
    # Empty / jump-only episodes are fine.
    rd.assert_chain_order([])
    rd.assert_chain_order([{"shot_id": "s0"}, {"shot_id": "s1"}])


def test_run_episode_refuses_an_inverted_chain(cloud_stub_registry,
                                               monkeypatch):
    """It fires from run_episode, before either walk spends anything."""
    monkeypatch.setenv("OTR_CLOUD_FANOUT", "4")
    led = _cloud_ledger(refused_index=-1, n=3)
    # shot_0000 chains from shot_0002, which the ledger lists LAST.
    led["video"]["shots"][0]["starts_on_last_frame_of"] = "shot_0002"
    with pytest.raises(rd.RenderError, match="chain order is inverted"):
        rd.run_episode(led)


def test_a_local_successor_of_a_floored_beat_is_never_stamped_cloud(caplog):
    """A LOCAL engine never floors, even downstream of a cloud floor.

    Caught in review 2026-09-16: `_floored_predecessor` has no engine gate, so
    a local-engine beat that merely chained from a floored cloud beat was
    stamped `cloud_floor="predecessor_floored"` and skipped -- a local shot
    counted in the cloud-failure tally and silently dropped under NO FALLBACKS.
    It is still unrenderable, so it must not be ATTEMPTED either; it fails
    LOUD, which is what a local fault gets.
    """
    import logging
    from nodes._otr_shared import cloud_media_backend as cmb
    refused = cmb.CloudMediaError(
        cmb.CloudErrorCode.CONTENT_REFUSED, "Content filtered")

    cloud_shot = {"shot_id": "s0", "engine_id": "cloud_ltx25_foley_plus"}
    local_next = {"shot_id": "s1", "engine_id": "humo",
                  "starts_on_last_frame_of": "s0"}
    # The predecessor itself floors, because it IS a cloud beat.
    assert rd._cloud_floor_reason(
        "s0", {"s0": refused}, cloud_shot) == "content_refused"
    # The local successor is found as cascaded...
    assert rd._floored_predecessor(local_next, {"s0"}) == "s0"
    # ...but it is NOT a cloud engine, so it must never carry a cloud floor.
    assert not rd._is_cloud_video_engine("humo")


def test_a_paid_foley_assembly_fault_is_stamped_corrupt():
    """assemble_beat_foley_segments runs AFTER the provider was paid.

    Unstamped, those FoleyStemErrors sailed past the cloud floor and dumped
    the episode. The helper must stamp them, and the assemble site must
    raise through it.
    """
    from nodes._otr_shared.cloud_media_backend import (
        CloudErrorCode, cloud_job_failure_code)
    from nodes._otr_video_engines.foley_stems import FoleyStemError

    err = FoleyStemError("assembled foley is 1 sample(s) for a beat of 25")
    stamped = rd._cloud_output_fault(err, "cloud_ltx25_foley_plus")
    assert cloud_job_failure_code(stamped) is CloudErrorCode.CORRUPT_OUTPUT
    src = pathlib.Path(rd.__file__).read_text(encoding="utf-8")
    assemble_at = src.find("assemble_beat_foley_segments")
    wrap_at = src.find("_cloud_output_fault(stem_exc", assemble_at)
    assert assemble_at > 0 and wrap_at > assemble_at, (
        "assemble_beat_foley_segments is not raised through "
        "_cloud_output_fault; a paid Foley concatenate still dumps the run")


def test_report_cloud_floors_counts_still_gaps_and_budget_floors_too(caplog):
    """The banner divided cloud floors by the WHOLE episode and undercounted.

    Four still gaps plus two cloud floors in a 12-beat act is 50% holes; the
    earlier draft reported 17%, stayed under the threshold, and said nothing.
    """
    import logging
    from nodes._otr_shared import still_receipt as _receipt
    shots = [{"shot_id": "ok%d" % i} for i in range(6)]
    shots += [rd._stamp_cloud_floor_shot({"shot_id": "c%d" % i}, "timeout")
              for i in range(2)]
    shots += [rd._stamp_budget_floor_shot({"shot_id": "b0"})]
    gap = {"shot_id": "g0", "status": _receipt.STATUS_SANCTIONED_GAP}
    shots += [gap, dict(gap, shot_id="g1"), dict(gap, shot_id="g2")]
    with caplog.at_level(logging.WARNING):
        got = rd._report_cloud_floors(shots)
    # 6 holes out of 12 beats = 50%, comfortably over the 25% threshold.
    assert len(got) == 6, got
    assert "SYSTEMIC CLOUD FAILURE" in caplog.text
    # Each KIND is named separately, so the tally stays truthful.
    assert "timeout: c0, c1" in caplog.text
    assert "budget: b0" in caplog.text
    assert "still_gap: g0, g1, g2" in caplog.text

