"""Additive CPU coverage for the in-process render driver (A-S7.5).

New, self-contained tests complementing tests/test_video_render_driver.py. The
pure helpers (engine_family / build_full_ledger / build_soak_fixture range guard
/ classify_failure kind map / make_fallback_of overlay) plus a CPU exercise of
the REAL render loop (run_episode) driven by STUB engines registered into a
snapshot/restore registry: it proves a HARD failure degrades LOUDLY down the
chain (decision appended + degradation trail), the frozen audio section is never
touched, the input ledger is not mutated, and two runs are deterministic. No GPU,
no model load, no production edits. UTF-8, no BOM, ASCII-only, SFW.
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
    assert (oom["engine_id"], oom["family"]) == ("hunyuan3d_talk", "character_3d")
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
    assert fb("hunyuan3d_talk") == "humo"               # default synth overlay
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


def test_run_episode_degrades_loudly_and_keeps_audio_frozen(stub_registry):
    ledger = _two_shot_ledger()
    res = rd.run_episode(ledger, fallback_of=_fb)

    assert len(res["clips"]) == 2
    out = res["ledger"]["video"]
    shot1 = {s["shot_id"]: s for s in out["shots"]}["shot_0001"]
    assert shot1["engine_id"] == "stub_ok"              # degraded to the fallback
    assert shot1["degradation_trail"] == ["stub_fail->stub_ok (crash_before_load)"]

    decisions = out["runtime_fallback_decisions"]
    assert len(decisions) == 1
    d = decisions[0]
    assert d["from_engine"] == "stub_fail" and d["to_engine"] == "stub_ok"
    assert d["failure_kind"] == "crash_before_load"
    assert d["block_class"] == "hard"
    assert d["video_revision"] == 1                     # restamp stays at rev 1
    assert out["video_revision"] == 1
    # frozen audio section is byte-identical after the run
    assert res["ledger"]["audio"]["master_audio_sha256"] == rd.FROZEN_AUDIO_SHA


def test_run_episode_does_not_mutate_input_ledger(stub_registry):
    ledger = _two_shot_ledger()
    rd.run_episode(ledger, fallback_of=_fb)
    # the original ledger is deep-copied; its shot is untouched
    assert ledger["video"]["shots"][1]["engine_id"] == "stub_fail"
    assert ledger["video"]["shots"][1]["degradation_trail"] == []
    assert "runtime_fallback_decisions" not in ledger["video"]


def test_run_episode_is_deterministic(stub_registry):
    a = rd.run_episode(_two_shot_ledger(), fallback_of=_fb)
    b = rd.run_episode(_two_shot_ledger(), fallback_of=_fb)
    assert a["trace"] == b["trace"]
    assert a["trace"][1]["attempts"] == ["stub_fail", "stub_ok"]
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
    assert req["timing"]["start_s"] == 2.0 and req["timing"]["dur_s"] == 1.6
    assert req["char_id"] == "c03"


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
