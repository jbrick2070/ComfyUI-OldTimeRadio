"""Canonical REPLAY (campaign item 0, 2026-09-02): the bundle import, the identity
helpers, the writer's short-circuit, the typed pass-through returns of the audio chain and
MetaBrief, the assembler's byte-copy seam, ShotLock's planned-section reuse, the dispatcher's
verify-and-restamp, the freeze script and the manifest's safety rules. All offline, no
CUDA, no model: the point of a replay is that none of those load.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))
os.environ.setdefault("OTR_TEST_MODE", "1")

from nodes import production_ledger as PL  # noqa: E402


# --------------------------------------------------------------------------
# a frozen episode on disk, and a bundle of it
# --------------------------------------------------------------------------

def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _make_episode(root: Path, ep_id="signal_lost_frozen_20260902_000000", with_plan=True):
    ep = root / ep_id
    (ep / "audio").mkdir(parents=True)
    (ep / "stills").mkdir()
    (ep / "portraits").mkdir()
    still = ep / "stills" / "scene_b001.png"; still.write_bytes(b"\x89PNG still bytes")
    portrait = ep / "portraits" / "c02.png"; portrait.write_bytes(b"\x89PNG portrait")
    # A REAL 3-second 16-bit stereo WAV, not a byte blob. The sequencer's replay
    # pass-through reads this file and hands its samples to the whole downstream
    # chain, and the procgen visualizer renders one frame per audio frame -- so
    # a fixture that is not openable by `wave` would silently exercise only the
    # fallback and PBUG-20260903-02 would pass unnoticed here.
    master = ep / "audio" / (ep_id + "_master.wav")
    import wave as _wave
    with _wave.open(str(master), "wb") as _fh:
        _fh.setnchannels(2)
        _fh.setsampwidth(2)
        _fh.setframerate(48000)
        _fh.writeframes(b"\x00\x10" * 2 * 48000 * 3)
    ledger = {
        "schema_version": "l4-2026-08-07", "episode_id": ep_id, "commit": "abc",
        "total_episode_dur_s": 90.0,
        "cast": [{"char_id": "c02", "name": "DR ZHANG", "gender": "male"}],
        "lines": [{"line_id": "b001", "speaker": "DR ZHANG", "char_id": "c02",
                   "text": "The sequence is initiated.", "start_s": 1.0, "dur_s": 5.0}],
        "beats": [], "scenes": [], "shots": [], "music": [], "clips": [],
        "audio": {"master_audio_sha256": _sha(master.read_bytes()), "ledger_frozen": True},
        "final_audio_path": str(master), "final_video_path": str(ep / "final.mp4"),
        "meta": {"episode_id": ep_id, "episode_title": "Frozen", "freeze_timestamp": "2026-09-02T00:00:00+00:00",
                 "cast_lock_revision": 3, "video_revision": 2, "technical_model": "google/gemma-4-E2B-it",
                 "visual_style": "anime", "render_engines": {"histogram": {"x": 1}},
                 "render_trace": [{"shot_id": "old"}], "phase_ms": {"a": 1},
                 # the receipts `otr_credits_roll` requires at mux time -- the
                 # fixture carries them so a replay that drops one is caught
                 # here rather than sixteen minutes into a real render
                 "image_engines": {"by_role": {}, "image_revision": 1},
                 "music_engine": "musicgen", "source_bank": "media_archive",
                 "paths": {"stills_dir": str(ep / "stills")}},
        "images": {"image_revision": 1, "images": [
            {"image_id": "i1", "kind": "scene_beat", "path": str(still), "pool_path": str(still),
             "beat_id": "b001"},
            {"image_id": "i2", "kind": "portrait", "path": str(portrait), "char_id": "c02"}]},
    }
    if with_plan:
        ledger["video"] = {"video_revision": 2, "shots": [
            {"shot_id": "shot_b001", "beat_id": "b001", "engine_id": "animatediff15_v3_haunted_video",
             "render_request_hash": "0123456789abcdef", "ghost_prompt": {"mode": "figure",
             "motif_cue": "a lean figure", "drawable_beat": "the figure turns"}}]}
    lp = ep / "audio" / (ep_id + "_ledger.json")
    lp.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    (ep / "episode_canon.json").write_text(json.dumps({"title": "Frozen"}), encoding="utf-8")
    return ep, ledger


def _freeze(ep: Path, out: Path, **kw) -> Path:
    import importlib
    fz = importlib.import_module("otr_freeze_replay_bundle")
    return fz.freeze(ep, out, **kw)


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    # every ledger the test creates lands under tmp, never in the real output tree
    monkeypatch.setattr(PL, "_default_out_dir", lambda ep=None: str(tmp_path / "episodes" / (ep or "pending") / "audio"))
    ep, ledger = _make_episode(tmp_path / "episodes")
    bundle = _freeze(ep, tmp_path / "bundles")
    return {"ep": ep, "ledger": ledger, "bundle": bundle, "tmp": tmp_path}


# --------------------------------------------------------------------------
# the freeze script and the manifest
# --------------------------------------------------------------------------

def test_freeze_writes_a_verified_manifest_with_relative_paths(frozen):
    man = json.loads((frozen["bundle"] / PL.REPLAY_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert man["schema_version"] == PL.REPLAY_MANIFEST_SCHEMA
    assert man["source_episode_id"] == frozen["ep"].name and man["planned_shots"] == 1
    paths = [f["path"] for f in man["files"]]
    assert all(not os.path.isabs(p) and ".." not in p for p in paths)
    assert man["ledger"] in paths and man["master_audio"] in paths
    assert {"stills/scene_b001.png", "portraits/c02.png", "episode_canon.json"} <= set(paths)
    for f in man["files"]:
        full = frozen["bundle"] / Path(f["path"])
        assert full.stat().st_size == f["bytes"] and _sha(full.read_bytes()) == f["sha256"]
    assert PL.load_replay_manifest(str(frozen["bundle"]))["source_episode_id"] == frozen["ep"].name


def test_freeze_refuses_a_ledger_without_the_planned_section(tmp_path, monkeypatch):
    ep, _ = _make_episode(tmp_path / "e", ep_id="noplan_20260902_000000", with_plan=False)
    with pytest.raises(SystemExit, match="planned video.shots"):
        _freeze(ep, tmp_path / "b")
    assert _freeze(ep, tmp_path / "b2", allow_no_plan=True).is_dir()


def test_freeze_is_immutable_and_refuses_an_existing_bundle(frozen):
    with pytest.raises(SystemExit, match="already exists"):
        _freeze(frozen["ep"], frozen["bundle"].parent)


@pytest.mark.parametrize("tamper", ["digest", "traversal", "absolute", "missing", "duplicate", "schema"])
def test_the_manifest_rejects_every_unsafe_shape(frozen, tamper):
    bundle = frozen["bundle"]
    mp = bundle / PL.REPLAY_MANIFEST_NAME
    man = json.loads(mp.read_text(encoding="utf-8"))
    if tamper == "digest":
        (bundle / "stills" / "scene_b001.png").write_bytes(b"tampered")
    elif tamper == "traversal":
        man["files"].append({"path": "../escape.txt", "bytes": 1, "sha256": "x"})
    elif tamper == "absolute":
        man["files"].append({"path": str(bundle / "stills" / "scene_b001.png"), "bytes": 1, "sha256": "x"})
    elif tamper == "missing":
        man["files"].append({"path": "stills/gone.png", "bytes": 1, "sha256": "x"})
    elif tamper == "duplicate":
        man["files"].append(dict(man["files"][-1], path=man["files"][-1]["path"].upper()))
    else:
        man["schema_version"] = "something_else"
    mp.write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(PL.ReplayBundleError):
        PL.load_replay_manifest(str(bundle))


# --------------------------------------------------------------------------
# the import: a new workspace, rebased, cleared, singleton rebound
# --------------------------------------------------------------------------

def test_import_replay_bundle_clones_into_a_new_workspace(frozen):
    led = PL.import_replay_bundle(str(frozen["bundle"]))
    data = led.data
    new_id = data["episode_id"]
    assert new_id.startswith(frozen["ep"].name + "_replay_") and new_id != frozen["ep"].name
    assert PL.peek_ledger() is led, "the singleton is rebound"
    meta = data["meta"]
    assert meta["replay_of_episode"] == frozen["ep"].name
    assert meta["replay_from"] == str(frozen["bundle"])
    assert meta["replay_workspace_id"] and meta["freeze_timestamp"] == "2026-09-02T00:00:00+00:00"
    assert meta["cast_lock_revision"] == 3 and meta["video_revision"] == 2
    for gone in ("render_engines", "render_trace", "phase_ms", "paths"):
        assert gone not in meta or gone == "paths"   # paths is re-stamped by save(); the source's is gone
    assert data["final_audio_path"] is None and data["final_video_path"] is None
    assert data["video"]["shots"][0]["render_request_hash"] == "0123456789abcdef"
    # assets materialized and rows rebased onto the new dir
    new_root = Path(led.out_dir).parent
    row = data["images"]["images"][0]
    assert Path(row["path"]).is_file() and Path(row["path"]).parent == new_root / "stills"
    assert Path(row["pool_path"]).parent == new_root / "stills"
    assert (new_root / "portraits" / "c02.png").read_bytes() == b"\x89PNG portrait"
    assert not (new_root / "audio" / (frozen["ep"].name + "_master.wav")).exists(), \
        "the master is node 7's to copy, not the import's"
    assert PL.replay_descriptor(meta)["replay_workspace_id"] == meta["replay_workspace_id"]
    assert PL.replay_descriptor({}) == {} and PL.replay_descriptor({"replay_from": ""}) == {}
    # durable on disk, with the planned section preserved
    on_disk = json.loads(Path(led.path).read_text(encoding="utf-8"))
    assert on_disk["video"]["shots"][0]["shot_id"] == "shot_b001"


def test_the_replay_keeps_every_receipt_the_credits_roll_requires(frozen):
    """PBUG-20260903-01: eight clips rendered and NOTHING published.

    `import_replay_bundle` cleared `meta.image_engines` as run-volatile, but the
    only thing that stamps it is `OTR_ImageGenDispatcher`, which a replay does
    not run -- a replay IMPORTS the source's stills instead of minting new ones.
    So `otr_credits_roll` raised `CreditsDataError: required credits receipt
    missing: meta.image_engines` at mux time, after sixteen minutes of render
    and eight good clips on disk.

    Carrying it forward is the CORRECT value, not a stale one: the replay shows
    the imported stills, so the engines that made them are the source's engines.

    This pins the whole requirement SET rather than the one key, because the
    next member added to the volatile list would fail exactly the same way, and
    the failure only surfaces at the very end of a real render.
    """
    led = PL.import_replay_bundle(str(frozen["bundle"]))
    meta = led.data["meta"]
    # `otr_credits_roll` calls `_require(meta, key, "meta")` on each of these,
    # which refuses None, "", {} and [].
    for key in ("episode_title", "visual_style", "image_engines",
                "music_engine", "source_bank"):
        assert meta.get(key) not in (None, "", {}, []), key
    # and the genuinely run-volatile ones are still cleared, because the replay
    # rebuilds every one of them for itself
    for key in ("render_engines", "render_trace", "phase_ms"):
        assert key not in meta, key


def test_the_replay_passes_the_REAL_master_through_not_a_short_placeholder(frozen):
    """PBUG-20260903-02: a one-second placeholder truncated the whole episode.

    The sequencer's replay branch returned one second of silence, on the
    reasoning that no mix is built when node 7 copies the frozen master. One
    consumer measures THIS wire rather than the file: the procgen visualizer
    renders `len(audio) / sample_rate` frames, so it produced a 1.00s overlay
    and `PostUpscaleProcgenBlend` cut an 85.7s episode down to one second of
    picture. It published green -- `obs_publish OK` -- and the episode was
    broken.

    The property that matters is the LENGTH, so that is what this asserts.
    """
    from nodes.scene_sequencer import _replay_master_audio

    led = PL.import_replay_bundle(str(frozen["bundle"]))
    meta = led.data["meta"]
    out, note = _replay_master_audio(meta)
    seconds = out["waveform"].shape[-1] / out["sample_rate"]
    assert seconds > 1.5, (
        "the replay wire is %.2fs -- anything that measures duration from it "
        "(the procgen visualizer, the blend) will truncate the episode: %s"
        % (seconds, note))
    assert out["waveform"].dim() == 3 and out["waveform"].shape[1] == 2


def test_the_wav_reader_does_not_need_torchaudio(frozen, monkeypatch):
    """PBUG-20260903-03: the assembler's torchaudio load ALWAYS raises here.

    `torchaudio.load` on this stack raises `ImportError: TorchCodec is required
    for load_with_torchcodec` -- torchaudio 2.10 moved decoding to torchcodec,
    which is not installed. The assembler wrapped it in a bare
    `except Exception` that fell through to one second of silence, so every
    replay handed the video node a one-second wire and the published episode
    came out with one second of picture.

    The stdlib reader is the one that has to work, so this asserts it directly
    with torchaudio made unavailable.
    """
    import builtins
    from nodes.scene_sequencer import wav_file_as_audio

    real_import = builtins.__import__

    def _no_torchaudio(name, *a, **k):
        if name == "torchaudio":
            raise ImportError("TorchCodec is required for load_with_torchcodec")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_torchaudio)
    master = frozen["ep"] / "audio" / (frozen["ep"].name + "_master.wav")
    audio, note = wav_file_as_audio(str(master))
    seconds = audio["waveform"].shape[-1] / audio["sample_rate"]
    assert abs(seconds - 3.0) < 0.01, note
    assert audio["waveform"].shape[1] == 2


@pytest.mark.parametrize("meta", [
    None, {}, {"replay_from": "/nope"},
    {"replay_from": "/nope", "replay_master_audio": "audio/missing.wav"},
])
def test_the_master_passthrough_degrades_safely_and_never_raises(meta):
    """A missing or unreadable master costs the LENGTH, never the run."""
    from nodes.scene_sequencer import _replay_master_audio

    out, note = _replay_master_audio(meta)
    assert out["waveform"].dim() == 3 and out["sample_rate"] > 0
    assert "fallback" in note


def test_two_imports_get_two_workspaces(frozen):
    a = PL.import_replay_bundle(str(frozen["bundle"]))
    a_id, a_ws = a.data["episode_id"], a.data["meta"]["replay_workspace_id"]
    b = PL.import_replay_bundle(str(frozen["bundle"]))
    assert b.data["episode_id"] != a_id and b.data["meta"]["replay_workspace_id"] != a_ws


def test_a_replay_workspace_is_never_the_sources_durable_run():
    src = {"episode_id": "e", "meta": {"freeze_timestamp": "t"}}
    rep = {"episode_id": "e_replay_1", "meta": {"freeze_timestamp": "t", "replay_workspace_id": "ws1"}}
    rep2 = {"episode_id": "e_replay_1", "meta": {"freeze_timestamp": "t", "replay_workspace_id": "ws1"}}
    other = {"episode_id": "e_replay_2", "meta": {"freeze_timestamp": "t", "replay_workspace_id": "ws2"}}
    assert PL._same_durable_run(src, copy.deepcopy(src))
    assert not PL._same_durable_run(src, rep), "same freeze receipt, but one side carries a workspace id"
    assert PL._same_durable_run(rep, rep2)
    assert not PL._same_durable_run(rep, other)
    from nodes import otr_shot_lock as SL
    assert SL._same_frozen_episode(rep, rep2)[0]
    assert not SL._same_frozen_episode(src, rep)[0]
    assert not SL._same_frozen_episode(rep, other)[0]


# --------------------------------------------------------------------------
# the nodes on replay: typed pass-through, no model
# --------------------------------------------------------------------------

def _replay_ledger_json(frozen):
    led = PL.import_replay_bundle(str(frozen["bundle"]))
    return json.dumps(led.data, ensure_ascii=True, separators=(",", ":")), led


def test_the_writer_short_circuits_on_replay_from(frozen, monkeypatch):
    from nodes import OTR_LedgerScriptWriter as W
    node = W.OTR_LedgerScriptWriter()
    # no roll, no bank gate, no LLM: all of those live after the branch
    monkeypatch.setattr(W, "_ROLLS", None)
    out = node.run(replay_from=str(frozen["bundle"]))
    script_text, script_json, news_used, est, technical_model = out
    data = json.loads(script_json)
    assert data["meta"]["replay_of_episode"] == frozen["ep"].name
    assert "DR ZHANG: The sequence is initiated." in script_text
    # the estimate rides the INT wire slot: 90 s -> 1.5 min -> rounded (half to even) to 2
    assert news_used == "" and est == 2 and isinstance(est, int)
    assert technical_model == "google/gemma-4-E2B-it"
    assert PL.peek_ledger().data["episode_id"] == data["episode_id"]


def test_the_freeze_cascade_passes_a_replay_through_unchanged(frozen):
    from nodes.OTR_LedgerFreezeCascade import OTR_LedgerFreezeCascade
    sj, led = _replay_ledger_json(frozen)
    out = OTR_LedgerFreezeCascade().run(script_text="t", script_json=sj, news_used="n",
                                        estimated_minutes=2, technical_model="m")
    assert out[1] == sj and out[6] == sj and out[4] == "replay"
    assert json.loads(out[1])["meta"]["freeze_timestamp"] == "2026-09-02T00:00:00+00:00"


def test_cast_lock_returns_before_its_revision_bump_on_replay(frozen):
    from nodes.cast_lock import CastLock
    sj, led = _replay_ledger_json(frozen)
    ledger_json, rev, report, done = CastLock().lock(sj, cast_voice_policy="auto_registry")
    assert rev == 3 and done == "cast_lock:replay"
    assert json.loads(ledger_json)["meta"]["cast_lock_revision"] == 3
    assert json.loads(ledger_json)["cast"] == led.data["cast"]


def test_voices_music_and_sequencer_pass_through_on_replay(frozen):
    from nodes.batch_character_voices import BatchCharacterVoices
    from nodes.announcer_voice import AnnouncerVoice
    from nodes.stable_audio_theme import StableAudioTheme
    from nodes.scene_sequencer import SceneSequencer
    sj, _ = _replay_ledger_json(frozen)
    for cls in (BatchCharacterVoices, AnnouncerVoice):
        audio, log_, done = cls().generate(sj, "kokoro", ledger_json=sj)
        assert done == "replay:passthrough" and audio["waveform"].shape[-1] == 0
    cue_audio, manifest, log_, done = StableAudioTheme().generate(sj, "stable_audio_3", ledger_json=sj)
    assert done == "replay:passthrough" and manifest == ""
    audio, log_ = SceneSequencer().sequence(sj)
    # THE SEQUENCER PASSES THE FROZEN MASTER, NOT A ONE-SECOND STUB. This line
    # used to assert `(1, 2, 48000)` -- exactly one second -- which is the
    # contract that shipped PBUG-20260903-02: the procgen visualizer renders one
    # frame per audio frame off this wire, so a one-second batch cut the whole
    # published episode down to one second of picture. The fixture's master is
    # three seconds, so the shape follows it.
    assert audio["sample_rate"] == 48000
    assert audio["waveform"].shape == (1, 2, 48000 * 3)
    assert audio["waveform"].device.type == "cpu"
    assert "replay" in log_


def test_meta_brief_and_dispatcher_pass_through_on_replay(frozen):
    from nodes.otr_meta_brief_image_prompt import OTRMetaBriefImagePromptGen
    from nodes import otr_image_gen_dispatcher as D
    sj, led = _replay_ledger_json(frozen)
    prompts_json, log_ = OTRMetaBriefImagePromptGen().generate(sj)
    assert json.loads(prompts_json) == {"replay": True, "objects": []}
    patched, image_done, report = D.OTRImageGenDispatcher().dispatch(sj, "{}", prompts_json)
    assert image_done == "image_done:replay" and "verified" in report
    assert json.loads(patched)["images"]["image_revision"] == 1, "nothing minted, nothing bumped"
    # a missing imported file fails loud, never a gen_fn call
    os.remove(led.data["images"]["images"][0]["path"])
    with pytest.raises(RuntimeError, match="no file on disk"):
        D.OTRImageGenDispatcher().dispatch(sj, "{}", prompts_json)


def test_the_assembler_copies_and_verifies_the_frozen_master_before_audio_done(frozen):
    from nodes.scene_sequencer import EpisodeAssembler
    sj, led = _replay_ledger_json(frozen)
    descriptor = json.dumps({"meta": led.data["meta"]})
    audio, out_path, info, done = EpisodeAssembler().assemble(
        None, "Frozen", music_cue_audio={"waveform": None, "sample_rate": 48000},
        music_cue_manifest_json="", replay_descriptor=descriptor)
    want = frozen["ledger"]["audio"]["master_audio_sha256"]
    assert done == "audio_done:replay:" + want[:12]
    assert Path(out_path).name == led.data["episode_id"] + "_master.wav"
    assert _sha(Path(out_path).read_bytes()) == want
    assert PL.peek_ledger().data["final_audio_path"] == out_path
    assert json.loads(info)["replay_of_episode"] == frozen["ep"].name


def test_the_assembler_withholds_audio_done_on_a_digest_mismatch(frozen):
    from nodes.scene_sequencer import EpisodeAssembler
    sj, led = _replay_ledger_json(frozen)
    led.data["meta"]["replay_master_sha256"] = "0" * 64
    descriptor = json.dumps({"meta": led.data["meta"]})
    with pytest.raises(RuntimeError, match="audio_done withheld"):
        EpisodeAssembler().assemble(None, "Frozen", replay_descriptor=descriptor)
    assert not (Path(led.out_dir) / (led.data["episode_id"] + "_master.wav")).exists()


def test_the_assembler_refuses_a_descriptor_for_another_workspace(frozen):
    from nodes.scene_sequencer import EpisodeAssembler
    sj, led = _replay_ledger_json(frozen)
    descriptor = json.dumps({"meta": dict(led.data["meta"], replay_workspace_id="not-this-one")})
    with pytest.raises(RuntimeError, match="not the bound"):
        EpisodeAssembler().assemble(None, "Frozen", replay_descriptor=descriptor)


def test_shot_lock_reuses_the_planned_section_without_an_llm(frozen, monkeypatch):
    from nodes import otr_shot_lock as SL
    sj, led = _replay_ledger_json(frozen)
    monkeypatch.setattr(SL, "_resolve_writer_llm",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM resolved on replay")))
    policy = json.dumps({"policy_version": 2, "canvas": {"fps": 25}})
    patched, revision, report, done, episode_id = SL.OTRShotLock().lock(sj, audio_done="", video_policy_json=policy)
    assert revision == 2 and done == "shot_lock:done:rev=2" and "replay" in report
    assert json.loads(patched)["video"]["shots"][0]["render_request_hash"] == "0123456789abcdef"
    assert episode_id == led.data["episode_id"]
    assert PL.peek_ledger().data["video"]["shots"][0]["shot_id"] == "shot_b001"


def test_shot_lock_refuses_a_replay_without_the_plan(tmp_path, monkeypatch):
    from nodes import otr_shot_lock as SL
    monkeypatch.setattr(PL, "_default_out_dir", lambda ep=None: str(tmp_path / "episodes" / (ep or "pending") / "audio"))
    ep, _ = _make_episode(tmp_path / "e", ep_id="noplan_20260902_000001", with_plan=False)
    bundle = _freeze(ep, tmp_path / "b", allow_no_plan=True)
    led = PL.import_replay_bundle(str(bundle))
    sj = json.dumps(led.data)
    with pytest.raises(ValueError, match="planned video"):
        SL.OTRShotLock().lock(sj, video_policy_json=json.dumps({"policy_version": 2}))


# --------------------------------------------------------------------------
# the harness and the whitelists
# --------------------------------------------------------------------------

def test_replay_from_is_whitelisted_in_both_copies_and_the_canonical_carries_the_wiring():
    import importlib.util
    spec = importlib.util.spec_from_file_location("otr_api_test", _REPO / "scripts" / "otr_api.py")
    api = importlib.util.module_from_spec(spec); spec.loader.exec_module(api)
    from nodes import _otr_workflow_apply as WA
    assert "replay_from" in api.CREATIVE_WHITELIST and "replay_from" in WA.CREATIVE_WHITELIST
    wf = json.loads((_REPO / "workflows" / "otr_canonical.json").read_text(encoding="utf-8"))
    by = {n["id"]: n for n in wf["nodes"]}
    assert by[1]["inputs"][-1]["name"] == "replay_from" and by[1]["widgets_values"][-1] == ""
    assert by[7]["inputs"][10]["name"] == "replay_descriptor"
    link = next(l for l in wf["links"] if l[0] == 289)
    assert link[1:5] == [62, 6, 7, 10] and 289 in by[62]["outputs"][6]["links"]
    assert wf["last_link_id"] == 289


def test_the_verifier_recomputes_the_receipt_sha():
    import importlib
    vr = importlib.import_module("otr_verify_replay")
    from nodes._otr_video_engines import render_driver as rd
    row = {k: ("x" if k != "sampler_inputs" else {"steps": 20}) for k in rd._RECEIPT_CAUSAL_KEYS}
    row["actual_request_sha"] = vr.recompute_sha(row)
    assert vr.recompute_sha(row) == row["actual_request_sha"]
    row["text_prompt"] = "changed"
    assert vr.recompute_sha(row) != row["actual_request_sha"]
