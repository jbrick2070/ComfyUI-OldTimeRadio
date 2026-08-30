"""CW-4 -- the NEW render path (OTR_SilentComposite + terminal OTR_MasterAudioMux)
+ the cheap radio-floor families. CPU tests.

The audio-critical assertions use real ffmpeg on SYNTHESIZED media (sine master +
black silent video) -- the same headless approach the A-S2 mux probe used -- so
the mux byte-identity (V-1 / C7) and the silent-composite 0-audio invariants are
proven without a GPU. The live end-to-end episode render is an INTERACTIVE
ComfyUI smoke (operator), not covered here.
"""
from __future__ import annotations

import os
import json
import shutil
import subprocess
import sys
import pathlib

import pytest

from nodes.otr_master_audio_mux import (
    mux_master_audio, audio_pcm_sha, OTRMasterAudioMux,
)
from nodes.otr_silent_composite import (
    normalize_to_silent_canonical, count_audio_streams, probe_video, OTRSilentComposite,
    plan_timeline_segments, assemble_silent_timeline, count_video_frames,
    timeline_quality_report,
)
from nodes._otr_video_engines import registry as vreg
from nodes._otr_shared import role_compat as rc

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe required")


def _ff(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args],
                   check=True, capture_output=True)


def _sine(path, dur=2.0):
    _ff("-f", "lavfi", "-i", f"sine=frequency=440:duration={dur}",
        "-ar", "24000", "-ac", "1", str(path))


def _silent_video(path, dur=2.0, with_audio=False):
    args = ["-f", "lavfi", "-i", f"color=c=black:s=64x64:d={dur}", "-r", "25"]
    if with_audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=300:duration={dur}"]
    args += ["-pix_fmt", "yuv420p", "-t", f"{dur}"]
    if not with_audio:
        args += ["-an"]
    args += [str(path)]
    _ff(*args)


# --------------------------------------------------------------------------- #
# OTR_MasterAudioMux -- the terminal, audio-critical node
# --------------------------------------------------------------------------- #
@needs_ffmpeg
def test_master_audio_mux_stream_hash_byte_identical(tmp_path):
    master = tmp_path / "master.wav"
    silent = tmp_path / "silent.mp4"
    out = tmp_path / "final.mkv"
    _sine(master, 2.0)
    _silent_video(silent, 2.0)
    final, report = mux_master_audio(str(silent), str(master), str(out), fps=25)
    assert os.path.isfile(final)
    # the output audio is byte-identical to the frozen master (no re-encode)
    assert audio_pcm_sha(final) == audio_pcm_sha(str(master))
    # and the final actually carries exactly one audio stream
    assert count_audio_streams(final) == 1
    assert "audio_mode=master_copy" in "\n".join(report)


@needs_ffmpeg
def test_master_audio_mux_no_shortest_and_credits_tail_guard(tmp_path, monkeypatch):
    master = tmp_path / "m.wav"
    silent = tmp_path / "s.mp4"
    _sine(master, 2.0)
    _silent_video(silent, 3.0)   # 1s longer than the audio
    # BUG-LOCAL-410: a MODERATE longer-than-audio video is the intentional
    # rolling-credits post-roll (the credits scroll in silence after the closing
    # theme) -> now ALLOWED, and the audio stays byte-identical (-c:a copy).
    out_ok = tmp_path / "ok.mkv"
    final, _r = mux_master_audio(str(silent), str(master), str(out_ok), fps=25)
    assert os.path.isfile(final)
    assert audio_pcm_sha(final) == audio_pcm_sha(str(master))
    # GROSS drift PAST the budget is REPORTED, NOT REFUSED (operator directive
    # 2026-08-30: "don't kill a duration mismatch, just let it fly"). By the
    # time this gate runs, the writer, voices, music, every video beat and the
    # full audio master are already rendered -- refusing to mux discards a
    # finished episode over a consistency judgement, which is the class the
    # operator has ruled must never kill a render. The number still reaches the
    # log at full precision, because a gross overshoot usually IS an upstream
    # frame-budget bug worth chasing.
    monkeypatch.setenv("OTR_MAX_CREDITS_TAIL_S", "0.25")
    drift_out = tmp_path / "drift.mkv"
    final_drift, _rd = mux_master_audio(
        str(silent), str(master), str(drift_out), fps=25)
    assert os.path.isfile(final_drift), "a gross-drift episode was not published"
    # and the audio is STILL byte-identical -- publishing anyway must never
    # become truncating (no -shortest), which is the invariant that matters.
    assert audio_pcm_sha(final_drift) == audio_pcm_sha(str(master))
    # the node never silently truncates (no -shortest); the source proves it -- the
    # only '-shortest' token in the mux module is the V-2 guard assert, never a cmd arg.
    src = (REPO_ROOT / "nodes" / "otr_master_audio_mux.py").read_text(encoding="utf-8")
    assert 'assert "-shortest" not in cmd' in src


def test_master_audio_mux_missing_inputs_fail_closed(tmp_path):
    # missing files -> ValueError (never a half-muxed episode)
    with pytest.raises(ValueError):
        mux_master_audio(str(tmp_path / "nope.mp4"), str(tmp_path / "nope.wav"),
                         str(tmp_path / "o.mkv"))


def test_a_failed_publish_FAILS_THE_RUN_it_never_reports_success(tmp_path, monkeypatch):
    """THE TERMINAL NODE MUST BE ABLE TO FAIL. A render with no episode is a
    FAILED render -- it must never be recorded green.

    OTR_MasterAudioMux is terminal: it muxes the master audio and PUBLISHES to obs.
    Every gate inside mux_master_audio() raises ValueError BY DESIGN ("raises
    ValueError on any gate failure (never produces a silently-wrong episode)").
    The node used to catch (ValueError, OSError) and `return ("", f"error: {exc}")`,
    which neutralized ALL of them: the graph completed, ComfyUI logged
    "Prompt executed", the API reported RESULT SUCCESS -- and no file existed.

    Live 2026-07-14, 420w scifi_codex re-leg (prompt 5ab3884b): the duration gate
    tripped on ~1 frame of concat rounding, this handler ate it, and the leg was
    recorded GREEN with nothing in otr\\obs\\. It would have entered the bake-off as
    a phantom episode. Only the operator's standing law caught it -- confirm the
    asset on disk; API success is not proof.

    A node that cannot fail cannot be trusted, and a guard that cannot abort is not
    a guard.
    """
    node = OTRMasterAudioMux()
    # A missing silent video is one of the fail-closed gates.
    with pytest.raises((ValueError, OSError)):
        node.mux(
            silent_video_path=str(tmp_path / "does_not_exist.mp4"),
            master_audio_path=str(tmp_path / "also_missing.wav"),
            declared_credits_tail_s=0.0,
            fps=25,
            ffmpeg="ffmpeg",
            output_path=str(tmp_path / "out.mkv"),
        )

    # And the swallow must be gone from the CODE -- no `return ("", ...)` escape
    # hatch on the fail-closed arm. Scan executable lines only: a comment that
    # merely DESCRIBES the old bug (as the fix's own does) must not trip this.
    src = (REPO_ROOT / "nodes" / "otr_master_audio_mux.py").read_text(encoding="utf-8")
    code_lines = [
        ln.strip() for ln in src.splitlines()
        if not ln.strip().startswith("#")
    ]
    assert not any(ln.startswith('return ("",') for ln in code_lines), (
        "OTR_MasterAudioMux must RAISE when it cannot publish -- returning an "
        "empty path lets a render with no episode be recorded as SUCCESS."
    )


@needs_ffmpeg
def test_credits_tail_gate_absorbs_concat_frame_quantization(tmp_path):
    """The duration gate must not refuse a finished episode over a rounding.

    The published video is a CONCAT of body + silent credits roll, and ffmpeg lands
    each segment on a frame boundary, so the assembled duration can exceed
    (audio + declared_credits_tail) by a fraction of a frame per segment plus
    timebase slop. The tolerance was ONE frame, which does not cover it.

    Live 2026-07-14: video 294.1600s, audio 218.9307s -> an excess of 75.2293s
    against a declared tail of ~75.18s. Over by ~0.05s -- about one and a quarter
    frames -- and the gate refused to publish. Three hundredths of a second cost a
    25-minute render.

    The GROSS-drift protection is unchanged and still proven below.
    """
    master = tmp_path / "m.wav"
    silent = tmp_path / "s.mp4"
    _sine(master, 2.0)
    _silent_video(silent, 3.0)

    # Declare a credits tail that is a hair SHORT of the real excess -- exactly the
    # quantization case. 1 frame of slack rejects this; the real bound accepts it.
    declared_just_under = (3.0 - 2.0) - (1.5 / 25.0)   # ~1.25 frames short
    final, _r = mux_master_audio(
        str(silent), str(master), str(tmp_path / "quant.mkv"), fps=25,
        declared_credits_tail_s=declared_just_under,
    )
    assert os.path.isfile(final)
    assert audio_pcm_sha(final) == audio_pcm_sha(str(master))

    # GROSS drift is still DETECTED but no longer aborts (see the sibling test
    # and the operator directive of 2026-08-30). The quantization slack remains
    # a real bound rather than a blind widening -- what changed is the penalty
    # for exceeding it: a loud log line instead of a discarded episode.
    gross, _rg = mux_master_audio(
        str(silent), str(master), str(tmp_path / "gross.mkv"), fps=25,
        declared_credits_tail_s=0.01,
    )
    assert os.path.isfile(gross)
    assert audio_pcm_sha(gross) == audio_pcm_sha(str(master))


@needs_ffmpeg
def test_mux_report_declares_unconditional_master_copy(tmp_path):
    """rip-sfx 2026-08-06: the SFX mix branch is gone, so EVERY mux is the
    -c:a copy passthrough. The report must say so and must carry none of the
    retired sfx_mixed fields."""
    master = tmp_path / "master.wav"
    silent = tmp_path / "silent.mp4"
    final = tmp_path / "final.mkv"
    _sine(master, 2.0)
    _silent_video(silent, 2.0)
    out, report = mux_master_audio(str(silent), str(master), str(final), fps=25)
    text = "\n".join(report)
    assert out == str(final)
    assert "audio_mode=master_copy" in text
    assert "audio_mode=sfx_mixed" not in text
    assert "sfx_bed_path" not in text
    assert "sfx_gain" not in text


def test_master_audio_mux_is_output_node():
    assert OTRMasterAudioMux.OUTPUT_NODE is True
    it = OTRMasterAudioMux.INPUT_TYPES()
    assert "silent_video_path" in it["required"] and "master_audio_path" in it["required"]
    assert "audio_done" in it["optional"]            # gate mirrors audio_done


def test_master_audio_mux_default_out_peels_credits_suffix(monkeypatch, tmp_path):
    """Credits enrichment 2026-07-03: OTR_CreditsRoll appends "_with_credits" to
    node 93's "_procgen_blended" stem before the mux. _default_out must peel ALL
    the post-chain suffixes so the final lands in the episode's OWN folder
    otr/episodes/<ep>/ (§6 canonical-path contract), not a spurious
    <...>_procgen_blended_with_credits/ dir."""
    import types
    monkeypatch.setitem(sys.modules, "folder_paths", types.SimpleNamespace(
        get_output_directory=lambda: str(tmp_path)))
    node = OTRMasterAudioMux()
    got = node._default_out(str(
        tmp_path / "ep042_silent_procgen_blended_captioned_with_credits.mp4"))
    ep_dir = os.path.join(str(tmp_path), "otr", "episodes", "ep042")
    assert os.path.dirname(got) == ep_dir, got
    # legacy (no credits) input still resolves to the same episode folder
    got2 = node._default_out(str(tmp_path / "ep042_procgen_blended.mp4"))
    assert os.path.dirname(got2) == ep_dir, got2


def test_master_audio_mux_default_out_uses_live_ledger_episode_authority(
        monkeypatch, tmp_path):
    """A live ledger owns the episode directory even when a future terminal
    enrichment adds a suffix that the mux has never seen before.

    The filename remains useful for the final asset name, but it must not be
    reinterpreted as a sibling episode id. A mismatched stale singleton is
    rejected by the episode-id prefix check.
    """
    import types
    from nodes import _otr_ledger as otr_ledger

    monkeypatch.setitem(sys.modules, "folder_paths", types.SimpleNamespace(
        get_output_directory=lambda: str(tmp_path)))
    episode_dir = tmp_path / "otr" / "episodes" / "ep042"
    ledger_path = episode_dir / "audio" / "ep042_ledger.json"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        otr_ledger, "in_flight_ledger_path", lambda: ledger_path)

    node = OTRMasterAudioMux()
    future_stem = (
        "ep042_silent_procgen_blended_captioned_future_fx_with_credits.mp4")
    got = pathlib.Path(node._default_out(str(tmp_path / future_stem)))
    assert got.parent == episode_dir
    assert got.name == future_stem[:-4] + "_final.mp4"

    # A stale in-flight pointer from a prior episode cannot redirect this run.
    stale_dir = tmp_path / "otr" / "episodes" / "ep_old"
    stale_ledger = stale_dir / "audio" / "ep_old_ledger.json"
    monkeypatch.setattr(
        otr_ledger, "in_flight_ledger_path", lambda: stale_ledger)
    got_stale = pathlib.Path(node._default_out(str(
        tmp_path / "ep099_silent_procgen_blended_captioned_with_credits.mp4")))
    assert got_stale.parent == tmp_path / "otr" / "episodes" / "ep099"


def test_master_audio_mux_terminal_stamp_owns_all_final_paths(
        monkeypatch, tmp_path):
    from nodes import _otr_ledger as otr_ledger

    ep_id = "ep_terminal"
    otr_root = tmp_path / "output" / "otr"
    episode_root = otr_root / "episodes" / ep_id
    ledger_path = episode_root / "audio" / f"{ep_id}_ledger.json"
    ledger_path.parent.mkdir(parents=True)
    final_audio = ledger_path.parent / f"{ep_id}_master.wav"
    final_video = episode_root / f"{ep_id}_with_credits_final.mp4"
    obs_video = otr_root / "obs" / final_video.name
    obs_video.parent.mkdir(parents=True)
    for path in (final_audio, final_video, obs_video):
        path.write_bytes(b"asset")
    ledger_path.write_text(json.dumps({
        "episode_id": ep_id,
        "final_audio_path": "",
        "final_video_path": "old-intermediate.mp4",
        "meta": {"paths": {"obs_final": str(otr_root / "obs" / f"{ep_id}.mp4")}},
    }), encoding="utf-8")
    monkeypatch.setattr(
        otr_ledger, "in_flight_ledger_path", lambda: ledger_path)

    status = OTRMasterAudioMux()._stamp_terminal_paths(
        str(final_video), str(obs_video), str(final_audio))
    saved = json.loads(ledger_path.read_text(encoding="utf-8"))

    assert "stamped ledger" in status
    assert saved["final_audio_path"] == str(final_audio)
    assert saved["final_video_path"] == str(final_video)
    assert saved["meta"]["obs_final_path"] == str(obs_video.resolve())
    assert saved["meta"]["paths"]["obs_final"] == str(obs_video.resolve())
    for key in ("final_audio_path", "final_video_path"):
        assert pathlib.Path(saved[key]).is_file()
    assert pathlib.Path(saved["meta"]["paths"]["obs_final"]).is_file()


def _publishable_episode(tmp_path, monkeypatch, ep_id):
    """An episode whose ledger CLEARS publication, wired as the in-flight one.

    Since 2026-08-15 (build contract D5a) the mux publishes to obs only when the
    ledger carries an eligibility receipt that says so, and it fails CLOSED --
    no ledger means no OBS copy. That is deliberate (a stale singleton must
    never publish one episode under another's rights), and it means a test about
    the PUBLISHED copy now has to supply the episode the copy belongs to.
    """
    from nodes import _otr_ledger as otr_ledger
    from nodes import _otr_publication_eligibility as pub

    ledger_path = (tmp_path / "out" / "otr" / "episodes" / ep_id / "audio"
                   / f"{ep_id}_ledger.json")
    ledger_path.parent.mkdir(parents=True)
    led = {"episode_id": ep_id, "lines": [], "meta": {"source_bank": "original"}}
    # Stamp through the real producer, so this fixture cannot drift away from
    # the receipt shape the mux actually reads.
    assert pub.stamp_publication_eligibility(led).eligible is True
    ledger_path.write_text(json.dumps(led), encoding="utf-8")
    monkeypatch.setattr(otr_ledger, "in_flight_ledger_path", lambda: ledger_path)
    return ledger_path


@needs_ffmpeg
def test_master_audio_mux_publishes_final_to_obs(tmp_path, monkeypatch):
    """OUTPUT HYGIENE (2026-06-09): the muxed FINAL episode mp4 is the
    deliverable and must ALSO land in <output>/otr/obs as the WATCHABLE copy
    (video stream untouched, audio AAC -- standard players reject raw
    PCM-in-MP4). The archival byte-identical PCM final stays in episodes/."""
    master = tmp_path / "master.wav"
    silent = tmp_path / "silent.mp4"
    _sine(master)
    _silent_video(silent)
    import types
    fake_fp = types.SimpleNamespace(
        get_output_directory=lambda: str(tmp_path / "out"))
    monkeypatch.setitem(sys.modules, "folder_paths", fake_fp)
    monkeypatch.delenv("OTR_OBS_DIR", raising=False)
    _publishable_episode(tmp_path, monkeypatch, "silent")
    node = OTRMasterAudioMux()
    final, status = node.mux(str(silent), str(master))
    assert final and os.path.isfile(final)
    obs = tmp_path / "out" / "otr" / "obs" / os.path.basename(final)
    assert obs.is_file(), "final mp4 was not published to otr/obs"
    assert "obs_publish OK" in status
    # the obs copy carries PLAYABLE aac audio; the archival final stays pcm.
    def _codecs(path, kind):
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", kind,
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, check=True).stdout.split()
        return out
    assert _codecs(obs, "a") == ["aac"]
    assert _codecs(final, "a")[0].startswith("pcm")
    # video stream is copied, not re-encoded (same codec both files).
    assert _codecs(obs, "v") == _codecs(final, "v")


# --------------------------------------------------------------------------- #
# OTR_SilentComposite -- always-silent canonical output (V-1)
# --------------------------------------------------------------------------- #
@needs_ffmpeg
def test_silent_composite_strips_audio_and_is_canonical(tmp_path):
    base = tmp_path / "base.mp4"
    out = tmp_path / "silent.mp4"
    _silent_video(base, 2.0, with_audio=True)        # base HAS audio
    assert count_audio_streams(str(base)) == 1
    silent, report = normalize_to_silent_canonical(
        str(base), str(out), w=320, h=240, fps=25,
    )
    # V-1: the composite is ALWAYS silent
    assert count_audio_streams(silent) == 0
    info = probe_video(silent)
    assert info.get("pix_fmt") == "yuv420p"
    assert int(info["width"]) % 2 == 0 and int(info["height"]) % 2 == 0


@needs_ffmpeg
def test_post_chain_zero_audio_then_mux_roundtrip(tmp_path):
    """End-to-end (CPU): base(+audio) -> SilentComposite (0 audio) -> MasterAudioMux
    -> final whose audio is byte-identical to the frozen master."""
    base = tmp_path / "base.mp4"
    silent = tmp_path / "silent.mp4"
    final = tmp_path / "final.mkv"
    _silent_video(base, 2.0, with_audio=True)
    normalize_to_silent_canonical(str(base), str(silent), w=320, h=240, fps=25)
    assert count_audio_streams(str(silent)) == 0      # post-chain zero audio

    # synth the master to the composite's ACTUAL duration so the 1/fps gate holds
    from nodes.otr_master_audio_mux import _probe_float
    dur = _probe_float(str(silent), "v:0")
    master = tmp_path / "master.wav"
    _sine(master, round(dur, 3))
    out, _r = mux_master_audio(str(silent), str(master), str(final), fps=25)
    assert count_audio_streams(out) == 1
    assert audio_pcm_sha(out) == audio_pcm_sha(str(master))


# --------------------------------------------------------------------------- #
# cheap radio-floor families -- registered, model-agnostic
# --------------------------------------------------------------------------- #
def test_cheap_families_registered():
    names = set(vreg.all_engine_names())
    # "visualizer" graduated to a real engine (eng_visualizer.py) 2026-06-18; it is
    # no longer a cheap floor family.
    # C0 2026-06-30: abstract + station_card retired -> assert the surviving floors.
    for fam in ("still_motion", "still_pan", "still_flat"):
        assert fam in names, f"cheap family {fam} not registered in the video registry"


def test_cheap_family_usable_and_role_filtered():
    # C2 (2026-06-30): eligibility is CAPABILITY. still_motion (text-only) fits
    # every surviving role, so it is usable there.
    assert vreg.assert_usable("still_motion", "character_video") == "still_motion"
    # the SHARED role filter (AS-1, capability) offers floors per their inputs
    descs = [
        {"engine_id": n, "roles": tuple(vreg.get_engine(n).roles),
         "required_inputs": tuple(getattr(vreg.get_engine(n), "required_inputs", ()))}
        for n in vreg.all_engine_names()
    ]
    cv = rc.filter_engines_for_role("character_video", descs)
    assert "still_motion" in cv                    # text-only floor fits by capability
    # rip-sfx-broll (2026-07-01): the input-poor roles are GONE; a dead role
    # token raises through the same shared filter path.
    with pytest.raises(rc.RoleCompatError):
        rc.filter_engines_for_role("retired_role_b", descs)


def test_cheap_families_cold_import_clean():
    """The render namespace + the new render nodes import NO heavy lib (V-12)."""
    code = (
        "import sys;"
        "import nodes._otr_video_engines;"            # triggers cheap_families register
        "import nodes._otr_video_engines.cheap_families;"
        "import nodes.otr_silent_composite;"
        "import nodes.otr_master_audio_mux;"
        "heavy=[m for m in ('torch','transformers','diffusers') if m in sys.modules];"
        "print('HEAVY', heavy); sys.exit(1 if heavy else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT),
                       capture_output=True, text=True)
    assert r.returncode == 0, f"heavy libs at import:\n{r.stdout}\n{r.stderr}"


def test_new_render_nodes_have_no_shortest_cmd():
    """Neither new render node passes -shortest to ffmpeg. The token may appear
    ONLY in MasterAudioMux's V-2 guard assert/comment, never in a cmd list."""
    comp = (REPO_ROOT / "nodes" / "otr_silent_composite.py").read_text(encoding="utf-8")
    # composite builds a cmd list; -shortest must not be an element of it
    assert '"-shortest"' not in comp.replace('assert "-shortest" not in cmd', "")


# --------------------------------------------------------------------------- #
# OTR_SilentComposite per-beat assemble (Chunk C): a frame-accurate CFR timeline
# from a clip manifest -- frame counts only, gap-filled from the floor/black,
# assembled length asserted == the audio-derived budget (pre-mux A/V guard).
# --------------------------------------------------------------------------- #
def test_plan_timeline_segments_clip_floor_black_and_total():
    manifest = {"clips": [
        {"shot_id": "s0", "target_frame_count": 20, "path": "/x/a.mp4",
         "exists": True, "engine_id": "humo"},
        {"shot_id": "s1", "target_frame_count": 30, "path": "", "exists": False,
         "engine_id": "still_motion"},                   # gap -> floor / black
        {"shot_id": "s2", "target_frame_count": 0, "path": "/x/c.mp4",
         "exists": True},                                # 0 frames -> skipped
    ]}
    segs, total = plan_timeline_segments(manifest, floor_available=True,
                                         floor_frames=200)
    assert total == 50 and len(segs) == 2
    assert segs[0]["source"] == "clip" and segs[0]["n_frames"] == 20
    assert segs[1]["source"] == "floor" and segs[1]["src_start_frame"] == 20
    # with no floor the gap beat degrades to black so the episode still assembles
    segs2, total2 = plan_timeline_segments(manifest, floor_available=False)
    assert total2 == 50 and segs2[1]["source"] == "black"


@needs_ffmpeg
def test_assemble_silent_timeline_frame_accurate_and_silent(tmp_path):
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    floor = tmp_path / "floor.mp4"
    _silent_video(a, 2.0)             # ~50 frames -> conformed to 20 (truncate)
    _silent_video(b, 0.4)             # ~10 frames -> conformed to 30 (hold last)
    _silent_video(floor, 3.0)
    manifest = {"fps": 25, "clips": [
        {"shot_id": "s0", "target_frame_count": 20, "path": str(a), "exists": True},
        {"shot_id": "s1", "target_frame_count": 30, "path": str(b), "exists": True},
    ]}
    out = tmp_path / "assembled.mp4"
    res, report = assemble_silent_timeline(manifest, str(floor), str(out),
                                           w=320, h=240, fps=25)
    # assembled length == the master/base length (the pre-mux A/V-sync guard);
    # the 20+30 beat clips sit at the head, the floor tail-fills to the base.
    assert abs(count_video_frames(str(out)) - count_video_frames(str(floor))) <= 2
    assert count_video_frames(str(out)) >= 50        # the per-beat clips are included
    assert count_audio_streams(str(out)) == 0        # V-1: always silent


# test_assemble_extends_to_floor_for_credits_tail_bug410 RETIRED (credits
# enrichment 2026-07-03). Under the silent-tail model the composite no longer
# extends PAST the master mix to the procgen floor to carry a credits scroll:
# the unified credits roll is a SILENT tail appended LATE by OTR_CreditsRoll.
# The composite now ends at the MASTER length (A/V-sync fill + looped-last-clip
# closing-theme backdrop; see test_plan_timeline_segments_positions_by_start_s_
# and_fills_to_master below). The looped-last-clip credits backdrop + silent
# append contract moved to tests/test_credits_roll_spec.py
# (test_backdrop_is_last_existing_clip_looped_never_black,
#  test_append_credits_extends_body_and_stays_silent).


@needs_ffmpeg
def test_assemble_gap_fills_missing_clip_from_floor(tmp_path):
    a = tmp_path / "a.mp4"
    floor = tmp_path / "floor.mp4"
    _silent_video(a, 1.0)
    _silent_video(floor, 5.0)
    manifest = {"fps": 25, "clips": [
        {"shot_id": "s0", "target_frame_count": 25, "path": str(a), "exists": True},
        {"shot_id": "s1", "target_frame_count": 25, "path": "", "exists": False},
        {"shot_id": "s2", "target_frame_count": 25, "path": str(a), "exists": True},
    ]}
    out = tmp_path / "asm.mp4"
    assemble_silent_timeline(manifest, str(floor), str(out), w=320, h=240, fps=25)
    assert abs(count_video_frames(str(out)) - count_video_frames(str(floor))) <= 2
    assert count_audio_streams(str(out)) == 0


@needs_ffmpeg
def test_silent_composite_node_assemble_mode_via_manifest(tmp_path):
    import json
    a = tmp_path / "a.mp4"
    floor = tmp_path / "floor.mp4"
    _silent_video(a, 1.0)
    _silent_video(floor, 3.0)
    manifest = {"fps": 25, "clips": [
        {"shot_id": "s0", "target_frame_count": 25, "path": str(a), "exists": True},
        {"shot_id": "s1", "target_frame_count": 15, "path": "", "exists": False},
    ]}
    out = tmp_path / "node_assembled.mp4"
    silent, report = OTRSilentComposite().composite(
        str(floor), canvas_w=320, canvas_h=240, fps=25,
        output_path=str(out), clip_manifest_json=json.dumps(manifest))
    assert silent == str(out) and "assemble" in report
    assert abs(count_video_frames(silent) - count_video_frames(str(floor))) <= 2
    assert count_audio_streams(silent) == 0


def test_plan_timeline_segments_positions_by_start_s_and_fills_to_master():
    # POSITION mode: beats placed by start_s, floor gap-fills head/gap/tail so the
    # assembled length == the master length (the +intro shift + closing theme).
    # THE CLOSING WINDOW IS PART OF A PRODUCTION-SHAPED MANIFEST (no-mirror
    # step 4, 2026-08-06). The tail may hold the last drama clip ONLY inside the
    # closing-theme window, so a fixture that wants the BUG-410 backdrop must
    # carry the window a real episode carries. Measured on 11 of 12 recent
    # shipped ledgers: each has exactly one music_close row in master_mix space,
    # and each mints a window whose end equals the master length.
    manifest = {"fps": 25,
                "closing_theme_frame_window": {"start": 340, "end": 1543},
                "clips": [
        {"shot_id": "s0", "target_frame_count": 50, "path": "/x/a.mp4",
         "exists": True, "start_s": 9.6},      # after a 9.6s floor intro -> frame 240
        {"shot_id": "s1", "target_frame_count": 40, "path": "/x/b.mp4",
         "exists": True, "start_s": 12.0},     # 10-frame inter-beat floor gap
    ]}
    segs, total = plan_timeline_segments(
        manifest, floor_available=True, floor_frames=2000,
        target_total_frames=1543, fps=25)      # ~61.7s master
    assert total == 1543                        # assembled == master length
    assert [s["source"] for s in segs] == ["floor", "clip", "floor", "clip", "clip"]
    assert segs[0]["n_frames"] == 240           # round(9.6*25) head intro
    assert segs[2]["n_frames"] == 10            # round(12.0*25) - (240+50) inter-beat gap
    # BUG-410: the closing tail now HOLDS the last drama clip as the backdrop for
    # the rolling credits (the 6/5 "credits over the scene" look), not the floor
    assert segs[-1]["source"] == "clip" and segs[-1]["path"] == "/x/b.mp4"
    assert sum(s["n_frames"] for s in segs) == 1543
    # no floor (black gap-fill) still reaches the master length
    segs2, total2 = plan_timeline_segments(
        manifest, floor_available=False, target_total_frames=1543, fps=25)
    assert total2 == 1543 and segs2[0]["source"] == "black"


def test_positioned_crossfades_partition_visible_timeline_without_duplication():
    """PBUG-20260721-17: render-work frames may overlap; output frames may not."""
    # Production-shaped: the closing tail holds the last clip only inside the
    # closing-theme window (no-mirror step 4). See the sibling test above.
    manifest = {"fps": 25, "total_target_frames": 538,
                "closing_theme_frame_window": {"start": 400, "end": 538},
                "clips": [
        {"shot_id": "open", "target_frame_count": 250,
         "frame_count": 250, "path": "/x/open.mp4", "exists": True,
         "start_s": 0.0, "engine_id": "abstract"},
        {"shot_id": "body", "target_frame_count": 138,
         "frame_count": 138, "path": "/x/body.mp4", "exists": True,
         "start_s": 9.5, "engine_id": "humo",
         "family": "audio_driven_face"},
        {"shot_id": "close", "target_frame_count": 175,
         "frame_count": 175, "path": "/x/close.mp4", "exists": True,
         "start_s": 14.5, "engine_id": "abstract"},
    ]}

    segments, total = plan_timeline_segments(
        manifest, target_total_frames=538, fps=25)

    by_shot = {}
    for segment in segments:
        by_shot.setdefault(segment.get("shot_id"), 0)
        by_shot[segment.get("shot_id")] += segment["n_frames"]
    assert total == 538
    assert sum(segment["n_frames"] for segment in segments) == 538
    assert by_shot == {"open": 238, "body": 124, "close": 176}
    assert sum(row["target_frame_count"] for row in manifest["clips"]) == 563

    qa = timeline_quality_report(manifest, segments)
    beats = {beat["shot_id"]: beat for beat in qa["beats"]}
    assert qa["delivered_frames_ok"] is True
    assert beats["open"]["overlap_trimmed_frame_count"] == 12
    assert beats["body"]["overlap_trimmed_frame_count"] == 14
    assert beats["close"]["overlap_trimmed_frame_count"] == 0
    assert beats["close"]["planned_visible_frame_count"] == 176
    assert beats["close"]["quality_status"] == "looped_fill"


@needs_ffmpeg
def test_positioned_assemble_reconciles_oversized_manifest_down_to_master(tmp_path):
    """The filesystem master cross-check may shrink a positioned timeline."""
    floor = tmp_path / "floor.mp4"
    master = tmp_path / "episode_master.wav"
    _silent_video(floor, 3.2)
    _sine(master, 2.1)
    manifest = {"fps": 25, "total_target_frames": 80,
                "timeline_total_frames": 80, "clips": [
        {"shot_id": "only", "target_frame_count": 80,
         "frame_count": 80, "path": str(floor), "exists": True,
         "start_s": 0.0, "engine_id": "abstract"},
    ]}
    out = tmp_path / "positioned.mp4"

    _, report = assemble_silent_timeline(
        manifest, str(floor), str(out), w=320, h=240, fps=25)

    assert count_video_frames(str(out)) == 53  # ceil(2.1 * 25)
    assert any("A/V-sync master reconcile: 80 -> 53" in row for row in report)


def test_title_reveal_resolves_early_and_holds_bug409():
    """BUG-LOCAL-409: the hero title decode/reveal must COMPLETE in the first
    fraction of its window and then HOLD solid -- not resolve only on the last
    frame (the operator saw the title scramble for the whole duration)."""
    from nodes.video_engine import _title_reveal_progress
    w0, me, frac = 0, 100, 0.4
    assert _title_reveal_progress(0, w0, me, False, frac) == 0.0
    assert _title_reveal_progress(20, w0, me, False, frac) < 1.0      # mid-reveal
    assert _title_reveal_progress(40, w0, me, False, frac) >= 1.0     # resolved by ~40%
    assert _title_reveal_progress(80, w0, me, False, frac) == 1.0     # held solid
    assert _title_reveal_progress(99, w0, me, False, frac) == 1.0     # solid right before POP
    assert _title_reveal_progress(50, w0, me, True, frac) == 1.0      # dock -> solid
    # the OLD full-window stretch would still read p==0.4 at frame 40; the new
    # pacing resolves strictly earlier than the linear whole-window progress
    assert _title_reveal_progress(40, w0, me, False, frac) > (40 - w0) / (me - w0)


def test_hero_title_text_draws_black_outline_for_readability():
    """The title-card hero text needs a real dark edge over moving CRT art."""
    from PIL import Image, ImageDraw
    from nodes.video_engine import (
        _draw_crt_overstrike_text, _load_font, CRT_GREEN,
    )

    bg = (80, 0, 80)
    im = Image.new("RGB", (360, 120), bg)
    draw = ImageDraw.Draw(im)
    _draw_crt_overstrike_text(
        draw, (18, 22), "SLANTED INK", fill=CRT_GREEN,
        font=_load_font(42), stroke_width=4, stroke_fill=(0, 0, 0),
    )
    px = im.load()
    black = green = 0
    for yy in range(im.height):
        for xx in range(im.width):
            val = px[xx, yy]
            black += int(val == (0, 0, 0))
            green += int(val == CRT_GREEN)
    assert black > 0, "hero title stroke must add black pixels"
    assert green > 0, "hero title fill must stay phosphor green"
