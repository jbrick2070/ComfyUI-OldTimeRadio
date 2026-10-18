"""Every path a workflow names is READ only inside the approved folders.

Registry review flagged 2.3.21 and 2.3.22 "policy-v0.6: arbitrary-file-read,
path-traversal" (versions API, include_status_reason=true): a path input -- or
a path riding inside a JSON input or the ledger -- reached ffmpeg, ``open()``
or ``json.load`` verbatim, so a graph sent to ComfyUI's unauthenticated
``/prompt`` could read any file on the box into an output it then fetched.

The approved folders (operator, 2026-10-18: "usually otr/episodes for most
files"): ``otr/episodes``, ``otr/obs``, ComfyUI's input and temp folders, and
``$OTR_EXTRA_OUTPUT_ROOTS``. The check is the one comfyui-darkroom shipped to
go from Flagged to Active: text refusals first, then containment lexically and
again after ``realpath``.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_paths as P  # noqa: E402
from nodes._otr_paths import OtrPathContractError  # noqa: E402


@pytest.fixture
def roots(tmp_path, monkeypatch):
    """An output tree, an input folder and an unrelated folder beside them."""
    out = tmp_path / "output"
    episodes = out / "otr" / "episodes" / "ep"
    episodes.mkdir(parents=True)
    (out / "otr" / "obs").mkdir(parents=True)
    other_pack = out / "some_other_pack"
    other_pack.mkdir()
    inp = tmp_path / "input"
    inp.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    monkeypatch.setenv("OTR_OUTPUT_DIR", str(out))
    monkeypatch.setenv("OTR_INPUT_DIR", str(inp))
    monkeypatch.delenv("OTR_OBS_DIR", raising=False)
    monkeypatch.delenv("OTR_EXTRA_OUTPUT_ROOTS", raising=False)
    return {"out": out, "episodes": episodes, "input": inp,
            "other_pack": other_pack, "outside": outside}


# ---------------------------------------------------------------- the helper

def test_a_file_in_an_episode_or_the_input_folder_is_read(roots):
    for folder in (roots["episodes"], roots["input"]):
        f = folder / "x.mp4"
        assert P.confine_to_read_roots(str(f), "video_path") == os.path.normpath(str(f))


def test_an_empty_value_means_use_the_default(roots):
    assert P.confine_to_read_roots("", "video_path") == ""


@pytest.mark.parametrize("where", ["outside", "other_pack"])
def test_a_file_outside_the_approved_folders_is_refused(roots, where):
    # Not even another pack's folder under output/: not this pack's to read.
    with pytest.raises(OtrPathContractError, match="outside every permitted read root"):
        P.confine_to_read_roots(str(roots[where] / "secret.txt"), "video_path")


def test_a_relative_path_anchors_at_the_output_folder_and_cannot_climb(roots):
    got = P.confine_to_read_roots("otr/episodes/ep/x.mp4", "video_path")
    assert got == os.path.normpath(str(roots["episodes"] / "x.mp4"))
    with pytest.raises(OtrPathContractError, match="traversal"):
        P.confine_to_read_roots("otr/episodes/../../../elsewhere/x", "video_path")


@pytest.mark.parametrize("value, match", [
    ("bad\x00name", "NUL"),
    (r"\\attacker-host\share\x.wav", "UNC"),
    ("//attacker-host/share/x.wav", "UNC"),
    ("file:///etc/passwd", "URL"),
])
def test_text_refusals_come_before_any_disk_access(roots, value, match):
    with pytest.raises(OtrPathContractError, match=match):
        P.confine_to_read_roots(value, "video_path")


@pytest.mark.skipif(os.name != "nt", reason="Windows path forms")
@pytest.mark.parametrize("value, match", [
    ("C:secret.txt", "drive-relative"),
    ("x.wav:hidden", "alternate data stream"),
])
def test_windows_only_path_forms_are_refused(roots, value, match):
    with pytest.raises(OtrPathContractError, match=match):
        P.confine_to_read_roots(value, "video_path")


def test_a_link_inside_a_root_that_leads_out_of_it_is_refused(roots):
    target = roots["outside"] / "secret.txt"
    target.write_text("x", encoding="utf-8")
    link = roots["episodes"] / "innocent.txt"
    try:
        os.symlink(str(target), str(link))
    except (OSError, NotImplementedError):
        pytest.skip("this account cannot create symlinks")
    with pytest.raises(OtrPathContractError, match="leads through a link"):
        P.confine_to_read_roots(str(link), "video_path")


@pytest.mark.skipif(os.name != "nt", reason="NTFS junctions are Windows-only")
def test_a_junction_inside_a_root_that_leads_out_of_it_is_refused(roots):
    # A junction needs no privilege, so it is the link a real attacker uses.
    import _winapi
    (roots["outside"] / "secret.txt").write_text("x", encoding="utf-8")
    junction = roots["episodes"] / "looks_local"
    _winapi.CreateJunction(str(roots["outside"]), str(junction))
    try:
        with pytest.raises(OtrPathContractError, match="leads through a link"):
            P.confine_to_read_roots(str(junction / "secret.txt"), "video_path")
    finally:
        os.rmdir(str(junction))


def test_the_predicate_never_raises_and_says_no(roots):
    assert P.is_readable_in_roots(str(roots["episodes"] / "x.mp4"))
    for value in (str(roots["outside"] / "x"), r"\\host\share\x", "a\x00b", None):
        assert P.is_readable_in_roots(value) is False


def test_voice_references_may_also_live_under_a_models_root(roots, tmp_path, monkeypatch):
    models = tmp_path / "models_root"
    (models / "TTS" / "refs").mkdir(parents=True)
    monkeypatch.setenv("OTR_COMFYUI_MODELS_ROOT", str(models))
    ref = models / "TTS" / "refs" / "a.wav"
    assert P.is_readable_in_roots(str(ref), roots=P.voice_read_roots())
    assert not P.is_readable_in_roots(str(ref))  # not an ordinary read root


@pytest.mark.parametrize("bad", ["C:", "D:evil", "ep:stream"])
def test_an_episode_id_with_a_colon_is_refused(bad):
    with pytest.raises(OtrPathContractError, match="separator/traversal"):
        P._validate_episode_id(bad)


def test_a_portrait_hash_must_be_a_hex_digest():
    from nodes._otr_shared import portrait_ledger as pl
    for bad in ("../../x", "C:x", "a/b", "zz" * 32):
        with pytest.raises(pl.PortraitUnresolved):
            pl.portrait_path_for_hash(bad)


# ---------------------------------------------------------------- the nodes

def test_caption_burn_refuses_a_ledger_outside_the_folders(roots):
    from nodes import otr_caption_burn as burn_mod
    node = burn_mod.OTRCaptionBurn()
    with pytest.raises(OtrPathContractError, match="ledger_path"):
        node.burn(video_path=str(roots["episodes"] / "ep_silent.mp4"),
                  burn_captions=True,
                  ledger_path=str(roots["outside"] / "notes.json"))


def test_the_mux_refuses_a_master_outside_the_folders(roots):
    from nodes import otr_master_audio_mux as mux_mod
    node = mux_mod.OTRMasterAudioMux()
    with pytest.raises((OtrPathContractError, ValueError), match="master_audio_path"):
        node.mux(silent_video_path=str(roots["episodes"] / "ep_silent.mp4"),
                 master_audio_path=str(roots["outside"] / "master.wav"))


def test_the_composite_refuses_a_clip_outside_the_folders(roots):
    from nodes import otr_silent_composite as comp_mod
    node = comp_mod.OTRSilentComposite()
    manifest = json.dumps({"clips": [{"path": str(roots["outside"] / "clip.mp4")}]})
    with pytest.raises(OtrPathContractError, match="clip path"):
        node.composite(base_video_path=str(roots["episodes"] / "ep.mp4"),
                       clip_manifest_json=manifest)


def test_the_credits_roll_refuses_a_video_outside_the_folders(roots):
    from nodes import otr_credits_roll as cr
    node = cr.OTRCreditsRoll()
    with pytest.raises(OtrPathContractError, match="video_path"):
        node.roll(video_path=str(roots["outside"] / "ep.mp4"), clip_manifest_json="{}")


def test_the_render_batch_refuses_a_master_outside_the_folders(roots):
    from nodes.otr_video_render_batch import OTRVideoRenderBatch
    with pytest.raises(OtrPathContractError, match="master_audio_path"):
        OTRVideoRenderBatch().render(master_audio_path=str(roots["outside"] / "m.wav"))


def test_a_replay_bundle_outside_the_folders_is_refused(roots):
    from nodes import production_ledger as pl
    with pytest.raises(OtrPathContractError, match="outside every permitted read root"):
        pl.load_replay_manifest(str(roots["outside"]))


def test_a_voice_reference_outside_the_folders_resolves_to_nothing(roots):
    from nodes import _otr_voice_node_common as vc
    ref = roots["outside"] / "voice.wav"
    ref.write_bytes(b"RIFF")
    assert vc._resolve_ref_to_disk(str(ref)) is None


def test_an_engine_never_receives_a_ledger_path_outside_the_folders(roots):
    from nodes._otr_video_engines import render_driver as rd
    inside = str(roots["episodes"] / "still.png")
    assert rd._confined_ledger_path(inside, "init_image") == os.path.normpath(inside)
    assert rd._confined_ledger_path(str(roots["outside"] / "x.png"), "init_image") == ""
    assert rd._confined_ledger_path("", "audio_ref") == ""


def _humo_ledger(still, wav):
    """One live audio_driven_face shot whose still and line wav are ``still``
    and ``wav`` -- the two files the engine would open."""
    return {
        "audio": {"master_audio_sha256": "21aa71f6a4e5master", "ledger_frozen": True},
        "lines": [{"line_id": "b010", "char_id": "c07", "speaker_role": "character",
                   "start_s": 4.0, "dur_s": 2.4, "bark_wav_path": wav}],
        "images": {"images": [{"object_id": "c07", "path": still}]},
        "video": {"video_revision": 1, "fps": 25, "shots": [
            {"shot_id": "shot_b010", "source_line_ids": ["b010"],
             "role": "character_video", "engine_id": "humo",
             "family": "audio_driven_face", "group_id": "grp_char",
             "target_frame_count": 60, "degradation_trail": [],
             "creative": {"text_prompt": "c07 speaks"}}]},
    }


def test_the_request_builder_is_wired_to_the_guard(roots):
    # The WIRING, not the helper: a ledger naming files outside the approved
    # folders yields a request with neither, and one naming episode files
    # keeps both.
    from nodes._otr_video_engines import render_driver as rd
    out_led = _humo_ledger(str(roots["outside"] / "c07.png"),
                           str(roots["outside"] / "b010.wav"))
    req = rd.build_request_from_shot(out_led["video"]["shots"][0], out_led)
    assert not req["asset_refs"].get("init_image") and not req.get("audio_ref")
    still, wav = roots["episodes"] / "c07.png", roots["episodes"] / "b010.wav"
    in_led = _humo_ledger(str(still), str(wav))
    req = rd.build_request_from_shot(in_led["video"]["shots"][0], in_led)
    assert req["asset_refs"]["init_image"] == os.path.normpath(str(still))
    assert req["audio_ref"]["path"] == os.path.normpath(str(wav))


def test_a_soft_check_hands_back_the_spelling_it_checked(roots, tmp_path, monkeypatch):
    # Check-A/open-B (Cursor review): a relative value is CHECKED against the
    # output folder, so the caller must OPEN that same absolute path -- never
    # the raw value, which open() would resolve against the process CWD.
    from nodes._otr_video_engines import render_driver as rd
    monkeypatch.chdir(roots["outside"])
    got = rd._confined_ledger_path("otr/episodes/ep/still.png", "init_image")
    assert got == os.path.normpath(str(roots["episodes"] / "still.png"))
    assert os.path.isabs(got)


def test_the_dispatcher_copies_the_pool_path_it_trusted(roots, monkeypatch):
    from nodes import otr_image_gen_dispatcher as disp
    from nodes.otr_image_gen_dispatcher import _MIN_PNG_BYTES, _PNG_MAGIC
    still = roots["episodes"] / "pool.png"
    still.write_bytes(_PNG_MAGIC + b"\0" * _MIN_PNG_BYTES)
    monkeypatch.chdir(roots["outside"])
    # Same file in the CWD-relative spelling: must NOT be what comes back.
    (roots["outside"] / "otr" / "episodes" / "ep").mkdir(parents=True)
    (roots["outside"] / "otr" / "episodes" / "ep" / "pool.png").write_bytes(
        _PNG_MAGIC + b"\1" * _MIN_PNG_BYTES)
    got = disp._trusted_pool_source("otr/episodes/ep/pool.png", str(roots["out"]))
    assert got == os.path.normpath(str(still))
    assert disp._trusted_pool_source(str(roots["outside"] / "x.png"), str(roots["out"])) == ""


# ------------------------------------------- Sonnet QA follow-ups, 2026-10-18

def test_a_unc_hidden_behind_the_models_prefix_is_never_statted(monkeypatch):
    # `models/\host\share\x.wav` passes the text refusals; stripped, it is a
    # UNC path that os.path.join lets REPLACE the models root.
    from nodes._otr_audio_engines.base import resolve_voice_ref_path
    from nodes._otr_voice_node_common import _resolve_ref_to_disk
    statted = []
    real_exists = os.path.exists
    monkeypatch.setattr(os.path, "exists",
                        lambda p: statted.append(str(p)) or real_exists(p))
    b = chr(92)
    for ref in ("models/" + b + b + "attacker" + b + "share" + b + "x.wav",
                "models//attacker/share/x.wav", "models/C:/Windows/win.ini"):
        assert resolve_voice_ref_path(ref) == ""
        assert _resolve_ref_to_disk(ref) is None
    assert not [p for p in statted if "attacker" in p or "win.ini" in p]


def test_the_per_segment_voice_slice_reads_only_confined_audio():
    # The line wav and the master both feed ffmpeg in the slicers, BEFORE the
    # request-boundary guard: the confinement must sit at the source.
    import inspect
    from nodes._otr_video_engines import render_driver as rd
    src = inspect.getsource(rd.build_request_from_shot)
    assert "_confined_ledger_path(_voice_audio_for_line(line)" in src
    assert src.index("_confined_ledger_path(master_audio_path") < src.index(
        "_slice_master_audio(")


def test_a_master_outside_the_folders_is_never_sliced(roots, monkeypatch):
    from nodes._otr_video_engines import render_driver as rd
    master = roots["outside"] / "master.wav"
    master.write_bytes(b"RIFF")
    sliced = []
    monkeypatch.setattr(rd, "_slice_master_audio",
                        lambda path, *a, **k: sliced.append(path) or "")
    led = _humo_ledger(str(roots["episodes"] / "c07.png"), "")
    led["lines"][0].pop("bark_wav_path")
    rd.build_request_from_shot(led["video"]["shots"][0], led,
                               master_audio_path=str(master))
    assert sliced == []


def test_the_cloud_floor_refuses_a_still_outside_the_folders(roots, monkeypatch):
    from nodes._otr_video_engines import render_driver as rd
    from nodes._otr_video_engines.render_errors import RenderError
    still = roots["outside"] / "b003.png"
    still.write_bytes(b"\x89PNG")
    monkeypatch.setattr(rd, "_still_index", lambda _led: {"b003": str(still)})
    monkeypatch.setattr(rd, "render_shot", lambda *a, **k: pytest.fail("rendered"))
    with pytest.raises(RenderError, match="no clip"):
        rd._render_still_floor({"shot_id": "shot_b003", "source_line_ids": ["b003"]}, {})


@pytest.mark.parametrize("bad", ["../../x", "C:/Windows/win", "a/b", r"\\host\s\x", ""])
def test_a_kokoro_voice_id_is_a_bare_name(bad):
    from nodes._otr_audio_engines import eng_kokoro as k
    path = k._kokoro_voice_path(bad)
    assert os.path.dirname(path) == k._kokoro_voices_dir()
    assert os.path.basename(path) == "_refused_voice_id_.pt"
    assert os.path.basename(k._kokoro_voice_path("af_heart")) == "af_heart.pt"


@pytest.mark.skipif(os.name != "nt", reason="NTFS junctions are Windows-only")
def test_a_root_typed_through_a_junction_still_admits_its_real_spelling(
        tmp_path, monkeypatch):
    # The mux and caption burn hand on `.resolve()`d paths; a root typed
    # through a junction (or an 8.3 name, or a mapped drive) must still match.
    import _winapi
    real_out = tmp_path / "real_output"
    (real_out / "otr" / "episodes" / "ep").mkdir(parents=True)
    typed_out = tmp_path / "typed_output"
    _winapi.CreateJunction(str(real_out), str(typed_out))
    try:
        monkeypatch.setenv("OTR_OUTPUT_DIR", str(typed_out))
        monkeypatch.delenv("OTR_EXTRA_OUTPUT_ROOTS", raising=False)
        resolved = real_out / "otr" / "episodes" / "ep" / "x.mp4"
        assert P.confine_to_read_roots(str(resolved), "video_path")
        assert P.confine_to_read_roots(
            str(typed_out / "otr" / "episodes" / "ep" / "x.mp4"), "video_path")
        with pytest.raises(OtrPathContractError):
            P.confine_to_read_roots(str(real_out / "secret.txt"), "video_path")
    finally:
        os.rmdir(str(typed_out))


def test_the_configured_output_and_mesh_cache_are_read_roots(roots, tmp_path, monkeypatch):
    import sys
    import types
    configured = tmp_path / "configured_output"
    fake = types.ModuleType("folder_paths")
    fake.get_output_directory = lambda: str(configured)
    monkeypatch.setitem(sys.modules, "folder_paths", fake)
    pin = tmp_path / "mesh_pin"
    monkeypatch.setenv("OTR_MESH_CACHE_DIR", str(pin))
    assert P.is_readable_in_roots(str(configured / "otr" / "episodes" / "_shared" / "m.glb"))
    assert P.is_readable_in_roots(str(pin / "m.glb"))
    assert not P.is_readable_in_roots(str(configured / "other_pack" / "m.glb"))
