"""CPU tests for the radio-floor ffmpeg render (cheap_families.render_clip).

render_clip produces the platform's silent bt709 / yuv420p clip with ffmpeg (no
heavy model), and it is proven end-to-end here on the build box (ffmpeg
present); the ffmpeg-running tests skip cleanly without it.

"The cheap families are the always-succeeds radio floor ... the fallback-chain
terminus the A-S6 chain humo -> humo_1.7B -> still_motion converges on" OPENED
THIS DOCSTRING and is no longer true, twice over. The chain was RIPPED on
2026-07-02 (no UNIVERSAL_FLOOR, no auto-default role, nothing degrades here),
and as of lanes 15-16 `still_motion` and `still_pan` REFUSE a missing still
rather than painting a black beat -- so three of the four families now fail
LOUD on the case the "always succeeds" claim was really about. `still_flat` is
the last one carrying the dark lavfi floor and that is lane 17's call.

UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys

import pytest

from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import cheap_families  # noqa: F401  (registers floor)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_HAS_FFMPEG = shutil.which("ffmpeg") is not None
_HAS_FFPROBE = shutil.which("ffprobe") is not None
# 2026-06-18: "visualizer" graduated from a cheap floor stub to the real procedural
# CRT engine (eng_visualizer.py) -- it is no longer a _CheapFamilyBase, so it is
# dropped from the cheap-family render matrix (covered by test_video_visualizer.py).
# C0 2026-06-30: "abstract" + "station_card" retired -> dropped from the matrix.
_FAMILIES = ("still_motion", "still_pan", "still_flat")


def _req(frames=6, w=96, h=64, fps=25, **extra):
    r = {"shot_id": "s1", "canvas": {"w": w, "h": h, "fps": fps},
         "timing": {"target_frame_count": frames}, "text_prompt": "test card"}
    r.update(extra)
    return r


def _probe(path, *entries):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=" + ",".join(entries),
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True)
    return out.stdout.split()


def _has_audio(path):
    a = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    return a.stdout.strip() != ""


# --- clip-dict contract + pure helpers (no ffmpeg) ------------------------- #
def test_floor_clip_contract_shape():
    eng = vreg.get_engine("still_motion")
    clip = eng._floor_clip(_req(), "C:/o.mp4", 25, 30)
    assert clip["has_audio"] is False and clip["pixel_format"] == "yuv420p"
    assert clip["color_primaries"] == clip["transfer"] == clip["matrix"] == "bt709"
    assert clip["engine_id"] == "still_motion"
    assert clip["family"] == "static_motion"
    assert clip["frame_count"] == 30 and clip["fps"] == 25 and clip["clip_id"] == "s1"


def test_still_flat_uses_static_cmd_pan_uses_motion(monkeypatch, tmp_path):
    """still_flat holds the still FLAT (ffmpeg_still_static_cmd: fit+pad, no crop)
    while still_motion pans it (ffmpeg_still_motion_cmd) -- the 'stills, no
    motion, no face-crop' contract. Asserts WHICH command each engine selects, with
    a real still present (no ffmpeg run -- the builders are stubbed)."""
    from nodes._otr_video_engines import wrapper_bridge as wb
    still = tmp_path / "s.png"
    still.write_bytes(b"\x89PNG\r\n\x1a\n")          # presence is all _still_path checks
    calls = {}
    monkeypatch.setattr(wb, "ffmpeg_still_static_cmd",
                        lambda *a, **k: calls.setdefault("static", a) or ["ffmpeg"])
    monkeypatch.setattr(wb, "ffmpeg_still_motion_cmd",
                        lambda *a, **k: calls.setdefault("motion", a) or ["ffmpeg"])
    monkeypatch.setattr(wb, "run_ffmpeg", lambda cmd: None)
    # M7 (2026-07-27): render_clip now ffprobes the emitted mp4 before
    # _floor_clip self-declares its contract. run_ffmpeg is stubbed here, so no
    # file is ever written and the real probe would raise on a missing path --
    # the proof is stubbed too, and ASSERTED, so that this test says out loud
    # that the proof ran rather than silently depending on it not existing.
    from nodes._otr_video_engines import cheap_families as cf
    monkeypatch.setattr(cf, "ffprobe_clip_fields", lambda p: {"probed": p})
    monkeypatch.setattr(cf, "validate_silent_clip_contract",
                        lambda fields, fps: calls.setdefault("proved", fields))
    # M7 COUNT half (2026-07-28): the frame count is read back off the file
    # too. Same reasoning as the line above -- no file exists on this stubbed
    # path, so the count proof is stubbed AND asserted rather than silently
    # depended upon. It returns a number the caller could not have computed,
    # so the engine stamping its own `n` instead would be visible here.
    monkeypatch.setattr(wb, "proven_frame_count",
                        lambda path, declared, **kw:
                        calls.setdefault("counted", declared) and 0 or 4242)
    req = _req(asset_refs={"init_image": str(still)})

    raw = vreg.get_engine("still_flat").render_clip(req)
    assert "static" in calls and "motion" not in calls   # flat = NO pan
    assert "proved" in calls, "the emitted clip was never proven"
    assert "counted" in calls, "the emitted clip's frame count was never proven"
    assert raw["frame_count"] == 4242, (
        "still_flat stamped %r -- its own requested count, not the proven one"
        % (raw["frame_count"],))
    calls.clear()
    raw = vreg.get_engine("still_motion").render_clip(req)
    assert "motion" in calls and "static" not in calls   # still_motion = the pan path
    assert "proved" in calls, "the emitted clip was never proven"
    assert "counted" in calls, "the emitted clip's frame count was never proven"
    assert raw["frame_count"] == 4242


def test_still_flat_registered_validated_all_roles():
    eng = vreg.get_engine("still_flat")
    assert eng.family == "static_image_gen"
    assert eng.commercial_clean is True
    assert getattr(eng, "accepts_still", False) is True   # coverage gate mints its still
    assert getattr(eng, "_still_motion", True) is False   # flat hold, not a pan
    for role in ("announcer_visual", "music_visual", "character_video"):
        assert role in eng.roles
    assert "still_flat" in vreg.all_engine_names()        # shows in the dropdown (C4)


def test_canvas_and_frame_defaults():
    eng = vreg.get_engine("still_motion")
    assert eng._canvas_dims({}) == (832, 480, 25)        # platform defaults
    assert eng._canvas_dims(
        {"canvas": {"w": 1280, "h": 720, "fps": 24}}) == (1280, 720, 24)
    assert eng._frame_count({}, 25) == 25                # 1s default when absent
    assert eng._frame_count({"timing": {"target_frame_count": 50}}, 25) == 50


def test_still_path_extraction():
    eng = vreg.get_engine("still_motion")
    assert eng._still_path({"asset_refs": {"init_image": "C:/p.png"}}) == "C:/p.png"
    assert eng._still_path({"asset_refs": {"still": {"path": "C:/s.png"}}}) == "C:/s.png"
    assert eng._still_path({"asset_refs": {}}) == ""


def test_uses_still_flags():
    assert vreg.get_engine("still_motion").uses_still is True
    assert vreg.get_engine("still_pan").uses_still is True
    assert vreg.get_engine("still_flat").uses_still is True


def test_still_pan_fits_all_roles_bug401():
    """BUG-LOCAL-401: still_pan is the fast 'just a still' pick and must be valid
    in EVERY video role -- it needs only text_prompt, which every role supplies.
    A missing role tag (music_visual) made
    music_video_model='still_pan' fail OTR_VideoDirector validation at execute
    even though a still is perfectly valid for a music beat."""
    from nodes._otr_shared import role_compat as rc
    eng = vreg.get_engine("still_pan")
    desc = {"engine_id": "still_pan", "roles": tuple(eng.roles),
            "required_inputs": tuple(eng.required_inputs)}
    for role in ("announcer_visual", "music_visual", "character_video"):
        assert rc.engine_fits_role(desc, role), f"still_pan must fit role {role!r}"


# --- real ffmpeg renders (the floor leaf) ---------------------------------- #
@pytest.mark.skipif(not (_HAS_FFMPEG and _HAS_FFPROBE), reason="ffmpeg not on PATH")
@pytest.mark.parametrize("name", _FAMILIES)
def test_each_family_renders_silent_clip(name, tmp_path):
    """Every cheap family emits a valid silent clip. THE SUBJECT IS THE CLIP
    CONTRACT -- yuv420p, exact frame count, no audio stream -- not whether a
    family will render from nothing.

    FIXED AT THE FIXTURE, lane 15 (2026-08-11). `still_motion` now sets
    `_require_still` (S8b-12(b): a missing still was painting a black beat), so
    this test's old "call render_clip with no asset_refs" shape made it assert
    the ABSENCE of that refusal for one family, which was never its point.
    Every still-consuming family is handed a real still; the families that
    synthesise their own floor are unchanged. Lane 8's rule: give the new gate
    what it wants, never weaken the gate to keep a proxy test green.
    """
    eng = vreg.get_engine(name)
    kwargs = {}
    if getattr(eng, "uses_still", False):
        still = tmp_path / "still.png"
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi",
                        "-i", "color=c=0x223344:s=80x80", "-frames:v", "1",
                        str(still)], check=True, capture_output=True)
        kwargs["asset_refs"] = {"init_image": str(still)}
    clip = eng.render_clip(_req(frames=6, **kwargs), None)
    p = pathlib.Path(clip["path"])
    try:
        assert p.exists() and p.stat().st_size > 0
        assert clip["frame_count"] == 6 and clip["has_audio"] is False
        fields = _probe(p, "nb_read_frames", "pix_fmt")
        assert "yuv420p" in fields and "6" in fields
        assert not _has_audio(p)
        assert eng.canonicalize(clip, _req(), {}) is clip   # identity
    finally:
        p.unlink(missing_ok=True)


@pytest.mark.skipif(not (_HAS_FFMPEG and _HAS_FFPROBE), reason="ffmpeg not on PATH")
@pytest.mark.parametrize("name", ("still_flat",))
def test_the_no_still_LAVFI_FLOOR_still_renders_for_the_families_that_keep_it(name):
    """The coverage the lane-15 fixture fix would otherwise have cost.

    `test_each_family_renders_silent_clip` stages a real still for every
    still-consuming family -- correct, because the refusing families need one.
    That stopped driving `render_clip`'s `else:` branch (the synthesized dark
    lavfi floor) end to end, so it gets its own case here.

    THE PARAMETER LIST IS THE LIVE SCOPE OF THE DARK FLOOR, and it shrinks as
    lanes rule: `still_pan` was here until lane 16 gave it `_require_still`
    (2026-08-11), leaving `still_flat` as the LAST family that still paints a
    floor rather than refusing. When lane 17 rules, this test goes away with the
    branch -- and if this list ever empties while the `else:` branch survives,
    that branch is dead code and should be deleted, not left as a floor nobody
    reaches.
    """
    eng = vreg.get_engine(name)
    clip = eng.render_clip(_req(frames=6), None)          # no asset_refs
    p = pathlib.Path(clip["path"])
    try:
        assert p.exists() and p.stat().st_size > 0
        assert clip["frame_count"] == 6 and clip["has_audio"] is False
        fields = _probe(p, "nb_read_frames", "pix_fmt")
        assert "yuv420p" in fields and "6" in fields
        assert not _has_audio(p)
    finally:
        p.unlink(missing_ok=True)


def test_ffmpeg_is_gated_at_PREFLIGHT_for_every_cheap_family(monkeypatch):
    """S8b-12(a), lane 15 (2026-08-11) -- the SHARED-BASE ffmpeg gate.

    `assert_usable` used to `return self.name` unconditionally, under a comment
    saying the real check runs in `render_clip`. That is true and is the whole
    problem: `render_clip` runs mid-beat, after the writer, the TTS, the master
    freeze and the stills are already paid for. Every viz_* lane gates ffmpeg at
    both boundaries; these four gated it at neither.

    Asserted over ALL FOUR families, because the fix is on `_CheapFamilyBase`
    and lesson L13 says a shared-mechanism fix must be shown to cover every
    adapter sharing it -- not just the lane that happened to find it.
    """
    from nodes._otr_shared import scope_draw as sd

    monkeypatch.setattr(sd, "find_ffmpeg", lambda *a, **k: "")
    for name in _FAMILIES + ("still_word",):
        eng = vreg.get_engine(name)
        with pytest.raises(vreg.EngineUnusable) as exc:
            eng.assert_usable(host_caps={}, profile={})
        assert exc.value.reason == vreg.EngineUsabilityReason.MISSING_MODEL
        assert "ffmpeg" in str(exc.value)
        assert name in str(exc.value)      # names ITSELF, not the base class


@pytest.mark.parametrize("name", ("still_motion", "still_pan"))
def test_the_refusing_families_REFUSE_a_missing_still_not_a_black_beat(name,
                                                                      tmp_path):
    """S8b-12(b) -- THE BLACK-BEAT DEFECT, per refusing family.

    Parametrized rather than duplicated: `still_pan` took the same refusal in
    lane 16 on the same evidence, and one body asserting both keeps the two
    lanes from drifting into two slightly different refusals.
    """
    eng = vreg.get_engine(name)
    assert eng._require_still is True
    with pytest.raises(RuntimeError) as exc:
        eng.render_clip(_req(frames=6), None)
    assert name in str(exc.value)
    assert "image phase" in str(exc.value).lower()

    # A still that is DECLARED but absent from disk must refuse too -- the
    # common shape of a failed mint is a path written into the ledger and never
    # produced.
    ghost = tmp_path / "never_minted.png"
    with pytest.raises(RuntimeError):
        eng.render_clip(
            _req(frames=6, asset_refs={"init_image": str(ghost)}), None)


def test_which_still_families_REFUSE_a_missing_still_is_pinned_per_lane():
    """The scope guard on S8b-12(b) -- and it has already earned its keep.

    Lane 15 wrote it asserting `still_motion` alone refused. Lane 16 gave
    `still_pan` the same refusal and THIS TEST WENT RED, which is precisely the
    job: a behaviour change spreading across the shelf must be an explicit act
    per lane, never a side effect, because the base default is shared.

    The state below is therefore the LEDGER of who has ruled, not an
    implementation detail:

      still_word    True   -- always did (Sprint B; a word card cannot be black)
      still_motion  True   -- lane 15, closing the black-beat defect
      still_pan     True   -- lane 16, same evidence, its own lane
      still_flat    False  -- LANE 17'S CALL, not yet made
      the BASE      False  -- so a new cheap family opts IN, never inherits it

    When lane 17 rules, update this list in the same commit. If the base default
    ever flips instead, that is a shelf-wide decision and it needs its own
    argument -- not a quiet edit here.
    """
    from nodes._otr_video_engines.cheap_families import _CheapFamilyBase

    ruled = {"still_word": True, "still_motion": True, "still_pan": True,
             "still_flat": False,
             # mesh_stage extends this base for the frame contract and the
             # canvas/still helpers, but OVERRIDES render_clip completely (hy3d
             # -> Blender), so it never reaches the branch `_require_still`
             # guards and the flag is INERT there. It refuses a missing still by
             # its own FileNotFoundError instead -- asserted below, because
             # "the flag does not apply here" is only safe if something else
             # does the refusing.
             "mesh_stage": False}

    assert _CheapFamilyBase._require_still is False
    for name, expected in ruled.items():
        assert vreg.get_engine(name)._require_still is expected, name

    import inspect
    mesh_src = inspect.getsource(type(vreg.get_engine("mesh_stage")).render_clip)
    assert "_require_still" not in mesh_src            # the flag is unread there
    assert "FileNotFoundError" in mesh_src             # and it refuses anyway

    # GENERIC over the registry, not just the four named above: a NEW cheap
    # family must appear in `ruled` before it ships, so nobody can add one that
    # silently inherits (or hand-sets) a refusal without a lane ruling on it.
    live = {n for n in vreg.all_engine_names()
            if isinstance(vreg.get_engine(n), _CheapFamilyBase)}
    unruled = sorted(live - set(ruled))
    assert not unruled, (
        "cheap families with no recorded ruling on _require_still: %s. Add "
        "each to `ruled` in the lane packet that decides it -- the point of "
        "this test is that the decision is explicit per lane, and a family "
        "nobody ruled on is exactly what it exists to catch." % unruled)


@pytest.mark.skipif(not (_HAS_FFMPEG and _HAS_FFPROBE), reason="ffmpeg not on PATH")
def test_still_motion_over_a_real_still(tmp_path):
    still = tmp_path / "still.png"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=0x223344:s=80x80",
                    "-frames:v", "1", str(still)], check=True, capture_output=True)
    eng = vreg.get_engine("still_motion")
    clip = eng.render_clip(_req(frames=8, asset_refs={"init_image": str(still)}), None)
    p = pathlib.Path(clip["path"])
    try:
        assert p.exists() and clip["frame_count"] == 8
        fields = _probe(p, "nb_read_frames", "pix_fmt")
        assert "yuv420p" in fields and "8" in fields
        assert not _has_audio(p)
    finally:
        p.unlink(missing_ok=True)


# --- cold-import + ASCII / no BOM ------------------------------------------ #
def test_cheap_families_cold_import_no_heavy_libs():
    code = ("import sys;"
            "import nodes._otr_video_engines.cheap_families;"
            "heavy=[m for m in ('torch','transformers','diffusers','numpy') "
            "if m in sys.modules];"
            "print('HEAVY', heavy); sys.exit(1 if heavy else 0)")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT),
                       capture_output=True, text=True)
    assert r.returncode == 0, f"heavy libs at import:\n{r.stdout}\n{r.stderr}"


def test_cheap_families_source_ascii_no_bom():
    p = REPO_ROOT / "nodes" / "_otr_video_engines" / "cheap_families.py"
    raw = p.read_bytes()
    assert raw[:3] != b"\xef\xbb\xbf"
    src = raw.decode("utf-8")
    assert chr(0x2014) not in src
    src.encode("ascii")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
