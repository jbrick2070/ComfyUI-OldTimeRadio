"""NO-MIRROR, step 1 -- every video producer declares how its frames got there.

This is the PRODUCER half of the no-mirror build, and it is deliberately driven
off the LIVE REGISTRY rather than a hand-written list of eleven adapters.

The reason is the defect this whole change exists to close. Eleven hand-written
per-adapter tests prove eleven adapters and say nothing about the twelfth, so
the day a new engine is registered the grader is armed against a producer that
never fires -- ``PBUG-20260805-04``, which is the fourth
armed-consumer-without-a-producer defect found in one week. A registry walk
fails BY NAME on an engine that has not been taught to answer, before it can
ship a beat nothing can grade.

WHAT EACH ENGINE OWES, and it is not the same for all of them (spec 7.3):

* EVERY delivered ``type == "video"`` clip owes ``extension_mode`` -- a string
  from ``acceptance.EXTENSION_MODES``. Silence is the one answer a padding lane
  and an honest one give identically.
* A BOUNDED engine -- ``frame_contract.can_split`` is True, meaning it has a
  ceiling a beat can overflow, so its beats get split into real segments --
  ALSO owes ``native_frame_count``. An UNBOUNDED lane (the procedural
  visualizers, the still families) can never be split, so it owes no
  native-count evidence and must not invent any: a flat still hold claiming
  every frame was "natively rendered" is an over-claim, and an over-claim is
  what a padded clip wants to make.
* ``type == "directory"`` is not a video row (``mesh_stage``), and owes nothing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from nodes._otr_shared import cloud_media_canonical as _cmc
from nodes._otr_video_engines import acceptance as acc
from nodes._otr_video_engines import frame_contract as fc
from nodes._otr_video_engines import registry as reg

_FFMPEG = shutil.which("ffmpeg")

#: A raw a local adapter's pure builder can shape. It carries the receipts a
#: real render would have stamped, so a builder that DROPS them on the floor --
#: the ``_clip_from_raw`` passthrough seam, which is how ``eng_ltx_8gb`` lost
#: its receipt originally -- fails here rather than in production.
_RAW = {"out_path": "probe.mp4", "frame_count": 50,
        "native_frame_count": 50, "extension_mode": "none"}

_REQUEST = {"shot_id": "shot_probe", "request_id": "req_probe",
            "canvas": {"w": 832, "h": 480, "fps": 25},
            "timing": {"target_frame_count": 50}}


def _stub_asset(tmp_path):
    return _cmc.CanonicalAsset(
        path=tmp_path / "provider.mp4", sha256="0" * 64, media_type="video",
        duration_s=2.0, width=832, height=480, fps=25.0, container="mp4",
        provider_job_id="job-probe")


def _make_av_fixture(tmp_path) -> Path:
    """A real 2s test clip with a tone, for the one engine that harvests
    audio out of its own provider file before the picture gets canonicalized
    (see the ``cloud_ltx25_foley_plus`` branch of ``_clip_for`` below)."""
    out = tmp_path / "provider_av.mp4"
    subprocess.run(
        [_FFMPEG, "-v", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=size=128x72:rate=25:duration=2",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(out)],
        check=True, capture_output=True, timeout=120)
    return out


def _clip_for(engine, tmp_path, monkeypatch):
    """This engine's delivered clip dict, however it happens to build one.

    Four shapes exist in the tree and all four are production paths: a local
    adapter's ``_clip_from_raw``, the cheap-family ``_floor_clip``,
    ``mesh_stage``'s ``_directory_clip``, and a provider ``canonicalize`` that
    shapes the dict itself. The provider lanes import their canonicalizer INSIDE
    the function, so patching the module attribute reaches them -- and it must,
    because a provider clip cannot be built without a real downloaded asset and
    these lanes cannot run on this box at all.

    ``_directory_clip`` is checked FIRST because ``mesh_stage`` also INHERITS
    ``_floor_clip`` from the cheap-family base without ever calling it. Asking
    the inherited helper would have tested a path this engine does not take and
    reported a video row where production ships a directory -- so the ordering
    here is what makes the "a directory clip owes nothing" exemption honest
    rather than an assumption.

    ``cloud_ltx25_foley_plus`` is a FIFTH shape: it harvests the provider's
    native audio bed out of the raw file BEFORE calling the parent
    ``canonicalize`` (docstring: "harvested BEFORE canonicalize_video strips
    the picture"), so its ``canonicalize`` reads ``raw["path"]`` straight off
    disk with real ffmpeg rather than going through the
    ``cloud_media_canonical.canonicalize_video`` seam every other provider
    lane uses. The generic stub raw (no real media file) 500s here with
    CORRUPT_OUTPUT, not because the engine forgot its receipt -- it never gets
    that far -- so this branch hands it a real fixture instead of skipping the
    engine outright.
    """
    directory = getattr(engine, "_directory_clip", None)
    if callable(directory):
        return directory(_REQUEST, str(tmp_path), 25, 50)
    builder = getattr(engine, "_clip_from_raw", None)
    if callable(builder):
        return builder(_RAW, _REQUEST)
    floor = getattr(engine, "_floor_clip", None)
    if callable(floor):
        return floor(_REQUEST, "probe.mp4", 25, 50)
    if getattr(engine, "name", None) == "cloud_ltx25_foley_plus":
        if not _FFMPEG:
            pytest.skip("ffmpeg not on PATH; cloud_ltx25_foley_plus "
                        "harvests real audio and cannot be probed without it")
        from nodes._otr_video_engines import foley_stems as fs
        monkeypatch.setattr(fs, "durable_foley_dir", lambda: str(tmp_path))
        src = _make_av_fixture(tmp_path)
        raw = {"path": str(src), "content_type": "video/mp4",
               "duration_s": None, "provider_job_id": "probe",
               "raw_meta": {}}
        asset = _stub_asset(tmp_path)
        monkeypatch.setattr(_cmc, "canonicalize_video", lambda *a, **k: asset)
        return engine.canonicalize(raw, _REQUEST, {})
    asset = _stub_asset(tmp_path)
    monkeypatch.setattr(_cmc, "canonicalize_video", lambda *a, **k: asset)
    return engine.canonicalize(dict(_RAW), _REQUEST, {})


def _registered_engines():
    names = []
    for engine_id in sorted(reg.all_engine_names()):
        try:
            reg.get_engine(engine_id)
        except Exception:                       # noqa: BLE001 -- dark scaffold
            continue
        names.append(engine_id)
    return names


@pytest.mark.parametrize("engine_id", _registered_engines())
def test_every_registered_engine_declares_how_its_frames_got_there(
        engine_id, tmp_path, monkeypatch):
    engine = reg.get_engine(engine_id)
    try:
        clip = _clip_for(engine, tmp_path, monkeypatch)
    except NotImplementedError:
        pytest.skip("%s is a dark scaffold and delivers no clip" % engine_id)
    assert isinstance(clip, dict), (
        "%s did not produce a clip dict" % engine_id)
    if str(clip.get("type") or "video") != "video":
        return                                  # a directory clip is not a video row

    mode = clip.get("extension_mode")
    assert mode in acc.EXTENSION_MODES, (
        "%s delivers a video clip declaring extension_mode=%r. Every delivered "
        "video row owes a mode from %r -- an engine that never says how its "
        "frames got there cannot be graded, and 'no receipt' is exactly what a "
        "lane that pads without saying so looks like."
        % (engine_id, mode, acc.EXTENSION_MODES))

    native = clip.get("native_frame_count")
    if not fc.can_split(engine):
        # Unbounded: no native-count evidence is owed, and none may be invented.
        return
    delivered = acc.frame_count(clip.get("frame_count"))
    assert acc.frame_count(native) is not None, (
        "%s is BOUNDED (frame_contract.can_split is True), so its beats are "
        "split into real segments and it owes native_frame_count -- got %r"
        % (engine_id, native))
    assert acc.frame_count(native) <= delivered, (
        "%s claims %r native frame(s) in a clip carrying %r. The count is "
        "EMITTED-scope, so it can never exceed the clip's own length; a native "
        "count above it is an impossible receipt, not a large one."
        % (engine_id, native, clip.get("frame_count")))


def test_the_registry_walk_actually_covered_the_engines_it_claims_to():
    """The parametrization is the test, so an EMPTY one must fail loudly.

    A registry that failed to import would collect zero cases and the file
    above would pass by vacuum -- green because nothing ran. DERIVED checks,
    not a hand-typed floor (a count rots the next time an engine is added or
    retired -- rip-sfx 2026-08-06 removed five): the live roster must be
    non-empty, equal the declared CAPABILITIES table, carry the named anchors
    the multi-clip rules actually interrogate, and carry NONE of the retired
    ids.
    """
    names = _registered_engines()
    assert names, "registry discovered no engines at all"
    assert set(names) == set(reg.CAPABILITIES), (
        "live roster and CAPABILITIES disagree: only-live=%s only-declared=%s"
        % (sorted(set(names) - set(reg.CAPABILITIES)),
           sorted(set(reg.CAPABILITIES) - set(names))))
    for required in ("humo", "humo_1.7B", "humo_1.7B_169", "humo_14B_169",
                     "ltx_8gb", "razzle_ltx_8gb", "ltx25_video",
                     "ltx25_native_audio_in_16gb"):
        assert required in names, required
    from nodes._otr_shared.public_engines import RETIRED_ENGINE_IDS
    assert not (set(names) & RETIRED_ENGINE_IDS), (
        "retired engine id(s) re-registered: %s"
        % sorted(set(names) & RETIRED_ENGINE_IDS))


def test_an_unbounded_lane_does_not_claim_native_frames_it_cannot_evidence():
    """The still and procedural families answer the mode and STOP.

    Stated as its own test because the omission is deliberate and looks like an
    oversight. ``still_flat`` holds ONE image for the whole beat: every frame is
    ffmpeg's output, so "all of them were rendered" is true of the encode and
    misleading about the picture. Under the 7.3 ruling an unbounded lane owes no
    native-count evidence, so it supplies none rather than the flattering answer.
    """
    for engine_id in ("still_flat", "still_pan", "viz_green", "viz_camera"):
        engine = reg.get_engine(engine_id)
        assert not fc.can_split(engine), engine_id
        clip = engine._floor_clip(_REQUEST, "probe.mp4", 25, 50) \
            if hasattr(engine, "_floor_clip") \
            else engine._clip_from_raw(_RAW, _REQUEST)
        assert clip.get("extension_mode") == "none", engine_id
        assert clip.get("native_frame_count") is None, engine_id
