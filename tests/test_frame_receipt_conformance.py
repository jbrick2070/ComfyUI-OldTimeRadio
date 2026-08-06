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
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from nodes._otr_shared import cloud_media_canonical as _cmc
from nodes._otr_video_engines import acceptance as acc
from nodes._otr_video_engines import frame_contract as fc
from nodes._otr_video_engines import registry as reg

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
    asset = _stub_asset(tmp_path)
    monkeypatch.setattr(_cmc, "canonicalize_video", lambda *a, **k: asset)
    monkeypatch.setattr(_cmc, "extract_sfx_bed_from_provider_video",
                        lambda *a, **k: asset)
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
    above would pass by vacuum -- green because nothing ran. This pins the
    floor: the shipped roster is 30+ engines, and the four HuMo tiers and both
    WAN adapters are the ones the multi-clip rules actually interrogate.
    """
    names = _registered_engines()
    assert len(names) >= 30, names
    for required in ("humo", "humo_1.7B", "humo_1.7B_169", "humo_14B_169",
                     "ltx_video", "ltx_audio_in", "ltx_8gb",
                     "wan_ti2v", "wan_i2v", "fastwan_8gb"):
        assert required in names, required


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
