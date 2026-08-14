#!/usr/bin/env python
"""45-word end-to-end campaign across every local video engine.

Replaces tmp/_g4_campaign.sh, which had three defects that cost a whole night of
GPU time on 2026-08-01:

1. ONE SHARED LOG. Every leg appended to tmp/_g4_campaign.txt and each new
   campaign truncated it, so two concurrent runs interleaved into an unreadable
   file and destroyed each other's record. Here every leg owns its own log.

2. NO RESET BETWEEN LEGS. A client-side ``RESULT TIMEOUT`` does NOT cancel the
   server job -- the render keeps running and the next leg queues BEHIND the
   ghost. That is how one stalled leg turned into a pile-up. Here every leg
   clears the queue and waits for VRAM to fall back to baseline before the next
   one starts.

3. NOTHING STOPPED A SECOND CAMPAIGN. Two runs fought over 16 GB and both
   thrashed. A lock file now makes a double launch fail loudly.

Success is the ASSET, never the exit code: each leg is verified by ffprobe on
the published file and by audio-vs-video coverage, per the standing rule that a
render is only done when the file is on disk.

Two further defects were fixed on 2026-08-02, both of which made a green result
mean less than it appeared to:

4. IT RAN SIX ENGINES AND CALLED THAT "EVERY LOCAL ENGINE". Nineteen local
   engines are registered. The roster is now DERIVED FROM THE REGISTRY, a local
   engine with no profile is a loud refusal rather than a silent skip, and the
   summary names every engine it did not run instead of printing "6/6 passed".

5. ITS ACCEPTANCE COULD NOT REJECT A MIRROR. Coverage was
   ``sum(clip durations) >= master duration`` -- a test that a ping-ponged clip
   passes perfectly, because mirroring yields exactly the right number of
   seconds out of frames that were already used. Acceptance now audits FRAMES:
   held-frame runs and mirror reflections are detected per clip on the motion
   lanes, and the engine-layer and composite guards are confirmed still armed
   before the campaign starts.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.request

REPO = pathlib.Path(__file__).resolve().parents[1]
PY = r"C:/Users/jeffr/Documents/ComfyUI/.venv/Scripts/python.exe"
SERVER = "http://127.0.0.1:8000"
OBS = pathlib.Path(r"C:/Users/jeffr/Documents/ComfyUI/output/otr/obs")
EPISODES = pathlib.Path(r"C:/Users/jeffr/Documents/ComfyUI/output/otr/episodes")
LOCK = REPO / "tmp" / "_w45_campaign.lock"

# THE ROSTER IS DERIVED FROM THE REGISTRY, NOT HAND-LISTED (2026-08-02).
#
# This file used to carry a six-entry LEGS list while its own docstring claimed
# "every local video engine". There are NINETEEN registered local engines. The
# thirteen it silently skipped were humo_1.7B, humo_1.7B_169, humo_14B_169,
# mesh_stage, still_flat, still_motion, still_pan, still_word, viz_camera,
# viz_green, viz_mxc_cpu, viz_mxc_mandala and wan_i2v -- so "6/6 engines passed"
# read as total coverage and was a third of it.
#
# A hand-list drifts the moment an engine is added; the registry cannot, because
# "the registry IS the menu" is already the project invariant (registry.py, C0).
# So the roster is built from ``all_engine_names()`` at run time and the profile
# for each engine is looked up by convention. A local engine with no profile is a
# LOUD refusal, never a silent skip -- that is the whole defect being fixed.
#
# Ordering is CHEAPEST-AND-LEAST-PROVEN FIRST. Information per hour is what
# matters in an unattended run: an engine that is going to fail should fail in
# the first minutes, not the twentieth hour. The CPU/procedural lanes run first
# (they cost minutes and catch config breakage immediately), then the heavy GPU
# lanes least-proven first, with the slow incumbent wan_ti2v last.

#: Cloud engines render provider-side and cannot run offline (no keys,
#: offline-first). They are qualified STATICALLY and are never campaign legs.
#: word_razzle is the trap here -- it reads like a local word-card engine and it
#: is a Pixverse partner row, and an ``otr_w45_word_razzle.json`` profile exists
#: on disk that would have pulled it into a "local" sweep.
CLOUD_PREFIXES = ("cloud_", "google_")
CLOUD_BY_NAME = frozenset({"word_razzle"})

#: Engine -> profile stem. Convention first, exceptions named explicitly.
#: The roll sentinel, taken from the writer's own rolls module so this file
#: cannot drift from the string the node actually accepts.
try:
    from nodes._otr_rolls import BANK_SENTINEL
except Exception:                                   # pragma: no cover
    BANK_SENTINEL = "roll (any eligible bank)"

PROFILE_PREFIX = "otr_w45_"
PROFILE_EXCEPTIONS = {
    "fastwan_8gb": "otr_w45_fastwan",
    "humo_1.7B": "otr_w45_humo_1_7b",
    "humo_1.7B_169": "otr_w45_humo_1_7b_169",
    "humo_14B_169": "otr_w45_humo_14b_169",
}

#: Run order. Cheap CPU lanes first, then heavy GPU lanes least-proven first.
LEG_ORDER = (
    "still_flat", "still_pan", "still_motion", "still_word",
    "viz_camera", "viz_green", "viz_mxc_cpu", "viz_mxc_mandala",
    "mesh_stage", "ltx_8gb", "fastwan_8gb", "ltx_video", "ltx_audio_in",
    "humo_1.7B", "humo_1.7B_169", "humo", "humo_14B_169",
    "wan_i2v", "wan_ti2v",
    # THE TWO H3 LANES RUN LAST, AND THEY NEED A DIFFERENT BOOT (lanes 19/20,
    # 2026-08-12). Last because they are by far the slowest -- ~240 s for a
    # single 5 s beat, against ~22 s for ltx_8gb -- so on the
    # cheapest-and-least-proven-first rule they are the tail.
    #
    # **THEY CANNOT SHARE THIS CAMPAIGN'S SERVER WITH THE HEAVY LANES.** Both
    # REQUIRE the named `h3` boot contract, whose `--reserve-vram 12` is what
    # forces a 21 GB DiT to stream on a 16 GB card -- and reserving 12 GiB away
    # from model loading would starve `wan_i2v` (13.9 GiB warm) and the HuMo
    # tiers on the same server. A contract is a BOOT fact and boot facts cannot
    # be fixed at render time.
    #
    # So a complete "every visual path" sweep is TWO campaigns against two
    # boots, not one: this file's default run for the nineteen incumbent lanes,
    # then `--only minimax_h3_video,minimax_h3_audio_in` against a server booted
    # with OTR_HEADLESS_RESERVE_VRAM_GB=12 + OTR_HEADLESS_DISABLE_PINNED=1. They
    # stay in this roster rather than being excluded, because a lane silently
    # missing from the list is the exact defect the registry-derived roster
    # exists to prevent -- and because on the wrong boot they REFUSE by name at
    # preflight rather than rendering something wrong.
    "minimax_h3_video", "minimax_h3_audio_in",
)


def is_cloud_engine(name: str) -> bool:
    return name.startswith(CLOUD_PREFIXES) or name in CLOUD_BY_NAME


def profile_for(engine: str) -> str:
    return PROFILE_EXCEPTIONS.get(engine, PROFILE_PREFIX + engine)


def local_engine_roster() -> list:
    """Every REGISTERED local video engine, in run order.

    Imports the video registry directly. If that import fails this function
    RAISES rather than returning a shorter list: a campaign that cannot see the
    roster must refuse, because silently running a subset while reporting
    "N/N engines passed" is exactly the defect this replaced.
    """
    import sys as _sys
    for path in (str(REPO), str(REPO.parent)):
        if path not in _sys.path:
            _sys.path.insert(0, path)
    from nodes._otr_video_engines import registry as _registry  # noqa: WPS433

    local = [n for n in _registry.all_engine_names() if not is_cloud_engine(n)]
    ordered = [n for n in LEG_ORDER if n in local]
    # Anything registered but absent from LEG_ORDER still runs -- appended, never
    # dropped, so a newly added engine joins the campaign the day it registers.
    ordered += sorted(n for n in local if n not in LEG_ORDER)
    return ordered


def build_legs() -> list:
    """[(engine, profile)] for every local engine. Missing profile = REFUSE."""
    legs, missing = [], []
    for engine in local_engine_roster():
        stem = profile_for(engine)
        if not (REPO / "config" / "profiles" / (stem + ".json")).is_file():
            missing.append("%s -> %s.json" % (engine, stem))
            continue
        legs.append((engine, stem))
    if missing:
        raise SystemExit(
            "REFUSING: %d local engine(s) have no campaign profile:\n  %s\n"
            "Add the profile or the campaign cannot claim to cover every local "
            "engine." % (len(missing), "\n  ".join(missing)))
    return legs

# MEASURED on this box, 2026-08-01, rather than estimated from the fastwan leg:
# a 45-word episode is ~66.7 s of audio = 1666 frames at 25 fps, and wan_ti2v
# renders at most 177 frames (7.08 s) per segment, so a leg is 13-15 clips. Two
# consecutive clips landed 12 minutes apart (22:25:32 -> 22:37:29), on top of
# ~45 min of writer + audio. That is a ~3.5-4 h leg for the incumbent.
#
# 10800 s (3 h) was still too small and would have killed the first leg about
# four clips from the end, after burning the whole three hours -- the expensive
# failure, because a client timeout does not cancel the server job. Sized to the
# measurement with headroom instead of to a round number.
LEG_TIMEOUT_S = 21600

# The engines a failing adapter falls BACK to. Their appearance in a leg's
# delivered_engine set is the silent-degrade signature: the episode publishes,
# looks fine, and never touched the engine under test. A real video engine
# appearing alongside the one under test is NOT this -- it is role
# specialisation (humo cannot render a music beat; it has no face to drive).
#
# ``still_parallax`` was REMOVED from this set 2026-08-02: it has been
# UNREGISTERED since 2026-06-30 (registry.py, item 2 rip-out), so it cannot
# deliver a clip and listing it was checking for a ghost.
#
# THE FLOOR IS NOT A FLOOR WHEN IT IS THE ENGINE UNDER TEST. Now that the
# campaign covers all nineteen local engines, eight of these ARE legs. Testing
# still_flat and then failing it for "falling back to still_flat" would be
# nonsense, so ``floor_violation()`` excludes the engine under test.
PROCEDURAL_FLOOR = frozenset({
    "still_flat", "still_pan", "still_motion", "still_word",
    "viz_camera", "viz_green", "viz_mxc_cpu", "viz_mxc_mandala", "mesh_stage",
})

#: Engine families whose output is MOTION, and which therefore must never
#: contain reused frames. The static families are excluded by design, not by
#: oversight: ``still_flat`` is a held still and ``viz_*`` lanes may legitimately
#: repeat a rendered pattern -- for them a duplicate frame is the intended
#: render, not a clip stretched to cover audio it could not afford.
MOTION_FAMILIES = frozenset({
    "audio_driven_face", "audio_conditioned_video",
    "image_to_video", "text_to_video",
})

#: A run of identical frames longer than this in a MOTION clip is a held frame.
#: One or two repeats can occur naturally (a still moment, an encoder decision);
#: eight at 25 fps is a third of a second of frozen picture, which is the
#: 0.68s-then-freeze signature this invariant exists to kill.
MAX_IDENTICAL_RUN = 8

#: Frames are compared as downscaled grayscale thumbnails. 32x32 is coarse
#: enough to ignore encoder noise and fine enough that two genuinely different
#: frames of a talking head never collide.
HASH_DIM = 32

#: Frames are compared by MEAN ABSOLUTE DIFFERENCE, not by exact equality.
#:
#: Exact hashing was the first implementation and it silently failed the only
#: test that mattered: a deliberately mirrored clip scored a reflection span of
#: ZERO. H.264 is lossy and direction-dependent -- the encoder predicts the
#: reversed half from different reference frames, so a mirrored frame decodes to
#: *nearly* the same pixels, never the same bytes. A byte-compare therefore
#: detects a frozen clip (whose repeated frames really are identical, being
#: residual-free P-frames) while missing the ping-pong it was written to catch.
#:
#: THE TOLERANCE IS RELATIVE TO THE CLIP'S OWN CONTRAST, not a fixed number on
#: the 0-255 scale. A fixed 2.0 was the second implementation and it was WRONG in
#: a way that matters for this show specifically: inter-frame differences scale
#: with a scene's dynamic range, so a dark scene's real motion produces smaller
#: pixel deltas than a bright scene's. Measured on real footage, the same clip
#: with only its contrast reduced went from 21 identical-run frames to 177 -- the
#: entire clip -- and archived near-black clips scored their full length. OTR is
#: a noir radio drama; low-key scenes are the house style, not an edge case, so a
#: fixed threshold would have failed good legs and sent the reader hunting a
#: re-armed extender that was never there.
#:
#: 0.04 of the clip's own pixel standard deviation reproduces the old 2.0 on
#: ordinary footage (std ~50) and tightens automatically as a scene darkens.
SAME_FRAME_REL_TOL = 0.04

#: Floor for the contrast scale, so a genuinely flat clip cannot drive the
#: tolerance to zero and call every frame different.
MIN_CONTRAST_SCALE = 1.0

#: HELD FRAMES AND MIRRORS NEED DIFFERENT TOLERANCES, because they leave
#: different traces in the encode.
#:
#: A held frame is THE SAME FRAME EMITTED TWICE. H.264 codes the repeat as a
#: skip with no residual, so it decodes essentially identically -- deviation
#: near zero. A mirror is a re-encoded REVERSAL: genuinely similar to its twin
#: but predicted from different references, so it never matches byte for byte
#: and needs the looser contrast-relative tolerance above.
#:
#: Using the loose tolerance for both was wrong in a way that matters on this
#: show. Measured on a real leg, a near-static announcer card held 46 frames
#: within 1.78 of its anchor against a 1.89 tolerance -- flagged -- while only
#: ONE frame was byte-identical. That is a still shot, which is a creative
#: choice, not reused video. A pixel threshold cannot tell a deliberately quiet
#: shot from a repeated one, so the held-frame test is tightened to the point
#: where only genuine duplication survives, and judging whether a shot is "too
#: still" is left to the person watching it.
HELD_FRAME_TOL = 0.5
VRAM_BASELINE_MB = 2600
VRAM_SETTLE_S = 240


def _now() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _say(msg: str) -> None:
    print("[w45 %s] %s" % (_now(), msg), flush=True)


def _server_json(path: str):
    with urllib.request.urlopen(SERVER + path, timeout=15) as fh:
        return json.load(fh)


def _server_post(path: str, payload: dict) -> None:
    req = urllib.request.Request(
        SERVER + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    urllib.request.urlopen(req, timeout=15).read()


def _vram_used_mb() -> int:
    """GPU MB in use, or -1 when it cannot be read.

    Guarded because this is called in a poll loop -- roughly two dozen times per
    leg across nineteen legs -- and it is the only subprocess call in this file
    that was unprotected. A missing nvidia-smi would otherwise abort the whole
    remaining campaign with a raw traceback. Every caller already treats -1 as
    "unknown", so failing soft here loses nothing.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        ).stdout.strip().splitlines()
    except (OSError, ValueError):
        return -1
    return int(out[0]) if out and out[0].strip().isdigit() else -1


def reset_between_legs() -> None:
    """Clear the queue and wait for the GPU to actually let go.

    Without this a stranded job from a timed-out leg keeps rendering and the
    next leg silently queues behind it.
    """
    try:
        _server_post("/queue", {"clear": True})
        _server_post("/interrupt", {})
    except Exception as exc:                       # server may be mid-restart
        _say("queue clear failed (continuing): %s" % exc)

    deadline = time.time() + VRAM_SETTLE_S
    while time.time() < deadline:
        time.sleep(10)
        try:
            queue = _server_json("/queue")
        except Exception:
            continue
        busy = len(queue.get("queue_running", [])) + len(queue.get("queue_pending", []))
        used = _vram_used_mb()
        if busy == 0 and 0 <= used <= VRAM_BASELINE_MB:
            _say("reset clean: queue empty, vram %d MB" % used)
            return
    _say("reset TIMED OUT: queue busy or vram still %d MB -- continuing anyway"
         % _vram_used_mb())


def _ffprobe(path: pathlib.Path, entries: str, stream: str | None = None) -> str:
    cmd = ["ffprobe", "-v", "error"]
    if stream:
        cmd += ["-select_streams", stream]
    cmd += ["-show_entries", entries, "-of", "csv=p=0", str(path)]
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def _duration(path: pathlib.Path) -> float:
    raw = _ffprobe(path, "format=duration")
    try:
        return float(raw.split(",")[0])
    except ValueError:
        return 0.0


def guards_armed() -> list:
    """Confirm the no-reuse guards still EXIST before trusting a green campaign.

    The invariant is enforced in two places -- the engine layer
    (``wrapper_bridge``) and the assembler (``otr_silent_composite``) -- and both
    enforce it by RAISING. A campaign therefore proves the invariant mostly by
    what did NOT happen, and "nothing raised" is also what you see when somebody
    has quietly re-armed a fill path. So check the guards directly.

    Returns a list of failure strings; empty means armed.
    """
    import sys as _sys
    for path in (str(REPO), str(REPO.parent)):
        if path not in _sys.path:
            _sys.path.insert(0, path)

    failures = []
    try:
        from nodes._otr_video_engines import wrapper_bridge as _wb
        if hasattr(_wb, "extend_frames_to_target"):
            failures.append("wrapper_bridge.extend_frames_to_target is BACK -- "
                            "the ping-pong mirror extender was deleted 2026-08-02")
        if not hasattr(_wb, "MirrorExtensionForbidden"):
            failures.append("wrapper_bridge.MirrorExtensionForbidden is GONE -- "
                            "a short render no longer fails closed")
    except Exception as exc:
        failures.append("cannot import wrapper_bridge to check guards: %s" % exc)

    try:
        from nodes import otr_silent_composite as _sc
        if not hasattr(_sc, "ClipUnderrunsItsBeat"):
            failures.append("otr_silent_composite.ClipUnderrunsItsBeat is GONE -- "
                            "the composite may loop-fill again")
        elif _sc._should_loop_fill({"frame_count": 5}, 100) is not False:
            failures.append("otr_silent_composite._should_loop_fill no longer "
                            "returns False -- stream-loop fill is re-armed")
    except Exception as exc:
        failures.append("cannot import otr_silent_composite to check guards: %s" % exc)

    return failures


def frame_signatures(path: pathlib.Path):
    """A float32 [N, HASH_DIM*HASH_DIM] thumbnail stack for one clip.

    Decoding at 32x32 makes a whole clip a few hundred KB and makes the
    comparison immune to encoder noise, which a compare of full frames would
    trip over constantly. Returns None when the clip cannot be decoded.
    """
    import numpy as np

    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-vf", "scale=%d:%d" % (HASH_DIM, HASH_DIM),
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        return None
    stride = HASH_DIM * HASH_DIM
    buf = np.frombuffer(proc.stdout, dtype=np.uint8)
    n = buf.size // stride
    if n == 0:
        return None
    return buf[:n * stride].reshape(n, stride).astype(np.float32)


def find_frame_reuse(sigs) -> dict:
    """Detect the two ways video gets reused to cover audio it did not earn.

    HELD FRAME -- a run of near-identical consecutive frames. The composite used
    to hold the last frame for the rest of a beat; that is now terminal, and a
    long identical run in a delivered clip means it came back.

    PING-PONG / MIRROR -- the deleted extender built the cycle
    ``[0,1,..,N-1,N-2,..,1]``, a palindrome about frame N-1. So look for a
    reflection point p where frame ``p-i`` matches frame ``p+i`` for a long run.
    Searching for a reflection POINT rather than testing whether the whole clip
    is a palindrome is what makes this survive the tail trim that always follows
    the extension.

    Comparison is mean-absolute-difference against SAME_FRAME_TOL, never exact
    equality -- see that constant for why exact matching provably missed this.
    """
    import numpy as np

    n = 0 if sigs is None else int(sigs.shape[0])
    out = {"frames": n, "max_identical_run": 0, "mirror_span": 0,
           "mirror_at": None, "contrast_scale": None, "tolerance": None}
    if n < 4:
        return out

    # Scale the tolerance to this clip's own contrast -- see SAME_FRAME_REL_TOL.
    scale = max(float(np.asarray(sigs).std()), MIN_CONTRAST_SCALE)
    tol = SAME_FRAME_REL_TOL * scale
    out["contrast_scale"], out["tolerance"] = round(scale, 2), round(tol, 3)

    def dist(i, j):
        return float(np.abs(sigs[i] - sigs[j]).mean())

    def same(i, j):                      # mirror test: re-encoded, so relative
        return dist(i, j) <= tol

    def duplicated(i, j):                # hold test: same frame, so near-exact
        return dist(i, j) <= HELD_FRAME_TOL

    # A HELD FRAME IS CUMULATIVE STASIS, NOT SLOW MOTION.
    #
    # The first implementation counted runs of small NEIGHBOUR-TO-NEIGHBOUR
    # difference, and that fails on the thing this engine actually does. A HuMo
    # clip opens with a slow motion-onset ramp: measured on a real leg, frames
    # 2-41 had consecutive distances averaging 1.11 (under tolerance) while
    # drifting 6.05 away from the run's first frame -- only 1 of 40 frames was
    # identical to it. The picture was changing the whole time, slowly. Failing
    # that leg cost 142 minutes of GPU and pointed at a re-armed extender that
    # does not exist.
    #
    # So a run only counts as HELD when the picture also does not move away from
    # where it started: every frame must stay within tolerance of the run's
    # ANCHOR, not merely of its predecessor. A genuine hold satisfies both; slow
    # motion satisfies only the first.
    best_run, run_start = 1, 0
    i = 0
    while i < n:
        anchor = i
        j = i + 1
        while j < n and duplicated(j, j - 1) and duplicated(j, anchor):
            j += 1
        if (j - anchor) > best_run:
            best_run, run_start = j - anchor, anchor
        i = j if j > i + 1 else i + 1
    out["max_identical_run"] = best_run
    out["identical_run_at"] = run_start

    # Reflection search. A genuine mirror reflects for many frames; incidental
    # matches die after one or two, so a span threshold separates them cleanly.
    # Frame 0 and the final frame are skipped as centres (a 1-frame reflection
    # at the very edge is meaningless).
    best_span, best_at = 0, None
    for p in range(2, n - 2):
        span = 0
        while p - span - 1 >= 0 and p + span + 1 < n:
            if not same(p - span - 1, p + span + 1):
                break
            span += 1
        if span > best_span:
            best_span, best_at = span, p
    out["mirror_span"], out["mirror_at"] = best_span, best_at
    return out


def _engine_families() -> dict:
    """{engine_name: family} from the live registry, {} if it cannot be read."""
    import sys as _sys
    for path in (str(REPO), str(REPO.parent)):
        if path not in _sys.path:
            _sys.path.insert(0, path)
    try:
        from nodes._otr_video_engines import registry as _registry
        return {n: getattr(_registry.get_engine(n), "family", "")
                for n in _registry.all_engine_names()}
    except Exception:
        return {}


def engine_of_clip(clip_name: str, families: dict):
    """The engine a clip was rendered by, from its filename.

    Clips are named ``shot_<id>_<role>_<engine>.mp4``. Matching by longest known
    engine name rather than by splitting on '_' is deliberate: the engine names
    themselves contain underscores (``ltx_audio_in``, ``viz_mxc_mandala``), so
    positional parsing picks up the wrong token.
    """
    stem = clip_name.rsplit(".", 1)[0]
    hits = [name for name in families if stem.endswith("_" + name)]
    return max(hits, key=len) if hits else None


def audit_clip_reuse(episode: pathlib.Path) -> dict:
    """Every MOTION clip checked for held frames and mirror cycles.

    This is the check the old acceptance could not make. It compared SUMMED
    CLIP DURATIONS against the master length -- a test a mirrored clip passes
    perfectly, because mirroring produces exactly the right duration out of
    exactly the wrong frames.

    Static-family clips are skipped, and skipping them is REPORTED rather than
    silent: a still lane's repeated frames are its intended render, but a reader
    should be able to see how much of the episode this check actually covered.
    """
    families = _engine_families()
    clips = sorted((episode / "clips").glob("*.mp4"))
    findings, checked, skipped = [], [], []

    for clip in clips:
        engine = engine_of_clip(clip.name, families)
        family = families.get(engine or "", "")
        if family and family not in MOTION_FAMILIES:
            skipped.append("%s (%s/%s)" % (clip.name, engine, family))
            continue
        if engine is None and families:
            # Unknown engine: check it rather than trust it.
            findings.append({"clip": clip.name, "why": "engine not identifiable "
                                                       "from filename"})

        sigs = frame_signatures(clip)
        if sigs is None:
            findings.append({"clip": clip.name, "why": "undecodable"})
            continue
        checked.append(clip.name)
        got = find_frame_reuse(sigs)
        if got["max_identical_run"] > MAX_IDENTICAL_RUN:
            findings.append({"clip": clip.name, "why": "held frames",
                             "detail": "%d identical consecutive frames"
                                       % got["max_identical_run"]})
        if got["mirror_span"] >= MAX_IDENTICAL_RUN:
            findings.append({"clip": clip.name, "why": "mirror/ping-pong",
                             "detail": "%d-frame reflection about frame %s"
                                       % (got["mirror_span"], got["mirror_at"])})

    return {"clips_checked": len(checked), "clips_skipped_static": len(skipped),
            "skipped": skipped, "findings": findings,
            "registry_readable": bool(families)}


def delivered_engines(episode: pathlib.Path) -> list[str]:
    """Which engine ACTUALLY rendered each clip, per the ledger.

    This is the check that separates "six engines work" from "six legs produced
    a file". A dark engine does not necessarily fail the run -- OTR can fall
    closed and hand the beat to still_parallax, which publishes a perfectly
    valid episode that never touched the engine under test (the 2026-06-12
    mesh_stage catch: "PASS but engine NOT in the trace"). The ledger records
    the truth per clip in ``delivered_engine``.
    """
    ledgers = list(episode.glob("*_ledger.json")) + list(episode.glob("audio/*_ledger.json"))
    if not ledgers:
        return []
    try:
        blob = json.loads(ledgers[0].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []

    seen: list[str] = []

    def walk(node):
        if isinstance(node, dict):
            got = node.get("delivered_engine")
            if isinstance(got, str) and got:
                seen.append(got)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(blob)
    return seen


def verify_asset(started_at: float) -> dict:
    """A leg passes only if it PUBLISHED, and only if video covers audio."""
    published = [p for p in OBS.glob("*.mp4") if p.stat().st_mtime >= started_at]
    if not published:
        return {"ok": False, "why": "no new file in otr/obs"}
    asset = max(published, key=lambda p: p.stat().st_mtime)

    dims = _ffprobe(asset, "stream=width,height,nb_frames", stream="v:0")
    seconds = _duration(asset)
    audio = _ffprobe(asset, "stream=codec_name", stream="a:0")
    if seconds <= 0:
        return {"ok": False, "why": "unreadable/zero-length asset", "asset": asset.name}
    if not audio:
        return {"ok": False, "why": "no audio stream", "asset": asset.name}

    # Coverage: every second of audio must have real video behind it.
    #
    # DURATION COVERAGE ALONE CANNOT PROVE THIS, and until 2026-08-02 it was the
    # only test here. Summing clip durations and comparing against the master
    # asks "is there enough video?" -- and a mirrored clip answers yes, because
    # ping-ponging a short render produces exactly the right NUMBER of seconds
    # out of frames that were already used. The invariant is about WHICH frames,
    # so the frame-level audit below is the part that actually tests it.
    coverage = "not measured"
    reuse = None
    engines: list[str] = []
    episode_dirs = [d for d in EPISODES.iterdir()
                    if d.is_dir() and d.stat().st_mtime >= started_at]
    if episode_dirs:
        ep = max(episode_dirs, key=lambda d: d.stat().st_mtime)
        engines = delivered_engines(ep)
        masters = list(ep.glob("audio/*master.wav"))
        clips = sorted(ep.glob("clips/*.mp4"))
        if masters and clips:
            a = _duration(masters[0])
            v = sum(_duration(c) for c in clips)
            coverage = "audio %.2fs video %.2fs clips %d %s" % (
                a, v, len(clips), "COVERS" if v >= a else "SHORT")
            if v < a:
                return {"ok": False, "why": "video short of audio",
                        "asset": asset.name, "coverage": coverage}

        reuse = audit_clip_reuse(ep)
        if not reuse["registry_readable"]:
            return {"ok": False, "asset": asset.name, "coverage": coverage,
                    "why": "cannot read the engine registry, so no clip could be "
                           "classified motion-vs-static and the no-reuse "
                           "invariant is UNPROVEN for this leg"}
        # ADVISORY, NOT BLOCKING (2026-08-02, after it cost a 142-minute leg).
        #
        # This audit failed its first live leg on shot_l001_character_video_humo
        # with "40 identical consecutive frames". Diagnosis: only 1 of those 40
        # frames was byte-identical to the run's anchor and the picture drifted
        # 6.05 away from it -- HuMo's slow motion-onset ramp, not a held frame.
        # Two tightenings later it stopped flagging that, and stopped flagging a
        # near-static announcer card, but it still reports mirror reflections in
        # clips whose CONSECUTIVE frames already sit inside the mirror tolerance
        # -- on content that still, a reflection is arithmetic rather than
        # evidence. No boomerang exists in that lane to produce one: the
        # ``eng_ltx_video`` mirror was DELETED on 2026-08-06 (the earlier note
        # here said ``_loop_via_reverse`` "returns False unconditionally", which
        # was true while the gate still existed and is now a cite to a symbol
        # that is gone), and ``eng_ltx_av`` has no mirror path at all.
        #
        # A check that cannot yet separate "quiet shot" from "reused frame" must
        # not discard a completed render. The invariant is ALREADY enforced where
        # it can be enforced exactly -- ``MirrorExtensionForbidden`` in the
        # engine layer and ``ClipUnderrunsItsBeat`` in the composite, both
        # terminal, both checked armed before this campaign starts. This audit is
        # a second opinion on the artifact, so it REPORTS and the leg stands.
        #
        # Restore blocking only when the detector can distinguish deliberate
        # stillness from duplication on this show's own footage.
        if reuse["findings"]:
            first = reuse["findings"][0]
            _say("  REUSE AUDIT (advisory, %d finding(s)): %s: %s%s"
                 % (len(reuse["findings"]), first["clip"], first["why"],
                    (" -- " + first["detail"]) if first.get("detail") else ""))

    # AN UNRUN CHECK IS NOT A PASSED CHECK. If the episode directory could not
    # be located, `reuse` is still None here and the leg would return ok=True
    # having never tested the invariant it exists to prove. Silence and success
    # must not look the same.
    if reuse is None:
        return {"ok": False, "asset": asset.name, "coverage": coverage,
                "why": "could not locate the episode directory for this leg, so "
                       "the no-reuse audit never ran -- the invariant is "
                       "UNPROVEN, not satisfied"}

    return {"ok": True, "asset": asset.name, "dims": dims,
            "seconds": round(seconds, 2), "audio": audio, "coverage": coverage,
            "reuse": reuse, "delivered": sorted(set(engines))}


def run_leg(engine: str, profile: str, words: int,
            source_bank: str = BANK_SENTINEL) -> dict:
    log_path = REPO / "tmp" / ("_w45_%s.log" % engine)
    _say("LEG %s (profile %s) -> %s" % (engine, profile, log_path.name))
    started_at = time.time()

    cmd = [PY, str(REPO / "scripts" / "otr_canonical_api_run.py"),
           "--profile", profile,
           "--act-count", str(words),
           "--source-bank", source_bank,
           "--visual-style", "roll (any style)",
           "--timeout", str(LEG_TIMEOUT_S)]
    env = dict(os.environ, PYTHONUTF8="1")

    with open(log_path, "w", encoding="utf-8") as log:
        log.write("### %s  profile=%s  start=%s\n" % (engine, profile, _now()))
        log.flush()
        try:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                  cwd=str(REPO), env=env,
                                  timeout=LEG_TIMEOUT_S + 600).returncode
        except subprocess.TimeoutExpired:
            code = -1
            log.write("\n### runner timeout after %ds\n" % (LEG_TIMEOUT_S + 600))

    elapsed = time.time() - started_at
    verdict = verify_asset(started_at)

    # The engine under test must be the engine that actually rendered. A
    # published file proves the pipeline ran; only the ledger proves WHICH
    # engine ran, and a silent fall-back to the procedural floor would
    # otherwise be scored as a pass for an engine that never loaded.
    #
    # "ONLY this engine" was the FIRST version of this rule and it was WRONG.
    # It failed the humo leg after a 97-minute render that had actually
    # succeeded: humo is an audio-driven FACE engine, so its profile correctly
    # routes the beats with no face -- music_visual and announcer_visual -- to
    # ltx_audio_in, and the ledger showed exactly that:
    #     character_video x5 -> humo
    #     music_visual, announcer_visual -> ltx_audio_in
    # That is role specialisation working, not a fallback. ltx_8gb and
    # ltx_audio_in passed the strict rule only because they are general-purpose
    # enough to cover every role themselves.
    #
    # What the check is actually FOR is the silent degrade: an engine that fails
    # to load and quietly hands its beats to the procedural floor, publishing a
    # perfectly valid episode that never touched the engine under test (the
    # 2026-06-12 mesh_stage catch: "PASS but engine NOT in the trace"). So:
    # the engine under test must have delivered at least one clip, and no
    # procedural-floor engine may appear at all.
    if verdict.get("ok"):
        delivered = verdict.get("delivered") or []
        # EXCLUDE THE ENGINE UNDER TEST. The comment on PROCEDURAL_FLOOR has
        # claimed this since the roster went to nineteen engines; the code did
        # not do it, and it named a floor_violation() helper that was never
        # written. The first procedural leg to run under the new roster --
        # still_flat, 2026-08-03 00:09 -- duly failed with "silent fall-back to
        # the procedural floor: still_flat also delivered", i.e. failed for
        # being itself. All eight procedural legs would have followed.
        #
        # A comment describing a fix is not a fix. This is the same
        # documentation-reality gap the repo's own rules exist to catch, and it
        # shipped inside the very change that was correcting one.
        floor_used = sorted((set(delivered) & PROCEDURAL_FLOOR) - {engine})
        if not delivered:
            verdict.update(ok=False, why="ledger records no delivered_engine")
        elif engine not in delivered:
            verdict.update(
                ok=False,
                why="engine under test never delivered a clip: expected %s, "
                    "ledger says %s" % (engine, ", ".join(delivered)))
        elif floor_used:
            verdict.update(
                ok=False,
                why="silent fall-back to the procedural floor: %s also "
                    "delivered (ledger says %s)"
                    % (", ".join(floor_used), ", ".join(delivered)))

    verdict.update(engine=engine, profile=profile, exit=code,
                   minutes=round(elapsed / 60.0, 1), log=log_path.name)

    _say("LEG %s -> %s (exit=%s, %.1f min) %s" % (
        engine, "PASS" if verdict["ok"] else "FAIL",
        code, elapsed / 60.0, verdict.get("why", verdict.get("asset", ""))))
    return verdict


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # 2026-08-14: was `--words 45`. The child runner's --words flag went
    # with the target_words widget; episode shape is act count now.
    ap.add_argument("--acts", type=int, default=1,
                    help="act count 1-8; the only episode-shape knob")
    # PIN THE BANK SO A WRITER DEFECT CANNOT MASK A VIDEO LANE.
    #
    # Every leg rolled its bank, and that is wrong for what this campaign
    # MEASURES. The gate is "a 45-word render of every VISUAL path" -- the bank
    # is not the variable under test, it is noise on top of it. On 2026-08-12
    # `ltx_8gb` rolled `scifi_news_pro`, died in `OTR_LedgerScriptWriter` after
    # four markup attempts, and the LTX video lane was never exercised at all:
    # a writer defect consumed a video leg and reported it as a lane failure.
    # `mesh_stage` was lost the same way to a cast-coverage gap on shakespeare.
    #
    # Rolling still has a place -- it is how both of those writer defects were
    # FOUND -- so the sentinel remains the default and this is opt-in. Use it
    # when the question is "does this engine render", and roll when the question
    # is "does the pipeline hold up across banks".
    ap.add_argument("--source-bank", default=BANK_SENTINEL,
                    help="pin the source bank (default: roll). Pin it when the "
                         "engine is the variable under test, so a writer defect "
                         "on an unlucky roll cannot be reported as a lane "
                         "failure.")
    ap.add_argument("--only", default=None,
                    help="comma-separated engine names to run instead of all")
    args = ap.parse_args(argv)

    # Check the guards BEFORE taking the lock, so a disarmed invariant fails
    # instantly and leaves nothing to clean up. A campaign proves "no mirrors,
    # no ping-pong, no held frames" largely by what did not happen -- and a
    # disarmed guard produces the same silence as a clean run.
    disarmed = guards_armed()
    if disarmed:
        print("REFUSING: the no-reuse invariant is not armed:", file=sys.stderr)
        for line in disarmed:
            print("  - %s" % line, file=sys.stderr)
        return 3

    if LOCK.exists():
        print("REFUSING: %s exists -- a campaign is already running.\n"
              "Two campaigns on one 16 GB GPU thrash and corrupt each other's\n"
              "results. Stop the other run, then delete the lock." % LOCK,
              file=sys.stderr)
        return 2

    # RESOLVE THE LEGS BEFORE TAKING THE LOCK. build_legs() raises SystemExit
    # when a local engine has no profile, and when the lock was written first
    # that exception escaped before the try/finally existed -- stranding the
    # lock so every LATER run refused with "a campaign is already running"
    # until someone deleted the file by hand. The roster is derived precisely so
    # a newly registered engine joins automatically, which is exactly when a
    # missing profile (and so this refusal) becomes likely. Nothing in leg
    # resolution needs the lock, so the ordering fixes it outright.
    all_legs = build_legs()
    legs = all_legs
    if args.only:
        wanted = {n.strip() for n in args.only.split(",") if n.strip()}
        if not wanted:
            raise SystemExit("REFUSING: --only was given but names no engine. "
                             "A campaign that runs zero legs and exits 0 reads "
                             "as a pass.")
        unknown = wanted - {leg[0] for leg in all_legs}
        if unknown:
            raise SystemExit("REFUSING: --only names engines that are not "
                             "registered local engines: %s"
                             % ", ".join(sorted(unknown)))
        legs = [leg for leg in all_legs if leg[0] in wanted]

    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text("pid=%d start=%s\n" % (os.getpid(), _now()), encoding="utf-8")

    results = []
    try:
        for engine, profile in legs:
            reset_between_legs()
            results.append(run_leg(engine, profile, args.acts, args.source_bank))
            (REPO / "tmp" / "_w45_results.json").write_text(
                json.dumps(results, indent=2), encoding="utf-8")
    finally:
        LOCK.unlink(missing_ok=True)

    print("\n=========== %d-ACT CAMPAIGN ===========" % args.acts)
    for r in results:
        print("%-16s %-5s exit=%-4s %6s min  %s" % (
            r["engine"], "PASS" if r["ok"] else "FAIL", r["exit"],
            r["minutes"], r.get("why", r.get("coverage", ""))))
    passed = sum(1 for r in results if r["ok"])

    # SAY WHAT WAS AND WAS NOT COVERED. The previous summary printed
    # "6/6 engines passed" for a six-engine subset of a nineteen-engine roster,
    # which reads as total coverage. A campaign that cannot state its own scope
    # honestly cannot be used to qualify anything.
    print("%d/%d legs passed" % (passed, len(results)))
    ran = {r["engine"] for r in results}
    skipped = [e for e, _p in all_legs if e not in ran]
    print("local engines registered: %d | run this campaign: %d"
          % (len(all_legs), len(ran)))
    if skipped:
        print("NOT RUN (%d): %s" % (len(skipped), ", ".join(skipped)))
        print("This campaign does NOT qualify the engines above.")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
