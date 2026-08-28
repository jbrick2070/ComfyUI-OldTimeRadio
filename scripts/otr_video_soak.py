"""A-S7.5 FULL-EPISODE soak harness -- the CPU portion (the GPU run is operator).

NO FALLBACKS (operator directive 2026-07-02: NO fallbacks / NO auto-defaults
anywhere). The old CPU harness proved the fallback-chain restamp machinery;
that machinery is RIPPED (Sprint A, E1). What this harness proves now:

* CLEAN legs: a synthetic 40-beat, all-roles / all-families episode driven
  through an INJECTED fake renderer completes with every beat producing a clip,
  the frozen audio section byte-identical before and after, and two
  back-to-back runs deterministic with no cross-episode carryover.
* LOUD-failure contract leg: a forced mid-episode OOM on the synthetic
  ``soak_oom_heavy`` heavy-engine stub PROPAGATES as a raise -- NO swap, NO
  restamp, NO degradation trail. The soak asserts the raise.

The LIVE GPU soak (the real OTR_VideoRenderBatch + the real engines on the
5080, VRAM <= 14.5 GB, render-twice pixels, the real audio byte-identical mux)
is the operator gate -- ``--mode gpu`` prints those steps and refuses to report
a pass (no faking). Cold-import clean (stdlib + the dep-free shared modules).
UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import argparse
import copy
import logging
import os
import sys

# Put the repo root on sys.path so ``from nodes import ...`` resolves when this
# script is run directly (python scripts/otr_video_soak.py), not only via pytest.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Register the real engines (the fixture names real engine ids).
from nodes._otr_video_engines import eng_humo            # noqa: E402,F401
from nodes._otr_video_engines import cheap_families      # noqa: E402,F401

_LOG = logging.getLogger("otr.video.soak")

#: The M1 frozen master-audio PCM sha256 marker -- the soak threads this through
#: an audio ledger section and asserts it is byte-identical after the run (the
#: render layer must never touch frozen audio). The real byte-identical proof
#: is tests/test_audio_byte_identical.py + the GPU mux.
FROZEN_AUDIO_SHA = "21aa71f6a4e5master_audio_pcm_marker"

#: (role, engine, family) rotation covering the 3 roles + the non-3D families.
#: rip-sfx-broll (2026-07-01): the scene_broll/background_abstract legs died
#: with their roles (kept in sync with render_driver._PROFILES).
_PROFILES = (
    ("announcer_visual", "humo", "audio_driven_face"),
    # re-seated 2026-08-28: ltx_video moved to image_to_video; the
    # remaining local text_to_video engine keeps the family covered.
    ("music_visual", "animatediff15_v3_haunted_video", "text_to_video"),
    # The image_to_video seat -- `wan_i2v` until the 14B was retired on
    # 2026-08-26, `wan_ti2v` (the 5B) after. KEEP THIS TUPLE IDENTICAL to
    # `render_driver._PROFILES`: the GPU soak must walk the same shape the CPU
    # harness proves, so a row removed from one and not the other is a silent
    # divergence.
    ("character_video", "wan_ti2v", "image_to_video"),
    ("character_video", "still_motion", "static_motion"),
    ("music_visual", "still_flat", "static_image_gen"),
    ("announcer_visual", "still_pan", "static_image_gen"),
)

#: The forced-OOM group: the synthetic ``soak_oom_heavy`` stub, standing in
#: for the LIVE audio_driven_face family (rebased off character_3d 2026-08-23
#: ahead of that family's retirement). Its forced OOM must RAISE -- there is
#: no chain and no floor (NO FALLBACKS).
_HEAVY_OOM = ("character_video", "soak_oom_heavy", "audio_driven_face")

#: The engines the LOUD-contract leg forces to OOM on the injected shot.
OOM_ENGINES = frozenset({"soak_oom_heavy", "humo", "humo_1.7B"})


class OomSignal(RuntimeError):
    """The harness stand-in for a render-time CUDA OOM (a HARD failure)."""


class SoakError(AssertionError):
    """A soak invariant was violated (the soak FAILED)."""


def build_soak_fixture(n_beats: int = 40, oom_index=None):
    """Build a synthetic ``ledger['video']`` section + meta (pure).

    ``oom_index=None`` builds a CLEAN all-profiles fixture; an integer injects
    the synthetic ``soak_oom_heavy`` heavy-engine stub at that index for the
    LOUD-failure contract leg. Returns ``(section, meta)``.
    """
    if oom_index is not None and not 0 <= oom_index < n_beats:
        raise ValueError("oom_index %d out of range for %d beats"
                         % (oom_index, n_beats))
    shots = []
    for i in range(n_beats):
        role, engine, family = _HEAVY_OOM if i == oom_index \
            else _PROFILES[i % len(_PROFILES)]
        shots.append({
            "shot_id": "shot_%04d" % i,
            "beat_id": "b%04d" % i,
            "role": role,
            "engine_id": engine,
            "family": family,
            "group_id": "grp_%04d" % i,
            "target_frame_count": 25 + i,
            "degradation_trail": [],
        })
    section = {"video_revision": 1, "fps": 25, "shots": shots}
    meta = {"oom_shot_id": ("shot_%04d" % oom_index)
            if oom_index is not None else None,
            "oom_index": oom_index, "n_beats": n_beats}
    return section, meta


def build_full_ledger(section: dict) -> dict:
    """Wrap a video section in a full ledger with a FROZEN audio section, so the
    soak can prove the audio is byte-identical after the run."""
    return {
        "audio": {"master_audio_sha256": FROZEN_AUDIO_SHA, "ledger_frozen": True},
        "video": section,
    }


class SoakRenderer:
    """An injected fake renderer: returns a clip, except it raises ``OomSignal``
    for the target shot on any engine in ``oom_engines`` (the LOUD-failure
    contract leg). Records every call for determinism checks."""

    def __init__(self, oom_shot_id=None, oom_engines=frozenset()):
        self.oom_shot_id = oom_shot_id
        self.oom_engines = frozenset(oom_engines)
        self.calls = []

    def render(self, shot_id: str, engine: str) -> dict:
        self.calls.append((shot_id, engine))
        if shot_id == self.oom_shot_id and engine in self.oom_engines:
            raise OomSignal("forced OOM: shot=%s engine=%s" % (shot_id, engine))
        return {"shot_id": shot_id, "engine_id": engine, "ok": True}


def run_episode_soak(ledger: dict, *, renderer) -> dict:
    """Drive one episode end-to-end (pure). NO FALLBACKS: each shot renders on
    its selected engine EXACTLY as planned; a render failure PROPAGATES (the
    caller asserts the raise on the LOUD-contract leg). Deep-copies ``ledger``
    (no input mutation / no carryover) and NEVER touches ``ledger['audio']``.
    Returns ``{ledger, clips}``.
    """
    ledger = copy.deepcopy(ledger)
    section = ledger["video"]
    clips = {}
    for shot in section["shots"]:
        sid = shot["shot_id"]
        clips[sid] = renderer.render(sid, shot["engine_id"])
    return {"ledger": ledger, "clips": clips}


def run_two_episode_soak(*, n_beats: int = 40, oom_index: int = 20) -> dict:
    """Run the CLEAN episode end-to-end TWICE back-to-back, then the forced-OOM
    LOUD-failure contract leg.

    Both clean runs consume the SAME input ledger; ``run_episode_soak``
    deep-copies it, so neither run mutates the shared fixture (the no-carryover
    guarantee). Fresh renderers each run. The contract leg builds its own
    fixture with the ``soak_oom_heavy`` stub and asserts the forced OOM RAISES.
    Returns both result ledgers + the render-call sequences + the contract
    outcome.
    """
    section, meta = build_soak_fixture(n_beats=n_beats, oom_index=None)
    ledger = build_full_ledger(section)
    r1 = SoakRenderer()
    r2 = SoakRenderer()
    e1 = run_episode_soak(ledger, renderer=r1)
    e2 = run_episode_soak(ledger, renderer=r2)
    # LOUD-failure contract leg (NO FALLBACKS): the forced OOM must RAISE.
    oom_section, oom_meta = build_soak_fixture(n_beats=n_beats,
                                               oom_index=oom_index)
    oom_ledger = build_full_ledger(oom_section)
    oom_renderer = SoakRenderer(oom_meta["oom_shot_id"], OOM_ENGINES)
    contract = {"raised": False, "error_type": "", "detail": ""}
    try:
        run_episode_soak(oom_ledger, renderer=oom_renderer)
    except OomSignal as exc:
        contract = {"raised": True, "error_type": "OomSignal",
                    "detail": str(exc)}
    return {"meta": meta, "input_ledger": ledger, "e1": e1, "e2": e2,
            "render_calls_1": r1.calls, "render_calls_2": r2.calls,
            "oom_contract": contract, "oom_input_ledger": oom_ledger,
            "oom_meta": oom_meta}


def _episode_facts(epresult: dict, meta: dict) -> dict:
    led = epresult["ledger"]
    sec = led["video"]
    return {
        "n_clips": len(epresult["clips"]),
        "all_clips": all(epresult["clips"].values()),
        "video_revision": sec["video_revision"],
        "audio_sha": led["audio"]["master_audio_sha256"],
        "trails": [s.get("degradation_trail") for s in sec["shots"]],
    }


def assert_soak_ok(result: dict):
    """Assert every soak invariant; raise :class:`SoakError` on any violation.
    Returns the list of passed-check descriptions for the report.

    NO-TRAIL LOUD contract: the clean episodes complete deterministically with
    empty degradation trails and untouched frozen audio; the forced-OOM leg
    must have RAISED (never a swap)."""
    meta = result["meta"]
    n = meta["n_beats"]
    checks = []
    facts = {"episode-1": _episode_facts(result["e1"], meta),
             "episode-2": _episode_facts(result["e2"], meta)}
    for tag, f in facts.items():
        if f["n_clips"] != n or not f["all_clips"]:
            raise SoakError("%s: not every beat produced a clip (%d/%d)"
                            % (tag, f["n_clips"], n))
        if any(f["trails"][i] for i in range(len(f["trails"]))):
            raise SoakError("%s: a degradation trail was stamped -- the "
                            "fallback machinery is ripped; nothing may restamp"
                            % tag)
        if f["audio_sha"] != FROZEN_AUDIO_SHA:
            raise SoakError("%s: frozen audio sha changed (%r) -- the soak must "
                            "never touch audio" % (tag, f["audio_sha"]))
        if f["video_revision"] != 1:
            raise SoakError("%s: video_revision bumped to %r (rendering never "
                            "re-locks the plan)" % (tag, f["video_revision"]))
        checks.append("%s: %d beats, all clips real, no trails, frozen audio "
                      "untouched" % (tag, n))
    if result["render_calls_1"] != result["render_calls_2"]:
        raise SoakError("non-deterministic: the two episodes' render-call "
                        "sequences differ")
    # no carryover: the shared input fixture was not mutated by either run.
    for s in result["input_ledger"]["video"]["shots"]:
        if s["degradation_trail"]:
            raise SoakError("carryover: the input fixture was mutated by a run")
    checks.append("determinism: two back-to-back episodes identical; input "
                  "fixture unmutated (no carryover)")
    # LOUD-failure contract: the forced OOM raised; nothing restamped.
    oc = result["oom_contract"]
    if not oc.get("raised") or oc.get("error_type") != "OomSignal":
        raise SoakError("LOUD-failure contract violated: the forced OOM did "
                        "not raise (raised=%s error_type=%r) -- NO FALLBACKS"
                        % (oc.get("raised"), oc.get("error_type")))
    oom_in = {s["shot_id"]: s for s in
              result["oom_input_ledger"]["video"]["shots"]}
    oom_shot = oom_in[result["oom_meta"]["oom_shot_id"]]
    if oom_shot["engine_id"] != "soak_oom_heavy" or oom_shot["degradation_trail"]:
        raise SoakError("LOUD-failure contract: the forced-OOM fixture was "
                        "restamped (engine=%r trail=%r) -- a failure must "
                        "never swap engines"
                        % (oom_shot["engine_id"], oom_shot["degradation_trail"]))
    checks.append("LOUD-failure contract: forced OOM raised OomSignal; no "
                  "swap, no restamp, no trail")
    return checks


GPU_GATE_MESSAGE = (
    "A-S7.5 GPU soak is the OPERATOR gate -- the CPU harness cannot certify "
    "it.\n"
    "On the 5080, wire the live OTR_VideoRenderBatch + the real engines and run "
    "the 40-beat all-roles fixture end-to-end TWICE back-to-back, with:\n"
    "  - every beat rendering on its SELECTED engine (NO FALLBACKS -- a "
    "failure raises RenderError LOUD and the run STOPS);\n"
    "  - VRAM peak <= 14.5 GB at every inter-engine boundary;\n"
    "  - render-twice determinism (identical per-shot request_hash);\n"
    "  - tests/test_audio_byte_identical.py GREEN (output audio PCM sha == the "
    "frozen master).\n"
    "This command exits non-zero so it is never mistaken for a pass."
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="A-S7.5 full-episode soak (CPU portion; GPU run is operator).")
    ap.add_argument("--mode", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--beats", type=int, default=40)
    ap.add_argument("--oom-index", type=int, default=20)
    args = ap.parse_args(argv)
    if args.mode == "gpu":
        print(GPU_GATE_MESSAGE)
        return 2                                   # never a CPU-certifiable pass
    result = run_two_episode_soak(n_beats=args.beats, oom_index=args.oom_index)
    for c in assert_soak_ok(result):
        print("[PASS] " + c)
    print("A-S7.5 CPU soak PASS: 2 clean episodes x %d beats deterministic; "
          "forced OOM raised LOUD (no swap); frozen audio untouched."
          % args.beats)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
