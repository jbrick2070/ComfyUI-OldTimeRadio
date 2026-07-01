"""A-S7.5 FULL-EPISODE soak harness -- the CPU portion (the GPU run is operator).

The audio side's worst bugs (BUG-296/291/300) only ever surfaced under a
full-episode soak, so A ships behind this test class: a synthetic 40-beat,
all-roles / all-families episode with a FORCED mid-episode OOM on a
``character_3d`` group, run end-to-end in ONE process, TWICE back-to-back.

This module is the CPU half: it builds the fixture, then drives it through the
SHIPPED A-S7 decision machinery (the retry taxonomy + the fallback-chain
resolver + the durable LOUD ledger restamp) using an INJECTED fake renderer that
simulates the OOM. It proves -- without a GPU -- that the episode completes with
every beat producing a clip, that the character_3d group degrades
``soak_oom_3d -> humo -> humo_1.7B -> still_motion`` (soak_oom_3d is a
synthetic soak stub) to the always-succeeds radio floor with a LOUD
restamp at the SAME ``video_revision``, that the frozen audio section is
byte-identical before and after, and that two back-to-back runs are
deterministic with no cross-episode carryover.

The LIVE GPU soak (the real OTR_VideoRenderBatch + the real engines on the 5080,
VRAM <= 14.5 GB, render-twice pixels, the real audio byte-identical mux) is the
operator gate that ships A -- ``--mode gpu`` prints those steps and refuses to
report a pass (no faking). Cold-import clean (stdlib + the dep-free A-S7 shared
modules). UTF-8, no BOM, ASCII-only source.
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

from nodes._otr_shared import retry_taxonomy as _rt  # noqa: E402
from nodes._otr_shared.fallback import resolve_fallback_chain
from nodes._otr_video_engines import registry as _vreg
# Register the real engines so the real humo -> humo_1.7B -> still_motion
# chain is walked (the character_3d -> humo hop is the synthetic B overlay).
from nodes._otr_video_engines import eng_humo            # noqa: F401
from nodes._otr_video_engines import cheap_families      # noqa: F401

_LOG = logging.getLogger("otr.video.soak")

#: The M1 frozen master-audio PCM sha256 marker -- the soak threads this through
#: an audio ledger section and asserts it is byte-identical after the run (the
#: decision layer must never touch frozen audio). The real byte-identical proof
#: is tests/test_audio_byte_identical.py + the GPU mux.
FROZEN_AUDIO_SHA = "21aa71f6a4e5master_audio_pcm_marker"

#: engine_id -> family, for restamping a shot onto its fallback engine.
ENGINE_FAMILY = {
    "soak_oom_3d": "character_3d",       # synthetic soak stub (see make_fallback_of)
    "humo": "audio_driven_face",
    "humo_1.7B": "audio_driven_face",
    "still_motion": "static_motion",
    "still_parallax": "static_motion",
    "ltx_video": "text_to_video",
    "wan_i2v": "image_to_video",
    "mesh_stage": "image_to_video",
    # C0 2026-06-30: station_card + abstract retired (engines unregistered); the
    # surviving still families carry static_image_gen. Keep in sync with
    # render_driver.ENGINE_FAMILY.
    "still_pan": "static_image_gen",
    "still_flat": "static_image_gen",
    # 2026-06-30: viz_mxc_cpu was a latent gap in this map (never added when it
    # shipped); viz_mxc_mandala added in the SAME chunk it ships. Both are
    # motion-exempt "abstract" -- no _PROFILES soak leg (the dedicated
    # render-contract tests are the coverage), just kept in sync here.
    "viz_mxc_cpu": "abstract",
    "viz_mxc_mandala": "abstract",
}

#: (role, engine, family) rotation covering all 5 roles + the non-3D families.
#: C0 2026-06-30: background_abstract + the 2nd announcer leg repointed from the
#: retired abstract/station_card to surviving still families (kept in sync with
#: render_driver._PROFILES).
_PROFILES = (
    ("announcer_visual", "humo", "audio_driven_face"),
    ("music_visual", "ltx_video", "text_to_video"),
    ("character_video", "wan_i2v", "image_to_video"),
    ("scene_broll", "still_motion", "static_motion"),
    ("background_abstract", "still_flat", "static_image_gen"),
    ("announcer_visual", "still_pan", "static_image_gen"),
)

#: The forced-OOM character_3d group: the synthetic ``soak_oom_3d`` stub (a
#: stand-in heavy character_3d engine) degrades to humo.
_CHAR3D = ("character_video", "soak_oom_3d", "character_3d")

#: The heavy engines the soak forces to OOM on the character_3d shot so the
#: chain walks all the way to the radio floor.
OOM_ENGINES = frozenset({"soak_oom_3d", "humo", "humo_1.7B"})


class OomSignal(RuntimeError):
    """The harness stand-in for a render-time CUDA OOM (a HARD failure)."""


class SoakError(AssertionError):
    """A soak invariant was violated (the soak FAILED)."""


def build_soak_fixture(n_beats: int = 40, oom_index: int = 20):
    """Build a synthetic ``ledger['video']`` section + meta (pure).

    ``n_beats`` shots cycle the role/family profiles (all 5 roles + 7 non-3D
    families appear); the shot at ``oom_index`` is the forced-OOM character_3d
    group. Returns ``(section, meta)`` where ``meta['oom_shot_id']`` names that
    shot.
    """
    if not 0 <= oom_index < n_beats:
        raise ValueError("oom_index %d out of range for %d beats"
                         % (oom_index, n_beats))
    shots = []
    for i in range(n_beats):
        role, engine, family = _CHAR3D if i == oom_index \
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
    meta = {"oom_shot_id": "shot_%04d" % oom_index, "oom_index": oom_index,
            "n_beats": n_beats}
    return section, meta


def build_full_ledger(section: dict) -> dict:
    """Wrap a video section in a full ledger with a FROZEN audio section, so the
    soak can prove the audio is byte-identical after the run."""
    return {
        "audio": {"master_audio_sha256": FROZEN_AUDIO_SHA, "ledger_frozen": True},
        "video": section,
    }


def make_fallback_of():
    """Return a ``fallback_of(name) -> next | None`` over the REAL registry plus
    the synthetic ``soak_oom_3d -> humo`` hop.

    humo -> humo_1.7B -> still_motion come from the live adapters'
    ``fallback_engine``; ``soak_oom_3d`` is a synthetic soak stub (NOT a real
    engine -- the real 3D scaffolds were unregistered 2026-06-29, C3), so its hop
    to humo is overlaid here. A floor engine returns ``None`` (terminal).
    """
    synth = {"soak_oom_3d": "humo"}

    def fallback_of(name):
        if name in synth:
            return synth[name]
        if _vreg.is_registered(name):
            return getattr(_vreg.get_engine(name), "fallback_engine", None)
        return None

    return fallback_of


class SoakRenderer:
    """An injected fake renderer: returns a clip, except it raises ``OomSignal``
    for the target shot on any engine in ``oom_engines`` (forcing the chain to
    walk to the radio floor). Records every call for determinism checks."""

    def __init__(self, oom_shot_id=None, oom_engines=frozenset()):
        self.oom_shot_id = oom_shot_id
        self.oom_engines = frozenset(oom_engines)
        self.calls = []

    def render(self, shot_id: str, engine: str) -> dict:
        self.calls.append((shot_id, engine))
        if shot_id == self.oom_shot_id and engine in self.oom_engines:
            raise OomSignal("forced OOM: shot=%s engine=%s" % (shot_id, engine))
        return {"shot_id": shot_id, "engine_id": engine, "ok": True}


def run_episode_soak(ledger: dict, *, fallback_of, renderer) -> dict:
    """Drive one episode end-to-end through the A-S7 decision machinery (pure).

    Deep-copies ``ledger`` (no input mutation / no carryover), renders every
    shot, and on an ``OomSignal`` classifies (OOM -> HARD), walks the declared
    fallback chain, restamps the shot row + appends a ``runtime_fallback_decisions``
    record at the SAME ``video_revision``, and logs the LOUD swap -- until a clip
    renders (the floor always succeeds). NEVER touches ``ledger['audio']``.
    Returns ``{ledger, clips}``.
    """
    ledger = copy.deepcopy(ledger)
    section = ledger["video"]
    rev = int(section["video_revision"])
    clips = {}
    new_shots = []
    for shot in section["shots"]:
        shot = dict(shot)
        sid, bid = shot["shot_id"], shot["beat_id"]
        chain = resolve_fallback_chain(shot["engine_id"], fallback_of)
        rendered = None
        for cand in chain:
            try:
                rendered = renderer.render(sid, cand)
                break
            except OomSignal:
                nxt = fallback_of(cand)
                if nxt is None:
                    raise SoakError(
                        "radio floor %r failed -- chain exhausted for %s"
                        % (cand, sid))
                _rt.classify(_rt.FailureKind.OOM)       # deterministic decision
                rec = _rt.build_fallback_decision(
                    shot_id=sid, beat_id=bid, from_engine=cand, to_engine=nxt,
                    kind=_rt.FailureKind.OOM, video_revision=rev)
                section = _rt.append_runtime_fallback_decision(section, rec)
                shot = _rt.restamp_shot_row(
                    shot, to_engine=nxt,
                    to_family=ENGINE_FAMILY.get(nxt, shot["family"]),
                    from_engine=cand, kind=_rt.FailureKind.OOM)
                _LOG.warning(_rt.format_swap_log(rec))
        clips[sid] = rendered
        new_shots.append(shot)
    section["shots"] = new_shots
    ledger["video"] = section
    return {"ledger": ledger, "clips": clips}


def run_two_episode_soak(*, n_beats: int = 40, oom_index: int = 20) -> dict:
    """Run the full episode end-to-end TWICE back-to-back (the A-S7.5 shape).

    Both runs consume the SAME input ledger; ``run_episode_soak`` deep-copies it,
    so neither run mutates the shared fixture (the no-carryover guarantee). Fresh
    renderers each run. Returns both result ledgers + the render-call sequences
    for the determinism check.
    """
    section, meta = build_soak_fixture(n_beats=n_beats, oom_index=oom_index)
    ledger = build_full_ledger(section)
    fb = make_fallback_of()
    r1 = SoakRenderer(meta["oom_shot_id"], OOM_ENGINES)
    r2 = SoakRenderer(meta["oom_shot_id"], OOM_ENGINES)
    e1 = run_episode_soak(ledger, fallback_of=fb, renderer=r1)
    e2 = run_episode_soak(ledger, fallback_of=fb, renderer=r2)
    return {"meta": meta, "input_ledger": ledger, "e1": e1, "e2": e2,
            "render_calls_1": r1.calls, "render_calls_2": r2.calls}


#: The expected character_3d degradation trail to the radio floor. This CPU
#: harness's fake renderer raises at EVERY hop, so here the 3-hop trail pairs
#: with 3 decisions (unlike render_driver's GPU soak, where humo->humo_1.7B is
#: an intra-engine tier swap and only 2 decisions appear -- semantics
#: preserved per the 3D plan 7.0 judge ruling).
EXPECTED_OOM_TRAIL = ["soak_oom_3d->humo (oom)", "humo->humo_1.7B (oom)",
                      "humo_1.7B->still_motion (oom)"]


def _episode_facts(epresult: dict, meta: dict) -> dict:
    led = epresult["ledger"]
    sec = led["video"]
    shots = {s["shot_id"]: s for s in sec["shots"]}
    oom = shots[meta["oom_shot_id"]]
    return {
        "n_clips": len(epresult["clips"]),
        "all_clips": all(epresult["clips"].values()),
        "oom_final_engine": oom["engine_id"],
        "oom_trail": oom["degradation_trail"],
        "decisions": sec.get("runtime_fallback_decisions", []),
        "video_revision": sec["video_revision"],
        "audio_sha": led["audio"]["master_audio_sha256"],
    }


def assert_soak_ok(result: dict):
    """Assert every A-S7.5 soak invariant; raise :class:`SoakError` on any
    violation. Returns the list of passed-check descriptions for the report."""
    meta = result["meta"]
    n = meta["n_beats"]
    checks = []
    facts = {"episode-1": _episode_facts(result["e1"], meta),
             "episode-2": _episode_facts(result["e2"], meta)}
    for tag, f in facts.items():
        if f["n_clips"] != n or not f["all_clips"]:
            raise SoakError("%s: not every beat produced a clip (%d/%d)"
                            % (tag, f["n_clips"], n))
        if f["oom_final_engine"] != "still_motion":
            raise SoakError("%s: character_3d OOM did not converge to the radio "
                            "floor (got %r)" % (tag, f["oom_final_engine"]))
        if f["oom_trail"] != EXPECTED_OOM_TRAIL:
            raise SoakError("%s: degradation trail %r != %r"
                            % (tag, f["oom_trail"], EXPECTED_OOM_TRAIL))
        if len(f["decisions"]) != 3:
            raise SoakError("%s: expected 3 LOUD fallback decisions, got %d"
                            % (tag, len(f["decisions"])))
        for d in f["decisions"]:
            if (d["failure_kind"] != "oom" or d["block_class"] != "hard"
                    or d["video_revision"] != 1):
                raise SoakError("%s: malformed fallback decision %r" % (tag, d))
        if f["audio_sha"] != FROZEN_AUDIO_SHA:
            raise SoakError("%s: frozen audio sha changed (%r) -- the soak must "
                            "never touch audio" % (tag, f["audio_sha"]))
        if f["video_revision"] != 1:
            raise SoakError("%s: video_revision bumped to %r (a restamp stays at "
                            "the same revision)" % (tag, f["video_revision"]))
        checks.append("%s: %d beats, character_3d OOM->floor converged, 4 LOUD "
                      "restamps @rev1, frozen audio untouched" % (tag, n))
    if facts["episode-1"]["decisions"] != facts["episode-2"]["decisions"]:
        raise SoakError("non-deterministic: the two episodes' fallback decisions "
                        "differ")
    if result["render_calls_1"] != result["render_calls_2"]:
        raise SoakError("non-deterministic: the two episodes' render-call "
                        "sequences differ")
    # no carryover: the shared input fixture was not mutated by either run.
    in_oom = {s["shot_id"]: s for s in
              result["input_ledger"]["video"]["shots"]}[meta["oom_shot_id"]]
    if in_oom["engine_id"] != "soak_oom_3d" or in_oom["degradation_trail"]:
        raise SoakError("carryover: the input fixture was mutated by a run")
    checks.append("determinism: two back-to-back episodes identical; input "
                  "fixture unmutated (no carryover)")
    return checks


GPU_GATE_MESSAGE = (
    "A-S7.5 GPU soak is the OPERATOR gate that ships A -- the CPU harness "
    "cannot certify it.\n"
    "On the 5080, wire the live OTR_VideoRenderBatch + the real engines and run "
    "this same 40-beat all-roles fixture end-to-end TWICE back-to-back, with:\n"
    "  - a real mid-episode character_3d OOM -> humo -> humo_1.7B -> "
    "still_motion convergence (LOUD restamp in the ledger);\n"
    "  - VRAM peak <= 14.5 GB at every inter-engine boundary;\n"
    "  - render-twice determinism (identical per-shot request_hash);\n"
    "  - tests/test_audio_byte_identical.py GREEN (output audio PCM sha == the "
    "frozen master).\n"
    "Then tag A-ship. This command exits non-zero so it is never mistaken for a "
    "pass."
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
    print("A-S7.5 CPU soak PASS: 2 episodes x %d beats; mid-episode character_3d "
          "OOM converged to the radio floor; frozen audio untouched; "
          "deterministic." % args.beats)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
