"""S16.6: the production workflow JSON must pass the full validator
in default mode. Strict-unknown-types mode is exercised at
production-loader time (S14.2.1) where every OTR class is
registered; the bare test env has known import-skip cases for
heavy optional deps (HuMo / LTX / Upscale) that legitimately can't
import.

This test is the cumulative gate for S16.1 (widget-name scrub),
S16.2 (extended check 5), S16.3 (positional widget-drift), S16.4
(FluxPortrait.ledger_json wired), and S16.5 (link-tuple + dup-dedup).
Any of those regressing will fire here.
"""
from __future__ import annotations

import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from nodes._workflow_validation import (  # noqa: E402
    validate_workflow_contract,
)


CANONICAL_WORKFLOW = REPO_ROOT / "workflows" / "otr_canonical.json"


def _otr_mappings_via_existing_test_helper() -> dict:
    """Reuse the AST-walk + importlib helper from
    test_workflow_contract_validation.py so we don't duplicate the
    parsing logic. Failing imports are silently skipped -- that's
    fine for default mode; strict mode is exercised elsewhere.
    """
    import test_workflow_contract_validation as twv
    return twv._otr_node_class_mappings()


def test_production_workflow_passes_default_validation():
    wf = json.loads(CANONICAL_WORKFLOW.read_text(encoding="utf-8"))
    mappings = _otr_mappings_via_existing_test_helper()
    # Default mode: strict_unknown_types=False so test-env class-
    # skip doesn't fire. The S14.2.1 production loader path enables
    # strict mode where every class IS registered.
    validate_workflow_contract(wf, mappings)


def test_production_workflow_visual_structure_pinned():
    """PRODUCTION RESTORE pin (operator directive 2026-06-10): the SAVED
    workflow itself -- not a headless runner's submit-time patches -- must
    carry the episode's full visual structure, so a ComfyUI Desktop render
    matches the prior quality bar. Pins the three restored pieces:

      1. burned-in SDH captions: node 86 OTR_CaptionBurn OWNS the burn
         (burn_captions=True, sdh_standard) and now sits AFTER the procgen blend
         (2026-07-04 widget-audit Batch 3 -- single caption owner); node 93
         OTR_PostUpscaleProcgenBlend no longer carries the burn_captions /
         caption_style widgets;
      2. the LTX radio open: node 87 OTR_VideoDirector routes
         announcer_video_model AND music_video_model to ltx_video;
      3. the credits stage: node 12 OTR_SignalLostVideo's procgen feeds BOTH
         the composite base (node 84) and the blend texture (node 93), and the
         chain runs 84 -> 93 -> 86 -> 95 -> 85: captions (node 86) burn BEFORE
         the late OTR_CreditsRoll (node 95, credits enrichment 2026-07-03), which
         appends the unified SILENT credits roll and declares its tail duration to
         the credits-aware terminal mux (node 85).

    If any of these regress to headless-only defaults again, this fires.
    """
    wf = json.loads(CANONICAL_WORKFLOW.read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in wf["nodes"]}
    links = {l[0]: l for l in wf["links"]}

    # -- 1. caption ownership (86-owner migration, 2026-07-04 widget-audit) ----
    n93 = nodes[93]
    assert n93["type"] == "OTR_PostUpscaleProcgenBlend"
    wv93 = n93["widgets_values"]
    # post-migration vector (11): [src, pgn, blend_mode, opacity, ffmpeg, bypass,
    #                              out_suffix, crush, green_only, scopes, audio_bars]
    assert len(wv93) == 11, (
        "node 93 widgets_values must be 11 (caption widgets removed): %r" % wv93)
    assert wv93[5] is False, "node 93 bypass must stay OFF"
    n93_input_names = [i.get("name") for i in n93["inputs"]]
    assert ("burn_captions" not in n93_input_names
            and "caption_style" not in n93_input_names), (
        "node 93 must NOT carry caption widgets -- ownership moved to node 86")
    n86 = nodes[86]
    assert n86["type"] == "OTR_CaptionBurn"
    assert n86["widgets_values"][0] is True, (
        "node 86 OTR_CaptionBurn now OWNS the burn (burn_captions must be ON)")
    assert n86["widgets_values"][1] == "sdh_standard", "node 86 caption style regressed"

    # -- 2. lean visual defaults ---------------------------------------------
    n87 = nodes[87]
    assert n87["type"] == "OTR_VideoDirector"
    wv87 = n87["widgets_values"]
    # 2026-07-07: the repo canonical is the operator-saved 30-word test canvas,
    # with CPU visualizers and z_image_turbo stills. Heavy/video profiles are
    # explicit profile overrides, not saved-workflow defaults.
    assert wv87[0].startswith("viz_mxc_cpu"), (
        "announcer_video_model regressed off the lean visualizer default: %r" % wv87[0])
    assert wv87[1].startswith("viz_mxc_mandala"), (
        "music_video_model regressed off the lean mandala default: %r" % wv87[1])
    # rip-sfx-broll (2026-07-01): widgets_values shrank 19 -> 15; clean-UI removals
    # (2026-07-03) dropped allow_auto_fallback (15->14), episode_duration_target
    # (14->13), then consolidated the legacy catch-all video slot -> character promoted to video
    # slot 3 (13->12; character_video_model is widgets index 2 now).
    # S5 platform-portability (2026-07-10): OLD pin 12 -> NEW pin 14
    # (+device_policy="cuda" appended at index 12, +dtype_policy="fp8_ok"
    # appended at index 13; character_video_model stays at index 2).
    # WAN 8GB launch contract (2026-07-24): OLD pin 14 -> NEW pin 15
    # (+max_render_frames=0 appended at index 14 -- 0 = UNPINNED, so the saved
    # canonical keeps today's behaviour and only a tier profile pins a ceiling).
    assert len(wv87) == 15, wv87
    assert wv87[14] == 0, "canonical must ship the render ceiling UNPINNED"
    assert wv87[2].startswith("viz_camera"), (
        "character_video_model must stay on the lean camera visualizer lane: %r"
        % wv87[2])
    assert wv87[3:6] == ["z_image_turbo", "z_image_turbo", "z_image_turbo"]

    # -- 3. the credits-bearing procgen wiring + chain order ------------------
    out12 = set(nodes[12]["outputs"][0].get("links") or [])
    assert {246, 265} <= out12, (
        "node 12 procgen must feed the composite base (246) AND the blend "
        "texture (265); got %r" % sorted(out12))
    assert links[246][1:5] == [12, 0, 84, 0]
    assert links[265][1:5] == [12, 0, 93, 1]
    assert links[247][1:5] == [84, 0, 93, 0]   # composite -> blend source
    assert links[266][1:5] == [93, 0, 86, 0]   # blend -> caption node
    assert links[273][1:5] == [94, 0, 93, 9]   # scopes -> blend (dst_slot 11->9 post strip)
    # credits enrichment 2026-07-03 + 86-owner migration 2026-07-04: the chain is
    # 84 -> 93 -> 86 -> 95 -> 85. Captions (node 86) burn BEFORE the credits roll
    # (node 95) so the *_with_credits concat is never re-encoded and captions land
    # on dialogue only. OTR_CreditsRoll stays terminal-before-mux.
    assert links[250][1:5] == [86, 0, 95, 0]   # caption final -> credits roll
    assert links[274][1:5] == [95, 0, 85, 0]   # credits roll -> terminal mux
    assert links[275][1:5] == [92, 1, 95, 1]   # clip manifest -> credits roll
    assert links[276][1:5] == [95, 1, 85, 3]   # declared credits tail -> mux
    n95 = nodes[95]
    assert n95["type"] == "OTR_CreditsRoll"
    assert not n95.get("widgets_values"), "node 95 has zero widgets (two forceInputs)"

    # -- 4. the W4 image-before-video gate (still-spine ST-0.2) ---------------
    # OTR_ImageGenDispatcher.image_done (91 out 1) must reach the render
    # node's image_done input so video render NEVER starts before every
    # episode still exists on disk. The ledger data edge (260) orders the
    # pair today; this explicit gate pins the contract against rewiring.
    n92 = nodes[92]
    assert n92["type"] == "OTR_VideoRenderBatch"
    gate = [i for i in n92["inputs"] if i.get("name") == "image_done"]
    assert gate and gate[0].get("link") == 267, (
        "node 92 image_done gate input missing or unwired (W4)")
    assert links[267][1:3] == [91, 1], (
        "image_done gate must come from OTR_ImageGenDispatcher out 1")
    assert links[260][1:5] == [91, 0, 92, 0]   # dispatcher ledger -> render

    # -- 5. the episode_id wire (still-spine ST-6 / DS-3) ---------------------
    # ShotLock's additive episode_id output (out 4) feeds the dispatcher's
    # episode_id input so every still lands in episodes/<ep>/stills/.
    n91 = nodes[91]
    epi = [i for i in n91["inputs"] if i.get("name") == "episode_id"]
    assert epi and epi[0].get("link") == 268, (
        "dispatcher episode_id input missing or unwired (ST-6)")
    assert links[268][1:3] == [90, 4], (
        "episode_id must come from OTR_ShotLock out 4")

    # -- 6. pending-to-final rename gate ------------------------------------
    # ShotLock must read the durable ledger only after SignalLostVideo has
    # moved/rebased it, so image/video consumers receive the final episode id
    # and paths rather than activating newest-ledger rescue logic.
    n90 = nodes[90]
    rename_gate = [i for i in n90["inputs"] if i.get("name") == "gate_in"]
    assert rename_gate and rename_gate[0].get("link") == 284
    assert links[284] == [284, 12, 0, 90, 4, "STRING"]
