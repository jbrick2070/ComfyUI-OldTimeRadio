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


CANONICAL_WORKFLOW = REPO_ROOT / "workflows" / "otr_scifi_16gb_full.json"


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

      1. burned-in SDH captions: node 93 OTR_PostUpscaleProcgenBlend owns the
         final burn (burn_captions=True, sdh_standard); node 86 OTR_CaptionBurn
         stays a pass-through (False) so captions never double-burn;
      2. the LTX radio open: node 87 OTR_VideoDirector routes
         announcer_video_model AND music_video_model to ltx_video;
      3. the rolling-credits stage: node 12 OTR_SignalLostVideo's procgen
         (which carries the credits post-roll) feeds BOTH the composite base
         (node 84) and the blend texture (node 93), and the chain runs
         84 -> 86 -> 93 -> 85 so the credits-extended composite reaches the
         terminal mux.

    If any of these regress to headless-only defaults again, this fires.
    """
    wf = json.loads(CANONICAL_WORKFLOW.read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in wf["nodes"]}
    links = {l[0]: l for l in wf["links"]}

    # -- 1. caption ownership ------------------------------------------------
    n93 = nodes[93]
    assert n93["type"] == "OTR_PostUpscaleProcgenBlend"
    wv93 = n93["widgets_values"]
    # preserved-mode vector: [src, pgn, blend_mode, opacity, ffmpeg, bypass,
    #                         out_suffix, crush, green_only, burn, style]
    assert wv93[9] is True, "node 93 burn_captions must stay ON (the owner)"
    assert wv93[10] == "sdh_standard", "caption style regressed"
    assert wv93[5] is False, "node 93 bypass must stay OFF"
    n86 = nodes[86]
    assert n86["type"] == "OTR_CaptionBurn"
    assert n86["widgets_values"][0] is False, (
        "node 86 must stay a pass-through -- 93 owns the burn (double-burn "
        "guard)")

    # -- 2. the LTX radio open ------------------------------------------------
    n87 = nodes[87]
    assert n87["type"] == "OTR_VideoDirector"
    wv87 = n87["widgets_values"]
    # 2026-06-29: keep the broad/cheap visible video slots on viz_green
    # (renamed from visualizer 2026-06-30, item 2), but route character beats
    # through the Route-A 14B wide HuMo lane. Values are bare engine ids so the
    # capability-profile validator + 16gb-identity profile stay consistent.
    assert wv87[0] == "viz_green", (
        "announcer_video_model regressed off the viz_green default: %r" % wv87[0])
    assert wv87[1] == "viz_green", (
        "music_video_model regressed off the viz_green default: %r" % wv87[1])
    # rip-sfx-broll (2026-07-01): widgets_values shrank 19 -> 15; the deprecated
    # allow_auto_fallback widget removal (2026-07-03, clean-UI directive) dropped
    # it 15 -> 14 (slot 11 + its rogue input socket); character_video_model is the
    # FINAL value now.
    assert len(wv87) == 14, wv87
    assert wv87[13] == "humo_14B_169", (
        "character_video_model must stay on the Route-A 14B motion lane: %r"
        % wv87[13])

    # -- 3. the credits-bearing procgen wiring + chain order ------------------
    out12 = set(nodes[12]["outputs"][0].get("links") or [])
    assert {246, 265} <= out12, (
        "node 12 procgen must feed the composite base (246) AND the blend "
        "texture (265); got %r" % sorted(out12))
    assert links[246][1:5] == [12, 0, 84, 0]
    assert links[265][1:5] == [12, 0, 93, 1]
    assert links[247][1:5] == [84, 0, 86, 0]   # composite -> caption node
    assert links[266][1:5] == [86, 0, 93, 0]   # caption -> blend source
    assert links[250][1:5] == [93, 0, 85, 0]   # blend final -> terminal mux

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
