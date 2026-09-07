"""The music engine must be visible to the asset planner.

PBUG-20260907-02. Selecting `stable_audio_3` in the UI and pressing Run logged

    [OTR.assets] READY engines=ltx_8gb,z_image_turbo files=5

with the music engine simply absent, and the render then died 5m40s in -- after
the entire script had been written -- with

    audio engine 'stable_audio_3' is not usable for role 'music': missing_model
    -- SA3 checkpoint not found ... fetch Comfy-Org/stable-audio-3 first

So the licence-clean music engine was unusable on every card, and `musicgen`
(CC-BY-NC) was the only one that worked. Every published episode carries a
non-commercial bed because of this.

TWO CAUSES, and the fix needed both:

1. `plan_prompt` collected only nodes whose `gate_in` named the validator
   DIRECTLY. The audio branch is two hops out
   (validator -> BatchCharacterVoices -> AnnouncerVoice -> StableAudioTheme), so
   the music node was never in scope and the `_MUSIC_NODE` scan was dead code.
2. `OTR_BatchCharacterVoices.gate_in` was `link=None` in the SHIPPED GRAPH --
   the whole audio chain hung off an unwired gate, so even a transitive walk had
   nothing to follow. That also meant voices could begin before the validator
   passed, which is the ordering the gate exists to guarantee.

These tests pin both halves, because either alone leaves the engine unplanned.
Pure structure: no server, no download, no render.
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nodes import _otr_visual_assets as VA  # noqa: E402

VALIDATOR = "OTR_WorkflowValidator"
AUDIO_HEAD = "OTR_BatchCharacterVoices"
MUSIC = "OTR_StableAudioTheme"


def _prompt():
    """API-shaped prompt mirroring the shipped gate topology."""
    return {
        "63": {"class_type": VALIDATOR, "inputs": {}},
        "1": {"class_type": "OTR_LedgerScriptWriter",
              "inputs": {"gate_in": ["63", 0], "replay_from": ""}},
        "87": {"class_type": "OTR_VideoDirector",
               "inputs": {"gate_in": ["63", 0],
                          "announcer_video_model": "still_pan (16:9)",
                          "music_video_model": "still_pan (16:9)",
                          "character_video_model": "still_pan (16:9)",
                          "announcer_image_model": "z_image_turbo",
                          "music_image_model": "z_image_turbo",
                          "character_image_model": "z_image_turbo"}},
        "81": {"class_type": AUDIO_HEAD, "inputs": {"gate_in": ["63", 0]}},
        "82": {"class_type": "OTR_AnnouncerVoice", "inputs": {"gate_in": ["81", 0]}},
        "83": {"class_type": MUSIC,
               "inputs": {"gate_in": ["82", 0], "engine": "stable_audio_3"}},
    }


def _plan(prompt):
    return VA.plan_prompt(prompt, "63",
                          resolve_video=lambda v: v.split(" (")[0],
                          freeze_video=lambda d: d)


class PlannerReachesTheMusicNodeTests(unittest.TestCase):
    def test_a_music_engine_two_hops_out_is_planned(self):
        """The regression itself. One-hop scoping returned only the video and
        image engines and the render died at the music node."""
        engines = _plan(_prompt())["engines"]
        self.assertIn("stable_audio_3", engines,
                      "the music engine is two gate hops from the validator and "
                      "was invisible to the planner; got %s" % sorted(engines))

    def test_another_validators_subgraph_still_cannot_leak(self):
        """The isolation the one-hop rule was protecting. A frozen replay bundle
        behind a DIFFERENT validator must not trigger downloads from its live
        widgets, so reachability must follow gate edges from THIS validator only.
        """
        prompt = _prompt()
        prompt["98"] = {"class_type": VALIDATOR, "inputs": {}}
        prompt["99"] = {"class_type": MUSIC,
                        "inputs": {"gate_in": ["98", 0], "engine": "musicgen"}}
        engines = _plan(prompt)["engines"]
        self.assertIn("stable_audio_3", engines)
        self.assertNotIn("musicgen", engines,
                         "another validator's music engine leaked into this plan")

    def test_an_ungated_audio_chain_is_still_invisible(self):
        """Guards the guard: with node 81's gate unwired -- the exact shipped
        state that caused this -- the music node must NOT be found, or this
        suite would pass without the graph fix."""
        prompt = _prompt()
        prompt["81"]["inputs"].pop("gate_in")
        engines = _plan(prompt)["engines"]
        self.assertNotIn("stable_audio_3", engines,
                         "fixture no longer reproduces the orphaned chain")

    def test_a_deep_chain_does_not_loop_forever(self):
        """The walk is iterative; a cycle in gate_in must terminate."""
        prompt = _prompt()
        prompt["81"]["inputs"]["gate_in"] = ["83", 0]   # 81 <- 83 <- 82 <- 81
        _plan(prompt)  # must return rather than hang


class ShippedGraphWiresTheAudioChainTests(unittest.TestCase):
    """The planner fix is useless if the graph leaves the chain orphaned."""

    def _graphs(self):
        yield "otr_canonical.json", json.loads(
            (ROOT / "workflows" / "otr_canonical.json").read_text(encoding="utf-8"))
        for path in sorted((ROOT / "workflows" / "variants").glob("*.json")):
            if path.name.endswith(".env.json"):
                continue
            yield path.name, json.loads(path.read_text(encoding="utf-8"))

    def test_every_shipped_graph_gates_its_audio_chain_to_the_validator(self):
        orphaned = []
        for name, graph in self._graphs():
            byid = {n["id"]: n for n in graph["nodes"]}
            heads = [n for n in graph["nodes"] if n.get("type") == AUDIO_HEAD]
            validators = [n["id"] for n in graph["nodes"]
                          if n.get("type") == VALIDATOR]
            if not heads or not validators:
                continue
            for head in heads:
                names = [i.get("name") for i in (head.get("inputs") or [])]
                if "gate_in" not in names:
                    orphaned.append("%s: %s has no gate_in" % (name, AUDIO_HEAD))
                    continue
                link = head["inputs"][names.index("gate_in")].get("link")
                src = [row[1] for row in graph["links"] if row[0] == link]
                if not src or src[0] not in validators:
                    orphaned.append("%s: gate_in link=%s src=%s (validators %s)"
                                    % (name, link, src, validators))
        self.assertFalse(
            orphaned,
            "these graphs leave the audio chain ungated, so voices may run "
            "before validation and the music engine cannot be planned:\n  "
            + "\n  ".join(orphaned[:10]))


if __name__ == "__main__":
    unittest.main()
