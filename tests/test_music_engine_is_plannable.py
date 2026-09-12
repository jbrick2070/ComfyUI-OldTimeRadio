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
        "82": {"class_type": "OTR_AnnouncerVoice", "inputs": {"gate_in": ["81", 2]}},
        "83": {"class_type": MUSIC,
               "inputs": {"gate_in": ["82", 2], "engine": "stable_audio_3"}},
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

    def test_the_walk_does_not_care_which_output_slot_carries_the_gate(self):
        """The bug in the FIRST cut of this fix, and the reason the live run
        still logged 'READY engines=ltx_8gb,z_image_turbo' with the graph
        correctly wired.

        The old code required `gate[1] == 0`. That is true of the validator's
        single output and FALSE of every chain hop -- the audio nodes pass their
        gate on from output slot 2, so the shipped prompt reads ['81', 2] and
        ['82', 2]. Requiring slot 0 rejected exactly the hops the walk exists to
        follow. `gate_in` is keyed by NAME in the API prompt, so which slot
        feeds it is the source node's business.
        """
        for slot in (0, 1, 2, 7):
            prompt = _prompt()
            prompt["82"]["inputs"]["gate_in"] = ["81", slot]
            prompt["83"]["inputs"]["gate_in"] = ["82", slot]
            self.assertIn("stable_audio_3", _plan(prompt)["engines"],
                          "gate carried on output slot %d was not followed" % slot)

    def test_a_deep_chain_does_not_loop_forever(self):
        """The walk is iterative; a cycle in gate_in must terminate."""
        prompt = _prompt()
        prompt["81"]["inputs"]["gate_in"] = ["83", 2]   # 81 <- 83 <- 82 <- 81
        _plan(prompt)  # must return rather than hang


class PlannedEngineBecomesADownloadRequestTests(unittest.TestCase):
    """Being PLANNED is not the same as being FETCHED, and that gap was real.

    Once the gate fix let the planner see the music engine, the live run logged

        READY engines=ltx_8gb,stable_audio_3,z_image_turbo files=5

    -- the engine named, and still only the five VISUAL files requested.
    `native_requests` had branches for the image and video engines only, so
    nothing ever turned `stable_audio_3` into a request. Membership in
    `_COVERED` even suppressed the "coverage unavailable" note that would
    otherwise have said so out loud, which is why it looked handled.
    """

    class _FolderPaths:
        """Nothing is installed, so every request resolves to 'missing'."""

        @staticmethod
        def get_full_path(category, token):
            return None

        @staticmethod
        def get_folder_paths(category):
            return [str(ROOT / "_nonexistent" / category)]

    class _SA3:
        """The adapter surface the preflight uses: `resolve_ckpt()` -- the
        name it WILL load, which on an empty disk is the BASE fetch default
        (2026-09-12) -- and the text-encoder constant. `_CKPT` is the
        operator's override and is empty by default (PBUG-20260912-05); the
        preflight must never read it as a filename."""
        _CKPT = ""
        _FETCH_DEFAULT = "stable_audio_3_small_music_base.safetensors"
        _TENC = "t5gemma_b_b_ul2.safetensors"

        class StableAudio3Engine:
            @staticmethod
            def resolve_ckpt():
                return ("stable_audio_3_small_music_base.safetensors", True)

    def test_selecting_the_music_engine_requests_its_two_files(self):
        reqs = VA.native_requests({"stable_audio_3"},
                                  folder_paths=self._FolderPaths(),
                                  sa3=self._SA3(), env={})
        got = {(r["category"], r["token"]) for r in reqs}
        self.assertEqual(
            got,
            {("checkpoints", self._SA3._FETCH_DEFAULT),
             ("text_encoders", self._SA3._TENC)},
            "the music engine must request the checkpoint AND its text encoder")
        for r in reqs:
            self.assertIsNotNone(
                r["spec"], "%s has no MANIFEST spec, so it can never download"
                % r["token"])

    def test_both_files_are_in_the_allowlist(self):
        """download_verified refuses anything not in MANIFEST.values(), so a
        request whose spec is absent would fail at transfer rather than here."""
        for category, token in (("checkpoints", self._SA3._FETCH_DEFAULT),
                                ("text_encoders", self._SA3._TENC)):
            self.assertIn((category, token), VA.MANIFEST,
                          "%s/%s is not allowlisted" % (category, token))

    def test_it_refuses_rather_than_silently_skipping_without_an_adapter(self):
        with self.assertRaises(VA.VisualAssetError):
            VA.native_requests({"stable_audio_3"},
                               folder_paths=self._FolderPaths(),
                               sa3=None, env={})

    def test_the_real_adapter_on_an_empty_disk_requests_its_fetch_default(self):
        """THE WIRING, NOT THE STUB (cursor, finished-diff review 2026-09-12).
        The stub above mirrors the adapter's intended answer, so it cannot
        notice the adapter changing its mind. This hands `native_requests`
        the REAL engine module with nothing installed and asserts that the
        checkpoint it asks for is the engine's own fetch default -- a base
        file -- and that the manifest can supply it. Until 2026-09-12 this
        path requested the post-trained file on every fresh install."""
        import sys
        import types
        from nodes._otr_audio_engines import eng_stable_audio_3 as real_sa3
        fake = types.ModuleType("folder_paths")
        fake.get_full_path = lambda kind, name: None
        saved_ckpt = real_sa3._CKPT
        saved_module = sys.modules.get("folder_paths")
        real_sa3._CKPT = ""
        sys.modules["folder_paths"] = fake
        try:
            reqs = VA.native_requests({"stable_audio_3"},
                                      folder_paths=self._FolderPaths(),
                                      sa3=real_sa3, env={})
        finally:
            real_sa3._CKPT = saved_ckpt
            if saved_module is None:
                sys.modules.pop("folder_paths", None)
            else:
                sys.modules["folder_paths"] = saved_module
        ckpt = [r for r in reqs if r["category"] == "checkpoints"]
        self.assertEqual(len(ckpt), 1, reqs)
        self.assertEqual(ckpt[0]["token"], real_sa3._FETCH_DEFAULT)
        self.assertTrue(ckpt[0]["token"].endswith("_base.safetensors"),
                        "a fresh install must be sent to a BASE checkpoint")
        self.assertIsNotNone(ckpt[0]["spec"],
                             "the fetch default has no manifest spec, so it "
                             "could never download")


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
