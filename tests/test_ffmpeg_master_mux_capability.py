"""The master mux asks before spending, like captions do (PBUG-20260913-03).

A pod on Ubuntu 22.04's ffmpeg 4.4 rendered two whole episodes and left 0-byte
finals: the mux copies the PCM master into the MP4 losslessly, which only
FFmpeg 6.1+ can write, and nothing asked that question up front. These tests
pin the gap sentence to the probe's answer without spawning any ffmpeg.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "nodes"))
from _otr_shared import ffmpeg as FF  # noqa: E402

_REAL_PROBE = FF.probe_ffmpeg_capabilities


class MasterMuxGap(unittest.TestCase):
    def setUp(self):
        FF._CAPABILITY_CACHE.clear()

    def tearDown(self):
        FF._CAPABILITY_CACHE.clear()
        FF.probe_ffmpeg_capabilities = _REAL_PROBE

    def _with(self, caps):
        FF.probe_ffmpeg_capabilities = lambda path=None: dict(caps)

    def test_no_ffmpeg_names_the_install_not_the_mux(self):
        self._with({"path": None, FF.CAPTION_FILTER: None, FF.CAPTION_ENCODER: None,
                    FF.MASTER_MUX_PCM: None})
        gap = FF.master_mux_support_gap()
        self.assertIsNotNone(gap)
        self.assertIn("no ffmpeg was found", gap)
        self.assertIn("6.1", gap)

    def test_a_build_that_cannot_write_pcm_into_mp4_is_named_with_the_fix(self):
        self._with({"path": "/usr/bin/ffmpeg", FF.CAPTION_FILTER: True,
                    FF.CAPTION_ENCODER: True, FF.MASTER_MUX_PCM: False})
        gap = FF.master_mux_support_gap("/usr/bin/ffmpeg")
        self.assertIsNotNone(gap)
        self.assertIn("/usr/bin/ffmpeg", gap)
        self.assertIn("6.1", gap)
        self.assertIn("OTR_FFMPEG", gap)

    def test_a_capable_build_has_no_gap(self):
        self._with({"path": "/usr/local/bin/ffmpeg", FF.CAPTION_FILTER: True,
                    FF.CAPTION_ENCODER: True, FF.MASTER_MUX_PCM: True})
        self.assertIsNone(FF.master_mux_support_gap("/usr/local/bin/ffmpeg"))

    def test_an_unanswerable_probe_does_not_refuse(self):
        # None is "we could not ask", never "it said no" -- the caption gap's
        # rule, kept identical here so a failed subprocess cannot refuse a box.
        self._with({"path": "/opt/ffmpeg", FF.CAPTION_FILTER: None,
                    FF.CAPTION_ENCODER: None, FF.MASTER_MUX_PCM: None})
        self.assertIsNone(FF.master_mux_support_gap("/opt/ffmpeg"))

    def test_a_dead_path_answers_none_for_the_mux_capability_too(self):
        caps = FF.probe_ffmpeg_capabilities("/nonexistent/ffmpeg")
        self.assertIsNone(caps["path"])
        self.assertIsNone(caps[FF.MASTER_MUX_PCM])


class TheGapIsActuallyAsked(unittest.TestCase):
    """A probe with no caller is this repo's most repeated defect: correct,
    tested code that nothing reaches. `caption_support_gap` sat built and
    tested with zero callers for days. These assert the CALL at its real
    sites -- the one job source inspection is the right tool for."""

    ROOT = Path(__file__).resolve().parents[1] / "nodes"

    def test_the_validator_asks_before_anything_is_fetched(self):
        src = (self.ROOT / "_otr_workflow_validator.py").read_text(encoding="utf-8")
        self.assertIn("_mux_gap = master_mux_gap_for_prompt(prompt)", src,
                      "the validator must ask the mux question for the graph")
        ask_at = src.index("_mux_gap = master_mux_gap_for_prompt(prompt)")
        # Beside the story admission, which is documented as running before any
        # asset is fetched in both branches. A question asked after the
        # spending buys nothing.
        admit_at = src.index("self._admit_story_input(prompt, unique_id)")
        self.assertLess(admit_at, ask_at)
        self.assertLess(
            ask_at,
            src.index("from ._workflow_validation import validate_workflow_contract"),
            "the question comes before the contract audit, not after")

    def test_the_mux_asks_before_it_spawns_ffmpeg(self):
        src = (self.ROOT / "otr_master_audio_mux.py").read_text(encoding="utf-8")
        self.assertIn("master_mux_support_gap(", src,
                      "the mux itself must refuse rather than emit a 0-byte MP4")


class OnlyAGraphThatEndsAtAMux(unittest.TestCase):
    """Codex's finding on the first cut, and it held: asking the question of
    every run refuses work that never mixes audio. A validator -> writer ->
    freeze wiring makes a script and no media; a text-only replay makes
    neither. The graph says which this is, and the validator can see it."""

    def setUp(self):
        import _otr_workflow_validator as WV
        self.WV = WV
        self._real = FF.master_mux_support_gap
        FF.master_mux_support_gap = lambda _p=None: "PRETEND THIS BUILD CANNOT"

    def tearDown(self):
        FF.master_mux_support_gap = self._real

    @staticmethod
    def _prompt(*nodes):
        return {str(i): n for i, n in enumerate(nodes)}

    def test_a_script_only_graph_is_not_asked(self):
        p = self._prompt({"class_type": "OTR_LedgerScriptWriter", "inputs": {}},
                         {"class_type": "OTR_LedgerFreezeCascade", "inputs": {}})
        self.assertIsNone(self.WV.master_mux_gap_for_prompt(p))

    def test_a_graph_with_a_mux_is_refused_on_a_build_that_cannot_finish(self):
        p = self._prompt({"class_type": "OTR_MasterAudioMux",
                          "inputs": {"output_path": ""}})
        self.assertEqual(self.WV.master_mux_gap_for_prompt(p),
                         "PRETEND THIS BUILD CANNOT")

    def test_an_empty_destination_is_the_default_which_is_an_mp4(self):
        p = self._prompt({"class_type": "OTR_MasterAudioMux", "inputs": {}})
        self.assertIsNotNone(self.WV.master_mux_gap_for_prompt(p))

    def test_a_matroska_destination_is_not_gated(self):
        p = self._prompt({"class_type": "OTR_MasterAudioMux",
                          "inputs": {"output_path": "/tmp/ep_final.mkv"}})
        self.assertIsNone(self.WV.master_mux_gap_for_prompt(p))

    def test_no_prompt_at_all_is_never_a_refusal(self):
        self.assertIsNone(self.WV.master_mux_gap_for_prompt(None))
        self.assertIsNone(self.WV.master_mux_gap_for_prompt({}))
        self.assertIsNone(self.WV.master_mux_gap_for_prompt("not a prompt"))


class OnlyTheContainersThatCare(unittest.TestCase):
    """Codex's finding on the first cut of this fix, and it held: the mux takes
    a caller-chosen destination, and this pack's own byte-identity test muxes to
    .mkv. Matroska carried PCM long before FFmpeg 6.1, so gating an MKV mux on
    the MP4 capability would be a fault invented by the guard."""

    def setUp(self):
        import otr_master_audio_mux as MUX
        self.MUX = MUX
        self._real = FF.master_mux_support_gap
        FF.master_mux_support_gap = lambda _p=None: "PRETEND THIS BUILD CANNOT"

    def tearDown(self):
        FF.master_mux_support_gap = self._real

    def _mux_to(self, suffix):
        """Call the mux with absent inputs; return the refusal text."""
        with self.assertRaises(ValueError) as caught:
            self.MUX.mux_master_audio(
                "no_such_video.mp4", "no_such_master.wav",
                "destination" + suffix)
        return str(caught.exception)

    def test_an_mp4_destination_is_refused_on_a_build_that_cannot_write_it(self):
        self.assertIn("PRETEND THIS BUILD CANNOT", self._mux_to(".mp4"))

    def test_a_matroska_destination_is_not_refused(self):
        # It still fails -- the inputs do not exist -- but NOT for ffmpeg's
        # MP4 capability, which Matroska does not need.
        self.assertNotIn("PRETEND THIS BUILD CANNOT", self._mux_to(".mkv"))

    def test_the_gated_containers_are_the_isobmff_family(self):
        self.assertIn(".mp4", self.MUX.PCM_STRICT_CONTAINERS)
        self.assertIn(".mov", self.MUX.PCM_STRICT_CONTAINERS)
        self.assertNotIn(".mkv", self.MUX.PCM_STRICT_CONTAINERS)
        self.assertNotIn(".wav", self.MUX.PCM_STRICT_CONTAINERS)


class AProbeThatCannotRunSaysSoInstead(unittest.TestCase):
    """Codex's other finding: the probe needs lavfi, anullsrc and a writable
    temp dir; the real mux needs none of them. Without a control, a machine
    with lavfi disabled or a read-only TMPDIR answered "this build cannot write
    PCM into MP4" and was refused for a capability it actually had."""

    def setUp(self):
        FF._CAPABILITY_CACHE.clear()
        self._real = FF._write_silence

    def tearDown(self):
        FF._write_silence = self._real
        FF._CAPABILITY_CACHE.clear()

    def _answers(self, wav, mp4):
        calls = []

        def fake(resolved, proc, container, suffix):
            calls.append(container)
            return wav if container == "wav" else mp4

        FF._write_silence = fake
        return FF._probe_pcm_in_mp4("/fake/ffmpeg", None), calls

    def test_a_failing_control_means_we_could_not_ask(self):
        answer, calls = self._answers(wav=False, mp4=False)
        self.assertIsNone(answer, "a broken probe is not evidence about ffmpeg")
        self.assertEqual(calls, ["wav"], "the MP4 attempt is pointless then")

    def test_an_unrunnable_control_means_we_could_not_ask(self):
        answer, _ = self._answers(wav=None, mp4=True)
        self.assertIsNone(answer)

    def test_a_passing_control_makes_a_failing_mp4_mean_the_container(self):
        answer, calls = self._answers(wav=True, mp4=False)
        self.assertIs(answer, False)
        self.assertEqual(calls, ["wav", "mp4"])

    def test_both_passing_is_a_capable_build(self):
        answer, _ = self._answers(wav=True, mp4=True)
        self.assertIs(answer, True)


if __name__ == "__main__":
    unittest.main()
