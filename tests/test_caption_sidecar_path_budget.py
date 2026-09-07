"""The caption sidecar must not blow Windows MAX_PATH on a deep install root.

PBUG-20260907-01, found by the alpha.25 clean-install drill. A 40-minute render
on the 4060 completed the script, the voices, the music, every image and all
eight LTX video beats, and then died at the caption burn with:

    OTR_CaptionBurn: no captions (could not write
    ...signal_lost_..._silent_procgen_blended_captioned.ass:
    FileNotFoundError: [Errno 2] No such file or directory)

The directory in that message plainly existed. Errno 2 is what Windows returns
for a path over MAX_PATH, because the failure surfaces as a missing parent.

THE ARITHMETIC, measured on the box rather than argued. A ComfyUI Desktop
install puts the output tree at
`AppData\\Local\\Comfy-Desktop\\ComfyUI-Installs\\ComfyUI\\ComfyUI\\output`, so
96 characters are spent before `episodes\\` ends. The episode id then appears
TWICE -- once as the folder, once as the filename stem -- and a 65-character id
plus `_silent_procgen_blended_captioned.ass` reached 264 units against a 260
limit. The 254-unit `_silent_procgen_blended.mp4` sitting beside it wrote fine.
Four characters decided a 40-minute render.

WHY ONLY THE SIDECAR IS COMPACTED HERE. The `.ass` is scratch: it is consumed
once by the very next ffmpeg call, and `_ass_filter_arg` hands ffmpeg the
BASENAME with cwd set to the folder, so its name has no contract with anything.
The captioned MP4 is NOT scratch -- `otr_credits_roll` identifies its input by
matching the stem against the episode id -- which is why `_credits_artifact_paths`
already states the rule this module now follows: compact the generated scratch,
never the episode identity.

Pure path arithmetic. Nothing here writes a file, renders, or spawns ffmpeg.
"""
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nodes.otr_caption_burn import (  # noqa: E402
    _WINDOWS_PATH_BUDGET, _ass_sidecar_path, _path_fits)

# The exact episode that failed on the 4060, 2026-09-07 00:12.
DEEP_ROOT = (r"C:\Users\jeffr\AppData\Local\Comfy-Desktop\ComfyUI-Installs"
             r"\ComfyUI\ComfyUI\output\otr\episodes")
LONG_ID = "signal_lost_a_play_to_drown_the_murder_accusation_20260906_234753"
LONG_OUT = os.path.join(DEEP_ROOT, LONG_ID,
                        LONG_ID + "_silent_procgen_blended_captioned.mp4")


class SidecarBudgetTests(unittest.TestCase):
    def test_the_production_failure_now_fits(self):
        """The regression itself: this exact output path produced a 264-unit
        sidecar and FileNotFoundError."""
        naive = os.path.splitext(os.path.abspath(LONG_OUT))[0] + ".ass"
        self.assertGreater(len(naive), 260,
                           "fixture no longer reproduces the overflow")
        chosen = _ass_sidecar_path(LONG_OUT)
        self.assertTrue(_path_fits(chosen),
                        "sidecar is still over the %d-unit budget: %d"
                        % (_WINDOWS_PATH_BUDGET, len(chosen)))

    def test_it_stays_in_the_episode_folder(self):
        """ffmpeg is given cwd = the sidecar's folder, and a stray file in the
        server's working directory is the bug the ass_out plumbing exists to
        prevent. Compacting must not move the file."""
        chosen = _ass_sidecar_path(LONG_OUT)
        self.assertEqual(os.path.dirname(chosen),
                         os.path.dirname(os.path.abspath(LONG_OUT)))
        self.assertTrue(chosen.endswith(".ass"))

    def test_it_is_deterministic(self):
        """A re-run must overwrite its own sidecar rather than litter one per
        attempt, and two burns in one folder must not collide."""
        self.assertEqual(_ass_sidecar_path(LONG_OUT), _ass_sidecar_path(LONG_OUT))
        other = LONG_OUT.replace("_captioned.mp4", "_other_captioned.mp4")
        self.assertNotEqual(_ass_sidecar_path(LONG_OUT), _ass_sidecar_path(other))

    def test_short_paths_keep_their_ordinary_name(self):
        """Nothing that currently works may change name -- the compaction is a
        fallback, not a rename."""
        short = os.path.join("C:" + os.sep, "ep", "short_captioned.mp4")
        self.assertEqual(_ass_sidecar_path(short),
                         os.path.splitext(os.path.abspath(short))[0] + ".ass")

    def test_the_name_carries_no_filtergraph_syntax(self):
        """The compacted name is interpolated into an UNQUOTED ffmpeg
        filtergraph via ass={name}; `_reject_filtergraph_syntax` would refuse
        any of , ; : = [ ] ' \\ ."""
        from nodes.otr_caption_burn import _FILTERGRAPH_SYNTAX

        name = os.path.basename(_ass_sidecar_path(LONG_OUT))
        self.assertFalse(set(name) & _FILTERGRAPH_SYNTAX,
                         "compacted sidecar name would break the filtergraph")

    def test_the_budget_matches_the_credits_node(self):
        """Two modules solving one rule is how the first one drifted. If these
        ever disagree, the render fails in whichever stage has the larger
        number."""
        import nodes.otr_credits_roll as credits

        source = Path(credits.__file__).read_text(encoding="utf-8")
        self.assertIn("<= 250", source,
                      "the credits node's budget moved; keep the two in step")
        self.assertEqual(_WINDOWS_PATH_BUDGET, 250)


if __name__ == "__main__":
    unittest.main()
