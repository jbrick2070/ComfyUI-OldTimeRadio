"""EVERY path the render chain writes must fit Windows MAX_PATH.

PBUG-20260907-01. A 40-minute render on an 8 GB 4060 wrote the script, the
voices, the music, every still and all eight LTX video beats, then died at the
caption burn with `FileNotFoundError: [Errno 2]` naming a directory that plainly
existed. Errno 2 is what Windows returns past MAX_PATH -- the overflow surfaces
as a missing parent, so the message points at the wrong thing.

WHY THIS TEST IS CHAIN-WIDE AND NOT PER NODE. When the caption stage was fixed
in isolation, FOUR more stages were still over the line and nobody knew, because
each node owned its own arithmetic:

    ProcgenBlend __nobars_tmp   266   over
    ProcgenBlend __bars_tmp     264   over
    CreditsRoll .concat.txt     265   over
    CreditsRoll .scroll.png     260   over
    Mux _final.mp4              260   over

Three of those had never fired only because the procgen node was bypassed on
that run and the render never reached the mux. That is luck, not correctness.
Worse, `otr_credits_roll` ALREADY had a compaction and it was self-defeating:
its compacted tuple kept `joined` unchanged while its own fits() check derived
`.concat.txt` and `_final.mp4` from `joined`, so the compact set failed for
exactly the reason the legacy set did and it silently returned the long names.

So the contract under test is the CHAIN, walked end to end with the real episode
id that failed, on the real ComfyUI Desktop output depth. A node that regresses
its own naming fails here even if its own module tests still pass.

Pure path arithmetic: nothing is written, rendered, or spawned.
"""
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nodes._otr_shared.pathbudget import (WINDOWS_PATH_BUDGET,  # noqa: E402
                                          compact_artifact, compact_scratch,
                                          path_length)

#: The real failing case, 2026-09-07 00:12. ComfyUI Desktop spends 96 units
#: before `episodes\` ends, and the episode id is then spelled TWICE.
DEEP_EPISODES_ROOT = (
    r"C:\Users\jeffr\AppData\Local\Comfy-Desktop\ComfyUI-Installs"
    r"\ComfyUI\ComfyUI\output\otr\episodes")
EPISODE_ID = "signal_lost_a_play_to_drown_the_murder_accusation_20260906_234753"

#: Windows fails AT 260 (the limit includes the terminating NUL).
HARD_LIMIT = 260


def _chain(episodes_root=DEEP_EPISODES_ROOT, episode_id=EPISODE_ID):
    """Every path the chain writes, in render order, as (label, path)."""
    from nodes.otr_caption_burn import _ass_sidecar_path, _captioned_name
    from nodes.otr_credits_roll import _credits_artifact_paths

    ep = os.path.join(episodes_root, episode_id)
    out = []

    silent = os.path.join(ep, episode_id + "_silent.mp4")
    out += [("silent", silent), ("silent.qa.json", silent + ".qa.json")]

    src = Path(silent)
    blend = compact_artifact(str(src.parent), src.stem, "_procgen_blended",
                             src.suffix)
    out.append(("procgen_blended", blend))
    bp = Path(blend)
    for tag in ("__nobars_tmp", "__bars_tmp"):
        out.append(("procgen" + tag,
                    compact_scratch(str(bp.parent), bp.stem + tag, bp.suffix)))

    captioned = _captioned_name(ep, bp.stem)
    out += [("captioned", captioned), ("captions.ass", _ass_sidecar_path(captioned))]

    clip, backdrop, joined = _credits_artifact_paths(captioned)
    out += [("credits_clip", clip), ("credits_backdrop", backdrop),
            ("credits.base.png", clip + ".base.png"),
            ("credits.scroll.png", clip + ".scroll.png"),
            ("with_credits", joined),
            ("concat_list",
             compact_scratch(os.path.dirname(joined),
                             os.path.basename(joined) + ".concat", ".txt"))]

    out.append(("final",
                compact_artifact(ep,
                                 os.path.splitext(os.path.basename(joined))[0],
                                 "_final", ".mp4")))
    return out


@unittest.skipIf(os.name != "nt", "MAX_PATH is a Windows limit")
class RenderChainBudgetTests(unittest.TestCase):
    def test_no_stage_exceeds_the_hard_limit(self):
        """The one that would have saved the 40-minute render."""
        over = [(label, path_length(p)) for label, p in _chain()
                if path_length(p) >= HARD_LIMIT]
        self.assertFalse(
            over,
            "these stages exceed Windows MAX_PATH (%d) and will fail with "
            "FileNotFoundError on a deep install:\n  %s"
            % (HARD_LIMIT, "\n  ".join("%s: %d" % row for row in over)))

    def test_no_stage_exceeds_the_budget(self):
        """The budget is 250, not 260, and the ten units are load-bearing: a
        name that merely fits gets handed to a later stage that appends to it.
        `_final.mp4` landed on exactly 260 that way -- legal-looking and dead."""
        over = [(label, path_length(p)) for label, p in _chain()
                if path_length(p) > WINDOWS_PATH_BUDGET]
        self.assertFalse(
            over,
            "over the %d-unit budget:\n  %s"
            % (WINDOWS_PATH_BUDGET,
               "\n  ".join("%s: %d" % row for row in over)))

    def test_the_fixture_still_reproduces_the_original_overflow(self):
        """Guards the guard. If the id or the root shortened, this suite would
        pass while proving nothing -- the naive names must still overflow."""
        ep = os.path.join(DEEP_EPISODES_ROOT, EPISODE_ID)
        naive = os.path.join(
            ep, EPISODE_ID + "_silent_procgen_blended_captioned.ass")
        self.assertGreaterEqual(
            path_length(naive), HARD_LIMIT,
            "the fixture no longer reproduces the production overflow")

    def test_every_deliverable_still_reduces_to_the_episode_id(self):
        """Compaction may drop STAGE suffixes and must never touch identity.

        Both otr_credits_roll and otr_master_audio_mux recover the episode id by
        stripping those suffixes from the stem; a compacted name that no longer
        reduces to the id would silently drop them into their legacy branches --
        which is PBUG-20260904-06, a name-bound reader refusing a renamed file.
        """
        from nodes._otr_shared.pathbudget import strip_stage_suffixes

        deliverables = {"silent", "procgen_blended", "captioned",
                        "with_credits", "final"}
        for label, p in _chain():
            if label not in deliverables:
                continue
            stem = os.path.splitext(os.path.basename(p))[0]
            reduced = strip_stage_suffixes(stem)
            # `_final` is not a pipeline stage suffix; peel it first.
            if reduced.endswith("_final"):
                reduced = strip_stage_suffixes(reduced[: -len("_final")])
            self.assertEqual(
                reduced.casefold(), EPISODE_ID.casefold(),
                "%s (%s) no longer reduces to the episode id" % (label, stem))

    def test_short_installs_keep_their_ordinary_names(self):
        """The compaction is a fallback, not a rename. A shallow install must
        produce byte-identical names to before this change."""
        rows = dict(_chain(episodes_root=r"C:\otr", episode_id="ep1"))
        self.assertTrue(rows["procgen_blended"].endswith("ep1_silent_procgen_blended.mp4"),
                        rows["procgen_blended"])
        self.assertTrue(rows["captioned"].endswith(
            "ep1_silent_procgen_blended_captioned.mp4"), rows["captioned"])
        self.assertTrue(rows["concat_list"].endswith(".concat.txt"),
                        rows["concat_list"])


if __name__ == "__main__":
    unittest.main()
