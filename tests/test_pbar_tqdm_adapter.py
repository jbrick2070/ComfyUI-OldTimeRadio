"""The download progress adapter must satisfy the REAL huggingface_hub contract.

PBUG-20260906-08. alpha.25 wired a ComfyUI ProgressBar into
`auto_download_if_missing`. The adapter it used was a hand-rolled tqdm
look-alike that stored its counters privately and implemented only
update/close/__enter__/__exit__/__iter__. On the first cold-cache run of the
clean-install drill every LLM download died with:

    AttributeError: '_PBarTqdm' object has no attribute 'total'

because `snapshot_download` builds two parent bars from the class and then, per
file, executes `_snapshot_download._AggregatedTqdm.__init__`:

    reconstruct_progress.total = (reconstruct_progress.total or 0) + total
    transfer_progress.total = (transfer_progress.total or 0) + total
    reconstruct_progress.refresh()

That is an attribute READ, an attribute WRITE, and a method the stand-in did not
have. The 36.8 GB of visual assets in the same run succeeded, which is precisely
why nothing caught it earlier: the asset planner calls `hf_hub_download`
directly and never passes a `tqdm_class`, so only the LLM lane was affected.

WHY THESE TESTS DRIVE THE REAL LIBRARY. The bug survived because it is a
CONTRACT bug, not a logic bug. A test that constructs the adapter by hand and
asserts on its own mock would have passed against the broken version. So the
first test below calls huggingface_hub's own `_create_progress_bar` with the
exact keyword arguments `_snapshot_download` passes, and then replays the exact
aggregation lines. If huggingface_hub changes what it touches, this fails here
rather than on a user's first run.

Nothing here downloads anything or touches a model.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nodes"))

import _otr_model_catalog as CATALOG  # noqa: E402


class _RecordingPBar:
    """Stands in for comfy.utils.ProgressBar."""

    def __init__(self):
        self.calls = []

    def update_absolute(self, value, total=None):
        self.calls.append((value, total))


class _ThrowingPBar:
    def update_absolute(self, value, total=None):
        raise RuntimeError("the ProgressBar exploded")


def _hf_kwargs(name):
    """The exact kwargs huggingface_hub/_snapshot_download.py passes."""
    import logging

    return dict(
        log_level=logging.INFO,
        name=name,
        desc="Reconstructing (incomplete total...)",
        total=0,
        initial=0,
        unit="B",
        unit_scale=True,
        bar_format="{l_bar}{bar}",
    )


class RealHuggingfaceContractTests(unittest.TestCase):
    """The regression that matters: drive huggingface_hub's own constructor."""

    def setUp(self):
        try:
            from huggingface_hub.utils.tqdm import _create_progress_bar  # noqa: F401
        except Exception as exc:  # pragma: no cover
            self.skipTest("huggingface_hub not importable here: %s" % exc)

    def test_the_exact_aggregation_that_crashed_alpha25(self):
        from huggingface_hub.utils.tqdm import _create_progress_bar

        cls = CATALOG._make_pbar_tqdm_adapter(_RecordingPBar())
        reconstruct = _create_progress_bar(
            cls=cls, **_hf_kwargs("huggingface_hub.snapshot_download"))
        transfer = _create_progress_bar(
            cls=cls, **_hf_kwargs("huggingface_hub.snapshot_download.transfer"))

        # Verbatim from _snapshot_download._AggregatedTqdm.__init__, run once
        # per file in the repo. This is the line that raised AttributeError.
        for size in (12309866400, 8044982048, 335304388):
            reconstruct.total = (reconstruct.total or 0) + size
            transfer.total = (transfer.total or 0) + size
            reconstruct.refresh()

        self.assertEqual(reconstruct.total, 12309866400 + 8044982048 + 335304388)
        self.assertEqual(transfer.total, reconstruct.total)

    def test_progress_actually_reaches_the_comfyui_bar(self):
        """A bar that never forwards is the failure the disable=True version had:
        tqdm short-circuits update() when disabled, so display() never runs and
        the adapter silently reports nothing while looking correct."""
        from huggingface_hub.utils.tqdm import _create_progress_bar

        pbar = _RecordingPBar()
        cls = CATALOG._make_pbar_tqdm_adapter(pbar)
        bar = _create_progress_bar(cls=cls, **_hf_kwargs("x"))
        bar.total = 1000
        bar.update(250)
        bar.refresh()

        self.assertTrue(pbar.calls, "nothing was forwarded to the ProgressBar")
        value, total = pbar.calls[-1]
        self.assertEqual(total, 1000)
        self.assertEqual(value, 250)

    def test_nothing_is_written_to_the_console(self):
        """The bar is mirrored into the ComfyUI queue UI, not drawn as ASCII in
        the server log."""
        cls = CATALOG._make_pbar_tqdm_adapter(_RecordingPBar())
        bar = cls(total=100)
        bar.update(50)
        bar.refresh()
        self.assertEqual(bar.fp.getvalue(), "",
                         "the adapter drew a progress bar into its stream")
        bar.close()


class NeverBreaksTheDownloadTests(unittest.TestCase):
    """A progress indicator that kills a 24 GB download is worse than none."""

    def test_a_throwing_progressbar_does_not_propagate(self):
        cls = CATALOG._make_pbar_tqdm_adapter(_ThrowingPBar())
        bar = cls(total=100)
        bar.update(10)
        bar.refresh()
        bar.close()

    def test_none_and_zero_totals_survive(self):
        """total is None before aggregation starts and 0 at construction."""
        cls = CATALOG._make_pbar_tqdm_adapter(_RecordingPBar())
        bar = cls(total=0)
        bar.refresh()
        bar.total = None
        bar.refresh()
        bar.total = 0
        bar.refresh()
        bar.close()

    def test_use_after_close_and_context_manager(self):
        cls = CATALOG._make_pbar_tqdm_adapter(_RecordingPBar())
        bar = cls(total=10)
        bar.close()
        bar.update(1)
        with cls(total=10) as ctx:
            ctx.update(5)

    def test_a_name_kwarg_would_not_raise(self):
        """huggingface_hub injects name= only for its own tqdm subclass today.
        If that ever widens, vanilla tqdm answers TqdmKeyError, so the adapter
        drops the key defensively."""
        cls = CATALOG._make_pbar_tqdm_adapter(_RecordingPBar())
        bar = cls(total=10, name="huggingface_hub.snapshot_download")
        bar.close()


class SubclassShapeTests(unittest.TestCase):
    def test_it_is_a_real_tqdm_subclass(self):
        """The whole point of the fix: inherit the surface instead of guessing
        which attributes huggingface_hub will touch next."""
        from tqdm.std import tqdm as tqdm_base

        cls = CATALOG._make_pbar_tqdm_adapter(_RecordingPBar())
        self.assertTrue(issubclass(cls, tqdm_base))

    def test_the_attributes_hf_touches_are_public_and_writable(self):
        cls = CATALOG._make_pbar_tqdm_adapter(_RecordingPBar())
        bar = cls(total=5)
        for attr in ("total", "n", "desc"):
            self.assertTrue(hasattr(bar, attr), "missing public %r" % attr)
        for method in ("update", "refresh", "close", "set_description",
                       "display", "reset"):
            self.assertTrue(callable(getattr(bar, method, None)),
                            "missing method %r" % method)
        bar.total = 99          # the write that raised AttributeError
        self.assertEqual(bar.total, 99)
        bar.close()


if __name__ == "__main__":
    unittest.main()
