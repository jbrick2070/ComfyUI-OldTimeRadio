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
sys.path.insert(0, str(ROOT))

# Imported as part of the `nodes` PACKAGE, not flat off nodes/ as most of the
# suite does. auto_download_if_missing lazily runs `from ._otr_hf_auth import
# resolve_hf_token_runtime`, which raises "attempted relative import with no
# known parent package" under a flat import -- so the flat form can construct
# the adapter but can never reach the function that installs it.
from nodes import _otr_model_catalog as CATALOG  # noqa: E402


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


class LockInvariantTests(unittest.TestCase):
    """tqdm's refresh() leaks its lock if display() raises ANYTHING.

    Installed tqdm 4.70.0, std.py refresh():

        self._lock.acquire()
        self.display()
        self._lock.release()

    No try/finally. TqdmDefaultWriteLock holds th_lock as a CLASS attribute, so
    a single escaped exception blocks the next refresh() on every tqdm in the
    process -- a whole-server deadlock, not a lost progress bar. `except
    Exception` is not enough because KeyboardInterrupt is a BaseException.
    """

    def test_a_keyboardinterrupt_from_the_sink_does_not_escape(self):
        class _Interrupting:
            def update_absolute(self, value, total=None):
                raise KeyboardInterrupt

        cls = CATALOG._make_pbar_tqdm_adapter(_Interrupting())
        bar = cls(total=1000)
        bar.update(500)
        bar.refresh()   # would strand the lock if display() raised
        bar.close()

    def test_the_shared_tqdm_lock_still_works_afterwards(self):
        """The real symptom: an unrelated bar hangs. If the lock were stranded
        this call would block forever rather than fail."""
        from tqdm.std import tqdm as tqdm_base

        class _Interrupting:
            def update_absolute(self, value, total=None):
                raise KeyboardInterrupt

        cls = CATALOG._make_pbar_tqdm_adapter(_Interrupting())
        bar = cls(total=10)
        bar.update(1)
        bar.refresh()
        bar.close()

        import io as _io

        other = tqdm_base(total=3, file=_io.StringIO())
        other.update(1)
        other.refresh()
        other.close()


class CompletionTests(unittest.TestCase):
    """The bar must reach 100% even when nothing transferred.

    A file already in the blob cache returns before any progress object is
    built, so a resumed download aggregates only the remaining shards and a
    fully-cached repo aggregates nothing at all. Without an explicit write on
    the way out the node's bar sits where it was left while the download had
    actually finished.
    """

    def test_auto_download_drives_the_bar_to_complete(self):
        """The real function, with the downloader injected. Simulates the
        fully-cached case: the fake transfers nothing, so the mirrored bars
        never publish, and only the explicit completion write can finish it."""
        import tempfile

        pbar = _RecordingPBar()
        seen = {}

        def _fake_snapshot(**kwargs):
            seen.update(kwargs)
            return "C:/fake/snapshot"

        with tempfile.TemporaryDirectory() as tmp:
            out = CATALOG.auto_download_if_missing(
                "Qwen/Qwen3.5-4B",
                hub_root=Path(tmp),
                progress_pbar=pbar,
                _snapshot_download=_fake_snapshot,
            )

        self.assertEqual(out, "C:/fake/snapshot")
        self.assertIn("tqdm_class", seen, "the adapter must still be wired in")
        self.assertTrue(pbar.calls, "the bar never moved at all")
        self.assertEqual(
            pbar.calls[-1], (1000, 1000),
            "the last write must complete the bar; without it a resumed or "
            "fully-cached download leaves the node's bar part-way while the "
            "download has actually finished")

    def test_a_throwing_bar_cannot_fail_the_download(self):
        """The completion write is on the success path -- if it could raise, a
        finished 24 GB download would be reported as a failed node."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out = CATALOG.auto_download_if_missing(
                "Qwen/Qwen3.5-4B",
                hub_root=Path(tmp),
                progress_pbar=_ThrowingPBar(),
                _snapshot_download=lambda **kw: "C:/fake/snapshot",
            )
        self.assertEqual(out, "C:/fake/snapshot")

    def test_the_completion_write_source_is_present(self):
        """Guards the call itself, since the surrounding function needs a real
        hub root to execute end to end here."""
        import inspect

        source = inspect.getsource(CATALOG.auto_download_if_missing)
        self.assertIn("update_absolute(1000, 1000)", source,
                      "auto_download_if_missing must complete the bar on exit")
        self.assertIn("except BaseException", source,
                      "the completion write must not be able to fail a download")


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
