"""Tiny-file downloader contracts: no ComfyUI imports or real network calls."""
from contextlib import contextmanager
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "nodes" / "_otr_visual_asset_download.py"
SPEC = importlib.util.spec_from_file_location("visual_asset_download_tested", MODULE_PATH)
download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download)


class ComfyLikeInterrupt(BaseException):
    """Match Comfy's interruption hierarchy without importing its runtime."""


class FetchVerifiedTests(unittest.TestCase):
    """`fetch_verified` is the LIBRARY transport, added 2026-09-07.

    The hand-rolled `download_verified` owned a socket (removed 2026-10-08, with
    its tests), and comparing four published zips showed the Comfy Registry
    scanner Flags exactly the versions that ship a bespoke downloader -- nine of
    fourteen, and a Flagged version never resolves as `latest_version`, so
    Manager's default button never offers it. `fetch_verified` hands the bytes to
    huggingface_hub and keeps every guarantee that made the loop trustworthy. The
    helpers it shares -- lock, cancellation, validation, no-clobber publish -- are
    pinned here through it.

    THE POINT OF THESE TESTS: the library verified its own TRANSFER; it did not
    verify our pinned CONTENT. A stale or poisoned cache entry, a truncated
    file, or a fetch that returns the wrong path must all still be refused here.
    """

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.root, ignore_errors=True))
        self.body = b"visual weight fixture payload"
        self.source = self.root / "cache" / "blob.safetensors"
        self.source.parent.mkdir(parents=True, exist_ok=True)
        self.source.write_bytes(self.body)
        self.destination = self.root / "models" / "checkpoints" / "final.safetensors"
        self.spec = {"repo_id": "Comfy-Org/fixture", "filename": "blob.safetensors"}
        self.metadata = {
            "commit": "a" * 40,
            "sha256": hashlib.sha256(self.body).hexdigest(),
            "size": len(self.body),
        }

    def invoke(self, **kwargs):
        arguments = {
            "fetch": lambda spec, meta, progress=None: str(self.source),
            "disk_free": lambda _p: self.metadata["size"] + download.DISK_MARGIN_BYTES,
        }
        arguments.update(kwargs)
        return download.fetch_verified(
            self.spec, self.destination, self.metadata, **arguments)

    @contextmanager
    def failing_unlock(self):
        if os.name == "nt":
            import msvcrt as lock_module
            function_name = "locking"
            unlock_mode = lock_module.LK_UNLCK
        else:
            import fcntl as lock_module
            function_name = "flock"
            unlock_mode = lock_module.LOCK_UN
        real_lock = getattr(lock_module, function_name)

        def injected_lock(*args):
            if args[1] == unlock_mode:
                raise PermissionError("injected unlock failure")
            return real_lock(*args)

        with mock.patch.object(lock_module, function_name, side_effect=injected_lock):
            yield

    def test_a_verified_fetch_publishes_the_final(self):
        receipt = self.invoke()
        self.assertEqual(receipt["status"], "downloaded")
        self.assertTrue(receipt["verified"])
        self.assertEqual(receipt["bytes_verified"], len(self.body))
        self.assertEqual(receipt["sha256"], self.metadata["sha256"])
        self.assertEqual(self.destination.read_bytes(), self.body)
        # No URL is returned in a receipt (a signed one must never reach a log),
        # and the .lock file deliberately persists: its existence is not a lock.
        self.assertNotIn("url", receipt)
        self.assertTrue(
            self.destination.with_name(self.destination.name + ".lock").exists())

    def test_a_content_hash_mismatch_is_refused(self):
        """The library checked its own transfer, not our pinned sha256."""
        self.metadata["sha256"] = "b" * 64
        with self.assertRaises(download.VisualAssetDownloadError):
            self.invoke()
        self.assertFalse(self.destination.exists())

    def test_a_truncated_fetch_is_refused(self):
        self.source.write_bytes(self.body[:-3])
        with self.assertRaises(download.VisualAssetDownloadError):
            self.invoke()
        self.assertFalse(self.destination.exists())

    def test_a_fetch_that_returns_no_file_is_refused(self):
        with self.assertRaises(download.VisualAssetDownloadError):
            self.invoke(fetch=lambda s, m, progress=None: str(self.root / "absent"))
        self.assertFalse(self.destination.exists())

    def test_an_existing_destination_is_preserved_untouched(self):
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.destination.write_bytes(b"operator's own file")
        receipt = self.invoke(fetch=self._forbidden_fetch)
        self.assertEqual(receipt["status"], "exists")
        self.assertFalse(receipt["verified"])
        self.assertEqual(self.destination.read_bytes(), b"operator's own file")
        # A BROKEN symlink is lexists=True / exists=False, and it is preserved
        # too. Simulated, so no symlink (or elevated Windows privilege) is needed.
        self.destination.unlink()
        with mock.patch.object(download.os.path, "lexists", return_value=True):
            receipt = self.invoke(fetch=self._forbidden_fetch)
        self.assertEqual(receipt["status"], "exists")
        self.assertFalse(self.destination.exists())

    def _forbidden_fetch(self, *_args, **_kwargs):
        raise AssertionError("an existing destination must not be re-fetched")

    def test_insufficient_disk_refuses_before_fetching(self):
        one_byte_short = self.metadata["size"] + download.DISK_MARGIN_BYTES - 1
        with self.assertRaisesRegex(download.VisualAssetDownloadError, "insufficient"):
            self.invoke(fetch=self._forbidden_fetch,
                        disk_free=lambda _p: one_byte_short)
        self.assertFalse(self.destination.exists())

    def test_the_allowlist_and_pin_validation_still_run(self):
        invalid = [
            ("size", True), ("size", 0), ("size", -1), ("size", 2.0),
            ("commit", "a" * 39), ("commit", "g" * 40),
            ("sha256", "a" * 63), ("sha256", "g" * 64),
        ]
        for key, value in invalid:
            with self.subTest(key=key, value_type=type(value).__name__):
                metadata = dict(self.metadata, **{key: value})
                with self.assertRaises(ValueError):
                    download.fetch_verified(self.spec, self.destination, metadata,
                                            fetch=self._forbidden_fetch)
        self.assertFalse(self.destination.exists())

    def test_cancellation_is_honoured_before_any_publish(self):
        class Stop(Exception):
            pass

        def cancel():
            raise Stop()

        with self.assertRaises(Stop):
            self.invoke(fetch=self._forbidden_fetch, cancel=cancel)
        self.assertFalse(self.destination.exists())
        # The boolean seam: a truthy return becomes DownloadCancelled. Either
        # way the check precedes every filesystem effect, so no folder appears.
        with self.assertRaises(download.DownloadCancelled):
            self.invoke(fetch=self._forbidden_fetch, cancel=lambda: True)
        self.assertFalse(self.destination.parent.exists())

    def test_progress_reaches_full_only_after_verification(self):
        seen = []
        self.invoke(progress=lambda done, total: seen.append((done, total)))
        self.assertEqual(seen[0], (0, len(self.body)))
        self.assertEqual(seen[-1], (len(self.body), len(self.body)))

    def test_it_reports_resume_support(self):
        """The loop could not resume and said so in the planner's own log line;
        the library can, and the receipt records the difference."""
        self.assertTrue(self.invoke()["resume_supported"])

    def test_a_fetch_that_returns_the_caches_symlink_publishes_the_real_file(self):
        """PBUG-20260926-02, reproduced in the Hugging Face cache's own shape:
        the fetch returns `snapshots/<commit>/<file>`, a RELATIVE symlink to
        `blobs/<sha256>`. The published final must be a real file that reads
        the payload -- not a copy of that symlink, whose relative target means
        nothing from the model folder. (On Windows os.link does not follow
        symlinks, which is how the broken link was made.)"""
        blob = self.root / "hub" / "models--fixture" / "blobs" / self.metadata["sha256"]
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(self.body)
        snapshot = (self.root / "hub" / "models--fixture" / "snapshots"
                    / self.metadata["commit"] / "blob.safetensors")
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(os.path.join("..", "..", "blobs", self.metadata["sha256"]), snapshot)
        except (OSError, NotImplementedError) as exc:
            self.skipTest("this machine cannot create symlinks (%s)" % exc)
        receipt = self.invoke(fetch=lambda spec, meta, progress=None: str(snapshot))
        self.assertEqual(receipt["status"], "downloaded")
        self.assertFalse(os.path.islink(self.destination),
                         "the final is a symlink copied out of the cache")
        self.assertTrue(os.path.isfile(self.destination))
        self.assertEqual(self.destination.read_bytes(), self.body)

    def test_a_cross_device_fetch_still_publishes_via_copy(self):
        """The fetched file may live in an HF cache on another VOLUME, where
        os.link raises EXDEV. This caught a real gap: the first cut's fallback
        copied to a temp and then called os.link again, so a filesystem with no
        hard links at all raised a bare OSError instead of publishing.
        """
        def cross_device(src, dst):
            raise OSError(18, "Invalid cross-device link")

        with mock.patch.object(download.os, "link", side_effect=cross_device):
            receipt = self.invoke()
        self.assertEqual(receipt["status"], "downloaded")
        self.assertTrue(receipt["verified"])
        self.assertEqual(self.destination.read_bytes(), self.body)

    def test_the_fallback_is_still_no_clobber(self):
        """os.link was chosen because it cannot overwrite. The fallback keeps
        that with O_CREAT|O_EXCL rather than an os.replace that would clobber."""
        def cross_device(src, dst):
            raise OSError(18, "Invalid cross-device link")

        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.destination.write_bytes(b"someone else's final")
        with mock.patch.object(download.os, "link", side_effect=cross_device):
            # The pre-lock existence check returns first; force past it to
            # exercise the publish itself.
            self.assertFalse(download._publish_link(self.source, self.destination))
        self.assertEqual(self.destination.read_bytes(), b"someone else's final")

    def test_the_fallback_leaves_no_partial_final_on_failure(self):
        def cross_device(src, dst):
            raise OSError(18, "Invalid cross-device link")

        class Boom(Exception):
            pass

        def exploding_copy(*_args, **_kwargs):
            raise Boom()

        self.destination.parent.mkdir(parents=True, exist_ok=True)
        with mock.patch.object(download.os, "link", side_effect=cross_device), \
                mock.patch.object(download.shutil, "copyfileobj",
                                  side_effect=exploding_copy):
            with self.assertRaises(Boom):
                download._publish_link(self.source, self.destination)
        self.assertFalse(self.destination.exists(),
                         "a partial final was left for the native loader")

    def test_uncooperative_publish_race_preserves_final(self):
        real_link = os.link

        def racing_link(source, destination):
            Path(destination).write_bytes(b"other writer")
            real_link(source, destination)

        with mock.patch.object(download.os, "link", racing_link):
            receipt = self.invoke()
        self.assertEqual(receipt["status"], "exists")
        self.assertFalse(receipt["verified"])
        self.assertEqual(self.destination.read_bytes(), b"other writer")

    def test_transport_error_not_retried_and_lock_reusable(self):
        failure = OSError("full transport error " + "x" * 120)
        transport = mock.Mock(side_effect=failure)
        with self.assertRaises(OSError) as caught:
            self.invoke(fetch=transport)
        self.assertIs(caught.exception, failure)
        self.assertEqual(transport.call_count, 1)
        self.assertFalse(self.destination.exists())
        self.assertEqual(self.invoke()["status"], "downloaded")

    def test_callback_cannot_mutate_approved_metadata(self):
        def progress(done, total):
            self.metadata["size"] = 1
            self.metadata["sha256"] = "0" * 64
        receipt = self.invoke(progress=progress)
        self.assertEqual(receipt["status"], "downloaded")
        self.assertEqual(receipt["bytes_verified"], len(self.body))
        self.assertEqual(receipt["sha256"], hashlib.sha256(self.body).hexdigest())

    def test_unlock_failure_preserves_original_error_and_closes_lock_handle(self):
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        lock = self.destination.with_name(self.destination.name + ".lock")
        for failure in (OSError("original transport failure"),
                        ComfyLikeInterrupt("original Comfy interruption")):
            with self.subTest(error=type(failure).__name__):
                with self.failing_unlock():
                    with self.assertRaises(type(failure)) as caught:
                        with download._destination_lock(lock):
                            raise failure
                self.assertIs(caught.exception, failure)
                details = getattr(failure, "__notes__", []) + [
                    getattr(failure, "visual_asset_lock_cleanup_error", "")
                ]
                self.assertIn("destination lock release failed: injected unlock failure", details)
                with download._destination_lock(lock):
                    pass

    def test_unlock_failure_without_original_error_is_not_hidden(self):
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        lock = self.destination.with_name(self.destination.name + ".lock")
        with self.failing_unlock():
            with self.assertRaisesRegex(PermissionError, "injected unlock failure"):
                with download._destination_lock(lock):
                    pass
        with download._destination_lock(lock):
            pass

    def test_locked_competitor_process_and_released_lock(self):
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        lock = self.destination.with_name(self.destination.name + ".lock")
        code = (
            "import importlib.util, pathlib, sys\n"
            "s=importlib.util.spec_from_file_location('d', sys.argv[1])\n"
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
            "try:\n"
            "    with m._destination_lock(pathlib.Path(sys.argv[2])):\n"
            "        raise AssertionError('competing lock was acquired')\n"
            "except m.DownloadLocked:\n"
            "    print('LOCKED')\n"
        )
        with download._destination_lock(lock):
            with self.assertRaises(download.DownloadLocked):
                self.invoke(fetch=self._forbidden_fetch)
            child = subprocess.run(
                [sys.executable, "-B", "-c", code, str(MODULE_PATH), str(lock)],
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(child.returncode, 0, child.stderr)
            self.assertEqual(child.stdout.strip(), "LOCKED")
        self.assertTrue(lock.exists())
        self.assertEqual(self.invoke()["status"], "downloaded")


if __name__ == "__main__":
    unittest.main()
