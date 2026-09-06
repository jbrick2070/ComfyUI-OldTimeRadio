"""Tiny-file downloader contracts: no ComfyUI imports or real network calls."""
from contextlib import contextmanager
import hashlib
import importlib.util
import io
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


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.destination = self.root / "diffusion_models" / "fixture.safetensors"
        self.payload = b"tiny verified weight fixture"
        self.spec = {"repo_id": "Example/fixture", "filename": "weights/fixture.safetensors"}
        self.metadata = {
            "commit": "a" * 40,
            "sha256": hashlib.sha256(self.payload).hexdigest(),
            "size": len(self.payload),
            "url": "https://example.invalid/fixture?signature=do-not-log",
        }
        self.opened = []

    @contextmanager
    def transport(self, url):
        self.opened.append(url)
        with io.BytesIO(self.payload) as stream:
            yield stream

    def invoke(self, **kwargs):
        arguments = {
            "open_stream": self.transport,
            "disk_free": lambda _path: self.metadata["size"] + download.DISK_MARGIN_BYTES,
        }
        arguments.update(kwargs)
        return download.download_verified(self.spec, self.destination, self.metadata, **arguments)

    def assert_no_partial(self):
        self.assertEqual(list(self.root.rglob("*.part")), [])

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

    def test_success_exact_receipt_progress_and_persistent_lock(self):
        events = []
        with mock.patch.object(download, "CHUNK_BYTES", 5):
            receipt = self.invoke(progress=lambda done, total: events.append((done, total)))
        self.assertEqual(self.destination.read_bytes(), self.payload)
        self.assertEqual(receipt["status"], "downloaded")
        self.assertTrue(receipt["verified"])
        self.assertEqual(receipt["bytes_verified"], len(self.payload))
        self.assertEqual(receipt["sha256"], self.metadata["sha256"])
        self.assertFalse(receipt["resume_supported"])
        self.assertEqual(events[0], (0, len(self.payload)))
        self.assertEqual(events[-1], (len(self.payload), len(self.payload)))
        self.assertEqual([d for d, _ in events], [0, 5, 10, 15, 20, 25, len(self.payload)])
        self.assertNotIn("url", receipt)
        self.assertNotIn("signature", str(receipt))
        self.assertTrue(self.destination.with_name(self.destination.name + ".lock").exists())
        self.assert_no_partial()

    def test_wrong_hash_does_not_publish(self):
        self.metadata["sha256"] = "0" * 64
        with self.assertRaisesRegex(download.VisualAssetDownloadError, "SHA-256 mismatch"):
            self.invoke()
        self.assertFalse(self.destination.exists())
        self.assert_no_partial()

    def test_short_read_does_not_publish(self):
        self.metadata["size"] += 1
        with self.assertRaisesRegex(download.VisualAssetDownloadError, "size mismatch"):
            self.invoke()
        self.assertFalse(self.destination.exists())
        self.assert_no_partial()

    def test_overlong_rejected_before_writing_extra_chunk(self):
        self.metadata["size"] -= 1
        events = []
        with self.assertRaisesRegex(download.VisualAssetDownloadError, "exceeds declared size"):
            self.invoke(progress=lambda done, total: events.append((done, total)))
        self.assertEqual(events, [(0, self.metadata["size"])])
        self.assertFalse(self.destination.exists())
        self.assert_no_partial()

    def test_cancel_before_start_does_not_open_or_create_parent(self):
        with self.assertRaises(download.DownloadCancelled):
            self.invoke(cancel=lambda: True)
        self.assertEqual(self.opened, [])
        self.assertFalse(self.destination.parent.exists())

    def test_cancel_during_stream_cleans_only_own_temp(self):
        self.destination.parent.mkdir()
        other = self.destination.parent / "other-transfer.part"
        other.write_bytes(b"not ours")
        events = []
        with mock.patch.object(download, "CHUNK_BYTES", 5):
            with self.assertRaises(download.DownloadCancelled):
                self.invoke(
                    progress=lambda done, total: events.append(done),
                    cancel=lambda: bool(events and events[-1] >= 5),
                )
        self.assertFalse(self.destination.exists())
        self.assertEqual(other.read_bytes(), b"not ours")
        self.assertEqual(list(self.root.rglob("*.part")), [other])

    def test_raised_interrupt_propagates_unchanged(self):
        interruption = RuntimeError("injected Comfy interruption")
        def cancel():
            raise interruption
        with self.assertRaises(RuntimeError) as caught:
            self.invoke(cancel=cancel)
        self.assertIs(caught.exception, interruption)
        self.assertEqual(self.opened, [])

    def test_midstream_baseexception_closes_transport_cleans_temp_and_releases_lock(self):
        self.destination.parent.mkdir()
        other = self.destination.parent / "other-transfer.part"
        other.write_bytes(b"not ours")
        interruption = ComfyLikeInterrupt("injected mid-stream Comfy interruption")
        events = []
        stream = io.BytesIO(self.payload)

        @contextmanager
        def transport(url):
            self.opened.append(url)
            with stream:
                yield stream

        def cancel():
            if events and events[-1] >= 5:
                raise interruption

        with mock.patch.object(download, "CHUNK_BYTES", 5):
            with self.assertRaises(ComfyLikeInterrupt) as caught:
                self.invoke(open_stream=transport, cancel=cancel,
                            progress=lambda done, total: events.append(done))
        self.assertIs(caught.exception, interruption)
        self.assertEqual(events, [0, 5])
        self.assertTrue(stream.closed)
        self.assertFalse(self.destination.exists())
        self.assertEqual(other.read_bytes(), b"not ours")
        self.assertEqual(list(self.root.rglob("*.part")), [other])
        # No stale-lock repair: normal transfer can acquire the persistent lock.
        self.assertEqual(self.invoke()["status"], "downloaded")
        self.assertEqual(other.read_bytes(), b"not ours")

    def test_unlock_failure_preserves_original_error_and_closes_lock_handle(self):
        self.destination.parent.mkdir()
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
        self.destination.parent.mkdir()
        lock = self.destination.with_name(self.destination.name + ".lock")
        with self.failing_unlock():
            with self.assertRaisesRegex(PermissionError, "injected unlock failure"):
                with download._destination_lock(lock):
                    pass
        with download._destination_lock(lock):
            pass

    def test_preexisting_final_is_explicitly_unverified_and_unchanged(self):
        self.destination.parent.mkdir()
        self.destination.write_bytes(b"existing selection")
        receipt = self.invoke(open_stream=mock.Mock(side_effect=AssertionError("network called")))
        self.assertEqual(receipt["status"], "exists")
        self.assertFalse(receipt["verified"])
        self.assertEqual(receipt["bytes_verified"], 0)
        self.assertEqual(self.destination.read_bytes(), b"existing selection")
        self.assert_no_partial()

    def test_broken_symlink_lexists_is_not_overwritten(self):
        self.destination.parent.mkdir()
        # Simulate a broken link's lexists=True / exists=False semantics without
        # creating any symlink or requiring elevated Windows privileges.
        with mock.patch.object(download.os.path, "lexists", return_value=True):
            receipt = self.invoke(open_stream=mock.Mock(side_effect=AssertionError("network called")))
        self.assertEqual(receipt["status"], "exists")
        self.assertFalse(self.destination.exists())

    def test_filesystem_without_hardlinks_fails_closed(self):
        # There must never be a copy/replace fallback on filesystems that cannot
        # provide atomic, no-clobber hard-link publication.
        with mock.patch.object(download.os, "link", side_effect=OSError("hard links unsupported")):
            with self.assertRaisesRegex(OSError, "hard links unsupported"):
                self.invoke()
        self.assertFalse(self.destination.exists())
        self.assert_no_partial()

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
        self.assert_no_partial()

    def test_invalid_metadata_never_opens_transport(self):
        invalid = [
            ("size", True), ("size", 0), ("size", -1), ("size", 2.0),
            ("commit", "a" * 39), ("commit", "g" * 40),
            ("sha256", "a" * 63), ("sha256", "g" * 64),
            ("url", "http://example.invalid/file"), ("url", "https://"),
            ("url", "https://user:secret@example.invalid/file"),
            ("url", "https://example.invalid/file#secret"),
        ]
        original = self.metadata.copy()
        for key, value in invalid:
            with self.subTest(key=key, value_type=type(value).__name__):
                self.metadata = dict(original, **{key: value})
                with self.assertRaises(ValueError) as caught:
                    self.invoke()
                self.assertNotIn("secret", str(caught.exception))
                self.assertEqual(self.opened, [])
        self.assertFalse(self.destination.exists())
        self.assert_no_partial()

    def test_insufficient_disk_does_not_download(self):
        with self.assertRaisesRegex(download.VisualAssetDownloadError, "insufficient"):
            self.invoke(disk_free=lambda path: self.metadata["size"] + download.DISK_MARGIN_BYTES - 1)
        self.assertEqual(self.opened, [])
        self.assertFalse(self.destination.exists())
        self.assert_no_partial()

    def test_transport_error_not_retried_and_lock_reusable(self):
        failure = OSError("full transport error " + "x" * 120)
        transport = mock.Mock(side_effect=failure)
        with self.assertRaises(OSError) as caught:
            self.invoke(open_stream=transport)
        self.assertIs(caught.exception, failure)
        self.assertEqual(transport.call_count, 1)
        self.assertFalse(self.destination.exists())
        self.assert_no_partial()
        self.assertEqual(self.invoke()["status"], "downloaded")

    def test_fsync_failure_does_not_publish(self):
        with mock.patch.object(download.os, "fsync", side_effect=OSError("fsync failed")):
            with self.assertRaisesRegex(OSError, "fsync failed"):
                self.invoke()
        self.assertFalse(self.destination.exists())
        self.assert_no_partial()

    def test_cleanup_failure_does_not_mask_transport_error(self):
        failure = OSError("original full transport failure")
        with mock.patch.object(download.Path, "unlink", side_effect=PermissionError("temp busy")):
            with self.assertRaises(OSError) as caught:
                self.invoke(open_stream=mock.Mock(side_effect=failure))
        self.assertIs(caught.exception, failure)
        details = getattr(failure, "__notes__", []) + [
            getattr(failure, "visual_asset_cleanup_error", "")
        ]
        self.assertIn("owned temporary cleanup failed: temp busy", details)
        self.assertFalse(self.destination.exists())
        # The one owned temp remains because its injected cleanup failed; no
        # unrelated file was touched. TemporaryDirectory cleans this fixture.
        self.assertEqual(len(list(self.root.rglob("*.part"))), 1)

    def test_callback_cannot_mutate_approved_metadata(self):
        def progress(done, total):
            self.metadata["size"] = 1
            self.metadata["sha256"] = "0" * 64
        receipt = self.invoke(progress=progress)
        self.assertEqual(receipt["status"], "downloaded")
        self.assertEqual(receipt["bytes_verified"], len(self.payload))
        self.assertEqual(receipt["sha256"], hashlib.sha256(self.payload).hexdigest())

    def test_locked_competitor_process_and_released_lock(self):
        self.destination.parent.mkdir()
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
                self.invoke()
            self.assertEqual(self.opened, [])
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
