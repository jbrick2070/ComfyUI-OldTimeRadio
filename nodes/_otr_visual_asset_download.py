"""Verified, no-clobber visual-weight transfer; standard library only.

The caller owns metadata discovery, source allowlisting, native path resolution,
and the network transport. This module never imports ComfyUI/model code and has
no default network callable. Transfers do not retry or resume: a later explicit
attempt starts a fresh temporary file. No URL is logged or returned in receipts.
Transport/cancellation exceptions propagate unchanged to the caller.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from urllib.parse import urlsplit


DISK_MARGIN_BYTES = 512 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024


class VisualAssetDownloadError(RuntimeError):
    """A transfer was refused or failed verification; no final was published."""


class DownloadLocked(VisualAssetDownloadError):
    """Another process holds this destination's advisory lock."""


class DownloadCancelled(VisualAssetDownloadError):
    """The injected cancellation callback requested cancellation."""


def _validate(spec, metadata):
    if not isinstance(spec, dict) or not isinstance(metadata, dict):
        raise ValueError("spec and metadata must be dictionaries")
    for key in ("repo_id", "filename"):
        value = spec.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError("spec requires a nonempty " + key)
    size = metadata.get("size")
    if type(size) is not int or size <= 0:
        raise ValueError("metadata size must be a positive exact integer")
    for key, length in (("commit", 40), ("sha256", 64)):
        value = metadata.get(key)
        if not isinstance(value, str) or not re.fullmatch(
            r"[0-9a-fA-F]{%d}" % length, value
        ):
            raise ValueError("metadata " + key + " must be exact hexadecimal")
    url = metadata.get("url")
    try:
        parsed = urlsplit(url) if isinstance(url, str) else None
        valid_url = (
            parsed is not None
            and parsed.scheme == "https"
            and bool(parsed.hostname)
            and parsed.username is None
            and parsed.password is None
            and not parsed.fragment
        )
    except ValueError:
        valid_url = False
    if not valid_url:
        # Never interpolate an invalid/signed URL into an exception.
        raise ValueError("metadata url must be an HTTPS URL without userinfo or fragment")


def _check_cancel(cancel):
    # Comfy's interrupt check raises; a simple boolean test seam also works.
    if cancel is not None and cancel():
        raise DownloadCancelled("visual asset download cancelled; resume unsupported")


@contextmanager
def _destination_lock(path):
    """Nonblocking OS lock, released by the OS even after process death.

    The .lock file deliberately persists. Its existence is not a lock, and no
    stale-file deletion or other process's temporary-file cleanup is attempted.
    """
    with open(path, "a+b") as handle:
        # msvcrt locks a byte range. A persistent first byte works on both OSes.
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise DownloadLocked("visual asset destination is locked by another transfer") from exc
        try:
            yield
        finally:
            original_error = sys.exc_info()[1]
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError as cleanup_error:
                if original_error is None:
                    raise
                # Closing the handle still releases its OS lock. Preserve a
                # transport failure or Comfy's BaseException interruption if
                # explicit unlock also fails, and retain both error details.
                detail = "destination lock release failed: " + str(cleanup_error)
                if hasattr(original_error, "add_note"):
                    original_error.add_note(detail)
                else:  # Python 3.10: retain detail without replacing error.
                    original_error.visual_asset_lock_cleanup_error = detail


def _existing_ancestor(path):
    while not path.exists():
        previous = path
        path = path.parent
        if path == previous:
            raise VisualAssetDownloadError("destination has no existing filesystem ancestor")
    return path


def download_verified(
    spec: dict,
    destination: Path,
    metadata: dict,
    *,
    open_stream,
    cancel=None,
    progress=None,
    disk_free=None,
) -> dict:
    """Transfer exactly one caller-authorized artifact without overwriting.

    ``open_stream(url)`` must return a context manager yielding a binary object
    with ``read(n)``. ``cancel()`` may raise or return truthy. ``progress(done,
    total)`` receives exact byte counts. ``disk_free(existing_path)`` optionally
    replaces ``shutil.disk_usage(existing_path).free`` for deterministic tests.

    ``status='exists'`` means *unverified*: the caller must re-resolve the native
    loader token. Even a broken destination symlink is preserved. A successful
    transfer publishes with an atomic hard link, not an overwriting rename;
    unsupported filesystems fail closed. Only this call's unique temp is cleaned.
    """
    _validate(spec, metadata)
    # Callbacks must not be able to change the approved size/hash mid-transfer.
    spec = spec.copy()
    metadata = metadata.copy()
    if not callable(open_stream):
        raise TypeError("open_stream must be an injected callable")
    destination = Path(destination)
    _check_cancel(cancel)
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "repo_id": spec["repo_id"],
        "filename": spec["filename"],
        "commit": metadata["commit"].lower(),
        "sha256": metadata["sha256"].lower(),
        "size": metadata["size"],
        "destination": str(destination),
        "status": "exists",
        "verified": False,
        "bytes_verified": 0,
        "resume_supported": False,
    }
    lock_path = destination.with_name(destination.name + ".lock")
    with _destination_lock(lock_path):
        _check_cancel(cancel)
        if os.path.lexists(destination):
            return receipt
        filesystem_path = _existing_ancestor(destination.parent)
        free = (disk_free(filesystem_path) if disk_free is not None
                else shutil.disk_usage(filesystem_path).free)
        if type(free) is not int or free < metadata["size"] + DISK_MARGIN_BYTES:
            raise VisualAssetDownloadError(
                "insufficient destination disk space: need %d bytes plus %d bytes margin"
                % (metadata["size"], DISK_MARGIN_BYTES)
            )
        temporary = None
        try:
            fd, name = tempfile.mkstemp(
                prefix="." + destination.name + ".", suffix=".part",
                dir=destination.parent,
            )
            temporary = Path(name)
            digest = hashlib.sha256()
            received = 0
            with os.fdopen(fd, "wb") as output:
                _check_cancel(cancel)
                if progress is not None:
                    progress(0, metadata["size"])
                with open_stream(metadata["url"]) as stream:
                    while True:
                        _check_cancel(cancel)
                        chunk = stream.read(CHUNK_BYTES)
                        _check_cancel(cancel)
                        if not isinstance(chunk, bytes):
                            raise VisualAssetDownloadError("transport must return binary bytes")
                        if not chunk:
                            break
                        if received + len(chunk) > metadata["size"]:
                            raise VisualAssetDownloadError(
                                "download exceeds declared size %d bytes" % metadata["size"]
                            )
                        output.write(chunk)
                        digest.update(chunk)
                        received += len(chunk)
                        if progress is not None:
                            progress(received, metadata["size"])
                if received != metadata["size"]:
                    raise VisualAssetDownloadError(
                        "download size mismatch: received %d, expected %d bytes"
                        % (received, metadata["size"])
                    )
                if digest.hexdigest() != metadata["sha256"].lower():
                    raise VisualAssetDownloadError("download SHA-256 mismatch")
                output.flush()
                os.fsync(output.fileno())
            _check_cancel(cancel)
            # Unlike replace/rename on POSIX, link is atomically no-clobber.
            # If an uncooperative writer races us, preserve its final too.
            try:
                os.link(temporary, destination)
            except FileExistsError:
                return receipt
            receipt.update(status="downloaded", verified=True, bytes_verified=received)
            return receipt
        finally:
            if temporary is not None:
                original_error = sys.exc_info()[1]
                try:
                    temporary.unlink(missing_ok=True)
                except OSError as cleanup_error:
                    if original_error is None:
                        raise
                    # Preserve the full transport/verification/interruption
                    # error even when antivirus/permissions delay temp cleanup.
                    detail = "owned temporary cleanup failed: " + str(cleanup_error)
                    if hasattr(original_error, "add_note"):
                        original_error.add_note(detail)
                    else:  # Python 3.10: retain detail without replacing error.
                        original_error.visual_asset_cleanup_error = detail
