"""The terminal mux's finishing boundaries (Muse finishing review, 2026-10-14).

Four guarantees, each pinned against the real code with CPU fixtures:

1. A final file -- the archival PCM mux or the OBS viewing copy -- is replaced
   only by bytes that passed validation. Candidates are encoded beside it (OBS:
   in a staging subfolder), validated, then renamed over it in one step; a
   rejected candidate is kept as evidence and the old final is never touched.
2. The terminal ledger stamp writes ONLY the ledger that answers for the
   incoming video's stem -- never a foreign singleton.
3. A Cancel pending after the mux is honoured at each finishing boundary, and
   Comfy's own exception propagates exactly as raised.
4. A file missing its audio (or video) stream is refused by name; the container
   duration no longer stands in for a stream that is not there.

Real ffmpeg / ffprobe make the media where a probe must read a real file; the
stubs replace only the step under test.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from nodes import _otr_ledger as OTRL
from nodes import otr_master_audio_mux as MUX
from nodes._otr_shared import ffprobe as ffp


# --------------------------------------------------------------------------- #
# media fixtures
# --------------------------------------------------------------------------- #
def _ffmpeg():
    fb = MUX._ffmpeg_bin("ffmpeg")
    if not fb:
        pytest.skip("ffmpeg not available")
    return fb


def _make(path, *args):
    subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", *args, str(path)],
                   check=True)
    return Path(path)


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    d = tmp_path_factory.mktemp("mux_media")
    color = ["-f", "lavfi", "-i", "color=c=blue:s=64x48:r=25"]
    sine = ["-f", "lavfi", "-i", "sine=f=440:sample_rate=48000"]
    return {
        "av": _make(d / "av.mp4", *color, *sine, "-t", "2", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest"),
        "short_av": _make(d / "short_av.mp4", *color, *sine, "-t", "1",
                          "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest"),
        "video_only": _make(d / "video_only.mp4", *color, "-t", "2",
                            "-pix_fmt", "yuv420p"),
        "audio_only": _make(d / "audio_only.mp4", *sine, "-t", "2", "-c:a", "aac"),
        "silent": _make(d / "silent.mp4", *color, "-t", "2", "-pix_fmt", "yuv420p"),
        "master": _make(d / "master.wav", *sine, "-t", "2", "-c:a", "pcm_s16le"),
    }


@pytest.fixture
def obs(tmp_path, monkeypatch):
    folder = tmp_path / "output" / "otr" / "obs"
    folder.mkdir(parents=True)
    monkeypatch.setattr(MUX, "_obs_dir", lambda: folder)
    return folder


SENTINEL = b"the previously published, known-good bytes"


def _seed(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(SENTINEL)
    return Path(path)


def _writes(src):
    """A `_run` stand-in that "encodes" by copying ``src`` to the output arg."""
    def run(argv, **kw):
        shutil.copyfile(src, argv[-1])
        return subprocess.CompletedProcess(argv, 0, "", "")
    return run


def _staged(obs_dir):
    staging = Path(obs_dir) / MUX.OBS_STAGING_DIRNAME
    return sorted(staging.iterdir()) if staging.is_dir() else []


# --------------------------------------------------------------------------- #
# 4. a missing stream is not a stream without a duration
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("backend", ["binary", "pyav"])
def test_a_missing_stream_is_named_not_filled_from_the_container(
        media, monkeypatch, backend):
    if backend == "pyav":
        monkeypatch.setattr(ffp, "resolve_ffprobe", lambda *a, **k: None)
    elif not ffp.resolve_ffprobe():
        pytest.skip("no ffprobe binary on this box")
    assert MUX._probe_float(str(media["video_only"]), "a:0") == MUX.STREAM_MISSING
    assert MUX._probe_float(str(media["audio_only"]), "v:0") == MUX.STREAM_MISSING
    assert MUX._probe_float(str(media["video_only"]), "v:0") == pytest.approx(2.0, abs=0.05)
    assert MUX._probe_float(str(media["av"]), "a:0") > 1.5


def test_a_document_without_a_streams_key_is_unmeasured_not_missing(
        media, monkeypatch):
    """Only an explicit EMPTY list means "read, and not there"."""
    real = ffp.probe_json

    def no_key(path, entries=None, **kw):
        if entries == "stream=duration":
            return {}
        return real(path, entries, **kw)
    monkeypatch.setattr(ffp, "probe_json", no_key)
    assert MUX._probe_float(str(media["av"]), "a:0") == pytest.approx(2.0, abs=0.1)


@pytest.mark.parametrize("payload,label", [("video_only", "audio"),
                                           ("audio_only", "video")])
def test_obs_refuses_a_copy_missing_a_stream_and_keeps_the_old_one(
        media, obs, monkeypatch, payload, label):
    final = str(media["av"])
    dst = _seed(obs / MUX._obs_basename(final))
    monkeypatch.setattr(MUX, "_run", _writes(media[payload]))
    with pytest.raises(OSError, match="%s stream is MISSING" % label):
        MUX.OTRMasterAudioMux()._publish_to_obs(final)
    assert dst.read_bytes() == SENTINEL, "the published copy was replaced"
    kept = _staged(obs)
    assert len(kept) == 1 and kept[0].stat().st_size > 0, "the evidence was lost"
    assert "_final" not in kept[0].name, "a rejected candidate reads as published"


def test_a_first_publish_that_fails_creates_no_published_file(
        media, obs, monkeypatch):
    final = str(media["av"])
    monkeypatch.setattr(MUX, "_run", _writes(media["video_only"]))
    with pytest.raises(OSError):
        MUX.OTRMasterAudioMux()._publish_to_obs(final)
    assert not (obs / MUX._obs_basename(final)).exists()
    assert [p for p in obs.iterdir() if p.is_file()] == [], \
        "nothing may appear in the top-level listing he watches"


# --------------------------------------------------------------------------- #
# 1. candidate, validate, then replace -- OBS
# --------------------------------------------------------------------------- #
def test_a_good_publish_replaces_the_old_copy_only_after_validation(
        media, obs, monkeypatch):
    final = str(media["av"])
    dst = _seed(obs / MUX._obs_basename(final))
    seen = []
    real_run = MUX._run

    def run(argv):
        seen.append(list(argv))
        assert Path(dst).read_bytes() == SENTINEL, "replaced before validation"
        return real_run(argv)
    monkeypatch.setattr(MUX, "_run", run)

    out = MUX.OTRMasterAudioMux()._publish_to_obs(final)
    assert out == str(dst)
    assert dst.read_bytes() != SENTINEL
    assert MUX._probe_float(out, "v:0") > 1.5 and MUX._probe_float(out, "a:0") > 1.5
    argv = seen[0]
    # the encode itself is unchanged: video copy, AAC 320k at 48 kHz
    assert argv[argv.index("-c:v") + 1] == "copy"
    assert argv[argv.index("-c:a") + 1] == "aac"
    assert argv[argv.index("-b:a") + 1] == "320k"
    assert argv[argv.index("-ar") + 1] == "48000"
    assert "-shortest" not in argv
    # ...into a candidate in the staging subfolder, never the published name
    assert Path(argv[-1]).parent.name == MUX.OBS_STAGING_DIRNAME
    assert "_final" not in Path(argv[-1]).name
    assert not (obs / MUX.OBS_STAGING_DIRNAME).exists(), "empty staging left behind"


def test_a_truncated_copy_is_refused_and_the_old_one_stands(media, obs, monkeypatch):
    final = str(media["av"])
    dst = _seed(obs / MUX._obs_basename(final))
    monkeypatch.setattr(MUX, "_run", _writes(media["short_av"]))
    with pytest.raises(OSError, match="TRUNCATED"):
        MUX.OTRMasterAudioMux()._publish_to_obs(final)
    assert dst.read_bytes() == SENTINEL


@pytest.mark.parametrize("partial", [True, False], ids=["partial_bytes", "nothing"])
def test_a_failed_encode_keeps_the_old_copy_and_only_drops_an_empty_reservation(
        media, obs, monkeypatch, partial):
    final = str(media["av"])
    dst = _seed(obs / MUX._obs_basename(final))

    def run(argv, **kw):
        if partial:
            Path(argv[-1]).write_bytes(b"half an mp4")
        return subprocess.CompletedProcess(argv, 1, "", "encoder exploded")
    monkeypatch.setattr(MUX, "_run", run)
    with pytest.raises(OSError, match="encoder exploded"):
        MUX.OTRMasterAudioMux()._publish_to_obs(final)
    assert dst.read_bytes() == SENTINEL
    kept = _staged(obs)
    assert len(kept) == (1 if partial else 0)


windows_only = pytest.mark.skipif(
    os.name != "nt", reason="needs a real Windows file handle")


def _held(winerror=32):
    """A PermissionError carrying a Windows hold code, on ANY platform (a
    subclass attribute, so the retry logic is exercised off Windows too)."""
    cls = type("_Hold", (PermissionError,), {"winerror": winerror})
    return cls(13, "being used by another process")


def test_a_replacement_that_fails_never_deletes_the_old_copy(media, obs, monkeypatch):
    """Windows refuses a rename over a file a player holds open. The old copy
    stays; it is never deleted to make room."""
    final = str(media["av"])
    dst = _seed(obs / MUX._obs_basename(final))
    monkeypatch.setattr(MUX, "_run", _writes(media["av"]))
    monkeypatch.setattr(MUX, "_REPLACE_WAIT_S", 0.0)
    tries = []

    def busy(src, target):
        tries.append(1)
        raise _held()
    monkeypatch.setattr(MUX.os, "replace", busy)
    with pytest.raises(OSError, match="could not replace"):
        MUX.OTRMasterAudioMux()._publish_to_obs(final)
    assert dst.read_bytes() == SENTINEL
    assert len(_staged(obs)) == 1, "the accepted candidate must be kept"
    assert len(tries) == MUX._REPLACE_ATTEMPTS, "a held file is retried, boundedly"


def test_a_briefly_held_candidate_is_retried_not_failed(media, obs, monkeypatch):
    """An antivirus or indexer holding the fresh candidate for a moment must
    not fail the last step of a long render (sonnet QA on 733c2386)."""
    final = str(media["av"])
    dst = _seed(obs / MUX._obs_basename(final))
    monkeypatch.setattr(MUX, "_run", _writes(media["av"]))
    monkeypatch.setattr(MUX, "_REPLACE_WAIT_S", 0.0)
    real_replace = os.replace
    held = {"left": 2}

    def scanned_twice(src, target):
        if held["left"]:
            held["left"] -= 1
            raise _held()
        return real_replace(src, target)
    monkeypatch.setattr(MUX.os, "replace", scanned_twice)
    out = MUX.OTRMasterAudioMux()._publish_to_obs(final)
    assert out == str(dst) and dst.read_bytes() != SENTINEL


def test_a_permission_error_that_is_not_a_hold_is_tried_once(media, obs, monkeypatch):
    """A read-only or wrong-type destination is not transient: no retries
    (sonnet QA on b4a1f00f)."""
    final = str(media["av"])
    _seed(obs / MUX._obs_basename(final))
    monkeypatch.setattr(MUX, "_run", _writes(media["av"]))
    tries = []

    def read_only(src, target):
        tries.append(1)
        raise PermissionError(13, "read-only destination")   # no Windows hold code
    monkeypatch.setattr(MUX.os, "replace", read_only)
    with pytest.raises(OSError, match="could not replace"):
        MUX.OTRMasterAudioMux()._publish_to_obs(final)
    assert len(tries) == 1


@pytest.mark.parametrize("dest", ["writable_file", "absent"])
def test_access_denied_on_a_writable_or_absent_destination_is_a_hold(
        media, obs, monkeypatch, dest):
    """ACCESS_DENIED (5) is how a real hold often surfaces; it is retried
    when the destination could take the file (pins 5 in the set)."""
    final = str(media["av"])
    dst = obs / MUX._obs_basename(final)
    if dest == "writable_file":
        _seed(dst)
    monkeypatch.setattr(MUX, "_run", _writes(media["av"]))
    monkeypatch.setattr(MUX, "_REPLACE_WAIT_S", 0.0)
    tries = []

    def denied(src, target):
        tries.append(1)
        raise _held(5)
    monkeypatch.setattr(MUX.os, "replace", denied)
    with pytest.raises(OSError, match="could not replace"):
        MUX.OTRMasterAudioMux()._publish_to_obs(final)
    assert len(tries) == MUX._REPLACE_ATTEMPTS


@pytest.mark.parametrize("dest", ["read_only_file", "directory"])
def test_access_denied_on_a_read_only_or_directory_destination_is_tried_once(
        media, obs, monkeypatch, dest):
    """The same code 5 from a destination that can never take the file is not
    a hold: one try, no ~1.25 s of retries (sonnet QA on 2c901787)."""
    import stat

    final = str(media["av"])
    dst = obs / MUX._obs_basename(final)
    if dest == "directory":
        dst.mkdir()
    else:
        _seed(dst)
        os.chmod(dst, stat.S_IREAD)
    monkeypatch.setattr(MUX, "_run", _writes(media["av"]))
    tries = []

    def denied(src, target):
        tries.append(1)
        raise _held(5)
    monkeypatch.setattr(MUX.os, "replace", denied)
    try:
        with pytest.raises(OSError, match="could not replace"):
            MUX.OTRMasterAudioMux()._publish_to_obs(final)
    finally:
        if dest == "read_only_file":
            os.chmod(dst, stat.S_IREAD | stat.S_IWRITE)
    assert len(tries) == 1


def test_a_cancel_during_a_held_retry_is_honoured_before_the_next_try(
        media, obs, monkeypatch, caplog):
    """The Cancel poll runs before EVERY try, so a hold never delays a Cancel;
    the cancel is logged, the candidate removed, the empty staging folder
    gone and the old copy untouched (sonnet QA on b4a1f00f)."""
    final = str(media["av"])
    dst = _seed(obs / MUX._obs_basename(final))
    monkeypatch.setattr(MUX, "_run", _writes(media["av"]))
    monkeypatch.setattr(MUX, "_REPLACE_WAIT_S", 0.0)
    state = _install_comfy(monkeypatch)
    tries = []

    def held_then_cancelled(src, target):
        tries.append(1)
        if len(tries) == 2:
            state["cancel"] = True          # noticed by the poll before try 3
        raise _held()
    monkeypatch.setattr(MUX.os, "replace", held_then_cancelled)
    with caplog.at_level("WARNING", logger="OTR"):
        with pytest.raises(InterruptProcessingException):
            MUX.OTRMasterAudioMux()._publish_to_obs(final)
    assert len(tries) == 2, "a third replace ran after the Cancel was pending"
    assert "cancelled before replacing" in caplog.text
    assert dst.read_bytes() == SENTINEL
    assert not (obs / MUX.OBS_STAGING_DIRNAME).exists()


@windows_only
def test_a_cancel_with_the_candidate_still_held_is_logged_not_silent(
        media, obs, monkeypatch, caplog):
    """The scanner the retry exists for can hold the candidate when Cancel
    arrives: removal fails, and that must be said, not swallowed."""
    final = str(media["av"])
    dst = _seed(obs / MUX._obs_basename(final))
    state = _install_comfy(monkeypatch)
    handles = []
    copy = _writes(media["av"])

    def encode_hold_cancel(argv, **kw):
        result = copy(argv)
        handles.append(open(argv[-1], "rb"))     # a scanner's handle
        state["cancel"] = True
        return result
    monkeypatch.setattr(MUX, "_run", encode_hold_cancel)
    try:
        with caplog.at_level("WARNING", logger="OTR"):
            with pytest.raises(InterruptProcessingException):
                MUX.OTRMasterAudioMux()._publish_to_obs(final)
    finally:
        for h in handles:
            h.close()
    assert "cancelled before replacing" in caplog.text
    assert "could not remove the cancelled candidate" in caplog.text
    assert dst.read_bytes() == SENTINEL


def test_a_failed_removal_after_cancel_is_logged_on_any_platform(
        media, obs, monkeypatch, caplog):
    final = str(media["av"])
    dst = _seed(obs / MUX._obs_basename(final))
    state = _install_comfy(monkeypatch)
    copy = _writes(media["av"])

    def encode_then_cancel(argv, **kw):
        result = copy(argv)
        state["cancel"] = True
        return result
    monkeypatch.setattr(MUX, "_run", encode_then_cancel)

    def held_remove(path, *a, **k):
        raise _held(32)
    monkeypatch.setattr(MUX.os, "remove", held_remove)
    with caplog.at_level("WARNING", logger="OTR"):
        with pytest.raises(InterruptProcessingException):
            MUX.OTRMasterAudioMux()._publish_to_obs(final)
    monkeypatch.undo()
    assert "could not remove the cancelled candidate" in caplog.text
    assert dst.read_bytes() == SENTINEL


def test_the_alias_refusal_does_not_wait_for_the_ffmpeg_gate(media, tmp_path, monkeypatch):
    """With no usable ffmpeg the alias is still named, not "ffmpeg not found"
    (sonnet QA on 2c901787: the order was unpinned)."""
    silent = tmp_path / "silent.mp4"
    shutil.copyfile(media["silent"], silent)
    monkeypatch.setattr(MUX, "_ffmpeg_bin", lambda *a, **k: "")
    with pytest.raises(ValueError, match="silent video input"):
        MUX.mux_master_audio(str(silent), str(media["master"]), str(silent))


def test_an_encode_that_raises_leaves_no_reservation_or_staging(
        media, obs, tmp_path, monkeypatch):
    final = str(media["av"])
    dst = _seed(obs / MUX._obs_basename(final))

    def timed_out(argv, **kw):
        raise TimeoutError("ffmpeg hung")
    monkeypatch.setattr(MUX, "_run", timed_out)
    with pytest.raises(TimeoutError):
        MUX.OTRMasterAudioMux()._publish_to_obs(final)
    assert dst.read_bytes() == SENTINEL
    assert not (obs / MUX.OBS_STAGING_DIRNAME).exists()
    out = _seed(tmp_path / "ep" / "ep_final.mkv")
    with pytest.raises(TimeoutError):
        MUX.mux_master_audio(str(media["silent"]), str(media["master"]), str(out))
    assert [p.name for p in out.parent.iterdir()] == [out.name]


def test_a_staging_folder_removed_by_a_peer_mid_reservation_is_a_retry(
        tmp_path, monkeypatch):
    """A peer's rmdir between our mkdir and our create is a retry, not a
    failed publish (sonnet QA on 2c901787)."""
    folder = tmp_path / "obs" / MUX.OBS_STAGING_DIRNAME
    real_open = os.open
    raced = {"done": False}

    def racing_open(path, flags, *a):
        if not raced["done"]:
            raced["done"] = True
            os.rmdir(folder)                     # the peer empties and removes it
            raise FileNotFoundError(2, "the folder vanished")
        return real_open(path, flags, *a)
    monkeypatch.setattr(MUX.os, "open", racing_open)
    got = MUX._reserve_candidate(str(tmp_path / "obs" / "ep_final.mp4"), str(folder))
    assert raced["done"] and os.path.isfile(got)


def test_no_name_fits_creates_no_folder(tmp_path, monkeypatch):
    folder = tmp_path / "obs" / MUX.OBS_STAGING_DIRNAME
    monkeypatch.setattr(MUX, "path_fits", lambda path, budget=0: False)
    with pytest.raises(OSError, match="path budget"):
        MUX._reserve_candidate(str(tmp_path / "obs" / "ep_final.mp4"), str(folder))
    assert not folder.exists()


def test_a_non_sharing_replace_error_is_not_retried(media, obs, monkeypatch):
    final = str(media["av"])
    _seed(obs / MUX._obs_basename(final))
    monkeypatch.setattr(MUX, "_run", _writes(media["av"]))
    tries = []

    def gone(src, target):
        tries.append(1)
        raise FileNotFoundError(2, "the volume went away")
    monkeypatch.setattr(MUX.os, "replace", gone)
    with pytest.raises(OSError, match="could not replace"):
        MUX.OTRMasterAudioMux()._publish_to_obs(final)
    assert len(tries) == 1


def test_an_output_that_is_an_input_is_refused_by_name(media, tmp_path):
    """ffmpeg used to refuse "Output same as Input" itself; a candidate renamed
    over the destination would now overwrite the input (sonnet QA on 733c2386)."""
    silent = tmp_path / "silent.mp4"
    master = tmp_path / "master.wav"
    shutil.copyfile(media["silent"], silent)
    shutil.copyfile(media["master"], master)
    before = (silent.read_bytes(), master.read_bytes())
    with pytest.raises(ValueError, match="silent video input"):
        MUX.mux_master_audio(str(silent), str(master), str(silent))
    with pytest.raises(ValueError, match="master audio input"):
        MUX.mux_master_audio(str(silent), str(master), str(master))
    assert (silent.read_bytes(), master.read_bytes()) == before


def test_a_published_name_that_is_the_archival_final_is_refused(obs, media, monkeypatch):
    """An operator output_path that lands the archival final on the exact
    published name: the AAC copy would replace the PCM final it reads."""
    final = obs / "ep_final.mp4"
    shutil.copyfile(media["av"], final)
    before = final.read_bytes()
    monkeypatch.setattr(MUX, "_obs_basename", lambda path: os.path.basename(path))
    with pytest.raises(OSError, match="archival final itself"):
        MUX.OTRMasterAudioMux()._publish_to_obs(str(final))
    assert final.read_bytes() == before


# --------------------------------------------------------------------------- #
# 1. candidate, validate, then replace -- the archival mux
# --------------------------------------------------------------------------- #
def _mux_capable():
    from nodes._otr_shared.ffmpeg import master_mux_support_gap
    gap = master_mux_support_gap(_ffmpeg())
    if gap:
        pytest.skip(gap)


@pytest.mark.parametrize("ext", [".mp4", ".mkv"])
def test_the_archival_mux_replaces_a_final_only_with_a_proven_one(
        media, tmp_path, ext):
    if ext == ".mp4":
        _mux_capable()
    out = _seed(tmp_path / "ep" / ("ep_final" + ext))
    result, report = MUX.mux_master_audio(str(media["silent"]), str(media["master"]),
                                          str(out))
    assert result == str(out)
    assert out.read_bytes() != SENTINEL
    assert any(line.startswith("audio_byte_identical OK") for line in report)
    assert MUX.audio_pcm_sha(str(out), _ffmpeg()) == \
        MUX.audio_pcm_sha(str(media["master"]), _ffmpeg())
    assert sorted(p.name for p in out.parent.iterdir()) == [out.name], \
        "a candidate was left beside the final"


def test_a_pcm_mismatch_keeps_the_old_final_and_the_evidence(
        media, tmp_path, monkeypatch):
    out = _seed(tmp_path / "ep" / "ep_final.mkv")
    shas = iter(["a" * 64, "b" * 64])
    monkeypatch.setattr(MUX, "audio_pcm_sha", lambda path, fb=None: next(shas))
    with pytest.raises(ValueError, match="NOT byte-identical"):
        MUX.mux_master_audio(str(media["silent"]), str(media["master"]), str(out))
    assert out.read_bytes() == SENTINEL
    others = [p for p in out.parent.iterdir() if p != out]
    assert len(others) == 1 and others[0].name.startswith("otr_cand_")
    assert others[0].suffix == ".mkv", "the candidate must keep the final's container"


@pytest.mark.parametrize("partial", [True, False], ids=["partial_bytes", "nothing"])
def test_a_failed_archival_encode_leaves_the_old_final(
        media, tmp_path, monkeypatch, partial):
    """ffmpeg that opened its output and then failed: with -y on the final
    itself, the old bytes were gone before the failure was known."""
    out = _seed(tmp_path / "ep" / "ep_final.mkv")

    def run(argv, **kw):
        if partial:
            Path(argv[-1]).write_bytes(b"half a container")
        return subprocess.CompletedProcess(argv, 1, "", "mux refused")
    monkeypatch.setattr(MUX, "_run", run)
    with pytest.raises(ValueError, match="mux refused"):
        MUX.mux_master_audio(str(media["silent"]), str(media["master"]), str(out))
    assert out.read_bytes() == SENTINEL
    others = [p for p in out.parent.iterdir() if p != out]
    # bytes are evidence and are kept; an EMPTY reservation of this
    # invocation is removed
    assert len(others) == (1 if partial else 0)


def test_an_archival_file_with_no_video_is_refused(media, tmp_path, monkeypatch):
    out = _seed(tmp_path / "ep" / "ep_final.mkv")
    real = MUX._probe_float

    def no_video_in_candidates(path, stream):
        if Path(path).name.startswith("otr_cand_") and stream == "v:0":
            return MUX.STREAM_MISSING
        return real(path, stream)
    monkeypatch.setattr(MUX, "_probe_float", no_video_in_candidates)
    with pytest.raises(ValueError, match="NO video stream"):
        MUX.mux_master_audio(str(media["silent"]), str(media["master"]), str(out))
    assert out.read_bytes() == SENTINEL


# --------------------------------------------------------------------------- #
# 1. reservation: unique, exclusive, path-budgeted, non-ASCII safe
# --------------------------------------------------------------------------- #
def test_two_reservations_never_share_a_path(tmp_path, monkeypatch):
    final = tmp_path / "ep_final.mp4"
    first = MUX._reserve_candidate(str(final))
    # force a collision on the next draw: the exclusive create must refuse it
    # and draw again rather than share the reservation
    fixed = Path(first).stem[len("otr_cand_"):]
    draws = iter([fixed, "f" * 12])

    class _Fake:
        def __init__(self, h):
            self.hex = h + "0" * 20
    monkeypatch.setattr(MUX.uuid, "uuid4", lambda: _Fake(next(draws)))
    second = MUX._reserve_candidate(str(final))
    assert first != second
    assert os.path.isfile(first) and os.path.isfile(second)
    assert Path(second).suffix == ".mp4"


def test_a_deep_non_ascii_folder_reserves_and_publishes(media, tmp_path, monkeypatch):
    folder = tmp_path / ("épisode " + chr(0x65E5) + chr(0x672C) + " " + chr(0x2605))
    folder.mkdir()
    monkeypatch.setattr(MUX, "_obs_dir", lambda: folder)
    out = MUX.OTRMasterAudioMux()._publish_to_obs(str(media["av"]))
    assert Path(out).parent == folder and Path(out).is_file()


@pytest.mark.parametrize("room,prefix", [(25, "otr_cand_"), (17, "oc_"), (13, "c")])
def test_a_deep_folder_gets_a_shorter_candidate_not_a_refusal(
        tmp_path, monkeypatch, room, prefix):
    """A short operator final (`ep_final.mp4`, 12) can fit where the long
    candidate (25) would not; refusing there failed a render that used to
    write (codex QA on 733c2386)."""
    monkeypatch.setattr(MUX, "path_fits",
                        lambda path, budget=0: len(os.path.basename(path)) <= room)
    got = Path(MUX._reserve_candidate(str(tmp_path / "ep_final.mp4")))
    assert got.name.startswith(prefix) and got.suffix == ".mp4"
    assert len(got.name) <= room and "_final" not in got.name


def test_an_archival_final_inside_obs_stages_its_candidate_out_of_the_listing(
        media, obs, monkeypatch):
    """An operator `output_path` may name the watch folder on a cleared
    episode; a rejected archival candidate must still not appear in it."""
    out = _seed(obs / "chosen_final.mkv")
    shas = iter(["a" * 64, "b" * 64])
    monkeypatch.setattr(MUX, "audio_pcm_sha", lambda path, fb=None: next(shas))
    with pytest.raises(ValueError, match="NOT byte-identical"):
        MUX.mux_master_audio(str(media["silent"]), str(media["master"]), str(out))
    assert out.read_bytes() == SENTINEL
    assert sorted(p.name for p in obs.iterdir() if p.is_file()) == [out.name]
    assert len(_staged(obs)) == 1


def test_a_candidate_over_the_path_budget_is_refused_before_any_write(
        tmp_path, monkeypatch):
    final = _seed(tmp_path / "ep_final.mp4")
    monkeypatch.setattr(MUX, "path_fits", lambda path, budget=0: False)
    with pytest.raises(OSError, match="path budget"):
        MUX._reserve_candidate(str(final))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ep_final.mp4"]
    assert final.read_bytes() == SENTINEL


# --------------------------------------------------------------------------- #
# 3. Cancel at the finishing boundaries -- Comfy's exception, untouched
# --------------------------------------------------------------------------- #
class InterruptProcessingException(BaseException):
    """A faithful stand-in: the current runtime derives it from BaseException."""


def _install_comfy(monkeypatch, exc_type=InterruptProcessingException):
    state = {"cancel": False}

    def check():
        if state["cancel"]:
            state["cancel"] = False        # Comfy clears the flag as it raises
            raise exc_type("Processing interrupted")

    comfy = types.ModuleType("comfy")
    mm = types.ModuleType("comfy.model_management")
    mm.throw_exception_if_processing_interrupted = check
    comfy.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    return state


def test_no_comfy_means_no_cancel_check(monkeypatch):
    monkeypatch.setitem(sys.modules, "comfy.model_management", None)
    assert MUX._poll_interrupt() is None


def test_an_exception_derived_interrupt_is_not_swallowed(monkeypatch):
    """Older ComfyUI derived its interrupt from Exception; the old poll caught
    it with its own `except Exception` and the Cancel vanished."""
    class OldInterrupt(Exception):
        pass
    state = _install_comfy(monkeypatch, OldInterrupt)
    state["cancel"] = True
    with pytest.raises(OldInterrupt):
        MUX._poll_interrupt()


@pytest.fixture
def episode_run(tmp_path, monkeypatch, obs):
    """Drive the real mux() with the media stubbed, recording every finishing
    side effect, on an episode the publication decision clears."""
    master = tmp_path / "ep" / "ep_master.wav"
    master.parent.mkdir(parents=True)
    master.write_bytes(b"RIFF")
    final = str(tmp_path / "ep" / "ep_silent_final.mp4")
    calls = []
    monkeypatch.setattr(MUX, "_publication_decision", lambda path: types.SimpleNamespace(
        publishable=True, summary=lambda: "cleared"))
    monkeypatch.setattr(MUX.OTRMasterAudioMux, "_default_out", lambda self, p: final)
    monkeypatch.setattr(MUX.OTRMasterAudioMux, "_stamp_terminal_paths",
                        lambda self, *a: calls.append("stamp") or "stamped")
    monkeypatch.setattr(MUX.OTRMasterAudioMux, "_asset_cleanup",
                        lambda self, *a: calls.append("cleanup") or (None, False))
    monkeypatch.setattr(MUX, "_canvas_preview",
                        lambda f, o: calls.append("preview") or {})
    from nodes import _otr_janitor
    monkeypatch.setattr(_otr_janitor, "sweep_shared_tmp",
                        lambda: calls.append("janitor") or types.SimpleNamespace(deleted=[]))

    def publish(self, f):
        calls.append("publish")
        dst = obs / "published.mp4"
        dst.write_bytes(b"published")
        return str(dst)
    monkeypatch.setattr(MUX.OTRMasterAudioMux, "_publish_to_obs", publish)
    monkeypatch.setattr(MUX, "mux_master_audio",
                        lambda *a, **k: calls.append("mux") or (final, ["mux OK"]))

    def run():
        return MUX.OTRMasterAudioMux().mux(str(tmp_path / "ep" / "ep_silent.mp4"),
                                           str(master))
    return types.SimpleNamespace(run=run, calls=calls, obs=obs)


def test_a_cancel_during_the_mux_stops_before_publication(episode_run, monkeypatch):
    state = _install_comfy(monkeypatch)
    real_mux = MUX.mux_master_audio

    def mux_then_cancel(*a, **k):
        out = real_mux(*a, **k)
        state["cancel"] = True
        return out
    monkeypatch.setattr(MUX, "mux_master_audio", mux_then_cancel)
    with pytest.raises(InterruptProcessingException):
        episode_run.run()
    assert episode_run.calls == ["mux"], episode_run.calls


def test_a_cancel_during_the_obs_encode_keeps_the_old_copy(media, obs, monkeypatch):
    final = str(media["av"])
    dst = _seed(obs / MUX._obs_basename(final))
    state = _install_comfy(monkeypatch)
    copy = _writes(media["av"])

    def encode_then_cancel(argv, **kw):
        result = copy(argv)
        state["cancel"] = True
        return result
    monkeypatch.setattr(MUX, "_run", encode_then_cancel)
    with pytest.raises(InterruptProcessingException):
        MUX.OTRMasterAudioMux()._publish_to_obs(final)
    assert dst.read_bytes() == SENTINEL, "a cancelled publish replaced the copy"
    assert _staged(obs) == [], "a cancelled candidate must not be orphaned"


def test_a_cancel_after_publication_skips_the_janitor_and_keeps_the_copy(
        episode_run, monkeypatch):
    state = _install_comfy(monkeypatch)
    original = MUX.OTRMasterAudioMux._stamp_terminal_paths

    def stamp_then_cancel(self, *a):
        state["cancel"] = True
        return original(self, *a)
    monkeypatch.setattr(MUX.OTRMasterAudioMux, "_stamp_terminal_paths", stamp_then_cancel)
    with pytest.raises(InterruptProcessingException):
        episode_run.run()
    assert episode_run.calls == ["mux", "publish", "stamp"], episode_run.calls
    assert (episode_run.obs / "published.mp4").read_bytes() == b"published"


def test_a_cancel_during_the_preview_skips_the_asset_cleanup(episode_run, monkeypatch):
    state = _install_comfy(monkeypatch)

    def preview_then_cancel(f, o):
        episode_run.calls.append("preview")
        state["cancel"] = True
        return {}
    monkeypatch.setattr(MUX, "_canvas_preview", preview_then_cancel)
    with pytest.raises(InterruptProcessingException):
        episode_run.run()
    assert episode_run.calls == ["mux", "publish", "stamp", "janitor", "preview"]
    assert (episode_run.obs / "published.mp4").read_bytes() == b"published"


def test_without_a_cancel_every_finishing_step_runs(episode_run, monkeypatch):
    _install_comfy(monkeypatch)
    episode_run.run()
    assert episode_run.calls == ["mux", "publish", "stamp", "janitor", "preview",
                                 "cleanup"]


# --------------------------------------------------------------------------- #
# 2. the terminal stamp writes only THIS episode's ledger
# --------------------------------------------------------------------------- #
@pytest.fixture
def episodes(tmp_path, monkeypatch):
    root = tmp_path / "output" / "otr" / "episodes"
    root.mkdir(parents=True)
    monkeypatch.setattr(MUX, "_episodes_root", lambda: root)

    def ledger_for(ep_id):
        path = root / ep_id / "audio" / ("%s_ledger.json" % ep_id)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({
            "episode_id": ep_id, "final_video_path": "theirs.mp4",
            "final_audio_path": "theirs.wav",
            "meta": {"obs_final_path": "", "paths": {}}}), encoding="utf-8")
        return path
    return types.SimpleNamespace(root=root, ledger_for=ledger_for)


@pytest.mark.parametrize("ledger_ep,video_ep", [("ep10", "ep1"), ("ep1", "ep10")],
                         ids=["ep10_ledger_ep1_video", "ep1_ledger_ep10_video"])
def test_a_foreign_singleton_is_never_stamped(episodes, monkeypatch, ledger_ep, video_ep):
    foreign = episodes.ledger_for(ledger_ep)
    before = foreign.read_bytes()
    monkeypatch.setattr(OTRL, "in_flight_ledger_path", lambda: foreign)
    line = MUX.OTRMasterAudioMux()._stamp_terminal_paths(
        "mine_final.mp4", None, "mine.wav", "%s_silent_with_credits" % video_ep)
    assert line.startswith("terminal path stamp skipped")
    assert foreign.read_bytes() == before, "another episode's ledger was written"


def test_the_renamed_episode_still_stamps(episodes, monkeypatch, tmp_path):
    ep = "signal_lost_the_three_boxes_20261014_123150"
    ledger = episodes.ledger_for(ep)
    monkeypatch.setattr(OTRL, "in_flight_ledger_path", lambda: ledger)
    final = episodes.root / ep / ("%s_final.mp4" % ep)
    final.write_bytes(b"final")
    line = MUX.OTRMasterAudioMux()._stamp_terminal_paths(
        str(final), None, "master.wav",
        ep + "_silent_procgen_blended_captioned_with_credits")
    assert line.startswith("stamped ledger"), line
    saved = json.loads(ledger.read_text(encoding="utf-8"))
    assert saved["final_video_path"] == str(final)


def test_a_singleton_that_moves_mid_run_is_not_written(
        episodes, monkeypatch, tmp_path, obs):
    """The singleton names THIS episode when the mux starts and another one by
    the time the stamp runs: the stamp skips, the other ledger is untouched."""
    mine = episodes.ledger_for("ep1")
    theirs = episodes.ledger_for("ep10")
    before = theirs.read_bytes()
    current = {"path": mine}
    monkeypatch.setattr(OTRL, "in_flight_ledger_path", lambda: current["path"])
    monkeypatch.setattr(MUX, "_publication_decision", lambda path: types.SimpleNamespace(
        publishable=False, summary=lambda: "withheld for the test"))
    final = str(episodes.root / "ep1" / "ep1_silent_final.mp4")

    def mux_then_move(*a, **k):
        current["path"] = theirs
        return final, ["mux OK"]
    monkeypatch.setattr(MUX, "mux_master_audio", mux_then_move)
    monkeypatch.setattr(MUX, "_canvas_preview", lambda f, o: {})
    master = episodes.root / "ep1" / "audio" / "ep1_master.wav"
    master.write_bytes(b"RIFF")
    _path, report = MUX.OTRMasterAudioMux().mux(
        str(episodes.root / "ep1" / "ep1_silent.mp4"), str(master))["result"]
    assert theirs.read_bytes() == before
    assert "terminal path stamp skipped" in report
