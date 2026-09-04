"""``has_nvenc(ffmpeg)`` takes a binary; its cache must be keyed by it.

The probe was cached in one process-global boolean (kibitz runpod-found-fixes
r2, Codex must-fix 6): the first answer was reused for every later binary.
Once resolution is centralized, two callers can still legally name different
builds -- an explicit widget path beside the pinned default -- and the second
must be probed on its own.
"""
from __future__ import annotations

from types import SimpleNamespace

from nodes._otr_shared import encode_sink as es


def test_two_binaries_get_two_verdicts_and_each_is_cached(monkeypatch):
    monkeypatch.setattr(es, "_NVENC_PROBE", {})
    probed = []

    def run_by_binary(cmd, **_kw):
        probed.append(cmd[0])
        return SimpleNamespace(returncode=0 if "good" in cmd[0] else 1)

    monkeypatch.setattr(es.otr_proc, "run", run_by_binary)

    assert es.has_nvenc("/opt/good/ffmpeg") is True
    assert es.has_nvenc("/opt/bad/ffmpeg") is False
    assert es.has_nvenc("/opt/good/ffmpeg") is True
    assert es.has_nvenc("/opt/bad/ffmpeg") is False
    assert probed == ["/opt/good/ffmpeg", "/opt/bad/ffmpeg"], probed


def test_the_same_binary_spelled_two_ways_is_one_probe(monkeypatch):
    monkeypatch.setattr(es, "_NVENC_PROBE", {})
    probed = []

    def run_ok(cmd, **_kw):
        probed.append(cmd[0])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(es.otr_proc, "run", run_ok)
    import os
    here = os.path.abspath("ffmpeg.exe")
    assert es.has_nvenc(here) is True
    assert es.has_nvenc(os.path.join(os.path.dirname(here), ".", "ffmpeg.exe")) is True
    assert len(probed) == 1, probed


def test_signal_lost_asks_the_owner_for_every_binary(monkeypatch):
    """`video_engine._check_nvenc` kept a second, process-global cache IN FRONT
    of the owner (kibitz r3), so the first binary's verdict came back for every
    later one and the owner's per-binary key was never consulted."""
    from nodes import video_engine as ve
    monkeypatch.setattr(es, "_NVENC_PROBE", {})
    monkeypatch.setattr(ve, "_NVENC_ANNOUNCED", set())
    probed = []

    def run_by_binary(cmd, **_kw):
        probed.append(cmd[0])
        return SimpleNamespace(returncode=0 if "good" in cmd[0] else 1)

    monkeypatch.setattr(es.otr_proc, "run", run_by_binary)
    assert ve._check_nvenc("/opt/good/ffmpeg") is True
    assert ve._check_nvenc("/opt/bad/ffmpeg") is False
    assert ve._check_nvenc("/opt/good/ffmpeg") is True
    assert probed == ["/opt/good/ffmpeg", "/opt/bad/ffmpeg"], probed
    assert ve._check_nvenc(None) is False
    assert ve._check_nvenc("") is False


def test_an_empty_binary_is_false_and_never_keys_the_cache_on_cwd(monkeypatch):
    monkeypatch.setattr(es, "_NVENC_PROBE", {})
    calls = []
    monkeypatch.setattr(es.otr_proc, "run",
                        lambda cmd, **_kw: calls.append(cmd) or SimpleNamespace(returncode=0))
    assert es.has_nvenc("") is False
    assert es.has_nvenc(None) is False
    assert calls == []
    assert es._NVENC_PROBE == {}


def test_a_bare_name_is_resolved_through_the_owner_before_it_keys(monkeypatch):
    """`has_nvenc("ffmpeg")` used to key on <cwd>/ffmpeg and probe whatever
    PATH said (agy r4). A bare name goes through the owner -- the pin, then
    PATH -- so the key and the probed binary are the one this box runs."""
    monkeypatch.setattr(es, "_NVENC_PROBE", {})
    monkeypatch.setattr(es, "resolve_ffmpeg", lambda p=None: "/x/pinned/ffmpeg")
    probed = []
    monkeypatch.setattr(es.otr_proc, "run",
                        lambda cmd, **_kw: probed.append(cmd[0]) or SimpleNamespace(returncode=0))
    assert es.has_nvenc("ffmpeg") is True
    assert probed == ["/x/pinned/ffmpeg"]
    import os
    assert list(es._NVENC_PROBE) == [os.path.normcase(os.path.abspath("/x/pinned/ffmpeg"))]
    # an explicit path is the caller's choice: probed as given, never re-resolved
    assert es.has_nvenc("/opt/explicit/ffmpeg") is True
    assert probed[-1] == "/opt/explicit/ffmpeg"


def test_an_unresolvable_bare_name_is_false_and_never_keys_the_cache(monkeypatch):
    """cursor r4: with nothing to resolve to, `has_nvenc("ffmpeg")` must
    not fall back to keying <cwd>/ffmpeg and probing a name that is not
    there."""
    monkeypatch.setattr(es, "_NVENC_PROBE", {})
    monkeypatch.setattr(es, "resolve_ffmpeg", lambda p=None: None)
    calls = []
    monkeypatch.setattr(es.otr_proc, "run",
                        lambda cmd, **_kw: calls.append(cmd) or SimpleNamespace(returncode=0))
    assert es.has_nvenc("ffmpeg") is False
    assert calls == []
    assert es._NVENC_PROBE == {}


def test_a_probe_that_explodes_is_false_and_cached_for_that_binary(monkeypatch):
    monkeypatch.setattr(es, "_NVENC_PROBE", {})
    calls = []

    def boom(cmd, **_kw):
        calls.append(cmd[0])
        raise OSError("no such binary")

    monkeypatch.setattr(es.otr_proc, "run", boom)
    assert es.has_nvenc("/nowhere/ffmpeg") is False
    assert es.has_nvenc("/nowhere/ffmpeg") is False
    assert len(calls) == 1
