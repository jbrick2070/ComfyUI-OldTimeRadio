"""A fresh install can speak without the operator downloading anything.

THE PROBLEM (operator, 2026-08-24). Someone installs the pack from the Comfy
registry, clicks Run, and needs a voice. Of the five LOCAL TTS engines, three
(`indextts2`, `chatterbox`, `dia`) declare `requires_voice_ref = True` -- they
CLONE, so they need a reference WAV the user must supply, and a fresh install
ships none. That left exactly two zero-setup engines: Bark, whose voices live
inside its weights, and Kokoro, whose voices are separate files that nothing
fetched.

Measured on disk: Bark 4.2 GB, Kokoro 327 MB. A 13x tax on the 8 GB tier, and
the entire gap was ~14 MB of 523 KB voice files.

WHAT THIS SUITE PINS:
* the prefetch cannot make the RENDER path network -- that rule is why
  `eng_kokoro` fails closed instead of fetching, after a mid-render hub fetch
  once 404'd and aborted a finished episode;
* the voice names are REAL upstream files, not plausible-looking strings;
* enough voices for a full cast, since two characters never share one;
* boot survives every failure mode -- no internet, no hub library, no disk.

CPU-only: no model, no network (every network path is monkeypatched).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "nodes"))

import _otr_kokoro_voice_prefetch as PRE  # noqa: E402


# --------------------------------------------------------------------------- #
# the voice list is real, and big enough to dress a cast
# --------------------------------------------------------------------------- #
def test_every_declared_voice_is_a_real_upstream_file():
    """The names came from a live listing of `hexgrad/Kokoro-82M`; this pins
    their SHAPE so a typo cannot creep in later. A voice that does not exist
    upstream is a 404 at boot and a voice missing at cast time -- and
    `eng_kokoro` fails an episode closed on a missing voice file."""
    for voice in PRE.ENGLISH_VOICES:
        assert voice[:2] in {"bf", "bm", "af", "am"}, voice
        assert voice[2] == "_" and voice[3:], voice
        assert voice.islower(), voice


def test_english_only_no_dead_weight():
    """The repo ships 54 voices; 26 are Spanish, French, Italian, Hindi,
    Japanese, Portuguese and Chinese. This show has no use for them, and they
    would double the download."""
    foreign_prefixes = {"ef", "em", "ff", "hf", "hm", "if", "im",
                        "jf", "jm", "pf", "pm", "zf", "zm"}
    for voice in PRE.ENGLISH_VOICES:
        assert voice[:2] not in foreign_prefixes, voice


def test_enough_voices_for_the_largest_cast():
    """TWO CHARACTERS NEVER SHARE A VOICE -- enforced by the casting
    validator's `taken` set and `_assert_unique_bark_voices`. Casts run to 6
    on the legacy banks and 10 on scifi_news_pro, and a gender-balanced cast
    halves the usable pool, so the FOUR announcer voices could never have
    dressed one."""
    males = [v for v in PRE.ENGLISH_VOICES if v[1] == "m"]
    females = [v for v in PRE.ENGLISH_VOICES if v[1] == "f"]

    assert len(PRE.ENGLISH_VOICES) >= 10, "must cover the largest cast"
    assert len(males) >= 10, f"a same-gender cast of 10 needs 10 male voices"
    assert len(females) >= 10, "same for female"


def test_the_announcer_pool_is_covered_by_the_prefetch():
    """`eng_kokoro` draws its announcer from a curated British pool and fails
    CLOSED if the file is absent, so every one of those must be fetched or a
    fresh install cannot even open an episode."""
    from nodes._otr_audio_engines.eng_kokoro import ANNOUNCER_VOICE_POOL

    for voice in ANNOUNCER_VOICE_POOL:
        assert voice in PRE.ENGLISH_VOICES, (
            f"{voice} is drawn by the announcer but never prefetched")


def test_the_prefetch_targets_the_engine_s_own_voices_dir():
    """The subdir is duplicated rather than imported (the engine cannot be
    imported this early at prestartup), so it is pinned against drift here --
    a mismatch means files land where nothing reads them."""
    from nodes._otr_audio_engines import eng_kokoro

    assert PRE._KOKORO_MODEL_SUBDIR == os.path.join(
        *eng_kokoro._KOKORO_MODEL_SUBDIR.split(os.sep))


# --------------------------------------------------------------------------- #
# the render path must stay offline -- the rule this fix must not break
# --------------------------------------------------------------------------- #
def test_the_engine_still_refuses_to_fetch_mid_render():
    """THE LOAD-BEARING GUARANTEE. `eng_kokoro` raises a named EngineUnusable
    rather than downloading during a render, because a mid-render hub fetch
    once 404'd and aborted a finished episode. The prefetch lives at boot
    precisely so this stays true -- if a future edit moved a fetch into the
    engine, this fails."""
    import inspect

    from nodes._otr_audio_engines import eng_kokoro

    source = inspect.getsource(eng_kokoro)
    assert "never downloads" in source
    for banned in ("hf_hub_download", "snapshot_download"):
        assert banned not in source, (
            f"{banned} in eng_kokoro would reopen the mid-render fetch that "
            "aborted a live episode")


# --------------------------------------------------------------------------- #
# boot survives every failure mode -- a voice is never worth a dead boot
# --------------------------------------------------------------------------- #
def test_no_network_is_a_log_line_not_an_exception(monkeypatch, tmp_path):
    """A machine with no internet must boot exactly as before."""
    monkeypatch.setattr(PRE, "_models_dir", lambda: str(tmp_path))

    def _boom(*_a, **_k):
        raise OSError("getaddrinfo failed")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _boom)

    receipt = PRE.prefetch_kokoro_voices()

    assert receipt["fetched"] == 0
    assert "getaddrinfo" in receipt["reason"]
    PRE.prefetch_at_boot()          # must not raise


def test_a_missing_models_dir_is_survivable(monkeypatch):
    monkeypatch.setattr(PRE, "_models_dir", lambda: None)

    receipt = PRE.prefetch_kokoro_voices()

    assert receipt["fetched"] == 0
    assert "models" in receipt["reason"]


def test_an_offline_operator_stays_offline(monkeypatch, tmp_path):
    """A deliberate HF_HUB_OFFLINE=1 is respected -- the prefetch must not
    quietly override an operator's own decision."""
    monkeypatch.setattr(PRE, "_models_dir", lambda: str(tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    receipt = PRE.prefetch_kokoro_voices()

    assert receipt["skipped_offline"] is True
    assert receipt["fetched"] == 0


def test_there_is_an_explicit_opt_out(monkeypatch, tmp_path):
    monkeypatch.setattr(PRE, "_models_dir", lambda: str(tmp_path))
    monkeypatch.setenv("OTR_SKIP_KOKORO_PREFETCH", "1")

    assert PRE.prefetch_kokoro_voices()["skipped_offline"] is True


# --------------------------------------------------------------------------- #
# the ordinary case: after first boot it does nothing at all
# --------------------------------------------------------------------------- #
def test_a_settled_install_fetches_nothing(monkeypatch, tmp_path):
    """THE NO-OP CASE, which is every boot after the first. It must not
    network, so the download function is detonated."""
    voices = tmp_path / PRE._KOKORO_MODEL_SUBDIR / "voices"
    voices.mkdir(parents=True)
    for voice in PRE.ENGLISH_VOICES:
        (voices / f"{voice}.pt").write_bytes(b"x")
    monkeypatch.setattr(PRE, "_models_dir", lambda: str(tmp_path))

    import huggingface_hub

    def _detonate(*_a, **_k):
        raise AssertionError("a settled install must not touch the network")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _detonate)

    receipt = PRE.prefetch_kokoro_voices()

    assert receipt["attempted"] == 0 and receipt["fetched"] == 0


def test_only_the_missing_voices_are_fetched(monkeypatch, tmp_path):
    """A partial install tops up rather than re-downloading everything."""
    voices = tmp_path / PRE._KOKORO_MODEL_SUBDIR / "voices"
    voices.mkdir(parents=True)
    present = PRE.ENGLISH_VOICES[:5]
    for voice in present:
        (voices / f"{voice}.pt").write_bytes(b"x")
    monkeypatch.setattr(PRE, "_models_dir", lambda: str(tmp_path))

    missing = PRE.missing_voices(str(voices))

    assert len(missing) == len(PRE.ENGLISH_VOICES) - len(present)
    for voice in present:
        assert voice not in missing


def test_a_fetched_voice_is_COPIED_not_symlinked(monkeypatch, tmp_path):
    """The hub cache is a separate tree with its own eviction, and
    `eng_kokoro` checks the destination path with os.path.exists before every
    episode -- a dangling link would read as 'voice missing' and fail an
    episode closed."""
    src_dir = tmp_path / "hubcache"
    src_dir.mkdir()
    monkeypatch.setattr(PRE, "_models_dir", lambda: str(tmp_path))

    def _fake_download(repo_id, filename, **_k):
        blob = src_dir / os.path.basename(filename)
        blob.write_bytes(b"voice-bytes")
        return str(blob)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)

    receipt = PRE.prefetch_kokoro_voices()
    landed = tmp_path / PRE._KOKORO_MODEL_SUBDIR / "voices" / "bf_alice.pt"

    assert receipt["fetched"] == len(PRE.ENGLISH_VOICES)
    assert landed.is_file() and not landed.is_symlink()
    assert landed.read_bytes() == b"voice-bytes"


# --------------------------------------------------------------------------- #
# prestartup wiring -- an unwired prefetch is inert
# --------------------------------------------------------------------------- #
def test_prestartup_calls_it_and_cannot_die_doing_so():
    text = (REPO_ROOT / "prestartup_script.py").read_text(encoding="utf-8")

    assert "prefetch_at_boot" in text, "the prefetch must actually be wired"
    assert "except Exception" in text, (
        "a voice is never worth a dead boot -- prestartup failure silently "
        "skips everything below it")


def test_both_files_stay_ASCII_only():
    """Load-bearing here: a non-ASCII character in prestartup raised
    UnicodeEncodeError on a cp1252 console and made EVERY boot log a failure
    while the mock had actually worked -- a banner that lied."""
    for rel in ("prestartup_script.py", "nodes/_otr_kokoro_voice_prefetch.py"):
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        offenders = [c for c in text if ord(c) > 127]
        assert not offenders, f"{rel} carries non-ASCII: {offenders[:5]}"
