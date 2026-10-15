"""Replay episode ids are recognized wherever an id is parsed (plan item 19).

`production_ledger.replay_episode_id` appends `_rp<8 hex>` to the source id,
and replaying a frozen replay appends again. The OBS identity-tail parser only
knew the older `_replay_<date>_<time>_<us>` stamp, so a replay published under
its native fallback name instead of its gloss; the API runner's receipt parser
knew neither and named the SOURCE episode.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from nodes import production_ledger as pl  # noqa: E402
from nodes._otr_shared import obs_name  # noqa: E402

SOURCE = "signal_lost_the_quiet_relay_20261014_181214"


def _replay(source):
    return pl.replay_episode_id(source)


def test_the_producer_still_mints_rp_plus_eight_hex():
    rid = _replay(SOURCE)
    assert re.fullmatch(re.escape(SOURCE) + r"_rp[0-9a-f]{8}", rid)


@pytest.mark.parametrize("episode_id,tail", [
    (SOURCE, "_20261014_181214"),
    (SOURCE + "_replay_20261014_190000_123456", "_20261014_181214_replay_20261014_190000_123456"),
])
def test_ordinary_and_legacy_ids_keep_their_tails(episode_id, tail):
    assert obs_name.identity_tail(episode_id) == tail


def test_a_replay_and_a_replay_of_a_replay_keep_their_full_identity():
    once = _replay(SOURCE)
    twice = _replay(once)
    assert obs_name.identity_tail(once) == "_20261014_181214" + once[len(SOURCE):]
    assert obs_name.identity_tail(twice) == "_20261014_181214" + twice[len(SOURCE):]
    assert obs_name.obs_stem("Quiet relay", once) == (
        "quiet_relay_20261014_181214" + once[len(SOURCE):])
    # Two replays of one source never collapse to the same published stem.
    assert obs_name.obs_stem("Quiet relay", once) != obs_name.obs_stem(
        "Quiet relay", _replay(SOURCE))


def test_a_language_iso_still_has_to_match_the_ledger():
    es = "signal_lost_negativo_en_peligro_es_20261014_181214"
    rid = _replay(es)
    assert obs_name.identity_tail(rid, "es") == "_es_20261014_181214" + rid[len(es):]
    assert obs_name.identity_tail(rid, "fr") == ""
    # A title word that looks like a language code is not read as one.
    it = "signal_lost_made_of_it_20261014_181214"
    assert obs_name.identity_tail(_replay(it)).startswith("_20261014_181214_rp")


@pytest.mark.parametrize("bad", [
    SOURCE + "_rp1234567",          # seven hex
    SOURCE + "_rp123456789",        # nine hex
    SOURCE + "_rpGGGGGGGG",         # not hex
    SOURCE + "_rp",                 # empty
    SOURCE + "_rp12345678x",        # trailing junk
])
def test_partial_or_invalid_stamps_are_not_an_identity(bad):
    assert obs_name.identity_tail(bad) == ""
    assert obs_name.obs_stem("Quiet relay", bad) == ""


def test_a_missing_gloss_still_falls_back_to_the_native_name():
    assert obs_name.obs_stem("", _replay(SOURCE)) == ""


def _history(*paths):
    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"p": {"prompt": [0, "p", {"1": {"inputs": {"replay_from": paths[0]}}}, {}],
                          "outputs": {"81": {"files": [{"path": p} for p in paths[1:]]}}}}
    return _Response()


@pytest.mark.parametrize("sep", ["\\", "/"])
def test_the_api_receipt_names_the_replay_not_its_source(monkeypatch, sep):
    import requests
    import otr_canonical_api_run as canonical
    rid = _replay(SOURCE)
    src = sep.join(["C:", "out", "otr", "episodes", SOURCE, "replay", "bundle.json"])
    out = sep.join(["C:", "out", "otr", "episodes", rid, "audio", "master.wav"])
    monkeypatch.setattr(requests, "get", lambda *_a, **_k: _history(src, out))
    assert canonical.episode_of_prompt("p") == rid
    chained = _replay(rid)
    out2 = sep.join(["C:", "out", "otr", "episodes", chained, "clips", "b.mp4"])
    monkeypatch.setattr(requests, "get", lambda *_a, **_k: _history(out, out2))
    assert canonical.episode_of_prompt("p") == chained
