"""Cloud ElevenLabs CAST MAPPING (2026-07-04).

The elevenlabs adapter fails LOUD ("no provider_voice_id on this cast entry")
unless CastLock stamps the provider voice_id. That needs (a) an elevenlabs voice
POOL in the bank (rows with engine=='elevenlabs' + a real provider_voice_id) and
(b) CastLock resolving the char engine to elevenlabs under the 'elevenlabs_cloud'
bank. These tests guard both, plus the invariant that LOCAL banks never gain a
provider_voice_id (byte-identical local cast).
"""
import json
import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OTR_TEST_MODE", "1")

from nodes._otr_voice_bank import VoiceCastingError, load_voice_bank
from nodes.cast_lock import CastLock


_STOCK_ELEVENLABS_PARTNER_VOICES = {
    "CwhRBWXzGAHq8TQ4Fs17": ("el_roger", "male", "american"),
    "EXAVITQu4vr4xnSDxMaL": ("el_sarah", "female", "american"),
    "FGY2WhTYpPnrIDTdsKH5": ("el_laura", "female", "american"),
    "IKne3meq5aSn9XLyUdCD": ("el_charlie", "male", "australian"),
    "JBFqnCBsd6RMkjVDRZzb": ("el_george", "male", "british"),
    "N2lVS1w4EtoT3dr4eOWO": ("el_callum", "male", "american"),
    "SAz9YHcvj6GT2YYXdXww": ("el_river", "neutral", "american"),
    "SOYHLrjzK2X1ezoPC6cr": ("el_harry", "male", "american"),
    "TX3LPaxmHKxFdv7VOQHJ": ("el_liam", "male", "american"),
    "Xb7hH8MSUJpSbSDYk0k2": ("el_alice", "female", "british"),
    "XrExE9yKIg1WjnnlVkGX": ("el_matilda", "female", "american"),
    "bIHbv24MWmeRgasZH58o": ("el_will", "male", "american"),
    "cgSgspJ2msm6clMCkdW9": ("el_jessica", "female", "american"),
    "cjVigY5qzO86Huf0OWal": ("el_eric", "male", "american"),
    "hpp4J3VqNfWAUOO0d1Us": ("el_bella", "female", "american"),
    "iP95p4xoKVk53GoZ742B": ("el_chris", "male", "american"),
    "nPczCjzI2devNBz1zQrb": ("el_brian", "male", "american"),
    "onwK4e9ZLuTAKqWW03F9": ("el_daniel", "male", "british"),
    "pFZP5JQG7iQjIQuC4Bku": ("el_lily", "female", "british"),
    "pNInz6obpgDQGcFmaJgB": ("el_adam", "male", "american"),
    "pqHfZKP75CvOlQylNhV4": ("el_bill", "male", "american"),
}


def test_bank_has_elevenlabs_pool_with_provider_ids():
    entries, _sha = load_voice_bank()
    el = [e for e in entries if e.engine == "elevenlabs"]
    assert el, "no elevenlabs voice rows in the bank"
    assert all(e.provider_voice_id for e in el), \
        "every elevenlabs row must carry a provider_voice_id"
    genders = {e.gender for e in el}
    assert {"male", "female"} <= genders, \
        "elevenlabs pool must cover male + female for the caster gender match"


def test_bank_matches_stock_partner_voice_catalog():
    entries, _sha = load_voice_bank()
    el = [e for e in entries if e.engine == "elevenlabs"]
    by_provider = {e.provider_voice_id: e for e in el}
    assert set(by_provider) == set(_STOCK_ELEVENLABS_PARTNER_VOICES)
    for provider_id, (voice_ref_id, gender, accent) in _STOCK_ELEVENLABS_PARTNER_VOICES.items():
        entry = by_provider[provider_id]
        assert entry.voice_ref_id == voice_ref_id
        assert entry.gender == gender
        assert accent in entry.timbre
        assert entry.ref_path == "cloud:elevenlabs:%s" % provider_id
        assert entry.ref_sha256 == "cloud"


def test_elevenlabs_announcer_pool_has_castable_provider_ids():
    entries, _sha = load_voice_bank()
    announcers = [
        e for e in entries
        if e.engine == "elevenlabs" and "announcer_voice" in e.roles
    ]
    assert len(announcers) >= 3
    assert all(e.provider_voice_id for e in announcers)
    assert {"male", "female"} <= {e.gender for e in announcers}


def _led():
    return {"meta": {"episode_seed": 12345}, "lines": [
        {"line_id": "b002", "char_id": "c02", "speaker_role": "character", "text": "hi"},
        {"line_id": "b003", "char_id": "c03", "speaker_role": "character", "text": "yo"},
    ]}


def test_elevenlabs_cloud_bank_stamps_provider_voice_id():
    cast = [
        {"char_id": "c02", "name": "MARGOT", "gender": "female"},
        {"char_id": "c03", "name": "DOLPH", "gender": "male"},
        {"name": "ANNOUNCER", "speaker_role": "announcer", "char_id": "announcer"},
    ]
    CastLock()._auto_registry(_led(), cast, "elevenlabs_cloud", False, [])
    chars = [e for e in cast if e.get("name") != "ANNOUNCER"]
    for e in chars:
        assert e.get("voice_engine") == "elevenlabs", (e.get("name"), e)
        assert e.get("provider_voice_id"), \
            "%s got no provider_voice_id -> adapter would fail loud" % e.get("name")
    # deterministic + gender-appropriate id assigned (no bare fail-loud)
    assert chars[0]["provider_voice_id"] != chars[1]["provider_voice_id"]


def test_explicit_elevenlabs_voice_engines_stamp_characters_and_announcer():
    cast = [
        {"char_id": "c02", "name": "MARGOT", "gender": "female",
         "voice_preset": "v2/en_speaker_2"},
        {"name": "ANNOUNCER", "speaker_role": "announcer",
         "char_id": "announcer", "voice_preset": "v2/en_speaker_6"},
    ]
    led = _led()
    report = []
    CastLock()._auto_registry(
        led, cast, "elevenlabs_cloud", False, report,
        char_voice_engine="elevenlabs",
        announcer_voice_engine="elevenlabs")
    by_id = {e["char_id"]: e for e in cast}
    assert by_id["c02"]["voice_engine"] == "elevenlabs"
    assert by_id["c02"]["provider_voice_id"]
    assert by_id["announcer"]["voice_engine"] == "elevenlabs"
    assert by_id["announcer"]["provider_voice_id"]
    assert led["meta"]["char_voice_engine"] == "elevenlabs"
    assert led["meta"]["announcer_voice_engine"] == "elevenlabs"


def test_preserve_ledger_stamps_explicit_elevenlabs_voice_engines():
    led = {
        "meta": {"episode_seed": 12345},
        "cast": [
            {"char_id": "c02", "name": "MARGOT", "gender": "female",
             "voice_preset": "v2/en_speaker_2"},
            {"name": "ANNOUNCER", "speaker_role": "announcer",
             "char_id": "announcer", "voice_preset": "v2/en_speaker_6"},
        ],
        "lines": [
            {"line_id": "b002", "char_id": "c02",
             "speaker_role": "character", "text": "hi"},
            {"line_id": "b001", "char_id": "announcer",
             "speaker_role": "announcer", "text": "Tonight."},
        ],
    }
    ledger_json, _rev, report, _done = CastLock().lock(
        json.dumps(led),
        voice_bank="elevenlabs_cloud",
        cast_voice_policy="preserve_ledger",
        allow_voice_reuse=False,
        char_voice_engine="elevenlabs",
        announcer_voice_engine="elevenlabs",
    )
    out = json.loads(ledger_json)
    assert "preserve_ledger" in report
    assert out["meta"]["voice_bank_id"] == "elevenlabs_cloud"
    assert out["meta"]["char_voice_engine"] == "elevenlabs"
    assert out["meta"]["announcer_voice_engine"] == "elevenlabs"


def test_explicit_elevenlabs_rejects_incompatible_voice_bank():
    entries, _sha = load_voice_bank()
    with pytest.raises(VoiceCastingError, match="voice_bank .* not allowed"):
        CastLock._resolve_char_engine("default", entries, "elevenlabs")


def test_default_bank_leaves_provider_voice_id_unset():
    cast = [{"char_id": "c02", "name": "MARGOT", "gender": "female"}]
    CastLock()._auto_registry({"meta": {"episode_seed": 1}, "lines": []},
                              cast, "default", False, [])
    assert cast[0].get("provider_voice_id") in (None, ""), \
        "local (default) bank must NOT stamp a provider_voice_id"


# --------------------------------------------------------------------------- #
# Dispatch compatibility: even if an older workflow stamped a LOCAL voice_ref_id
# but the render node is set to ElevenLabs, the TTS adapter still resolves a
# deterministic provider voice_id instead of falling back to a local engine.
# --------------------------------------------------------------------------- #
def test_provider_voice_id_auto_resolves_when_bank_left_local():
    from nodes._otr_voice_node_common import _resolve_provider_voice_id
    f = {"char_id": "c02", "gender": "female", "voice_ref_id": "vz_caro_davy"}
    m = {"char_id": "c03", "gender": "male", "voice_ref_id": "vz_donor_kditz"}
    pf = _resolve_provider_voice_id("elevenlabs", f, 12345)
    pm = _resolve_provider_voice_id("elevenlabs", m, 12345)
    assert pf and pm, "an elevenlabs char must auto-resolve a provider voice_id"
    assert pf == _resolve_provider_voice_id("elevenlabs", f, 12345), \
        "resolution must be deterministic per character (C7)"
    assert pf != pm, "distinct characters must get distinct voices"


def test_provider_voice_id_none_for_local_engine():
    from nodes._otr_voice_node_common import _resolve_provider_voice_id
    # a LOCAL engine has no provider id -> None; the clone-ref path handles it,
    # and the change is inert for the whole local lane.
    assert _resolve_provider_voice_id(
        "indextts2", {"char_id": "c02", "gender": "female"}, 1) is None
