"""Cloud ElevenLabs CAST MAPPING (2026-07-04).

The elevenlabs adapter fails LOUD ("no provider_voice_id on this cast entry")
unless CastLock stamps the provider voice_id. That needs (a) an elevenlabs voice
POOL in the bank (rows with engine=='elevenlabs' + a real provider_voice_id) and
(b) CastLock resolving the char engine to elevenlabs under the 'elevenlabs_cloud'
bank. These tests guard both, plus the invariant that LOCAL banks never gain a
provider_voice_id (byte-identical local cast).
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OTR_TEST_MODE", "1")

from nodes._otr_voice_bank import load_voice_bank
from nodes.cast_lock import CastLock


def test_bank_has_elevenlabs_pool_with_provider_ids():
    entries, _sha = load_voice_bank()
    el = [e for e in entries if e.engine == "elevenlabs"]
    assert el, "no elevenlabs voice rows in the bank"
    assert all(e.provider_voice_id for e in el), \
        "every elevenlabs row must carry a provider_voice_id"
    genders = {e.gender for e in el}
    assert {"male", "female"} <= genders, \
        "elevenlabs pool must cover male + female for the caster gender match"


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


def test_default_bank_leaves_provider_voice_id_unset():
    cast = [{"char_id": "c02", "name": "MARGOT", "gender": "female"}]
    CastLock()._auto_registry({"meta": {"episode_seed": 1}, "lines": []},
                              cast, "default", False, [])
    assert cast[0].get("provider_voice_id") in (None, ""), \
        "local (default) bank must NOT stamp a provider_voice_id"
