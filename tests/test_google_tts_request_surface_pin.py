"""Google TTS: what is sent AFTER the cache key is cut is pinned to
``engine_impl_version``.

The audio cache key hashes ``prepared_text`` (``prepare_text``, which already
carries the delivery prefix), the model, the voice and ``engine_impl_version``.
``generate_voice`` then shapes the provider request further, outside that key:
``_apply_voice_prompt`` wraps the announcer voices in a style instruction, and
``_interaction_payload`` builds the request body. Reword either and Google
renders different audio for a line whose key has not moved, so a cached sidecar
serves the old take as if this build had made it.

``engine_impl_version`` is the profile's declared "the engine's behaviour
changed" statement (see ``test_voice_identity_fix``), and it keys the cache.
This test makes that statement impossible to forget: the request surface is
fingerprinted by BEHAVIOUR (what the two functions return, not their source),
and each impl version owns exactly one fingerprint.

When this fails:
  * The change alters what Google is asked to render -> bump
    ``engine_impl_version`` on BOTH Google profiles in
    ``config/audio_engine_profiles.yaml`` and add the new fingerprint under the
    new version here. Old cached lines then miss and re-render, which is the
    point.
  * The change is transport-only and cannot alter the audio (a renamed wire
    field the API now requires, say) -> re-pin the CURRENT version's
    fingerprint and say why in the commit. That edit is visible in review on
    purpose.

Decode (``_audio_from_response``) is a separate surface and is not pinned here.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from nodes._otr_audio_engines import eng_google_tts as G

GOOGLE_PROFILES = ("char_google_tts_v1", "announcer_google_tts_v1")

#: engine_impl_version -> fingerprint of the request surface it shipped with.
PINNED_SURFACE = {
    "1": "e0192f027cf6637c3ed401b009fd5ba1e0d22932cedb9d6f198583657e71b26a",
}

_PROBE = "OTR PROBE LINE"
_SAY_PROBE = "Say it plainly: OTR PROBE LINE"


def _request_surface() -> dict:
    """Everything ``generate_voice`` adds to the request after the key is cut.

    Only voices the prompt actually changes are recorded, so adding a new
    supported voice WITHOUT a style does not trip the pin (its lines key on the
    new voice id anyway), while styling any voice -- new or existing -- does.
    """
    styled = {}
    for voice in G._SUPPORTED_VOICES:
        spoken = G._apply_voice_prompt(_PROBE, voice)
        if spoken != _PROBE:
            styled[voice] = {
                "plain": spoken,
                "say_prefixed": G._apply_voice_prompt(_SAY_PROBE, voice),
            }
    return {
        "voice_prompt": styled,
        "payload": G._interaction_payload("<model>", "<text>", "<voice>"),
    }


def _fingerprint() -> str:
    blob = json.dumps(_request_surface(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _google_impl_versions() -> dict:
    from nodes._otr_engine_profiles import load_resolver

    resolver = load_resolver()
    return {pid: str(resolver.get(pid).engine_impl_version) for pid in GOOGLE_PROFILES}


def test_both_google_profiles_share_one_impl_version():
    """One adapter serves both rows, so a surface change moves both or neither."""
    versions = _google_impl_versions()
    assert len(set(versions.values())) == 1, versions


def test_request_surface_matches_the_pin_for_its_impl_version():
    version = next(iter(_google_impl_versions().values()))
    assert version in PINNED_SURFACE, (
        f"Google engine_impl_version is {version!r} but no request-surface "
        f"fingerprint is pinned for it. Add {version!r}: {_fingerprint()!r} "
        f"to PINNED_SURFACE.")
    assert _fingerprint() == PINNED_SURFACE[version], (
        "The Google TTS request surface (_apply_voice_prompt / "
        "_interaction_payload) changed while engine_impl_version stayed "
        f"{version!r}. Cached audio would replay the old take. Bump "
        "engine_impl_version on both Google profiles in "
        "config/audio_engine_profiles.yaml and pin the new fingerprint "
        f"{_fingerprint()!r}, or, for a transport-only change, re-pin the "
        "current version and say why in the commit.")


def test_the_pin_covers_the_announcer_styles():
    """The surface must actually contain the styled voices, or the pin guards
    nothing."""
    styled = _request_surface()["voice_prompt"]
    assert set(styled) == set(G._ANNOUNCER_STYLE_BY_VOICE), styled
    for voice, style in G._ANNOUNCER_STYLE_BY_VOICE.items():
        assert style in styled[voice]["plain"], voice


@pytest.mark.parametrize("voice,style", [
    ("Charon", "a gravelly American radio announcer style"),   # reworded
    ("Puck", "a brisk newsreel narrator style"),               # newly styled
])
def test_a_style_change_moves_the_fingerprint(monkeypatch, voice, style):
    before = _fingerprint()
    monkeypatch.setitem(G._ANNOUNCER_STYLE_BY_VOICE, voice, style)
    assert _fingerprint() != before


def test_a_payload_change_moves_the_fingerprint(monkeypatch):
    before = _fingerprint()
    original = G._interaction_payload

    def _with_temperature(model, text, voice):
        payload = original(model, text, voice)
        payload["generation_config"]["temperature"] = 0.4
        return payload

    monkeypatch.setattr(G, "_interaction_payload", _with_temperature)
    assert _fingerprint() != before
