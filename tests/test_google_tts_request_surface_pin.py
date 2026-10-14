"""Google TTS: what is sent AFTER the cache key is cut is pinned to
``engine_impl_version``.

The audio cache key hashes ``prepared_text`` (``prepare_text``, which already
carries the delivery prefix), the model, the voice and ``engine_impl_version``.
``generate_voice`` then shapes the provider request further, outside that key:
it strips the line, ``_apply_voice_prompt`` wraps the announcer voices in a
style instruction, ``_interaction_payload`` builds the body, and
``_post_interaction`` picks the endpoint path. Change any of that and Google
renders different audio for a line whose key has not moved, so a cached sidecar
serves the old take as if this build had made it.

``engine_impl_version`` is the profile's declared "the engine's behaviour
changed" statement (see ``test_voice_identity_fix``), and it keys the cache.
This test makes that statement impossible to forget. It drives the REAL
``generate_voice`` for every supported model and voice, stubbing only the
shared client's ``post_json`` (no network) and decode, captures the path and
body it would POST, and fingerprints them. Each impl version owns exactly one
fingerprint.

The model and voice ids are folded out of each request, and a voice that is
rendered the same way on every model is recorded once. Adding a supported
model or voice that behaves like the rest therefore does not trip the pin --
new ids key separately and strand no cached audio -- while a style, a branch
or a payload field that depends on the real voice or model does.

When this fails:
  * The change alters what Google is asked to render -> bump
    ``engine_impl_version`` on EVERY google_tts profile in
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
from collections import Counter

import pytest

from nodes import _otr_audio_engines as AE
from nodes._otr_audio_engines import eng_google_tts as G
from nodes._otr_google_api import client as GC

#: engine_impl_version -> fingerprint of the request surface it shipped with.
PINNED_SURFACE = {
    "1": "e61a9d68500d519a5ac4539b36d4aaa1265bc94d909752c6cc3026484556cf4d",
}

#: Prepared lines as generate_voice receives them: plain, padded with a stage
#: tag, the "Say ..." shapes a delivery prefix produces (cased, padded), and a
#: line that merely starts with the letters "say".
_PROBES = (
    "OTR PROBE LINE",
    "  OTR PROBE LINE, with [sighs] a stage tag.  ",
    "Say quietly: OTR PROBE LINE",
    "say it plainly: OTR PROBE LINE",
    "  Say quietly: OTR PROBE LINE  ",
    "Sayonara, OTR PROBE LINE.",
)
_PLAIN = _PROBES[0]


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _fold_ids(node, model: str, voice: str):
    """Replace values EQUAL to the model or voice id with a placeholder."""
    if isinstance(node, dict):
        return {k: _fold_ids(v, model, voice) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [_fold_ids(v, model, voice) for v in node]
    if node == model:
        return "<model>"
    if node == voice:
        return "<voice>"
    return node


def _sent_requests(monkeypatch) -> dict:
    """(model, voice, probe) -> the folded {path, body} generate_voice POSTs."""
    sent = []

    def _capture(path, payload, **_kw):
        sent.append(_canon({"path": path, "body": payload}))
        return {}

    monkeypatch.setattr(G, "_resolve_api_key", lambda: "otr-test-key")
    monkeypatch.setattr(GC, "post_json", _capture)
    monkeypatch.setattr(G, "_audio_from_response",
                        lambda response: {"waveform": None, "sample_rate": 24000})
    engine = AE.get_engine("google_tts")
    requests = {}
    for model in G._SUPPORTED_MODELS:
        for voice in G._SUPPORTED_VOICES:
            for probe in _PROBES:
                del sent[:]
                engine.generate_voice(probe, voice, None, 0,
                                      disable_retry=True, resolved_model=model)
                assert len(sent) == 1, (model, voice, probe, len(sent))
                # Folding writes these markers; a request that already carries
                # one would fold to a legitimate request's surface and hide.
                assert "<model>" not in sent[0] and "<voice>" not in sent[0], (
                    "request carries a fold placeholder literally", sent[0])
                requests[(model, voice, probe)] = _fold_ids(json.loads(sent[0]), model, voice)
    return requests


def _request_surface(monkeypatch) -> dict:
    """Per probe: the majority request, plus every voice (or model|voice) that
    departs from it."""
    requests = _sent_requests(monkeypatch)
    surface = {}
    for probe in _PROBES:
        blobs = {(m, v): _canon(r) for (m, v, p), r in requests.items() if p == probe}
        counts = Counter(blobs.values())
        default = min(counts, key=lambda blob: (-counts[blob], blob))
        departures = {}
        for voice in G._SUPPORTED_VOICES:
            per_model = {m: blobs[(m, voice)] for m in G._SUPPORTED_MODELS}
            if len(set(per_model.values())) == 1:
                only = next(iter(per_model.values()))
                if only != default:
                    departures[f"voice:{voice}"] = json.loads(only)
            else:
                for model, blob in per_model.items():
                    if blob != default:
                        departures[f"{model}|{voice}"] = json.loads(blob)
        surface[probe] = {"default": json.loads(default), "departures": departures}
    return surface


def _fingerprint(monkeypatch) -> str:
    return hashlib.sha256(_canon(_request_surface(monkeypatch)).encode("utf-8")).hexdigest()


def _google_impl_versions() -> dict:
    """Every google_tts profile, discovered -- a third one is covered too."""
    from nodes._otr_engine_profiles import load_resolver

    resolver = load_resolver()
    versions = {
        pid: str(resolver.get(pid).engine_impl_version)
        for pid in resolver.profile_ids()
        if resolver.get(pid).engine == "google_tts"
    }
    assert {"char_google_tts_v1", "announcer_google_tts_v1"} <= set(versions), versions
    return versions


def test_every_google_profile_shares_one_impl_version():
    """One adapter serves every row, so a surface change moves all or none."""
    versions = _google_impl_versions()
    assert len(set(versions.values())) == 1, versions


def test_request_surface_matches_the_pin_for_its_impl_version(monkeypatch):
    version = next(iter(_google_impl_versions().values()))
    actual = _fingerprint(monkeypatch)
    assert version in PINNED_SURFACE, (
        f"Google engine_impl_version is {version!r} but no request-surface "
        f"fingerprint is pinned for it. Add {version!r}: {actual!r} "
        f"to PINNED_SURFACE.")
    assert actual == PINNED_SURFACE[version], (
        "The Google TTS request generate_voice sends changed while "
        f"engine_impl_version stayed {version!r}. Cached audio would replay "
        "the old take. Bump engine_impl_version on every google_tts profile "
        "in config/audio_engine_profiles.yaml and pin the new fingerprint "
        f"{actual!r}, or, for a transport-only change, re-pin the current "
        "version and say why in the commit.")


def test_the_surface_is_the_real_request_for_every_model_and_voice(monkeypatch):
    """Non-vacuity: every supported pair was driven through generate_voice, and
    the departures are exactly the announcer styles, model-independent."""
    requests = _sent_requests(monkeypatch)
    assert len(requests) == (
        len(G._SUPPORTED_MODELS) * len(G._SUPPORTED_VOICES) * len(_PROBES))
    assert G._ANNOUNCER_STYLE_BY_VOICE, "no styled voice: the pin would guard nothing"
    surface = _request_surface(monkeypatch)
    assert surface[_PLAIN]["default"]["path"] == G._INTERACTIONS_PATH
    departures = surface[_PLAIN]["departures"]
    assert set(departures) == {f"voice:{v}" for v in G._ANNOUNCER_STYLE_BY_VOICE}, departures
    for voice, style in G._ANNOUNCER_STYLE_BY_VOICE.items():
        assert style in departures[f"voice:{voice}"]["body"]["input"], voice


def _restyle_charon(monkeypatch):
    monkeypatch.setitem(G._ANNOUNCER_STYLE_BY_VOICE, "Charon",
                        "a gravelly American radio announcer style")


def _style_puck(monkeypatch):
    monkeypatch.setitem(G._ANNOUNCER_STYLE_BY_VOICE, "Puck",
                        "a brisk newsreel narrator style")


def _loosened_say_guard(monkeypatch):
    def _prompt(text, voice):
        style = G._ANNOUNCER_STYLE_BY_VOICE.get(voice)
        if not style or str(text).lower().startswith("say"):
            return text
        return "Say in %s: %s" % (style, text)

    monkeypatch.setattr(G, "_apply_voice_prompt", _prompt)


def _puck_say_branch(monkeypatch):
    original = G._apply_voice_prompt

    def _prompt(text, voice):
        if voice == "Puck" and str(text).lower().startswith("say "):
            return text + " [whispers]"
        return original(text, voice)

    monkeypatch.setattr(G, "_apply_voice_prompt", _prompt)


def _charon_only_temperature(monkeypatch):
    original = G._interaction_payload

    def _payload(model, text, voice):
        body = original(model, text, voice)
        if voice == "Charon":
            body["generation_config"]["temperature"] = 0.4
        return body

    monkeypatch.setattr(G, "_interaction_payload", _payload)


def _one_model_field(monkeypatch):
    original = G._interaction_payload

    def _payload(model, text, voice):
        body = original(model, text, voice)
        if model == "gemini-3.8-flash-tts":
            body["generation_config"]["seed"] = 7
        return body

    monkeypatch.setattr(G, "_interaction_payload", _payload)


def _new_endpoint_path(monkeypatch):
    monkeypatch.setattr(G, "_INTERACTIONS_PATH", "/v1/interactions")


@pytest.mark.parametrize("mutate", [
    _restyle_charon, _style_puck, _loosened_say_guard, _puck_say_branch,
    _charon_only_temperature, _one_model_field, _new_endpoint_path,
], ids=lambda f: f.__name__.lstrip("_"))
def test_a_request_change_moves_the_fingerprint(monkeypatch, mutate):
    before = _fingerprint(monkeypatch)
    mutate(monkeypatch)
    assert _fingerprint(monkeypatch) != before


def test_a_literal_placeholder_in_the_request_is_refused_not_folded(monkeypatch):
    """Sending "<model>" in place of the real id is a request change that
    folding alone would erase (codex QA on f1aeb2ee)."""
    original = G._interaction_payload

    def _payload(model, text, voice):
        body = original(model, text, voice)
        body["model"] = "<model>"
        return body

    monkeypatch.setattr(G, "_interaction_payload", _payload)
    with pytest.raises(AssertionError, match="fold placeholder"):
        _fingerprint(monkeypatch)


@pytest.mark.parametrize("attr,extra", [
    ("_SUPPORTED_VOICES", "Otrprobevoice"),
    ("_SUPPORTED_MODELS", "otr-probe-tts-model"),
])
def test_adding_an_ordinary_model_or_voice_keeps_the_pin(monkeypatch, attr, extra):
    """New ids key separately, so they strand no cached audio: no false alarm."""
    before = _fingerprint(monkeypatch)
    monkeypatch.setattr(G, attr, getattr(G, attr) + (extra,))
    assert _fingerprint(monkeypatch) == before
