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
``generate_voice`` for every supported model and voice over a set of CASES --
each probe line with a populated delivery vector and with ``None`` (production
passes ``None`` when delivery is off or failed), always with a non-zero seed --
stubbing only the API-key lookup, the shared client's ``post_json`` (no
network) and decode. It captures the path and body each call would POST and
pins one short digest PER CASE, so a failure names the cases that moved.

The model and voice ids are folded out of each request, and a voice that is
rendered the same way on every model is recorded once. Adding a supported
model or voice that behaves like the rest therefore does not trip the pin --
new ids key separately and strand no cached audio -- while a style, a branch
or a payload field that depends on the real voice or model does.

THE PROBES ARE REPRESENTATIVE, NOT EXHAUSTIVE. They are the line shapes OTR's
prepared text actually takes (plain, padded, stage tags, the "Say ..." delivery
prefixes and look-alikes, long, digits and markup, ASCII and typographic
quotes, non-Latin scripts). A branch keyed on a shape no probe has will pass
this test unseen: when you add one, add a probe for it in the same change.

When this fails:
  * The change alters what Google is asked to render for existing ids -> bump
    ``engine_impl_version`` on EVERY google_tts profile in
    ``config/audio_engine_profiles.yaml`` and add the new version's digests
    here (the failure message prints them ready to paste). Old cached lines
    then miss and re-render, which is the point.
  * The change is transport-only and cannot alter the audio (a renamed wire
    field the API now requires, say) -> re-pin the CURRENT version and say why
    in the commit. That edit is visible in review on purpose.
  * The change only adds a NEW model or voice id whose request has its own
    shape (a newly styled voice, a field only a new model takes) -> re-pin
    the CURRENT version and say why. New ids key separately, so no cached
    line goes stale and a version bump would only re-bill the whole cache.
  * You added or renamed a probe -> pin its digest under the current version.

Decode (``_audio_from_response``) is a separate surface and is not pinned here.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import unicodedata
from collections import Counter

import pytest

from nodes import _otr_audio_engines as AE
from nodes._otr_audio_engines import eng_google_tts as G
from nodes._otr_google_api import client as GC

HERE = "tests/" + pathlib.Path(__file__).name

#: engine_impl_version -> {case: digest} of the request surface it shipped with.
PINNED_SURFACE = {
    "1": {
        "plain/delivery": "f07bdc305411c391",
        "plain/no_delivery": "f07bdc305411c391",
        "padded_tag/delivery": "911e79050ce08ff1",
        "padded_tag/no_delivery": "911e79050ce08ff1",
        "say_prefix/delivery": "3c5aca44ad0915e9",
        "say_prefix/no_delivery": "3c5aca44ad0915e9",
        "say_lower/delivery": "c186c729ed2454aa",
        "say_lower/no_delivery": "c186c729ed2454aa",
        "say_padded/delivery": "3c5aca44ad0915e9",
        "say_padded/no_delivery": "3c5aca44ad0915e9",
        "sayonara/delivery": "420509c92485c35a",
        "sayonara/no_delivery": "420509c92485c35a",
        "say_comma/delivery": "dc8073fabbed1776",
        "say_comma/no_delivery": "dc8073fabbed1776",
        "say_midline/delivery": "b943adae71d6289a",
        "say_midline/no_delivery": "b943adae71d6289a",
        "long/delivery": "560566da21d7fbc9",
        "long/no_delivery": "560566da21d7fbc9",
        "digits_markup/delivery": "ddb5ef5b68dcba97",
        "digits_markup/no_delivery": "ddb5ef5b68dcba97",
        "ascii_quote/delivery": "8a140bc7a20e5438",
        "ascii_quote/no_delivery": "8a140bc7a20e5438",
        "typographic/delivery": "8bfc03f618a2cb2c",
        "typographic/no_delivery": "8bfc03f618a2cb2c",
        "non_latin/delivery": "6bf134cc24973142",
        "non_latin/no_delivery": "6bf134cc24973142",
        "stage_tags/delivery": "7b99bcc90f6a00ae",
        "stage_tags/no_delivery": "7b99bcc90f6a00ae",
        "pause_dash_bang/delivery": "6d5fa8168bec77ba",
        "pause_dash_bang/no_delivery": "6d5fa8168bec77ba",
        "short/delivery": "9c2fb441f36df619",
        "short/no_delivery": "9c2fb441f36df619",
        "tag_tail/delivery": "b0c3d2c6b814c6a2",
        "tag_tail/no_delivery": "b0c3d2c6b814c6a2",
        "tag_only/delivery": "82ee728effe996e8",
        "tag_only/no_delivery": "82ee728effe996e8",
        "tag_spaced_comma/delivery": "38f51ccee601ffeb",
        "tag_spaced_comma/no_delivery": "38f51ccee601ffeb",
        "trailing_pause/delivery": "5f15241c73ec3abe",
        "trailing_pause/no_delivery": "5f15241c73ec3abe",
        "leading_pause/delivery": "a2e8ae830dd3e215",
        "leading_pause/no_delivery": "a2e8ae830dd3e215",
        "pause_only/delivery": "2ed545c21022a194",
        "pause_only/no_delivery": "2ed545c21022a194",
        "cut_off/delivery": "dcd4bf4b986b2e90",
        "cut_off/no_delivery": "dcd4bf4b986b2e90",
        "shouting/delivery": "c2bcdba165dfd68a",
        "shouting/no_delivery": "c2bcdba165dfd68a",
        "japanese/delivery": "93bfd506bbf6e248",
        "japanese/no_delivery": "93bfd506bbf6e248",
        "chinese/delivery": "7bd7e5a4b358bcb2",
        "chinese/no_delivery": "7bd7e5a4b358bcb2",
        "hindi/delivery": "f8b14ca5c428d2a2",
        "hindi/no_delivery": "f8b14ca5c428d2a2",
    },
}


def _text(*parts) -> str:
    """A probe from str pieces and code points, so this source stays ASCII."""
    return "".join(chr(p) if isinstance(p, int) else p for p in parts)


#: probe name -> a prepared line as generate_voice receives it.
_PROBES = {
    "plain": "OTR PROBE LINE",
    "padded_tag": "  OTR PROBE LINE, with [sighs] a stage tag.  ",
    "say_prefix": "Say quietly: OTR PROBE LINE",
    "say_lower": "say it plainly: OTR PROBE LINE",
    "say_padded": "  Say quietly: OTR PROBE LINE  ",
    "sayonara": "Sayonara, OTR PROBE LINE.",
    "say_comma": "Say, what's that? OTR PROBE LINE",
    "say_midline": "I would say OTR PROBE LINE",
    "long": ("OTR PROBE LINE, " * 400).strip(),
    "digits_markup": "Act 3, scene 2: a < b & c > d, OTR PROBE LINE",
    "ascii_quote": 'He said "stop," OTR PROBE LINE.',
    "typographic": _text("OTR ", 0x2014, " it", 0x2019, "s a ", 0x201C,
                         "probe", 0x201D, " line"),
    "non_latin": _text("Caf", 0xE9, " ", 0xFC, "ber ",
                       0x416, 0x438, 0x437, 0x43D, 0x44C, " ",
                       0x4F60, 0x597D, " OTR PROBE LINE"),
    "stage_tags": "[whispers] OTR PROBE LINE [laughs] then [gasp].",
    # prepare_text manufactures "..." pauses (_otr_script_prep: any run of
    # dots); dashes, bangs and very short lines pass straight through.
    "pause_dash_bang": "Well... OTR PROBE LINE -- really!",
    "short": "Hm.",
    "tag_tail": "Fine. [sighs]",
    "tag_only": "[laughs]",
    "tag_spaced_comma": "Oh [sighs] , OTR PROBE LINE",
    # Pauses at the edges and alone, and speech cut off with a dash -- all
    # passed through or manufactured by prepare_text.
    "trailing_pause": "I do not know...",
    "leading_pause": "...OTR PROBE LINE",
    "pause_only": "...",
    "cut_off": "I was going to--",
    "shouting": "STOP!!! Now?!",
    # Google TTS is admitted on every episode language (ja, zh, hi among
    # them). Japanese lines carry fullwidth punctuation and ideographic
    # stops; Chinese uses fullwidth commas; Hindi ends on a danda.
    "japanese": _text(0x3053, 0x3093, 0x306B, 0x3061, 0x306F, 0x3001,
                      "OTR PROBE LINE", 0x3002, " ", 0xFF01, 0xFF1F),
    "chinese": _text(0x4F60, 0x597D, 0xFF0C, 0x4E16, 0x754C, 0x3002),
    "hindi": _text(0x0928, 0x092E, 0x0938, 0x094D, 0x0924, 0x0947, 0x0964),
}

#: Production passes a delivery vector, or None when delivery is off or failed
#: (_otr_voice_node_common); a request that came to depend on either must show.
_DELIVERIES = {"delivery": {"angry": 0.9, "calm": 0.1}, "no_delivery": None}
_SEED = 12345

#: case -> (probe text, delivery vector)
_CASES = {f"{p}/{d}": (text, vec)
          for p, text in _PROBES.items() for d, vec in _DELIVERIES.items()}
_PLAIN = "plain/delivery"


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
    """(model, voice, case) -> the folded {path, body} generate_voice POSTs."""
    sent = []

    def _capture(path, payload, **_kw):
        sent.append(_canon({"path": path, "body": payload}))
        return {}

    monkeypatch.setattr(G, "_resolve_api_key", lambda: "otr-test-key")
    monkeypatch.setattr(GC, "post_json", _capture)
    # _post_interaction imports post_json at call time today; if that import
    # ever moves to module scope, stub the module's copy too so no test can
    # reach the network.
    if hasattr(G, "post_json"):
        monkeypatch.setattr(G, "post_json", _capture)
    monkeypatch.setattr(G, "_audio_from_response",
                        lambda response: {"waveform": None, "sample_rate": 24000})
    engine = AE.get_engine("google_tts")
    requests = {}
    for model in G._SUPPORTED_MODELS:
        for voice in G._SUPPORTED_VOICES:
            for case, (text, delivery) in _CASES.items():
                del sent[:]
                engine.generate_voice(
                    text, voice, dict(delivery) if delivery else delivery, _SEED,
                    disable_retry=True, resolved_model=model)
                assert len(sent) == 1, (model, voice, case, len(sent))
                # Folding writes these markers; a request that already carries
                # one would fold to a legitimate request's surface and hide.
                assert "<model>" not in sent[0] and "<voice>" not in sent[0], (
                    "request carries a fold placeholder literally", sent[0])
                requests[(model, voice, case)] = _fold_ids(json.loads(sent[0]), model, voice)
    return requests


def _request_surface(monkeypatch) -> dict:
    """Per case: the majority request, plus every voice (or model|voice) that
    departs from it."""
    requests = _sent_requests(monkeypatch)
    surface = {}
    for case in _CASES:
        blobs = {(m, v): _canon(r) for (m, v, c), r in requests.items() if c == case}
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
        surface[case] = {"default": json.loads(default), "departures": departures}
    return surface


def _digests(surface: dict) -> dict:
    return {case: hashlib.sha256(_canon(row).encode("utf-8")).hexdigest()[:16]
            for case, row in surface.items()}


def _fingerprint(monkeypatch) -> dict:
    return _digests(_request_surface(monkeypatch))


def _leaves(node, path=""):
    """Flatten a request to {dotted.path: value}, so it can be shown or diffed
    field by field instead of truncated as one string."""
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            out.update(_leaves(v, f"{path}.{k}" if path else str(k)))
        return out
    if isinstance(node, list):
        out = {}
        for i, v in enumerate(node):
            out.update(_leaves(v, f"{path}[{i}]"))
        return out
    return {path: node}


def _show(value) -> str:
    """A value, with a long string as its length plus head and tail."""
    if isinstance(value, str) and len(value) > 160:
        return f"<{len(value)} chars> {value[:60]!r} ... {value[-80:]!r}"
    return repr(value)


def _explain(surface: dict, changed) -> str:
    """For each moved case: every field of the default request, and for each
    departing voice/model only the fields that differ from that default."""
    out = []
    for case in changed[:6]:
        row = surface.get(case)
        if row is None:
            out.append(f"  {case}: no longer a case (removed probe?)")
            continue
        base = _leaves(row["default"])
        out.append(f"  {case}: default request")
        out.extend(f"      {k} = {_show(v)}" for k, v in sorted(base.items()))
        for who, req in sorted(row["departures"].items()):
            mine = _leaves(req)
            diff = sorted(k for k in set(base) | set(mine) if base.get(k) != mine.get(k))
            out.append(f"    {who} departs on: " + "; ".join(
                f"{k} = {_show(mine.get(k, '<absent>'))}" for k in diff))
    if len(changed) > 6:
        out.append(f"  ... and {len(changed) - 6} more")
    return "\n".join(out)


def _paste(version: str, digests: dict) -> str:
    rows = "".join(f'        "{c}": "{d}",\n' for c, d in digests.items())
    return f'    "{version}": {{\n{rows}    }},'


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
    surface = _request_surface(monkeypatch)
    actual = _digests(surface)
    assert version in PINNED_SURFACE, (
        f"Google engine_impl_version is {version!r} but nothing is pinned for "
        f"it in PINNED_SURFACE in {HERE}. Add:\n{_paste(version, actual)}")
    pinned = PINNED_SURFACE[version]
    changed = sorted(c for c in set(actual) | set(pinned)
                     if actual.get(c) != pinned.get(c))
    assert not changed, (
        "The Google TTS request generate_voice sends changed while "
        f"engine_impl_version stayed {version!r}; cached audio would replay "
        f"the old take. Cases that moved: {changed}.\n"
        f"See the docstring of {HERE} for which of its four cases applies:\n"
        "  - audio changes for existing ids: bump engine_impl_version on every "
        "google_tts profile in config/audio_engine_profiles.yaml and add the "
        "new version below to PINNED_SURFACE;\n"
        "  - transport-only, a NEW model/voice id with its own shape, or an "
        "added probe: replace the current version's entry and say why in the "
        "commit.\n"
        f"Current request for the moved cases:\n{_explain(surface, changed)}\n"
        f"Digests to paste:\n{_paste(version, actual)}")


def test_the_surface_is_the_real_request_for_every_model_and_voice(monkeypatch):
    """Non-vacuity: every supported pair was driven through generate_voice for
    every case, and the departures are exactly the announcer styles."""
    requests = _sent_requests(monkeypatch)
    assert len(requests) == (
        len(G._SUPPORTED_MODELS) * len(G._SUPPORTED_VOICES) * len(_CASES))
    assert G._ANNOUNCER_STYLE_BY_VOICE, "no styled voice: the pin would guard nothing"
    surface = _request_surface(monkeypatch)
    assert surface[_PLAIN]["default"]["path"] == G._INTERACTIONS_PATH
    departures = surface[_PLAIN]["departures"]
    assert set(departures) == {f"voice:{v}" for v in G._ANNOUNCER_STYLE_BY_VOICE}, departures
    for voice, style in G._ANNOUNCER_STYLE_BY_VOICE.items():
        assert style in departures[f"voice:{voice}"]["body"]["input"], voice
    assert set(PINNED_SURFACE[next(iter(_google_impl_versions().values()))]) == set(_CASES), (
        "every case must be pinned, or a moved case could go unreported")


# --------------------------------------------------------------------------- #
# Mutants: each is a plausible edit to the request path, and each must move
# at least one case. Probes and mutants are NOT one-to-one: several shapes are
# covered jointly (japanese / chinese / non_latin share the normalisation
# mutants; plain / padded / say share the style and guard mutants), so deleting
# a probe is not guaranteed to fail this suite. Review a probe deletion by hand.
# --------------------------------------------------------------------------- #
def _restyle_charon(monkeypatch):
    monkeypatch.setitem(G._ANNOUNCER_STYLE_BY_VOICE, "Charon",
                        "a gravelly American radio announcer style")


def _style_puck(monkeypatch):
    monkeypatch.setitem(G._ANNOUNCER_STYLE_BY_VOICE, "Puck",
                        "a brisk newsreel narrator style")


def _voice_prompt(monkeypatch, guard):
    def _prompt(text, voice):
        style = G._ANNOUNCER_STYLE_BY_VOICE.get(voice)
        if not style or guard(str(text).lower()):
            return text
        return "Say in %s: %s" % (style, text)

    monkeypatch.setattr(G, "_apply_voice_prompt", _prompt)


def _loosened_say_guard(monkeypatch):
    _voice_prompt(monkeypatch, lambda t: t.startswith("say"))


def _say_anywhere_guard(monkeypatch):
    _voice_prompt(monkeypatch, lambda t: "say " in t)


def _puck_say_branch(monkeypatch):
    original = G._apply_voice_prompt

    def _prompt(text, voice):
        if voice == "Puck" and str(text).lower().startswith("say "):
            return text + " [whispers]"
        return original(text, voice)

    monkeypatch.setattr(G, "_apply_voice_prompt", _prompt)


def _payload_field(monkeypatch, when):
    original = G._interaction_payload

    def _payload(model, text, voice):
        body = original(model, text, voice)
        if when(model, text, voice):
            body["generation_config"]["temperature"] = 0.4
        return body

    monkeypatch.setattr(G, "_interaction_payload", _payload)


def _charon_only_temperature(monkeypatch):
    _payload_field(monkeypatch, lambda m, t, v: v == "Charon")


def _one_model_field(monkeypatch):
    _payload_field(monkeypatch, lambda m, t, v: m == "gemini-3.8-flash-tts")


def _long_line_field(monkeypatch):
    _payload_field(monkeypatch, lambda m, t, v: len(t) > 300)


def _digit_field(monkeypatch):
    _payload_field(monkeypatch, lambda m, t, v: any(ch.isdigit() for ch in t))


def _new_endpoint_path(monkeypatch):
    monkeypatch.setattr(G, "_INTERACTIONS_PATH", "/v1/interactions")


def _rewrite_text(monkeypatch, change):
    """An edit inside generate_voice: the line it sends, given its inputs."""
    cls = type(AE.get_engine("google_tts"))
    original = cls.generate_voice

    def generate_voice(self, text, provider_voice_id, delivery_vector, seed, **kw):
        return original(self, change(text, delivery_vector, seed),
                        provider_voice_id, delivery_vector, seed, **kw)

    monkeypatch.setattr(cls, "generate_voice", generate_voice)


def _delivery_none_branch(monkeypatch):
    _rewrite_text(monkeypatch, lambda t, d, s: t + " (neutral)" if d is None else t)


def _delivery_set_branch(monkeypatch):
    _rewrite_text(monkeypatch, lambda t, d, s: t if d is None else "Say warmly: " + t)


def _seed_in_request(monkeypatch):
    _rewrite_text(monkeypatch, lambda t, d, s: f"{t} #{s}" if s else t)


def _truncate_200(monkeypatch):
    _rewrite_text(monkeypatch, lambda t, d, s: t[:200])


def _straighten_quotes(monkeypatch):
    table = {0x2019: "'", 0x201C: '"', 0x201D: '"'}
    _rewrite_text(monkeypatch, lambda t, d, s: t.translate(table))


def _ascii_quote_rewrite(monkeypatch):
    _rewrite_text(monkeypatch, lambda t, d, s: t.replace('"', "'"))


def _fold_accents(monkeypatch):
    def fold(t, d, s):
        return "".join(ch for ch in unicodedata.normalize("NFKD", t)
                       if not unicodedata.combining(ch))

    _rewrite_text(monkeypatch, fold)


def _whispers_only(monkeypatch):
    _rewrite_text(monkeypatch, lambda t, d, s: t.replace("[whispers]", "(whispers)"))


def _nfkc(monkeypatch):
    _rewrite_text(monkeypatch, lambda t, d, s: unicodedata.normalize("NFKC", t))


def _pause_to_comma(monkeypatch):
    _rewrite_text(monkeypatch, lambda t, d, s: t.replace("...", ","))


def _dash_to_hyphen(monkeypatch):
    _rewrite_text(monkeypatch, lambda t, d, s: t.replace("--", "-"))


def _bang_to_period(monkeypatch):
    _rewrite_text(monkeypatch, lambda t, d, s: t.replace("!", "."))


def _pad_short_lines(monkeypatch):
    _rewrite_text(monkeypatch, lambda t, d, s: t + " ..." if len(t.strip()) < 6 else t)


def _drop_trailing_tag(monkeypatch):
    def drop(t, d, s):
        body = t.rstrip()
        if body.endswith("]") and "[" in body and not body.startswith("["):
            return body[:body.rindex("[")].rstrip()
        return t

    _rewrite_text(monkeypatch, drop)


def _close_spaced_comma(monkeypatch):
    _rewrite_text(monkeypatch, lambda t, d, s: t.replace(" ,", ","))


def _tag_only_to_parens(monkeypatch):
    def rewrite(t, d, s):
        body = t.strip()
        if body.startswith("[") and body.endswith("]") and body.count("[") == 1:
            return "(" + body[1:-1] + ")"
        return t

    _rewrite_text(monkeypatch, rewrite)


def _trailing_pause_to_period(monkeypatch):
    # A pause that ENDS speech, after a word -- not the pause-only line.
    _rewrite_text(monkeypatch, lambda t, d, s: re.sub(r"(\w)\.\.\.\s*$", r"\1.", t))


def _fullwidth_comma(monkeypatch):
    _rewrite_text(monkeypatch, lambda t, d, s: t.replace(chr(0xFF0C), ","))


def _strip_leading_pause(monkeypatch):
    _rewrite_text(monkeypatch, lambda t, d, s: t.lstrip(".") or t)


def _punctuation_only_line(monkeypatch):
    _rewrite_text(monkeypatch,
                  lambda t, d, s: t if any(c.isalnum() for c in t) else "(pause)")


def _cut_off_to_pause(monkeypatch):
    _rewrite_text(monkeypatch, lambda t, d, s: re.sub(r"--\s*$", "...", t))


def _collapse_bangs(monkeypatch):
    _rewrite_text(monkeypatch, lambda t, d, s: re.sub(r"!{2,}", "!", t))


def _cjk_stops(monkeypatch):
    table = {0x3002: ". ", 0x3001: ", ", 0xFF0C: ", "}
    _rewrite_text(monkeypatch, lambda t, d, s: t.translate(table))


def _danda_to_period(monkeypatch):
    _rewrite_text(monkeypatch, lambda t, d, s: t.replace(chr(0x0964), "."))


_MUTANTS = [
    _restyle_charon, _style_puck, _loosened_say_guard, _say_anywhere_guard,
    _puck_say_branch, _charon_only_temperature, _one_model_field,
    _long_line_field, _digit_field, _new_endpoint_path, _delivery_none_branch,
    _delivery_set_branch, _seed_in_request, _truncate_200, _straighten_quotes,
    _ascii_quote_rewrite, _fold_accents, _whispers_only, _nfkc,
    _pause_to_comma, _dash_to_hyphen, _bang_to_period, _pad_short_lines,
    _drop_trailing_tag, _close_spaced_comma, _tag_only_to_parens,
    _trailing_pause_to_period, _strip_leading_pause, _punctuation_only_line,
    _cut_off_to_pause, _collapse_bangs, _cjk_stops, _danda_to_period,
    _fullwidth_comma,
]


@pytest.mark.parametrize("mutate", _MUTANTS, ids=lambda f: f.__name__.lstrip("_"))
def test_a_request_change_moves_the_fingerprint(monkeypatch, mutate):
    before = _fingerprint(monkeypatch)
    mutate(monkeypatch)
    after = _fingerprint(monkeypatch)
    assert [c for c in before if after[c] != before[c]], "no case moved"


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
