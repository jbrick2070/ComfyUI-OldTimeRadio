"""The music receipt reaches the ledger (build plan r2, M7, 2026-09-11).

A listening verdict ("the music needs to be improved") must be traceable to
what the engine actually heard and did: the composed engine prompt, the
negative prompt, the palette, the engine's own parameters, and the peak facts
of what came back. The theme node writes `render_receipt` on every manifest
row; `_music_row_from_manifest` and `reconcile_ledger_music` carry it into
`ledger.music[]` on create AND on an identity match; `Ledger.set_music`
projects it like the other render-owned fields. Opus's refutation found both
projections silently dropping every unlisted key -- this file is the pin.
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from nodes import _otr_cue_manifest as CM  # noqa: E402
from nodes import _otr_music_palette as P  # noqa: E402


def _stub(monkeypatch, recorder):
    from nodes._otr_audio_engines import get_engine
    engine = get_engine("musicgen")

    def _gen(prompt, duration_s, seed, **kwargs):
        recorder.append({"prompt": prompt, "seed": int(seed), **kwargs})
        wave = torch.full((1, 1, 64), 0.5, dtype=torch.float32)
        wave[0, 0, 3] = 1.0  # one full-scale sample the receipt must count
        return {"waveform": wave, "sample_rate": 32000,
                "receipt": {"engine": "stub", "seed": int(seed), "cfg": 7.0}}
    monkeypatch.setattr(engine, "generate_clip", _gen)


def _script(meta):
    return json.dumps({"schema_version": "l3-2026-05-14", "cast": [],
                       "lines": [], "meta": meta})


_META = {"source_bank": "shakespeare", "source_meta": {"year": "c. 1595"},
         "music_mood_terms": ["tense", "moonlit"]}


def _render(monkeypatch):
    from nodes.stable_audio_theme import StableAudioTheme
    calls = []
    _stub(monkeypatch, calls)
    out = StableAudioTheme().generate(script_json=_script(_META), engine="musicgen")
    # the legacy lane synthesizes the two fixed slots (opening, closing)
    return calls, CM.parse_manifest(out[1], batch_size=2)


def test_every_cue_row_carries_what_the_engine_heard_and_did(monkeypatch):
    calls, manifest = _render(monkeypatch)
    assert [r["cue_id"] for r in manifest["cues"]] == ["opening", "closing"]
    for row in manifest["cues"]:
        receipt = row["render_receipt"]
        heard = calls[row["batch_index"]]
        assert receipt["engine_prompt"] == heard["prompt"]
        assert receipt["engine_prompt"].startswith(P.EARLY_CONSORT.instruments)
        assert receipt["negative_prompt"] == heard["negative_prompt"]
        assert "hiss" in receipt["negative_prompt"]
        assert receipt["palette_key"] == "early_consort"
        assert receipt["params"] == {"engine": "stub", "seed": heard["seed"], "cfg": 7.0}
        assert receipt["sample_rate"] == 32000
        assert receipt["clipped_samples"] == 1
        assert receipt["peak_dbfs"] == 0.0
        # the ROW text (identity) is untouched by the engine prompt
        assert row["prompt"].startswith("tense, moonlit")
        assert not row["prompt"].startswith(P.EARLY_CONSORT.instruments)


def test_the_receipt_reaches_the_ledger_on_create_and_on_an_identity_match(monkeypatch):
    _, manifest = _render(monkeypatch)
    ledger = {"music": []}
    CM.reconcile_ledger_music(ledger, manifest)  # legacy lane: rows materialize
    assert [r["cue_id"] for r in ledger["music"]] == [
        r["cue_id"] for r in manifest["cues"]]
    for row in ledger["music"]:
        assert isinstance(row["render_receipt"], dict)
        assert row["render_receipt"]["palette_key"] == "early_consort"
    # a re-render of the SAME cue spec matches on identity and refreshes it
    for row in manifest["cues"]:
        row["render_receipt"] = dict(row["render_receipt"],
                                     params={"engine": "stub", "cfg": 9.0})
    stats = CM.reconcile_ledger_music(ledger, manifest)
    assert stats["matched"] == 2 and stats["created"] == 0
    assert all(r["render_receipt"]["params"]["cfg"] == 9.0 for r in ledger["music"])
    # and the cue-spec identity never moved: the receipt is outside it
    from nodes.production_ledger import music_cue_spec_sha256
    for row in ledger["music"]:
        assert row["cue_spec_sha256"] == music_cue_spec_sha256(row)


def test_set_music_projects_the_receipt_like_the_other_render_fields():
    from nodes.production_ledger import Ledger, music_cue_spec_sha256
    ledger = Ledger.__new__(Ledger)
    ledger.data = {}
    row = {"cue_id": "opening", "description": "d", "generation_prompt": "g",
           "placement": "opening", "target_duration_s": 12.0,
           "wav_path": "x.wav", "render_receipt": {"palette_key": "k", "params": {}}}
    ledger.set_music([row, dict(row, cue_id="closing", render_receipt="junk")])
    music = ledger.data["music"]
    assert music[0]["render_receipt"] == {"palette_key": "k", "params": {}}
    assert music[1]["render_receipt"] is None, "a non-dict receipt is dropped, not carried"
    assert music[0]["cue_spec_sha256"] == music_cue_spec_sha256(music[0])
    assert music[0]["cue_spec_sha256"] == music_cue_spec_sha256(
        {k: v for k, v in music[0].items() if k != "render_receipt"}), \
        "the receipt is outside the cue-spec identity"
