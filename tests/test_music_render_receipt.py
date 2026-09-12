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
        wave[0, 0, 3] = 1.0  # one full-scale sample the CEILING must take down
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
        # The bus ceiling ran before the wav writer, so the full-scale sample
        # is at -1 dBFS and NOTHING is clipped. Before 2026-09-12 this read
        # peak 0.0 dBFS with one clipped sample, and on 14% of real cues it
        # read thousands.
        assert receipt["clipped_samples"] == 0
        assert receipt["peak_dbfs"] == pytest.approx(-1.0, abs=0.05)
        # the ROW text (identity) is untouched by the engine prompt
        assert row["prompt"].startswith("tense, moonlit")
        assert not row["prompt"].startswith(P.EARLY_CONSORT.instruments)


def test_the_music_bus_has_a_ceiling_and_leaves_a_quiet_cue_alone():
    """MEASURED 2026-09-12: 280 of 1,984 cue wavs on the dev box carried more
    than four hard-clipped samples (worst 4,710), because ComfyUI pins a
    Stable Audio render's RMS but not its peak, only `eng_musicgen`
    normalises, and `_write_cue_wav` casts to int16 with a hard clip."""
    from nodes.stable_audio_theme import StableAudioTheme as _Theme

    hot = {"waveform": torch.full((1, 2, 4096), 3.0), "sample_rate": 44100}
    limited, info = _Theme._ceiling_the_cue(hot, "hot")
    assert info["engaged"] is True
    assert float(limited["waveform"].abs().max()) <= 10.0 ** (-1.0 / 20.0) + 1e-6
    assert limited["sample_rate"] == 44100

    quiet = {"waveform": torch.full((1, 2, 4096), 0.2), "sample_rate": 44100}
    same, info = _Theme._ceiling_the_cue(quiet, "quiet")
    assert info["engaged"] is False
    assert torch.equal(same["waveform"], quiet["waveform"]), (
        "a cue under the ceiling is untouched -- this is a ceiling, not a "
        "normalisation, so the music never moves against the dialogue")

    # a ceiling is worth having and never worth an episode
    broken, info = _Theme._ceiling_the_cue({"sample_rate": 44100}, "broken")
    assert info["engaged"] is False and broken == {"sample_rate": 44100}


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
