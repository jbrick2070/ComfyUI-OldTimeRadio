"""Every voice Dia can be cast with carries the words of its sample clip.

Dia clones a voice from a short sample and expects that sample's transcript
in front of the new line, so it knows where the sample ends. The pack's
samples shipped without transcripts, and on 2026-09-29 a published Dia
episode opened its dialogue in gibberish. The same line, voice and seed,
rendered through the real worker on CPU: without the transcript, 19 s of
nonsense; with it, word for word. config/dia_ref_transcripts.json is the
fix, and these tests keep a voice from joining the bank without one.
"""
from __future__ import annotations

import json
import logging
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
TRANSCRIPTS = json.loads((REPO / "config" / "dia_ref_transcripts.json").read_text(encoding="utf-8"))


def _dia_voices():
    bank = json.loads((REPO / "config" / "voice_reference_bank.json").read_text(encoding="utf-8"))
    return [v for v in bank["voices"] if v.get("engine") == "dia"]


def test_every_dia_voice_has_its_sample_transcribed():
    voices = _dia_voices()
    assert voices, "the bank lists no dia voices; this test would prove nothing"
    missing = [(v["voice_ref_id"], pathlib.PurePosixPath(v["ref_path"]).name)
               for v in voices
               if not TRANSCRIPTS.get(pathlib.PurePosixPath(v["ref_path"]).name, "").strip()]
    assert not missing, "dia voices whose sample has no transcript: %s" % missing


def test_the_transcripts_are_plain_text_the_worker_can_prepend():
    for name, text in TRANSCRIPTS.items():
        assert name.endswith(".wav"), name
        assert text == " ".join(text.split()), "%s: stray whitespace" % name
        assert not text.startswith("[S"), "%s: the worker adds the speaker tag" % name


def test_the_adapter_finds_a_samples_transcript_by_its_file_name():
    from nodes._otr_audio_engines.eng_dia import DiaEngine

    engine = DiaEngine()
    path = r"C:\anywhere\models\TTS\refs\indextts2\vz_donor_hannah.wav"
    assert engine._resolve_transcript(path) == TRANSCRIPTS["vz_donor_hannah.wav"]


def test_a_sample_with_no_transcript_warns_once(caplog):
    from nodes._otr_audio_engines.eng_dia import DiaEngine

    engine = DiaEngine()
    with caplog.at_level(logging.WARNING, logger="OTR"):
        assert engine._resolve_transcript("/refs/someone_new.wav") == ""
        assert engine._resolve_transcript("/refs/someone_new.wav") == ""
    warnings = [r for r in caplog.records if "someone_new.wav" in r.getMessage()]
    assert len(warnings) == 1


def test_the_request_to_the_worker_carries_the_transcript(monkeypatch):
    """The helper proves the lookup; this captures the request generate_voice
    actually writes to the worker and reads the sample's words out of it."""
    import io

    from nodes._otr_audio_engines import _otr_sidecar as SC
    from nodes._otr_audio_engines.eng_dia import DiaEngine

    class CapturedWorker:
        def __init__(self):
            self.stdin = io.StringIO()

        def poll(self):
            return None

    engine = DiaEngine()
    engine._proc = CapturedWorker()
    sample = r"C:\refs\vz_bill_boerst.wav"
    monkeypatch.setattr(engine, "load", lambda: None)
    monkeypatch.setattr(engine, "_resolve_ref", lambda ref: ref)
    monkeypatch.setattr(SC, "read_protocol_line", lambda proc, timeout, what: json.dumps(
        {"ok": True, "out_path": "line.wav", "sample_rate": 44100}))
    monkeypatch.setattr(SC, "load_wav_as_audio", lambda path, rate: {"sample_rate": rate})
    engine.generate_voice("Kick it!", sample, None, 7)
    sent = json.loads(engine._proc.stdin.getvalue().splitlines()[0])
    assert sent["text"] == "Kick it!"
    assert sent["ref_clip"] == sample
    assert sent["ref_transcript"] == TRANSCRIPTS["vz_bill_boerst.wav"]
