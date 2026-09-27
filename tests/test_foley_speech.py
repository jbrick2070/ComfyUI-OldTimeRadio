"""Speech ducking: mocked models, real PCM/ledger boundaries, no GPU access."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nodes import _otr_ledger as ledger
from nodes import otr_master_audio_mux as mux
from nodes._otr_video_engines import foley_speech as speech
from nodes._otr_video_engines import foley_stems as fs

RATE, FPS, STEP = 48000, 25, 1920
FOLEY, MIME = "ltx25_foley_16gb", "ltx25_mime_16gb"


def _row(beat_id="b1", **kwargs):
    return dict(beat_id=beat_id, foley_path=beat_id + ".wav", engine_id=FOLEY,
                start_s=0.0, frame_count=2, **kwargs)


def _stub_models(monkeypatch, *, positive=None, texts=None):
    monkeypatch.setenv("OTR_TEST_MODE", "0")
    events = []
    monkeypatch.setattr(speech, "_load_vad", lambda: (object(), object()))
    monkeypatch.setattr(speech, "_vad_positive",
                        lambda path, *_: positive is None or path in positive)
    model = SimpleNamespace(model=SimpleNamespace(unload_model=lambda: events.append("release")))
    monkeypatch.setattr(speech, "_load_whisper", lambda: events.append("whisper") or model)

    def transcribe(path, model):
        events.append(path)
        value = (texts or {}).get(path, "words")
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(speech, "_transcribe", transcribe)
    return events


def _stub_judge_slot(monkeypatch, answer):
    from nodes import otr_shot_lock, _otr_model_loader
    calls, releases = [], []

    def generate(messages, **kwargs):
        calls.append((messages, kwargs))
        if isinstance(answer, Exception):
            raise answer
        return answer if isinstance(answer, str) else json.dumps(answer)

    monkeypatch.setattr(otr_shot_lock, "_resolve_writer_llm_binding",
                        lambda meta, warnings: (generate, "episode-model"))
    monkeypatch.setattr(_otr_model_loader, "unload_llm_if_local_resident",
                        lambda: releases.append(True))
    return calls, releases


@pytest.mark.parametrize("cuda_encode_works", [True, False])
def test_whisper_proves_cuda_with_a_real_encode_else_uses_the_cpu(monkeypatch, cuda_encode_works):
    """ctranslate2 4.x loads cuBLAS 12 only at the FIRST encode, so on a CUDA 13
    torch stack a CUDA device 'exists' and every transcription raised (the
    5080 and the pod, 2026-09-27) -- the duck silently ducked nothing. The
    loader proves CUDA with one real encode and falls back to the CPU."""
    import ctranslate2
    import faster_whisper
    import huggingface_hub
    built = []

    class FakeWhisper:
        def __init__(self, path, *, device, compute_type, **kw):
            self.device = device
            built.append((device, compute_type))

        def transcribe(self, audio, **kw):
            if self.device == "cuda" and not cuda_encode_works:
                raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
            return iter([]), None

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: "unused")
    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeWhisper)
    monkeypatch.setattr(ctranslate2, "get_cuda_device_count", lambda: 1)
    monkeypatch.setattr(ctranslate2, "get_supported_compute_types", lambda d: {"float16", "int8"})
    monkeypatch.setattr(speech, "_models_dir", lambda: Path("unused"))
    model = speech._load_whisper()
    if cuda_encode_works:
        assert model.device == "cuda" and built == [("cuda", "float16")]
    else:
        assert model.device == "cpu" and built == [("cuda", "float16"), ("cpu", "int8")]


def test_vad_negative_never_loads_whisper_or_llm(monkeypatch):
    events = _stub_models(monkeypatch, positive=set())
    monkeypatch.setattr(speech, "_judge_transcripts", lambda *_: pytest.fail("LLM called"))
    result = speech.detect_foley_speech([_row()], {})
    assert result["b1"] == dict(vad=False, transcript="", verdict=False,
                                reason="VAD found no speech", ducked=False)
    assert events == []


def test_empty_vad_positive_ducks_without_llm(monkeypatch):
    events = _stub_models(monkeypatch, texts={"b1.wav": ""})
    monkeypatch.setattr(speech, "_judge_transcripts", lambda *_: pytest.fail("LLM called"))
    result = speech.detect_foley_speech([_row()], {})
    assert result["b1"]["ducked"] is True
    assert result["b1"]["vad"] is True
    assert events == ["whisper", "b1.wav", "release"]


@pytest.mark.parametrize("text", ["[music]", "[applause]", "a door creaks"])
def test_sound_descriptions_reach_judge_and_never_become_empty_babble(monkeypatch, text):
    _stub_models(monkeypatch, texts={"b1.wav": text})
    calls, _ = _stub_judge_slot(monkeypatch, {
        "b1": {"speech": False, "reason": "Sound description, not a human utterance"}})
    result = speech.detect_foley_speech([_row()], {})
    assert result["b1"]["transcript"] == text and result["b1"]["ducked"] is False
    assert json.loads(calls[0][0][-1]["content"]) == {"b1": text}
    # The judge is told sound labels are not words (the operator's rule,
    # 2026-09-26: any human words duck, and unsure means duck).
    system = calls[0][0][0]["content"]
    assert "sound labels" in system and "are NOT words" in system
    assert "When you are unsure, answer true" in system


def test_missing_whisper_weights_is_no_duck_with_receipt(monkeypatch):
    _stub_models(monkeypatch)

    def unavailable():
        raise FileNotFoundError("Whisper base is not cached and hub is offline")

    monkeypatch.setattr(speech, "_load_whisper", unavailable)
    result = speech.detect_foley_speech([_row()], {})
    assert result["b1"]["ducked"] is False
    assert "not cached" in result["b1"]["reason"]


def test_one_batched_strict_verdict_after_whisper_release(monkeypatch):
    events = _stub_models(monkeypatch, positive={"b1.wav", "b2.wav"},
                          texts={"b1.wav": "Get down!", "b2.wav": "A creak"})
    calls, releases = _stub_judge_slot(monkeypatch, {
        "b1": {"speech": True, "reason": "Human words"},
        "b2": {"speech": False, "reason": "Environmental noise"},
    })
    judge = speech._judge_transcripts

    def check_order(transcripts, meta):
        assert events[-1] == "release"
        return judge(transcripts, meta)

    monkeypatch.setattr(speech, "_judge_transcripts", check_order)
    rows = [_row("b1"), _row("b2"), _row("b3")]
    result = speech.detect_foley_speech(rows, {"technical_model": "episode-model"})
    assert [result[key]["ducked"] for key in ("b1", "b2", "b3")] == [True, False, False]
    assert len(calls) == len(releases) == 1
    payload = json.loads(calls[0][0][-1]["content"])
    assert payload == {"b1": "Get down!", "b2": "A creak"}
    assert all("foley_speech_duck" not in row for row in rows)


@pytest.mark.parametrize("answer", [
    "not JSON", {}, {"wrong": {"speech": True, "reason": "yes"}},
    {"b2": {"speech": "false", "reason": "no"}},
    {"b2": {"speech": 1, "reason": "yes"}},
    {"b2": {"speech": True}},
    {"b2": {"speech": True, "reason": "yes", "extra": 3}},
    {"b2": {"speech": True, "reason": ""}}, RuntimeError("slot failed"),
])
def test_bad_batch_rolls_back_even_wordless_ducks(monkeypatch, answer):
    _stub_models(monkeypatch, texts={"b1.wav": "", "b2.wav": "Help"})
    calls, releases = _stub_judge_slot(monkeypatch, answer)
    result = speech.detect_foley_speech([_row("b1"), _row("b2")], {})
    assert all(not item["ducked"] for item in result.values())
    assert all("LLM failed" in item["reason"] for item in result.values())
    assert len(calls) == len(releases) == 1


@pytest.mark.parametrize("stage", ["VAD", "Whisper", "transcribe", "unload"])
def test_stage_exception_never_leaves_partial_ducks(monkeypatch, stage):
    events = _stub_models(monkeypatch, texts={"b1.wav": "", "b2.wav": "words"})

    def fail(*args):
        raise RuntimeError("injected failure")

    if stage == "VAD":
        monkeypatch.setattr(speech, "_vad_positive", fail)
    elif stage == "Whisper":
        monkeypatch.setattr(speech, "_load_whisper", fail)
    elif stage == "unload":
        monkeypatch.setattr(speech, "_release_whisper", fail)
    else:
        original = speech._transcribe
        monkeypatch.setattr(speech, "_transcribe",
                            lambda path, m: fail() if path == "b2.wav" else original(path, m))
    result = speech.detect_foley_speech([_row("b1"), _row("b2")], {})
    assert all(item["ducked"] is False and "failed" in item["reason"]
               for item in result.values())
    if stage == "transcribe":
        assert events[-1] == "release"


@pytest.mark.parametrize("lane", [MIME, "ltx25_mime_24gb", "ltx25_audio_in_16gb", "still"])
def test_non_foley_lanes_never_reach_detector_models(monkeypatch, lane):
    monkeypatch.setattr(speech, "_load_vad", lambda: pytest.fail("VAD loaded"))
    row = _row()
    row["engine_id"] = lane
    assert speech.detect_foley_speech([row], {}) == {}


@pytest.mark.parametrize("rows", [[_row("" )], [_row("b1"), _row("b1")]])
def test_missing_or_duplicate_identity_is_no_duck(monkeypatch, rows):
    monkeypatch.setattr(speech, "_load_vad", lambda: pytest.fail("VAD loaded"))
    receipt = speech.detect_foley_speech(rows, {})
    assert all(item["ducked"] is False and "beat_id" in item["reason"]
               for item in receipt.values())


def _pcm_rows(tmp_path):
    rows = [_row("b1"), _row("b2")]
    for i, row in enumerate(rows):
        row["foley_path"] = str(tmp_path / (row["beat_id"] + ".wav"))
        row["start_s"] = i / FPS  # overlap by one frame
        fs.write_pcm16_wav(row["foley_path"], np.full((2, 2 * STEP), 0.2), RATE)
    return rows


def test_mix_halves_only_flagged_bed_and_preserves_voice_envelope(tmp_path):
    rows = _pcm_rows(tmp_path)
    master = np.full((2, 4 * STEP), 0.3, dtype=np.float32)
    before, stats_before = fs.mix_foley_under_master(master, RATE, rows, fps=FPS)
    rows[0]["foley_speech_duck"] = True
    rows[1]["foley_speech_duck"] = "false"  # never treat a string as a verdict
    after, stats = fs.mix_foley_under_master(master, RATE, rows, fps=FPS)
    stem, _ = fs.read_pcm16_wav(rows[0]["foley_path"])
    expected = before.copy()
    expected[:, :2 * STEP] -= stem * fs.FOLEY_GAIN * 0.5
    np.testing.assert_allclose(after, expected, atol=3e-8)
    bed, _ = fs.mix_foley_under_master(np.zeros_like(master), RATE, rows, fps=FPS)
    np.testing.assert_allclose(after - bed, master * fs.MASTER_GAIN_UNDER_FOLEY, atol=3e-8)
    assert stats["global_master_gain"] == stats_before["global_master_gain"]
    assert stats["speech_ducked_beats"] == ["b1"] and stats["speech_ducked"] == 1


def test_mime_cannot_be_ducked_even_with_stale_true_flag(tmp_path):
    rows = _pcm_rows(tmp_path)[:1]
    rows[0].update(engine_id=MIME, foley_speech_duck=True)
    master = np.ones((2, 4 * STEP), dtype=np.float32)
    mixed, stats = fs.mix_foley_under_master(master, RATE, rows, fps=FPS)
    stem, _ = fs.read_pcm16_wav(rows[0]["foley_path"])
    np.testing.assert_array_equal(mixed[:, :2 * STEP], stem)
    assert stats["speech_ducked"] == 0


def _episode(tmp_path, monkeypatch):
    from nodes import scene_sequencer
    path = tmp_path / "audio" / "episode_ledger.json"
    path.parent.mkdir()
    data = dict(episode_id="episode", schema_version="1.0",
                meta={"technical_model": "episode-model", "keep": "original"},
                cast=[], lines=[], beats=[], scenes=[], shots=[], music=[], clips=[])
    assert ledger.save_ledger_safe(path, data)
    monkeypatch.setattr(ledger, "in_flight_ledger_path", lambda: path)
    monkeypatch.setattr(scene_sequencer, "_master_loudness", lambda audio, **kw: (audio, {}))
    master = tmp_path / "episode_master.wav"
    fs.write_pcm16_wav(master, np.full((2, 4 * STEP), 0.3), RATE)
    return path, master


def test_mux_wires_flags_and_receipt_into_real_ledger(monkeypatch, tmp_path):
    ledger_path, master = _episode(tmp_path, monkeypatch)
    rows = _pcm_rows(tmp_path)
    rows[1]["engine_id"] = MIME
    seen = []

    def detect(bearing, meta):
        seen.extend(bearing)
        assert meta["technical_model"] == "episode-model"
        # A later owner wrote while detection ran: stamp must reload.
        fresh = ledger.load_ledger_safe(ledger_path)
        fresh["meta"]["keep"] = "updated"
        assert ledger.save_ledger_safe(ledger_path, fresh)
        return {"b1": dict(vad=True, transcript="Help", verdict=True,
                           reason="Human words", ducked=True)}

    monkeypatch.setattr(speech, "detect_foley_speech", detect)
    output, report = mux._compile_foley_master(str(master), json.dumps({"clips": rows}), FPS)
    assert [row["beat_id"] for row in seen] == ["b1"]
    result = ledger.load_ledger_safe(ledger_path)
    assert result["meta"]["keep"] == "updated"
    assert result["meta"]["foley_speech"]["b1"]["ducked"] is True
    assert "foley speech duck: 1 of 1 beats (vad 1, llm 1)" in report
    rows[0]["foley_speech_duck"] = True
    original, _ = fs.read_pcm16_wav(master)
    expected, _ = fs.mix_foley_under_master(original, RATE, rows, fps=FPS)
    expected_path = tmp_path / "expected.wav"
    fs.write_pcm16_wav(expected_path, expected, RATE)
    assert Path(output).read_bytes() == expected_path.read_bytes()


def test_detector_exception_preserves_mix_bytes_and_records_reason(monkeypatch, tmp_path):
    ledger_path, master = _episode(tmp_path, monkeypatch)
    rows = _pcm_rows(tmp_path)
    original, _ = fs.read_pcm16_wav(master)
    expected, _ = fs.mix_foley_under_master(original, RATE, rows, fps=FPS)
    expected_path = tmp_path / "expected.wav"
    fs.write_pcm16_wav(expected_path, expected, RATE)
    rows[0]["foley_speech_duck"] = True

    def broken(bearing, meta):
        bearing[1]["foley_speech_duck"] = True
        raise RuntimeError("detector broke")

    monkeypatch.setattr(speech, "detect_foley_speech", broken)
    output, report = mux._compile_foley_master(str(master), json.dumps({"clips": rows}), FPS)
    assert Path(output).read_bytes() == expected_path.read_bytes()
    receipt = ledger.load_ledger_safe(ledger_path)["meta"]["foley_speech"]
    assert all(not r["ducked"] and "detector broke" in r["reason"] for r in receipt.values())


def test_mime_only_mux_never_calls_detector(monkeypatch, tmp_path):
    _, master = _episode(tmp_path, monkeypatch)
    rows = _pcm_rows(tmp_path)
    for row in rows:
        row.update(engine_id=MIME, foley_speech_duck=True)
    monkeypatch.setattr(mux, "_analyse_foley_speech", lambda *_: pytest.fail("detector called"))
    _, report = mux._compile_foley_master(str(master), json.dumps({"clips": rows}), FPS)
    assert not any("speech duck" in line for line in report)


@pytest.mark.parametrize("cuda", [True, False])
def test_whisper_auto_download_is_ungated_small_external_and_device_aware(monkeypatch, tmp_path, cuda):
    calls, downloads = [], []
    monkeypatch.setattr(speech, "_models_dir", lambda: tmp_path / "models")
    monkeypatch.setitem(sys.modules, "ctranslate2", SimpleNamespace(
        get_cuda_device_count=lambda: int(cuda),
        get_supported_compute_types=lambda device: {"float16", "int8_float16"}))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        snapshot_download=lambda *a, **kw: downloads.append((a, kw)) or kw["local_dir"]))
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(
        WhisperModel=lambda *a, **kw: calls.append((a, kw))))
    speech._load_whisper()
    assert downloads[0][0] == ("Systran/faster-whisper-base",)
    assert downloads[0][1]["token"] is False
    assert Path(downloads[0][1]["local_dir"]).is_relative_to(tmp_path / "models")
    assert calls[0][1]["device"] == ("cuda" if cuda else "cpu")
    assert calls[0][1]["compute_type"] == ("float16" if cuda else "int8")
    assert calls[0][1]["local_files_only"] is True


def test_audio_input_is_contiguous_mono_16khz_and_transcription_is_exhausted(tmp_path):
    rows = _pcm_rows(tmp_path)
    seen = []

    def transcribe(audio, **kwargs):
        assert audio.ndim == 1 and audio.dtype == np.float32
        assert len(audio) == 2 * STEP // 3
        assert audio.flags.c_contiguous
        assert kwargs["vad_filter"] is False and kwargs["condition_on_previous_text"] is False
        assert kwargs["task"] == "transcribe" and kwargs["suppress_tokens"] == [-1]
        assert kwargs["suppress_blank"] is True

        def segments():
            seen.append("iterated")
            yield SimpleNamespace(text="  ")
            yield SimpleNamespace(text=" Hello ")
        return segments(), None

    assert speech._transcribe(rows[0]["foley_path"], SimpleNamespace(transcribe=transcribe)) == "Hello"
    assert seen == ["iterated"]


def test_mux_and_detector_import_without_torch_or_optional_packages(tmp_path):
    probe = tmp_path / "cold_import.py"
    root = Path(__file__).resolve().parents[1]
    probe.write_text("""import importlib.abc, sys
class BlockHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'silero_vad', 'faster_whisper', 'ctranslate2'}:
            raise AssertionError('heavy import: ' + fullname)
sys.meta_path.insert(0, BlockHeavy())
sys.path.insert(0, sys.argv[1])
from nodes import otr_master_audio_mux
from nodes._otr_video_engines import foley_speech
assert otr_master_audio_mux.OTRMasterAudioMux.INPUT_TYPES()['optional']
assert 'torch' not in sys.modules
""", encoding="utf-8")
    result = subprocess.run([sys.executable, str(probe), str(root)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
