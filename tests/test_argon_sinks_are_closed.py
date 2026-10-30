"""The 2026-10-30 Argon registry claims that survived grounding, each
reproduced and refused.

Gemini 4 Argon (run by the operator) answered "what else would the registry's
reviewer flag"; a Sonnet fan-out grounded every claim against the real files
with executed probes. These are the ones that held: misaki's English G2P would
pip-install spaCy's model mid-render on a fresh install; a control character
in a filename injected a line into an ffmpeg concat list; a ledger's ElevenLabs
voice id was interpolated into the Comfy proxy's URL path; a repository could
ship the pack's own download receipt and its keys were stat'ed unconfined; the
unauthenticated ledger route scrubbed only whole-string paths; the chatterbox
sidecar used tempfile.mktemp; and the tokenizer allow prefix with a deny list
beneath it was case-sensitive and never complete.
"""
from __future__ import annotations

import inspect
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


# --- RANK 1 ---------------------------------------------------------------- #

def test_english_kokoro_refuses_before_misaki_can_download_spacy(monkeypatch):
    from nodes._otr_audio_engines import _kokoro_backends as KB
    import importlib.util
    real = importlib.util.find_spec

    def absent(name, *a, **k):
        return None if name == "en_core_web_sm" else real(name, *a, **k)
    monkeypatch.setattr(importlib.util, "find_spec", absent)
    touched = []
    fake = types.ModuleType("kokoro")

    class KPipeline:  # noqa: N801 -- the real class name
        def __init__(self, **kw):
            touched.append(kw)
    fake.KPipeline = KPipeline
    monkeypatch.setitem(sys.modules, "kokoro", fake)
    with pytest.raises(KB.BackendUnavailable, match="spacy download en_core_web_sm"):
        KB.TorchKokoroBackend("cpu", lang_code="b").load()
    assert touched == []                       # misaki was never given the chance
    KB.TorchKokoroBackend("cpu", lang_code="e").load()     # espeak languages need no spaCy
    assert touched and touched[0]["lang_code"] == "e"


# --- RANK 4 ---------------------------------------------------------------- #

def test_a_control_character_in_a_path_is_refused_by_text(tmp_path):
    from nodes._otr_paths import OtrPathContractError, confine_to_read_roots, confined_or_empty
    for bad in ("a\nfile /etc/hosts\nfile b.mp4", "a\rb.mp4", "a\x0bb.mp4", "a\x7fb.mp4"):
        with pytest.raises(OtrPathContractError, match="control character"):
            confine_to_read_roots(str(tmp_path / bad), "video_path", roots=[str(tmp_path)])
        assert confined_or_empty(str(tmp_path / bad), roots=[str(tmp_path)]) == ""
    assert confined_or_empty(str(tmp_path / "fine.mp4"), roots=[str(tmp_path)]) == str(tmp_path / "fine.mp4")


def test_a_listed_frame_name_with_a_newline_is_not_a_frame(tmp_path, monkeypatch):
    from nodes._otr_video_engines import directory_clip as DC
    clip = tmp_path / "clip"
    clip.mkdir()
    (clip / "0001.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    import os
    monkeypatch.setattr(os, "listdir", lambda p: ["0001.png", "0002\nfile x.png"])
    with pytest.raises(ValueError, match="control character"):
        DC.list_directory_frames(str(clip))


# --- RANK 5 ---------------------------------------------------------------- #

def test_an_elevenlabs_voice_id_that_is_not_an_id_never_reaches_the_proxy(monkeypatch):
    from nodes._otr_audio_engines import eng_cloud_elevenlabs as EL
    from nodes._otr_shared import cloud_media_invoke as I
    recorded = []
    monkeypatch.setattr(I, "invoke_partner_node",
                        lambda *a, **k: recorded.append((a, k)) or pytest.fail("reached the proxy"))
    eng = EL.ElevenLabsCloudVoice()
    for hostile in ("../../../users/me", "abc?output_format=x", "a#b", "x" * 65, "id with space",
                    "a/b", "%2e%2e"):
        with pytest.raises(ValueError, match="not an ElevenLabs voice id"):
            eng.generate_voice("hello", hostile, None, 0)
    assert recorded == []


# --- RANK 6 + RANK 2 ------------------------------------------------------- #

def test_the_download_never_fetches_a_receipt_or_a_tokenizer_named_payload():
    from nodes import _otr_model_catalog as MC
    got = MC._fetched_paths([
        ".otr_download_complete.json", "sub/.otr_download_complete.json",
        "tokenizer.EXE", "tokenizer.js", "tokenizer_payload.PY", "tokenizer.model.exe",
        "tokenizer.json", "tokenizer_config.json", "tokenizer.model", "tokenizer.model.v3",
        "tokenizer.tiktoken", "tekken.json", "model-00001-of-00002.safetensors", "config.json",
    ])
    assert got == {"tokenizer.json", "tokenizer_config.json", "tokenizer.model",
                   "tokenizer.model.v3", "tokenizer.tiktoken", "tekken.json",
                   "model-00001-of-00002.safetensors", "config.json"}


def test_a_receipt_key_outside_its_folder_is_no_receipt(tmp_path):
    from nodes import _otr_llm_folder as LF
    folder = tmp_path / "LLM" / "org--repo"
    folder.mkdir(parents=True)
    (folder / "config.json").write_text("{}", encoding="utf-8")
    (folder / "model.safetensors").write_bytes(b"0123456789")
    (tmp_path / "outside.txt").write_bytes(b"1234567")

    def receipt(files):
        (folder / LF.RECEIPT_NAME).write_text(json.dumps(
            {"schema": LF.RECEIPT_SCHEMA, "files": files}), encoding="utf-8")
    receipt({"../outside.txt": 7, "model.safetensors": 10})
    assert LF._receipt(folder) is None
    receipt({str(tmp_path / "outside.txt"): 7})
    assert LF._receipt(folder) is None
    receipt({"model.safetensors": 10, "config.json": 2})
    assert LF._receipt(folder) is not None


# --- RANK 8 ---------------------------------------------------------------- #

def _scrub():
    src = (REPO / "__init__.py").read_text(encoding="utf-8")
    start = src.index("    import re as _otr_re")
    end = src.index('    @_otr_PromptServer.instance.routes.get("/otr/latest_ledger")')
    body = src[start:end].split("\n")
    block = "\n".join(line[4:] if line.startswith("    ") else line for line in body)
    ns: dict = {}
    exec(block, ns)  # noqa: S102 -- shipped code, as test_widget_cannot_name_the_binary loads it
    return ns["_otr_scrub_paths"]


def test_the_ledger_route_scrubs_a_path_wherever_it_sits():
    scrub = _scrub()
    b = chr(92)
    win = "C:" + b + "Users" + b + "jeffr" + b + "a.wav"
    assert "Users" not in scrub("warning: could not open " + win + " (errno 2)")
    assert scrub("warning: could not open " + win + " (errno 2)").endswith("a.wav (errno 2)")
    assert "Users" not in scrub("file:///C:/Users/jeffr/a.wav")
    assert "Users" not in scrub("line one\nsee " + win + "\nline three")
    assert "Users" not in scrub("No such file or directory: '/home/jeffr/x/a.wav'")
    assert "Users" not in scrub(r"\\host\share\Users\jeffr\a.wav")
    assert scrub({win: 1}) == {"a.wav": 1}                      # keys too
    deep = {"k": "v"}
    for _ in range(26):
        deep = {"n": deep}
    assert "v" not in json.dumps(scrub(deep)) and "<redacted>" in json.dumps(scrub(deep))
    # prose and URLs are left alone
    assert scrub("https://example.com/path/to/thing") == "https://example.com/path/to/thing"
    assert scrub("the fraction 3/4 and a /dev word") == "the fraction 3/4 and a /dev word"


# --- RANK 10 --------------------------------------------------------------- #

def test_the_chatterbox_sidecar_path_comes_from_the_pack_allocator():
    from nodes._otr_audio_engines import eng_chatterbox as CBX
    src = inspect.getsource(CBX)
    assert "tempfile.mktemp(" not in src
    assert 'otr_engine_tmp_path("otr_cbx_", ".wav")' in inspect.getsource(CBX.ChatterboxEngine.generate_voice)
