"""The kokoro engine's two backends (2026-09-02, queue item 2).

What matters, one assertion each: `auto` picks torch iff the torch package
imports, else kokoro-onnx, else a NAMED EngineUnusable that says both pip lines;
a forced backend never falls through; the torch call shape is ONE KPipeline call
over the full line with split_pattern (that is what keeps the RTX 5080's bytes
identical -- the sha256 proof rides on it); the ONNX call kwargs are pinned; the
voice table is derived from the .pt files under a digest-named file, never a
replace-in-place; the ledger's cuda stamp is logged, not raised, under ONNX; the
cache hook is empty under torch; the prefetch's gate never raises on a
sys.modules fake; nothing in the backends can reach the network; and the five
espeak rows reach kokoro-onnx as the torch pipeline's own phonemes, while
Japanese and Mandarin go to torch or refuse by name.
"""
from __future__ import annotations

import importlib
import inspect
import logging
import os
import sys
import types

import numpy as np
import pytest

os.environ.setdefault("OTR_TEST_MODE", "1")

from nodes._otr_audio_engines import _kokoro_backends as kb  # noqa: E402
from nodes._otr_audio_engines import eng_kokoro  # noqa: E402
from nodes._otr_audio_engines.registry import (  # noqa: E402
    EngineUnusable, EngineUsabilityReason,
)
import nodes._otr_kokoro_voice_prefetch as prefetch  # noqa: E402


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
def _fake_torch_kokoro(monkeypatch):
    monkeypatch.setitem(sys.modules, "kokoro",
                        types.SimpleNamespace(KPipeline=lambda **kw: None))


def _fake_onnx(monkeypatch):
    monkeypatch.setitem(sys.modules, "kokoro_onnx", types.SimpleNamespace(Kokoro=object))


def _absent(monkeypatch, name):
    monkeypatch.setitem(sys.modules, name, None)   # `import name` raises ImportError


def test_auto_prefers_torch_when_both_import(monkeypatch):
    _fake_torch_kokoro(monkeypatch)
    _fake_onnx(monkeypatch)
    assert kb.select_backend_name(None) == "torch"


def test_auto_falls_to_onnx_when_torch_kokoro_is_absent(monkeypatch):
    _absent(monkeypatch, "kokoro")
    _fake_onnx(monkeypatch)
    assert kb.select_backend_name("auto") == "onnx"


def test_neither_installed_names_both_pip_lines(monkeypatch):
    _absent(monkeypatch, "kokoro")
    _absent(monkeypatch, "kokoro_onnx")
    with pytest.raises(kb.BackendUnavailable) as exc:
        kb.select_backend_name(None)
    msg = str(exc.value)
    assert "pip install kokoro " in msg and "pip install kokoro-onnx" in msg


def test_forced_backend_never_falls_through(monkeypatch):
    _fake_torch_kokoro(monkeypatch)
    _absent(monkeypatch, "kokoro_onnx")
    with pytest.raises(kb.BackendUnavailable, match="OTR_KOKORO_BACKEND=onnx"):
        kb.select_backend_name("onnx")
    _absent(monkeypatch, "kokoro")
    _fake_onnx(monkeypatch)
    with pytest.raises(kb.BackendUnavailable, match="OTR_KOKORO_BACKEND=torch"):
        kb.select_backend_name("torch")
    with pytest.raises(kb.BackendUnavailable, match="auto, torch or onnx"):
        kb.select_backend_name("cuda")


def test_selection_is_re_evaluated_per_call(monkeypatch):
    _fake_torch_kokoro(monkeypatch)
    _fake_onnx(monkeypatch)
    assert kb.select_backend_name(None) == "torch"
    _absent(monkeypatch, "kokoro")
    assert kb.select_backend_name(None) == "onnx"   # no process-level cache


def test_provider_list_parsing_rejects_empty():
    assert kb.parse_onnx_providers(None) == ["CPUExecutionProvider"]
    assert kb.parse_onnx_providers("CUDAExecutionProvider, CPUExecutionProvider") == [
        "CUDAExecutionProvider", "CPUExecutionProvider"]
    with pytest.raises(kb.BackendUnavailable):
        kb.parse_onnx_providers(" , ")


# --------------------------------------------------------------------------- #
# the torch call shape (the byte-identity contract)
# --------------------------------------------------------------------------- #
def test_torch_backend_is_one_pipeline_call_over_the_full_line(monkeypatch):
    calls = []

    class _Pipe:
        def __init__(self, **kw):
            calls.append(("init", kw))

        def __call__(self, text, **kw):
            calls.append(("call", text, kw))
            yield ("g", "p", np.full(10, 0.5, dtype=np.float32))
            yield ("g", "p", np.full(5, -0.25, dtype=np.float32))

    monkeypatch.setitem(sys.modules, "kokoro", types.SimpleNamespace(KPipeline=_Pipe))
    backend = kb.TorchKokoroBackend("cpu")
    backend.load()
    out = backend.synthesize("line one\n\nline two", "bm_george", 0.95)
    init = [c for c in calls if c[0] == "init"][0][1]
    assert init["lang_code"] == "b" and init["device"] == "cpu"
    call = [c for c in calls if c[0] == "call"]
    assert len(call) == 1, "the torch backend must NOT pre-split the line"
    assert call[0][1] == "line one\n\nline two"
    assert call[0][2] == {"voice": "bm_george", "speed": 0.95, "split_pattern": r"\n+"}
    assert out.dtype == np.float32 and out.size == 15   # concatenated, not normalized here


# --------------------------------------------------------------------------- #
# the ONNX call contract
# --------------------------------------------------------------------------- #
class _StubKokoro:
    def __init__(self, voices=("bm_george", "bf_emma"), rate=24000):
        self.calls = []
        self._voices = list(voices)
        self._rate = rate

    def get_voices(self):
        return list(self._voices)

    def create(self, text, **kw):
        self.calls.append((text, kw))
        return np.full(100, 0.5, dtype=np.float32), self._rate


def _onnx_backend_with_stub(stub):
    backend = kb.OnnxKokoroBackend("model.onnx", "voices.npz")
    backend._kokoro = stub
    backend._session = object()
    return backend


def test_onnx_create_kwargs_are_pinned_and_split_on_newlines():
    stub = _StubKokoro()
    backend = _onnx_backend_with_stub(stub)
    out = backend.synthesize("first line\n\n   \nsecond line\n", "bf_emma", 0.95)
    assert [c[0] for c in stub.calls] == ["first line", "second line"]   # empties skipped
    for _, kw in stub.calls:
        assert kw == {"voice": "bf_emma", "speed": 0.95, "lang": "en-gb", "trim": False,
                      "sentence_pause": 0.25, "clause_pause": 0.1}
    assert out.dtype == np.float32 and out.size == 200


def test_onnx_unknown_voice_and_wrong_rate_fail_by_name():
    backend = _onnx_backend_with_stub(_StubKokoro())
    with pytest.raises(kb.BackendUnavailable, match="not in the ONNX voice table"):
        backend.synthesize("hello", "vz_donor_lemmy", 0.95)
    backend = _onnx_backend_with_stub(_StubKokoro(rate=22050))
    with pytest.raises(RuntimeError, match="22050"):
        backend.synthesize("hello", "bm_george", 0.95)


def test_onnx_session_is_built_with_the_explicit_provider_list(monkeypatch):
    seen = {}

    class _Session:
        def __init__(self, path, sess_options=None, providers=None):
            seen["path"] = path
            seen["providers"] = providers
            seen["threads"] = sess_options.intra_op_num_threads

        def get_providers(self):
            return list(seen["providers"])

    class _Options:
        intra_op_num_threads = 0

    fake_ort = types.SimpleNamespace(
        InferenceSession=_Session, SessionOptions=_Options,
        get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setitem(sys.modules, "kokoro_onnx", types.SimpleNamespace(
        Kokoro=types.SimpleNamespace(from_session=lambda s, v: _StubKokoro())))
    monkeypatch.setenv("ONNX_PROVIDER", "CUDAExecutionProvider")   # kokoro-onnx's own knob is ignored
    backend = kb.OnnxKokoroBackend("m.onnx", "v.npz")
    backend.load()
    assert seen["providers"] == ["CPUExecutionProvider"]
    assert 1 <= seen["threads"] <= kb.ONNX_THREAD_CAP
    bad = kb.OnnxKokoroBackend("m.onnx", "v.npz", providers=["DmlExecutionProvider"])
    with pytest.raises(kb.BackendUnavailable, match="not available"):
        bad.load()


# --------------------------------------------------------------------------- #
# the voice table (npz) derived from the .pt files
# --------------------------------------------------------------------------- #
def _write_voice(dirpath, name, shape=(510, 1, 256)):
    import torch

    torch.save(torch.full(shape, 0.5, dtype=torch.float32), os.path.join(dirpath, name + ".pt"))


def test_voice_table_keys_are_bare_ids_and_digest_named(tmp_path):
    voices = tmp_path / "voices"
    voices.mkdir()
    for v in ("bm_george", "bf_emma", "af_heart"):
        _write_voice(str(voices), v)
    path = kb.ensure_voices_npz(str(voices))
    assert os.path.basename(path).startswith("_onnx_voices.") and path.endswith(".npz")
    with np.load(path) as table:
        assert sorted(table.files) == ["af_heart", "bf_emma", "bm_george"]
        assert table["bm_george"].shape == (510, 1, 256) and table["bm_george"].dtype == np.float32
    assert kb.ensure_voices_npz(str(voices)) == path            # reused, not rebuilt


def test_voice_table_changes_name_when_the_set_changes_and_skips_a_corrupt_file(tmp_path):
    voices = tmp_path / "voices"
    voices.mkdir()
    _write_voice(str(voices), "bm_george")
    first = kb.ensure_voices_npz(str(voices))
    (voices / "am_adam.pt").write_bytes(b"not a tensor")
    _write_voice(str(voices), "bf_emma")
    second = kb.ensure_voices_npz(str(voices))
    assert second != first and not os.path.exists(first)       # new digest, stale one removed
    with np.load(second) as table:
        assert sorted(table.files) == ["bf_emma", "bm_george"]  # the corrupt one skipped


def test_voice_table_with_no_voices_is_a_named_error(tmp_path):
    with pytest.raises(kb.BackendUnavailable, match="no kokoro voice files"):
        kb.ensure_voices_npz(str(tmp_path))


# --------------------------------------------------------------------------- #
# the engine over the ONNX backend
# --------------------------------------------------------------------------- #
class _StubOnnxBackend:
    name = "onnx"
    instances: list = []

    def __init__(self, model_path, voices_npz, providers=None, threads=None, lang_code="b"):
        self.model_path, self.voices_npz = model_path, voices_npz
        self.lang_code = lang_code
        self.providers_active = ["CPUExecutionProvider"]
        self.threads = 4
        self.closed = False
        _StubOnnxBackend.instances.append(self)

    def load(self):
        pass

    def synthesize(self, text, voice_id, speed):
        return np.full(48, 0.5, dtype=np.float32)

    def close(self):
        self.closed = True


def _onnx_engine(monkeypatch, tmp_path, *, model_exists=True):
    model = tmp_path / "onnx" / "model.onnx"
    model.parent.mkdir(parents=True)
    if model_exists:
        model.write_bytes(b"onnx")
    voices = tmp_path / "voices"
    voices.mkdir()
    (voices / "bm_george.pt").write_bytes(b"pt")
    monkeypatch.setattr(eng_kokoro, "_kokoro_model_dir", lambda: str(tmp_path))
    monkeypatch.setattr(kb, "select_backend_name", lambda env=None: "onnx")
    monkeypatch.setattr(kb, "ensure_voices_npz", lambda d: str(voices / "_onnx_voices.x.npz"))
    monkeypatch.setattr(kb, "OnnxKokoroBackend", _StubOnnxBackend)
    eng = eng_kokoro.KokoroEngine()
    eng.requested_device = "cuda"
    return eng


def test_engine_onnx_path_logs_the_ledger_device_and_returns_the_contract(monkeypatch, tmp_path, caplog):
    eng = _onnx_engine(monkeypatch, tmp_path)
    with caplog.at_level(logging.INFO, logger="OTR"):
        out = eng.generate_voice("hello there", "bm_george", None, 7)
    assert tuple(out["waveform"].shape) == (1, 1, 48) and out["sample_rate"] == 24000
    assert float(out["waveform"].abs().max()) == pytest.approx(0.9)
    assert eng._backend_name == "onnx"
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "backend=onnx" in text and "voice_device='cuda'" in text and "CPU by design" in text
    assert "lang=b" in text
    eng.unload()
    assert eng._backend is None and _StubOnnxBackend.instances[-1].closed


def test_engine_missing_onnx_model_names_the_offline_fetch(monkeypatch, tmp_path):
    eng = _onnx_engine(monkeypatch, tmp_path, model_exists=False)
    with pytest.raises(EngineUnusable) as exc:
        eng.generate_voice("hello", "bm_george", None, 7)
    assert exc.value.reason == EngineUsabilityReason.MISSING_MODEL
    assert "huggingface-cli download onnx-community/Kokoro-82M-v1.0-ONNX" in str(exc.value)


def test_engine_with_no_backend_installed_is_a_named_error(monkeypatch, tmp_path):
    voices = tmp_path / "voices"
    voices.mkdir()
    (voices / "bm_george.pt").write_bytes(b"pt")
    monkeypatch.setattr(eng_kokoro, "_kokoro_model_dir", lambda: str(tmp_path))
    _absent(monkeypatch, "kokoro")
    _absent(monkeypatch, "kokoro_onnx")
    monkeypatch.delenv("OTR_KOKORO_BACKEND", raising=False)
    eng = eng_kokoro.KokoroEngine()
    with pytest.raises(EngineUnusable) as exc:
        eng.generate_voice("hello", "bm_george", None, 7)
    assert exc.value.reason == EngineUsabilityReason.MISSING_MODEL
    assert "pip install kokoro-onnx" in str(exc.value)


def test_render_time_params_is_empty_under_torch_and_stamped_under_onnx(monkeypatch, tmp_path):
    eng = eng_kokoro.KokoroEngine()
    monkeypatch.delenv("OTR_KOKORO_BACKEND", raising=False)
    monkeypatch.setattr(eng_kokoro, "_spec_present", lambda n: True)
    assert eng.render_time_params() == {}                       # torch: caching stays "static"
    monkeypatch.setattr(eng_kokoro, "_spec_present", lambda n: n == "kokoro_onnx")
    monkeypatch.setattr(eng_kokoro, "_kokoro_model_dir", lambda: str(tmp_path))
    assert eng.render_time_params() == {"backend": "onnx", "onnx_model": "missing"}
    model = tmp_path / "onnx" / "model.onnx"
    model.parent.mkdir()
    model.write_bytes(b"onnx")
    params = eng.render_time_params()
    assert params["backend"] == "onnx" and params["onnx_model"].startswith("4:")


# --------------------------------------------------------------------------- #
# prestartup prefetch of the ONNX model
# --------------------------------------------------------------------------- #
def test_spec_probe_never_raises_on_a_sys_modules_fake(monkeypatch):
    monkeypatch.setitem(sys.modules, "kokoro", types.SimpleNamespace())   # no __spec__
    assert prefetch._spec_exists("kokoro") is False
    assert eng_kokoro._spec_present("kokoro") is False


def test_onnx_backend_wanted_gate(monkeypatch):
    monkeypatch.setenv("OTR_KOKORO_BACKEND", "onnx")
    assert prefetch.onnx_backend_wanted() is True
    monkeypatch.setenv("OTR_KOKORO_BACKEND", "torch")
    assert prefetch.onnx_backend_wanted() is False
    monkeypatch.delenv("OTR_KOKORO_BACKEND")
    monkeypatch.setattr(prefetch, "_spec_exists", lambda n: n == "kokoro_onnx")
    assert prefetch.onnx_backend_wanted() is True
    monkeypatch.setattr(prefetch, "_spec_exists", lambda n: True)
    assert prefetch.onnx_backend_wanted() is False


def test_onnx_prefetch_receipts(monkeypatch, tmp_path):
    monkeypatch.setattr(prefetch, "_models_dir", lambda: str(tmp_path))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("OTR_SKIP_KOKORO_PREFETCH", raising=False)
    dest = tmp_path / "TTS" / "KokoroTTS" / "onnx" / "model.onnx"

    monkeypatch.setattr(prefetch, "onnx_backend_wanted", lambda: False)
    r = prefetch.prefetch_kokoro_onnx_model()
    assert r["attempted"] == 0 and "not selected" in r["reason"]

    monkeypatch.setattr(prefetch, "onnx_backend_wanted", lambda: True)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    r = prefetch.prefetch_kokoro_onnx_model()
    assert r["skipped_offline"] and r["attempted"] == 0
    monkeypatch.delenv("HF_HUB_OFFLINE")

    fetched = {}

    def _fake_download(repo_id, filename, local_dir):
        fetched["args"] = (repo_id, filename, local_dir)
        target = os.path.join(local_dir, *filename.split("/"))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as fh:
            fh.write(b"onnx")
        return target

    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(hf_hub_download=_fake_download))
    r = prefetch.prefetch_kokoro_onnx_model()
    assert r["fetched"] == 1 and dest.exists()
    assert fetched["args"] == (prefetch.KOKORO_ONNX_REPO_ID, "onnx/model.onnx",
                               str(tmp_path / "TTS" / "KokoroTTS"))
    r = prefetch.prefetch_kokoro_onnx_model()
    assert r["present"] is True and r["attempted"] == 0          # second boot: nothing to do


def test_prefetch_and_engine_agree_on_the_onnx_path_and_boot_stays_light():
    assert prefetch._KOKORO_ONNX_REL_PATH == eng_kokoro._KOKORO_ONNX_REL_PATH
    assert prefetch._KOKORO_MODEL_SUBDIR == eng_kokoro._KOKORO_MODEL_SUBDIR
    assert prefetch.KOKORO_ONNX_REPO_ID == eng_kokoro.KOKORO_ONNX_REPO_ID
    source = inspect.getsource(prefetch)
    for heavy in ("import torch", "import numpy", "import kokoro", "import onnxruntime"):
        assert heavy not in source, "%s must not be imported at prestartup" % heavy


def test_backends_module_cannot_reach_the_network_and_imports_nothing_heavy_at_top():
    source = inspect.getsource(kb)
    for banned in ("hf_hub_download", "snapshot_download", "requests.", "urllib"):
        assert banned not in source
    top = [n for n in importlib.import_module(kb.__name__).__dict__
           if n in ("torch", "numpy", "onnxruntime", "kokoro", "kokoro_onnx")]
    assert top == [], "heavy modules bound at module top: %s" % top


# --- PBUG-20260918-06: the Japanese row's empty MeCab dictionary -------------
# The seven-language sweep's Japanese leg died at t=229s inside
# fugashi.Tagger() with a path that does not exist, because misaki[ja] pulls
# `unidic` (a downloader, not a dictionary). These pin the two halves that
# actually cost time: the message must name the UNINSTALL as well as the
# install, and it must never reword an error it did not diagnose.

_MECAB_ERROR = RuntimeError(
    "Failed initializing MeCab. ... param.cpp(69) [ifs] no such file or "
    r"directory: C:\...\site-packages\unidic\dicdir\mecabrc")


def test_mecab_message_names_both_halves_of_the_fix():
    out = kb._mecab_dictionary_error(_MECAB_ERROR, "j")
    assert out is not _MECAB_ERROR
    text = str(out)
    # Installing unidic-lite ALONE leaves the error unchanged, because fugashi
    # prefers `unidic` whenever it imports. A message that named only the
    # install would send the reader in a circle.
    assert "pip uninstall -y unidic" in text
    assert "pip install unidic-lite" in text
    assert "not optional" in text
    assert "Original error" in text and "mecabrc" in text


def test_a_mecab_failure_on_a_non_japanese_row_is_left_alone():
    # No other language row loads MeCab, so this signature on `z` or `b` means
    # something else entirely and must keep its own words.
    for lang_code in ("z", "b", "e", "f", ""):
        assert kb._mecab_dictionary_error(_MECAB_ERROR, lang_code) is _MECAB_ERROR


def test_other_kpipeline_failures_stay_exactly_as_loud():
    other = RuntimeError("CUDA error: device-side assert triggered")
    assert kb._mecab_dictionary_error(other, "j") is other


def test_the_japanese_row_is_the_only_one_that_loads_mecab():
    assert kb._MECAB_LANG_CODES == ("j",)


# --------------------------------------------------------------------------- #
# the espeak languages on ONNX (2026-09-28)
# --------------------------------------------------------------------------- #
# ComfyUI Desktop and the portable build run Python 3.13, where neither the
# torch package nor misaki installs, and three of the operator's randomized
# runs died at the first Spanish line after the writer had spent two minutes.
# The torch pipeline voices these five rows through misaki's EspeakG2P; the
# ONNX backend now does the same, and the proof is phoneme equality with
# misaki itself wherever misaki is installed.

#: One sentence per espeak row: a borrowed English name (the language-switch
#: flags misaki strips), brackets and angle quotes (its bracket handling), and
#: the tied sounds it folds into Kokoro's own symbols.
_ESPEAK_SAMPLES = {
    "es": "Buenas noches. Esta es la se\xf1al perdida (dice Wendy): \xab\xa1muchas gracias, Chihuahua!\xbb",
    "fr-fr": "Bonsoir. Ici le signal perdu, et Wendy r\xe9pond : \xab merci beaucoup ! \xbb",
    "it": "Buonasera. Questo \xe8 il segnale perduto (dice Wendy): grazie mille!",
    "pt-br": "Boa noite. Este \xe9 o sinal perdido, e Wendy diz: muito obrigado!",
    "hi": "\u0936\u0941\u092d \u0938\u0902\u0927\u094d\u092f\u093e\u0964 \u092f\u0939 \u0916\u094b\u092f\u093e \u0939\u0941\u0906 \u0938\u0902\u0915\u0947\u0924 \u0939\u0948, \u0914\u0930 \u0935\u0947\u0902\u0921\u0940 \u0915\u0939\u0924\u0940 \u0939\u0948: \u092c\u0939\u0941\u0924 \u0927\u0928\u094d\u092f\u0935\u093e\u0926!",
}


def _kokoro_lang_codes():
    """``kokoro.pipeline.LANG_CODES``, read from the installed source. Importing
    the module would leave ``kokoro`` (and torch) in sys.modules, and the node
    tests that pin "importing the node loads no engine library" would then fail
    whenever they run after this file."""
    import ast
    import importlib.util
    try:
        spec = importlib.util.find_spec("kokoro")
    except (ImportError, ValueError):
        spec = None
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("the torch kokoro package is not installed")
    path = os.path.join(list(spec.submodule_search_locations)[0], "pipeline.py")
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(target, "id", "") == "LANG_CODES" for target in node.targets):
            return {kw.arg: ast.literal_eval(kw.value) for kw in node.value.keywords}
    pytest.fail("LANG_CODES not found in %s" % path)


def test_the_espeak_set_is_the_torch_pipelines_own():
    own_g2p = ("a", "b", "j", "z")        # misaki's English, Japanese, Chinese
    assert kb.ESPEAK_LANGUAGES == {
        code: lang for code, lang in _kokoro_lang_codes().items() if code not in own_g2p}
    assert sorted(_ESPEAK_SAMPLES) == sorted(kb.ESPEAK_LANGUAGES.values())


def test_every_non_english_row_is_espeak_or_needs_torch():
    from nodes import _otr_episode_languages as langs
    rows, _by_label, _by_iso = langs.load_registry()
    codes = {row.engines["kokoro"]["lang_code"] for row in rows if row.iso != "en"}
    assert codes - set(kb.ESPEAK_LANGUAGES) == {"j", "z"}


@pytest.mark.parametrize("language", sorted(_ESPEAK_SAMPLES))
def test_the_onnx_phonemes_are_the_torch_pipelines_phonemes(language):
    misaki_espeak = pytest.importorskip("misaki.espeak")
    text = _ESPEAK_SAMPLES[language]
    ours = kb.EspeakPhonemizer(language)(text)
    theirs, _ = misaki_espeak.EspeakG2P(language=language)(text)
    assert ours and ours == theirs


class _StubG2P:
    language = "es"

    def __call__(self, text):
        return "" if text == "..." else "PH<%s>" % text


def test_an_espeak_row_goes_to_kokoro_onnx_as_phonemes():
    stub = _StubKokoro(voices=("ef_dora",))
    backend = _onnx_backend_with_stub(stub)
    backend._g2p = _StubG2P()
    out = backend.synthesize("hola\n...\nadios", "ef_dora", 0.95)
    assert [c[0] for c in stub.calls] == ["PH<hola>", "PH<adios>"]   # no phonemes: skipped
    for _, kw in stub.calls:
        assert kw == {"voice": "ef_dora", "speed": 0.95, "is_phonemes": True, "lang": "es",
                      "trim": False, "sentence_pause": 0.25, "clause_pause": 0.1}
    assert out.size == 200


def _fake_onnx_runtime(monkeypatch):
    class _Session:
        def __init__(self, path, sess_options=None, providers=None):
            self.providers = providers

        def get_providers(self):
            return list(self.providers)

    class _Options:
        intra_op_num_threads = 0

    monkeypatch.setitem(sys.modules, "onnxruntime", types.SimpleNamespace(
        InferenceSession=_Session, SessionOptions=_Options,
        get_available_providers=lambda: ["CPUExecutionProvider"]))
    monkeypatch.setitem(sys.modules, "kokoro_onnx", types.SimpleNamespace(
        Kokoro=types.SimpleNamespace(from_session=lambda s, v: _StubKokoro())))


def test_load_builds_the_row_phonemizer_and_refuses_what_onnx_cannot_speak(monkeypatch):
    _fake_onnx_runtime(monkeypatch)
    built = []
    monkeypatch.setattr(kb, "EspeakPhonemizer",
                        lambda language: built.append(language) or _StubG2P())
    spanish = kb.OnnxKokoroBackend("m.onnx", "v.npz", lang_code="e")
    spanish.load()
    assert built == ["es"] and spanish._g2p is not None
    english = kb.OnnxKokoroBackend("m.onnx", "v.npz")
    english.load()
    assert english._g2p is None and built == ["es"]
    _absent(monkeypatch, "onnxruntime")          # refused before any import
    for code in ("j", "z"):
        with pytest.raises(kb.BackendUnavailable, match="no phonemizer for lang_code '%s'" % code):
            kb.OnnxKokoroBackend("m.onnx", "v.npz", lang_code=code).load()


def test_the_engine_speaks_an_espeak_row_on_onnx(monkeypatch, tmp_path):
    eng = _onnx_engine(monkeypatch, tmp_path)
    monkeypatch.setattr(eng_kokoro, "_spec_present", lambda n: False)   # no torch kokoro
    eng._lang_code = "e"
    eng.load()
    assert eng._backend_name == "onnx"
    assert _StubOnnxBackend.instances[-1].lang_code == "e"


def test_a_cjk_row_without_torch_names_the_python_it_needs(monkeypatch, tmp_path):
    eng = _onnx_engine(monkeypatch, tmp_path)
    monkeypatch.setattr(eng_kokoro, "_spec_present", lambda n: False)
    eng._lang_code = "j"
    with pytest.raises(EngineUnusable) as exc:
        eng.load()
    assert exc.value.reason == EngineUsabilityReason.MISSING_MODEL
    assert "'j'" in str(exc.value) and "Python 3.12" in str(exc.value)


def test_a_cjk_row_moves_to_torch_when_torch_is_installed(monkeypatch, tmp_path):
    eng = _onnx_engine(monkeypatch, tmp_path)
    monkeypatch.setattr(eng_kokoro, "_spec_present", lambda n: True)
    loaded = []

    class _Torch:
        def __init__(self, device, lang_code="b"):
            loaded.append(lang_code)

        def load(self):
            pass

    monkeypatch.setattr(kb, "TorchKokoroBackend", _Torch)
    eng._lang_code = "z"
    eng.load()
    assert eng._backend_name == "torch" and loaded == ["z"]


#: misaki 0.9.4's EspeakG2P on _ESPEAK_SAMPLES (espeak-ng 1.52 through
#: espeakng_loader 0.2.4), recorded 2026-09-28. The parity test above needs
#: misaki, which does not install on Python 3.13 -- the very boxes this path is
#: for -- so these hold the same line there.
_ESPEAK_GOLDEN = {
    'es': 'bw\u02c8enas n\u02c8o\u02a7es. \u02c8esta \u02c8es la se\u0272\u02c8al pe\u027e\xf0\u02c8i\xf0a (d\u02c8i\u03b8e w\u02c8\u025bndi): \u201c\xa1m\u02c8u\u02a7as \u0263\u027e\u02c8a\u03b8jas, \u02a7iw\u02c8awa!\u201d',
    'fr-fr': 'b\u0254\u0303sw\u02c8a\u0281. is\u02c8i l\u0259 sinj\u02c8al p\u025b\u0281d\u02c8y, e w\u02c8\u025bndi \u0281ep\u02c8\u0254\u0303 : \u201c m\u025b\u0281s\u02c8i bok\u02c8u ! \u201d',
    'hi': '\u0283\u02c8\u028ab\u02b0 s\u02c8\u028cnd\u02b0ja\u02d0 j\u02cc\u0259h k\u02b0\u02c8o\u02d0ja\u02d0 h\u02c8\u028aa\u02d0 s\u0259\u014bk\u02c8e\u02d0t h\u025b\u02d0, \u0254\u02d0\u027e \u028b\u02c8e\u0303\u02d0\u0256i k\u02c8\u028ch\u0259t\u02cci h\u025b\u02d0: b\u02c8\u028ch\u028at d\u02b0\u02cc\u0259nj\u0259\u028b\u02c8a\u02d0d!',
    'it': 'bw\u02cc\u0254nas\u02c8era. kw\u02c8esto \u02cc\u025b il se\u0272\u02c8ale perd\u02c8uto (d\u02c8i\u02a7e w\u02c8\u025bnd\u026a): \u0261r\u02c8a\u02a6je m\u02c8ille!',
    'pt-br': 'b\u02c8o\xe6 n\u02c8o\u026a\u02a7y. \u02cces\u02a7y \u025b \u028a sin\u02c8W p\u02cce\u027e\u0259\u02a4\u02c8id\u028a, i w\u02c8A\u014bdi \u02a4\u02c8is: mw\u02c8i\u014btw \u02ccobri\u0261\u02c8ad\u028a!',
}


@pytest.mark.parametrize("language", sorted(_ESPEAK_SAMPLES))
def test_the_onnx_phonemes_match_misakis_recorded_output(language):
    pytest.importorskip("phonemizer")
    loader = pytest.importorskip("espeakng_loader")
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    EspeakWrapper.set_library(loader.get_library_path())
    EspeakWrapper.set_data_path(loader.get_data_path())
    ours = kb.EspeakPhonemizer(language)(_ESPEAK_SAMPLES[language])
    assert ours == _ESPEAK_GOLDEN[language]


class _EchoG2P:
    language = "es"

    def __call__(self, text):
        return text


def test_a_chunk_the_model_has_no_symbol_for_is_skipped_not_fatal():
    # kokoro-onnx refuses a batch with no vocabulary symbol in it; the torch
    # pipeline speaks a quarter second of nothing. A lone inverted question
    # mark must not take the whole line with it.
    stub = _StubKokoro(voices=("ef_dora",))
    stub.tokenizer = types.SimpleNamespace(
        known=lambda ps: "".join(c for c in ps if c not in "\xbf\xa1[]{}"))
    backend = _onnx_backend_with_stub(stub)
    backend._g2p = _EchoG2P()
    backend.synthesize("hola\n\xbf\nadios", "ef_dora", 0.95)
    assert [c[0] for c in stub.calls] == ["hola", "adios"]


def test_an_espeak_row_without_its_phonemizer_never_reads_as_english():
    backend = _onnx_backend_with_stub(_StubKokoro(voices=("ef_dora",)))
    backend.lang_code = "e"
    with pytest.raises(RuntimeError, match="will not read the line as English"):
        backend.synthesize("hola", "ef_dora", 0.95)


def test_a_phonemizer_that_fails_to_build_leaves_the_backend_unloaded(monkeypatch):
    _fake_onnx_runtime(monkeypatch)

    def _broken(language):
        raise RuntimeError("espeak-ng missing")

    monkeypatch.setattr(kb, "EspeakPhonemizer", _broken)
    backend = kb.OnnxKokoroBackend("m.onnx", "v.npz", lang_code="e")
    with pytest.raises(RuntimeError, match="espeak-ng missing"):
        backend.load()
    assert backend._kokoro is None and backend._g2p is None   # a retry loads again


def test_a_forced_onnx_backend_keeps_the_espeak_rows_on_onnx(monkeypatch, tmp_path):
    # OTR_KOKORO_BACKEND=onnx on a box with the torch build: the five espeak
    # rows now stay on ONNX (they went to torch while ONNX was English-only);
    # Japanese and Mandarin still go to torch.
    eng = _onnx_engine(monkeypatch, tmp_path)          # selection forced to onnx
    monkeypatch.setattr(eng_kokoro, "_spec_present", lambda n: True)
    eng._lang_code = "f"
    eng.load()
    assert eng._backend_name == "onnx"
    assert _StubOnnxBackend.instances[-1].lang_code == "f"
