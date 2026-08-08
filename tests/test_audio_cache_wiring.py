"""Cloud-audio-cache chunk 2 (2026-08-08) wiring tests.

Covers the get/put/load cycle, atomic writes, corruption handling, adapter
identity hook, retry-control kwarg, ledger stamp reload-before-save, and the
IS_CHANGED classmethod. Every test uses an isolated ``tmp_path`` cache dir via
``OTR_AUDIO_CACHE_DIR`` so no artifacts leak into the operator's canonical
output tree.

Non-goals covered:
- Production release-gate wiring (not in this chunk).
- Bounded same-model 429 retry (not in this chunk).
- Local engine cache activation (profile-opt-in; local profiles default off).
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib

import numpy as np
import pytest
import torch

from nodes._otr_audio_cache import (
    AudioCacheRecord,
    CACHE_SCHEMA_VERSION,
    FileAudioCache,
    cache_key_for,
    record_from_request,
)
from nodes._otr_resolved_request import (
    IN_KEY_FIELDS,
    REQUEST_SCHEMA_VERSION,
    build_resolved_request,
)


def _req(**over):
    base = dict(
        role="announcer_voice",
        engine_name="google_tts",
        engine_profile_id="announcer_google_tts_v1",
        engine_impl_version="1",
        char_id="ann",
        line_id="L1",
        prepared_text="Hello there.",
        voice_ref_id="Zephyr",
        commercial_clean=True,
        sample_rate=24000,
        channels=1,
        provider_model_id="gemini-2.5-flash-preview-tts",
        provider_voice_id="Zephyr",
    )
    base.update(over)
    return build_resolved_request(**base)


def _audio(n_samples=16, sr=24000, channels=1):
    return {
        "waveform": torch.zeros(1, channels, n_samples, dtype=torch.float32),
        "sample_rate": sr,
    }


# ----------------------------------------------------------------------------
# In_KEY exposure + schema versions
# ----------------------------------------------------------------------------
def test_provider_fields_in_key():
    assert "provider_model_id" in IN_KEY_FIELDS
    assert "provider_voice_id" in IN_KEY_FIELDS


def test_schema_versions_bumped():
    assert REQUEST_SCHEMA_VERSION == "2"
    assert CACHE_SCHEMA_VERSION == "2"


def test_model_flip_changes_cache_key():
    a = _req(provider_model_id="gemini-2.5-flash-preview-tts")
    b = _req(provider_model_id="gemini-2.5-pro-preview-tts")
    assert a.cache_key != b.cache_key


def test_voice_flip_changes_cache_key():
    a = _req(provider_voice_id="Zephyr")
    b = _req(provider_voice_id="Charon")
    assert a.cache_key != b.cache_key


# ----------------------------------------------------------------------------
# FileAudioCache.load: verified miss on every corruption path
# ----------------------------------------------------------------------------
def test_put_load_roundtrip(tmp_path):
    cache = FileAudioCache(str(tmp_path))
    req = _req()
    a = _audio()
    cache.put(req, a, allowed_for_release=True, actual_sample_rate=24000,
              provider_model_id="gemini-2.5-flash-preview-tts")
    loaded = cache.load(req)
    assert loaded is not None
    audio, record = loaded
    assert torch.allclose(audio["waveform"], a["waveform"])
    assert audio["sample_rate"] == 24000
    assert record.provider_model_id == "gemini-2.5-flash-preview-tts"
    assert record.actual_sample_rate == 24000


def test_load_returns_none_on_missing_sidecar(tmp_path):
    cache = FileAudioCache(str(tmp_path))
    assert cache.load(_req()) is None


def test_load_returns_none_when_npy_missing(tmp_path):
    cache = FileAudioCache(str(tmp_path))
    req = _req()
    cache.put(req, _audio(), allowed_for_release=True, actual_sample_rate=24000)
    # Delete the .npy leaving the sidecar behind.
    key = cache_key_for(req)
    os.unlink(str(tmp_path / f"{key}.npy"))
    assert cache.load(req) is None


def test_load_returns_none_on_sha_mismatch(tmp_path):
    cache = FileAudioCache(str(tmp_path))
    req = _req()
    cache.put(req, _audio(), allowed_for_release=True, actual_sample_rate=24000)
    key = cache_key_for(req)
    # Tamper the npy so the hash no longer matches.
    np.save(str(tmp_path / f"{key}.npy"), np.ones((1, 1, 16), dtype=np.float32))
    assert cache.load(req) is None


def test_load_returns_none_on_rate_mismatch(tmp_path):
    cache = FileAudioCache(str(tmp_path))
    req = _req(sample_rate=24000)
    cache.put(req, _audio(sr=24000), allowed_for_release=True, actual_sample_rate=24000)
    other = _req(sample_rate=22050)
    assert cache.load(other) is None


def test_load_returns_none_on_channels_mismatch(tmp_path):
    cache = FileAudioCache(str(tmp_path))
    req = _req(channels=1)
    cache.put(req, _audio(channels=1), allowed_for_release=True, actual_sample_rate=24000)
    other = _req(channels=2)
    assert cache.load(other) is None


def test_load_returns_none_on_cache_key_mismatch(tmp_path):
    cache = FileAudioCache(str(tmp_path))
    req = _req()
    cache.put(req, _audio(), allowed_for_release=True, actual_sample_rate=24000)
    key = cache_key_for(req)
    # Rewrite the sidecar to claim a different cache_key.
    sidecar = tmp_path / f"{key}.json"
    doc = json.loads(sidecar.read_text(encoding="utf-8"))
    doc["cache_key"] = "different_key"
    sidecar.write_text(json.dumps(doc), encoding="utf-8")
    assert cache.load(req) is None


def test_load_ignores_sidecar_audio_path_and_derives_from_cache_dir(tmp_path):
    """r4 Fable gate SF#3: sidecar audio_path is forensics-only; load derives
    <cache_dir>/<key>.npy so a rename of the cache dir does not invalidate
    hits. Rewriting the sidecar's audio_path to a bogus location must NOT
    turn a valid cache entry into a miss."""
    cache = FileAudioCache(str(tmp_path))
    req = _req()
    cache.put(req, _audio(), allowed_for_release=True, actual_sample_rate=24000)
    key = cache_key_for(req)
    sidecar = tmp_path / f"{key}.json"
    doc = json.loads(sidecar.read_text(encoding="utf-8"))
    doc["audio_path"] = "/nonexistent/lied/about/path.npy"
    sidecar.write_text(json.dumps(doc), encoding="utf-8")
    loaded = cache.load(req)
    assert loaded is not None, "load should derive the real path from cache_dir"


def test_load_survives_cache_dir_rename(tmp_path):
    """r4 Fable gate SF#3: rename the cache dir (simulating rename_episode);
    the sidecar+npy pair moves together and load still returns the audio."""
    old_dir = tmp_path / "old_ep"
    old_dir.mkdir()
    cache = FileAudioCache(str(old_dir))
    req = _req()
    cache.put(req, _audio(), allowed_for_release=True, actual_sample_rate=24000)
    # Rename the whole cache dir.
    new_dir = tmp_path / "renamed_ep"
    old_dir.rename(new_dir)
    # Point a new cache at the new location; the same request must hit.
    cache2 = FileAudioCache(str(new_dir))
    loaded = cache2.load(req)
    assert loaded is not None
    audio, record = loaded
    assert torch.allclose(audio["waveform"], _audio()["waveform"])


# ----------------------------------------------------------------------------
# Atomic write survives partial crash
# ----------------------------------------------------------------------------
def test_atomic_write_survives_partial_crash(monkeypatch, tmp_path):
    cache = FileAudioCache(str(tmp_path))
    req = _req()

    def _boom(*args, **kwargs):
        raise IOError("simulated crash")

    monkeypatch.setattr(np, "save", _boom)
    with pytest.raises(IOError):
        cache.put(req, _audio(), allowed_for_release=True, actual_sample_rate=24000)
    # No sidecar (sidecar is published LAST), no orphan .npy, no leftover .tmp.
    files = list(tmp_path.iterdir())
    for f in files:
        assert not f.name.endswith(".npy")
        assert not f.name.endswith(".json")


# ----------------------------------------------------------------------------
# GoogleTTSVoice adapter -- identity_params + disable_retry + resolved_model
# ----------------------------------------------------------------------------
def test_google_identity_params_returns_model_and_voice():
    from nodes._otr_audio_engines.eng_google_tts import GoogleTTSVoice

    g = GoogleTTSVoice()
    p = g.identity_params(
        resolved_voice_ref="Charon",
        resolved_model="gemini-2.5-pro-preview-tts",
    )
    assert p == {"model": "gemini-2.5-pro-preview-tts", "provider_voice": "Charon"}


def test_google_identity_params_falls_back_to_env(monkeypatch):
    from nodes._otr_audio_engines import eng_google_tts as _egt

    monkeypatch.setenv("OTR_GOOGLE_TTS_MODEL_ID", "gemini-3.1-flash-tts-preview")
    g = _egt.GoogleTTSVoice()
    p = g.identity_params(resolved_voice_ref="Zephyr")
    assert p["model"] == "gemini-3.1-flash-tts-preview"


def test_models_to_try_allow_retry_false_returns_single():
    from nodes._otr_audio_engines.eng_google_tts import _models_to_try

    got = _models_to_try("gemini-2.5-flash-preview-tts", allow_retry=False)
    assert got == ("gemini-2.5-flash-preview-tts",)


def test_generate_voice_accepts_disable_retry_and_resolved_model(monkeypatch):
    """The signature accepts the new kwargs and threads resolved_model into
    the API call so identity and actual match. Mock _post_interaction to
    capture the model_id used."""
    from nodes._otr_audio_engines import eng_google_tts as _egt

    monkeypatch.setenv("OTR_GOOGLE_API_KEY", "test-key")
    captured = []

    def _mock_post(api_key, payload):
        captured.append(payload["model"])
        # Return a minimal audio response the extractor handles.
        return {"output_audio": {
            "data": "AAA=",  # 3-byte base64 => 2 bytes decoded, but len%2==0
            "sample_rate": 24000,
            "mime_type": "audio/l16",
        }}

    monkeypatch.setattr(_egt, "_post_interaction", _mock_post)
    g = _egt.GoogleTTSVoice()
    try:
        g.generate_voice(
            "hello", "Zephyr", None, 42,
            disable_retry=True, resolved_model="gemini-2.5-pro-preview-tts",
        )
    except _egt.GoogleTTSError:
        pass  # 2-byte payload may or may not parse; we only care about the model captured
    # Assert exactly one attempt was made, and it used the resolved_model.
    assert captured == ["gemini-2.5-pro-preview-tts"]


# ----------------------------------------------------------------------------
# IS_CHANGED classmethod on OTRVoiceNodeBase
# ----------------------------------------------------------------------------
def _is_nan(x):
    return isinstance(x, float) and x != x


def test_IS_CHANGED_returns_static_when_use_cache_false():
    from nodes.batch_character_voices import BatchCharacterVoices

    got = BatchCharacterVoices.IS_CHANGED(engine="bark")
    assert got == "static"


def test_IS_CHANGED_returns_nan_when_use_cache_true():
    from nodes.announcer_voice import AnnouncerVoice

    got = AnnouncerVoice.IS_CHANGED(engine="google_tts")
    assert _is_nan(got)


def test_IS_CHANGED_returns_nan_on_resolution_failure():
    from nodes.batch_character_voices import BatchCharacterVoices

    # Unknown engine name -> resolver raises -> fail open with NaN.
    got = BatchCharacterVoices.IS_CHANGED(engine="not_a_real_engine")
    assert _is_nan(got)


# ----------------------------------------------------------------------------
# _audio_cache_dir_for helper
# ----------------------------------------------------------------------------
def test_audio_cache_dir_env_override(tmp_path, monkeypatch):
    from nodes._otr_voice_node_common import _audio_cache_dir_for

    monkeypatch.setenv("OTR_AUDIO_CACHE_DIR", str(tmp_path))
    got = _audio_cache_dir_for({})
    assert pathlib.Path(got) == tmp_path.resolve()


def test_audio_cache_dir_from_meta_paths(monkeypatch, tmp_path):
    from nodes._otr_voice_node_common import _audio_cache_dir_for

    monkeypatch.delenv("OTR_AUDIO_CACHE_DIR", raising=False)
    meta = {"paths": {"audio_dir": str(tmp_path / "epX")}}
    got = _audio_cache_dir_for(meta)
    assert got.endswith(os.sep + "audio_cache")


def test_audio_cache_dir_empty_when_no_source(monkeypatch):
    from nodes._otr_voice_node_common import _audio_cache_dir_for

    monkeypatch.delenv("OTR_AUDIO_CACHE_DIR", raising=False)
    assert _audio_cache_dir_for({}) == ""
    assert _audio_cache_dir_for({"paths": {}}) == ""


# ----------------------------------------------------------------------------
# _persist_ledger_stamps reload-before-save preserves prior role stamps
# ----------------------------------------------------------------------------
def test_persist_ledger_stamps_preserves_prior_role_stamps(tmp_path):
    """Character stamps land, announcer stamps land, both survive the
    two-write sequence because _persist_ledger_stamps reloads the ledger
    from disk before patching."""
    import logging

    from nodes._otr_voice_node_common import _persist_ledger_stamps
    from nodes._otr_ledger import save_ledger_safe

    ledger_path = tmp_path / "ep_ledger.json"
    ledger = {
        "schema_version": "l4-2026-08-07",
        "lines": [
            {"line_id": "l001", "char_id": "c01"},
            {"line_id": "l002", "char_id": "ann"},
        ],
        "meta": {"episode_id": "ep_test"},
    }
    # Bootstrap: write ledger through save_ledger_safe so meta.paths gets
    # populated with paths.ledger_path pointing at this file.
    save_ledger_safe(pathlib.Path(ledger_path), ledger)
    # After bootstrap, meta.paths.ledger_path is set. Simulate a per-role
    # stamp cycle: character role stamps l001; then announcer role stamps l002.
    meta = {"paths": {"ledger_path": str(ledger_path)}}
    log = logging.getLogger("test")

    # Character role stamps l001 first.
    stamps_char = [("l001", {
        "tts_engine": "google_tts",
        "audio_cache_key": "char_key_1",
        "audio_sha256": "char_sha_1",
    })]
    degraded = _persist_ledger_stamps(meta, stamps_char, log)
    assert degraded == 0

    # Announcer role runs next; must NOT clobber the character stamp.
    stamps_ann = [("l002", {
        "tts_engine": "google_tts",
        "audio_cache_key": "ann_key_1",
        "audio_sha256": "ann_sha_1",
    })]
    degraded = _persist_ledger_stamps(meta, stamps_ann, log)
    assert degraded == 0

    # Both roles' stamps survive on disk.
    with open(ledger_path, "r", encoding="utf-8") as fh:
        final = json.load(fh)
    line_map = {ln["line_id"]: ln for ln in final["lines"]}
    assert line_map["l001"].get("audio_cache_key") == "char_key_1"
    assert line_map["l001"].get("audio_sha256") == "char_sha_1"
    assert line_map["l002"].get("audio_cache_key") == "ann_key_1"
    assert line_map["l002"].get("audio_sha256") == "ann_sha_1"


def test_persist_ledger_stamps_missing_path_returns_all_degraded(tmp_path):
    import logging

    from nodes._otr_voice_node_common import _persist_ledger_stamps

    meta = {"paths": {"ledger_path": ""}}
    log = logging.getLogger("test")
    degraded = _persist_ledger_stamps(
        meta, [("l1", {"tts_engine": "x"})], log,
    )
    assert degraded == 1


# ----------------------------------------------------------------------------
# ASCII-only source guard (matches the sibling on _otr_audio_cache.py)
# ----------------------------------------------------------------------------
def test_source_is_ascii_no_em_dash():
    # em-dash is U+2014; kept as a codepoint literal here so this file itself
    # stays fully ASCII (grep for em-dash on this file returns 0 hits).
    em_dash = chr(0x2014)
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    for rel in [
        "nodes/_otr_voice_node_common.py",
        "nodes/_otr_audio_cache.py",
        "nodes/_otr_audio_engines/eng_google_tts.py",
    ]:
        src = (repo_root / rel).read_text(encoding="utf-8")
        assert em_dash not in src, f"em-dash in {rel}"
        src.encode("ascii")


# ----------------------------------------------------------------------------
# End-to-end wiring: AnnouncerVoice + google_tts + use_cache=True
# ----------------------------------------------------------------------------
def _one_line_google_ledger():
    """Minimal ledger driving a google_tts announcer render for one line."""
    return json.dumps({
        "schema_version": "l4-2026-08-07",
        "cast": [{
            "char_id": "ann",
            "name": "ANNOUNCER",
            "voice_ref_id": "Zephyr",
        }],
        "lines": [{
            "line_id": "a1",
            "char_id": "ann",
            "speaker_role": "announcer",
            "text": "Hello there.",
        }],
        "meta": {
            "episode_seed": "seed-1",
            "cast_lock_revision": 1,
        },
    })


def _stub_google_generate_voice(monkeypatch, recorder):
    """Point the singleton google_tts adapter at a stub that records calls."""
    from nodes._otr_audio_engines import get_engine
    from nodes import _otr_engine_profiles as ep

    g = get_engine("google_tts")

    def _gen(text, voice_ref, delivery_vector, seed, *, disable_retry=False, resolved_model=None):
        recorder.append({
            "text": text, "voice_ref": voice_ref, "seed": seed,
            "disable_retry": disable_retry, "resolved_model": resolved_model,
        })
        return {"waveform": torch.zeros(1, 1, 16, dtype=torch.float32),
                "sample_rate": 24000}

    monkeypatch.setattr(g, "generate_voice", _gen)
    # Neuter the profile preflights so we do not need a live API key or model file.
    monkeypatch.setattr(ep, "assert_token_for_profile", lambda p: None)
    monkeypatch.setattr(ep, "assert_model_available", lambda p: None)
    monkeypatch.setenv("OTR_GOOGLE_API_KEY", "test-key")
    return g


def test_end_to_end_google_tts_cache_miss_then_hit(tmp_path, monkeypatch):
    """MUST-FIX #2 from Sonnet 5 QA: exercise _render_per_line with
    use_cache=True. First call misses (stub called once); second call hits
    (stub NOT called). P-OBS line count matches line count (would have
    caught the pre-fix double-emit)."""
    from nodes.announcer_voice import AnnouncerVoice

    recorder = []
    _stub_google_generate_voice(monkeypatch, recorder)
    monkeypatch.setenv("OTR_AUDIO_CACHE_DIR", str(tmp_path))
    node = AnnouncerVoice()

    # First call: MISS.
    audio_a, log_a, done_a = node.generate(
        script_json=_one_line_google_ledger(), engine="google_tts",
    )
    assert len(recorder) == 1
    assert recorder[0]["disable_retry"] is True
    # resolved_model comes from identity_params()["model"] which reads env or
    # profile default -- the default profile default is gemini-2.5-flash-preview-tts.
    assert recorder[0]["resolved_model"] in (
        "gemini-2.5-flash-preview-tts",
        "gemini-3.1-flash-tts-preview",
        "gemini-2.5-pro-preview-tts",
    )

    # Second call, identical input: HIT (no new adapter call).
    audio_b, log_b, done_b = node.generate(
        script_json=_one_line_google_ledger(), engine="google_tts",
    )
    assert len(recorder) == 1, "cache hit should not have called generate_voice again"

    # P-OBS line count == line count (not double). One-line episode has 1 P-OBS.
    def _pobs_lines(log_str):
        return [ln for ln in log_str.split("\n") if " -> voice_ref_id=" in ln]

    assert len(_pobs_lines(log_a)) == 1, f"miss log P-OBS lines: {_pobs_lines(log_a)}"
    assert len(_pobs_lines(log_b)) == 1, f"hit log P-OBS lines: {_pobs_lines(log_b)}"
    # And the hit line carries cache=hit; the miss carries cache=miss.
    assert " cache=miss" in log_a
    assert " cache=hit" in log_b


def test_end_to_end_google_tts_cache_off_byte_identity(tmp_path, monkeypatch):
    """When the profile's use_cache is False the wiring is byte-identical to
    the pre-chunk behavior: generate_voice is called EVERY time, no cache dir
    is opened, P-OBS emits its bare form (no `cache=` token)."""
    from nodes._otr_audio_engines import get_engine
    from nodes.announcer_voice import AnnouncerVoice
    from nodes import _otr_engine_profiles as ep

    # Force the resolver to return a profile with use_cache=False even though
    # the YAML has True -- this simulates "the operator did not opt in".
    real_resolve = ep.EngineProfileResolver.resolve_casting_plan

    def _mock_resolve(self, *, role, engine, voice_bank=None):
        p = real_resolve(self, role=role, engine=engine, voice_bank=voice_bank)
        # Rebuild without use_cache using pydantic model_copy.
        return p.model_copy(update={"use_cache": False})

    monkeypatch.setattr(
        ep.EngineProfileResolver, "resolve_casting_plan", _mock_resolve,
    )
    recorder = []
    _stub_google_generate_voice(monkeypatch, recorder)
    monkeypatch.setenv("OTR_AUDIO_CACHE_DIR", str(tmp_path))
    node = AnnouncerVoice()

    node.generate(
        script_json=_one_line_google_ledger(), engine="google_tts",
    )
    node.generate(
        script_json=_one_line_google_ledger(), engine="google_tts",
    )
    assert len(recorder) == 2, "cache off: adapter must be called both times"
    # No cache dir should have been populated.
    npy_files = [f for f in tmp_path.iterdir() if f.name.endswith(".npy")]
    assert not npy_files, f"cache off should not write to disk, found: {npy_files}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
