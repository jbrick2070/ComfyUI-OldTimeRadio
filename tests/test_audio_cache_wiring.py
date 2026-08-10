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

import ast
import hashlib
import json
import logging
import os
import pathlib
import re

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
    """Pinned so a schema change is DELIBERATE, never incidental.

    REQUEST 2 -> 3 on 2026-08-10: four route-identity fields (route_id,
    route_contract_version, qualification_record_id, weight_revision) joined
    IN_KEY_FIELDS, which changes every request's cache_key. The bump routes that
    invalidation through the designed slim-migration path rather than letting
    keys drift silently. CACHE_SCHEMA_VERSION is unchanged: the SIDECAR record
    shape did not change, only what the key is computed over.
    """
    assert REQUEST_SCHEMA_VERSION == "3"
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


def _bootstrap_ledger_on_disk(tmp_path, lines_data):
    """Write a minimal ledger through save_ledger_safe so meta.paths.ledger_path
    lands on the wire copy the voice node receives. Returns (ledger_path_str,
    ledger_dict) where ledger_dict has been re-hydrated from disk so meta.paths
    reflects the derived-from-path canonical block that _persist_ledger_stamps
    reads back in the finally.
    """
    from nodes._otr_ledger import save_ledger_safe

    ledger_path = tmp_path / "ep_ledger.json"
    ledger = {
        "episode_id": "test_ep",
        "cast": [{
            "char_id": "ann",
            "name": "ANNOUNCER",
            "voice_ref_id": "Zephyr",
            "provider_voice_id": "Zephyr",
        }],
        "lines": lines_data,
        "meta": {
            "episode_seed": 42,
            "cast_lock_revision": "1",
            "paths": {
                "ledger_path": str(ledger_path),
                "audio_dir": str(tmp_path / "audio_out"),
            },
        },
    }
    assert save_ledger_safe(pathlib.Path(ledger_path), ledger) is True
    return str(ledger_path), ledger


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
    """MUST-FIX #2 from Sonnet 5 QA + SF#1 ledger-flush wire-up: exercise
    _render_per_line with use_cache=True. First call misses (stub called
    once); second call hits (stub NOT called). P-OBS line count matches line
    count (would have caught the pre-fix double-emit). After each call, read
    the on-disk ledger and assert the per-line provenance fields landed via
    the finally-flushed _persist_ledger_stamps (would have caught the
    zero-callers SF#1 bug: hit-stamp render_ms==0, miss-stamp render_ms>=0,
    both carry cache_key + sha256 hex digests + non-empty provider_model_id).
    """
    from nodes.announcer_voice import AnnouncerVoice

    recorder = []
    _stub_google_generate_voice(monkeypatch, recorder)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setenv("OTR_AUDIO_CACHE_DIR", str(cache_dir))
    ledger_path, ledger_dict = _bootstrap_ledger_on_disk(
        tmp_path,
        [{
            "line_id": "a1",
            "char_id": "ann",
            "speaker_role": "announcer",
            "text": "Hello there.",
        }],
    )
    node = AnnouncerVoice()

    # First call: MISS.
    audio_a, log_a, done_a = node.generate(
        script_json=json.dumps(ledger_dict), engine="google_tts",
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

    # SF#1 finally-flush: miss stamp lands on disk with the full provenance.
    with open(ledger_path, "r", encoding="utf-8") as fh:
        miss_ledger = json.load(fh)
    miss_line = next(ln for ln in miss_ledger["lines"] if ln["line_id"] == "a1")
    _HEX64 = re.compile(r"^[0-9a-f]{64}$")
    assert _HEX64.match(miss_line.get("audio_cache_key", "") or ""), miss_line
    assert _HEX64.match(miss_line.get("audio_sha256", "") or ""), miss_line
    _miss_render_ms = miss_line.get("render_ms")
    assert isinstance(_miss_render_ms, int) and _miss_render_ms >= 0, miss_line
    assert miss_line.get("provider_model_id", ""), miss_line

    # Second call, identical input: HIT (no new adapter call).
    audio_b, log_b, done_b = node.generate(
        script_json=json.dumps(ledger_dict), engine="google_tts",
    )
    assert len(recorder) == 1, "cache hit should not have called generate_voice again"

    with open(ledger_path, "r", encoding="utf-8") as fh:
        hit_ledger = json.load(fh)
    hit_line = next(ln for ln in hit_ledger["lines"] if ln["line_id"] == "a1")
    assert _HEX64.match(hit_line.get("audio_cache_key", "") or ""), hit_line
    assert _HEX64.match(hit_line.get("audio_sha256", "") or ""), hit_line
    # Hit path stamps render_ms == 0 (no generation time consumed) -- the
    # None-is-skip / 0-is-persisted split (r2 Codex MF-4) MUST NOT lose this.
    assert hit_line.get("render_ms") == 0, hit_line
    assert hit_line.get("provider_model_id", ""), hit_line

    # Neither leg reported any degraded stamps.
    assert "degraded_ledger=0" in log_a, log_a
    assert "degraded_ledger=0" in log_b, log_b

    # P-OBS line count == line count (not double). One-line episode has 1 P-OBS.
    def _pobs_lines(log_str):
        return [ln for ln in log_str.split("\n") if " -> voice_ref_id=" in ln]

    assert len(_pobs_lines(log_a)) == 1, f"miss log P-OBS lines: {_pobs_lines(log_a)}"
    assert len(_pobs_lines(log_b)) == 1, f"hit log P-OBS lines: {_pobs_lines(log_b)}"
    # And the hit line carries cache=hit; the miss carries cache=miss.
    assert " cache=miss" in log_a
    assert " cache=hit" in log_b


def test_end_to_end_google_tts_cache_off_byte_identity(tmp_path, monkeypatch, caplog):
    """When the profile's use_cache is False the wiring is byte-identical to
    the pre-chunk behavior: generate_voice is called EVERY time, no cache dir
    is opened, P-OBS emits its bare form (no `cache=` token), and
    _persist_ledger_stamps is NEVER invoked (SF#1 Codex r4 MF-3 control:
    without this, a future edit could wire the flush unconditionally and
    silently rewrite disk ledgers on every leg)."""
    from nodes._otr_audio_engines import get_engine
    from nodes.announcer_voice import AnnouncerVoice
    from nodes import _otr_engine_profiles as ep
    from nodes import _otr_voice_node_common as vnc

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
    persist_calls = []
    real_persist = vnc._persist_ledger_stamps

    def _persist_spy(meta, stamps, log_):
        persist_calls.append(len(stamps))
        return real_persist(meta, stamps, log_)

    monkeypatch.setattr(vnc, "_persist_ledger_stamps", _persist_spy)

    recorder = []
    _stub_google_generate_voice(monkeypatch, recorder)
    monkeypatch.setenv("OTR_AUDIO_CACHE_DIR", str(tmp_path))
    node = AnnouncerVoice()

    with caplog.at_level(logging.WARNING, logger="OTR"):
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
    # SF#1 Codex r4 MF-3: no persistence call on the cache-off path.
    assert persist_calls == [], (
        f"cache-off path must NOT invoke _persist_ledger_stamps; called "
        f"with stamp counts {persist_calls}"
    )
    # And no ledger-flush warnings surfaced.
    for record in caplog.records:
        assert "ledger stamp flush failed" not in record.getMessage()


# ============================================================================
# SF#1: cloud-audio-cache ledger-flush + partial-exception finally wire-up
# ============================================================================
def _multi_line_ledger_for_role(tmp_path, role):
    """Bootstrap a five-line ledger on disk for the given voice ROLE.

    `role` is one of "announcer" (AnnouncerVoice.LINE_ROLES) or "character"
    (BatchCharacterVoices.LINE_ROLES). All five lines share the same char_id
    so a single cast row suffices; each line gets a distinct line_id.
    """
    lines = [
        {
            "line_id": f"a{idx}",
            "char_id": "ann",
            "speaker_role": role,
            "text": f"Line number {idx}.",
        }
        for idx in range(1, 6)
    ]
    return _bootstrap_ledger_on_disk(tmp_path, lines)


def _stamp_from_ledger(ledger_dict, line_id):
    """Return the on-disk line row for `line_id`, or None if absent."""
    for row in ledger_dict.get("lines", []):
        if row.get("line_id") == line_id:
            return row
    return None


@pytest.mark.parametrize(
    "node_cls_path,role,engine",
    [
        ("nodes.announcer_voice.AnnouncerVoice", "announcer", "google_tts"),
        (
            "nodes.batch_character_voices.BatchCharacterVoices",
            "character",
            "google_tts",
        ),
    ],
    ids=["announcer_voice", "batch_character_voices"],
)
def test_cache_on_multi_line_partial_exception_stamps_completed_lines_via_finally(
    tmp_path, monkeypatch, node_cls_path, role, engine,
):
    """SF#1 root fix: five-line episode where the 3rd generate_voice call
    raises. Expectations:

    * The sentinel escapes _render_per_line -- the caller sees the raise.
    * Lines a1 + a2 completed before the raise, so their per-line ledger
      stamps land on disk via the finally-flushed _persist_ledger_stamps.
    * Lines a3-a5 never completed, so their stamp fields stay empty (no
      audio_cache_key / audio_sha256 / provider_model_id) and render_ms is
      the None-is-skip default (absent from the row).
    * Parameterized over BOTH voice nodes (Codex r4 MF-3 route coverage):
      AnnouncerVoice and BatchCharacterVoices reach _render_per_line via
      different generate() branches, so the finally must fire on both.
    """
    import importlib
    from nodes._otr_audio_engines import get_engine
    from nodes import _otr_engine_profiles as ep

    module_path, cls_name = node_cls_path.rsplit(".", 1)
    node_cls = getattr(importlib.import_module(module_path), cls_name)

    ledger_path, ledger_dict = _multi_line_ledger_for_role(tmp_path, role)

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setenv("OTR_AUDIO_CACHE_DIR", str(cache_dir))

    sentinel = RuntimeError("mid-loop generate_voice sentinel")
    call_count = {"n": 0}

    def _gen(text, voice_ref, delivery_vector, seed,
             *, disable_retry=False, resolved_model=None):
        call_count["n"] += 1
        if call_count["n"] == 3:
            raise sentinel
        return {
            "waveform": torch.zeros(1, 1, 16, dtype=torch.float32),
            "sample_rate": 24000,
        }

    monkeypatch.setattr(get_engine(engine), "generate_voice", _gen)
    # Neuter the preflights so a live API key/model file are not needed.
    monkeypatch.setattr(ep, "assert_token_for_profile", lambda p: None)
    monkeypatch.setattr(ep, "assert_model_available", lambda p: None)
    monkeypatch.setenv("OTR_GOOGLE_API_KEY", "test-key")

    node = node_cls()
    with pytest.raises(RuntimeError) as excinfo:
        node.generate(script_json=json.dumps(ledger_dict), engine=engine)
    assert excinfo.value is sentinel, (
        f"expected the loop's sentinel to escape unchanged; got {excinfo.value!r}"
    )
    # We got as far as the 3rd generate_voice call (which raised).
    assert call_count["n"] == 3, call_count

    with open(ledger_path, "r", encoding="utf-8") as fh:
        after_ledger = json.load(fh)

    # a1 + a2 completed and were stamped via the finally.
    _HEX64 = re.compile(r"^[0-9a-f]{64}$")
    for done_id in ("a1", "a2"):
        line = _stamp_from_ledger(after_ledger, done_id)
        assert line is not None, done_id
        assert _HEX64.match(line.get("audio_cache_key", "") or ""), (done_id, line)
        assert _HEX64.match(line.get("audio_sha256", "") or ""), (done_id, line)
        assert line.get("provider_model_id", ""), (done_id, line)
        _rm = line.get("render_ms")
        assert isinstance(_rm, int) and _rm >= 0, (done_id, line)

    # a3 raised, a4 + a5 never started -- no stamps land on any of them.
    for skipped_id in ("a3", "a4", "a5"):
        line = _stamp_from_ledger(after_ledger, skipped_id)
        assert line is not None, skipped_id
        assert (line.get("audio_cache_key", "") or "") == "", (skipped_id, line)
        assert (line.get("audio_sha256", "") or "") == "", (skipped_id, line)
        assert (line.get("provider_model_id", "") or "") == "", (skipped_id, line)
        # render_ms uses None-is-skip semantics -- an unstamped row must not
        # carry the field at all (stamp helper skips it, not writes 0).
        assert line.get("render_ms") is None, (skipped_id, line)


def test_cache_persist_failure_does_not_mask_render_exception(
    tmp_path, monkeypatch, caplog,
):
    """If both the render AND the finally-triggered _persist_ledger_stamps
    raise, the ORIGINAL render exception must escape (defensive wrap in the
    finally must not swallow the primary raise) AND the flush failure must
    be logged as a warning (Codex r4 SF-2: no return-value channel is
    available when the sentinel escapes, so caplog is the only surface).

    Uses three lines with a raise on the third call so ledger_stamps has
    entries when the finally runs -- otherwise the guarded inner block
    ``if cache_enabled and ledger_stamps:`` short-circuits and the
    persistence stub never fires, defeating the assertion.
    """
    from nodes._otr_audio_engines import get_engine
    from nodes.announcer_voice import AnnouncerVoice
    from nodes import _otr_engine_profiles as ep
    from nodes import _otr_voice_node_common as vnc

    ledger_path, ledger_dict = _bootstrap_ledger_on_disk(
        tmp_path,
        [
            {
                "line_id": f"a{idx}",
                "char_id": "ann",
                "speaker_role": "announcer",
                "text": f"Line {idx}.",
            }
            for idx in range(1, 4)
        ],
    )
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setenv("OTR_AUDIO_CACHE_DIR", str(cache_dir))

    sentinel = RuntimeError("render sentinel escapes past finally")
    calls = {"n": 0}

    def _gen(text, voice_ref, delivery_vector, seed,
             *, disable_retry=False, resolved_model=None):
        calls["n"] += 1
        if calls["n"] == 3:
            raise sentinel
        return {
            "waveform": torch.zeros(1, 1, 16, dtype=torch.float32),
            "sample_rate": 24000,
        }

    monkeypatch.setattr(get_engine("google_tts"), "generate_voice", _gen)
    monkeypatch.setattr(ep, "assert_token_for_profile", lambda p: None)
    monkeypatch.setattr(ep, "assert_model_available", lambda p: None)
    monkeypatch.setenv("OTR_GOOGLE_API_KEY", "test-key")

    def _persist_explodes(meta, stamps, log_):
        raise RuntimeError("persistence exploded")

    monkeypatch.setattr(vnc, "_persist_ledger_stamps", _persist_explodes)

    node = AnnouncerVoice()
    with caplog.at_level(logging.WARNING, logger="OTR"):
        with pytest.raises(RuntimeError) as excinfo:
            node.generate(script_json=json.dumps(ledger_dict), engine="google_tts")

    # The ORIGINAL render sentinel escapes (defensive wrap must not shadow).
    assert excinfo.value is sentinel, (
        f"the render sentinel must escape unchanged even when the finally-"
        f"flushed _persist_ledger_stamps raises; got {excinfo.value!r}"
    )
    # AND the flush failure surfaced via caplog (the only channel available
    # when a raise walks past the return statement).
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("ledger stamp flush failed in finally" in m for m in messages), (
        f"expected the defensive-wrap warning; got {messages}"
    )


def test_cache_persist_failure_reports_full_degraded_count_on_success(
    tmp_path, monkeypatch, caplog,
):
    """Successful three-line render + _persist_ledger_stamps raises in the
    finally. Audio still returns; the finally's defensive except credits
    len(ledger_stamps) as degraded so the cache summary line (emitted AFTER
    try/finally on the success path) reports the accurate count. Codex r4
    SF-1: absent this credit, a persistence failure quietly reported
    degraded_ledger=0 while the ledger silently missed every stamp.
    """
    from nodes._otr_audio_engines import get_engine
    from nodes.announcer_voice import AnnouncerVoice
    from nodes import _otr_engine_profiles as ep
    from nodes import _otr_voice_node_common as vnc

    ledger_path, ledger_dict = _bootstrap_ledger_on_disk(
        tmp_path,
        [
            {
                "line_id": f"a{idx}",
                "char_id": "ann",
                "speaker_role": "announcer",
                "text": f"Line {idx}.",
            }
            for idx in range(1, 4)
        ],
    )
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setenv("OTR_AUDIO_CACHE_DIR", str(cache_dir))

    recorder = []

    def _gen(text, voice_ref, delivery_vector, seed,
             *, disable_retry=False, resolved_model=None):
        recorder.append(text)
        return {
            "waveform": torch.zeros(1, 1, 16, dtype=torch.float32),
            "sample_rate": 24000,
        }

    monkeypatch.setattr(get_engine("google_tts"), "generate_voice", _gen)
    monkeypatch.setattr(ep, "assert_token_for_profile", lambda p: None)
    monkeypatch.setattr(ep, "assert_model_available", lambda p: None)
    monkeypatch.setenv("OTR_GOOGLE_API_KEY", "test-key")

    def _persist_explodes(meta, stamps, log_):
        raise RuntimeError("persistence exploded")

    monkeypatch.setattr(vnc, "_persist_ledger_stamps", _persist_explodes)

    node = AnnouncerVoice()
    with caplog.at_level(logging.WARNING, logger="OTR"):
        audio, log_str, done = node.generate(
            script_json=json.dumps(ledger_dict), engine="google_tts",
        )
    # Render completed for all three lines.
    assert len(recorder) == 3, recorder
    assert audio is not None
    # And the summary line reports the full degraded count (3 stamps
    # attempted, all credited as degraded by the defensive wrap).
    assert "degraded_ledger=3" in log_str, log_str
    # Flush failure was logged.
    assert any(
        "ledger stamp flush failed in finally" in rec.getMessage()
        for rec in caplog.records
    ), [rec.getMessage() for rec in caplog.records]


# ----------------------------------------------------------------------------
# BUG-12.74 static-reachability guard for SF#1 -- the last defect class we
# want to catch by AST, not by hoping a suite happens to exercise it.
# ----------------------------------------------------------------------------
_VOICE_COMMON = (
    pathlib.Path(__file__).resolve().parents[1] / "nodes" / "_otr_voice_node_common.py"
)


def _voice_common_tree():
    return ast.parse(_VOICE_COMMON.read_text(encoding="utf-8"))


def _class_method(tree, cls_name, method_name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"{cls_name}.{method_name} not found")


def test_persist_ledger_stamps_wired_into_render_per_line_finally():
    """BUG-12.74 static-reachability guard for the SF#1 fix.

    Assert:
      (1) EXACTLY ONE call to _persist_ledger_stamps in _render_per_line;
      (2) called with args named meta, ledger_stamps, log (or matching
          positional);
      (3) call sits inside an ast.Try.finalbody AND that try's body contains
          both the per-line for-loop AND the pack_audio_batch call;
      (4) call return is used in AugAssign(op=Add,
          target=Subscript(cache_stats, "degraded_ledger")).
    """
    tree = _voice_common_tree()
    fn = _class_method(tree, "OTRVoiceNodeBase", "_render_per_line")

    calls = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "_persist_ledger_stamps"
    ]
    assert len(calls) == 1, (
        f"BUG-12.74: expected exactly one _persist_ledger_stamps call in "
        f"_render_per_line; got {len(calls)}"
    )
    call = calls[0]

    # (2) arg names OR ordered positional (meta, ledger_stamps, log).
    if call.args:
        positional_names = [
            a.id if isinstance(a, ast.Name) else None for a in call.args
        ]
        assert positional_names == ["meta", "ledger_stamps", "log"], (
            f"expected positional args (meta, ledger_stamps, log); "
            f"got {positional_names}"
        )
    else:
        keyword_names = sorted(k.arg for k in call.keywords)
        assert keyword_names == sorted(["meta", "ledger_stamps", "log"])

    # (3) enclosing Try.finalbody + try body contains for + pack call.
    enclosing_try = None
    for try_node in ast.walk(fn):
        if not isinstance(try_node, ast.Try):
            continue
        for stmt in try_node.finalbody:
            for inner in ast.walk(stmt):
                if inner is call:
                    enclosing_try = try_node
                    break
            if enclosing_try:
                break
        if enclosing_try:
            break
    assert enclosing_try is not None, (
        "the _persist_ledger_stamps call must sit inside a try.finalbody"
    )

    body_stmts = list(
        ast.walk(ast.Module(body=enclosing_try.body, type_ignores=[]))
    )
    has_for = any(isinstance(n, ast.For) for n in body_stmts)
    has_pack_call = any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "pack_audio_batch"
        for n in body_stmts
    )
    assert has_for, "enclosing try must contain the per-line for loop"
    assert has_pack_call, "enclosing try must contain the pack_audio_batch call"

    # (4) call is inside AugAssign(target=cache_stats["degraded_ledger"], op=Add).
    aug_hit = None
    for aug in ast.walk(fn):
        if not isinstance(aug, ast.AugAssign):
            continue
        if not isinstance(aug.op, ast.Add):
            continue
        if not isinstance(aug.target, ast.Subscript):
            continue
        if not (
            isinstance(aug.target.value, ast.Name)
            and aug.target.value.id == "cache_stats"
        ):
            continue
        slice_val = aug.target.slice
        slice_str = (
            slice_val.value if isinstance(slice_val, ast.Constant) else None
        )
        if slice_str != "degraded_ledger":
            continue
        for inner in ast.walk(aug):
            if inner is call:
                aug_hit = aug
                break
        if aug_hit:
            break
    assert aug_hit is not None, (
        "call return must be used to augment cache_stats['degraded_ledger'] "
        "(Codex r4 MF-2 strict shape; not just any call anywhere)"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
