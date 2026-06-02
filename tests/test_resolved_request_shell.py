"""R0a contract tests for the ResolvedVoiceRequest shell, the seed reduction
helper, and the AUDIO-batch contract asserts."""
import dataclasses

import pytest
import torch

from nodes._otr_resolved_request import (
    IGNORED_FIELDS,
    IN_KEY_FIELDS,
    ResolvedVoiceRequest,
    _field_partition_is_complete,
    _seed_to_int64,
    assert_audio_batch_contract,
    empty_audio_batch,
)


# --- _seed_to_int64 -------------------------------------------------------

def test_seed_is_deterministic():
    assert _seed_to_int64("a", 1, b"x") == _seed_to_int64("a", 1, b"x")


def test_seed_in_positive_int64_range():
    for parts in [("a",), (1, 2, 3), ("role", "char", 7), (b"\x00\xff",)]:
        s = _seed_to_int64(*parts)
        assert isinstance(s, int)
        assert 0 <= s < (1 << 63)


def test_seed_order_sensitive():
    assert _seed_to_int64("a", "b") != _seed_to_int64("b", "a")


def test_seed_type_mix_distinct():
    # int 1 and str "1" must not collide into the same seed
    assert _seed_to_int64(1) != _seed_to_int64("1")


def test_seed_bool_distinct_from_int():
    assert _seed_to_int64(True) != _seed_to_int64(1)
    assert _seed_to_int64(False) != _seed_to_int64(0)


# --- ResolvedVoiceRequest identity ---------------------------------------

def test_field_partition_complete():
    assert _field_partition_is_complete()
    declared = {f.name for f in dataclasses.fields(ResolvedVoiceRequest)}
    assert declared == set(IN_KEY_FIELDS) | set(IGNORED_FIELDS)
    assert not (set(IN_KEY_FIELDS) & set(IGNORED_FIELDS))


def test_request_is_frozen():
    r = ResolvedVoiceRequest()
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.role = "narrator"  # type: ignore[misc]


def test_cache_key_stable_and_hex():
    r = ResolvedVoiceRequest(role="character", char_id="c1", line_id="L1")
    assert r.cache_key == r.cache_key
    assert len(r.cache_key) == 64
    int(r.cache_key, 16)  # valid hex


def test_cache_key_changes_with_in_key_field():
    base = ResolvedVoiceRequest(role="character", char_id="c1")
    other = dataclasses.replace(base, char_id="c2")
    assert base.cache_key != other.cache_key


def test_cache_key_ignores_non_identity_fields():
    base = ResolvedVoiceRequest(role="character", char_id="c1")
    # prepared_text (debug), engine_seed (derived), allowed_for_release (gate)
    noisy = dataclasses.replace(
        base,
        prepared_text="the quick brown fox",
        engine_seed=999999,
        allowed_for_release=True,
    )
    assert base.cache_key == noisy.cache_key


def test_in_key_dict_excludes_ignored():
    keys = set(ResolvedVoiceRequest().in_key_dict().keys())
    assert keys == set(IN_KEY_FIELDS)
    for ign in IGNORED_FIELDS:
        assert ign not in keys


def test_quantized_params_participate_in_key():
    a = ResolvedVoiceRequest(quantized_params=(("temp", 70), ("cfg", 30)))
    b = ResolvedVoiceRequest(quantized_params=(("temp", 71), ("cfg", 30)))
    assert a.cache_key != b.cache_key


# --- AUDIO-batch contract -------------------------------------------------

def test_empty_audio_batch_shape_and_passes_contract():
    eb = empty_audio_batch()
    assert eb["waveform"].shape == (1, 1, 0)
    assert eb["sample_rate"] == 24000
    assert assert_audio_batch_contract(eb) is eb


def test_valid_batch_passes():
    a = {"waveform": torch.zeros(1, 1, 16), "sample_rate": 48000}
    assert assert_audio_batch_contract(a, where="SceneSequencer") is a


def test_contract_rejects_non_dict():
    with pytest.raises(TypeError):
        assert_audio_batch_contract(torch.zeros(1, 1, 4))


def test_contract_rejects_missing_keys():
    with pytest.raises(ValueError):
        assert_audio_batch_contract({"waveform": torch.zeros(1, 1, 4)})


def test_contract_rejects_wrong_rank():
    with pytest.raises(ValueError):
        assert_audio_batch_contract({"waveform": torch.zeros(1, 4), "sample_rate": 24000})


def test_contract_rejects_bad_sample_rate():
    with pytest.raises(ValueError):
        assert_audio_batch_contract({"waveform": torch.zeros(1, 1, 4), "sample_rate": 0})
    with pytest.raises(ValueError):
        assert_audio_batch_contract({"waveform": torch.zeros(1, 1, 4), "sample_rate": True})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
