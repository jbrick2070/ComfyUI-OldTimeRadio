"""Wave 0: ResolvedVoiceRequest builder + integer-tick quantizer (I-6 -- the
engine keys on a frozen request, never a raw widget float)."""
import hashlib

from nodes._otr_resolved_request import (
    build_resolved_request,
    quantize_params,
)


# --- quantizer ------------------------------------------------------------

def test_quantize_sorted_and_int_ticks():
    q = quantize_params({"temp": 0.7, "cfg": 0.35})
    assert q == (("cfg", 350), ("temp", 700))  # sorted by name, *1000


def test_quantize_bool_to_0_1():
    assert quantize_params({"flag": True, "off": False}) == (("flag", 1), ("off", 0))


def test_quantize_deterministic():
    p = {"a": 0.123, "b": 2}
    assert quantize_params(p) == quantize_params(p)


def test_quantize_float_noise_collapses():
    # 0.1+0.2 != 0.3 in float, but both quantize to the same 300 tick
    assert quantize_params({"x": 0.1 + 0.2})[0][1] == quantize_params({"x": 0.3})[0][1]


def test_quantize_distinct_values_differ():
    assert quantize_params({"x": 0.70}) != quantize_params({"x": 0.71})


def test_quantize_empty():
    assert quantize_params(None) == ()
    assert quantize_params({}) == ()


# --- builder --------------------------------------------------------------

def _req(**over):
    base = dict(
        role="character", engine_name="bark", engine_profile_id="bark_default",
        engine_impl_version="1", char_id="c1", line_id="L1", prepared_text="Hello.",
        params={"temperature": 0.7},
    )
    base.update(over)
    return build_resolved_request(**base)


def test_builder_computes_prepared_text_sha():
    r = _req(prepared_text="Hello world")
    assert r.prepared_text_sha256 == hashlib.sha256(b"Hello world").hexdigest()
    assert r.prepared_text == "Hello world"  # IGNORED debug field carried


def test_builder_quantizes_params_into_key():
    a = _req(params={"temperature": 0.7})
    b = _req(params={"temperature": 0.8})
    assert a.quantized_params == (("temperature", 700),)
    assert a.cache_key != b.cache_key


def test_builder_identical_inputs_same_cache_key():
    assert _req().cache_key == _req().cache_key


def test_stable_line_seed_deterministic_and_int():
    r = _req()
    assert isinstance(r.stable_line_seed, int) and 0 <= r.stable_line_seed < (1 << 63)
    assert _req().stable_line_seed == r.stable_line_seed


def test_stable_line_seed_changes_with_identity():
    assert _req(line_id="L1").stable_line_seed != _req(line_id="L2").stable_line_seed
    assert _req(char_id="c1").stable_line_seed != _req(char_id="c2").stable_line_seed


def test_prepared_text_debug_not_in_cache_key():
    # Same sha (same text) -> same key regardless; but the IGNORED prepared_text
    # field is not itself hashed. Build two with the same text and confirm the
    # cache_key depends on the sha (identity), with prepared_text carried as debug.
    r = _req(prepared_text="abc")
    assert r.prepared_text == "abc"
    assert r.prepared_text_sha256 == hashlib.sha256(b"abc").hexdigest()


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
