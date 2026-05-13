"""S17.1 (IMP-11) + S17.3 + S17.4 drift guards for MusicGen cache keys.

Mirrors tests/test_audiogen_cache_keys.py -- the two cache surfaces
must follow the same contract (S12.3 standard + every output-
determining input hashed in canonical form).
"""
from __future__ import annotations

import pytest


def _import_cache_prefix():
    """Lazy import so test collection works even if optional deps
    (torch / transformers) blow up at import time. We only need the
    pure-Python cache_prefix function."""
    from nodes.musicgen_theme import _cache_prefix
    return _cache_prefix


def _base_kwargs() -> dict:
    return {
        "cue_id": "opening",
        "prompt": "1940s noir radio drama style, brass + strings",
        "duration_sec": 10.0,
        "episode_seed": "ep001",
        "model_id": "facebook/musicgen-medium",
        "guidance_scale": 3.0,
    }


def test_cache_prefix_length_is_12():
    """S12.4-style pin: the SHA suffix is 12 hex chars."""
    prefix = _import_cache_prefix()(**_base_kwargs())
    # Format: ``<cue_id>_<sha12>`` -> the part after the LAST underscore is the digest.
    digest = prefix.rsplit("_", 1)[-1]
    assert len(digest) == 12, (
        f"S17.1 contract: sha12 (12 hex chars). Got {len(digest)}."
    )


def test_cache_prefix_changes_on_model_id():
    """Different model_id must produce different prefixes."""
    cp = _import_cache_prefix()
    base = _base_kwargs()
    p1 = cp(**base)
    p2 = cp(**{**base, "model_id": "facebook/musicgen-large"})
    assert p1 != p2, (
        "S17.1: switching MusicGen models must invalidate cache."
    )


def test_cache_prefix_changes_on_guidance_scale():
    """Different guidance_scale must produce different prefixes."""
    cp = _import_cache_prefix()
    base = _base_kwargs()
    p1 = cp(**base)
    p2 = cp(**{**base, "guidance_scale": 5.0})
    assert p1 != p2, (
        "S17.1: switching CFG must invalidate cache."
    )


def test_cache_prefix_changes_on_cue_id():
    """Different cue_id must produce different prefixes."""
    cp = _import_cache_prefix()
    base = _base_kwargs()
    p1 = cp(**base)
    p2 = cp(**{**base, "cue_id": "closing"})
    assert p1 != p2


def test_cache_prefix_float_canonical_duration():
    """duration_sec is .3f canonical -- 0.1+0.2 == 0.3 cache identity."""
    cp = _import_cache_prefix()
    base = _base_kwargs()
    p_a = cp(**{**base, "duration_sec": 0.1 + 0.2})
    p_b = cp(**{**base, "duration_sec": 0.3})
    assert p_a == p_b, (
        "S17.1: duration_sec float-canonical (.3f) must collapse "
        "IEEE-754 drift."
    )


def test_cache_prefix_guidance_scale_2dp_canonical():
    """guidance_scale is .2f canonical."""
    cp = _import_cache_prefix()
    base = _base_kwargs()
    p_a = cp(**{**base, "guidance_scale": 3.121})
    p_b = cp(**{**base, "guidance_scale": 3.124})
    assert p_a == p_b, "guidance_scale must round to .2f."
    p_c = cp(**{**base, "guidance_scale": 3.13})
    assert p_a != p_c, "a 0.01 bump must change the key."


def test_cache_prefix_keyword_only_signature():
    """Positional calls must raise TypeError (intentional contract)."""
    cp = _import_cache_prefix()
    with pytest.raises(TypeError):
        cp("opening", "prompt", 10.0, "seed", "model", 3.0)


def test_cache_prefix_starts_with_cue_id():
    """Format pin: <cue_id>_<sha12>. cue_id readable up front."""
    cp = _import_cache_prefix()
    base = _base_kwargs()
    p = cp(**base)
    assert p.startswith("opening_"), p


# -------------------------------------------------------------------
# S17.4 -- episode_seed defensive coercion at the node boundary.
# -------------------------------------------------------------------


def test_episode_seed_coercion_does_not_raise_in_cache_prefix():
    """The cache_prefix path str()s episode_seed; a non-string at the
    cache layer must not raise. The node boundary coerces before
    reaching here, but pinning cache_prefix robustness is cheap."""
    cp = _import_cache_prefix()
    base = _base_kwargs()
    # Even if a future caller bypasses the node and passes weird
    # types directly to cache_prefix, the JSON-canonical str()
    # coercion in cache_prefix shouldn't crash.
    p1 = cp(**{**base, "episode_seed": "42"})
    p2 = cp(**{**base, "episode_seed": 42})  # int
    # Note: these will produce different prefixes because "42" != 42
    # after str() -- str("42") == "42", str(42) == "42", so actually
    # they're the same. Pin that.
    assert p1 == p2, "str(42) == str('42'); prefixes must match"
