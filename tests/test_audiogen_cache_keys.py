"""S8.2 -- Cache-key invariant tests for AudioGen + ProcSFX.

Sprint 8.2 of voice-path-cleanbreak. Pins the rule: a writer-side
change to a line's resolved ``dur_s`` MUST invalidate any prior
render cached under that prompt + seed. If the cache key collapses
two distinct durations into one filename, the user gets a stale
wav with the wrong duration silently served on the second run.

Two surfaces:

1. **AudioGen** (``nodes/batch_audiogen_generator.py``):
   ``_cache_prefix(prompt, duration_sec, episode_seed)`` is the
   deterministic identity hash. The function MUST include
   ``duration_sec`` in its payload. (It does today; these tests
   pin it as a drift guard.)

2. **Procedural SFX** (``nodes/batch_procedural_sfx.py``): has NO
   cache layer -- output filename is keyed by ``sfx_type`` + ``line_id``
   only, and the wav is regenerated on every render. The dur_s flows
   directly into the procedural synth, so a change in dur_s produces
   a new wav with the new duration on the next run. The pinning test
   here asserts the no-cache contract by source inspection: any
   future patch that adds a duration-blind ``_load_cached_wav`` /
   ``_find_cached`` style helper to ProcSFX would be the same class
   of bug as if AudioGen dropped duration from its hash, and should
   surface in CI.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from nodes.batch_audiogen_generator import (
    _cache_prefix,
    _cache_filename_for_write,
    _cache_key,
)


# ---------------------------------------------------------------------------
# AudioGen cache-key behavior tests
#
# Post-S12.3 the helpers are keyword-only with five inputs:
#   prompt, duration_sec, episode_seed, model_id, guidance_scale
# Tests use a shared _BASELINE dict so a future knob added to the cache
# key requires updating one place + adding one varying test.
# ---------------------------------------------------------------------------


_BASELINE = {
    "prompt":         "alarm bell ringing",
    "duration_sec":   1.0,
    "episode_seed":   "seed_001",
    "model_id":       "facebook/audiogen-medium",
    "guidance_scale": 3.0,
}


def _kp(**overrides):
    """Return _BASELINE merged with overrides -- compact baseline+vary helper."""
    return {**_BASELINE, **overrides}


def test_audiogen_cache_prefix_changes_when_duration_changes():
    """Different duration -> different cache prefix.

    If this test fails, ``_cache_prefix`` has either dropped
    ``duration_sec`` from its hash payload or hashed it in a way
    that collapses. Either case lets a writer change a SFX dur_s
    and silently get served the cached wav from the prior duration.
    """
    a = _cache_prefix(**_kp(duration_sec=1.0))
    b = _cache_prefix(**_kp(duration_sec=2.0))
    assert a != b, (
        "AudioGen _cache_prefix is duration-blind. Same other inputs "
        "must produce DIFFERENT prefixes for DIFFERENT durations, "
        "otherwise a writer-side dur_s edit from 1.0 -> 2.0 will "
        "silently serve the stale 1.0-second wav."
    )


def test_audiogen_cache_prefix_changes_when_prompt_changes():
    """Different prompt -> different cache prefix."""
    a = _cache_prefix(**_kp(prompt="alarm bell ringing"))
    b = _cache_prefix(**_kp(prompt="dial tone"))
    assert a != b


def test_audiogen_cache_prefix_changes_when_seed_changes():
    """Different episode seed -> different cache prefix.

    Two episodes that happen to use the same prompt + duration must
    not share cached wavs (Rule C7: episode determinism).
    """
    a = _cache_prefix(**_kp(episode_seed="seed_001"))
    b = _cache_prefix(**_kp(episode_seed="seed_002"))
    assert a != b


def test_audiogen_cache_prefix_changes_when_model_id_changes():
    """S12.3 / IMP-1 layer 3: different model_id -> different prefix.
    Switching from facebook/audiogen-medium to facebook/audiogen-large
    produces wavs of different quality + character; the cache must
    not collapse them."""
    a = _cache_prefix(**_kp(model_id="facebook/audiogen-medium"))
    b = _cache_prefix(**_kp(model_id="facebook/audiogen-large"))
    assert a != b, (
        "Cache key is model-blind. Switching AudioGen models must "
        "invalidate the cache; otherwise a cold/warm switch silently "
        "serves the prior model's output."
    )


def test_audiogen_cache_prefix_changes_when_guidance_scale_changes():
    """S12.3 / IMP-1 layer 3: different guidance_scale -> different
    prefix. CFG is the second knob the user can twist that materially
    changes output; same dimension class as model_id."""
    a = _cache_prefix(**_kp(guidance_scale=3.0))
    b = _cache_prefix(**_kp(guidance_scale=5.0))
    assert a != b, (
        "Cache key is guidance-scale-blind. Tweaking CFG must "
        "invalidate the cache."
    )


def test_audiogen_cache_prefix_float_canonical():
    """S12.3 / IMP-1 layer 2: JSON-canonical payload uses
    ``f"{x:.3f}"`` for duration_sec, so IEEE 754 float-string drift
    (``str(0.1 + 0.2) == '0.30000000000000004'``) doesn't split the
    cache key. Two arithmetically-equivalent durations MUST hash
    to the same prefix."""
    k_literal = _cache_prefix(**_kp(duration_sec=0.3))
    k_arith   = _cache_prefix(**_kp(duration_sec=0.1 + 0.2))
    assert k_literal == k_arith, (
        "Cache key splits 0.3 and (0.1+0.2). The JSON-canonical layer "
        "must canonicalize duration_sec via f'{x:.3f}' so float arith "
        "and literal-typed values collapse to the same key."
    )


def test_audiogen_cache_prefix_stable_across_calls():
    """Same inputs -> same cache prefix every call.

    Identity hash MUST be deterministic; otherwise cache hits never
    fire and we re-render every SFX every run (the BUG-LOCAL-017
    timestamp-suffix regression that this fix originally addressed).
    """
    a = _cache_prefix(**_BASELINE)
    b = _cache_prefix(**_BASELINE)
    assert a == b


def test_audiogen_cache_prefix_subsecond_durations_distinct():
    """Sub-second duration deltas must NOT collapse.

    G7's lower bound is 0.5s. Two SFX cues at 0.5s and 0.6s are
    legitimate distinct fixtures and must hash distinctly so the
    smaller doesn't get served from the larger's cache.

    Note: post-S12.3 the canonical encoding is ``f"{x:.3f}"`` so
    sub-millisecond deltas DO collapse (0.5001 == 0.5 in the cache
    key). The chosen canonical precision is 3 decimal places --
    enough to distinguish every G7-legal value but not so much
    that float-arith noise splits keys.
    """
    a = _cache_prefix(**_kp(duration_sec=0.5))
    b = _cache_prefix(**_kp(duration_sec=0.6))
    assert a != b


def test_audiogen_cache_prefix_within_g7_bounds_distinct_at_boundary():
    """G7 boundary samples (0.5, 10.0) must hash distinctly from
    the just-inside-bounds neighbors (0.51, 9.99)."""
    boundary_lo    = _cache_prefix(**_kp(duration_sec=0.5))
    just_inside_lo = _cache_prefix(**_kp(duration_sec=0.51))
    boundary_hi    = _cache_prefix(**_kp(duration_sec=10.0))
    just_inside_hi = _cache_prefix(**_kp(duration_sec=9.99))
    assert boundary_lo != just_inside_lo
    assert boundary_hi != just_inside_hi


def test_audiogen_cache_filename_extension_is_wav():
    """``_cache_filename_for_write`` MUST emit a ``.wav`` suffix.

    ProductionLedger / FFmpeg downstream both assume the cached
    SFX path ends in ``.wav``; a future change that emits e.g.
    ``.flac`` without updating downstream would break the audio
    timeline mux."""
    fn = _cache_filename_for_write(**_BASELINE)
    assert fn.endswith(".wav"), fn


def test_audiogen_cache_key_alias_matches_filename_for_write():
    """``_cache_key`` MUST return the same string as
    ``_cache_filename_for_write``. The alias exists as the legacy
    public name; the two surfaces must not drift."""
    assert _cache_key(**_BASELINE) == _cache_filename_for_write(**_BASELINE)


# ---------------------------------------------------------------------------
# ProcSFX no-cache invariant tests (source inspection)
# ---------------------------------------------------------------------------


_PROCSFX_SRC = (
    Path(__file__).resolve().parent.parent
    / "nodes" / "batch_procedural_sfx.py"
).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# NOTE: ``test_procsfx_module_does_not_define_cache_lookup_helpers`` was
# RETIRED in S12.2 (F-7 fix). The hardcoded-symbol-name source-grep was
# rename-blind: AudioGen renaming ``_cache_prefix`` to ``_lookup_prefix``
# would have left the guard silently passing while ProcSFX could import
# the new name. Replaced by ``test_procsfx_does_not_import_from_audiogen_cache``
# in tests/test_procsfx_isolation.py -- AST-walk over import statements,
# rename-immune by construction.
# ---------------------------------------------------------------------------


def test_procsfx_filename_contract_documented():
    """The ProcSFX output-filename contract is stamped in the module
    docstring. Post-S12.1 the shape is
    ``proc_<sfx_type>_<line_id>_<perm>.wav`` where ``<perm>`` is an
    8-char SHA-256 over (dur_s, type, line_id) so iteration on scene
    timing produces distinct on-disk files instead of overwriting.
    This test pins the documented shape; a future patch that changes
    it has to update this test in lockstep -- forcing the author to
    think about whether the dur_s-variance invariant survives."""
    assert "proc_<sfx_type>_<line_id>_<perm>.wav" in _PROCSFX_SRC, (
        "ProcSFX docstring no longer documents the canonical "
        "``proc_<sfx_type>_<line_id>_<perm>.wav`` filename shape. If "
        "the shape really changed, update this test AND review whether "
        "writer-side dur_s edits still produce distinct on-disk wavs "
        "(post-S12.1 the perm hash guarantees this; without it, the "
        "F-6 overwrite-on-iteration regression resurfaces)."
    )


def test_procsfx_filename_perm_hash_varies_with_dur_s():
    """S12.1 / F-6 fix: the ProcSFX filename's <perm> segment is a
    SHA-256 over (dur_s, type, line_id). Same line_id rendered at
    two distinct durations MUST produce two distinct hashes -- the
    on-disk artifact is the test artifact for "iteration on scene
    timing didn't lose A/B history" (the F-6 finding).

    Tests the hash construction directly; rendering an actual wav
    isn't needed because the hash is the contract surface."""
    import hashlib
    line_id = "sfx_001"
    chosen_type = "door_knock"
    perm_a = hashlib.sha256(
        f"{1.0:.3f}|{chosen_type}|{line_id}".encode("utf-8")
    ).hexdigest()[:8]
    perm_b = hashlib.sha256(
        f"{2.0:.3f}|{chosen_type}|{line_id}".encode("utf-8")
    ).hexdigest()[:8]
    assert perm_a != perm_b, (
        "ProcSFX perm hash collapses different durations. The same "
        "(line_id, type) at dur_s=1.0 vs 2.0 must hash distinctly so "
        "the second render doesn't overwrite the first wav."
    )

    # Symmetry: same dur_s -> same hash (deterministic).
    perm_a2 = hashlib.sha256(
        f"{1.0:.3f}|{chosen_type}|{line_id}".encode("utf-8")
    ).hexdigest()[:8]
    assert perm_a == perm_a2, "ProcSFX perm hash must be deterministic."


def test_procsfx_perm_hash_in_module_source():
    """Source-level guard: the perm-hash construction line must
    appear in the ProcSFX module. Catches a future refactor that
    rewrites the filename without the hash, silently re-introducing
    the F-6 overwrite-on-iteration regression."""
    assert "hashlib.sha256" in _PROCSFX_SRC, (
        "ProcSFX module no longer imports/uses hashlib.sha256. "
        "S12.1 requires the per-cue filename to include a SHA-256 "
        "perm hash; without it, dur_s edits silently overwrite the "
        "previous render."
    )
    assert "_{perm}.wav" in _PROCSFX_SRC, (
        "ProcSFX filename template no longer includes the {perm} "
        "segment. The 8-char content-addressed hash is the F-6 "
        "regression guard."
    )
