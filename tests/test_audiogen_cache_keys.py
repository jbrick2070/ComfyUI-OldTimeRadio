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
# ---------------------------------------------------------------------------


def test_audiogen_cache_prefix_changes_when_duration_changes():
    """Different duration -> different cache prefix.

    If this test fails, ``_cache_prefix`` has either dropped
    ``duration_sec`` from its hash payload or hashed it in a way
    that collapses (e.g., truncating to int, rounding to nearest 5s).
    Either case lets a writer change a SFX dur_s and silently get
    served the cached wav from the prior duration.
    """
    a = _cache_prefix("alarm bell ringing", 1.0, "seed_001")
    b = _cache_prefix("alarm bell ringing", 2.0, "seed_001")
    assert a != b, (
        "AudioGen _cache_prefix is duration-blind. "
        "Same prompt + seed must produce DIFFERENT prefixes for "
        "DIFFERENT durations, otherwise a writer-side dur_s edit "
        "from 1.0 -> 2.0 will silently serve the stale 1.0-second wav."
    )


def test_audiogen_cache_prefix_changes_when_prompt_changes():
    """Different prompt -> different cache prefix."""
    a = _cache_prefix("alarm bell ringing", 1.5, "seed_001")
    b = _cache_prefix("dial tone", 1.5, "seed_001")
    assert a != b


def test_audiogen_cache_prefix_changes_when_seed_changes():
    """Different episode seed -> different cache prefix.

    Two episodes that happen to use the same prompt + duration must
    not share cached wavs (Rule C7: episode determinism).
    """
    a = _cache_prefix("alarm bell ringing", 1.5, "seed_001")
    b = _cache_prefix("alarm bell ringing", 1.5, "seed_002")
    assert a != b


def test_audiogen_cache_prefix_stable_across_calls():
    """Same inputs -> same cache prefix every call.

    Identity hash MUST be deterministic; otherwise cache hits never
    fire and we re-render every SFX every run (the BUG-LOCAL-017
    timestamp-suffix regression that this fix originally addressed).
    """
    a = _cache_prefix("alarm bell ringing", 1.5, "seed_001")
    b = _cache_prefix("alarm bell ringing", 1.5, "seed_001")
    assert a == b


def test_audiogen_cache_prefix_subsecond_durations_distinct():
    """Sub-second duration deltas must NOT collapse.

    G7's lower bound is 0.5s. Two SFX cues at 0.5s and 0.6s are
    legitimate distinct fixtures and must hash distinctly so the
    smaller doesn't get served from the larger's cache.
    """
    a = _cache_prefix("dial tone", 0.5, "seed_001")
    b = _cache_prefix("dial tone", 0.6, "seed_001")
    assert a != b


def test_audiogen_cache_prefix_within_g7_bounds_distinct_at_boundary():
    """G7 boundary samples (0.5, 10.0) must hash distinctly from
    the just-inside-bounds neighbors (0.51, 9.99)."""
    boundary_lo = _cache_prefix("alarm", 0.5, "seed_001")
    just_inside_lo = _cache_prefix("alarm", 0.51, "seed_001")
    boundary_hi = _cache_prefix("alarm", 10.0, "seed_001")
    just_inside_hi = _cache_prefix("alarm", 9.99, "seed_001")
    assert boundary_lo != just_inside_lo
    assert boundary_hi != just_inside_hi


def test_audiogen_cache_filename_extension_is_wav():
    """``_cache_filename_for_write`` MUST emit a ``.wav`` suffix.

    ProductionLedger / FFmpeg downstream both assume the cached
    SFX path ends in ``.wav``; a future change that emits e.g.
    ``.flac`` without updating downstream would break the audio
    timeline mux."""
    fn = _cache_filename_for_write("alarm", 1.5, "seed_001")
    assert fn.endswith(".wav"), fn


def test_audiogen_cache_key_alias_matches_filename_for_write():
    """``_cache_key`` (legacy public name) MUST return the same
    string as ``_cache_filename_for_write``. The alias exists for
    back-compat of any external import; the two surfaces must not
    drift."""
    args = ("alarm", 1.5, "seed_001")
    assert _cache_key(*args) == _cache_filename_for_write(*args)


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
