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


def test_procsfx_module_does_not_define_cache_lookup_helpers():
    """ProcSFX has no cache layer by design (procedural is cheap +
    deterministic). If a future patch introduces a cache-lookup
    helper, that helper MUST include dur_s in its key -- the same
    invariant AudioGen pins above. This test fires when such a
    helper appears so the author has to wire dur_s into it
    deliberately, with their own test."""
    forbidden_names = (
        "_find_cached",       # AudioGen's lookup; ProcSFX shouldn't grow one
        "_load_cached_wav",   # AudioGen's wav reader
        "_cache_prefix",      # AudioGen's identity hash
        "_cache_key",         # AudioGen's legacy alias
    )
    hits = []
    for name in forbidden_names:
        # Match ``def <name>(`` -- a definition, not a use site.
        if re.search(rf"^def\s+{re.escape(name)}\s*\(", _PROCSFX_SRC, re.MULTILINE):
            hits.append(name)
    assert not hits, (
        "ProcSFX has grown one or more cache-lookup helpers: "
        f"{hits}. If this is intentional, the helper MUST include the "
        "line's resolved dur_s in its identity (otherwise a writer-side "
        "dur_s edit will serve a stale wav). Add a duration-aware test "
        "to test_audiogen_cache_keys.py and remove the symbol(s) from "
        "the forbidden list above."
    )


def test_procsfx_filename_contract_documented():
    """The ProcSFX output-filename contract is stamped in the module
    docstring (line ~13). This test pins the contract text so a future
    patch that changes the filename shape (e.g., adds a duration
    suffix, drops the line_id, switches to a hash) has to update this
    test in lockstep -- forcing the author to think about whether the
    dur_s invariant is preserved."""
    assert "proc_<sfx_type>_<line_id>.wav" in _PROCSFX_SRC, (
        "ProcSFX docstring no longer documents the canonical "
        "``proc_<sfx_type>_<line_id>.wav`` filename shape. If the "
        "shape really changed, update this test AND review whether "
        "writer-side dur_s edits still produce fresh wavs on the "
        "next run (they do today because there's no cache)."
    )
