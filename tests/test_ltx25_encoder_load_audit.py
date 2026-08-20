"""The encoder-load acceptance gate must not FAIL OPEN.

WHY THIS FILE EXISTS. `scripts/otr_ltx25_encoder_load_audit.py` is the only
thing that can prove the episode-scoped encoder cache actually works on a live
leg -- the failure mode it guards is silent, because a cache that never hits
renders correctly and just costs what it always cost.

A review lane pointed out that the first version had exactly one check,
``reads > expected``, and therefore **passed a log with no matched lines at
all**: a renamed loader line, a truncated log, or the wrong file entirely would
all have read as a clean run and been quoted as a receipt. It also had no tests,
which is how a gate rots without anyone noticing.

Every case below is a way the gate could lie.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

_AUDIT_PATH = (pathlib.Path(__file__).resolve().parents[1]
               / "scripts" / "otr_ltx25_encoder_load_audit.py")
_spec = importlib.util.spec_from_file_location("otr_encoder_audit", _AUDIT_PATH)
audit_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit_mod)


READ = ("[INFO] gguf qtypes: F16 (2), I16 (5), Q5_K (245), BF16 (346), "
        "Q6_K (88)")
PINNED = "[INFO] [ltx25_video] text encoder pinned to CPU (load=cpu offload=cpu)"
RENDER = "[INFO] [OTR video] ltx25_video PLAN dit=LTX-2.5-Distilled-Q3_K_M.gguf"
OPEN = "[INFO] [ltx25_video] encoder cache scope OPEN (episode-owned)"
CLOSE = "[INFO] [ltx25_video] encoder cache scope CLOSED; the episode's"
HIT = "[INFO] [ltx25_video] encoder cache HIT / negative HIT (scope open)"
MISS = "[INFO] [ltx25_video] encoder cache MISS / negative MISS (scope open)"
DROP = "[WARNING] [ltx25_video] dropping the cached text encoder -- stale"


def _log(tmp_path, lines, name="server.log"):
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)


def _cached_episode(shots=6):
    """What a WORKING cache looks like: one read, then hits."""
    out = [OPEN, RENDER, READ, PINNED, MISS]
    for _ in range(shots - 1):
        out += [RENDER, HIT]
    out.append(CLOSE)
    return out


def _uncached_episode(shots=6):
    """The pre-cache baseline: one read per shot."""
    out = []
    for _ in range(shots):
        out += [RENDER, READ, PINNED, MISS]
    return out


# --- the gate passes only real evidence ------------------------------------ #
def test_a_working_cache_PASSES(tmp_path):
    assert audit_mod.main([_log(tmp_path, _cached_episode())]) == 0


def test_the_PRE_CACHE_shape_fails(tmp_path):
    """The BEFORE picture -- one read per shot -- must never read as a pass."""
    assert audit_mod.main([_log(tmp_path, _uncached_episode())]) == 1


# --- the ways it could FAIL OPEN, which is the point of the file ----------- #
def test_a_log_with_NO_matched_reads_fails(tmp_path):
    """Regex drift or the wrong file. ``reads == 0`` is not ``reads <= 1``.

    This is the exact hole the r3 panel found: with only a ``reads > expected``
    check, this log PASSED and would have been quoted as proof.
    """
    lines = [OPEN] + [RENDER, HIT] * 6 + [CLOSE]
    assert audit_mod.main([_log(tmp_path, lines)]) == 1


def test_a_TRUNCATED_log_with_no_scope_fails(tmp_path):
    lines = [RENDER, READ, PINNED] * 2
    assert audit_mod.main([_log(tmp_path, lines)]) == 1


def test_a_SINGLE_shot_cannot_prove_reuse(tmp_path):
    """One render is one load no matter what; it demonstrates nothing."""
    lines = [OPEN, RENDER, READ, PINNED, MISS, CLOSE]
    assert audit_mod.main([_log(tmp_path, lines)]) == 1


def test_renders_with_ZERO_hits_fail(tmp_path):
    """The silent degradation: correct renders, cache never used."""
    lines = [OPEN] + [RENDER, READ, PINNED, MISS] * 4 + [CLOSE]
    assert audit_mod.main([_log(tmp_path, lines)]) == 1


def test_an_empty_log_fails(tmp_path):
    assert audit_mod.main([_log(tmp_path, ["nothing to see here"])]) == 1


# --- the leak and drift signals -------------------------------------------- #
def test_an_unbalanced_scope_fails(tmp_path):
    """Opened and never closed IS the 8.86 GiB leak, and it is invisible in
    the read/render ratio."""
    lines = _cached_episode()
    lines.remove(CLOSE)
    assert audit_mod.main([_log(tmp_path, lines)]) == 1


def test_a_cache_dropped_every_shot_fails(tmp_path):
    """Renders are correct and the wall clock is unchanged -- exactly the
    outcome that must not read as a pass."""
    lines = [OPEN]
    for _ in range(5):
        lines += [RENDER, DROP, READ, PINNED, MISS]
    lines.append(CLOSE)
    assert audit_mod.main([_log(tmp_path, lines)]) == 1


def test_diverging_loader_signals_fail(tmp_path):
    """If the two independent loader patterns disagree, one has drifted and
    NEITHER count can be trusted."""
    lines = [OPEN, RENDER, READ, READ, PINNED, MISS, RENDER, HIT, CLOSE]
    assert audit_mod.main([_log(tmp_path, lines)]) == 1


# --- the counter itself ----------------------------------------------------- #
def test_the_counts_are_what_they_claim(tmp_path):
    c = audit_mod.audit(_log(tmp_path, _cached_episode(shots=4)))
    assert c["renders"] == 4
    assert c["reads"] == 1
    assert c["pinned"] == 1
    assert c["hits"] == 3 and c["misses"] == 1
    assert c["opens"] == 1 and c["closes"] == 1
    assert c["drops"] == 0


def test_events_are_attributed_to_THEIR_OWN_scope(tmp_path):
    """The correlation the whole instrument turns on."""
    lines = _cached_episode(shots=3) + _cached_episode(shots=5)
    scopes, unscoped = audit_mod.parse_scopes(_log(tmp_path, lines))
    assert len(scopes) == 2
    assert [s["renders"] for s in scopes] == [3, 5]
    assert [s["reads"] for s in scopes] == [1, 1]
    assert all(s["closed"] for s in scopes)
    assert unscoped["renders"] == 0


def test_the_DiT_histogram_is_not_counted_as_the_encoder(tmp_path):
    """The transformer prints its own qtype line. Matching it would count the
    10.7 GiB DiT as an encoder read and make every leg look broken."""
    dit = "[INFO] gguf qtypes: F32 (2603), F16 (306), Q3_K (1072)"
    c = audit_mod.audit(_log(tmp_path, _cached_episode() + [dit] * 5))
    assert c["reads"] == 1, "the DiT histogram leaked into the encoder count"


@pytest.mark.parametrize("episodes,expected", [(1, 0), (2, 0), (5, 1)])
def test_expect_episodes_is_a_MINIMUM_SCOPE_COUNT(tmp_path, episodes, expected):
    """The flag's meaning has now changed TWICE, and both changes were fixes.

    v1 it was the read allowance -- which failed a healthy mixed-engine leg,
    because crossing engines legitimately reopens a scope. v2 it became a floor
    on that allowance. v3 (here) the allowance is gone entirely: each scope is
    gated on its OWN reads, so the only thing left for a flag to say is HOW MANY
    SCOPES MUST EXIST. Asking for five and finding two is now a failure, and
    that is right -- it is the difference between "I saw what I expected" and
    "I saw something".
    """
    lines = _cached_episode() + _cached_episode()
    assert audit_mod.main(
        [_log(tmp_path, lines), "--expect-episodes", str(episodes)]) == expected


def test_a_MIXED_ENGINE_leg_that_reopens_is_not_failed(tmp_path):
    """A checker that fails a HEALTHY run is worse than no checker, because the
    next real failure gets ignored along with it.

    An episode crossing engines (ltx25 -> other -> ltx25) closes the scope at
    the hand-off and opens a fresh one coming back, so it reads the encoder
    twice and that is CORRECT. Gating on ``--expect-episodes 1`` would have
    failed it; the allowance is the observed scope-opening count.
    """
    lines = _cached_episode(shots=3) + _cached_episode(shots=3)
    assert audit_mod.main([_log(tmp_path, lines)]) == 0


def test_but_an_extra_read_WITHOUT_an_extra_scope_still_fails(tmp_path):
    """The allowance follows scope openings, so it cannot be gamed by a stray
    reload inside one scope."""
    lines = _cached_episode(shots=3)
    lines.insert(-1, READ)
    lines.insert(-1, PINNED)
    assert audit_mod.main([_log(tmp_path, lines)]) == 1


# --- per-scope correlation: the r4 findings -------------------------------- #
def test_a_GOOD_scope_cannot_subsidise_a_BAD_one(tmp_path):
    """THE AGGREGATION HOLE. Summing every line into one counter set let a
    scope that reloaded twice pass, because a second EMPTY scope had raised the
    global allowance. Each scope is now judged on its own events."""
    bad = [OPEN, RENDER, MISS, READ, PINNED,
           RENDER, DROP, MISS, READ, PINNED,
           RENDER, HIT, CLOSE]
    empty = [OPEN, CLOSE]
    assert audit_mod.main([_log(tmp_path, bad + empty)]) == 1


def test_an_EMPTY_scope_grants_no_allowance(tmp_path):
    """Reachable for real: the driver opens the scope before building the
    request, so a request-building failure closes an empty one."""
    lines = [OPEN, CLOSE] + _cached_episode(shots=3)
    assert audit_mod.main([_log(tmp_path, lines)]) == 0
    lines_bad = [OPEN, CLOSE, OPEN, RENDER, MISS, READ, PINNED,
                 RENDER, MISS, READ, PINNED, CLOSE]
    assert audit_mod.main([_log(tmp_path, lines_bad)]) == 1


def test_an_UNSCOPED_diagnostic_render_does_not_fail_a_good_episode(tmp_path):
    """A perfect episode must not be failed because the same server log also
    contains the supported unscoped ``render_single`` path, whose load is
    deliberately uncached (`eng_ltx25` documents that)."""
    diagnostic = [RENDER, MISS, READ, PINNED]
    assert audit_mod.main(
        [_log(tmp_path, diagnostic + _cached_episode(shots=4))]) == 0


def test_a_scope_that_never_closes_still_fails(tmp_path):
    lines = _cached_episode(shots=3)
    lines.remove(CLOSE)
    assert audit_mod.main([_log(tmp_path, lines)]) == 1
