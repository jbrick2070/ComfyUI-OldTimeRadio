"""PBUG-20260825-04, orphan-lifecycle round -- two small, surgical fixes for
the CROSS-PROMPT collision window the three earlier fixes in this PBUG do
not close (that trio makes the exact SAME-PROMPT incident structurally
unreachable, per kibitz r1's grounded judgment; what remains is an orphan
from an already-failed prompt persisting into a later request).

Found by an independent (Fable) cold-take review of the orphan-lifecycle
design, and NOT identified by either mechanical reviewer (Codex, Cursor) in
the same round -- both focused on VRAM contention; this is a sharper hazard:

1. THE CACHE EPOCH. request_slot's Step-9 cache store (and the analogous
   GGUF-path store) writes to the shared LLM_CACHE dict UNCONDITIONALLY once
   load_llm()/backend.load() returns -- with no check for whether the
   CALLER is still wanted. If an abandoned worker's own load call finishes
   successfully AFTER the main thread has already invalidated the cache and
   moved on, that write lands anyway, and a completely unrelated LATER
   request_slot call can take a cache HIT on it -- adopting a model the
   orphan may still be actively generating with in another thread. That is
   not VRAM contention, it is two threads mutating one model's
   generation/KV-cache state concurrently. _CACHE_EPOCH closes this: every
   invalidation path bumps a counter; request_slot snapshots it on entry
   and skips its own cache store if the epoch moved on while it was
   working.

2. THE GENERATION DEADLINE. The codebase already had a working per-token
   deadline check (story_orchestrator.GemmaHeartbeatStreamer.put() raises
   TimeoutError past a thread-local deadline) -- but it lived on a
   different, legacy streamer that the SHARED make_generate_fn transport
   (which NewsCuration/NewsCurationDeep actually use) never wired in. So an
   abandoned worker ran to its FULL max_new_tokens budget -- minutes, at
   slow token rates -- before its thread could unwind. A
   StoppingCriteria-based deadline check, owned by the loader (the shared
   transport every caller already goes through) and set by
   _run_with_timeout, shrinks that to ~1 token.

A round-2 kibitz review of the first cut of both fixes (Codex, reviewing the
finished diff rather than a plan) found the epoch counter was invalidating
its OWN call's wanted store on every ordinary model switch (request_slot
tears its own prior resident model down via unload_llm() before loading the
replacement, in the SAME call whose store follows a few lines later -- a raw
"bump on every clear" counter cannot tell that self-inflicted bump apart
from an external one), that the epoch check and the cache publish were not
atomic with the bump, and that a deadline hit could let generate() return
truncated text as a silent SUCCESS if it raced _run_with_timeout's own
future.result(timeout=...). This file's second half covers those three
corrections directly:

* _detach_and_invalidate_locked() / _publish_cache_entry_if_current() put
  the clear+bump and the check+publish under one lock, and unload_llm()/
  invalidate_cache_no_gpu_teardown() now return the new epoch.
* _DeadlineStoppingCriteria latches (.hit) rather than raising mid-decode
  (mirrors _otr_decode_guard's degeneracy criterion); both make_generate_fn
  and make_polish_generate_fn check .hit after generate() returns and raise
  GenerationDeadlineExceededError instead of returning the truncated text.
* _run_with_timeout catches that exception alongside its own FuturesTimeout
  and routes both through the identical recovery path (cache invalidation +
  _LLMTimeoutWorkflowPause), and no longer swallows a failure to install
  the deadline -- a guard that can silently fail to attach is worse than
  none.

A round-3 kibitz review of THAT fix (Codex, a wiring/integration pass) found
the "self-triggered teardown re-baselines unconditionally on unload_llm()'s
return" shape was itself exploitable: the DECISION to self-unload is made
from an unlocked read a few lines before the actual call ("a different model
is resident"); if an external invalidation races into that exact gap, the
call's snapshot is already stale by the time it reaches its own teardown,
but the unconditional re-baseline would still let it adopt a fresh epoch --
laundering someone else's invalidation into a legitimate-looking
self-triggered one, reopening the exact publish-after-abandonment bug this
whole file exists to close. Also found the cache-HIT lookup (Step 2 /
GGUF-branch reuse check) was several unlocked reads followed by a return,
letting an invalidation land mid-check and produce either a None (violating
request_slot's documented dict-return contract) or a hit against an entry
that no longer exists.

Both are fixed with two new atomic-locked primitives:
* _self_unload(my_epoch, slot=...) -- request_slot's ONLY self-teardown
  entry point now (the public unload_llm() is unconditional and reserved
  for external callers). Atomically claims ownership of my_epoch before
  touching LLM_CACHE or the GPU; a stale claim is a complete no-op and
  returns my_epoch UNCHANGED rather than adopting a new one.
* _try_cache_hit_locked(normalized, slot, gguf_key=... | policy_key=...) --
  the entire hit-check-and-return under one lock acquisition.

Neither fix requires or attempts the larger orphan-occupancy registry
(deferred to a dedicated session -- see docs/PROD_BUG_LOG.md). All of it is
independently testable without CUDA, and provably inert on a box where no
orphan is ever created (the epoch never moves; the deadline is never set).
"""
from __future__ import annotations

import ast
import inspect
import sys
import threading
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nodes import _otr_model_loader as ml  # noqa: E402
from nodes import story_orchestrator as so  # noqa: E402


# --------------------------------------------------------------------------- #
# Fix 1: _CACHE_EPOCH -- basic counter behavior
# --------------------------------------------------------------------------- #

def test_epoch_starts_stable_and_bump_advances_it():
    """Drives the bump through the real production invalidation entry
    point (invalidate_cache_no_gpu_teardown) rather than a test-only bump
    primitive -- there is no production path that bumps the epoch without
    also clearing cache identity, so the tests should not exercise one
    either (r3 kibitz CUT-THESE #2, Codex)."""
    before = ml._current_cache_epoch()
    ml.invalidate_cache_no_gpu_teardown()
    after = ml._current_cache_epoch()
    assert after == before + 1


def test_bump_is_visible_across_threads():
    """The whole point: an orphan on one thread must see a bump made by the
    main thread while it was away doing its own (slow) work. Event-based,
    not sleep-based -- a fixed sleep duration is scheduler-dependent and
    can pass by luck even if the visibility guarantee is broken."""
    snapshotted = threading.Event()
    bumped = threading.Event()
    seen = {}

    def orphan():
        my_epoch = ml._current_cache_epoch()
        snapshotted.set()
        assert bumped.wait(timeout=2), "main thread never bumped"
        seen["still_current"] = ml._current_cache_epoch() == my_epoch

    t = threading.Thread(target=orphan)
    t.start()
    assert snapshotted.wait(timeout=2), "orphan never snapshotted"
    ml.invalidate_cache_no_gpu_teardown()  # main thread invalidates while orphan is "away"
    bumped.set()
    t.join(timeout=2)

    assert seen.get("still_current") is False, (
        "the orphan's snapshotted epoch must no longer match current after "
        "a concurrent bump -- this is the exact check request_slot performs "
        "before its own cache store"
    )


def test_epoch_helpers_are_thread_safe_under_concurrent_bumps():
    """No lost updates: N threads each invalidate once, final count == N."""
    start = ml._current_cache_epoch()
    n = 50
    threads = [
        threading.Thread(target=ml.invalidate_cache_no_gpu_teardown)
        for _ in range(n)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=2)
    assert ml._current_cache_epoch() == start + n


# --------------------------------------------------------------------------- #
# Fix 1, r2 correction: ownership (self-invalidation must not skip the
# call's own wanted store) + atomicity (check+publish must share the bump's
# lock). Both behavioral, not source-string -- this is exactly the bug class
# a source pin cannot catch (the old code also "had" an epoch check; the
# defect was in what specific value it was checked against).
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def _clean_llm_cache():
    """Every test below reads/writes the real module-level LLM_CACHE and
    _CACHE_EPOCH; isolate each test so ordering never matters."""
    saved = dict(ml.LLM_CACHE)
    yield
    ml.LLM_CACHE.clear()
    ml.LLM_CACHE.update(saved)


def test_unload_llm_and_invalidate_return_the_epoch_they_just_set():
    """Both invalidators must hand back a value equal to the epoch they
    themselves just installed -- the whole ownership scheme in request_slot
    depends on this being exact, not approximate."""
    new_epoch = ml.unload_llm()
    assert new_epoch == ml._current_cache_epoch()

    newer_epoch = ml.invalidate_cache_no_gpu_teardown()
    assert newer_epoch == ml._current_cache_epoch()
    assert newer_epoch > new_epoch


def test_self_unload_claims_ownership_and_stores_succeed():
    """THE r2 fix, reproduced directly against _self_unload (the real
    call target request_slot now uses -- NOT the public unload_llm()):
    tearing down ITS OWN prior resident model as ordinary control flow
    (exactly what request_slot does for a GGUF load-config change or a
    cross-model slot transition) must still let the call publish its own
    replacement load afterward."""
    my_epoch = ml._current_cache_epoch()

    my_epoch = ml._self_unload(my_epoch, slot="technical")

    published = ml._publish_cache_entry_if_current(my_epoch, {
        "model_id": "test/self-teardown-model",
        "slot": "technical",
        "cache_entry": {"marker": "self-teardown"},
    })

    assert published is True, (
        "a call's OWN teardown-then-load must not invalidate its own "
        "subsequent store"
    )
    assert ml.LLM_CACHE["model_id"] == "test/self-teardown-model"


def test_self_unload_cannot_launder_a_prior_external_invalidation():
    """THE r3 regression, reproduced directly. This is Codex's exact
    finding: request_slot DECIDES to self-unload from an unlocked read a
    few lines before the actual call ("a different model is resident"). If
    an external invalidation races into the gap between that decision and
    the call, the caller's original snapshot is ALREADY stale by the time
    _self_unload runs -- but _self_unload unconditionally clearing+
    bumping+returning-a-new-epoch (the first cut's shape) would let the
    caller adopt that new epoch anyway, laundering someone else's
    invalidation into a legitimate-looking self-triggered one. The fix:
    _self_unload only proceeds/adopts if the caller's snapshot was STILL
    current at the moment of the atomic claim; otherwise it is a no-op and
    hands back the ORIGINAL (now provably stale) epoch unchanged."""
    my_epoch = ml._current_cache_epoch()  # the call's own entry snapshot

    # An UNRELATED external invalidation races in before the call's own
    # self-unload runs -- e.g. a timeout handler on a different phase.
    ml.invalidate_cache_no_gpu_teardown()

    # The call now runs its own (already-decided) self-unload. It must NOT
    # be allowed to adopt a fresh epoch here -- it was already orphaned.
    result_epoch = ml._self_unload(my_epoch, slot="technical")

    assert result_epoch == my_epoch, (
        "a self-unload whose snapshot was ALREADY stale before it ran must "
        "return the caller's original epoch unchanged, never a freshly "
        "claimed one -- adopting a new epoch here is exactly the r3 "
        "laundering bug"
    )

    # And the caller's eventual publish attempt (using the correctly-
    # unchanged, now-provably-stale epoch) must still be rejected.
    published = ml._publish_cache_entry_if_current(result_epoch, {
        "model_id": "test/laundered-model",
        "slot": "technical",
        "cache_entry": {"marker": "laundered"},
    })
    assert published is False, (
        "if the laundering bug were present, this store would wrongly "
        "succeed -- an abandoned call publishing a model a later, "
        "unrelated caller could take a cache hit on"
    )
    assert ml.LLM_CACHE.get("model_id") != "test/laundered-model"


def test_self_unload_does_not_touch_the_gpu_when_it_cannot_claim():
    """A self-unload that fails to claim ownership must be a complete
    no-op on LLM_CACHE (not just skip re-baselining) -- otherwise it could
    still clear/overwrite a DIFFERENT, legitimately-current entry that
    landed in the same window."""
    my_epoch = ml._current_cache_epoch()
    ml.invalidate_cache_no_gpu_teardown()  # external invalidation races in

    # A different, legitimate caller now publishes a fresh entry at the
    # new (current) epoch -- this must survive the stale call's self_unload.
    current_epoch = ml._current_cache_epoch()
    ml._publish_cache_entry_if_current(current_epoch, {
        "model_id": "test/legitimate-fresh-model",
        "slot": "technical",
        "cache_entry": {"marker": "fresh"},
    })

    ml._self_unload(my_epoch, slot="technical")  # the stale call, still trying

    assert ml.LLM_CACHE.get("model_id") == "test/legitimate-fresh-model", (
        "a self-unload that could not claim ownership must not clear a "
        "different, legitimately-current entry out from under it"
    )


def test_stale_snapshot_without_rebaseline_is_correctly_rejected():
    """Negative control: a call that does NOT re-baseline after its own
    unload_llm() gets its store correctly rejected -- proving the guard
    still does real work and the fixes above aren't just disabling it."""
    my_epoch = ml._current_cache_epoch()

    ml.unload_llm()  # bumps the epoch; caller deliberately does NOT re-baseline

    published = ml._publish_cache_entry_if_current(my_epoch, {
        "model_id": "test/stale-snapshot-model",
        "slot": "technical",
        "cache_entry": {"marker": "stale"},
    })

    assert published is False
    assert ml.LLM_CACHE.get("model_id") != "test/stale-snapshot-model"


def test_a_genuinely_external_invalidation_still_skips_the_store():
    """The scenario the whole mechanism exists for: this call is off doing
    slow work (simulated by simply not touching the epoch), a DIFFERENT
    caller invalidates the cache (a timeout handler, or a concurrent
    request_slot on another slot), and this call's late store must be
    skipped -- it does not own the epoch it snapshotted."""
    my_epoch = ml._current_cache_epoch()

    ml.invalidate_cache_no_gpu_teardown()  # an unrelated caller, e.g. a timeout

    published = ml._publish_cache_entry_if_current(my_epoch, {
        "model_id": "test/orphaned-model",
        "slot": "technical",
        "cache_entry": {"marker": "orphan"},
    })

    assert published is False
    assert ml.LLM_CACHE.get("model_id") != "test/orphaned-model"


def test_cache_hit_lookup_correctly_hits_then_misses_after_invalidation():
    """r3 MUST-FIX #3: the cache-HIT check-then-return must be one atomic
    locked operation (_try_cache_hit_locked), not several unlocked reads
    followed by a return -- otherwise an invalidation landing mid-check can
    make request_slot() return None (violating its documented dict
    contract) or a hit against an entry that no longer exists.

    This test proves SEQUENTIAL correctness (a hit returns the live entry;
    a lookup made AFTER an invalidation has already completed misses
    cleanly) -- it does not itself inject a race DURING one
    _try_cache_hit_locked call (r4 kibitz, Cursor: the earlier docstring
    here overclaimed that). The actual "check and write/read happen inside
    one critical section" property is a structural fact about the function
    body, proven at the source level in
    test_publish_detach_invalidate_and_cache_hit_share_one_lock below --
    that is what makes the sequential behavior asserted here also hold
    under real concurrency, since nothing outside that one lock can ever
    observe or act on a half-completed check."""
    real_entry = {"marker": "still-resident"}
    ml._publish_cache_entry_if_current(ml._current_cache_epoch(), {
        "model_id": "test/hit-target",
        "slot": "technical",
        "cache_entry": real_entry,
        "policy_key": "policy-a",
        "gguf_load_key": "gguf-a",
    })

    # A hit must return the exact live entry.
    hit = ml._try_cache_hit_locked(
        "test/hit-target", "technical", policy_key="policy-a",
    )
    assert hit is real_entry

    # A concurrent invalidation between two attempts must never leave a
    # caller holding a reference to a dict LLM_CACHE no longer owns --
    # each call is a fresh atomic snapshot, so post-invalidation it misses
    # cleanly rather than returning stale data.
    ml.invalidate_cache_no_gpu_teardown()
    miss = ml._try_cache_hit_locked(
        "test/hit-target", "technical", policy_key="policy-a",
    )
    assert miss is None, (
        "a hit-check after invalidation must miss cleanly, never return "
        "the now-cleared entry"
    )


def test_publish_detach_invalidate_and_cache_hit_share_one_lock_each():
    """Source-level pin for the atomicity property: _publish_cache_entry_if_
    current, _detach_and_invalidate_locked, and _try_cache_hit_locked must
    each perform their entire epoch/LLM_CACHE read-or-write under
    _CACHE_EPOCH_LOCK, in ONE `with` block, not as separate acquire/release
    steps a concurrent caller could interleave with. This is a structural
    question ("is there one critical section or two") that source
    inspection answers directly and correctly, the same way this codebase
    already pins StoppingCriteria wiring at the source level
    (test_decode_guard_covers_every_local_route.py) -- and it is what makes
    test_cache_hit_lookup_correctly_hits_then_misses_after_invalidation's
    sequential behavior also hold under real concurrent access: nothing
    outside this one lock can ever observe or act on a half-completed
    check."""
    for fn in (
        ml._publish_cache_entry_if_current,
        ml._detach_and_invalidate_locked,
        ml._try_cache_hit_locked,
    ):
        src = inspect.getsource(fn)
        tree = ast.parse(src)
        func_node = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
        )
        body = func_node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(
            body[0].value, ast.Constant,
        ):
            body = body[1:]  # skip a leading docstring
        # global declarations (e.g. `global _CACHE_EPOCH`) are fine outside
        # the lock -- they touch no shared state by themselves.
        body = [n for n in body if not isinstance(n, ast.Global)]

        assert len(body) == 1 and isinstance(body[0], ast.With), (
            f"{fn.__name__}'s entire epoch read/write must be ONE `with "
            f"_CACHE_EPOCH_LOCK:` block -- any statement outside it is a "
            f"gap a concurrent bump could land in, exactly the race "
            f"MUST-FIX #2 described. Found top-level body: "
            f"{[type(n).__name__ for n in body]}"
        )
        lock_name = ast.dump(body[0].items[0].context_expr)
        assert "_CACHE_EPOCH_LOCK" in lock_name


def test_concurrent_invalidation_during_a_publish_window_is_never_missed():
    """Deterministic two-thread proof of the atomicity property (not just
    the source pin above): a publisher snapshots epoch E0, is paused via an
    Event at the exact moment BEFORE it attempts to publish, an invalidator
    bumps the epoch while the publisher is paused there, and only then is
    the publisher released. The publish must be rejected -- this is the
    real-world shape of the race MUST-FIX #2 described (a concurrent
    invalidation landing in the gap between snapshot and publish)."""
    paused_before_publish = threading.Event()
    invalidated = threading.Event()
    result = {}

    my_epoch = ml._current_cache_epoch()

    def delayed_publisher():
        paused_before_publish.set()
        assert invalidated.wait(timeout=2), "invalidator never ran"
        result["published"] = ml._publish_cache_entry_if_current(my_epoch, {
            "model_id": "test/raced-model",
            "slot": "technical",
            "cache_entry": {"marker": "raced"},
        })

    t = threading.Thread(target=delayed_publisher)
    t.start()
    assert paused_before_publish.wait(timeout=2)
    ml._detach_and_invalidate_locked()
    invalidated.set()
    t.join(timeout=2)

    assert result.get("published") is False
    assert ml.LLM_CACHE.get("model_id") != "test/raced-model"


def test_request_slot_source_routes_self_unload_through_ownership_claim():
    """Source-level pin: every internal self-teardown call inside
    request_slot's own control flow must go through _self_unload (the
    ownership-checked claim), not the public unconditional unload_llm(),
    or that call site reintroduces the r3 laundering regression for its
    specific branch (GGUF load-config change, GGUF slot transition,
    transformers policy change, transformers slot transition)."""
    src = inspect.getsource(ml.request_slot)
    assert "_my_cache_epoch = _current_cache_epoch()" in src, (
        "request_slot must snapshot the epoch before doing any local work"
    )
    rebaseline_count = src.count(
        "_my_cache_epoch = _self_unload(_my_cache_epoch, slot=slot)"
    )
    assert rebaseline_count == 4, (
        f"expected exactly 4 self-triggered _self_unload() call sites (GGUF "
        f"load-config change, GGUF slot transition, transformers policy "
        f"change, transformers slot transition); found {rebaseline_count}. "
        f"A self-unload site that calls the raw unload_llm() instead "
        f"reintroduces the r3 laundering regression for that branch."
    )
    publish_count = src.count("_publish_cache_entry_if_current(_my_cache_epoch,")
    assert publish_count == 2, (
        f"expected exactly 2 epoch-guarded publish calls (GGUF path + "
        f"transformers Step 9); found {publish_count}"
    )
    hit_count = src.count("_try_cache_hit_locked(normalized, slot,")
    assert hit_count == 2, (
        f"expected exactly 2 atomic cache-hit checks (GGUF path + "
        f"transformers Step 2); found {hit_count}. An unlocked multi-read "
        f"hit check reintroduces the r3 cache-hit-atomicity regression."
    )

    # r4 kibitz finding (Cursor): the source-count check above only counts
    # _self_unload call sites -- it never asserted the ABSENCE of the raw,
    # unconditional unload_llm() anywhere else in request_slot's body. That
    # gap is exactly how the load-failure cleanup paths (both the
    # transformers and GGUF load-except blocks) kept calling unload_llm()
    # unconditionally after the success-path sites were fixed: an orphaned
    # call whose load fails could tear down or epoch-bump a completely
    # different, legitimate caller's freshly-published model. AST-walk
    # (not a substring check, which would also match the word inside a
    # comment) and forbid ANY unload_llm( Call node in request_slot's body.
    tree = ast.parse(src)
    request_slot_node = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "request_slot"
    )
    forbidden_calls = [
        node.lineno for node in ast.walk(request_slot_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "unload_llm"
    ]
    assert not forbidden_calls, (
        f"request_slot must never call the raw, unconditional unload_llm() "
        f"-- every self-triggered teardown (success path AND load-failure "
        f"cleanup) must go through the ownership-checked _self_unload, or "
        f"an abandoned call's failure-path cleanup can tear down a "
        f"different, legitimate caller's live model. Found unload_llm( "
        f"call(s) at relative line(s): {forbidden_calls}"
    )


# --------------------------------------------------------------------------- #
# Fix 2: generation deadline -- criterion behavior
# --------------------------------------------------------------------------- #

def test_deadline_criteria_false_when_no_deadline_set():
    ml.set_generation_deadline(None)
    crit = ml._DeadlineStoppingCriteria()
    assert crit(input_ids=None, scores=None) is False
    assert crit.hit is False


def test_deadline_criteria_true_once_past_deadline():
    ml.set_generation_deadline(time.monotonic() - 1.0)  # already in the past
    crit = ml._DeadlineStoppingCriteria()
    try:
        assert crit(input_ids=None, scores=None) is True
        assert crit.hit is True
    finally:
        ml.set_generation_deadline(None)


def test_deadline_criteria_false_before_deadline():
    ml.set_generation_deadline(time.monotonic() + 60.0)  # far in the future
    crit = ml._DeadlineStoppingCriteria()
    try:
        assert crit(input_ids=None, scores=None) is False
        assert crit.hit is False
    finally:
        ml.set_generation_deadline(None)


def test_deadline_criteria_latches_even_after_the_deadline_is_cleared():
    """The criterion instance is asked again on every subsequent decode
    step; once it has latched it must keep returning True even if
    something clears the thread-local deadline mid-generate (which does
    not happen in production, but the LATCH -- not the live deadline value
    -- is what generate()'s loop must see from here on)."""
    ml.set_generation_deadline(time.monotonic() - 1.0)
    crit = ml._DeadlineStoppingCriteria()
    assert crit(input_ids=None, scores=None) is True
    ml.set_generation_deadline(None)
    assert crit(input_ids=None, scores=None) is True, (
        "a latched criterion must keep returning True even once the live "
        "deadline value is gone -- it should never re-evaluate"
    )


def test_deadline_is_per_thread_not_global():
    """A deadline set on one thread must not affect generation on another --
    otherwise one timed-out phase would truncate every OTHER concurrent
    LLM call in the process, including ones with no timeout at all."""
    results = {}

    def other_thread():
        # This thread never calls set_generation_deadline at all.
        crit = ml._DeadlineStoppingCriteria()
        results["other"] = crit(input_ids=None, scores=None)

    ml.set_generation_deadline(time.monotonic() - 1.0)  # expired, on THIS thread
    try:
        t = threading.Thread(target=other_thread)
        t.start()
        t.join(timeout=2)
        assert results.get("other") is False, (
            "a deadline set on one thread leaked into another -- "
            "_GENERATION_DEADLINE must be threading.local(), not a plain "
            "module attribute"
        )
        # And THIS thread's own criteria still correctly reports expired.
        assert ml._DeadlineStoppingCriteria()(input_ids=None, scores=None) is True
    finally:
        ml.set_generation_deadline(None)


def test_both_generate_call_sites_wire_in_the_deadline_criteria():
    """Source-level pin: both model.generate() call sites in
    make_generate_fn/make_polish_generate_fn must include a bound
    _DeadlineStoppingCriteria instance alongside the existing degeneracy
    guard, AND check its .hit after generate() returns, or an abandoned
    worker either keeps running to its full token budget (missing wiring)
    or silently returns truncated text as success (missing the .hit check)."""
    src = inspect.getsource(ml)
    assert src.count("_deadline_guard = _DeadlineStoppingCriteria()") == 2, (
        "expected exactly 2 bound _DeadlineStoppingCriteria instances "
        "(make_generate_fn + make_polish_generate_fn)"
    )
    assert src.count("[_guard, _deadline_guard]") == 2, (
        "expected exactly 2 model.generate() call sites wiring in "
        "_deadline_guard alongside _guard"
    )
    assert src.count("if _deadline_guard.hit:") == 2, (
        "expected exactly 2 post-generate() checks of _deadline_guard.hit -- "
        "without this, a deadline hit returns truncated text as success "
        "instead of raising GenerationDeadlineExceededError"
    )
    assert src.count("raise GenerationDeadlineExceededError(") == 2


def test_run_with_timeout_sets_and_clears_the_loader_deadline():
    """Source-level pin on story_orchestrator._run_with_timeout: it must
    call _otr_model_loader.set_generation_deadline() when starting a worker
    and clear it (None) in that worker's own finally, so the deadline
    cannot outlive the call that set it."""
    src = inspect.getsource(so._run_with_timeout)
    assert "set_generation_deadline(deadline)" in src
    assert "set_generation_deadline(None)" in src
    # 2026-08-25: this used to pin the literal
    # `set_generation_deadline(_TIMEOUT_CTX.deadline)`, which derived the
    # loader deadline from the legacy streamer context INSIDE the worker.
    # There is now ONE absolute deadline computed BEFORE submit and shared by
    # the worker and the parent, because the old shape let the worker's
    # deadline outlive the parent's timeout by the executor's scheduling
    # delay. Pin the new invariants so that skew cannot come back:
    assert "time.monotonic() + timeout_sec" in src, (
        "the deadline must be monotonic and computed from the budget once; "
        "an epoch clock can step and a per-worker recomputation reintroduces "
        "the parent/worker skew"
    )
    assert "deadline - time.monotonic()" in src, (
        "the parent must wait only the REMAINING duration, not a fresh full "
        "timeout_sec measured from after the worker was scheduled"
    )
    assert src.index("deadline = time.monotonic()") < src.index("executor.submit"), (
        "the shared deadline must be computed BEFORE the worker is submitted"
    )


def test_ast_confirms_deadline_clear_is_in_a_finally_block():
    """Stronger than a substring check: the clear must be reachable even if
    fn() raises, or a timed-out (i.e. exactly the exceptional) call would
    leak its deadline onto whatever this worker thread does next."""
    src = inspect.getsource(so._run_with_timeout)
    tree = ast.parse(src)
    worker_fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_worker"
    )
    finally_blocks = [
        n.finalbody for n in ast.walk(worker_fn) if isinstance(n, ast.Try)
    ]
    found = any(
        "set_generation_deadline" in ast.dump(stmt)
        for body in finally_blocks
        for stmt in body
    )
    assert found, (
        "set_generation_deadline(None) must be inside a finally block in "
        "_worker(), not just called unconditionally after fn() -- an "
        "exception from fn() (the timeout case itself) would otherwise "
        "skip the clear"
    )


# --------------------------------------------------------------------------- #
# Fix 2, r2 correction: a deadline hit must raise, never return truncated
# text as success. Behavioral, via a fake model/tokenizer -- no torch, no
# CUDA, no network. Mirrors the established fake-transport harness in
# tests/test_a1_output_limit_raise_carries_evidence.py.
# --------------------------------------------------------------------------- #

class _PromptTensor:
    shape = (1, 12)


class _GeneratedIds:
    def __init__(self, count: int) -> None:
        self.shape = (count,)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key):
        raise AssertionError("deadline path never indexes generated_ids")


class _Output:
    def __init__(self, generated: _GeneratedIds) -> None:
        self._generated = generated

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self._generated
        raise AssertionError("the batch row is taken whole")


class _Inputs(dict):
    def __init__(self) -> None:
        super().__init__(input_ids=_PromptTensor())

    def to(self, _device):
        return self


class _Tokenizer:
    eos_token_id = 0

    def apply_chat_template(self, _messages, **_kwargs):
        return "serialized prompt"

    def __call__(self, _prompt, *, return_tensors):
        assert return_tensors == "pt"
        return _Inputs()

    def decode(self, _tokens, *, skip_special_tokens):
        raise AssertionError(
            "a deadline-exceeded call must raise before ever decoding text"
        )


class _DeadlineHittingModel:
    """Stands in for a real transformers model: 'runs' generate() by
    invoking every passed stopping_criteria once (as the real generation
    loop would on its first step), then returns fixed output -- enough to
    exercise the _deadline_guard.hit latch without a real decode loop."""
    device = "cpu"

    def __init__(self, generated_count: int = 3) -> None:
        self._generated = _GeneratedIds(generated_count)

    def generate(self, **kwargs):
        for criterion in kwargs["stopping_criteria"]:
            criterion(input_ids=None, scores=None)
        return [_Output(self._generated)]


def _make_cache_entry(model):
    return {
        "model": model,
        "tokenizer": _Tokenizer(),
        "model_id": "test/deadline-model",
        "context_cap": 8192,
        "_system_role_supported": True,
    }


def test_make_generate_fn_raises_deadline_exceeded_instead_of_returning_text():
    ml.set_generation_deadline(time.monotonic() - 1.0)  # already expired
    try:
        generate = ml.make_generate_fn(_make_cache_entry(_DeadlineHittingModel()))
        with pytest.raises(ml.GenerationDeadlineExceededError) as caught:
            generate(
                [{"role": "user", "content": "hello"}],
                temperature=0.7,
                max_new_tokens=64,
            )
        assert caught.value.generated_tokens == 3
    finally:
        ml.set_generation_deadline(None)


class _NormalTokenizer(_Tokenizer):
    def decode(self, _tokens, *, skip_special_tokens):
        assert skip_special_tokens is True
        return "ok"


class _NormalModel:
    device = "cpu"

    def generate(self, **kwargs):
        for criterion in kwargs["stopping_criteria"]:
            criterion(input_ids=None, scores=None)
        return [_Output(_GeneratedIds(2))]


def test_make_generate_fn_is_unaffected_with_no_deadline_set():
    """CONTROL: a call with no deadline registered must behave exactly as
    before -- the guard is inert when nobody set a deadline."""
    ml.set_generation_deadline(None)

    entry = _make_cache_entry(_NormalModel())
    entry["tokenizer"] = _NormalTokenizer()
    generate = ml.make_generate_fn(entry)
    result = generate(
        [{"role": "user", "content": "hello"}], temperature=0.7, max_new_tokens=64,
    )
    assert result == "ok"


def test_make_polish_generate_fn_raises_deadline_exceeded_instead_of_returning_text():
    ml.set_generation_deadline(time.monotonic() - 1.0)
    try:
        polish = ml.make_polish_generate_fn(
            _make_cache_entry(_DeadlineHittingModel()),
        )
        with pytest.raises(ml.GenerationDeadlineExceededError):
            polish(
                [{"role": "user", "content": "hello"}],
                temperature=0.4,
                max_new_tokens=64,
            )
    finally:
        ml.set_generation_deadline(None)


# --------------------------------------------------------------------------- #
# Fix 2, r2 correction: _run_with_timeout must route a deadline-exceeded
# worker through the SAME recovery path as its own FuturesTimeout, and must
# not silently no-op when the deadline install itself fails.
# --------------------------------------------------------------------------- #

def test_run_with_timeout_routes_deadline_exceeded_like_futures_timeout(
    monkeypatch,
):
    """The worker's own generate() call raised GenerationDeadlineExceededError
    (simulated directly here, since exercising a real generate() call is
    covered above) -- _run_with_timeout must catch it exactly like its own
    FuturesTimeout: invalidate the cache and raise _LLMTimeoutWorkflowPause,
    never let it surface as a bare, unclassified error."""
    invalidated = {"n": 0}
    monkeypatch.setattr(
        ml, "invalidate_cache_no_gpu_teardown",
        lambda: invalidated.__setitem__("n", invalidated["n"] + 1) or 999,
    )

    def _fn():
        raise ml.GenerationDeadlineExceededError("cut short", generated_tokens=1)

    with pytest.raises(so._LLMTimeoutWorkflowPause) as exc:
        so._run_with_timeout(_fn, timeout_sec=5, phase_label="TEST")

    assert invalidated["n"] == 1
    assert isinstance(exc.value.__cause__, ml.GenerationDeadlineExceededError)


def test_run_with_timeout_fails_loudly_if_deadline_install_fails(monkeypatch):
    """MUST-FIX #4: a construction/install failure for the deadline guard
    must surface, not be swallowed -- a guard that can silently fail to
    attach is worse than none, because the log then claims protection that
    was never actually installed."""
    def _boom(_deadline):
        raise RuntimeError("deadline install exploded")

    monkeypatch.setattr(ml, "set_generation_deadline", _boom)

    with pytest.raises(RuntimeError, match="deadline install exploded"):
        so._run_with_timeout(lambda: "unreachable", timeout_sec=5, phase_label="TEST")
