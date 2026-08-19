"""The stale-ledger voice guard is gone -- and the log line meant to replace it
COULD NOT BE ADDED. That second half is the finding worth keeping.

WHAT THE OPERATOR VOTED (2026-08-19). The partial-wiring sweep found
`assert_registry_ledger_has_voice_ref_id` (`nodes/_otr_audio_cache.py`)
written, documented, tested, and called by ZERO production code. He ruled:
delete it. That was right, and for a better reason than "unused":

    the guard RAISED. It refused to render a stale ledger.

THE LAW says an audit may improve a story and may never FAIL one. A guard that
rejects an episode because its ledger predates a field is the same shape as
PBUG-20260729-03 -- a bound that refuses instead of degrading. Wiring it up
would have shipped a fresh instance of a defect class this repo has already
been burned by. So it is deleted, and this file pins that it stays deleted.

THE PROBLEM IT WAS AIMED AT IS REAL AND IS STILL OPEN:

  * `_resolve_clone_ref_path` and `_resolve_provider_voice_id`
    (`nodes/_otr_voice_node_common.py`) read `cast["voice_ref_id"]` and, when
    it is missing, assign a voice DETERMINISTICALLY BY GENDER instead.
  * Measured 2026-08-19: **51 of 683 cast-locked ledgers on disk carry at
    least one cast row with no `voice_ref_id`.**
  * Re-render one and the character comes back as a different person, with no
    log line anywhere saying so.

THE OBVIOUS FIX IS BLOCKED, AND HERE IS WHY -- DO NOT REDISCOVER THIS THE HARD
WAY. The driver added exactly that missing warning to
`_otr_voice_node_common.py`. One comment block and one `log.warning`, purely
additive, no logic touched. It turned FOUR tests red across
`test_voice_identity_fix.py` and `test_cast_lock_policy_repin.py`.

The cause is not the warning. It is that `nodes/_otr_voice_node_common.py` is
one of the four files hashed by
``_otr_voice_route.RUNTIME_FINGERPRINT_SOURCES["indextts2"]``. A qualified
voice route records the fingerprint of the code that will render it; when the
live hash stops matching, ``select_policy_route`` withdraws the route and the
voice falls back to an ordinary draw. **A comment changed the hash, so Lemmy
lost his qualified Cockney.** The gate did its job perfectly.

So the trade was: make a rare stale-replay degrade audible, at the cost of
de-qualifying the shipped voice route on EVERY episode until someone
re-auditions. That is a bad trade, and the warning was reverted. What replaces
it is this file -- the constraint, written down where the next person to try
will find it.
"""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIO_CACHE = REPO_ROOT / "nodes" / "_otr_audio_cache.py"
VOICE_ROUTE = REPO_ROOT / "nodes" / "_otr_voice_route.py"


# --------------------------------------------------------------------------
# The guard is gone, and must stay gone
# --------------------------------------------------------------------------

def test_the_raising_guard_is_gone_and_stays_gone():
    """It refused a render, and THE LAW forbids failing a story.

    If this symbol comes back, whoever restores it should have to read this
    test and decide again rather than rediscovering the ruling.
    """
    src = AUDIO_CACHE.read_text(encoding="utf-8")
    assert "assert_registry_ledger_has_voice_ref_id" not in src
    assert "CacheMigrationError" not in src, (
        "the error class existed only so that guard could refuse a render"
    )


def test_no_production_code_CALLS_OR_IMPORTS_the_deleted_guard():
    """AST, not text search, and the difference is deliberate.

    A COMMENT naming the guard is exactly what should survive -- explaining
    why something was removed is the point of a comment. A grep-based version
    of this test failed on such comments, which would pressure a future reader
    into deleting the explanation to get green. AST ignores comments by
    construction, so this asserts the real invariant: nothing CALLS or IMPORTS
    it.
    """
    dead = "assert_registry_ledger_has_voice_ref_id"
    offenders = []
    for sub in ("nodes", "scripts"):
        for path in (REPO_ROOT / sub).rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id == dead:
                    offenders.append("%s:%d (reference)" % (path.name, node.lineno))
                elif isinstance(node, ast.Attribute) and node.attr == dead:
                    offenders.append("%s:%d (attribute)" % (path.name, node.lineno))
                elif isinstance(node, ast.ImportFrom) and any(
                    a.name == dead for a in node.names
                ):
                    offenders.append("%s:%d (import)" % (path.name, node.lineno))
    assert not offenders, "deleted guard still wired in: %s" % offenders


# --------------------------------------------------------------------------
# The constraint that blocked the replacement -- pinned so it is discoverable
# --------------------------------------------------------------------------

def test_the_voice_resolver_is_OUT_of_the_indextts2_fingerprint():
    """INVERTED 2026-08-19, and the inversion is the story.

    This test was written earlier the same day asserting the OPPOSITE -- that
    `_otr_voice_node_common.py` IS inside the fingerprint -- with a failure
    message reading: *"either the protection was weakened, or this constraint
    is finally lifted and the stale-ledger warning can now be added; check
    which."* It then fired on the recipe change and forced exactly that
    check. It is the answer, so it is now pinned the other way.

    WHY THE FILE LEFT THE RECIPE, measured rather than argued:
      * 19 commits in 60 days, because it is shared dispatch code.
      * Of those 19, exactly ONE touched the seed path that was the stated
        reason for including it (`62fb6a1f`, the voice-identity fix).
      * So the whole-file hash produced 18 false demotions and 1 true one.
      * And `62fb6a1f` ALSO edited `eng_indextts2.py`, which stays in the
        recipe -- so narrowing loses nothing on the only real event in the
        window.

    THE RESIDUAL RISK IS REAL AND IS ACCEPTED: a seed-path change touching no
    engine-specific file would now escape this fingerprint. It did not happen
    once in 60 days. `weight_revision` and `reference.source_ref_sha256` still
    gate independently.
    """
    from nodes import _otr_voice_route as ROUTE

    sources = ROUTE.RUNTIME_FINGERPRINT_SOURCES["indextts2"]
    assert "nodes/_otr_voice_node_common.py" not in sources, (
        "the shared dispatcher is back in the fingerprint -- that reinstates "
        "18-in-19 false demotions; if it was restored deliberately, say why "
        "here and invert this test again"
    )
    # The engine-SPECIFIC files must stay, or the gate proves nothing at all.
    assert "nodes/_otr_audio_engines/eng_indextts2.py" in sources
    assert "scripts/_otr_indextts2_worker.py" in sources


def test_editing_the_shared_dispatcher_no_longer_costs_the_voice():
    """The product test for the recipe change.

    Proves the thing that actually matters: the exact edit that de-qualified
    Lemmy earlier today -- appending a comment to the shared dispatcher -- now
    leaves the route selected. Asserted against the REAL shipped policy.
    """
    from config import cast_pools as POOLS
    from nodes import _otr_voice_route as ROUTE

    ROUTE._LIVE_FINGERPRINT_CACHE.clear()
    before = ROUTE.live_engine_impl_version("indextts2")

    path = REPO_ROOT / "nodes" / "_otr_voice_node_common.py"
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"\n# transient probe comment\n")
        ROUTE._LIVE_FINGERPRINT_CACHE.clear()
        assert ROUTE.live_engine_impl_version("indextts2") == before, (
            "a comment in the shared dispatcher still moves the fingerprint"
        )
        assert ROUTE.select_policy_route(
            POOLS.LEMMY_VOICE_POLICY, "indextts2") is not None, (
            "a comment in the shared dispatcher still withholds the voice"
        )
    finally:
        path.write_bytes(original)
        ROUTE._LIVE_FINGERPRINT_CACHE.clear()


def test_every_fingerprinted_source_actually_exists():
    """A fingerprint over a path that does not exist would hash nothing and
    fail OPEN -- the gate would pass while proving nothing."""
    from nodes import _otr_voice_route as ROUTE

    for engine, sources in ROUTE.RUNTIME_FINGERPRINT_SOURCES.items():
        for rel in sources:
            assert (REPO_ROOT / rel).is_file(), (
                "%s fingerprints %r, which is not on disk" % (engine, rel)
            )


def test_the_fingerprint_is_recorded_for_indextts2_only():
    """Pins today's REAL coverage rather than the coverage one might assume.

    Only indextts2 has a recipe. bark / kokoro / chatterbox / dia have none,
    which is a known open item -- their routes cannot be fingerprint-gated at
    all. This test documents that honestly so nobody reads the mechanism as
    covering every engine.
    """
    from nodes import _otr_voice_route as ROUTE

    assert set(ROUTE.RUNTIME_FINGERPRINT_SOURCES) == {"indextts2"}, (
        "a new engine gained a fingerprint recipe -- good, but update this "
        "test and the open item that tracks the four missing recipes"
    )
