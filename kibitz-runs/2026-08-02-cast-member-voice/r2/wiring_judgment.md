# Wiring judgment -- Sonnet 5 review, grounded by the driver

Reviewer: Sonnet 5 (wiring/integration/sequencing lane, read-only).
Grounded against the real Windows files before folding anything in.

## THE PANELS DISAGREED, AND SONNET WAS RIGHT

**codex MUST-FIX 3 said:** "Replace the impossible 'one exact call site both
lanes pass through' requirement. There is no common pre-proof boundary."

**Sonnet B6 said:** `_otr_content_authorship.stamp_receipt` IS that boundary.

**VERIFIED -- Sonnet is correct:**

    nodes/_otr_scifi_codex.py:3174   stamp_receipt(
    nodes/_otr_scifi_fable2.py:2538  stamp_receipt(

One function, called by both content-owned lanes, immediately before
`build_receipt` mints the line proofs -- and its only positional parameter is
`ledger_data`, which already carries `cast[]` AND `lines[]`. So the gate needs
NO new plumbing and NO new interface threaded through two producers.

This materially simplifies the design codex proposed (one pure validator invoked
at two producer-specific boundaries). It is one validator at ONE boundary.
I accepted codex's version an hour ago; that acceptance is withdrawn.

Caveat kept from codex, still valid: a gate at `stamp_receipt` protects the
MINTING moment only. Fable2 has no tail finalizer, so a later tail mutation can
still empty a row after the proof is stamped -- which is exactly what the live
ledger shows happened. Both are needed: the pre-mint gate AND fable2's missing
final check.

## THE HIGHEST-VALUE FINDING: REMOVAL IS ACTIVELY UNSAFE TODAY

`_otr_voice_node_common.py:109-127`. When a line's `char_id` no longer resolves
to a cast row, `cast_lookup` returns `{}` (by design, never raises), and the
"gender-agnostic last resort" branch fires:

    _seed = f"{episode_seed}_{cast.get('char_id', '')}_anyref"
    entry = _random.Random(_seed).choice(cands)

VERIFIED by reading the branch. It returns a REAL, PLAYABLE voice file, seeded
on an empty string. That branch is deliberate and correct for its intended case
-- an unservable GENDER, where a clone engine must not silently drop to bark
(PD1) -- but it also swallows the DANGLING-REFERENCE case, where the only
correct answer is to fail.

**So implementing the operator's "remove the character" half today would produce
a ghost: cast row gone, lines still pointing at it, a randomly-seeded real voice
speaking them, and a render that passes.** That directly violates this repo's
fail-loud rule. It must raise before any removal path ships.

This is also why codex's CUT 1 (enforce BEFORE `_assemble_ledger`, so nothing
exists to clean up) is the right architecture: it means the dangerous removal
path never has to be built at all. The two findings agree.

## ACCEPTED -- Topic A (wan_ti2v)

**W1. ZERO extra stills.** Cheaper than feared, triple-confirmed:
`jump_still_requests` returns `()` unless `join_mode == JOIN_JUMP`
(`coverage_plan.py:661`), and wan_ti2v declares `strict_first_frame`, so a split
beat is always `JOIN_CHAIN`. Successors take the predecessor's real terminal
frame (`render_driver.py:3208-3241`), never a minted still. Both engines already
on the allowlist also declare `strict_first_frame`, so the jump-still seam has
never been exercised by anything.

**W2. `_planned_length` has NO VRAM guard, and this change widens its blast
radius.** Confirmed: it checks ladder legality and the configured ceiling only
(`eng_wan_ti2v.py:826-853`); `free_vram_mb`/`compute_real_frame_budget` appear
only in `_floor_length`. `VramPeakProbe` enforces nothing ("the peak is sampled
+ logged, never enforced"). Today the bypass only matters above 177 frames;
after the change it matters across the whole 82-530 band. The
`assert_frame_affordable` extraction was proposed in the no-mirror doc and never
landed -- it now becomes a prerequisite, not a nicety.

**W3. The mirror grep test should NOT be deleted.** `extend_frames_to_target`
legitimately remains on the single-clip branch. What inverts is the DOCSTRING's
premise -- the ceiling becomes dual-purpose (adapter cap AND planning cap). Keep
the grep, amend the prose, and ADD a test pinning the multi-clip refusal-on-
shortfall invariant (`eng_wan_ti2v.py:1051-1065`), which no test currently
covers and which is the actual new safety property.

**W4. Stale comments are load-bearing.** `frame_contract.py:276-309`,
`eng_wan_ti2v.py:813-821`, and the test module's own docstring all assert "WAN
stays out of PLANNING_CAP_ENGINES" as settled fact. Rewrite in the same commit
or the next reader trusts a lie.

## ACCEPTED -- Topic B, do not reinvent detection

`_otr_ledger_freeze._check_per_cast_invariants` (`:526-550`) ALREADY computes
"cast char_id referenced by zero non-skipped lines" -- but appends to
`warnings`, never `errors`. It is lane-agnostic and already runs for every bank.
Reuse that logic in the new gate rather than writing a third implementation of
the same cross-reference. Note it checks cast->lines only, never lines->cast
(the dangling direction), which is the check that does not exist anywhere.

## VERIFY-AT-BUILD (both reviewers flagged, neither resolved)

The exact downstream pass that empties a character's text AFTER fable2's
assemble-time gate has passed. Sonnet marked this UNVERIFIED and time-boxed;
codex assumed a stale bank registry. My own ledger read shows the rows arrive
`text='' skip=True speaker_role=character`, and that `build_receipt` and the
validator share the same `_voiced_rows` predicate -- so the mutation is real and
post-mint. The pass that does it is still unidentified. Find it before coding.
