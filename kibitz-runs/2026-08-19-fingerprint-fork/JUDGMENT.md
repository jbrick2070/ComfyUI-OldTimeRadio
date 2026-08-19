# JUDGMENT -- the voice fingerprint fork

Driver: Claude (Opus 5), sole judge. **Roster, stated exactly:** Fable ran r1
COLD (unanchored, no driver framing). **Codex gpt-5.6-sol hit its usage limit
mid-run and produced nothing** -- quota-held to 20:31. Per the 2026-08-17
directive a missing lane never blocks the arc, so a **Sonnet subagent filled
the Codex seat**, grounded. So: two lanes, one of them a substitute, and the
premium Codex lane did NOT participate. This is not a full arc and is not
described as one.

## THE DRIVER'S OWN LEAN WAS WRONG, AND BOTH LANES KILLED IT

My anchor picked **B** (hash normalised code, ignore comments). Both lanes
refuted it, and I verified the refutation myself rather than taking it:

* **B would not have prevented the incident that started this.** My edit added
  a comment block AND a `log.warning`. A log call is a new AST statement, so B
  trips on it too. Verified by direct experiment.
* **Measured over real history** (substitute lane, replaying all 44 commits
  that ever touched the four hashed files): raw-byte fingerprint changed
  **44/44**; a correctly-implemented AST fingerprint changed **43/44**. B buys
  **one commit in forty-four** -- about 2%.
* **B is also harder than it looks.** `ast.dump` KEEPS docstrings (they are
  `Expr(Constant)`), so the naive recipe still trips on prose. Verified:
  a prose-only edit changes a naive `ast.dump` hash. Same for
  "tokenize and drop COMMENT/NL" -- the docstring survives as a STRING token.
  Making B work needs an explicit docstring-stripping walk: new machinery whose
  own bugs would manifest as losing Lemmy's voice. That is exactly the
  "programmer thing" the operator said he does not want to own.

## WHAT BOTH LANES AGREED ON, INDEPENDENTLY

The defect is not WHAT is hashed. It is WHAT HAPPENS ON MISMATCH.

Fable put it best, cold: **"the punishment for a suspected wrong voice is a
guaranteed wrong voice."** The gate's response to "the code might have changed"
is to withdraw the approved voice and draw an ordinary one -- causing, with
certainty, the harm it exists to prevent. It fires on ordinary edits (19
commits in 60 days on one file, verified) and would catch a real voice change
approximately never.

Supporting evidence neither of us had at the start:
* **4 of 5 engines have no code fingerprint at all and nothing has gone wrong.**
  The project has been running without this gate in most of the house.
* **It already blocked a legitimate fix** -- the stale-ledger warning, reverted
  purely because of the hash.
* **Scalability is worse than the driver stated.** `_otr_voice_node_common.py`
  is the shared per-line dispatcher for EVERY engine, so gating all five would
  put the same file in all five recipes -- one edit would then demote all five
  routes at once. Friction multiplies by engines gated, it does not stay flat.

## THE RULING -- adopted, and it is neither lane's answer verbatim

**Remove the DEMOTION, keep everything else.** Concretely:

1. **Drop the two call sites that turn a code-hash mismatch into a demotion:**
   `_otr_voice_route.py:547` (`select_policy_route`) and `:1163`
   (`resolve_and_verify_reference`). Verified: those are the only two in
   production. ~15 lines, no redesign.
2. **KEEP `live_engine_impl_version()` and `RUNTIME_FINGERPRINT_SOURCES`.**
   Still compute it, still stamp it into new qualification records. It also has
   a SECOND consumer -- `_otr_resolved_request.py:84,118,260,304`, where it is
   part of the audio cache key, so different code still caches separately.
   Removing it outright would break that. (Substitute lane found this; the
   driver had missed it.)
3. **KEEP `weight_revision` and `reference.source_ref_sha256` gating exactly as
   they are.** Those hash the actual weights and the actual reference audio --
   things that change only by deliberate act. They guard a real, recurring
   failure class and cost nothing.
4. **ADD Fable's breadcrumb, which the substitute lane's version drops.** On a
   code-hash mismatch: keep the approved voice, and record that it drifted.
   **Not only a log line** -- the substitute lane correctly notes 126 files log
   to the same `OTR` logger, so a warning is easy to miss. Stamp it in the
   episode's own meta (`approved_under` vs `rendered_under`), the same
   present/absent receipt pattern `meta["title_work_anchor"]` and
   `meta["bank_roll"]` already use. Then a drifted episode says so on its own
   ledger, and his ear stays the judge.

**Why the synthesis beats either lane alone:** the substitute lane's C removes
the signal entirely, losing the breadcrumb Fable was right to value; Fable's D
proposes a log line, which is too easy to miss in this codebase. Keeping the
value stamped durably gives the audit trail at zero ceremony.

## WHAT THIS COSTS HIM

The guarantee that a code change can never ship ONE episode with a slightly
drifted Lemmy. Realistic worst case: an engine change shifts the voice, he
hears it in a daily, the ledger names what changed, he re-approves once. One
imperfect episode is the whole downside -- against a gate that currently
withdraws the voice on comment edits.

## STATUS: NOT IMPLEMENTED. AWAITING THE OPERATOR'S GO.

This changes how voices are selected. It is his call, and he has not given it.
