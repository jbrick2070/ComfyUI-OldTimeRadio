# Codex task -- the final cleaner pass (OTR dead / AI-slop code)

Repo: `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`,
branch `v2.0-alpha`. Read-only: change NOTHING, run no render, boot no server.

You are the independent check on a campaign that has already run five hunt
rounds plus a six-seat blind sweep. Two jobs, in this order. **Job A is the
one that matters most** -- it gates ~40 pending edits.

---

## JOB A -- adjudicate the PENDING queue before it is executed

Read `docs/2026-08-28-fresh-sweep-round1/CONSOLIDATED_FINDINGS.md`. Its
"PENDING -- the round-2 execution queue" section lists findings from blind
auditors that have NOT yet been executed. Treat every claim there as a
HYPOTHESIS, not a fact.

For each pending item, return one of:

* **CONFIRMED** -- you read the cited lines and every material claim holds.
  Give the exact edit list (files + line ranges to change, and the lines that
  must NOT change: live neighbors, tests that pin current behavior).
* **PARTIAL** -- the core claim holds but a cited detail is wrong. Say which.
* **MISREAD** -- cite the line that disproves it.
* **HAZARD** -- executing it as written would break something. Name what.

Method, mandatory, three layers per item: (1) read the real file at the cited
lines; (2) exact-name grep across `*.py`, `*.ps1`, `*.cmd`, `*.json`, `*.yaml`,
`*.md` and classify EVERY hit (a comment is not a caller; note any
`getattr`/env/string-dispatch risk); (3) say plainly what you could not check.

**Five house rules that override any finding -- apply them and say so when
they bite:**

1. **Video/engine lanes are INDEPENDENT (operator ruling 2026-08-23).**
   Duplicated helpers across `nodes/_otr_video_engines/` (and the image/audio
   engine packages) are DELIBERATE. Any "consolidate this shared helper"
   finding is REJECTED on sight -- do not revive it, do not re-report it.
2. **`nodes/_otr_audio_engines/eng_indextts2.py` is FROZEN for tidiness.**
   A shipped Lemmy voice-qualification record fingerprints that build; any
   edit -- even a behavior-identical dedup -- fails
   `tests/test_voice_identity_fix.py::test_the_shipped_lemmy_route_is_selected_again`
   and would cost a re-audition on the operator's ear. Reject edits there.
3. **A hole in the ledger is worse than a rip.** Removing a ledger/meta field
   requires every downstream reader accounted for first (TTS, slicing, video
   direction, captions, credits, obs publish). Prefer "needs an owner ruling"
   over "safe to delete" for any stamped field.
4. **Cache/profile identity is load-bearing.** A key that only feeds a hash
   still matters: say explicitly whether removal moves a PERSISTED cache key
   or ledger receipt, or only an in-process cache. (Precedent: the Bark
   `recommended_speakers` keys are KEPT for exactly this reason.)
5. **Widget removal is a THREE-part change** (`widgets_values`, the `inputs`
   descriptor array, and every link `dst_slot` after it -- repaired by
   IDENTITY, never arithmetic), plus regenerating all variants. Any finding
   that removes a widget must say so and be treated as high-risk.

Rank your output: HAZARDs first, then CONFIRMED-and-highest-value. Explicitly
call out the three items already believed to be real BUGS rather than
tidiness (an ffmpeg resolver bypass that can destroy a finished episode at the
mux, a second resolver that ignores `OTR_FFMPEG` while its shipped tooltip
promises otherwise, and a pair of SceneSequencer sockets whose declared
widgets the execute function cannot accept).

---

## JOB B -- hunt the slices the blind sweep did NOT cover

The six-seat sweep declared these coverage gaps. Hunt them fresh:

* `nodes/_otr_video_engines/render_driver.py` (~6,350 lines) -- scanned only
  for symbol duplication, NEVER read for docstring/comment lies or
  unreachable branches. This is the biggest unexamined surface in the repo.
* `nodes/_otr_video_engines/` support files not read: `acceptance.py`,
  `beat_session.py`, `cheap_families.py`, `coverage_plan.py`,
  `frame_contract.py`, `ghost_signal_author.py`, `wrapper_bridge.py`,
  `ltx25_recipe.py`.
* `nodes/_otr_shared/` files not read: `capability_profiles.py`,
  `cloud_media_backend.py`, `cloud_media_canonical.py`, `cloud_media_invoke.py`,
  `gpu_residency.py`, `portrait_ledger.py`, `content_oracle.py`,
  `llm_policy.py`, `slug_inventory.py`, `google_slug_verifier.py`,
  `ffprobe.py`, `still_plan_helpers.py`, `boot_contracts.py`.
* `config/source_banks/` bulk data: per-file dead-data needs a different
  method than reference-grepping -- propose one if you can, or say it is out
  of reach.

What counts as a finding: definitions with zero callers; unreachable branches;
comments/docstrings claiming behavior the body lacks; parameters accepted and
ignored; fields written and never read; defensive handlers for impossible
states; and cross-references to files/symbols that no longer exist (VERIFY
absence -- much was deleted recently, and a tombstone comment recording a past
removal is a RECEIPT, not a finding).

---

## Output format

    ### <short title>
    JOB: A | B
    VERDICT: CONFIRMED | PARTIAL | MISREAD | HAZARD   (Job A)
    CATEGORY: stale-claim | unreachable | debris | duplicate | inert-control | slop
    WHERE: path:lines
    WHAT / CONSUMED (proven how) / EDIT LIST / DO-NOT-TOUCH / RISK / PAYOFF

End with the honest sentence: "this pass found N defensible findings" or
"this pass is CLEAN -- zero confirmed findings". The campaign's stop rule is
two independent blind passes returning zero, so a real clean verdict is
valuable and a padded list costs a day. Do not pad; do not rubber-stamp.
