# My Story cross-machine repairs: evidence and triage

Initial review snapshot at 2499b411. GO_FORWARD_PLAN.md remains the only work queue.
The immutable original R1 input and subsequent corrections are in
kibitz-runs/2026-09-10-my-story-cross-machine/. This evidence index reflects
grounding corrections; a review proposal is not an implemented fix.
The operator requests all reported defects triaged and fixed, with at least
one independent reviewer at each step, then coordinated tests on the few
proven routes for each machine. No further baseline renders while fixing.

## Evidence and boundaries

- 4060: docs/2026-09-10-my-story-4060/README.md and drill Steps 116--121.
  Six one-act canonical runs at 0327850a: four complete stills publications,
  two P1 malformed-output failures, zero source-qualified outputs. Explicit
  living-mother/shared-dinner facts were inverted at interpretation, writing,
  or visual planning. A valid short coda was rewritten by shared cleanup.
- Mac: MAC_LESSONS_LEARNED.md section 10. A stale runtime copy was correctly
  refused at 37 saved versus 33 live widgets. After sync/restart, a generated
  unowned fourth speaker was correctly repaired. Later the OS killed the
  process at 19,163 MB during the writer/cleanup workload. A smaller uncached
  model then correctly failed the disk-space precheck. No full publication.
- 5080: final music-parent correction f829c920 produced one clean six-act
  writer/freeze component. RunPod's earlier three-act component had music
  parent warnings. Neither is a full canonical episode qualification.
- Shared source is now 2499b411. The incoming three commits contain docs only.
  Root is the sole production editor; reviewers do not edit code or tests.

## Grounded triage

| ID | Finding | Classification and owner |
|---|---|---|
| M1 | Constrained generation retains per-request token history on a resident model | Confirmed static defect; NOT active in the failed My Story Mac cleanup route, which never binds a grammar. `_otr_constrained_generate.py:123-142` caches parser/prefix functions by schema. Installed LMFE retains full-prefix histories. Repair this before introducing the P1 binder, without claiming it caused or fixes the Mac OS kill. |
| S1 | P1 repeatedly emits malformed JSON | Live failure; existing lazy `_otr_bind_schema` is available. Binding P1 alone is a proposed repair, but its hidden limits and M1 must be corrected first. |
| S2 | One act is described as having between-act music | Confirmed input-projection bug in My Story P0/P1. Actual assembly already emits zero boundaries. Derive one count from selected acts and breaks, reuse it in all prompts/assembly. Do not reject prose merely for mentioning a keyword. |
| F1 | P0/P1/P2/P3 can override the supplied story | Confirmed live correctness defect. Source bundle is already immutable raw/normalized data, but P0 is allowed to invent assumptions and P2/P3 receive only derived treatment/frame context. `fidelity_discrepancies` currently measures cast/gender differences, not semantics. |
| F2 | Visual scenes invert required shared presence | Confirmed live defect. MetaBrief `_build_char_scene_request` and `_compose_char_scene_prompt` own the prompts; ImageDirector routing alone is insufficient. Source facts and companion requirements are absent; first nonempty reply is accepted. |
| C1 | Cleanup substitutes valid authored speech | Confirmed live defect in `_otr_ledger_clean._repair_row`; cleanliness alone admits an unrelated replacement. The existing transaction reseals legal changes but cannot prove semantics. |
| C2 | Five dirty rows and six edits | Unverified as a separate bug: model-only dirty count excludes pattern-only targets. Need actual row-level receipt before attributing a sixth edit to an unowned target. |
| O1 | Pre-image missing-still warning sounds like a render failure | Diagnostic fix complete: cast-preflight INFO, unchanged request content and real post-image/render guard. See diagnostics_receipt.md. |
| O2 | LTX-open health complains on intentional still_pan | Confirmed intent-classification defect. Health must compare expected versus actual engine, and only demand LTX where selected. |
| O3 | Successful same-file rename reconciliation is logged loudly | Diagnostic fix complete in mux, image dispatcher and render manifest join. Failed identity/freeze/path handling is unchanged. See diagnostics_receipt.md. |
| D1 | Mac checkout and loaded extension differ | Deployment state, correctly detected. Retest instructions must verify the loaded extension path, commit/schema and restart, not alter canonical widgets. |
| D2 | Mac cannot download another model with current free space | Correct storage refusal. No cache deletion, invented fallback, or raised budget. Measure the actual generation/retirement boundary; M1 is not a Mac recovery claim. Incoming 0404aa86 explicitly scopes the failure to the measured M4/16 GB host. |

## Critical constraints for proposed repairs

1. There is ONE workflow: workflows/otr_canonical.json. All qualifying live
   tests use the shipped canonical runner with full execution through audio,
   video and an actual otr/obs file. No alternate graph, replay substitution
   or partial execution target. Normal story fields and sanctioned machine
   runtime settings are recorded. Update the canonical in the same change if
   a node, socket, widget or wiring surface changes; append optional widgets.
2. Preserve selected acts and flexible story-led cast. No RSS on My Story.
   No word, duration, story-length or taste rejection. Keep real provider,
   memory, storage and cancellation errors. Reuse existing repair ladders.
3. Direct listener facts are correctness, not prose quality. Do not implement
   a fixture-specific blacklist of death/fire/alone/waiting. Literal source
   quotes and byte hashes establish identity, not semantic entailment.
4. The raw input bundle is the authority before P0. A model-extracted fact list
   cannot become authoritative merely because its JSON is valid. Unstated
   status remains unstated; nostalgia does not imply bereavement or absence.
5. Preserve all accepted authored words outside verified cleanup complaint
   spans. Do not mark every My Story row as a protected Python fact to bypass
   cleanup. Resolve whole-row accusations explicitly; don't silently ban all
   legitimate descriptive-line repairs or let a vague complaint replace a coda.
6. Visual checks must distinguish story scenes from neutral headshots and
   announcer/music radio assets. Shared presence belongs to relevant scene
   assets; a neutral single-person portrait is not evidence of separation.
7. Publication eligibility currently concerns rights and identity. Do not turn
   an empty cast-discrepancy list into a semantic PASS, or quietly add a late
   gate that destroys a finished render. Any source receipt needs truthful
   scope, source/artifact hashes, lifecycle invalidation and the real consumer.
8. Adding P1 grammar today also activates a 2048-token open-string decode guard
   and LMFE's default array length 20. The installed LMFE supports disabling
   only its implicit array limit with max_json_array_length=0, but its prefix
   builder overwrites parser.config. Preserve explicit schema bounds and the
   tokenizer alphabet; do not use a huge substitute cap or global env mutation.
9. Keep expensive tokenizer preprocessing resident, but parser/prefix/enforcer
   state must belong to ONE generation, including repeat use of one closure and
   failure/cancellation paths. Don't introduce a new model cache or lose epochs.

## Proposed order, subject to grounded review

1. Fix M1 and grammar-policy boundaries; prove tokenizer preprocessing reuse,
   request-state release, dynamic-schema release, repeated closure calls and
   error paths. Then connect P1's existing binder once without changing other
   pass callables, retries or source preservation. Correct S2 at its producer.
2. Close C1 using validated complaint coordinates and the existing authorized
   transaction. Preserve semantic rejections in receipts and progressive repair
   selection; never let a later cleaner score override scope conservation.
3. Close F1/F2 across actual authoring owners. Reuse immutable source intake,
   shared structured-call/repair machinery and existing slot scheduler. Carry
   original source into P2/P3 and cleanup/visual requests. Review the smallest
   supported semantic-check design before implementation: raw source remains
   authority; any extracted anchors must point to exact source spans; authoring
   checks run before expensive media and receipts bind final accepted artifacts.
   No second retry engine or generic subjective judge. Define what an uncertain
   check means and how actual contradictions reach the existing repair owner.
4. Correct O1--O3 without weakening post-image, planned-engine, file-identity or
   freeze checks. Distinguish legacy absent-intent receipts explicitly.
5. Use focused tests, full Windows suite against the recorded failure baseline,
   explicit clean-OTR Bug Bible comparison, one finished-diff CLI review for
   every code chunk, canonical audit, commit/push together and HEAD equality.
6. Retest one common repaired revision. Start with the strongest reproduced
   vulnerability: one-act living-mother/shared-table source, full canonical
   publication. 5080 first; 4060 owner repeats its proven Z-Image/still_pan route
   with exactly one act; Mac owner verifies loaded runtime and existing model
   memory behavior on proven Mac dropdowns. Do not test every video/TTS engine.
   After recovery, vary acts/casts and repeat on a second compatible installed
   model family. RunPod stays stopped until it provides non-duplicate evidence.

## Review questions

- Verify priority, root-cause claims and any missing cross-machine owner.
- Propose the smallest source-preservation design that covers P0 through final
  spoken text and scene prompts without mistaking syntax or copied quotations
  for semantic fidelity. Distinguish deterministic checks from model judgments.
- Resolve whole-line cleanup complaints while preserving legitimate repairs.
- Identify hidden memory, grammar, cache, receipt/replay or late-gate risks.
- Separate actionable defects from correct resource/deployment refusals and
  unverified report claims. Cite actual files, not this document as proof.

The existing A2 capacity and Original/credits work remains in GO_FORWARD;
these new shared-memory/source-preservation blockers come first. No new live
test, source-quality PASS, complete stress campaign or registry publish is
claimed by this review input.
