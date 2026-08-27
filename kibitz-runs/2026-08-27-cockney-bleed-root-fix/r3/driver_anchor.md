VERDICT: build-ready as-is? yes-with-fixes. The code seams and canonical wrapper are now real, but the post-run group reconstruction must account for the final ledger splitting beat order from dialogue-slot identity, and the no-workflow-change rule needs an explicit validation sequence.

MUST-FIX BEFORE BUILD:

1. [P5.3.6] CONFIRMED — final live ledgers expose ordering in top-level `beats[]`, while accepted dialogue identity/text live in `lines[]`; observed published `beats[]` rows do not retain usable `dialogue_slot_id`. Pin the reconstruction: index `lines[]` by `beat_id`; walk the complete `beats[]` order; create a groupable slot only when the matching line supplies valid `d###` and non-reserved speaker; treat every other beat as a run break; call the real `group_voiced_beats`; then require the group's beat IDs to be a subset of the audit set. Otherwise the described audit is not implementable from the shipped ledger.
2. [P5.3.2] CONFIRMED — the exact command is valid only when executed directly in an existing PowerShell process, where `-Set @( ... )` binds one `string[]` parameter. State not to paste the array expression after a native `powershell.exe -File` boundary, where argument serialization differs. The implementation agent should invoke the script directly as written.
3. [P5.2/P5.3] CONFIRMED — order the gates: static workflow/INPUT_TYPES tests first; frozen canonical no-diff proof second; then the wrapper's own canonical validation and live run. Do not start a server merely for `/object_info` before tests, and do not let a successful publish waive a failed static contract check.

SHOULD-FIX:

1. [P2.3/P3.3] CONFIRMED — state that `cast` remains passed to `build_exchange_prompt` for the speaker/persona roster block; only the policy decision stops consuming it. Removing the parameter or roster block would be an integration regression unrelated to bleed.
2. [P5.3] CONFIRMED — after applying the two `-Set` values, verify the runner's printed/applied patch receipt contains exactly `source_bank='media_archive'` and `lemmy_cameo='always include'` before accepting the leg.
3. [P5.3.6] CONFIRMED — import and call the production `group_voiced_beats` helper in a temporary read-only audit; do not restate its chunking algorithm in a second implementation.
4. [P6] CONFIRMED — project code/tests/docs and Bug Bible promotion are separate green chunks. The PBUG/GO_FORWARD/HANDOFF record belongs with the project fix only after the live artifact; the Bible entry follows in its own repository.
5. [P4/P5] CONFIRMED — no node mappings, `INPUT_TYPES`, widgets, links, profiles, or canonical JSON values are changed. Any reviewer proposal to add an “active speaker” widget/input is a category error and must be rejected.

OPTIONAL / NICE-TO-HAVE:

- Include the reconstructed mixed group's beat IDs and speakers in the production receipt so a future reader need not rerun the audit.

CUT THESE (over-engineering):

1. Do not add persistent group-boundary metadata solely for this fix; existing beat order, line records, audit IDs, and the production grouping helper are sufficient.
2. Do not add a second live leg merely to force the per-line fallback. The captured line test is deterministic, and the canonical graph's fallback reachability is already code-grounded.
