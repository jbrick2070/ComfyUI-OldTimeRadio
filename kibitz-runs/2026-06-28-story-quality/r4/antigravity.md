VERDICT: yes-with-fixes. The plan is nearly converged but has some build-blocking ambiguities in S3, S2, S4, and G1, plus a signature mismatch in G1.1 and ungrounded wants parsing in S5.

MUST-FIX BEFORE BUILD:
1. [S3] Reroll accept rule is contradictory: "accept iff grounding passes" conflicts with "when both imperfect, score lower wins" because if the reroll fails grounding, it is rejected, but if the score is used when both are imperfect, it lacks a term for grounding failure, potentially preferring a grounding-failed reroll over an original draft. Fix: Define a total defect score that weights grounding failure highest: `score = 10*grounding_failed + 3*hard_leak + 2*trunc + 2*run_on + 1*roster_caps` (where `grounding_failed = 1` if `validate_composed_grounding` is False, else 0), and compare this score for both drafts (lower wins, original on tie).
2. [S3] Original draft `compose_flags` check misses defects on exchange paths: Lines generated via `use_exchange` bypass `compose_line` and set `beat_compose_flags = ()` (`nodes/OTR_LedgerScriptWriter.py#L4431-L4433`), so evaluating `hard_leak` using `compose_flags` will false-pass the original even if it contains leaks. Fix: Compute `hard_leak` directly by running `verify_and_repair_line` on the draft text for both original and reroll using the active `_episode_entity_policy`.
3. [G1.1] Signature mismatch for `_quality_reroll_hint`: The plan expects it to select a hint based on `req.story_quality_v2_enabled`, but its current signature only accepts `flags` (`nodes/_otr_line_composer.py#L2336`). Fix: Update the signature to `_quality_reroll_hint(flags, story_quality_v2_enabled: bool)` and pass the flag from its call site at `nodes/_otr_line_composer.py#L2462`.
4. [G1 / S4 / S2] Under-specified components [ASSUMPTION]: The plan names golden ledger files (`plancks`, `ledger_ink`, `dance`), curated cliche replacements in S4, and shape-keyed templates in S2, but none of these exist or are defined in the plan or codebase. Fix: Specify (a) where the golden ledger fixture files must be placed (e.g., `tests/fixtures/`), (b) the exact list of cliches and their replacements, and (c) the templates for each `arc_shape`.

SHOULD-FIX:
1. [G1.3 / S5 / S2] Helper import path ambiguity: The plan does not specify where the shared `derive_one_breath_cap` is defined. If defined in `_otr_line_composer.py`, importing it in `story_quality_scan.py` could cause import cycles or headless failures. Fix: Define `derive_one_breath_cap` in the stdlib-only leaf `nodes/_otr_line_hygiene.py`.
2. [G1.2] Ambiguous coordinating conjunctions: The term "coordinating conjunctions" for `_hard_clauses` is not defined. Fix: Explicitly define them as the standard "FANBOYS" set (`for, and, nor, but, or, yet, so`), matched case-insensitively with regex word boundaries (`\b`).
3. [S5] Wants name prefix parsing is ungrounded [ASSUMPTION]: Production and test wants fields (like `character_a_wants` or `character_b_wants`) are documented as verb phrases and do not consistently carry a speaker name prefix. Fix: Rely solely on top-2 by dialogue-line count to determine the two principals in the scan.

OPTIONAL / NICE-TO-HAVE:
- [G1.2] Make `_hard_clauses` public or export it so it can be explicitly unit-tested in isolation.

CUT THESE:
1. [S5] Cut the requirement to parse speaker names from `character_a_wants` / `character_b_wants`. It is safe to cut because wants are verb phrases and dialogue-line count is already a robust, grounded way to identify the top-2 principal speakers.

VERIFY-AT-BUILD checklist:
1. First-pass cap == reroll cap == scan cap (derived from the same `derive_one_breath_cap` helper).
2. `OTR_STORY_QUALITY_V2=0` (flag-off) is byte-identical to previous behavior, with no new meta keys stamped.
3. The new `tests/test_story_quality_golden.py` properly loads the three golden ledger fixtures and asserts `not is_truncated`, `_hard_clauses <= 3`, and word counts within the episode budget.
4. [ASSUMPTION] Pre-existing EntityPolicy/speaker_gender reroll gaps remain untouched and out of scope; verify no new S3 logic depends on those being reconstructed.
