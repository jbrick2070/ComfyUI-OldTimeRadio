<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The packet is not build-ready as-is because it asks the panel to judge a 51-script corpus it does not actually include, withholds the key costly-choice binding logic behind the main diagnosis, and contains several contradictions/out-of-scope invitations that will produce non-actionable recommendations.

MUST-FIX BEFORE BUILD:
1. [Appendix B / 0] The ask requires judging “all 51 published finals,” but the document only provides one full specimen plus quotes and file paths (`otr\obs`, `story_soak_results.csv`). [ASSUMPTION] API panel models will not have filesystem access. Concrete fix: include the full 51 scripts, or at minimum attach a clearly sampled bundle with each episode’s full ledger/script/metadata and sampling method. If attachments are expected, state exactly what files are attached and that the model may rely on them.

2. [2d / 3 Stage 5 / 3 Stage 7] The central diagnosis is that `costly_choice_beat` is not bound to a voiced line 76% of the time, but the document does not show the logic that creates `costly_choice_beat`, assigns `dialogue_slot_id`, populates the conditional dramatic fields, or runs `slot_drama_contracts_audit`. Stage 5’s shown prompt does not emit `costly_choice_beat` at all. Concrete fix: add the exact source excerpts for costly-choice generation, slot binding, line prompt dramatic block construction, and the audit, or remove requests for concrete logic-level fixes to that mechanism.

3. [3 “Logic constants” / Appendix A] The statement “The undershoot is mechanical” and “caps a character total at ~280-490 words no matter what `target_words` says” is overstated/false as written. Appendix A is 700 words despite the cited 14 character beats and 20-35 range, proving the range is not a hard output cap. Concrete fix: rephrase to “the budget target and prompt instructions make 864 unreachable if followed” and include the actual per-beat target derivation/allocation code so reviewers can propose a correct constant change.

4. [0 C4 / 2c-4 / 4] The document asks for fixes to premise/news-seed repetition, but the allowed levers exclude major architecture and the source packet includes no RSS/news-selection logic. Cross-episode dedup/recency guarding requires some persistent history or input queue policy, which is not shown. Concrete fix: either mark news-source selection as out of scope, or include the RSS/article selection code and explicitly allow a small local history file/window-based dedup rule.

5. [6 / 0 Scope guard] The act-bridge option touches `ACT_COUNT_CONFIG`, beat assembly, `start_s` timing, music interludes, workflow JSON, and possibly announcer-over-music behavior. That conflicts with the scope guard saying the render pipeline is healthy/out of scope and “do not propose render changes.” Concrete fix: split this into a clearly optional design experiment, or constrain it to text-only ledger changes already known to be renderer-safe. Add an explicit verification requirement: renderer accepts additional `speaker_role="announcer"` lines at act seams without schema/timing changes. [ASSUMPTION] Current renderer behavior with extra announcer lines is not proven by the document.

6. [6] Internal contradiction: “replace the three separate one-shot announcer calls (intro, outro)” names only two calls. Concrete fix: change to “replace the two separate one-shot announcer calls” or specify the missing third call if it exists.

7. [2b / 0 C3] The document lists `ledger_scrub_status = FAIL on 51/51`, `story_qa_verdict = SKIPPED on 51/51`, and external reviewer 404, but C3 forbids QA/reviewer/gate fixes. This will confuse panelists into either ignoring apparent release blockers or proposing forbidden QA work. Concrete fix: label these as non-story telemetry only, or move them to a separate “known pipeline defects, do not solve here” section with owner/status. If scrub failures affect renderable ledger validity, they are not story-quality-only and must be fixed outside this prompt before release.

SHOULD-FIX:
1. [3 header] The section claims prompts/logic are “verbatim,” but several parts are summarized or omitted: Stage 7 says “Other blocks emitted only when their field is set,” Stage 9 is summarized, Stage 10 is summarized, and post-script passes are not shown. Concrete fix: rename the section to “selected excerpts” or include the full emitted prompts for at least one representative normal line, costly-choice line, intro, and outro.

2. [3 Stage 7] The line-composer prompt is presented in fragments, but the strongest failure modes depend on the assembled final prompt order: stage directions leaking, self-addressing, news terms missing, and conditional dramatic fields. Concrete fix: include one fully rendered prompt example from a failing line and one from a good line, with all optional blocks in final order.

3. [2c] The 14-episode close-read sample is used heavily, but the sampling method is not stated. Concrete fix: say whether those 14 were random, recent, cherry-picked failures, strongest episodes, or stratified by model/target. Otherwise panelists cannot weight the qualitative symptoms against the 51-episode aggregate.

4. [2a / 2b] “ZERO errors” in the soak conflicts rhetorically with later “length pass errors on 33/51,” “scrub fail on 51/51,” and reviewer 404. Concrete fix: define “zero errors” narrowly as “no end-to-end generation crashes before frozen ledger emission” and separate generation success from post-pass/audit failures.

5. [8 Stage 8 / 2c-2] The “Tonight…” diagnosis mixes LLM convergence and deterministic fallback without fallback telemetry. Concrete fix: include counts for announcer validation failures/fallback firings, or phrase the cause as unconfirmed.

6. [5 / C2] C2 says no schema redesign and no renderer-learned structures, while Section 5 says optional new `meta` keys are allowed. That is probably compatible, but panelists may overuse it. Concrete fix: state: “Optional `meta.*` additions are allowed only if ignored by renderer and not required for render correctness.”

7. [3 Stage 3 / 2c-5] The document attributes voice sameness to portrait-heavy casting, but it does not show actual `cast[]` examples beyond Appendix A’s script. Concrete fix: include 3-5 cast records from affected episodes, including gender/name/description, so prompt fixes for speech register can be grounded.

8. [0 requested output] The requested “before -> after text” for every prompt edit is good, but the document does not specify whether models should preserve exact placeholder names and JSON schemas. Concrete fix: add: “Prompt edits must preserve existing placeholders unless explicitly proposing a small code change to supply a new one.”

OPTIONAL / NICE-TO-HAVE:
- [0] Add an explicit acceptance target for proposed fixes, e.g. “864 target should reach 85-105% without length pass,” “costly-choice audit valid on >90%,” “no contradicted announcer close in sample.”
- [7] Ask panelists to distinguish “first change to ship” from “highest theoretical leverage”; those may differ.
- [3] Include one compact ledger excerpt showing `lines[]`, `dialogue_slot_id`, `arc_phase`, and `meta.dramatic_state` for Appendix A.

CUT THESE (over-engineering):
1. [6] Cut the unified intro+bridges+outro announcer pass from the first build. It changes beat count, act seams, timing, tests, and workflow wiring. The minimal safe fix for the documented bug is to make the existing outro ending-aware and ban unresolved hedge language when the script resolved the question.

2. [6] Cut “announcer-over-music” for now. It creates render/timing ambiguity while the stated scope excludes render changes. If bridges are later approved, make them normal announcer lines first, not mixed music behavior.

3. [6] Cut act-bridge announcer lines as a length fix. They may add words, but they do not fix the core undershoot in character drama and risk padding the wrong layer. Fix beat budget/line targets first.

4. [0 / 7] Cut asking every panelist to fully design the act-bridge implementation unless that decision is actually in scope for this round. Keep it as one yes/no question plus minimal prompt suggestion; otherwise it will distract from the higher-severity costly-choice and line-budget defects.