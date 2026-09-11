<!-- requested_model: ~anthropic/claude-opus-latest | resolved_model: anthropic/claude-opus-5 -->

# Adversarial review — R2 driver plan

Verdict frame: the endpoint-ownership change is defensible and small. The rest of the plan mixes (a) genuine implementation gaps that will produce false-passing tests, (b) unverifiable prompt rhetoric sold as a fix, and (c) process ceremony unrelated to the diff. Must-fixes below are keyed to owners.

---

## A. Must-fix defects (with owners)

**A1. `_call` keyword collision and silent-drop path — owner `nodes/_otr_my_story.py:_call` (line 375) + `_pass_act` (587).**
Plan §2 says `_pass_act` "supplies that string" while every pass also forwards `**source_kwargs` from the orchestrator. If `source_rewrite_instruction` ever appears in `source_kwargs`, `_pass_act` raises `TypeError: multiple values for keyword`. Worse, the benign failure is silent: `_call` returns `authored` early whenever `source_rewrite_receipts is None` (see the `if source_rewrite_receipts is None: return authored` branch), so the scope instruction is accepted, ignored, and nothing records it. Must-fix: (i) `_pass_act` must not pass both; pop/assert instead; (ii) add a test that the instruction is inert-but-not-lost in the no-receipts path, and a test that the parameter is *explicitly consumed* (a test that fails if someone deletes the named parameter and lets it ride `**kwargs` into `structured_call`).

**A2. Two consumers, one undefined mechanism — owner `_pass_act`.**
§2 claims one string "tells the author and its correction the scope." Nothing in the plan says how it reaches the *author*: the author prompt is built inline in `_pass_act`'s user content, the corrector gets it via `instruction=`. Either two insertion points are needed (state them) or the claim about the author is false. As written the verification bullet ("test the actual target line/scope delivered to the model") is untestable because the plan never names which prompt slot carries it.

**A3. Self-contradicting act prompt — owner `_pass_act`.**
§2 keeps `json.dumps(treatment.model_dump(by_alias=True))` in the prompt while overriding the `- where it should leave the story:` slot with `treatment.ending`. On the final act the model then sees *both* `acts[-1].ending_state` (stale/local) and the global ending, stated as the target. Plan calls this "Local endpoint remains available in complete treatment JSON" as if a feature; it is the exact contradiction the R1 test case was written to detect. Must-fix: the overriding line must be self-labelling and supersessive ("ignore this act's planned ending_state where it conflicts"), or P1 coherence must be treated as the only mechanism. Do not ship two conflicting statements and call it defence-in-depth.

**A4. Blank-global fallback contradicts the scope string — owner `_pass_act`.**
§2 permits a blank endpoint, but the scope string asserts the "sole/final act must deliver the existing global endpoint." With `treatment.ending` blank (and possibly `plan.ending_state` blank too) the prompt instructs delivery of nothing and renders `- where it should leave the story: ` empty. The scope string must be conditional on a nonblank endpoint; add that as an explicit test case (currently "optional blank-global fallback" only tests the target line, not the scope line).

**A5. Test will false-pass on substring assertions — owner the new runner tests.**
Because the whole treatment JSON (including `ending`) is already in the prompt, any assertion of the form "global ending text appears / does not appear in prompt" passes or fails for the wrong reason. The plan half-recognises this but does not mandate the technique. Must-fix: assert on the parsed target line and scope line only (split on the literal `- where it should leave the story: ` prefix), and assert the earlier-act target line equals `plan.ending_state` exactly.

**A6. must_speak independence is under-tested — owner the new runner tests.**
§"Verification" only checks the final target "with must_speak empty." That proves nothing about the trap R1 identified. You need a 2×2: final act × {must_speak empty, non-empty} and intermediate act × same, asserting the endpoint/scope text is byte-identical across the must_speak axis and that the existing `unheard` block is unchanged.

**A7. Alias/field-name asymmetry in correction is asserted, not tested — owner `_call` + `_otr_story_source.rewrite_story_source`.**
§1 states "CastMember register alias already matches the pack; do not change it." But `_call` builds the correction payload as `authored.model_dump(mode="json")` **without** `by_alias=True`, while `_pass_act` dumps the treatment **with** `by_alias=True`. So the corrector's `draft` may carry Python field names (`speech_register`) while the author/prompt bank use `register`, and `_retain_omitted` / `schema.model_validate` round-trip through that shape. The plan's "do not change it" is a claim about code I cannot verify from the shown excerpt; it needs an explicit regression test that a corrected treatment/cast round-trips identical aliases, not a prose assurance.

**A8. Removing the deferral clause also de-scopes the spoken corrector — owner `nodes/_otr_story_source.py` common system text (line ~125) and `rewrite_spoken_from_source` (337).**
"An act need not repeat every fact; speculation is not a fact" is one sentence; §3 removes the first clause from *all* callers, including `ledger_clean_spoken`, whose `instruction=` block (line 337) contains no replacement. Combined with the retained "restore explicitly supplied people, relationships, actions or endings lost from this artifact's scope," the spoken corrector is now pushed toward maximal restoration into arbitrary existing rows — exactly the over-correction R1 said cannot be proved absent. Must-fix: state the exact replacement sentence for every remaining call site (interpret / treatment / act / spoken), not just `_pass_act`, and add an offline test that the spoken corrector's system text retains a conservation clause.

**A9. Live evidence claim is unsupported by the receipt schema — owner `rewrite_story_source` receipt construction.**
The plan forbids new receipt fields, yet promises to "preserve ... actual prompts/edits" from the canonical run. The attempt record stores `prompt_sha256` only (no prompt text) and the receipt stores no `instruction`. After the live run you will be unable to show which scope string was sent, only that *a* prompt hashed to something. Either accept that live prompt evidence is offline-only (and say so), or allow the single field. Do not claim both.

**A10. No acceptance criterion for the one canonical run — owner the driver.**
"If it still misses source facts, record remaining failure and diagnose" + "No auto rollback ... based on one stochastic outcome" + "No new rejection or change-rate threshold" leaves the run with no defined pass, no defined fail, and no defined action either way. That is an unfalsifiable gate. Must-fix: a binary, pre-registered check on that single episode (e.g. "the global ending's terminal event appears in the final act's spoken lines: yes/no", recorded, with a stated consequence for each branch even if the consequence is "diagnose, no rollback").

---

## B. Weak / over-engineered items

**B1. P0 redefinition should be cut.** §1 bullet 1 adds a paragraph about "brevity, nonspeaking reference or perceived plot importance" to `my_story_interpret_system`, whose `strength` field (`_otr_my_story.py:122`) R1 already conceded has **no deterministic consumer** and unproved causal effect. It also sits in tension with the same prompt's existing "An incidental mention is not [a requirement]", "Every noun is not a speaker", and the abandoned-alternatives rule — you are simultaneously telling the model to be stricter and looser about non-speaking mentions. Zero offline test can distinguish the outcomes. Recommend: drop from R2, or reduce to the single clause R1 authorised ("required = explicitly requested; preferred = source makes it optional") with nothing about brevity/plot importance.

**B2. Three mechanisms for one failure.** P1 coherence wording + `_pass_act` endpoint override + scope string all target the same defect. The override alone is deterministic and testable; P1 wording is stochastic and, if it works, makes the override a no-op (and if it doesn't, the override handles it). The scope string is the only added value for intermediate acts. Consider shipping override + scope string, and holding P1 wording unless a test distinguishes it.

**B3. "No late epilogue outside the authored acts" (§1) is unowned.** The frame stage (`my_story_frame_system`) authors `announcer_outro`/`coda`, and its rule "Introduce the story without giving away its ending" is not touched. If the intent is to stop the ending leaking into the frame, name the frame prompt; if not, delete the clause — as written it's a rule addressed to a pass that doesn't author epilogues.

**B4. Process ceremony unrelated to the diff.** "Canonical unchanged; re-run full validator/round-trip/**widget/link** audit", "No UI/schema/widget changes", "UTF8/noBOM/nonempty", "Python>=3.10" for a prompt-string + one-kwarg change. Keep the JSON-bank validator and round-trip; the widget/link audit and interpreter-version checks buy nothing here and pad the budget.

**B5. Regression-baseline arithmetic is unbudgeted.** You demand equality with `14411pass/51inherited/183skip/1xfail` while editing three prompt strings that existing tests plausibly assert on (the plan itself shows full prompt-stage text is fixtured). Any prompt-text assertion will fail by construction. Must-state: expected count delta and which existing prompt-text tests get updated, up front — otherwise "full regression versus latest baseline" is either a guaranteed fail or a silent licence to edit tests during the run.

**B6. "same two total calls" (§3) is asserted, not grounded.** `attempt_limit = min(SOURCE_REWRITE_ATTEMPTS, max_attempts)`; the value of `SOURCE_REWRITE_ATTEMPTS` is not in the excerpt. State it or say "unchanged from existing constant" rather than "two".

---

## C. Honest-limits check

The "Explicit limits" section is good and largely accurate. Two corrections:

- "P0 labels have an LLM consumer but no enforced meaning" — then §1's first bullet is an unmeasurable change, which argues for B1, not for shipping it with extra clauses.
- The `edit_shape_probe.json` conclusion is correctly hedged, but §3's "Actual schema can append a source-grounded ending to an existing permitted row" should also note the probe's row was *not* protected and that `rewrite_spoken_from_source` strips `PROTECTED_FACT_COMPONENT_FLAG` rows from the candidate (line 337) — so on a real episode the eligible row set may exclude the natural closing line. That's a live constraint the probe does not cover.

---

## D. Minimal must-fix list

1. `_pass_act`/`_call`: kwarg collision + explicit-consumption test + no-receipts inert-path test (A1).
2. Name the author-side insertion point for the scope string or drop the author claim (A2).
3. Make the final-act target supersessive over the in-prompt `ending_state`, or pick one mechanism (A3).
4. Conditional scope wording for blank global ending, with a test (A4).
5. Target/scope-line parsing assertions, not prompt substring matching (A5).
6. must_speak 2×2 independence test (A6).
7. Alias round-trip test for the correction payload (A7).
8. Specify the replacement conservation clause for *all four* correction call sites (A8).
9. Resolve the live-prompt-evidence contradiction (A9).
10. Pre-register the binary check and both consequences for the single canonical run (A10).
11. Cut or minimise the P0 paragraph (B1); state the regression-count delta and prompt-fixture updates (B5).