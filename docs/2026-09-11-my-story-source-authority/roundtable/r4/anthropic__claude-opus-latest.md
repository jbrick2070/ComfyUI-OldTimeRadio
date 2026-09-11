<!-- requested_model: ~anthropic/claude-opus-latest | resolved_model: anthropic/claude-opus-5 -->

## Must-fix defects

**D1 — Insertion targets for the two pack rules are ambiguous (Proposal §1–2).**
"after the existing incidental-requirement rule" and "after selected-act-count rule" do not uniquely identify a seam. The incidental rule ("A REQUIREMENT is something they are asking for. An incidental mention is not…") is in `my_story_interpret_system`; an act-count rule exists in **both** `my_story_interpret_system` ("The selected act count is binding…") and `my_story_treatment_system` ("Produce exactly the selected number of acts…"). The P1 rule concerns `acts[].ending_state` and `ending`, which only the treatment seam authors, so it must land in `my_story_treatment_system` — but the plan as written permits the wrong seam. Name the exact JSON key and the exact preceding bullet for each rule before coding.

**D2 — The reworded common clause leaves "scope" undefined for every caller that supplies no scope string (§5 + `_call`/`_pass_frame`).**
Only `_pass_act` sets `source_rewrite_instruction`. `_pass_frame` (`_otr_my_story.py:638`), and by symmetry the interpret/treatment callers, go through `_call` with the new default `""`. After the edit their correction system prompt says "A partial artifact need not repeat source facts outside its scope" with **no scope ever defined**, replacing a concrete permission ("An act need not repeat every fact"). The plausible failure mode is the opposite of the intent: a whole-artifact corrector concludes its scope is the entire source and starts injecting material into the frame/treatment. Either keep explicit wording for whole artifacts or give the non-act callers a one-line static scope. This is the sharpest regression risk in the change and is currently unmeasured (see L2).

**D3 — Residual "an act" in the spliced sentence (§5).**
The replacement only substitutes the prefix; the tail "…absence **from an act** is not death" survives. For the spoken and scene callers (`rewrite_spoken_from_source` at `:337`, `_rewrite_char_scene_from_source` at `:1739`) the resulting sentence mixes a generalized first clause with an act-specific third clause. If the point of the edit is generality, fix the whole sentence or state explicitly that the act-specific tail is retained deliberately.

**D4 — `act_scope` names a field the prompt no longer labels (§4).**
`endpoint = global_ending or plan.ending_state` replaces the value on the existing `- where it should leave the story:` line. For the final act the model is then told the target "supersedes **this act's planned ending_state** where they conflict" — but the only remaining occurrence of `ending_state` is inside the pasted treatment JSON, while the labelled line now shows the episode ending. The model is being asked to resolve a conflict between one visible bullet and an unlabelled JSON field. Either keep both endpoints as labelled lines, or reword the scope so it refers to "the act plan inside the accepted treatment".

**D5 — Multiline/marker endpoint breaks the bullet-list contract (§4).**
`treatment.ending` is bound unstripped and is spliced into a `- key: value` item in a hyphen list, directly above `ACT SCOPE:`. A two-sentence ending with a newline (the treatment schema invites "one or two sentences") yields a malformed list and, with "marker-like" text (which §Verification explicitly fixtures), text that reads as a new prompt section. The suffix-equality test proves nothing about this — it only proves the bytes are where you put them. At minimum decide and document the rendering (e.g. quoted/indented value) rather than deferring to a test that asserts the defect.

**D6 — §1 P0 rule collides with existing interpret rules.**
"Mark explicitly requested narrative directions required" sits next to "Examples, brainstorming alternatives and abandoned ideas are not requirements" and the worked example "maybe a train, or a ship -- go with the ship". "go with the ship" *is* an explicitly requested direction selected from brainstorming; the new rule and the old example now pull in different directions for the same sentence class. Add the disambiguation, or you have introduced the kind of upstream mismatch P1 exists to correct.

**D7 — Verification baseline is stated in unparseable numbers (§Verification).**
"14411pass/51inherited/183skip/1xfail", "saved06", "fullcanonical5080", "11.39", "live05/06" are not machine-checkable, yet the plan's own acceptance rule is exact failure-ID and normalized-payload comparison. Write the actual counts and the file/ID of the baseline failure list before coding, otherwise the comparison gate is unenforceable.

**D8 — Prompt-hash comparability is broken by the change itself.**
"Existing journals record prompt hashes" is cited as evidence infrastructure, but §1/§2/§4/§5 all change prompt bytes, so every recorded prompt hash differs from baseline by construction. The plan should state that hash comparison is only usable within the candidate run, not as a baseline/candidate diff.

## Risk / methodology

**R1 — Four coupled instruction edits, one measurement run.** §1, §2, §4 and §5 each independently alter author and corrector behavior, and the single canonical run cannot attribute any observed fidelity change to any of them. The plan disclaims "broad reliability claim" but still routes a pass/fail decision ("Source pass: preserve and proceed") through a confounded single observation. Either stage the edits or state explicitly that the run yields no per-change evidence.

**R2 — §2 (P1) and §4 are mutually masking.** If the treatment author obeys P1, the `global_ending`/`ending_state` conflict never arises and the override is dead code; if it disobeys, the override silences the mismatch, so P1 compliance becomes unobservable in the episode. Add a receipt-free but *