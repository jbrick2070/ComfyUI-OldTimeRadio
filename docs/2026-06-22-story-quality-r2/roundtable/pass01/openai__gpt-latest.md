<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The document is still a problem statement with unresolved design choices; the cheapest fixes are clear, but build would be risky without specifying the exact seams, gating, and regression tests.

MUST-FIX BEFORE BUILD:
1. [Q1/A] The “silent music only” option cannot be implemented by blanking the outline beat as written: `_otr_outline.Beat.intent` requires `min_length=4` and `target_words >= 3`, and `_assemble_outline()` currently creates `music_inter` beats with a non-empty placeholder intent. Concrete fix: keep a valid internal `intent` on `music_inter` beats, but make downstream transcript/caption/dialogue rendering suppress text for `speaker_role == "music_inter"` while preserving the ledger timing/music row. Do not key this only on `dialogue_slot_id is None`, because `music_open`, `music_close`, and `sfx` also have no slot id. Add a regression that no rendered transcript/caption line contains `"Musical interlude bridging"` and that the `music_inter` row/count still exists. verify: exact ledger/text rendering seam in `production_ledger.init_lines_from_outline` or equivalent.

2. [Q1/A] Replacing the placeholder with a “real in-world cue” is not sufficient if the row still renders as spoken/caption text. The observed defect is not only the wording; it is that a non-voiced `music_inter` beat becomes a visible/spoken line. Concrete fix: implement role-based suppression first; optionally change `_assemble_outline()`’s internal intent to something neutral like `"Bridge to the next phase with music only."` for diagnostics. Do not rely on prompt wording to solve this.

3. [Q2/B] The announcer close cannot be fixed by the existing character-line hygiene/critic path. `_otr_line_hygiene.py` states announcer open/close truncation routes to the dedicated announcer composer and that the critic excludes announcer lines as locked structural content. Concrete fix: patch the dedicated announcer close composer/prompt [ASSUMPTION: such composer exists per `_otr_line_hygiene.py`] with hard constraints: no thesis phrases such as `"Tonight's revelation"`, `"the lesson"`, `"reminding us"`, `"proving X right"`, `"this shows"`, and require one concrete final image/action tied to the changed ending. Add a post-generation deterministic scan for those banned close/meta phrases and reroll through the announcer composer, not the character reroll path.

4. [Q2/B] `_assemble_outline()` currently stamps the close intent as `"Close the episode and tag the broadcast."`, which invites generic summary copy. Concrete fix: change the close beat intent to a concrete-image contract, e.g. `"Close on a concrete final image showing what changed; no moral, thesis, or news-summary tag."` This is ledger-safe because it changes only outline content, not schema.

5. [Q3/C] The weak-model fix must operate at final line composition, not only outline Stage 3. Grounding shows Stage 3 `_BeatFleshout` only produces `intent` and `mood`; the cliches and stage-business examples are final dialogue defects. Concrete fix: add the opposed-want/objective constraints to the line-composer prompt and add a targeted final-line reroll/scrub pass for flagged dialogue lines. verify: exact line composer function and reroll seam.

6. [Q3/C] A prompt-only fix is under-specified and likely to fail on the weak end. Concrete fix: define a deterministic accept/reject gate before build:
   - reject exact/near-exact cliches from the grounded list: `"you're playing with fire"`, `"this changes everything"`, `"we're not leaving anything to chance"`;
   - reject pure stage-business lines with no pressure/reveal/refusal, starting from the observed patterns: `"I'll go check..."`, `"I'll double-check..."`, `"I'll lock down..."`, `"I've got this. No need..."`;
   - reroll only rejected lines with a prompt that includes the beat intent, speaker, opposed want, and the previous/next line context.
   Keep the list small and high-signal so it does not become a style police system.

7. [Q3/C] If the new pass relies on `DramaticState`, the current helper can generate generic defaults that do not encode real opposed wants. `_otr_dramatic_state.derive_dramatic_state_from_meta()` defaults to `"honor the established commitment..."` vs `"force a compromise..."`, and the validator only rejects identical strings, not non-opposition. Concrete fix: only use `DramaticState` wants as hard line constraints when they are source-derived/non-default, or add a stronger derivation/validation step before using them to drive rerolls.

8. [Q4] The plan does not specify opus/frontier regression protection. Since the strong model already works, round 2 must be gated to avoid rewriting good lines unnecessarily. Concrete fix: make the added craft pass targeted-only: music role suppression always; announcer close scan always; character line reroll only for deterministic flags or a very low critic score. Do not run a blanket rewrite over every line.

9. [Q5] The anti-goal is stated but not enforced. Concrete fix: explicitly prohibit changes to `EpisodeBudget`, beat count, target word allocation, and ledger schema for this round. Tests should compare that the number of `music_inter` rows and voiced slot ids remain stable before/after, while transcript text improves.

SHOULD-FIX:
1. [Q1/A] Add a fixture using an outline with `music_inter` beats from `_assemble_outline()` and assert `stamp_dialogue_slot_ids()` still leaves those beats with `dialogue_slot_id=None` while rendered transcript/captions omit them. This catches accidental treatment of music as dialogue.

2. [Q2/B] Add a close-line test set with the exact grounded failures:
   - `"Tonight's revelation: ..."`
   - `"Tesla's throne is now shared..."`
   - `"...reminding us to..."`
   The expected behavior should be reroll/reject, not regex deletion.

3. [Q2/B] Use `_otr_dramatic_state.is_resolved_ending_change()` and `HEDGE_LIST` only for hedge/open-ending handling; do not confuse that existing hedge check with the new “no thesis/meta summary” check. They are different failure modes.

4. [Q3/C] Require the line-composer prompt to turn each beat intent into an action verb under pressure: reveal, refuse, demand, bargain, accuse, conceal, choose. This is safer than asking for “better prose,” which weak models will satisfy with cliche.

5. [Q3/C] The reroll prompt should include the rejected line and the reason, e.g. `Rejected because it is stage-business without conflict`, but should not include a long ban list that consumes context and encourages the model to parrot the banned phrases.

6. [Q3/C] Add a cap on rerolls per episode, e.g. max 3-5 character-line rerolls plus close reroll, to keep latency bounded and prevent weak models from cycling.

7. [Q3/C] Track metrics separately: music placeholder count, meta-close count, cliche count, stage-business count. Do not collapse these into a single “story quality” score; failures need different fixes.

8. [Q4] Preserve `_otr_outline` Path C architecture. The grounded code already splits macro/phase/beat planning and has budget validators; round 2 should not redesign outline generation to solve final dialogue defects.

OPTIONAL / NICE-TO-HAVE:
- Add a tiny “final image” helper that extracts concrete nouns/actions from the last character beat or `DramaticState.ending_change` for the announcer close prompt. [ASSUMPTION: final ledger line context is available at close composition time.]
- Log every rerolled line with reason and before/after text for the next soak review.
- Add a small allowlist for intentional repeated phrases if a story uses a motif; otherwise a cliche scanner may reject a deliberate refrain.

CUT THESE (over-engineering):
1. [Q3/C] Cut a full extra LLM rewrite pass over every line. It is expensive, risks regressing strong-model output, and is unnecessary if deterministic flags catch the grounded weak-end failures. Use targeted reroll only.

2. [Q3/C] Cut a large global cliche ban-list. It will overfit, create false positives, and push weak models into alternate cliches. Start with the grounded phrases plus a small exact-pattern list.

3. [Q3/C] Cut critic-driven reroll as the primary gate unless the critic has deterministic thresholds and tests. A weak local model may be a weak critic too. Use deterministic rejection first; use LLM critic only as optional targeted scoring.

4. [Q1/A] Cut “replace the music placeholder with prettier in-world text” as the main fix. It leaves the core bug intact: music beats should not become spoken/caption dialogue.

5. [Q5] Cut any word-count/beat-count tuning from this round. The stated goal is craft-only, and `_otr_outline` already has substantial budget machinery that is orthogonal to these defects.