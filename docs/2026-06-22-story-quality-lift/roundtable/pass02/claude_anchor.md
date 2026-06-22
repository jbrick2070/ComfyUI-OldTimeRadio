ROUND 2 -- CLAUDE ANCHOR REVIEW (coding plan / implementability; grounded vs real code)

VERDICT: yes-with-fixes. pass01's altitude is right; the coding-round gaps are (a) the floor
classifier + segmentation spec, (b) the DEFECT 2 rerun-input mutation, (c) two interface details. All
fixes reuse existing, tested code -- no new subsystem.

MUST-FIX BEFORE BUILD:

1. [pass01 sec 2 Tier 3 / sec 7 Q1] The deterministic floor's "classify the outside-quote span as
   action" is unspecified at code level. SPEC: reuse `_otr_line_hygiene`'s tested guard vocabulary
   (`_COPULA_MODAL`, `_DIALOGUE_STARTER`, `_PRONOUN_ROOTS`, `_OBJ_PREP`) to add
   `is_third_person_action_clause(span) -> bool`: TRUE iff (a) the span contains NO `_PRONOUN_ROOTS`
   token (no i/we/you/my/your/our, matched before an apostrophe), (b) its head token is a
   present-tense action verb (3rd-person -s or bare imperative form) and NOT in `_COPULA_MODAL`,
   (c) the lead is not a `_DIALOGUE_STARTER`, (d) word-count <= a small cap (reuse
   MAX_STAGE_PREFIX_WORDS-class bound). The floor strips an outside-quote span ONLY when this is TRUE.
   Reusing the guards is why FP stays low (they already kill "looks can be deceiving, John").

2. [pass01 sec 2 Tier 3] Pin the segmentation rule precisely -- it is what keeps b015/b017 OUT of the
   floor: the floor acts ONLY when the line's quotes are BALANCED (even count of matched pairs). It
   segments into matched-quote spans + outside spans, tests each OUTSIDE span with predicate #1, strips
   only the action spans. b005/b010/b012 = balanced `"<spoken>." <action>` -> floor-safe. b015 has an
   ORPHAN close-quote (unbalanced) and b017 has NO quotes -> NOT segmentable -> fall through to the
   Tier-2 reroll, never the floor. (Note: the pronoun guard would also independently spare b015's
   "...I'll be presenting..." span because it contains "I'll" -- but the balanced-quote precondition is
   the primary, simplest gate.) Return contract: `Tuple[str, bool]`, matching `_strip_stage_directions`.

3. [pass01 sec 3 DEFECT 2 / sec 7 Q2] The repair mechanism has a determinism trap: an OTR episode
   rerun is SEED-KEYED deterministic, so a plain `needs_full_rerun` would REPRODUCE the same incoherent
   arc -- an infinite/no-op loop. FIX: the rerun MUST mutate its INPUT -- inject the coherence
   constraint into the outline/`DramaticState` before the rerun (pin the antagonist's `character_b_wants`
   + add "antagonist stance is consistent across beats; no reversal without an explicit turn beat" to
   the Stage-3 beat prompt), and cap it (1 escalation, reuse the cascade's structural-failure single
   short-circuit -- do NOT add a new loop). Without the input mutation + cap this is not buildable.

4. [pass01 sec 4 DEFECT 3] "char_id resolves to a known cast id" needs the cast set in scope at each
   coercion point. VERIFY/SPEC: `init_lines_from_outline` has `char_id_by_name` in scope (usable);
   confirm `set_lines` and the `_otr_ledger_reviewer` role_mismatch repair can see the cast roster to
   test membership -- if a point lacks it, coerce only where the set is available + add the final
   pre-freeze consistency pass (which has the whole ledger incl `cast`) as the catch-all. Provide the
   membership predicate (char_id in cast ids).

SHOULD-FIX:

1. [pass01 sec 2 Tier 2] Ordering: the new trailing/embedded detector in `compose_line` (2015-2060)
   must run on the RAW draft BEFORE `strip_line_formatting`/quote-normalization, or the quote
   boundaries it relies on are already gone. The freeze floor then re-checks the frozen text
   independently. State this ordering in the chunk.

2. [pass01 sec 3] The richer stance fields (character/object/prior/new/missing-turn) must ride
   free-form `meta` -- schema is FIXED. The only typed change is a new `FailedDimension` value
   ("stance"); verify FailedDimension is a Literal/enum with no exhaustive match that a new value would
   break (`_otr_reroll._scope_and_hints` folds it as a `[dim]` prefix -- additive, safe). Do NOT add
   fields to `RerollTarget`.

3. [pass01 sec 0/6] Name the strong-model NO-OP fixture. If no checked-in opus/frontier good ledger
   exists, create a small hand-authored clean ledger fixture; the "zero strips/rerolls fired" assert
   has nothing to run against otherwise.

OPTIONAL / NICE-TO-HAVE:
- Hand-authored 6-line stance-reversal fixture independent of Chandra (so DEFECT 2 tests aren't tied to
  one generated episode); a 6-line clean fixture for the DEFECT 1 negative set.

CUT THESE (over-engineering):
1. Do NOT add a FailedDimension-driven LINE reroll for DEFECT 2 -- that reintroduces the line-level fix
   R1 rejected. The stance axis DETECTS; repair is the episode escalation with input mutation (MUST-FIX 3).

[ASSUMPTION] `needs_full_rerun` is reachable from the critic verdict and accepts an input mutation --
seam map shows `decide_escalation_scope -> needs_full_rerun` for EPISODE-scope structural failure, but
whether a stance-reversal critical maps to episode scope (vs line scope) is unverified -> R3 wiring.
[ASSUMPTION] FailedDimension is a Literal; adding "stance" is non-breaking -- verify no exhaustive match.
