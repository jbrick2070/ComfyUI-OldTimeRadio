# LEAKING-WORDS STRATEGY -- R1-converged architecture (2026-06-25)

Synthesized from Claude's grounded anchor + a 3-model panel (gpt-5.5,
gemini-3.1-pro, deepseek-v4-pro), every claim checked against the real
`_otr_line_hygiene.py` / `_otr_ledger_scrub.py` / `_otr_config.py`. The panel
forced two corrections to the original problem doc (below). This is now a
DECISION, not a menu.

## Grounded root-cause correction (the panel earned its keep here)

The original doc blamed the `_NARRATION_VERBS` whitelist for the `Gasping,`
leak. **That was wrong.** Grounding `_otr_line_hygiene.py`:
- `scrub_leading_stage_direction` -> `_leading_stage_strip` GUARDS on
  `if not body or not body[0].islower(): return s` (line 271; docstring line 255:
  "the line starts lowercase"). It does NOT reference `_NARRATION_VERBS` at all.
- So `Gasping, "We're running out of time..."` is skipped because the lead is
  CAPITALISED, not because "gasping" is missing from a verb set.
- `_NARRATION_VERBS` is used by `is_third_person_action_clause` (line 412) -- a
  DIFFERENT detector for a different shape.

Therefore the fix is NOT widening the verb list (that would not touch this leak).
The fix is a narrow rule for the capitalised-participle-before-a-quote shape.

## CHOSEN ARCHITECTURE -- three layers; the correctness layer is deterministic + offline + mandatory

**Layer 1 -- Upstream prompt (defect-rate reduction, NOT enforcement).**
One line in the compose prompt: "spoken words only; no real-world proper names."
Cheap insurance. Never the enforcement layer (a weak local model still leaks --
that is the premise). [Option C, scoped down.]

**Layer 2 -- MANDATORY deterministic final verifier + repair (THE correctness
layer).** A per-line gate over the four leak classes using NARROW STRUCTURAL
extract-or-fail rules -- NOT broad destructive `-ing` scrubbing, NOT
verb-whitelist widening:
- *Stage-direction leak:* add a narrow rule for the grounded-missed shape --
  a leading capitalised participle + comma + a QUOTED remainder
  (`^["“]?[A-Z][a-z]+(ing|ed),\s*["“].+`) -> extract the quoted dialogue. The
  quote after the participle is the strong signal (dialogue is inside the quotes;
  the participle outside is the direction). Low false-positive because it requires
  the quote.
- *Caps-name vocative:* an ALL-CAPS token that EXACTLY matches a roster full name
  in a vocative position (`^FULLNAME[,!:-]+` / `[, ]+FULLNAME[.!?]?$`) ->
  title-case or drop the vocative. Do NOT generalise to mixed-case or in-line
  references. (verify-at-build: whether `scrub_self_vocative` already covers this
  and whether `scrub_ledger` even calls it -- GPT flagged it may be unwired.)
- *Malformed quotes:* fail-CLOSED to recompose on any INTERNAL odd quote;
  `sanitize_transcript_text` only balances a single edge wrapper today, so the
  internal-odd-quote class currently ships.
- *News-bleed (needs a policy, not a shape):* a per-episode named-entity policy --
  `allowed_proper_nouns` (cast + setting + premise/world) and
  `banned_source_proper_nouns` (entities from the raw news brief that were NOT
  abstracted into the fiction). A body line containing a banned source noun
  (e.g. "President Trump") -> reroll once. REQUIRES a news-abstraction step that
  turns raw news into fictional conflict objects and emits the banned set.
  (verify-at-build: `_otr_line_composer.build_allowed_roster` must NOT already
  whitelist news/key terms -- if it does, it would defeat this detector.)
- On any hit -> the existing one-repair/recompose budget; if exhausted, fail per
  an operator switch `strict_local_clean` (fail-closed) vs ship best-effort with
  telemetry (default).

**Layer 3 -- OPTIONAL online LLM cleaner (Option A, scoped to a typed repair).**
Off by default. Fires ONLY after Layer 2 flags a defect AND only when the operator
enables non-offline mode. A TYPED REPAIR, not a free rewrite: input = line +
speaker + cast + allowed/banned nouns; output JSON
`{clean_text, removed_spans, reason_codes, confidence}`; REJECT if empty,
over-diffed, quote-malformed, or it changes non-target words. Reuses the EXISTING
writer LLM plumbing (the OpenRouter/local slots) -> NO new workflow-JSON node.
Never runs on already-clean lines (no frontier degradation).

**Frontier writer (Option D) -- demoted to a product-tier recommendation, NOT the
correctness layer.** GPT shipped zero leaks today, but one day is not a guarantee.
The same Layer-2 verifier runs on frontier output (expected to no-op).

## Placement vs the audio spine (grounded invariant handling)

Layer 2/3 mutate spoken text, which is **AUDIO-AFFECTING** -- confirmed by the
existing pattern in `_otr_config.py` (lines 95/107: final-text-hygiene flags are
"DEFAULT OFF. AUDIO-AFFECTING ... ships dark"). So:
- The verifier runs in the WRITER, BEFORE TTS/freeze, so the audio is synthesised
  from the cleaned text (within-render byte-identity preserved: OBS audio == master
  mix, both from cleaned text).
- It ships DEFAULT-OFF / dark (it shifts the regression baseline), gets a live
  validation, then promotes -- exactly the existing audio-affecting-flag discipline.

## CUT
- **Option B (constrained generation / GBNF).** Not portable across the local
  transports (Ollama /v1 cannot take raw GBNF -- established in prior hardening),
  and "no real-world names" is a semantic constraint a context-free grammar cannot
  express. Cannot do the job; not model-agnostic.
- Any further investment in action-PRESERVATION (`split_stage_business` telemetry)
  for THIS problem -- the goal is preventing the leak from shipping, not archiving
  the leaked action.

## Acceptance gate (define before build)
A fixed regression corpus: the four observed shipped leaks (the `Gasping,` line,
the "President Trump" line, the "YUKI MARTIN" vocative, the unclosed-quote line)
as positive fixtures + NEGATIVE fixtures (a legitimate emphatic vocative, a
legitimate in-world proper noun from a premise, a non-stage `-ing` dialogue
opening like "Running to the door, I shouted..."). Require: 0 shipped instances of
the four classes across BOTH lanes AND 0 false-positives on the negatives.

## Invariants (unchanged)
Content-only (ledger schema frozen, no workflow-JSON node added -- Layer 3 reuses
existing plumbing); deterministic + offline at Layer 2 (the correctness layer);
model/transport-agnostic; must not degrade the already-clean frontier lane;
UTF-8 no BOM; SFW.
