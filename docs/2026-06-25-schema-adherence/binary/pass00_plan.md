# BINARY-DECISION DECOMPOSITION FOR LEDGER INTEGRITY -- ADDENDUM (for the roundtable)

Complementary to the converged schema-adherence plan (pass04_plan.md). That plan
hardens TOLERANCE -- accept whatever JSON an arbitrary writer model emits. THIS
addendum proposes the second model-agnostic lever the operator raised:
DECOMPOSITION -- where a structured pass is failure-prone across models, replace
the complex-object emission with a BINARY (or tiny-enum) decision any model
answers reliably, then assemble the ledger deterministically.

## THE OPERATOR'S PRINCIPLE
LLMs across families (local mistral/gemma AND frontier GPT/Claude/Grok) are
RELIABLE at binary classification and UNRELIABLE at complex structured emission
(live proof: Claude Opus tripped the normalize_length schema, burning ~90k tokens
on doomed retries). A binary question -- "is this clause spoken DIALOGUE (A) or a
STAGE DIRECTION (B)?" -- has a one-token answer space, the most schema-light call
possible, so it is the MOST model-agnostic primitive available. Use it to keep the
ledger intact: every ledger line should be clean, correctly-attributed spoken
text with no leaked stage directions / narration.

## GROUNDED CURRENT STATE (nodes/_otr_line_hygiene.py + _otr_repair_prompts.py)
The pipeline ALREADY tries to separate dialogue from stage business -- but
deterministically + brittlely, never as a clean binary LLM decision:
- `split_stage_business(text) -> (dialogue, action, reason)` (L7, :544): splits ONLY
  the "balanced-quote outside-span" class via regex (`segment_double_quotes` +
  `is_third_person_action_clause` + word-count ceilings); "not confident" -> returns
  `(text, "", "")` and PUNTS to the strip chain.
- `is_stage_direction_only(text)`: regex detector so the caller can RECOMPOSE a
  pure-stage-direction line.
- `scrub_parentheticals` / `scrub_leading_stage_direction` /
  `strip_quote_anchored_stage_direction`: deterministic strips.
- `narration_leak_repair` (a typed factory): a FULL LLM repair when action/plot
  prose leaks into a visual description.
So today the classification is EITHER brittle regex (model-agnostic but
inaccurate on the cases it punts) OR a full structured repair (accurate but
schema-heavy + the very failure mode pass04 is mopping up). There is no clean
binary-decision rung in between.

## THE PROPOSAL
A shared, ultra-tolerant BINARY-DECISION primitive used as the ESCALATION above
the deterministic chain (regex-first -> binary-LLM on the cases regex ABSTAINS ->
deterministic fallback if the binary parse is junk):
- `binary_decide(*, slot_fn, question, text, ...) -> Optional[bool]` -- a one-call,
  schema-free pass whose output is a single A/B (or yes/no) token; parse = the
  first decisive token (ultra-tolerant, reuses the pass04 strict-first spirit but
  there is barely any schema to violate); returns None (-> deterministic fallback)
  only on a genuinely undecidable reply. Deterministic given (model, prompt, seed).

## CANDIDATE LEDGER-INTEGRITY APPLICATIONS (the panel should prune)
1. **Dialogue vs stage-direction (the operator's example).** Escalate the cases
   `split_stage_business` PUNTS (the non-balanced-quote / undelimited classes) to
   `binary_decide("is this a spoken line or a stage direction?")`; if "stage
   direction", route to the existing RECOMPOSE seam. Keeps every ledger line clean.
2. **Edit vs no-op (the payload_null failure class).** The Script Doctor pass keeps
   emitting null payloads on annotation-only edits (BUG-LOCAL-275 -> the bespoke
   `payload_null_repair`). Reframe: ask `binary_decide("does this row need a real
   replacement, yes/no?")` BEFORE asking for the replacement object -- no payload
   schema to null out.
3. **Speaker membership.** `binary_decide("is the speaker of this line one of the
   locked cast?")` as a cheap integrity gate complementing the Levenshtein
   phantom-name resolver.
4. **Beat over-length (normalize_length).** The pass Opus tripped re-emits
   segmented beat objects. Reframe the hard part as a per-boundary binary "split
   here? yes/no" + deterministic re-assembly -- no segmentation object to mis-shape.

## CONSTRAINTS (carried from the main plan)
Model-agnostic + transport-agnostic (any local/remote model; never force a
choice); DETERMINISTIC + byte-identical for the local DEFAULT happy path (a new
LLM call where there was none changes output -> the binary call must live ONLY in
the ESCALATION path, fired only when the deterministic classifier ABSTAINS, so the
default path is unchanged); fail to a DETERMINISTIC fallback, never silent-wrong;
REUSE the `_otr_line_hygiene` chain (no regex duplication); offline-verifiable;
100% local-capable; UTF-8 no BOM; SFW.

## OPEN QUESTIONS FOR THE PANEL
1. Is a SHARED `binary_decide` primitive worth it, or is this per-pass? What output
   contract is the most model-agnostic + parseable (bare "A"/"B"? "yes"/"no"? a
   1-field bool object)? How tolerant must the parse be (first-decisive-token)?
2. Byte-identity: confirm the binary call belongs ONLY in the escalation path
   (regex abstains) so the local default is unchanged -- or is there a case for
   always-binary? What is the gate?
3. Cost/latency: a binary call per ambiguous line adds calls. Is the
   regex-first / binary-escalation / deterministic-fallback ordering the right
   cost-accuracy trade, or should binary fire only on a confidence signal?
4. Which of the 4 applications are genuinely better as binary vs which should stay
   deterministic (the regex floor is good enough)? Prune ruthlessly.
5. Failure mode: a binary call can still return prose / "it depends". The parse +
   the deterministic fallback must make that safe. How does this compose with the
   pass04 tolerance ladder -- is `binary_decide` a sibling of `structured_call`, or
   a thin wrapper over it with a 1-field bool schema?
6. Does decomposition REPLACE any complex pass (e.g. normalize_length) or only
   AUGMENT it? Where is replacing the object with binary decisions a net
   simplification vs a new moving part?

## OUT OF SCOPE
The pass04 tolerance work (complementary, already converged); prose quality; the
coda-bridge validator; the news-brief artifact.
