# Production Sprint Lessons

Use this guide when planning any OTR coding sprint, including story banks,
dynamic visual direction, audio/SFX layers, model integrations, and workflow
changes. The project rules in `AGENTS.md` and `CLAUDE.md` remain authoritative.

## 1. Define ownership before implementation

For every artifact and field, name:

- the one authoritative writer;
- every consumer;
- whether it is authored, derived, or measured;
- its lifecycle boundary (draft, accepted, frozen, or post-freeze extension);
- its durable storage and replay receipt.

If an input list becomes output rows, state the mapping exactly: one row per
input item, one singular owned value per row, exact reference coverage, and
which collections live at the top level. List the exact fields allowed in each
nested row. Do not assume a model infers ownership or nesting from a JSON
schema.

## 2. Keep five representations in lockstep

Every structured LLM pass has five representations that must agree:

1. base prompt;
2. typed schema;
3. worked fixture/example;
4. parser and deterministic validator;
5. repair prompt.

A change to one requires an audit of the other four. Explicitly forbid common
pseudo-shapes: numbered fields, `_secondary`/`_tertiary` variants, schema-path
keys, singular-vs-list aliases, and valid collections nested at the wrong
depth.

## 3. Separate authored decisions from mechanical repair

Python may safely repair only facts already determined by accepted artifacts:
IDs, ordering, references, exact enums, hashes, routing metadata, duplicate
references with an unambiguous first owner, and relocation of values whose
destination is uniquely declared.

Python must not invent dialogue, motives, clues, story causality, visual taste,
or sound-design decisions. If placement or meaning is ambiguous, return the
defect to the owning model and fail closed when the bounded ladder exhausts.

A deterministic repair must validate the complete downstream contract before
returning. A partially valid schema object can consume the repair rung and
prevent the intended LLM fallback.

## 4. Design retry ladders by failure class

- JSON syntax failure: same prompt at lower temperature may help.
- Typed/schema failure: send the failed artifact, exact error, schema, and
  explicit correction rules to typed repair.
- Semantic/graph failure: name the objective invariant and the owning item;
  do not ask a lower-temperature retry to guess.
- Incomplete repair JSON: retry the exact cached repair prompt only when the
  configured budget permits it.

Log which rung ran and why. Never silently truncate, silently coerce authored
content, or fall back to a canned artifact.

## 5. Size context from the real artifact

Do not derive output or context budgets from `target_words` alone. Include the
actual size drivers: accepted line count, evidence rows, graph width, schema
overhead, prior artifact size, and repair envelope size. The repair prompt is
often the largest call.

Resolve the true context cap for the selected model and fail loudly when a
provenance-sensitive prompt cannot fit. A model's native context and the
project's configured safe context are separate facts; measure both.

## 6. Test model diversity, not just repeatability

A prompt that works on one local LLM is not production-qualified. Different
families fail differently: integer IDs, extra fields, renamed forbidden keys,
wrong nesting, copied request envelopes, enum synonyms, or over-wide output.

Qualification ladder for model-sensitive work:

1. unit fixtures and full regression gates;
2. canonical 30-word end-to-end smoke on at least two different local model
   families and one configured cloud/frontier creative lane;
3. the same pairings at 120 words;
4. only then, 720-word qualification or bakeoff.

Record concrete model labels, slot assignments, prompt IDs, repair counts,
ledger paths, and final asset paths for every leg.

## 7. Prove the real workflow end to end

Always load `workflows/otr_canonical.json`. Code that is not wired into that
file is dead. Any node, input, widget, or link change updates and validates the
canonical JSON in the same commit.

An API `SUCCESS`, idle VRAM, or a resident server is not final proof. Verify:

- the saved ledger and required receipts;
- the canonical episode asset under `otr/episodes/<ep>/`;
- `obs_publish OK`;
- the final file exists under `otr/obs/`.

Conversely, resident VRAM after `Prompt executed` is not evidence that a run is
still alive. Check the queue, runner exit code, history result, log, and file.

## 8. Keep GPU experiments clean and comparable

Before every headless run, selectively stop only ComfyUI and its harnesses,
clear port 8000, and confirm baseline VRAM. Never blanket-kill Python. Boot with
UTF-8 through the canonical launcher. Save one server log and one leg log per
run, and use the watchdog for long renders.

Change one meaningful variable per comparison. Do not mix a model change, a
prompt change, a quant change, and a context change into one unexplained result.

## 9. Treat live failures as reusable project knowledge

Every bug that actually fails a live smoke, soak, or published episode gets an
append-only entry in `docs/PROD_BUG_LOG.md` with symptom, root cause, fix,
verification idea, and Bug Bible candidacy. Dev-only catches are fixed and
tested but do not enter the production log.

Promote recurring, machine-checkable failure classes to the Bug Bible later.
Also update the relevant planning guide and preflight so the next sprint avoids
the bug before implementation.

## 10. Preserve concurrency boundaries

One window owns code and the canonical workflow. Parallel windows may perform
read-only investigation or docs-only scoping, but they do not edit active code,
tests, registries, or workflow JSON. Preserve unrelated dirty-tree files and
stage only the owned change.

After every code change: focused tests, full Windows suite, Bug Bible, AST/JSON
checks, commit, push to `v2.0-alpha`, and verify `HEAD == origin` before live
qualification.

## 11. Prove semantic provenance, not just structural references

An ID can remain valid while its meaning disappears. If artifact A carries a
typed clue and artifact B carries only that clue's ID, a later model can emit a
self-consistent but unrelated story while every graph/reference validator
passes.

For content-owned pipelines, build a small immutable grounding contract from
the accepted source/draw. Carry exact positive anchors through every authoring
and retake seam, and verify their spoken evidence at the coordinates that own
them. Structural coverage, an LLM audit, a frozen ledger, and a published video
cannot override a deterministic semantic-fidelity failure.

## 12. Test channel isolation by changing one channel

Comments saying “visual style does not affect story” are not proof. Run the
same mocked story pipeline twice while changing only the visual-style value and
assert every captured story-model message is byte-identical. Apply the same
method to SFX, voice, render, and other downstream channels.

The story may feed visual direction; visual direction must not feed back into
story authorship unless a separately named, explicitly designed mode owns that
decision. Store selectors in separate ledger namespaces so a forensic audit can
distinguish correlation from causation.

## 13. Persist enough accepted artifacts to locate first drift

Response hashes prove identity only when the response is still available. They
cannot reveal where meaning first changed after a production run. Persist a
bounded receipt of accepted typed artifacts and their grounding evidence:
selected input, truth/plan, score/manifest, final authored output, audit, and
line/span coordinates.

Do not retain raw prompts, rejected outputs, or unlimited retry prose by
default. The goal is replayable accepted-state forensics, not an unbounded model
transcript.

## 14. Close every nested object, not only the artifact root

Listing required schema paths tells a model what must exist; it does not tell
the model which plausible bookkeeping fields must not exist. For every strict
structured artifact, state the exact key set at each nested ownership boundary,
especially music, media, provenance, delivery, and file/path-shaped objects.

Pair that prompt contract with a narrow structural normalizer only where Python
can prove the removed fields are non-authoritative metadata. Such a normalizer
must preserve all required authored values byte-for-byte and rerun the complete
strict schema plus semantic/graph validators. Never project an arbitrary model
object onto a schema when an unknown key could carry story meaning.

## 15. Make ordered graph topology executable

An array can contain every required typed row and still violate the graph's
meaning through order alone. State sequence rules with a concrete valid and
invalid example (for example A, A, B versus A, B, A), identify which manifest
owns the canonical group order, and enforce the invariant in Python.

Never reorder authored chronological rows as a structural repair: sequence is
story, even when every row object remains byte-identical. Repair only mechanical
identifier topology when it is provably unambiguous -- for example, split a
reopened A/B/A shot into A/B/A-return by cloning the shot metadata and retagging
the later run -- while holding the authored row sequence fixed. Then rerun the
complete graph, semantic, grounding, and landmark validators. Reject or fall
through to typed repair whenever the identifier split is ambiguous.

## 16. Run deterministic projections at every attempt boundary

A typed-repair prompt factory sees the failed base response, but it does not
necessarily see or normalize the model's typed-repair response. If a safe
projection exists only inside that factory, a model can repeat the identical
mechanical defect on the final repair attempt and bypass the projection.

Hash the actual raw model response first, then apply the same narrow projection
at the slot-output boundary for every base, syntax-retry, and typed-repair
response. Return the projected artifact only when its strict schema and complete
semantic/grounding validators pass; otherwise retain the raw response and let
the ordinary ladder report or repair the real remaining defect.

## 17. Put typed structural repair at the accepted-object boundary

Raw-string cleanup is appropriate for unambiguous JSON-shape defects, such as
lifting a declared collection from the wrong nesting level. It is not enough
for a projection whose safety depends on the fully parsed artifact and its
complete graph/grounding validator. Apply that kind of repair in the
schema-validated post-check over the same typed object that will be returned to
the caller.

Test the boundary itself: deliberately disable or bypass the earlier raw
normalizer, then prove both a base response and a typed-repair response still
accept only after the typed projection preserves authored values and clears all
validators. This prevents a duplicated pre-parse helper from becoming a
false-green proxy for the production acceptance path.

## 18. Group safe graph ownership repairs at one typed boundary

For a schema-valid score, related exact invariants such as contiguous shot
ownership and one-to-one clue ownership should share the same typed
post-validation boundary. The model owns the first authored placement and all
meaningful prose; Python may only derive a later mechanical identity or remove
an exact repeated reference when the complete typed graph proves that no story
meaning is lost.

Keep raw cleanup limited to defects that prevent parsing at all. Test each
typed repair with that raw path disabled, including a typed-repair response,
so a pre-parse helper cannot falsely appear to qualify the production guard.

## 19. Compose independent safe projections before global validation

Do not make each narrow mechanical repair demand that every other invariant is
already clean. A score can contain two independently provable defects, such as
a reopened shot run and a duplicate clue reference. Factor a projector from its
full-validation wrapper, apply only a small declared set of disjoint projectors
in deterministic order, and then run the complete graph, grounding, and safety
checks once over their shared result.

The composition must be bounded, must preserve authored content and chronology,
and must remain fail-closed for an unknown, missing, ambiguous, or still-invalid
condition. Add a regression where the base and typed-repair responses contain
the complete defect combination; testing each repair in isolation is not
evidence that the production boundary can combine them.

## 20. Repair localized semantic omissions with bounded typed patches

Do not resend an entire accepted artifact merely because one LLM-owned leaf
misses an immutable semantic anchor. Whole-document regeneration increases
context pressure, expands the failure surface, and can force the model to
recreate already-valid structure instead of correcting the actual omission.

Define a minimal patch schema that names the allowed targets and fields. Python
may derive those targets from the immutable contract and must verify exact
one-for-one coverage, literal anchor inclusion, and no changes outside the
declared patch scope. The model still authors the replacement prose. After the
merge, rerun the complete artifact, grounding, and authored-surface validators;
an unknown or broader defect must remain fail-closed rather than being squeezed
through the narrow tool.

## 21. Preserve accepted invariants in every replaced patch field

A patch that replaces an entire leaf value can remove a different invariant that
was already correct in that same value. Include both the newly required facts
and every immutable fact currently present in a selected target. This matters
when one beat carries multiple contracts, such as a reveal or closure beat that
also owns a clue.

Validate the *merged canonical artifact* in the structured-call post-validator,
not only the patch's local key set. A local patch can be schema-valid and carry
all newly requested literals while still breaking a graph, grounding, safety, or
authorship invariant elsewhere in the target field. Return the precise merged
error to the typed repair ladder and remain fail-closed if it cannot be cleared.

## 22. Apply localized semantic repair at every artifact boundary

Do not stop the bounded-patch pattern at a planning artifact. A complete
performance script can be just as expensive and unreliable to regenerate for a
single missing immutable phrase as a complete score. Keep the authoring model
responsible for replacement prose, but create a separate typed patch seam for
each artifact whose individual leaves have clear ownership and a full validator.

For spoken-script patches, target line IDs only; retain every valid literal
already spoken by that line, require every newly required literal, and rerun the
complete script graph, safety, and grounding checks after merge. The tool must
not become a broad retake in disguise: no title, roster, ordering, or unplanned
line edits are permitted.

## Sprint receipt

Record this at the end of every production sprint:

```text
SPRINT RECEIPT: PASS | FAIL
scope:
authoritative_writers:
durable_artifacts:
canonical_workflow_hash:
focused_tests:
full_suite:
bug_bible:
model_pairings:
30_word_receipts:
120_word_receipts:
720_word_receipts:
live_ledgers:
published_assets:
prod_bug_entries:
head:
origin:
remaining_risks:
```
