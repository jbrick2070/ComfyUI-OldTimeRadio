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
