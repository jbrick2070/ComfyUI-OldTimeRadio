# Problem statement: contract adapters for workflow LLM calls

## Decision to investigate

OldTimeRadio's source-bank workflow asks local models to create both authored
story material and strict structured artifacts. Models such as Gemma can make
good creative choices, but may repeatedly miss exact mechanical contracts:
JSON ownership, graph ordering, one-to-one clue assignment, enum values, and
cross-artifact IDs. A second free-form repair call often repeats the same
mistake because it shares the original model's representation and prompt
surface.

We should investigate small, versioned training adapters that teach a base
model the project's recurring artifact grammars and repair patterns. The
adapter is a quality and efficiency aid, not an authority: deterministic
validators and the canonical workflow remain the only acceptance gates.

## The problem

Today, each source-bank lane relies on a combination of prompt contracts,
bounded retries, typed validation, and narrow Python projections. This works
only after individual live defects have been discovered and hardened. It
creates three costs:

1. Local models spend long calls generating repair attempts that repeat exact
   contract failures.
2. Each new source bank can rediscover an already-known structured-output
   defect in a different artifact shape.
3. Runtime receipts record the outcome, but the model has not learned from the
   accepted/rejected history in a controlled, reproducible way.

The desired system should make a local creative model more likely to emit a
valid first artifact while preserving original story voice and never weakening
the existing fail-closed protections.

## Proposed direction

Train and load a small adapter per **base-model family and artifact contract**,
not one broad adapter for all OTR work. Examples:

- `gemma4_broadcast_score_contract_v1`
- `gemma4_truth_map_contract_v1`
- `mistral_nemo_critic_contract_v1`

Each adapter would be trained from provenance-approved examples:

- accepted structured artifacts with their exact system/user contract;
- minimal before/after repair pairs where Python can prove the edit preserved
  authored meaning;
- validator labels explaining the exact rejected invariant; and
- deliberately varied source-bank content so the adapter learns a grammar,
  not a single story.

At model load, the workflow resolves an adapter only when its manifest matches
the selected base model, artifact schema version, prompt-pack hash, and
validator-suite version. The call receipt records the resolved adapter and its
evaluation evidence. A mismatch means no adapter, never a guessed fallback.

## Non-negotiable guardrails

1. Python remains the acceptance authority for schemas, IDs, ordering,
   ownership, grounding anchors, and immutable provenance. An adapter can
   reduce retries; it cannot approve output.
2. Creative prose remains LLM-authored. Python may only make a proven
   mechanical repair from an already accepted typed graph.
3. Training, evaluation, and promotion data must be separate. Production test
   fixtures and live qualification seeds belong in held-out evaluation, not
   training.
4. Every adapter is immutable and versioned: base-model fingerprint, training
   corpus manifest/hash, schema and prompt hashes, trainer settings, evaluation
   report, and approval status are durable receipts.
5. The adapter must be independently removable. Disabling it returns to the
   current prompt plus validator behavior without changing the canonical
   workflow's story semantics or saved-widget positions.
6. Visual, SFX, voice, and media adapters are out of scope. This proposal is
   limited to textual workflow artifacts.

## Success criteria

An adapter is eligible for a guarded pilot only if it demonstrates all of the
following on held-out material:

- a materially lower invalid-first-response rate for its one artifact type;
- lower average model calls and latency without a lower validator standard;
- no drop in grounding, authorship, or story-quality blind-judge scores;
- exact reproducibility of adapter selection and receipt stamping; and
- clean 120-word qualification before it participates in a 320 or 720 word
  bakeoff.

An adapter must be rejected or demoted if it increases semantic drift, hides a
validator failure, narrows story variety, or behaves differently when only a
downstream visual-style selector changes.

## Evaluation design

Use four distinct partitions:

| Partition | Purpose | May train? |
| --- | --- | --- |
| Curated accepted artifacts | Learn artifact grammar | Yes |
| Minimal deterministic repair pairs | Learn recurring mechanical corrections | Yes, after provenance review |
| Development validation set | Tune adapter settings | No |
| Canonical qualification seeds and production regressions | Promotion gate | No |

Compare base-only and base-plus-adapter calls with the same model, seed,
source-bank payload, and canonical workflow. Measure schema validity,
post-validator validity, repair count, latency, accepted-artifact hashes,
grounding coverage, and blinded story quality separately. A better JSON pass
rate is insufficient if story quality or originality falls.

## Suggested phased plan

1. **Corpus audit.** Inventory accepted artifacts and exact deterministic repair
   receipts. Exclude raw rejected prose, unverified outputs, and any artifact
   without durable source provenance.
2. **Offline baseline.** Freeze a held-out contract suite for P3 truth maps and
   P5 broadcast scores across Gemma and Mistral. Record current pass/retry/
   quality metrics before training anything.
3. **Adapter experiment.** Train one narrow P5 contract adapter against a
   pinned Gemma base. Evaluate offline only; do not load it into ComfyUI.
4. **Shadow mode.** Generate adapter and base candidates for the same input;
   validate both, retain neither as production authority, and compare receipts.
5. **Guarded pilot.** Add an explicit adapter dropdown/input at the end of the
   relevant node's widgets, wire it into `workflows/otr_canonical.json`, and
   qualify a single source bank at 120 words with the adapter enabled and
   disabled.
6. **Promotion or rejection.** Promote only after reproducible improvement
   through the 120 -> 320 -> 720 ladder and blind story comparison. Otherwise
   retain the evidence and remove the runtime selection.

## Open questions

- Which local adapter format/runtime can coexist with the current 4-bit model
  loader without exceeding the 14.5 GB VRAM ceiling?
- Should the first pilot teach the full P5 schema, only the error-prone
  ownership subgrammar, or use constrained decoding before fine-tuning?
- What minimum number and diversity of accepted artifacts is enough to avoid
  overfitting one source bank's voice?
- Which evaluation metric best detects an adapter that produces valid but
  less-original stories?
- How should source-bank prompt-pack changes invalidate or require retraining
  an adapter?

## Recommendation

Start with a research-only, adapter-free corpus and baseline phase. The first
runtime pilot should be a narrow Gemma P5 contract adapter in shadow mode,
because P5 has the clearest typed boundary, durable accepted artifacts, and
recent live regression evidence. Do not begin with a general story-writing
adapter or give any adapter power to bypass Python validation.
