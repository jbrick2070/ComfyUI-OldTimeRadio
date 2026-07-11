# Original Codex 56SOL: Lost and Found Frequency -- Pre-Comparison Design Fingerprint V1

**Design lock:** 2026-07-11  
**Status:** creative architecture locked before detailed study or comparison of
registered bank implementations  
**Source mode:** no-source; original synthetic fiction  
**Proposed coordinates:**

- `source_bank_id`: `original_codex56sol`
- `story_model_id`: `original_codex56sol_v1`
- `story_pipeline_id`: `acoustic_puzzle_v1`

## Listener promise

The listener hears a warm, brisk, fair-play mystery built entirely for audio.
Several callers bring ordinary lost objects or small community problems to a
fictional evening radio desk. Their stories appear unrelated. Spoken recollections
of sounds, timings, call order, repeated phrases, and verbal details gradually reveal
one benign causal chain. The listener
should be able to form the answer shortly before the characters do. The resolution
returns or repurposes what was lost and leaves the callers better connected.

The governing listener question is:

> Can I solve this harmless mystery by ear before the characters do, and does the
> answer help the people involved?

There is no villain, peril, crime, weapon, injury, blood, violence, swearing, real
brand, real public figure, franchise, adaptation, or borrowed story world. Stakes
come from deadlines, promises, misunderstandings, embarrassment, disappointed
hopes, and competing benign needs.

## Target-derived material choices

1. Because the mystery must be solvable by ear, every decisive inference has one or
   more earlier audible clue IDs, and every clue is carried by dialogue, vocal
   imitation, timing, call order, or a recurring phrase rather than an unseen visual
   fact or an unrendered stage direction.
2. Because the experience must stay gentle without becoming shapeless, the hidden
   cause is a benign chain of ordinary actions while the dramatic pressure comes
   from intersecting promises and a near-term community deadline.
3. Because the pleasure is recognition rather than defeat, the reveal must connect
   all caller threads and produce a practical act of repair, return, exchange, or
   cooperation.
4. Because it is an original radio form, the operating desk is a character location,
   not an announcer exposition device; a separate brief station ident may frame the
   program but may not solve the puzzle or join character dialogue.
5. Because modest local models must be able to run the lane, each artifact is typed,
   self-contained, and bounded. Long-form writing is split only at stable artifact
   boundaries, never by arbitrary prose truncation.

## Six-dimension fingerprint

### 1. Source strategy

No external source is fetched or adapted. A bank-specific initializer selects a
small combination of non-spoken prompt ingredients from a packaged palette using OS
entropy, unless the existing explicit seed override is active. The creative slot
turns those ingredients into the premise, cast, clue system, and story. Python never
authors spoken prose. Provenance is stamped as original synthetic fiction with no
citation, author, URL, or license claim.

### 2. Pass DAG and slot assignments

The creative DAG is organized around fair-play audio causality:

1. **Possibility fan:** `creative_fn` proposes four materially different harmless
   object/sound/community-knot candidates in one typed artifact.
2. **Knot choice:** `technical_fn` checks contract facts and reference closure for
   each candidate; deterministic code corroborates cited defects; `creative_fn`
   chooses and, if necessary, repairs one candidate.
3. **Audible truth map:** `creative_fn` authors the hidden benign causal chain,
   caller goals, clue schedule, false-but-reasonable interpretations, reveal, and
   helpful consequence.
4. **Fair-play proof:** `technical_fn` audits whether every reveal step is supported
   by an earlier audible clue and whether any required fact is visual-only or
   withheld. Python validates all cited IDs and ordering.
5. **Broadcast score:** `creative_fn` authors typed cast, scenes, shots, beats,
   opening/closing music cues, exact ledger IDs, and a closed line manifest without
   spoken text. Music is never evidence, and the lane emits no interstitial cue.
6. **Graph proof:** deterministic validation proves structural closure; a bounded
   `technical_fn` repair handles schema/reference defects without inventing prose.
7. **Full performance draft:** `creative_fn` writes every canonical spoken line for
   the accepted manifest and supplies character voice descriptions and sonic
   direction.
8. **Blind-listener pass:** `technical_fn` reviews the draft from the listener's
   information state, citing exact line/clue IDs for unsupported leaps, premature
   reveals, indistinct voices, forbidden content, formatting defects, or missing
   closure. Taste notes remain warnings.
9. **Broadcast retake:** `creative_fn` returns a complete replacement script when a
   corroborated repairable defect exists; keep-better comparison preserves the
   accepted draft if the retake introduces a contract regression.
10. **Delivery proof:** deterministic code revalidates the final graph, canonical
    text, clue coverage, safety evidence, hashes, and ledger ownership before
    assembly. A final `technical_fn` audit may propose evidence-linked defects but
    cannot override deterministic contract results or block on taste.

Each structured call has a bounded base / structural retry / typed-repair ladder.
No pass asks a model to count words, lines, clues, or coverage. Python measures and
reports concrete defects to the responsible slot.

### 3. Role and authority graph

- **Possibility maker (creative):** owns divergent premises; cannot approve itself.
- **Causal architect (creative):** owns the chosen hidden truth and clue schedule.
- **Fair-play examiner (technical):** may cite missing or late evidence; cannot write
  a replacement clue or make a taste verdict fatal.
- **Radio scorer (creative):** owns the dramatic graph and exact ledger manifest.
- **Performer-writer (creative):** owns all canonical spoken text verbatim.
- **Blind listener (technical):** reports evidence-linked comprehension and contract
  findings; artistic preferences are warnings.
- **Retake writer (creative):** may replace model-authored fields while preserving
  accepted IDs and source-free provenance.
- **Python contract keeper:** owns parsing, IDs, ordering, references, enums, counts,
  hashes, prompt-fit calculation, and corroboration; it never authors or improves
  prose.

### 4. Artifact handoffs

- `PossibilitySlate`
- `SelectedKnot`
- `AudibleTruthMap` containing typed `CallerThread`, `CausalStep`, `AudibleClue`,
  `Interpretation`, and `ResolutionLink` items
- `FairPlayReport` with evidence-linked findings
- `BroadcastScore` containing typed cast/scene/shot/beat/line-manifest/music items
- `PerformanceScript` containing one verbatim line item per accepted line ID
- `BlindListenerReport` containing hard findings and non-fatal notes separately
- `DeliveryReceipt` containing graph, clue-order, authorship, safety, word, slot, and
  canonical-text hashes
- the one shared production `Ledger`

All model-authored collections use concrete nested item types. Namespaced receipts
live in `meta["original_codex56sol"]`; fixed ledger rows gain no ad hoc fields.

### 5. Retry and audit topology

- Serialization or schema failure: same-slot structural retry, then typed repair.
- Broken IDs/references/order: technical repair receives the exact deterministic
  defect list; exhaustion fails closed.
- Unsupported or visual-only reveal: creative repair receives exact clue/reveal IDs;
  fair-play proof reruns; exhaustion fails closed.
- Objectively detected forbidden content, speaker labels, stage directions,
  quotation wrappers, or broken declared-name references: creative full-artifact
  repair receives the cited lines; all downstream proofs rerun; exhaustion fails
  closed. A reviewer may note a suspected resemblance to existing material, but
  resemblance is not an objective runtime gate and never becomes an originality
  score.
- A retake that breaks an already-satisfied contract is rejected in favor of the
  previously accepted draft. This is not fallback story text; both candidates are
  model-authored artifacts from the same lane and episode.
- No source, pack, bank, canned prose, or model-provider fallback exists.

### 6. Ledger-write strategy

The lane uses `content_owned_readonly`. The accepted score fixes the graph and IDs;
the accepted performance artifact supplies every spoken text field verbatim. Python
mechanically constructs the shared Ledger through its setters, selects valid voice
metadata under the live no-reuse policy, stamps final counts and hashes, and proves
scene/shot/beat/line/cast plus clue/reveal closure. The shared tail may create only
its documented delivery fields such as fresh `text_for_tts` and canonical-text
source hashes. A lane finalizer verifies pre-save ownership and post-save receipt
integrity.

## Orientation and closure

The first scene contains a short station ident followed by the desk character's
plain statement of the evening's practical problem. The ident is framing only. The
last scene performs the actual return/exchange/repair, lets the affected callers
recognize the causal chain, and closes with the desk character logging what became of
each object. Tests identify the exact orientation and closure line IDs and prove they
belong to the accepted line manifest.

## Safety and originality posture

Safety is imposed at every creative seam: no guns, blood, violence, swearing,
threats, cruelty, crime plot, injury, intoxicants, or sexual content. A technical
audit must cite exact artifact and text evidence; Python verifies that citations
exist and routes corroborated defects back to the creative slot. Deterministic scans
are backstops for explicit contract terms, never substitutes for model repair.

Originality is a design property, not a score or taste gate. Prompts prohibit real
brands, public figures, franchises, copyrighted characters, lyrics, adaptations,
and imitation requests. The bank creates a fresh fictional town, station, people,
objects, causal knot, and dialogue each run. Runtime validation checks concrete
contract violations and declared-name closure. A technical pass may record a
specific suspected external reference as a warning for the receipt, but it does not
ask a model whether the story is "original enough" and it cannot block production
on resemblance or taste.

## Explicit non-goals

- no external source, news claim, adaptation, quotation, or factual coda
- no villain, crime solution, danger, combat, or violent jeopardy
- no deterministic prose rewrite, trimming, padding, or canned rescue text
- no third LLM slot, direct provider call, model loader, or new credential
- no parallel ledger, alternate workflow, generated workflow, or new widget unless
  later live-contract grounding proves one unavoidable
- no reuse of another bank's runner, role names, pass graph, prompt language,
  validators, or dramatic frame

## Design-lock rule

Post-lock contract grounding may change field names, exact signatures, registry
shapes, validation calls, or shared-tail adapters. It may not change the listener
promise, causal-audio form, authority graph, or pass topology merely to resemble an
existing bank. Any material creative redesign requires a new fingerprint version and
an explicit reason derived from this bank's listener promise.
