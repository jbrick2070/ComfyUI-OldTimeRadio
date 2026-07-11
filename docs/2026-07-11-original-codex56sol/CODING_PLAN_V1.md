# Original Codex 56SOL -- Coding Plan V1

## Outcome

Build one new runnable, no-source OTR bank whose listener-facing form is
**Lost and Found Frequency**: a warm fair-play audio mystery in which several
ordinary lost objects or small community problems turn out to share one benign
cause. The answer must be recoverable from spoken evidence, and the resolution
must help the callers rather than defeat a villain.

Locked routing coordinates:

- `source_bank_id`: `original_codex56sol`
- `story_model_id`: `original_codex56sol_v1`
- `story_pipeline_id`: `acoustic_puzzle_v1`

The exact spelling above normalizes the operator's `orginal_codex` /
`orghional_codex56sol` messages into a valid, consistent registry ID.

The creative architecture is locked in `DESIGN_FINGERPRINT_V1.md`; the
post-lock comparison is in `DESIGN_COMPARISON_MATRIX.md`. Contract grounding may
change adapter details, but it may not turn this into an existing lane with new
names or prompts.

## Current coordination state

Implementation authorization update (2026-07-11): the operator explicitly
authorized non-GPU coding while the prior Sci-Fi Codex live run remains active.
The four prior-owner dirty files and all live processes remain excluded; any
overlapping edit and Chunk D wait for release. The constrained claim is recorded
in `docs/GO_FORWARD_PLAN.md` at base SHA
`26952a7ea64d61a2178485ac2708e350b52f9b48`.

Planning-artifact SHA-256 values after the two required wording repairs:

- `DESIGN_FINGERPRINT_V1.md`:
  `b2a0c800583868fc85063fa595f0fbc973c88235976aa38a9289acf5a9684bf9`
- `DESIGN_COMPARISON_MATRIX.md`:
  `12c2adf3554f824b5d54af94d83a91565765fced3408c0550dac225fb6be568d`
- `CODING_PLAN_V1.md` (wording-repaired content before this embedded receipt):
  `fcd166726f5645536fcaf0ffe8327809d0f2e22ae84771c3250f2391dbfa1e89`

Planning is safe, but implementation must not begin until the existing coder slot
is released:

- `docs/GO_FORWARD_PLAN.md` currently claims the only coder slot for the Sci-Fi
  lane campaign and names `banks.json`, `pipelines.json`, and writer dispatch as
  serialized collision surfaces.
- A live canonical `scifi_codex` / Gemma smoke is running on port 8000.
- The tracked tree is clean at `d39b134c`, which equals `origin/v2.0-alpha`, but
  the workspace contains active untracked smoke/review artifacts that belong to
  the other work and must remain untouched.

At slot release, the implementing window must first fetch/re-read the new HEAD,
capture `git status`, refresh all file/line anchors and design-comparison hashes,
and claim the slot in `docs/GO_FORWARD_PLAN.md`. It must not reset or delete the
current smoke processes or artifacts.

This dated planning directory is intentionally matched by the repo's local-artifact
ignore rule. When the slot is claimed, force-add the locked fingerprint, comparison,
and final coding plan (or mirror the final plan to a tracked dated root file) in the
first docs commit so the implementation does not depend on an unpushed local-only
plan.

## Non-negotiable behavior

- Every generated title, premise, description, visual prompt, music prompt, and
  spoken line comes from an accepted artifact produced through the supplied
  `creative_fn` or `technical_fn`. Python may parse, validate, assign IDs and
  enums, select validated voices, calculate counts/hashes, and copy accepted text
  verbatim; it may not author, paraphrase, trim, pad, combine, or improve prose.
- Only `creative_fn` and `technical_fn` are used. There is no model loader, direct
  provider/API call, third slot, new key, or fallback model.
- The creative contract forbids guns, blood, violence, threats, crime plots,
  injury, swearing, intoxicants, sexual content, real brands/public figures,
  franchises, copyrighted characters, lyrics, adaptations, and imitation.
- Safety is designed into every creative seam. Deterministic checks cite defects
  and route them to bounded creative repair; they never censor or rewrite text.
- Originality is a design property, not a model score. Suspected resemblance can
  be recorded as a warning, but it is not a fatal taste gate.
- `target_words` is an advisory scale request and receipt only. Models do not
  calculate or report word, line, item, or coverage counts. Python measures; no
  trim, padding, line deletion, or length failure exists.
- No external source is fetched or adapted. Non-empty `source_ref` fails closed.
  `custom_premise`, when supplied, is an operator hint to the invention pass, not
  a source article.
- No SFX lane is introduced. Spoken evidence uses wording, vocal imitation,
  timing, call order, and repeated phrases. Opening and closing music are
  non-evidentiary bookends. No `music_inter` row or interstitial audio path is
  emitted.
- There is one shared production `Ledger`, one canonical workflow, no alternate
  graph, no generated substitute, no canned story, and no bank fallback.

## Root fixes required before the bank lands

These are not optional polish. Without them the new lane's receipts would make
claims the live code does not actually prove.

### R1. Generic content-authorship proof

Problem: `content_owned_readonly` is selected generically, but freeze hashing and
read-only verification are hard-coded to `meta.fable2.proof_map` in
`nodes/_otr_freeze_cascade.py`. Every other content-owned lane currently hashes an
empty proof and receives no shared accepted-artifact verification.

Root fix:

1. Add `nodes/_otr_content_authorship.py`, a pure helper defining one shared
   `meta.content_authorship` receipt:
   - schema version;
   - owner bank;
   - accepted artifact IDs and SHA-256 values;
   - exact `line_id -> SHA256(raw UTF-8 accepted canonical text)` rows;
   - exact coverage summary for all non-skipped voiced lines.
2. Validate owner/source-bank agreement, unique proof IDs, no missing or extra
   voiced-line proofs, referenced artifact existence, and hash equality against
   live canonical text.
3. Change freeze receipt hashing and read-only validation to consume only this
   generic contract. Missing, malformed, partial, extra, or mismatched proof is
   terminal for `content_owned_readonly`.
4. In one atomic green chunk, change EVERY read of `meta.fable2.proof_map` in
   the freeze cascade (`_sha256_proof_map`, the row-validation loop, and all
   entry/exit/capability hash consumers -- more than two sites), proven by an
   AST/no-reader regression (or a code-and-test-only search) that detects both
   dotted access and nested forms such as
   `.get("fable2").get("proof_map")`; documentation may legitimately retain
   the dotted phrase, so a repo-wide zero-match grep is not valid proof. Then
   migrate
   Fable2, Sci-Fi Codex, Gemini, and Sonnet to stamp the generic receipt from
   their accepted final artifact plus exact canonical `line_id/text` rows before
   shared assembly/finalization. Keep truly lane-specific evidence namespaced,
   but remove the Fable2-only shared proof lever rather than supporting two
   truths or permitting an intermediate empty-proof state.
   Pin the migration's accepted text sources to the live runner objects:
   - Sci-Fi Codex: final validated `script: ScriptArtifactV4` after any P9 rewrite;
   - Gemini: final validated `drafts[scene_id]: SceneDraftV4` values after each
     bounded scene rewrite, joined only by their accepted `OutlineV4` beat IDs;
   - Sonnet: final validated `events: list[DraftLineV4]` after Warden repair and
     P6 attestation expansion; Python may collect those typed lines but not alter
     their model-authored text.
5. Treat each runner/finalizer as the receipt producer and the writer
   before-save check, save/reopen check, and downstream freeze cascade as
   consumers. Prove identical receipt coverage and hashes at all three consumers.

### R2. Writer-tail ordering and truthful telemetry

Problems:

- consistency metadata is currently written after `tail_finalizer.before_save`
  can freeze the ledger;
- slot-call accounting is snapshotted before shared reflection calls and optional
  story-spine work finish;
- `gen_params_by_phase` describes legacy cast/outline/dialogue phases even for a
  custom dispatched runner;
- the refine gate is Fable2-worded and late instead of applying to every custom
  runner that does not support refine re-entry.

Root fix:

1. Run consistency evaluation and all other writer metadata mutation before the
   lane finalizer. This last-mutation-boundary subset lands before or atomically
   with the R1 receipt migration, never afterward.
2. Stamp slot transitions, calls-by-slot, calls-by-helper, per-phase transitions,
   and generation parameters only after the final writer LLM call.
3. Derive custom-lane generation receipts from the scheduler/helper journal and
   runner pass journal; the shared K-block must never stamp legacy cast, outline,
   or dialogue phases for a dispatched runner that did not execute them.
4. Reject refine mode once, early and generically, for every dispatched runner
   that declares no refine support. Keep legacy inline behavior unchanged.
5. Test that `before_save` is the last mutation boundary for a content-owned
   ledger and `after_save` observes a byte-identical serialized metadata hash.

### R3. Cross-layer routing proof

Problems:

- a pipeline can reference a declared seam that its pack does not supply;
- `executable=true` registry metadata does not prove a writer runner-map entry;
- the workflow-link helper does not prove saved input names exist in live
  `INPUT_TYPES`.

Root fix:

1. During registry cross-reference validation, require every pass `seam_ref` to
   exist in the selected pack. For non-production/custom seams, require exact
   three-way parity between pack prompt keys, `pipeline.declared_seams`, and the
   union of custom pass references. Production-allowlisted seams remain governed
   by `bank.required_seams` and shared inline routing; do not reject them merely
   because pipeline pass metadata does not reference them. A repair seam must be
   represented by an explicit pass row, not inferred from prose.
2. Add a cross-layer invariant test: every runnable non-inline pipeline has one
   `_RUNNER_BY_PIPELINE` entry, and every map entry resolves a registered
   executable pipeline.
3. Add a permanent canonical audit that every saved node input name is present in
   the live node's `INPUT_TYPES`, in addition to link referential integrity and
   widget-vector checks.

### R4. Honest local-seed ingress

Problem: any runnable bank with both `fetcher` and `interpreter` empty enters the
hard-coded `original_radio` spark path before custom runner dispatch.

Low-blast, contract-compliant fix for this bank:

1. Register `original_codex56sol_local_seed` in
   `nodes/_otr_source_payload.py` as a no-network local/package fetcher.
2. It validates a packaged constraint deck and performs the lane's only
   source/constraint entropy draw (voice selection later uses the existing
   shared cast-seed policy and is a separate, unaffected draw), using OS entropy
   unless the existing `OTR_ORIGINAL_SEED` override is active. It serializes the full immutable, hash-covered `ConstraintDraw` under
   `source_meta.constraint_draw`, then returns `SourceFetchResult` with the exact
   seven-key compatibility payload plus honest synthetic `source_meta` and
   `source_rights` sidecars. The runner's D0 reconstructs and validates that same
   draw; it never calls an RNG or silently re-draws.
3. Its `seed_source` is the existing truthful `original_llm` category. Note the
   printed HUD/news label does NOT come from `seed_source`: `_build_news_payload`
   falls back to `RSS Auto-Fetch` for every non-`custom_premise` seed_source
   unless `source_label` is supplied. The honest `Original (LLM)`-class label
   must come from the fetcher payload's `source` field and the bank's
   `source_material_label` default; a test pins the printed label itself, not
   `seed_source`.
4. Add a scalar bank default declaring `custom_premise_mode=operator_hint`.
   Immediately after resolving the bank, `_resolve_inputs` reads that mode. For
   `operator_hint`, it raises `SourcePayloadContractError` on non-empty
   `source_ref` before deck/entropy/network/model work, bypasses the generic
   `elif custom` source-override branch, and always follows the registered-fetcher
   path. After `normalize_fetch_result`, it attaches a non-empty premise only as
   sibling `source_meta.operator_hint`, outside `constraint_draw` and all draw/
   deck hashes. All other banks retain their current source-override order. Never
   substitute another seed or bank.
5. Tests cover all four blank/non-blank combinations of `source_ref` and
   `custom_premise`, with spies proving rejected references do no deck/entropy/
   network/model work and a premise never suppresses the constraint draw. A
   fixed seed produces byte-identical `constraint_draw` data with blank versus
   non-blank hints, while P1 receives the exact hint separately.
6. Tests prove zero network/model/file I/O at import or node discovery and prove
   `_otr_original_radio` is never called for this bank.

The bank row uses `source_kind="original_synthetic"`, an empty interpreter,
the local-seed fetcher above, empty production `required_seams`, and non-empty
scalar defaults for `story_form_label`, `source_material_label`,
`title_form_label`, `hud_origin_label`, `credits_source_line`, and
`custom_premise_mode`. The printed labels say original machine-generated fiction;
they do not claim a news source, human author, public-domain status, or license.

## Creative execution DAG

Every structured model call uses a base call, a lower-temperature JSON-syntax
retry only when syntax failed, and a typed repair prompt. Every ladder is bounded
and fails closed. Content-repair loops are separately bounded and rerun all
downstream proofs.

| Stage | Owner | Accepted artifact / action | Blocking rule |
|---|---|---|---|
| D0 constraint ingress | Python | typed `ConstraintIngress`: reconstruct and validate the fetcher's one immutable `ConstraintDraw`, then separately validate/carry the optional sibling operator hint | no RNG here; missing/malformed fetch receipt, invalid deck identity, explicit `source_ref`, or entropy/seed contract failure raises `SourcePayloadContractError` before any LLM call |
| P1 possibility fan | creative | `PossibilitySlate` of typed `PossibilityCard` items: ordinary objects, benign cause, caller needs, spoken clue modes, deadline, helpful outcome, fictional-name declarations | schema/declared-name/safety defects go through the structured ladder; no exact candidate count and no taste score |
| P2 contract triage | technical | `SlateTriage` with one typed evidence-linked assessment per returned candidate | Python corroborates IDs, exact quoted evidence, forbidden patterns, and visual-only dependencies; uncorroborated taste remains notes |
| P3 causal loom | creative | `AudibleTruthMap`: selected card, caller threads, causal steps, clue schedule, reasonable interpretations, reveal links, orientation, and helpful closure | must select an eligible card; objective defects return to creative repair; the technical slot never writes replacement story material |
| P4 fair-play proof | technical | `FairPlayReport` referencing exact causal-step, clue, and reveal IDs | Python proves every reveal reference resolves to an earlier spoken clue and no required fact is visual-only/unrendered; exhaustion fails closed |
| P5 broadcast score | creative | `BroadcastScore`: title, premise, setting, time of day, sound palette, cast descriptions, scenes, shots, beats, line intents, visual prompts, and named opening/closing music specs, with no spoken text | Python compiles IDs and boundaries, validates graph closure, and returns exact structural defects for one creative score repair |
| D1 manifest compile | Python | `ClosedLineManifest` with IDs, order, role, cast/scene/shot/beat ownership, boundary, clue/reveal refs, orientation ID, and closure ID | purely mechanical; no prose written; any unresolved reference stops |
| P6 performance draft | creative | `PerformanceScript` containing the final title and one verbatim `SpokenLine` for every manifest ID | exact manifest coverage, no extra IDs, accepted cast only, no labels/directions/wrappers, and all concrete safety checks must pass or route to creative repair |
| P7 blind listener | technical | `BlindListenerReport`: evidence-linked hard contract findings separated from non-fatal comprehension/taste notes; receives a pre-reveal packet made only from accepted verbatim lines | only deterministically corroborated, repairable contract defects can block; solvability, pacing, and enjoyment notes never become a hidden fatal gate |
| P8 broadcast retake | creative | hard-repair branch: a complete replacement `PerformanceScript` when P7 corroborates a contract defect; notes-improvement branch: at most one optional complete retake when P6 is already contract-valid | hard branch: P6 is invalid and repair rejection/exhaustion is terminal, with no fallback to P6; notes-only branch: reject an invalid optional retake and retain valid P6; every candidate reruns manifest, safety, graph, and authorship checks; no patch prose |
| P9 final audit | technical | `FinalContractAudit` over the current complete script, with typed exact evidence and separate warnings | never audits word count or originality; a corroborated hard finding makes its one creative repair and rerun mandatory, and rejection/exhaustion is terminal; warnings alone do not invalidate an otherwise contract-valid script |
| D2 assembly | Python | shared `Ledger`, namespaced lane receipts, generic content-authorship receipt, mutable EpisodeCanon-compatible object, and lane `TailFinalizer` | full graph closure, voice readiness, hashes, Phase 0/10, save/reopen parity, and downstream delivery must pass |

The blind-listener packet is a view, not rewritten content: Python selects accepted
verbatim lines that occur before the declared reveal and copies their IDs/text. It
does not summarize them.

## Typed artifact rules

`nodes/_otr_original_codex56sol.py` owns strict Pydantic models. Every
model-authored collection nests a concrete item type; no collection of things uses
`list[dict]`, `dict[str, Any]`, or `Any`. Identifier-keyed mappings are allowed only
when the key is the actual organization.

Minimum concrete types:

- `ConstraintDraw`, `ConstraintIngress`
- `FictionalName`, `PossibilityCard`, `PossibilitySlate`
- `CandidateFinding`, `CandidateAssessment`, `SlateTriage`
- `CallerThread`, `CausalStep`, `AudibleClue`, `Interpretation`,
  `ResolutionLink`, `AudibleTruthMap`
- `FairPlayFinding`, `FairPlayReport`
- `CastConcept`, `SceneConcept`, `ShotConcept`, `BeatConcept`, `LineIntent`,
  `MusicBookend`, `BroadcastScore`
- `ManifestLine`, `ClosedLineManifest`
- `SpokenLine`, `PerformanceScript`
- `ListenerFinding`, `ListenerNote`, `BlindListenerReport`
- `ContractFinding`, `FinalContractAudit`
- `OriginalCodex56SolTailParts` and `OriginalCodex56SolFinalizer`

Prompt seam, worked example, type schema, parser, post-validator, and repair prompt
must agree. Tests parse every worked example and compare required key paths against
the live models. A shared test rejects shapeless model-authored fields and any
model-reported count field.

## Prompt and context-fit policy

- All model instructions live in
  `nodes/story_packs/original_codex56sol/original_codex56sol_v1.json` under
  pipeline-declared `prompt_stages`; Python contains no fallback system prompt.
- Every call runs inside `slot_scheduler.helper_context(<pass_id>)` and uses only
  the pass's declared slot.
- Read the effective per-slot context cap from the same live scheduler/cache entry
  used by generation. Do not create a parallel context table.
- Calculate fit for base, possible syntax retry, and worst-case typed repair. The
  repair form includes the failed artifact and evidence list and is treated as the
  largest request.
- Mark all lane message lists as must-fit so the local generator raises instead of
  left-truncating. Remote/local response reservations use artifact size drivers:
  caller-thread and clue counts for the truth map, manifest-line count for the
  script, and finding count plus script size for repair. `target_words` may inform
  an advisory scale but is never the sole reservation driver or a gate.
- If the live scheduler does not expose its resolved cap read-only, add that narrow
  accessor to the scheduler rather than calling a stale catalog override.

## Ledger assembly and closure

The lane selects `content_owned_readonly` by omitting
`line_composer_system`. It calls the supplied `Ledger` setters in this order:

1. `set_cast`
2. `set_scenes`
3. `set_shots`
4. `set_beats`
5. `set_lines`
6. `set_music`
7. shared count stamping and save

`clips` remains the initialized empty list for downstream ownership. There is no
`sfx` table and no music sentinel line. `music` contains exactly two rows. Python
assigns the structural `cue_id` and `placement` from bookend ordinal position;
the model's `MusicBookend` supplies only the accepted music description and
generation prompt. The opening row receives `opening/opening`, and the closing
row receives `closing/closing`. Their anchors resolve to the orientation or
closure line. There is no interstitial cue.

Placement vocabulary is deliberately `opening`/`closing`, the values
`stable_audio_theme._canonical_placement` recognizes directly. The sibling
Codex/Gemini lanes declare `Literal["open", "inter", "close"]`, which that
mapper does not recognize. Their cue IDs are `music_open` and `music_close`,
which are likewise not recognized by `_canonical_placement`, so those rows fall
through to `interstitial`; there is no cue-ID rescue. The schema-parity sweep
must not force this lane onto those literals. This lane keeps the exact
`opening/opening` and `closing/closing` pairs.

The bank's runtime closure validator proves independently of shared freeze:

- every required top-level collection is a list;
- IDs are non-empty and unique per table;
- every shot resolves to one scene;
- every beat resolves to one shot and carries that shot's scene;
- every beat line ID resolves exactly once;
- every voiced line resolves to exactly one beat and shot;
- line and beat speaker/character identities agree;
- every character line resolves to cast;
- `shot_start`, `beat_start`, and `continue` match actual transitions;
- orientation and closure line IDs exist in the manifest;
- both music anchors resolve and cue ID/placement pairs are exactly
  `opening/opening` and `closing/closing`;
- every clue/reveal reference resolves in allowed order;
- no extra, missing, null, skipped, or unsupported-role row exists.

The performance artifact supplies every canonical text verbatim. After all
accepted creative repair, the runner stamps the generic content-authorship receipt
from that final artifact and the exact canonical `line_id/text` rows before shared
assembly/finalization. The writer tail then stamps fresh `text_for_tts` plus its
canonical-source hash without changing canonical text.

## Cast and delivery

- Cast is small and variable within the live 1-6 limit; `num_characters` is an
  advisory scale request recorded with actual cast size, not a model count gate.
- The station announcer uses `char_id="announcer"` only for a brief ident/frame and
  does not solve the puzzle or join character dialogue.
- The desk operator and callers are ordinary cast characters. The desk operator's
  first scene orients the practical problem; the last scene performs the
  return/exchange/repair and logs the result.
- Python selects real voice metadata from the live voice registry under the
  existing seed policy. Non-announcer presets are valid distinct `v2/*` values;
  the announcer receives a valid announcer preset. No hard-coded or invented voice
  ID is allowed.
- CastLock must preserve and validate the lane-owned voices. Missing presets,
  collisions under no-reuse, orphan character lines, stale `text_for_tts`, or TTS
  bus count mismatch fail loudly.

## Safety and rights validation

Create `nodes/story_rules/original_codex56sol.json`. JSON owns the bank-specific
explicit vocabulary/patterns; global profanity remains global. Content-owned runner
validation must inspect every authored surface, not dialogue alone:

- title and premise;
- character names and descriptions;
- scene/shot descriptions and visual prompts;
- line intents and spoken lines;
- music descriptions and generation prompts.

Concrete forbidden-term, stage-direction, wrapper, speaker-label, closed-name, and
reference defects are emitted as `field_path + item_id + exact_span + category +
allowed_correction`. Python verifies the span exists and sends the entire owning
artifact back to `creative_fn`; it never edits the span itself. Semantic reviewer
findings that cannot be objectively corroborated remain warnings.

Provenance is synthetic and truthful:

- `meta.source_bank = original_codex56sol`
- empty external URL/author/citation fields;
- deck version/hash and selected non-spoken constraints under `source_meta`;
- synthetic-original rights label only, with no fabricated license URL;
- printed HUD/credits disclosure from bank defaults;
- no factual coda and no claim that the story is news or an adaptation.

## Exact implementation surface

New files:

- `nodes/_otr_content_authorship.py`
- `nodes/_otr_original_codex56sol.py`
- `nodes/story_packs/original_codex56sol/original_codex56sol_v1.json`
- `nodes/story_packs/original_codex56sol/constraint_deck.json`
- `nodes/story_rules/original_codex56sol.json`
- `tests/test_content_authorship.py`
- `tests/test_original_codex56sol_registry.py`
- `tests/test_original_codex56sol_artifacts.py`
- `tests/test_original_codex56sol_runner.py`
- `tests/test_original_codex56sol_ledger.py`
- `docs/2026-07-11-original-codex56sol/SOURCE_BANK_PREFLIGHT_MATRIX.md`
- `docs/2026-07-11-original-codex56sol/SOURCE_BANK_PREFLIGHT_RECEIPT.md`

Existing files expected to change:

- `nodes/OTR_LedgerScriptWriter.py`
- `nodes/_otr_source_payload.py`
- `nodes/_otr_story_routing.py`
- `nodes/_otr_freeze_cascade.py`
- `nodes/_otr_scifi_fable2.py`
- `nodes/_otr_scifi_codex.py`
- `nodes/_otr_scifi_gemini.py`
- `nodes/_otr_scifi_sonnet.py`
- `nodes/story_packs/banks.json`
- `nodes/story_packs/pipelines.json`
- `tests/test_fable2_assembly.py`
- `tests/test_fable2_runner_ladders.py`
- `tests/test_freeze_policy_readonly.py`
- `tests/test_scifi_lane_schema_parity.py`
- focused shared tests for routing, freeze policy, source payload, widget selection,
  workflow input names, slot telemetry, and current custom lanes
- `docs/GO_FORWARD_PLAN.md` and `docs/HANDOFF_LOG.md` after the coder slot is claimed

`nodes/_otr_story_routing.py` must list `constraint_deck.json` as the exact sidecar
for this bank so the pack sweep does not parse it as a story pack.

Expected unchanged file:

- `workflows/otr_canonical.json`

The bank selector is registry-derived, the default remains `science_news`, and no
node/input/widget/link/default change is designed. Capture the canonical SHA-256
immediately before implementation and require byte identity after the bank lands.
If live grounding proves a workflow change truly unavoidable, stop that chunk,
update canonical JSON in the same code change, and run all positional widget/link/
input audits before proceeding.

## Green chunk sequence

### Chunk A -- true mutation boundary and generic content authorship

- First move consistency and every other writer metadata mutation before the lane
  finalizer and prove `before_save` is the last mutation boundary.
- Add the generic receipt helper; change both cascade consumers and migrate every
  current content-owned producer atomically from its accepted final artifact.
- Remove the Fable2-only shared proof path; never land a temporary dual or empty
  proof contract.
- Add tamper, coverage, owner, byte-hash save/reopen, and freeze-policy tests.
- Run focused tests, the full Windows suite, and Bug Bible.
- Commit and push immediately; verify `HEAD == origin/v2.0-alpha`, AST, no BOM,
  and no zero-byte files.

### Chunk B -- remaining tail/routing truthfulness

- Fix slot receipts, dispatched-runner per-phase generation telemetry, generic
  capability-aware refine rejection, custom pack/declared/pass seam parity,
  runner-map parity, and the live workflow input-name audit.
- Preserve production-allowlisted seam routing and pin `original_radio`'s existing
  13-stage pack as a non-regression fixture.
- Preserve byte behavior for legacy and existing custom lanes except for corrected
  receipts/order.
- Run focused tests, full suite, and Bug Bible; commit/push/verify.

### Chunk C -- atomic `original_codex56sol` lane

Land in one change:

- local seed fetcher and validated constraint deck;
- `_PACK_SIDECAR_FILENAMES_BY_BANK` entry for `constraint_deck.json` in the same
  change as the bank directory;
- bank row before `custom_source_bank`;
- executable pipeline and every declared pass/seam;
- prompt pack and story-rules pack;
- runner, strict artifacts, validators, repair ladders, ledger assembly, finalizer;
- writer lazy import and `_RUNNER_BY_PIPELINE` entry;
- all bank-owned tests;
- one atomic test resolving the bank row, pipeline, prompt pack, story rules,
  constraint-deck sidecar, registered fetcher, and writer runner entry;
- `runnable=true` and `executable=true` in the same change.

Do not commit an unwired runner, dormant pack, false executable flag, or
`runnable=false` staging state. Run focused tests, full suite, Bug Bible, canonical
audits, and offline canonical selector proof; commit/push/verify.

### Chunk D -- live-smoke root fixes

After a selective process/port reset and fresh UTF-8 boot, run the real canonical
workflow at 30 words. Any failure gets a root-cause code/prompt/schema fix, focused
regression, full suite, Bug Bible, and immediate commit/push before the next live
attempt. No post-hoc asset move, canned fallback, or temporary workflow is allowed.

After the 30-word gate is green, run at least one normal-length story roll to expose
context fit, multi-scene closure, clue ordering, cast separation, and pacing that a
micro-smoke cannot exercise. Stop at contract and listening convergence rather than
polishing indefinitely.

### Chunk E -- preflight receipt

Execute every hard item in `docs/SOURCE_BANK_PREFLIGHT.md`, write the numbered
`ID | status | evidence` matrix, hash the fingerprint/comparison/matrix, and fill the
final receipt. Commit and push this evidence only when every hard item is PASS or a
specifically allowed N/A.

## Focused test matrix

### Shared-root tests

- generic authorship happy path plus missing/extra/duplicate/tampered proof rows;
- all existing content-owned banks stamp the generic receipt;
- every Fable2 assembly/runner/freeze assertion migrates from
  `meta.fable2.proof_map` to `meta.content_authorship`, with no stale shared read
  path or dual-proof allowance;
- legacy packs still resolve `legacy_full`;
- finalizer is the last mutation boundary, with a byte-identical serialized hash
  from `before_save` exit through `after_save` entry;
- slot call totals include runner, reflections, and any shared tail calls;
- custom runner telemetry contains no legacy-only phase names;
- refine mode rejects before runner work;
- every pass seam exists in its pack and custom pack keys, declarations, and pass
  references have exact three-way parity in both directions: no declared custom
  seam is unused and no custom pack key is undeclared/unreferenced; production
  seams remain governed by their existing contract;
- runnable custom pipelines and writer runner-map entries are bijective;
- every canonical saved input name exists in live `INPUT_TYPES`.

### Bank tests

- registry, pack, pipeline, story-rules, sidecar, and three coordinate parity;
- the bank's non-empty fetcher makes `_bank_has_no_source_contract(...)` false,
  and actual writer dispatch never imports or calls `_otr_original_radio`;
- registry row remains before the intentionally non-runnable
  `custom_source_bank` last row;
- local seed payload has exact `SOURCE_PAYLOAD_KEYS`, all strings, non-empty
  `seed_text`, no network, one and only one constraint draw, deterministic
  explicit-seed replay, OS entropy otherwise, honest provenance, and loud
  `source_ref` rejection before any work;
- all four `source_ref`/`custom_premise` presence combinations obey precedence;
  `custom_premise` becomes only `source_meta.operator_hint` and cannot suppress
  the fetcher;
- every prompt example validates and every pass uses its declared slot/helper;
- the strict-lane JSON-representability/schema-parity sweep imports
  `nodes._otr_original_codex56sol` alongside Codex, Gemini, and Sonnet;
- all base/retry/repair prompts fit the resolved per-slot cap or fail before model
  invocation without truncation;
- no shapeless model-authored collection or model-reported count field;
- graph closure happy path plus one rejection fixture per invariant;
- no stage direction, speaker label, whole-line quote wrapper, gun, blood, violence,
  or profanity across every authored surface;
- repair success and exhausted-repair failure, with no Python prose mutation;
- P8 hard-repair exhaustion is terminal, while an invalid notes-only optional
  retake preserves the already-valid P6 artifact;
- no originality score/gate and no bank/model/source fallback;
- exact accepted-artifact line coverage and tamper rejection;
- correct `outline_view`, mutable canon, final title override, `run_story_spine=false`,
  and finalizer contract;
- content-owned freeze, fresh delivery hashes, valid unique voices, CastLock
  preservation, exact character/announcer TTS bus consumption;
- exact `opening/opening` and `closing/closing` cue pairs, resolving anchors, and
  no interstitial/SFX path, driven through StableAudioTheme, SceneSequencer, and
  EpisodeAssembler with zero inline insertion;
- selector includes the bank while canonical widget slot 23 stays `science_news`;
- canonical file byte hash remains unchanged.

## Required verification

Use the Windows venv with `PYTHONUTF8=1` and `pytest -q -p no:cacheprovider`.
Long full-suite jobs run in the background and are polled rather than exceeding the
command ceiling.

After every code chunk:

1. focused tests for touched contracts;
2. full repo suite;
3. Bug Bible from the separate survival-guide repo with the relative test path;
4. AST parse for touched Python;
5. UTF-8/no-BOM/no-zero-byte audit;
6. JSON duplicate-key parse and round-trip;
7. `HEAD == origin/v2.0-alpha` after immediate push.

Before live work:

1. canonical `OTR_WorkflowValidator`;
2. link referential integrity;
3. live input-name audit;
4. live `INPUT_TYPES` vs positional widget count;
5. canonical byte-hash comparison;
6. offline canonical API dry-run selecting `original_codex56sol`;
7. delete temporary prompt dumps.

Record the canonical pre-implementation hash and the post-Chunk-A, post-Chunk-B,
and post-Chunk-C hashes in the preflight receipt; all four must be byte-identical.

Live gate:

1. wait for the current smoke/coder slot to finish;
2. selectively kill only ComfyUI/server/harness processes by command line and port;
3. confirm port 8000 is free and VRAM is at baseline;
4. boot through the UTF-8 launcher;
5. load `workflows/otr_canonical.json` and select `original_codex56sol`;
6. verify both semantic LLM slots recorded real calls;
7. require saved ledger, generic authorship proof, bank graph proof, fresh
   `text_for_tts`, accepted Phase 10 verdict, and CastLock/TTS success;
8. require `obs_publish OK` and `Test-Path` on the exact final asset under
   `otr\obs`;
9. inspect the produced transcript and receipt for forbidden content, declared-name
   closure, comprehensible clue order, distinct voices, and a coherent helpful
   ending.

## Completion definition

The bank is complete only when:

- its design fingerprint and comparison remain independent;
- only the supplied LLM slots authored all creative text;
- every model artifact and repair path is typed, bounded, must-fit, and fail-closed;
- the shared ledger is fully closed and its accepted text is hash-proven;
- no guns, blood, violence, swearing, copyrighted adaptation, or factual-source
  pretense appears in the produced episode;
- the canonical selector reaches the bank without changing the shipped default;
- full suite, Bug Bible, canonical audits, live micro-smoke, and normal-length roll
  are green;
- the published asset exists in `otr\obs`;
- the source-bank preflight matrix and final receipt contain zero hard failures;
- every green commit is pushed and local HEAD equals origin.
