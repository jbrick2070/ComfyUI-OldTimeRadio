# R4 convergence contract

R3 is complete with Gemini3.8FlashHigh + CursorGrok4.6High. Read r3/judgment.md
and adaptive_coverage_followup.md. The following grounded amendments OVERRIDE
the preserved complete R2 integration contract below where more specific.
Review only NEW material; confirm independently settled chunks for implementation.

1. Grammar: fresh state before EVERY actual model.generate, including min_p
   compatibility retries. Remove stateful json_schema_parser/prefix_allowed_tokens_fn
   closure attributes; keep schema_model metadata. Construct JsonSchemaParser with
   CharacterLevelParserConfig(max_json_array_length=0), then restore0 on its
   effective config AFTER prefix builder sets alphabet. Both are necessary:
   root arrays capture their maximum at construction, nested arrays after builder.
   Test both with25items through real prefix_fn, plus explicit schema caps.
2. Native routing: identify curated remote/GGUF handles using existing suffix
   normalization/catalog before native HF side effects. Canonical HF root still
   precedes native validate_model_id's possible cache scan. No duplicate catalog.
   Fit inspection is on every bound/unbound scheduler closure, CPU-only and no
   generation counter increment. The organizer uses technical/P0. Remote/GGUF
   lack native inspection: preserve existing estimated admission with honest
   estimate labels and actual provider-capacity handling.
3. Source coordinates: one transient SourceDocument per nonempty raw field;
   Python string start_char/end_char offsets with coordinate version, UTF-8 hashes.
   Those offsets are not byte indices. Use an iterative queue with strict
   nonempty source reduction and note-group reduction, no fixed minimum window,
   recursion-depth or total work-item cap.
4. Review coverage: record actually reviewed source-span/candidate-part PAIRS,
   not merely coverage of each axis. Whole-candidate qualification requires full
   pair coverage within its declared scope; skipped/inferred-irrelevant pairs
   remain unresolved. Never hide quadratic review cost to obtain PASS: preserve
   honest partial scope when complete checking is unavailable. Validate evidence
   inside THAT CALL's delivered raw spans and candidate projection. A quote found
   elsewhere in saved source does not authorize an edit from an unseen packet.
5. Reviewer calls remain inside the existing author's post-validator after fast
   structural checks, with their own max2 structured_call and no recursive author
   validator. Inner exhausted schema/parse checks yield uncertain receipts;
   StructuredCallFailedError is NOT in outer _ATTEMPT_ERRORS. Never unwrap inner
   retryable capacity errors inside the author validator, reroll the creative slot
   because review failed, or put failed review JSON into author repair. Real
   provider/OOM/cancel failures retain their ownership/type. Only grounded direct
   contradiction returns the string that triggers existing author repair. Add
   tests for each branch. Do not add a second external author retry engine.
6. Partial cleaner replacement is a LIST of span-id/replacement pairs: duplicate
   keys must remain detectable. Empty localized replacements may remove a genuine
   direction; validate the final spoken row. Whole-row conversion still requires
   separate model authorization; count those calls in existing receipts.
7. Writer tail calls abort OR reconcile. Public abort uses guarded rollback and
   retains a terminal receipt; any later reconcile returns it without resealing.
   Idempotence can use a stored receipt, no new enum framework needed. A no-op
   transition does not itself delete the degradation key, contrary to one review.
8. Append a default base_prompt_hash compatibly with CURRENT NamedTuple attributes,
   indexing and old construction. That does not preserve arbitrary six-value
   unpacking; repo search found none. Current source receipt replaces old metadata
   in both cache/fresh branches; keep base versus final transformation scope honest.

Additional independent readers Einstein/Euler grounded coverage and installed
LMFE0.11.3 respectively. Root sole production editor. S2 is pushed; diagnostic
fixes are independently under finished-diff QA. No live media is claimed.

---

## Preserved R2 integration contract, with amendments above

Base59a40131 on v2.0-alpha. Root sole production editor. GO_FORWARD is the only
queue. R1/R2 completed Gemini3.8FlashHigh + CursorGrok4.6High. Read R2 judgment
for accepted/rejected claims. This document supersedes R1 final where amended;
R1 final retains detailed tests/owners. Review actual Windows code, read-only.
No architecture code is implemented. S2 boundary correction is already pushed,
72focused tests pass, full14134pass/54baselinefail/183skip/1xfail, canonical
unchanged. Do not propose reimplementing S2. Reviewers should focus on remaining
must-fix integration defects, not generate another generic wish list.

## User steering and priority

After R2 dispatch the user proposed first-pass organization: the LLM summarizes
large input into a working brief and uses smaller/additional passes if needed.
Model chooses organization; actual tokenizer/provider measures fit. Preserve raw
source and complete coverage, selected acts and flexible cast. No RSS, length,
duration or taste gate, silently unread tail or invented replacement story.
New adaptive-reading design and A2 dependency below must receive review now.

Implementation order after convergence: grammar/P1 shared fix; actual native
capacity and fit preparation; precise cleaner scope; source-preserving adaptive
authoring/scene prompts; Mac diagnostic and reported log corrections; canonical
recovery. Each code chunk gets focused/full/Bible checks and one finished-diff CLI
read, commit+push, parity. Existing credits/Original work stays later in GO_FORWARD.

## A. Grammar/P1 integration: settled changes, verify exact owners

`get_cached_transformers_schema_constraint` caches only tokenizer/tokenizer_data;
remove stale by_schema entries. Fresh parser/prefix is acquired INSIDE writer's
inner `generate_fn`, and inside `constrained_generate_fn`, on each invocation.
Never capture it in either outer factory. After real prefix builder, set
parser.config.max_json_array_length=0 on its live effective config. Preserve
alphabet and explicit maxItems/maxLength. No global environment changes.

ProviderCapacityMessages adds `_otr_unbounded_json_field=True`.
`make_degeneracy_criterion(..., max_open_string_tokens: int|None=2048)` skips
open-string tracker when None but retains cycle detection. Both bound transports
snapshot marker before message normalization and pass None when marked.
`_inherit_generation_contract` preserves ANY original list subtype even when
repair factory returns string: coerce target string to one user-message list,
then copy original subtype and markers. Do not branch on My Story or import its
schema. Correct halt messages and open_string_tokens in both bound transports;
unbound loader routes are cycle-only, so don't falsely claim an open-string case.

P1 `_pass_treatment` binds its local callable once using existing callable
`_otr_bind_schema(StoryTreatment)` when present. P2/P3 original callable stays.
Remote/GGUF existing dispatch unchanged. All old parsing/typed repairs still run;
no partial JSON salvage. MyStory attempt receipts retain raw_output from actual
return, plus SEPARATE raw_completion field on transport error. Current finally
counts only raw_output, so it never interprets truncated transport bytes as a
completed proposal. No widget/node/canonical surface changes.

Test real LMFE builder with21+items and explicit cap control, fresh state over
repeat closure/retries/errors, tokenizer scan-once, old histories/schema classes
collectible. Test >2048 nonrepeating tokens with marker, default-bound and real
cycle controls, marker preservation for string/list repairs, no bank branch,
binder scheduler accounting, and failed-completion evidence separation.

## B. Actual native capacity and exact prompt preparation

Revisit A2 from GO_FORWARD before adaptive chunk sizing. Shared catalog parser
handles dict or loaded config, positive native decoder context under text_config
before top-level max_position_embeddings/n_positions/n_ctx. Invalid/bool/nonpositive
values are absent. Actual native capacity beats curated estimate. Explicit
positive context pin yields min(native,pin); missing/invalid/nonpositive pin is
None. Support explicit positive values below512. Do not mutate model config.

Snapshot normalized pin once per native request. Native HF cache key is
(policy.cache_key(), normalized_pin) at BOTH reuse and publication; same textual
number with whitespace/leading zeros reuses; changed/unset pin invalidates reuse.
Keep policy.cache_key and GGUFLoadConfig.reuse_key unchanged for their other
consumers. No environment reread between lookup, loading and publication.

Canonical HF root must precede request_slot.validate_model_id (which may scan
uncurated cache), context discovery, download and snapshot selection. Pass same
hub_root explicitly; retain remote/GGUF early routing, with no new HF work there.
Finalize native capacity from actual AutoConfig and loaded decoder after first
download too. Remove load_llm's max_position_embeddings mutation. Preserve an
explicit load_llm(context_cap=...) parameter as a caller pin; don't pass an
estimate to that parameter. Add truthful capacity provenance and vram_priced_ctx
to entry. Native HF VRAM estimate remains weights-only; advertised window is not
KV allocation. Preserve admission-before-reuse and epoch/orphan safeguards.

Remove project-minimum model-window rejection in _otr_loader_backends and native
adapter precondition; small valid model windows are usable when prompt fits.
MIN_OUTPUT_TOKENS default becomes1; explicit min/require_full atomic contracts
remain. OpenRouter/Comfy already pass explicit floors; unknown/remote capacity
estimation/spend reservation stays disclosed existing behavior, not falsely fixed.

Factor a CPU prompt-preparation helper at native transport owner: exact same
system-role normalization, chat-template kwargs and tokenization used to generate.
Return prepared CPU inputs and primitive prompt_tokens/capacity/provenance; move
to device only at generation. Probe must use same helper so no duplicate estimate.
Expose scheduler-owned fit inspection via the existing slot callable's optional
capability, no new node/widget: acquire configured slot, prepare, return primitives,
drop entry/inputs, no generation-call increment. Actual generation revalidates
after slot changes. Respect selected technical/creative models; do not silently
reroute a technical check to creative to avoid legitimate configured swaps.

Tests owners: model_catalog_scan, loader_slot_primitives, native_text_decoder_load,
llm_runtime_policy, orphan_cache_epoch_and_deadline, loader_body_profiles,
context_window_precondition, generation_budget, gate_prices_the_policy_context,
hf_env_offline/model_catalog_download. Cover nested config, first download,
canonical root before admission scan, normalized pin cache invalidation, no native
config mutation, real one-token room, explicit atomic refusal, GGUF unchanged.

## C. Precise shared cleaner scope

Keep original row immutable. Add _otr_spoken_text_policy.f1_finding_spans using
real unmodified regex match offsets; existing Finding summaries unchanged.
Judge schema may carry optional original offsets; otherwise require one exact
occurrence. Ambiguous/repeated quote without localization grants no scope. A
small bounded localization request can resolve actual original offsets; never
authorize all occurrences automatically. Union only overlapping authorized spans.

Partial repair schema returns replacements keyed by approved span IDs. Require
exactly one replacement per ID, no unknown/duplicate IDs; validate before applying.
Python interleaves unchanged original slices and MODEL-WRITTEN replacements. This
is editing model-authored material, not Python inventing dialogue. It guarantees
untouched whitespace/punctuation by construction and avoids requiring the model
to copy the whole protected row perfectly. Whole-line conversion uses text only.
Remove _call_repair's blanket whitespace collapse; validate nonempty without
altering accepted bytes. No2000-character cap on a conserved input row.

Whole-row complaint has dedicated module-level authorization schema with verdict
already_spoken|localized_defect|whole_row_direction|unresolved and localized spans.
Use structured_call(max_attempts=2), original/speaker/context. Already-spoken:
no edit; localized: verify intervals; direction: full-row model conversion;
unresolved/malformed: original retained with receipt. No keyword blacklist.

Keep initial authorized intervals immutable across existing _MAX_ATTEMPTS.
Candidate reread receives candidate in marked neighbor; original repair context
continues to show original. Remaining previous-candidate diagnostics cannot widen
authority. Scope failure never reaches clean or best-progressive acceptance.
Source review of MyStory proposed edits likewise precedes either commit. Use
existing cleanup budget, no additional author retry engine. Unresolved edits
retain accepted text and render continues. Record authorization, exact before,
intervals/replacements, candidate identity and rejection reason in existing row
receipt. Extend context verification schema consistently if adding raw source.

## D. Source organization and fidelity: newly authorized adaptive path

Raw four creative fields are already persisted in source_meta.story_input.fields.
Use those exact bytes and explicit raw-coordinate version; normalized fields do
not share offsets. Reuse _otr_source_document's immutable documents/spans/coverage
validation; its objects remain transient, store only primitive hashes/offsets in
receipts. Empty optional fields require no document. No fixed fact taxonomy.

Try complete source with real reader prompt first. If prompt fits, P0 can organize
it directly; don't add compulsory summaries of short inputs. Working notes remain
advisory; raw source wins over P0 assumptions and P1 treatment. P0 schema may append
optional working notes/source references while preserving its current artifacts.

For source exceeding measured capacity, read exact advancing field spans; prefer
natural boundaries without gaps or skipped Unicode/unbroken text. The model gets
only current original spans and labelled previous notes. Code stamps delivered
input_span_ids independently of model citations. No model can declare unseen
tail read. Model chooses headings/notes/source_refs, not a fixed relationship
ontology. Validate references against supplied spans using the existing ladder.

Window sizing uses the real prompt/context. All available output room remains
available. If the returned organization cannot finish within real output capacity,
preserve that failed artifact and split that source window into smaller disjoint
children, using the same structured-call owner for each new work item. Do not
rerun a known-oversized packet indefinitely or truncate its source. Each split
advances coverage toward smaller nonempty intervals; the envelope itself unable
to fit is a real provider/config capacity failure.

If notes exceed the next prompt, merge fitting groups with the same organizer.
Each merge consumes at least2pending notes into1parent; retain children and exact
ancestry. A single-note compaction must measurably enable the next operation.
Apply bounded schema/organization repair for non-progress; no arbitrary global
story/pass limit and no loop on unchanged packet. Original source is never deleted.
Store separate read/organized/fidelity_checked states, measurements and identities.
Byte coverage proves delivery to calls, never semantic fidelity.

P0-P3 receive complete raw source whenever it fits. Otherwise organized notes and
relevant exact raw spans carry explicit partial-view labels; do not pretend the
author saw whole source. Scope an act to its actual plan, not every story event.
The organizer's notes cannot replace original authority in fidelity checking.

Shared module-level review schema: verdict no_direct_contradiction|contradiction|
uncertain; conflicts{source_field,source_quote,candidate_path:[str|int,...],
candidate_quote,reason}; uncertainty. Plain dict/list traversal validates paths;
no JSONPath dependency or substring fallback to another field. Ground exact
quotes against raw fields and specified candidate strings. Quote grounding is
provenance, not an infallible semantic proof. Compatible elaboration, facts not
repeated in a particular act, and dialogue speculation aren't contradictions.

Close over explicit review context (raw source/prepared views, technical_fn,
scheduler, source_reviews) in author post-validator composer; run structural
checks first. Add review context parameter to P0/P1/P2/P3 helpers. Reviewer uses
separate structured_call(max_attempts=2), no recursive author validator. Record
its calls separately. A grounded direct contradiction returns error string to
same author max3 ladder; full-artifact repair conserves all except named defect.
P0/P3 use full-artifact repair too. Never copy exception partial output into an
accepted proposal or raw_output field.

Review raw-source windows against complete candidate if it fits, otherwise
explicit candidate partitions too; track complete source/candidate coverage.
Cross-window ambiguity requests available context or stays uncertain. Notes may
guide retrieval but cannot mark unread raw source checked. Uncertain/unavailable
verifier is recorded, not repaired into confidence and not source-qualified PASS.
Only grounded unresolved direct contradiction exhausts authoring before media.
Provider/OOM/cancel failures keep real types; do not label them source conflicts.

Durable source_reviews: source digest/raw hashes, coordinate version, actual
delivered spans and candidate projection/hash, scope/pass, review outcome and
evidence, separate attempt IDs, configured versus known executed model identity,
accepted/rejected/unresolved disposition, parent revision. Any candidate edit
invalidates its old check. Persist at existing runner checkpoints/finally, not
only callbacks whose exceptions are swallowed. No late mux eligibility veto.

Tests: exact tail coverage/Unicode, unseen refs, nonprogress, changed context or
template, cancellation, summary omits fact but raw check catches conflict;
living/dead contradiction, pair separation, correction through same ladder,
compatible elaboration, fake evidence, uncertain not rejection/PASS, source beyond
400chars, true capacity, accepted hashes survive reload. Review does not prove
pixels or guarantee every semantic error is found; live recovery is still required.

## E. Cleaner transaction and final visuals

Raw source in MyStory cleaner judge/repair context. Proposed changed rows checked
before acceptance within its existing repair budget. After both cleaners, review
final spoken projection BEFORE reconcile. Add public CleanTransaction.abort(reason,
evidence=None) sharing guarded restore/degradation path, preserving attempted
receipts and supplied evidence through restore. On new grounded conflict restore
pre-clean rows/metrics/finalizer, then stamp review of actual retained projection;
no direct private _degrade call or unaudited restore/noop. Uncertain records no PASS.
Same-view hash avoids rechecking unchanged text; no global cache of semantic truth.

MetaBrief scene request gets raw/organized source context, current act/scene and
candidate companions with appearances. Model decides presence from explicit
current scene/source; speakers somewhere in the act are NOT automatically visible
in every shot. Preserve target face while allowing required companions. No new
fact inventory as authority. Neutral portraits and announcer/music scopes remain
distinct. Source checker treats prompt as typed {prompt: string} and scene scope.
Reuse max_reseed for concrete contradiction feedback; no new visual retry loop.
Fallback takes same scoped source context and is labelled unresolved if not
source-verified; never call it source-qualified merely because deterministic.

MetaBrief returns image_prompts_json, not ledger meta. Current source/context
receipt rides each object into dispatcher BOTH fresh/cache-hit paths and durable
stamp. Add base_prompt_hash to _NormalizedPrompt (preserve existing field access
compatibility); retain final prompt_hash. Record exact authorized normalization
chain: base, safety/style, banana selection, final identities. A base semantic
review is labelled as covering base only; explicit banana transformation is not
silently called fidelity-preserving. Cache identity includes current source/context
where it changes semantic request; mismatched receipt never labels a stale hit
accepted. Review actual final bytes if transformation makes new semantic claims.
Do not force a new model load in dispatcher solely to paint a false PASS; report
unreviewed transformation clearly and inspect final prompts/pixels in recovery.

## F. Mac and diagnostics; qualification

Incoming0404aa86 scopes memory failure to M4/16GB. Current MyStory cleanup does
not bind LMFE; M1 is not Mac cause/cure. Existing MPS flush stays. Do not switch
model.to(meta), delete models, add predictive story gates or claim a smaller
uncached model is qualified. Instrument actual generation return and retirement
with timestamp/phase, process memory and MPS live/driver allocations using
lightweight platform-aware observer, never per-token sync or a subprocess loop.
Prefer existing telemetry owner; fix stale unload reference comments. Actual
Mac owner must provide final log/footprint evidence to close memory diagnosis.

Reported diagnostic repairs after blockers: explicit cast-preflight phase gives
still-deferred info, real post-image missing-still remains loud; LTX-open health
compares planned/actual engine, intentional still_pan valid, requestedLTXfallback
invalid, missing intent unknown; successful same-file rename info in three known
owners, mismatches loud. IDs are campaign-local, not global O1 renames.

No node/socket/widget changes expected. Canonical remains real23node/63link graph,
37writerwidgets; run real validator+roundtrip+link/widget audit. Full recovery
loads ONLY workflows/otr_canonical.json with normal manual fields and sanctioned
runtime overrides, no replay/partial target/alternate graph. Start single one-act
living-mother/shared-dinner leg on5080, then4060 provenZImage/still_pan route;
Mac owner actualruntime/schema plus appropriate proven installedroute; collect
final prompts, ledger, audio/video and actualotr/obsfile. RunPod only for new
coverage after latestrepo/image verification, stopcompute afterward. Keep source
qualification separate from technical publication. Do not claim remaining fixes
or races complete from the shipped S2 component regression.
