# RECOVERED -- Lemmy cross-engine plan, sections 5-10

**What this is.** `docs/2026-08-10-OPEN-PLAN-lemmy-cross-engine.md` was reverted
on 2026-08-12 while it carried ~500 lines of Codex-hardened build spec that git
had never received. The disk copy now holds an earlier 199-line draft with a
different structure. The hardened version is otherwise lost.

**Where this text comes from.** I read sections 5-10 of the MODIFIED working copy
during the 2026-08-11 Lemmy build -- it is what plans 5.2 and 5.3 were built
from -- so the text below is transcribed from those reads, not reconstructed from
memory or inferred from the code. It is verbatim for the ranges I opened.

**WHAT IS AND IS NOT HERE, stated plainly so nobody trusts it further than it
earns:**

* **PRESENT, verbatim:** section 5 (incl. 5.1, 5.2, 5.3), section 6, section 7,
  section 8 (incl. 8.1, 8.2), section 9, and the start of section 10.
* **LOST:** sections 1-4 (incl. 2.1, 2.2, 4.1) and the tail of section 10. I
  never opened those ranges, so I have nothing for them and have not invented
  anything to fill the gap.
* The original heading list is preserved below so the shape of what is missing is
  visible.

**Original section map** (line numbers from the reverted file):

    1    # Lemmy Cross-Engine Architecture and Controlled Audition Plan
    10   ## 1. The decision that moves work forward          [LOST]
    29   ## 2. Data the implementer must have                 [LOST]
    31   ### 2.1 Candidate inventory: verified but not qualified   [LOST]
    62   ### 2.2 Required evidence packet before coding a route    [LOST]
    99   ## 3. Current constraints that the design must respect    [LOST]
    140  ## 4. One semantic qualification contract             [LOST]
    202  ### 4.1 Local WAV and direct-provider routes are intentionally different [LOST]
    235  ## 5. Exact implementation seams                      [RECOVERED]
    237  ### 5.1 Semantic validation and reference resolution  [RECOVERED]
    261  ### 5.2 CastLock ordering                             [RECOVERED]
    308  ### 5.3 Voice-node request/cache/receipt contract     [RECOVERED]
    348  ## 6. G1 controlled IndexTTS2 Test A and incumbent comparison [RECOVERED]
    396  ## 7. Branch A: qualified scalar Index route          [RECOVERED]
    414  ## 8. Branch B: generic per-cast routing              [RECOVERED]
    419  ### 8.1 One authoritative mode                        [RECOVERED]
    450  ### 8.2 All-or-nothing per-cast transaction           [RECOVERED]
    471  ## 9. Acceptance gates                                [RECOVERED]
    487  ## 10. Deliberate boundaries                          [PARTIAL]

---

## 5. Exact implementation seams

### 5.1 Semantic validation and reference resolution

Add validate_qualified_voice_route(record, now_utc). It verifies fields,
allowed statuses, hash syntax, local byte integrity where applicable, actual
bank entry, runtime/model identity, manifest, and rights state. The legacy
is_qualified_route may remain as a compatibility helper but cannot authorize a
selected route.

Add resolve_and_verify_reference used before both request builders:

- local_wav: checks the selected cast route, resolves the local reference,
  calculates actual SHA-256, and supplies source_ref_sha256;
- provider_voice: validates route/provider/model/voice fields without pretending
  a cloud URI is a local file; and
- legacy no-route row: preserves existing resolver behavior but can never be
  confused with a newly qualified route.

It receives the active scalar engine. A qualified `voice_route` must agree with
that engine and its same-engine bank entry. For a legacy row, treat the pair
`(voice_engine, voice_ref_id)` only as a declared bank reference after it
agrees with the bank and active scalar engine; do not use it as a claim that a
renderer ran. Preserve `tts_model`/`voice_preset` as Bark-preset provenance,
but never use them to select or override a present validated reference.

### 5.2 CastLock ordering

Refactor CastLock.lock as follows:

1. Calculate the new revision and assign meta["cast_lock_revision"] before
   resolving any route.
2. Resolve voice-bank entries/default engine once and stamp normal engine
   metadata for both preserve_ledger and auto_registry modes.
3. For every non-announcer cast row, match normalized row name or char_id
   against the policy character_key. Current Lemmy has positional char_id c02,
   so matching cannot assume char_id equals "lemmy".
4. Before hybrid LLM voice-fit and generic seeded selection, resolve a selected
   policy route. On a successful local route, obtain its same-engine unique bank
   entry, verify bytes, call existing CastLock stamp helper, write
   cast[].voice_route, and call _mark_used.
5. In auto_registry, the claimed row continues before hybrid/generic selection. In
   preserve_ledger, only the claimed policy row changes; all others preserve.
6. Continue ordinary revision/meta/durable save once. A selected route that
   fails qualification/match raises a route-named error; it never falls back.

The policy route is an explicit re-pin. It must not change `stable_cast_seed`
or the generic selector's behavior for unclaimed rows.

Marking a ref used happens regardless of allow_voice_reuse. It changes other
selections only if no-reuse is false. The current canonical workflow sets
allow_voice_reuse=true, so acceptance tests must exercise both true and false.

Cast row storage is immutable route identity:

~~~json
cast[].voice_route = {
  "route_id": "...",
  "route_contract_version": 1,
  "status": "qualified",
   "engine": "indextts2",
   "voice_ref_id": "idx_lemmy_algenib_cockney_v1",
   "reference_kind": "local_wav",
   "ref_path": "models/TTS/refs/indextts2/lemmy_algenib_cockney_v1.wav",
   "source_ref_sha256": "64-hex",
   "qualification_record_id": "...",
   "runtime": {"model_id": "...", "engine_impl_version": "...", "weight_revision": "..."}
}
~~~

It survives the whole-cast durable stamp. Do not add it to the root meta
whitelist.

### 5.3 Voice-node request/cache/receipt contract

Immediately after cast_lookup, the voice node reads its active scalar engine and
voice_route, proves the selected route/active-engine/bank-entry triple, then
calls resolve_and_verify_reference before either cached or local request
construction.

In one atomic change to _otr_resolved_request.py:

1. add route_id, route_contract_version, qualification_record_id, and
   weight_revision to ResolvedVoiceRequest;
2. add exactly those fields to IN_KEY_FIELDS;
3. preserve the dataclass field-partition invariant with IN_KEY_FIELDS and
   IGNORED_FIELDS;
4. extend build_resolved_request signature/propagation; and
5. update both request builders. Selected routes provide actual identity;
   non-policy legacy rows provide deterministic empty/zero defaults.

source_ref_sha256 is passed in both local and cache paths for local_wav routes.
A byte-only change at one path changes the request cache key.

Replace local IS_CHANGED="static" with a deterministic local-only fingerprint:

- local_wav: route fields, active profile/runtime fields, and actual reference
  bytes; unreadable expected local file returns NaN;
- provider_voice: route ID, provider, provider voice, model/runtime and policy
  values, without filesystem hashing; and
- per-cast Branch B: all active row routes/profiles, never a fake engine called
  per_cast.

Never make network calls in IS_CHANGED.

Store static route identity once on the cast row. Store current, bounded
per-line render evidence through _persist_ledger_stamps and
stamp_per_line_audio_meta for both local and cached renders: route ID,
`tts_engine` (the actual adapter), model, request/cache key where applicable,
sample rate, render duration, and audio_sample_hash/audio_sha256. Re-render
overwrites that line's current receipt; no unbounded history. Failure to persist
a selected-route receipt fails before returning audio output.

## 6. G1 controlled IndexTTS2 Test A and incumbent comparison

G1 answers both questions that matter: whether the candidate meets the configured
floor and whether it is a better qualified route than the historic, accidentally
pinned incumbent. It is a route comparison, not a claim that the 33 ledger rows
were necessarily 33 published listener exposures.

1. Preflight the three frozen reference inputs before loading the model:

   | Arm | Frozen reference input | Purpose |
   | --- | --- | --- |
   | A | `2_algenib_cockney.wav`, SHA `47E733...A60DB2` | Candidate route |
   | B | `vz_donor_marshal_indian.wav`, bank SHA `8F573D...AF4AA3F` | Historic IndexTTS2 incumbent |
   | C | `1_algenib_plain.wav`, SHA `D48AAD...283CFB` | Same-speaker non-Cockney control |

   Arm B resolves through its bank-relative path and must byte-match its bank
   hash. Arm C is fixed; `4_charon_plain_control.wav` is optional only as a
   sanity/distractor asset, never a substitute for a named three-arm control.
2. Freeze G0 decision, all three input hashes, local model/weight identity,
   neutral line, emotional line, settings, scorecard, rater identities, and
   threshold. Render all arms through the same installed IndexTTS2 implementation
   and settings. The neutral target text contains no Cockney lexical cue.
3. Randomize labels and retain output hashes, scorecards, request settings, and
   manifest. Do not expose arm identity to raters.
4. The scorecard has separate criteria:
   - absolute floor fit: configured gravelly/raspy character, Cockney, and
     intelligibility;
   - A-versus-B route comparison: A must beat B on the floor dimensions without
     losing intelligibility; and
   - A-versus-C same-speaker check: accent retention and Algenib identity.

   The present policy does not encode London working-class as a separate
   machine-checkable attribute. Add an explicit policy/rubric field before
   scoring it; never infer it from an asset identifier.
5. Recommended minimum is two named raters; A must score at least 4/5 on each
   floor dimension, win the B comparison, retain A/C identity and accent at
   least 4/5, have no intelligibility score below 3/5, and have no unresolved
   material disagreement.
6. A separate dated release/OBS audit decides whether the change is an editorial
   recast for an already-published audience. If it is, require that editorial
   approval in addition to G0/G1; do not invent exposure from ledger rows.
7. Only a full G0/G1 pass creates the local Index bank entry and qualified policy
   route. Fail/no authorization/ambiguity leaves native routes empty. It never
   installs the historical donor, an unqualified generic pin, or angry fallback.

Dia permits audio-only use today. Create/require a transcript schema only if a
later transcript-conditioned Dia route is chosen.

## 7. Branch A: qualified scalar Index route

Branch A starts only after G1 pass.
It is an explicit qualified re-pin of Lemmy; it does not alter generic
seed-derived casting for other rows.

1. Commit the local bank entry/reference, policy record, validator, CastLock
   resolver, unified request identity, local change fingerprint, and durable
   cast/line receipt work together.
2. Update tests that currently assert empty approved routes or mistake inherited
   Bark `tts_model` provenance for a bank-route runtime selector.
3. Prove selected route precedes hybrid/generic selection, marks used identity,
   fails loudly when invalid or when route/active-engine/bank engine disagree,
   leaves generic fallback unchanged, invalidates cache on byte change, and
   preserves cast/line data on durable reload.
4. Do not change workflow JSON for Branch A. Validate the real canonical graph,
   JSON round-trip, widget count, and links anyway.

## 8. Branch B: generic per-cast routing

Branch B begins only after G1 fails/no authorization and every chosen direct
route independently qualifies.

### 8.1 One authoritative mode

Add optional char_voice_routing_mode to OTR_CastLock at the end of its positional
INPUT_TYPES/widget order: scalar (default), per_cast. It stamps
meta.char_voice_routing_mode and that one root-meta field is added to the
durable whitelist.

Node 81 already consumes node 80 ledger_json over canonical link 235. It reads
that mode; its existing engine widget remains default for rows without a
selected route. Do not create a per_cast engine ID and do not add a duplicate
renderer mode widget.

Implementation contract:

~~~text
generate(...):
  if meta.char_voice_routing_mode == "per_cast":
      return _generate_per_cast(ledger, lines, default_engine=engine, ...)
  return existing scalar implementation unchanged
~~~

Branch B changes code and the real workflow atomically. Append per_cast to node
80 widgets_values, producing:

~~~text
["default", "auto_registry", true, "indextts2", "kokoro", "cuda", "per_cast"]
~~~

Update node 80 INPUT_TYPES/lock signature/root-meta whitelist, canonical JSON,
and workflow tests together. Never insert a positional widget.

### 8.2 All-or-nothing per-cast transaction

1. Collect active routes and preflight all profile, engine usability, token,
   model, rights, qualification, reference, and output-bus requirements before
   the first generation call.
2. Group lines by cast[].voice_route.engine when present; otherwise use default
   engine. Render into buffers keyed by original occurrence index.
3. Immediately normalize every generated clip through the existing
   _otr_audio_utils.resample_audio helper to the explicit 24,000 Hz mixed bus
   before it enters the occurrence buffer. This is project-standard CPU
   scipy.signal.resample_poly and returns AUDIO [B,C,T] float32.
4. In each engine-group finally block call existing _teardown(adapter) before
   loading another native adapter. Smoke this against the 16 GiB GPU budget.
5. Only after all groups succeed: verify no duplicate/missing occurrence,
   restore original order, emit ordered logs, durably persist line receipts,
   then call pack_audio_batch. A group failure returns no partial dialogue
   batch and no line receipts. Internal cache artifacts are not publication
   evidence without the final receipt.
6. Per-cast IS_CHANGED fingerprints all active row routes and profiles. It
   never calls profile resolution with per_cast as an engine.

## 9. Acceptance gates

| Area | Proof required |
| --- | --- |
| Evidence | Three-arm A/B/C hashes, manifest, scorecard, and rights decision all bind to exact route scope. |
| Qualification | Missing/tampered/mismatched local WAV, bad SHA syntax, invalid enum, wrong route, or denied/revoked/expired rights fail. |
| Routing | A selected route requires equality of route engine, active node-81 adapter engine, and bank-entry engine; legacy tts_model never overrides a validated bank reference. |
| Casting | Selected route runs before hybrid/generic selection, marks used, fails loud when invalid; unrelated fallback is unchanged. |
| Identity | Dataclass, field partition, both builder calls, request cache key, and local IS_CHANGED change when only WAV bytes change. |
| Persistence | Cast route and local/cached current line receipt with actual tts_engine survive reload; selected receipt-write failure fails. |
| Test A | Blinded three-arm scorecard proves floor fit, A-over-B route result, and A/C identity/accent result; a release audit records any separate editorial recast decision. |
| Branch A | Forced Lemmy scalar canonical render succeeds at native Index rate and SceneSequencer resampling is verified. |
| Branch B | Preflight, group teardown, ordered all-or-nothing output, 24 kHz packing, no silent fallback, and receipt tests pass. |
| Workflow | Branch-B code and append-only canonical node-80 widget pass OTR_WorkflowValidator, round-trip, link, input-name, and widget-count audits. |
| Final | Focused tests, full pytest, Bug Bible, then forced-Lemmy real canonical headless render with output asset existence verification. |

## 10. Deliberate boundaries

OTR ships 21 direct ElevenLabs IDs and no catalog search. Any future migration
needs a dated ID/type/access audit; the December 31, 2026 risk applies to
[ElevenLabs Default voices](https://elevenlabs.io/docs/help-center/product/voice-customization/my-voices/what-are-default-voices),
not automatically every configured ID.

The local Kokoro bank has four British male IDs and none is tagged gravelly.
The archived Bark audition was generic fixed-line fullness/least-thin work with

*(section 10 truncates here -- this is the end of what I read.)*

---

## One correction the recovered text needs

Section 8.2 step 3 says the mixed bus is **24,000 Hz**. It is **48,000**
(`scene_sequencer.py`: `sample_rate = 48000  # standardize output`; publish
encode `-ar 48000`). That error is IN the recovered plan, not introduced by this
transcription -- flagged here so Branch B is not built against it. The same
mistake was independently made and corrected in
`tests/test_lemmy_index_rate_to_bus.py` (`f22fa414`).
