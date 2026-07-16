# Fable2 S2 QA analysis, round 2

> **STATUS 2026-07-15 (baseline): SUPERSEDED -- runway complete; historical analysis only.**
> All three P0s folded @ 47bf50f2 (2026-07-10). P1.1/1.3/1.4 shipped as 720-bakeoff C1/C2/C3
> (9949bb6e / 2f335c28 / 6899d940); P1.2 landed inside the FreezePolicy fold; P1.5 landed via
> the S2 full-loop enable @ 95582643 (P1.5's item-by-item acceptance list was not re-verified
> in this baseline). scifi_fable2 subsequently ranked #1 in the 2026-07-15 720 bake-off
> verdict.

**Audit date:** 2026-07-10
**Audited baseline:** branch `v2.0-alpha`, fix commit `af378aad`
**Scope:** the real Windows worktree, the canonical workflow, focused regression tests, and
the two published episode ledgers named in the problem statement.

**Artifact citation key:** "Einstein ledger" is
`C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes\signal_lost_einsteins_echo_20260710_112823\audio\signal_lost_einsteins_echo_20260710_112823_ledger.json`;
"Butterfly ledger" is
`C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes\signal_lost_the_butterflys_gambit_20260710_134516\audio\signal_lost_the_butterflys_gambit_20260710_134516_ledger.json`.
"Einstein captions" and "Butterfly captions" are the corresponding
`signal_lost_einsteins_echo_20260710_112823_captions.ass` and
`signal_lost_the_butterflys_gambit_20260710_134516_captions.ass` files in those
same `audio` directories.

## Executive disposition

The round-1 fixes improved the disk-merge behavior and made the LTX run complete, but
they do **not** yet establish a proof-preserving freeze boundary. There are three P0
integrity failures:

1. The cascade invokes the text-mutating reviewer before it applies its Fable2
   capability gate, so a sealed proof map can diverge from the live canonical text.
2. A fresh `set_lines()` rebuild can restore a stale disk skip flag and its reasons
   onto a newly non-empty line.
3. A reviewer save can rebind the ledger root while the cascade continues to write
   terminal veto state to the old detached dictionary.

The green Einstein and Butterfly artifacts demonstrate successful rendering, not the
stronger S2 claim that canonical text, proof, freeze verdict, TTS input, music timing,
and final artifact all form an auditable immutable chain. Resolve all P0 items before
claiming a production-ready Fable2 lane or adding S2's full-loop generation work.

Focused evidence checks passed: `tests/test_ledger_merge_ownership.py` (3 passed) and
`tests/test_freeze_cascade_meta_persistence.py` (10 passed). Those tests do not cover
the P0 reproduction paths below.

## P0 - Proof and freeze integrity

### P0.1 - The Fable2 gate runs after Doctor mutation

**Verified evidence**

- Fable2 seals the proof map after collecting its constituents at
  `nodes/_otr_scifi_fable2.py:1661-1672` and assigns it into the ledger at
  `nodes/_otr_scifi_fable2.py:1800-1802`.
- The cascade invokes `review_ledger()` unconditionally at
  `nodes/_otr_freeze_cascade.py:784-800`. The lane capability decision does not occur
  until `nodes/_otr_freeze_cascade.py:862-894`, with the `clean()` substitute at
  `875-894`.
- The reviewer Doctor rewrites canonical `text` at
  `nodes/_otr_ledger_reviewer.py:1811-1835`; the Phase-7 readiness normalizer also
  rewrites canonical `text` and counts at `nodes/_otr_readiness.py:223-268`.
- The published Butterfly ledger records Doctor edits at `Butterfly ledger:1139-1146`
  and a `frozen_with_doctor_edits` verdict. Its live b2 text,
  `Not necessarily, Paul. Consider these.`, differs from the sealed proof text,
  `Not necessarily, Paul. Look at these.`, at `Butterfly ledger:282,763`,
  respectively. Its b5 text similarly differs: `You wouldn't.` at `345` versus
  `You didn't.` at `802`.
- The Einstein ledger independently shows Phase-7-style mutation: the frozen b3 text
  at `Einstein ledger:287-295` differs from its proof source at `757-766`.

**Root cause**

The current design treats "replace the 5B critic report" as a Fable2 protection. It is
not a protection for phases 1-2, deterministic reviewer repairs, Doctor edits,
readiness normalization, A3 target construction, escalation, A2, or any subsequent
canonical-text mutator. A proof seal is only meaningful if no unaccounted writer can
reach the sealed fields after the seal.

**Required fix**

Resolve one typed `FreezePolicy` before calling the reviewer. For
`meta.source_bank == "scifi_fable2"`, that policy must skip all legacy content-mutating
paths: reviewer cast repair, Doctor, 5B, A3, 5C, escalation, A2, and D3 mutation.
It must still run read-only structural validation: parse/line-shape validation,
speaker-to-cast validation, proof-map verification, required-boundary checks, and
non-mutating readiness checks. If a Fable2 pack cannot resolve, policy resolution must
raise a terminal structural error; it must not fail open into legacy mutation.

Do not emulate a reviewer result with `StoryCriticReport.clean()`. Instead expose a
capability disposition such as `reviewer_disposition: "not_applicable_fable2"`, with
the exact skipped phases. That makes the receipt truthful and prevents downstream code
from treating "clean" as "reviewed."

Add `meta.freeze_capability_receipt` before freeze finalization:

- policy name and source of resolution;
- skipped and executed phase identifiers;
- canonical-text and proof-map hashes before and after the cascade;
- `content_mutations: 0` for Fable2;
- structural-validation results and terminal error details.

**Regression sketch**

Build a full Fable2 ledger with a proof map, run the entire cascade under both default
and enabled critic-escalation settings, and assert all of the following:

- `review_ledger`, critic generation, reroll, and A2 are never invoked;
- canonical line text, word/character counts, proof map, and hashes are byte-identical;
- the receipt names each skipped legacy phase;
- malformed or missing pack resolution yields a terminal error, not a legacy review;
- a structural cast/proof error still stops the freeze.

### P0.2 - The merge fix still resurrects stale skip metadata after a line rebuild

**Verified evidence**

- The new denylist is only ten fields at
  `nodes/production_ledger.py:1318-1322`; merge suppression occurs only when the
  current row already has that key at `1356-1360`.
- `Ledger.set_lines()` constructs a fresh row without `skip`,
  `tts_skip_reason`, or `reviewer_skip_reason` at
  `nodes/production_ledger.py:1042-1061`.
- I reproduced the real save path: persist b001 as skipped with a reviewer reason,
  rebuild it with `set_lines()` and non-empty canonical text, then save. The disk row
  becomes a non-empty current line carrying `skip: true`,
  `tts_skip_reason: "reviewer_skip: old"`, and
  `reviewer_skip_reason: "old"`.
- Phase 10 treats a skipped non-empty line as a freeze error at
  `nodes/_otr_ledger_freeze.py:342-352`.

**Root cause**

The merge asks whether a current row contains an individual key. A fresh authoritative
row omits some fields by schema, so the merge incorrectly interprets omission as
"please restore old disk state." The ownership boundary must be at the row/field source
level, not inferred from falsiness or an incomplete in-memory dictionary.

**Required fix**

Give each array a declared merge policy and per-row authority/revision metadata. For a
current `lines[]` row, never restore line-owned fields from disk merely because a
fresh writer omitted them. A new line rebuild is authoritative for all line composition
state; durable renderer outputs may survive only when their source identity and revision
match. Do not fix this by pre-populating defaults alone: that hides the same semantic
bug for the next omitted line-owned field.

**Regression sketch**

Persist a skipped b001, rebuild b001 through the public `set_lines()` API with
non-empty text, save, reopen, and assert that neither skip flag nor either reason is
present. Run the same test with an explicitly empty replacement text and assert that
the result reflects the new authoritative row rather than stale disk state.

### P0.3 - A reviewer save can discard the terminal freeze veto

**Verified evidence**

- `Ledger.save()` replaces `self.data` with the merged root object at
  `nodes/production_ledger.py:1192-1234`.
- Reviewer commit and terminal paths save the ledger at
  `nodes/_otr_ledger_reviewer.py:2013-2034`, `2077-2099`, and `2119-2142`.
- The cascade writes phase telemetry immediately after review at
  `nodes/_otr_freeze_cascade.py:791-800` and terminal disposition at
  `812-835`, but refreshes its `ledger_data` reference only in later non-terminal
  paths at `862`, `1100`, and `1141`.
- `_build_terminal_skip_disposition()` writes `meta` through the passed object at
  `nodes/_otr_freeze_cascade.py:642-679`.
- A deep-rebinding probe returned `needs_full_rerun`, while the live ledger root had
  no `freeze_verdict` and no terminal phase records. This is safety-significant:
  `nodes/cast_lock.py:258-306` allows a missing verdict as a legacy path but halts on
  an explicit `needs_full_rerun`.

**Required fix**

Immediately after every possible reviewer save, assign
`ledger_data = led.data` before writing any phase record, disposition, receipt, or
terminal veto. Make root identity stable in the longer term, or protect save/rebind
with a revision/CAS contract so detached writes cannot silently win. The terminal
freeze function should operate on the ledger object or a fresh root accessor, not a
cached nested dictionary.

**Regression sketch**

Use a reviewer double that calls the real rebinding save and returns
`needs_full_rerun`. Assert that the live in-memory and persisted ledger each contain
the verdict, terminal disposition, and phase records; then feed that ledger to
CastLock and assert it halts. Repeat for a normal reviewer commit path.

## P1 - S2 contract and data-model work

### P1.1 - Complete the ownership model before S2 expands disk persistence

The old generic behavior intentionally restores missing row keys from disk
(`nodes/production_ledger.py:1179-1186`, `1326-1329`). Existing tests explicitly
exercise that behavior in `tests/test_production_ledger.py:248-294` and preserve an
empty music row from disk at `314-331`. That is a compatibility behavior, not proof
that it is correct for fresh authoritative composition rows.

Use the following classification as the S2 merge contract.

| Row | Owned by current composition/request | Durable only with matching source/revision identity |
| --- | --- | --- |
| lines | `line_id, beat_id, shot_id, char_id, speaker_role, boundary, text, char_count, word_count, traits, arc_phase, beat_intent, target_words, dialogue_slot_id, compose_flags, skip, tts_skip_reason, reviewer_skip_reason, reviewer_note, needs_render_realign` | generated WAV/cache/status fields, actual render metrics, `start_s, dur_s, start_s_space`, video/render stamps |
| music | `cue_id`, order, description, generation prompt, anchor/placement, target duration | WAV/cache/status/timing only when cue id plus prompt/spec hash match |
| clips | source/line binding, render request/spec, ordering | output path, cache/status, render metrics/timing only when source/render-spec hash match |

The line schema itself owns much more than the ten-field denylist:
`nodes/production_ledger.py:914-938` and `1042-1061`. Reviewer notes are written
at `nodes/_otr_ledger_reviewer.py:1855-1857`; render-realignment state is written at
`nodes/_otr_radio_editor.py:1034-1041`. Treating either as a durable blind restore
can attach a prior episode's editorial state to a replacement line.

`text_for_tts` should become a generated current field with a canonical source hash.
It may be retained only when that hash agrees; it must never be blindly restored after
the canonical text changes.

**Regression sketch:** table-drive every owned and durable field for lines, music, and
clips. Test replacement, deliberate clearing, same-spec persistence, changed-spec
invalidation, and a concurrent save/rebind. Retire only the old tests whose expectation
is explicitly superseded; preserve legacy read compatibility as a migration test.

### P1.2 - Capability gating must cover A3, escalation, and A2 as well as 5B/5C

**Verified evidence**

- `_legacy_line_compose_applicable()` fails open on resolution errors at
  `nodes/_otr_freeze_cascade.py:270-292`.
- After the 5B substitution, A3 still unconditionally builds reroll targets and can
  alter the unverified status at `918-963`.
- The default escalation route remains reachable at `1006-1078`, including a line
  reroll at `1071`; enabled escalation can progress to episode/A2 handling.
- The Butterfly ledger records line-escalation activity despite the lane being
  inapplicable to the legacy compose seam at `Butterfly ledger:1177-1179`.

**Required fix**

Make the policy chosen in P0.1 the only branch predicate. It must wrap the complete
reviewer-to-A2 subgraph, rather than accumulating local guards around individual
phases. There are only two valid states for a declared Fable2 lane:

1. known Fable2 policy: execute its read-only validation sequence and record all legacy
   content phases as not applicable;
2. policy resolution failure: fail terminally before any legacy content pass.

Do not reuse the current "unknown means legacy applicable" behavior for a declared
Fable2 source. Untagged legacy ledgers can retain the legacy route during migration.

**Regression sketch:** parameterize default and enabled escalation environments over a
Fable2 ledger and an untagged science-news ledger. Spy on A3, critic, reroll, and A2.
The Fable2 spies must all be zero calls; the legacy route must retain its existing call
behavior.

### P1.3 - Route pronunciation through text_for_tts without changing Fable2 proof text

**Verified evidence**

- Phase 7 currently normalizes the canonical text and counters in place at
  `nodes/_otr_readiness.py:223-268`, including number-to-word and `Dr.` to
  `Doctor` transformations.
- The common voice path selects ledger lines at
  `nodes/_otr_tts_engine_common.py:405-411`, reads `text` at `453-454`, derives
  delivery text at `466-478`, and passes it to engine generation at `592-595`.
  Bark, Kokoro, IndexTTS2, Chatterbox, and Dia therefore share this route today.
- Bark's per-line engine work occurs at `nodes/eng_bark.py:66-69` and `105-151`.
  SceneSequencer consumes the completed announcer audio bus rather than synthesizing
  new text at `nodes/scene_sequencer.py:833-850`.

**Required contract**

For a Fable2 ledger, Phase 7 writes only:

- `text_for_tts`;
- `text_for_tts_source_sha256` over canonical `text`;
- a normalization receipt with input/output hashes and applied rules.

It must stamp every non-skipped voiced line, even where the resulting delivery text is
identical, and must not change canonical `text`, char count, word count, or proof.
Numbers-to-words and `Dr.`-to-`Doctor` belong in this delivery-field generation.

Voice common must use:

    canonical = line["text"]
    delivery = line.get("text_for_tts") or canonical

For Fable2, absence, emptiness, or a source-hash mismatch is a terminal error rather
than a fallback. Use `delivery` for neutral preparation, engine-specific preparation,
delivery vector generation, resolved request receipts, and audio synthesis. Keep
`canonical` available for proof and caption diagnostics. The existing all-engine
common path means there is no separate Fable2 exemption to implement for the named
adapters; ensure future sidecars call this same resolved-delivery helper.

For `science_news`, retain the existing serialized input and resulting synthesized
request byte-for-byte. Its Phase-7 behavior must be explicitly versioned as legacy,
not silently changed by the new field.

Preserve ordering by attaching line id and delivery/request hash to each produced clip.
SceneSequencer should continue to use the ordered announcer clips and canonical text
for captions.

**Regression sketch**

- Full Fable2 cascade plus voice-common spies for every installed adapter: canonical
  text/proof/counters stay unchanged, and every synthesis input equals
  `text_for_tts`.
- Fail loud on missing, stale, or empty Fable2 delivery text; skipped lines bypass
  synthesis without being silently rendered.
- Assert line-id clip ordering through the 81/82-to-3 route and announcer bus.
- Serialize a science-news fixture before and after the S2 change and assert exact
  parity of its synthesis request, output ordering, and canonical ledger text.

### P1.4 - Replace the fixed theme slots with an authoritative cue manifest

**Verified evidence**

- Fable2 emits inter-scene cues at `nodes/_otr_scifi_fable2.py:1703-1744`.
- StableAudioTheme exposes fixed opening/closing/interstitial slots at
  `nodes/stable_audio_theme.py:38-43`, outputs them at `67-71`, and renders from
  fixed prompts at `208-225`; it does not consume `ledger.music[].generation_prompt`.
- `Ledger.set_music()` discards authored anchor fields at
  `nodes/production_ledger.py:1083-1095`.
- SceneSequencer passes music through at `nodes/scene_sequencer.py:753-831` and
  writes actual scene timing at `945-1006`; EpisodeAssembler only handles bookends
  at `nodes/episode_assembler.py:1043-1112` and `1342-1379`.
- In the canonical workflow, node 83 has output 0 linked to node 7 opening audio
  (link 241), output 1 linked to node 7 closing audio and node 12 closing audio
  (links 242 and 243), and output 2 is unwired. The `closing_audio` input of
  SignalLostVideo is dead at `nodes/video_engine.py:2252-2261`.

**Required contract**

`ledger.music[]` is the authoritative authored queue. Preserve, validate, and render
each `cue_id, anchor_line_id, placement, generation_prompt,` and
`target_duration_s`. A Fable2 cue must point at the exact boundary sentinel, and
duplicate cue ids or invalid anchors must fail loudly.

Replace the 3-slot output with:

- `cue_audio_clips: AUDIO`;
- `cue_manifest_json: STRING`;
- `render_log`;
- `done`.

Each manifest row must be unique and ordered, including cue id, anchor line id,
placement, batch index, prompt and prompt hash, seed, requested and actual duration,
sample count, sample rate, and canonical output path. Render each Fable2 cue directly
into the episode audio directory; never synthesize a post-hoc fixed prompt in place of
the ledger prompt.

SceneSequencer receives the queue and manifest, inserts inter-scene cues at their
authored boundaries, and stamps scene-relative timing. EpisodeAssembler takes the
same batch/manifest, extracts only opening/closing cues, shifts inter-scene timing into
master time, and mirrors timed rows into the final ledger. Science-news gets a legacy
manifest that reproduces its present three prompt/seed/order slots exactly.

**Exact canonical workflow delta**

Delete links 241, 242, and 243. Append (do not reorder) these inputs after existing
slot 6:

| Node | New input 7 | New input 8 |
| --- | --- | --- |
| 3 SceneSequencer | `music_cue_audio` | `music_cue_manifest_json` |
| 7 EpisodeAssembler | `music_cue_audio` | `music_cue_manifest_json` |

Append these four links, set `last_link_id` to 283, and update the affected endpoint
`links` arrays and input `link` fields:

| Link | Definition |
| --- | --- |
| 280 | `[280, 83, 0, 3, 7, "AUDIO"]` |
| 281 | `[281, 83, 1, 3, 8, "STRING"]` |
| 282 | `[282, 83, 0, 7, 7, "AUDIO"]` |
| 283 | `[283, 83, 1, 7, 8, "STRING"]` |

No widget is required for the base change. If a crossfade setting is later needed,
append it at the end of SceneSequencer's `widgets_values`; do not insert it.

**Regression sketch**

Use a ledger with opening, three inter-scene, and closing cues. Assert every prompt
was rendered, every cue manifest row is unique and ordered, and the final timing
anchors are correct. Test invalid/duplicate anchors as terminal errors. Add exact
science-news slot parity tests, fixed-slot consumer migration tests, a canonical JSON
round-trip, OTR_WorkflowValidator, and a link/widget audit in the same code-and-
workflow commit.

### P1.5 - Make S2 full-loop contracts executable before adding the loop

**Verified evidence**

- The ratified S2 architecture calls for P2a/P4/P5 at
  `docs/2026-07-10-scifi-fable2-architecture.md:896-902`.
- Current Fable2 rejects 120+ word full mode at
  `nodes/_otr_scifi_fable2.py:236-264`, builds one draft at `1990-1995`, makes one
  card at `2055-2067`, and proceeds P3 to P6 at `2098-2137`.
- PitchSelect and CriticNotes models exist at `322-324` and `422-448`, but no
  selection or revision contract presently uses them.
- The envelope rounds a scalar scene target at `755-764`; 350 words currently becomes
  117 x 3 = 351. P3 only aggregates budgets at `1428-1457`.
- P3 retries can grow the token budget by 25 percent at `1404-1415`, but the
  receipt records the initial budget at `2104-2105`, not the actual maximum used.
- `_script_view()` omits music at `1809-1826`, so a revision judge cannot protect
  music placement today.

**Required acceptance-shaped contract**

1. **Mode bounds and pitches.** At 30-119 words, retain the current compact mode and
   stamp P2a/P4/P5 as skipped. At 120-900 words, produce exactly three independently
   identified pitch cards with IDs 1, 2, and 3. A selection must name an existing
   integer pitch id and a valid selected member; malformed/missing/duplicate IDs are
   terminal errors. Reject 901 words before any creative call.

2. **Exact scene allocation.** `SceneEnvelope.scene_word_targets` must implement
   quotient/remainder allocation. At 350 with three scenes it must return
   `(117, 117, 116)`, preserve source order, and sum exactly to 350. Validate the
   total and each scene's configured word band in P3 and P5.

3. **Actual P3 accounting.** Store per-scene attempt count, every requested token
   budget, the actual maximum budget, truncation count, and selected successful
   attempt. The final receipt must derive from the observed attempts rather than the
   initial request.

4. **P4 and P5 input.** Normalize full markup before critique and revision. Pass the
   resolved story rules, scene envelope, selected pitch card, parsed original draft,
   normalized original source, critic notes, and all protected artifacts including
   title, cast, scene skeleton, coda/news boundary, and music queue.

5. **Pure revision validator.** Implement a pure function shaped as:

       validate_revision_contract(
           draft1_normalized, parsed1, draft2_normalized, parsed2,
           critic_notes, envelope, resolved_rules
       ) -> RevisionContractResult

   It returns eligibility; a deterministic reason list; named defects addressed;
   changed scopes; protected-field comparison; target/band results; and source hashes.
   It rejects title, cast, speaker set, scene order, settings, skeleton, coda/news
   boundary, or music changes; scene-zero ambiguity; a named-speaker change outside a
   critic-authorized scope; unnoted changes; and total/per-scene budget violations.
   Missing or malformed rules/notes must fail loudly rather than weakening validation.

6. **Deterministic keep-better judge.** Score both valid parsed drafts with the
   lexicographic tuple:

       (rule_pattern_hits, per_scene_band_violations,
        sum_scene_target_distance, total_target_distance,
        parser_normalization_count)

   Each axis is computed from the resolved rule set, not an implicit prompt. Draft 2
   wins only if eligible and strictly lower. A tie, worse score, or ineligibility keeps
   draft 1 and records the precise reason.

7. **Atomic final artifact.** Represent the winner as an immutable FinalDraft
   containing raw source, normalized source, parsed artifact, proof map, P3/P5
   receipts, score, and hashes. Atomically propagate that one object to P6, P7, P8,
   canonical `lines[]`, proof-map persistence, script/draft hashes, and downstream
   music-bearing inputs. Never update a script while retaining a proof map or P8 input
   from the losing draft.

**Regression sketch**

Add an acceptance matrix at 30, 119, 120, 350, 900, and 901 words. Run a fully mocked
350-word three-pitch loop; assert the three IDs, exact allocation, P3 actual-budget
receipt, protected-artifact preservation, score selection, and one coherent final
artifact hash set. Add negative revision cases for every protected field, named-speaker
scope escape, out-of-band scene, malformed critic note, tie, and worse draft.

## P2 - Retire unsafe legacy semantics and add production gates

### P2.1 - Retire Doctor skip as an editing outcome

The reviewer model permits `skip` at `nodes/_otr_ledger_reviewer.py:261-265`.
Consumers honor skipped lines in `nodes/_otr_ledger_consumers.py:71-105`, the voice
common route at `407-410`, SceneSequencer at `758`, and word accounting at
`nodes/production_ledger.py:405-410`. The new `tts_skip_reason` vocabulary has one
material enforcement point: the Phase-10 non-empty skipped-line error at
`nodes/_otr_ledger_freeze.py:342-352`. `reviewer_skip_reason` currently has no
semantic reader beyond its writer.

Retire new Doctor skip results globally. Legacy stored skips remain readable for
backward compatibility, but a reviewer may only rewrite, annotate, request escalation,
or fail. If an editor must remove content, model it as an explicit atomic composition
transaction that updates text, counts, proof eligibility, and boundaries together;
never as a QA-side mute. Remove `skip` from the reviewer result schema after migration
and validate that legacy reasons are not synthesized on new rows.

**Regression sketch:** reject a new reviewer `skip` result; load an old skipped
ledger without crashing; assert no new Fable2 or science-news review creates
`tts_skip_reason`; test the explicit editorial-removal transaction separately.

### P2.2 - Fix sentinel aliasing and HuMo's stale character guard

**Verified evidence**

- Captions use raw speaker names at `nodes/_otr_captions.py:180-187` and lookup at
  `242-270`; credits use the raw voice map at
  `nodes/otr_credits_roll.py:402-415`.
- The central cast lookup already handles aliases at
  `nodes/_otr_ledger_consumers.py:109-132`.
- ShotLock recognizes the c01 sentinel at `nodes/otr_shot_lock.py:240-264`, while
  HuMo's stale guard requires a literal character id at
  `nodes/render_driver.py:1262-1273`.
- The Einstein published captions omit an ANNOUNCER label for the sentinel around
  `Einstein captions:16,24-26`; Butterfly labels the intro sentinel but not the coda
  around `Butterfly captions:16,23,25`.

**Required fix**

Route caption and credits speaker/voice lookups through the same alias-aware cast
resolver used by ledger consumers. Preserve canonical display names in output, but do
not require a literal sentinel char-id in the render stale guard. For the sentinel,
match its role, source family, and portrait/shot-lock predicate instead.

**Regression sketch:** supply c01 intro and coda sentinels plus aliases across captions,
credits, ShotLock, and HuMo. Assert speaker labels and voice credits resolve, no stale
guard false positive occurs, and a non-sentinel similarly named row is still rejected.

### P2.3 - S3 soak metrics must be collected, not inferred

The current structured call retry defaults are three attempts
(`nodes/_otr_structured_call.py:65-81`, `482-710`); Fable2's news reader has two
outer attempts (`nodes/_otr_scifi_fable2.py:1243-1280`). P3 permits four temperature
steps (`176-182`), at most two budget retries (`185`, `1437-1457`), and one
truncation retry (`1404-1413`), for at most seven creative calls. Freeze phase
telemetry exists at `nodes/_otr_freeze_cascade.py:342-411` and `436-467`; reroll
records exist at `nodes/_otr_reroll.py:521-528` and `740-749`. The canonical
harness timeout is 5,400 seconds at `scripts/otr_headless_canonical.ps1:24`; the
watchdog hard-stall window is 300 seconds at
`scripts/otr_render_watchdog.ps1:5-15,21,87-91`.

Run four S3 cohorts: 30-word/default, 30-word/LTX, 350-word/default, and
350-word/LTX; N=30 random-RSS rolls each. A cohort is green only if at least 29/30
complete with zero structural/proof errors, zero `needs_full_rerun`, zero reroll
error, zero watchdog death, and all required canonical-path assets present.

| Metric | Acceptance threshold |
| --- | --- |
| Structured-call first-attempt rate | at least 95% per call type; at most one non-first attempt per 30-roll cohort |
| News outer retry | no more than 1/30; no exhausted outer retry |
| P3 ladder | at least 95% complete in <=2 calls; <=1 truncation and <=3 budget retries per cohort; no successful attempt 5-7 |
| Dossier/audit discards | 0 source-invalid events; alert above 10% dossier drop or above 2 audit discards/run |
| Freeze | 100% non-terminal on greens; `frozen_clean >= 29/30`, warns <=1, Doctor edits = 0, capability bypass = 0 |
| Runtime | establish five warm baselines per lane; p95 <=1.5x median, max <=2x median, absolute <=5,400 s, and no 300-s heartbeat gap |

Add telemetry for actual P3/P5 attempts, budgets, max budget, structured-call attempts,
news outer attempts, source-invalid events, dossier/audit discard reasons, policy
receipt, and a monotonic stage clock. Existing pass receipts, parse records,
`audit.discarded`, and normalization records are useful inputs, but they are not
sufficient to calculate every threshold above.

### P2.4 - Operator content pre-screen: advisory, not an automatic gate

This review inspected the published ledgers and captions only. It does not certify
audio performance, visual continuity, source attribution, or a legal/editorial policy.
No literal weapons, smoking, sex, branded-product, slang, machine-attribution, or
clear era-slip issue was found in the inspected text. No format-example name leakage
was demonstrated: example VERA/DOKU names occur in a source pack example, not in the
published spoken text.

| Episode | Verified advisory finding | Operator check |
| --- | --- | --- |
| Einstein's Echo | Cast c03 is LUCIA at `Einstein ledger:40-49`, yet c03 says `Lucia, I've reviewed...` at `245-249`, a likely self-address cue. The source/fictitious-person seam needs an editorial listen, though the coda is physically present at `371-417`. | Confirm intended addressee and whether the performers make the two-register contrast audible. |
| The Butterfly's Gambit | The treatment asks for distinct Dunn/Gray registers at `Butterfly ledger:614-632`; the compact spoken dialogue is at `257-366`. The ledger's `ending_unearned` advisory is recorded at `864-879`, and the coda/news block is at `403-450`. | Listen for speaker distinction and for a sufficiently earned turn into the news read. |

These are production-eyeball checks only. Do not turn them into automatic generation
gates without a ratified rubric and a measurable classifier or human review step.

**Required follow-up:** store this as an operator checklist attached to the final
episode dossier, with a named reviewer and explicit pass/needs-revision disposition.
There is no safe automated content rewrite implied by this advisory.

**Verification sketch:** use a fixture containing each sentinel/coda and two intentionally
similar cast voices to verify that the checklist is emitted, complete, and cannot be
mistaken for a passing automated freeze verdict.

## Recommended implementation order

1. Land P0.1 and P0.3 together: one early policy decision, no legacy mutation for
   Fable2, fresh root access after save, and a receipt that makes proof invariance
   testable.
2. Land P0.2 plus the complete P1.1 ownership/revision policy. This is the safe
   persistence foundation for the new delivery and cue fields.
3. Land P1.3 as a focused proof-preserving TTS slice, with the science-news parity
   fixture before any adapter changes.
4. Land P1.4 with the canonical workflow wiring in the same commit; validate the real
   workflow after editing it.
5. Land P1.5's pure contracts and fake full-loop tests before connecting a live P4/P5
   creative call.
6. Finish P2.1-P2.3, then run the measured S3 cohort. Keep the content screen as an
   operator review aid.

## Definition of production-ready for Fable2 S2

Fable2 is production-ready only when all P0 regressions are green; the canonical
workflow has a validated cue-manifest graph; a winning revision propagates one atomic
hash-consistent FinalDraft; all Fable2 TTS inputs are hash-linked delivery fields while
the proof map seals canonical text; the S3 cohort meets every stated threshold; and an
operator has completed the advisory listening/visual pass on the final artifacts.

Until then, the correct release status is **render-capable but not proof-safe**.
