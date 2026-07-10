# QA Analysis -- scifi_fable2 S1b freeze cascade and runway

**Audited:** 2026-07-10 against Windows working tree a8a32a83 on v2.0-alpha.

This was a read-only QA pass. No production code or canonical workflow was changed. The prior sentinel role-mismatch fix at 8e3d9228 is deliberately not re-reported here.

## Executive verdict

The LTX failure is not a Sprint-5C mutator. It is a pre-reroll corruption: Script Doctor asks to skip a row, then Ledger.save() merges stale disk text back into the intentionally cleared row. Sprint-5C subsequently targets the already-corrupt row, correctly fails on fable2's deliberately absent line_composer_system seam, and leaves the row for Phase 10 to reject.

The fable2 lane should not run legacy text-mutating quality passes. Its P4/P5/P8 loop owns that work. A fable2 capability gate is necessary, but it is not sufficient: the generic ledger merge and Script Doctor skip contract are separate production bugs.

| Priority | Verified finding | Effect |
|---|---|---|
| P0 | Reviewer skip plus stale-disk merge corrupts rows; fable2 then enters legacy reroll. | Blocks the LTX smoke and can corrupt any lane. |
| P0 | Legacy reviewer/readiness changes fable2 text after its proof artifact is sealed. | Breaks the lane's proof-map contract. |
| P1 | Inter-scene MUSIC rows are authored but neither rendered nor placed in master audio. | S2 multi-scene episodes lose intended transitions. |
| P1 | Caption and credits alias joins miss fable2's sentinel announcer id. | Delivered captions/credits lose announcer identity or voice receipt. |
| P2 | The HuMo stale-ledger radio-face guard can be bypassed after ShotLock maps announcer to c01. | Conditional profile-switch/reused-ledger face failure. |
| P1 before S2 | P2a/P4/P5 and the keep-better judge are schemas/seams only, not executable. | 120-900 word fable2 runs correctly fail today, but need full-loop contracts. |

## 1. Defect A: exact skip-mutator path

### Observed state

The LTX server log records this order:

1. Script Doctor completes and saves the ledger.
2. Stage 5B names shot_002_b3 as a target.
3. Sprint 5C attempts it twice and receives the expected StoryPackValidationError for line_composer_system.
4. A2 logs two residual structural errors and falls through.
5. Phase 10 rejects shot_002_b3 for non-empty text plus skip=True and a missing tts_skip_reason.

The persisted row in:

~~~text
C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes\pending_20260710_125522\audio\pending_20260710_125522_ledger.json
~~~

has the exact diagnostic shape:

~~~text
line_id='shot_002_b3'
speaker_role='character'
skip=True
reviewer_skip_reason='skip'
text='Abort? We'll lose the mission.'
char_count=0
word_count=0
tts_skip_reason absent
~~~

The earlier pending_20260710_105235 fable2 ledger has the same signature.

### Verified code path

1. The cascade invokes the reviewer before the story critic at nodes/_otr_freeze_cascade.py:759-763.

2. ReviewerEdit still permits action="skip" at nodes/_otr_ledger_reviewer.py:261-280. The Script Doctor prompt also offers that action at :1233-1266.

3. apply_doctor_edits() mutates the candidate at nodes/_otr_ledger_reviewer.py:1836-1842:

   - line["skip"] = True at :1837;
   - it records only reviewer_skip_reason at :1838;
   - it clears text, char_count, and word_count at :1840-1842.

4. review_ledger() commits that candidate and calls led.save() at nodes/_otr_ledger_reviewer.py:2114-2136.

5. Ledger.save() calls _merge_with_disk() at nodes/production_ledger.py:1175-1196. Its generic per-row rule copies every disk field into an in-memory row whenever the in-memory value is None, "", [], or {} at :1330-1337. It therefore treats the intentional empty text as absent and restores the old text from disk. It does not restore the zero counts because 0 is not considered absent. save() then replaces self.data with this merged payload at :1228-1234.

This proves the row is corrupt before Sprint-5C starts. It also explains the absent compose-flag breadcrumb: the writer did not compose that row in this path.

### What Sprint-5C actually does

run_targeted_reroll() filters only on speaker_role == "character" at nodes/_otr_reroll.py:623-642, not on skip. It builds the legacy LineRequest and calls compose_line() with the episode source bank at :660-667. The composer asks for line_composer_system at nodes/_otr_line_composer.py:2058-2072, while the fable2 pack correctly lacks that seam. The no-fallback accessor correctly raises at nodes/_otr_story_pack.py:237-244.

The broad error handler at nodes/_otr_reroll.py:679-688 only records the error and continues. There is no missing restore because reroll has not mutated the row. Its "keeping the original line" wording means keeping the pre-existing corrupt post-review state. A successful reroll would also be unsafe for a skipped row: update_line_text() at :701-704 updates text and counts but does not clear skip or skip reasons.

Phase 7 deliberately ignores skipped rows at nodes/_otr_readiness.py:223-235, so it cannot repair this state. Phase 10 correctly rejects both bad invariants at nodes/_otr_ledger_freeze.py:335-352.

### Recommendation: choose (c), plus two independent root fixes

Choose **(c)** for fable2: capability-gate legacy critic/reroll machinery before Stage-7 dispatch. Do not catch StoryPackValidationError and continue.

The governing capability should derive from the resolved story pipeline: fable2_multipass has no line_compose pass, while legacy_many_pass does; see nodes/story_packs/pipelines.json:6-31 and :76-116. A missing line_composer_system seam is a valid minimum check, but the pipeline pass graph is the real contract and avoids treating an accidentally broken legacy pack as an intentional no-op.

For an inapplicable lane, stamp an explicit disposition such as legacy_reroll_inapplicable, retain optional telemetry, bypass Stage-7 escalation and A2, and continue to normal structural freeze. Do not stamp needs_full_rerun for an intentionally bypassed legacy pass. This must cover both default Stage-7 fallback to LINE at nodes/_otr_story_select.py:365-387 and nodes/_otr_reroll_escalation.py:385-396, and the enabled escalation route that can turn uneven into EPISODE at _otr_story_select.py:303-329 and _otr_reroll_escalation.py:357-367.

Also make these generic fixes:

1. **Make the disk merge ownership-aware.** Preserve only explicitly out-of-band audio/video extension fields when absent. Never overwrite canonical owned fields such as text, counts, ids, role, skip, or skip reasons with stale disk state. This fixes the actual corrupt row without weakening the durable audio-field merge that BUG-LOCAL-108 was meant to preserve.

2. **Retire Script Doctor skip as a quality edit.** The reviewer itself says a mute is a pipeline cut and should not be used at nodes/_otr_ledger_reviewer.py:2096-2101, but its schema/prompt/apply path still permits it. Removing it is the consistent root fix. If the product intentionally keeps a mute action, it must atomically preserve empty text through save and stamp a valid tts_skip_reason; reviewer_skip_reason cannot satisfy Phase 10.

Options (a) and (b) are not primary fixes: reroll never sets skip, and restoring on its compose-error branch would mask the upstream ledger bug.

### A2 contract correction

A2 calls the gap audit at nodes/_otr_freeze_cascade.py:1051-1062, logs "never refusing" at :1063-1072, then Phase 10 hard-refuses the same structural errors. The correct contract is narrower: **residual quality findings can ship; structural gaps cannot.** A2 must not advertise an unconditional ship-through posture when its own preflight already finds Phase-10 errors.

The comments claiming a pre-reroll restore at :924-927 and :1033-1034 are stale; current reroll code explicitly keeps successful recompositions.

### Defect-A regression set

1. In tests/test_production_ledger.py, use a real on-disk Ledger: save a non-empty row, intentionally clear it with a valid skip state, save again, and assert the cleared text/counts survive while an unrelated disk-only audio field survives too.
2. In tests/test_phase3_ledger_reviewer.py, assert Script Doctor cannot emit/apply a mute action. If retained, assert the full valid skip shape and a real save/reload round trip.
3. In tests/test_otr_reroll.py, assert skipped targets are not composed. If a revive feature is introduced, require an explicit atomic clear of skip and every skip-reason field.
4. In tests/test_lfc_freeze_cascade_orchestrator.py, run fable2 fixtures with critic targets under both default and enabled Stage-7 escalation. Assert no legacy composer call, no A2 entry, an inapplicability breadcrumb, and a clean structural freeze.

Focused Windows reviewer/reroll/cascade checks passed (247), but their cascade fixtures use a no-op save() and cannot expose the real disk-merge failure.

## 2. Other freeze-cascade incompatibility: proof provenance

The fable2 runner seals meta.fable2.proof_map during P7 at nodes/_otr_scifi_fable2.py:1661-1672 and :1800-1802, then runs P8 before the freeze cascade at :2144-2158. The generic cascade allows two later canonical-text mutation paths:

- Script Doctor rewrite writes replacement text at nodes/_otr_ledger_reviewer.py:1811-1835.
- Phase 7 rewrites canonical line["text"] and counts at nodes/_otr_readiness.py:264-268.

The published green Einstein's Echo ledger is direct evidence. Its proof map still points to the fable2 draft while the frozen ledger contains a Script Doctor rewrite of shot_001_b3 and Phase-7 changes such as Dr. -> Doctor and 0.2 -> zero point two. The current code paths remain reachable after the sentinel role fix; they are not proof-preserving.

This violates the lane's stated law that every spoken row traces to a named LLM artifact. It also lets a legacy technical-model writer operate after fable2's P8 audit and outside planned P4/P5 revision protocol.

**Root fix:** for fable2, bypass all legacy text-mutating reviewer/critic/reroll passes, not only compose_line. Retain deterministic structural validation, but make it read-only. P4/P5 must be the only content revision route and must re-parse, rebuild proof spans, and re-run P8 before P6/P7 consume a new draft. For TTS pronunciation, preserve canonical text and use a dedicated delivery field that every TTS adapter explicitly consumes; the existing text_for_tts schema field is not currently a live replacement consumer.

Add an end-to-end fable2 freeze test that compares every frozen voiced line to its proof constituent(s) after the full cascade. Any post-proof canonical text change must either fail the proof check or be a new complete P5 artifact with fresh proofs.

## 3. Downstream media and delivery landmines

### 3.1 Inter-scene music is structurally unwired (P1)

Fable2 preserves every authored inter cue in ledger.music as inter_01, inter_02, and so on at nodes/_otr_scifi_fable2.py:1703-1744. This is the right assembly shape, but no downstream path consumes it:

- StableAudioTheme renders exactly fixed opening, closing, and interstitial slots at nodes/stable_audio_theme.py:38-43 and :208-225, using meta-derived prompts rather than ledger.music[].generation_prompt.
- In canonical workflows/otr_canonical.json:1, node 83's third interstitial_theme_audio output has no link. Node 7 only accepts opening and closing theme inputs, via links 241-243.
- SceneSequencer sends music_* lines through without a segment or timing stamp at nodes/scene_sequencer.py:753-831. EpisodeAssembler only places opening and closing at :1043-1112 and :1342-1379.

A green fable2 ledger already has inter_01 with null timing while opening and closing are timed. S2's three-scene plays will silently lose authored inter-scene audio and visual spans.

**Root fix:** replace the fixed three-output theme contract with a cue-list manifest keyed to source line/shot id. Render every ledger.music prompt, place each cue into master audio at its authored boundary, and stamp matched music row/sentinel with WAV path and timing. This is a code plus canonical-workflow change; update workflows/otr_canonical.json in the same patch.

Regression: two- and three-scene fable2 fixtures must prove every cue used its authored prompt, produced a WAV, has nonzero master timing, appears in master audio, and earns a nonzero visual span.

### 3.2 Announcer aliases disappear in captions and credits (P1 quality)

Fable2 correctly uses char_id="announcer" on announcer line rows at nodes/_otr_scifi_fable2.py:1692-1696 and :1759-1786 while retaining c01 ANNOUNCER for voice routing. Two exact-id consumers do not follow that alias:

- Captions build a char_id-to-name map from cast rows at nodes/_otr_captions.py:180-187. A sentinel row therefore loses its ANNOUNCER: speaker label.
- Credits uses an alias-aware display name, but maps voice_ref_id by raw cast id at nodes/otr_credits_roll.py:402-415. The announcer transcript entry loses its delivered-voice receipt.

Use the shared speaker/cast alias resolver for both fields. Add a c01 ANNOUNCER cast plus timed char_id="announcer" rows, then assert the caption label is ANNOUNCER: and the credits transcript carries the cast voice/ref label.

### 3.3 Conditional HuMo stale-ledger guard bypass (P2)

ShotLock deliberately converts the fable2 sentinel to cast id c01 for its beat join at nodes/otr_shot_lock.py:240-255. The render driver's fail-closed radio-face guard checks only literal char_id == "announcer" at nodes/_otr_video_engines/render_driver.py:1253 and :1268-1282. A reused or profile-switched ledger can therefore send a stale c01 radio_object portrait to an announcer_visual HuMo shot without the intended failure.

Fresh HUMO-on image generation currently produces console_face, and the requested LTX smoke uses character ltx_audio_in, so this does not block the next LTX run. It is still a real fail-closed gap for reused ledgers. Guard by announcer role/cast identity rather than one spelling of char_id. Regression: sentinel -> ShotLock c01 -> HuMo-on plus stale radio_object must raise RenderError.

### 3.4 Verified compatible media joins and one non-blocking inefficiency

No functional blocker was found in CastLock/TTS, character portrait creation, or a fresh character ltx_audio_in IA2V run:

- fable2 P6 supplies character_description on character cast rows at nodes/_otr_scifi_fable2.py:1521-1557;
- image prompts read it by char_id and create portraits/scene targets at nodes/otr_meta_brief_image_prompt.py:64-79, :1613-1643, and :1806-1839;
- fable2 character rows map to character_video at nodes/otr_shot_lock.py:55-84;
- IA2V TALKING requires a real portrait and fails loud when absent at nodes/_otr_video_engines/render_driver.py:1423-1458. Wan and generic cloud I2V paths correctly take a wide scene still instead.

There is a non-blocking efficiency defect: image-prompt generation creates both c01 ANNOUNCER and synthetic announcer portraits. Only the sentinel identity is routed by fable2 line/render paths. Deduplicate at the image-prompt boundary without removing c01, which remains needed for voice resolution.

## 4. S2 and long-episode runway

### Current ceiling matrix

S1b correctly rejects every target at or above 120 words before source work at nodes/OTR_LedgerScriptWriter.py:3205-3227 and nodes/_otr_scifi_fable2.py:236-264, :1990-1995.

| Target words | Current result | Envelope | Character band | P3 / retry budget | Micro cap |
|---:|---|---|---|---|---|
| 30 | allowed | 1 scene, 30 words | 5-55 | 1200 / 1500 | 4 lines |
| 120 | S2 required | 1 scene, 120 words | 95-145 | 1200 / 1500 | none |
| 350 | S2 required | 3 scenes, 117 words/scene | 280-420 | 1200 / 1500 | none |
| 900 | S2 required | 8 scenes, 112 words/scene | 720-1080 | 2180 / 2725 | none |
| 901 | rejected ceiling | n/a | n/a | n/a | n/a |

The envelope lives at nodes/_otr_scifi_fable2.py:755-764, aggregate band at :221-233, token budget at :215-218, one MISSING_END retry at :1404-1415, and sub-60 micro cap at :1283-1305. The 350-word raw formula is 970 but the actual floor is 1200; the 4200 cap does not bind until 1819 words. The current arithmetic is coherent.

Landmines before enabling S2:

1. _SCENE_WORD_BAND is prompt text only. P3 checks aggregate words but does not enforce per-scene allocation at _otr_scifi_fable2.py:1428-1457. A 350/900 play can meet its total while concentrating dialogue in one scene. Independently rounded targets also do not sum exactly at 350 or 900. Use an integer allocation and validate each scene against it.
2. The P3 receipt stamps the initial budget rather than a larger retry budget. This is observability drift, not a current failure; record the actual maximum used.
3. P2a/P4/P5 are staged only. The runner hardcodes one_pitch_one_draft, deals one card, sets critic=None, and jumps P3 to P6 at _otr_scifi_fable2.py:1995, :2055-2067, and :2098-2124. PitchSelect and CriticNotes schemas exist at :322-324 and :422-448, but _pass_select, _pass_critic, _pass_revision, and _defect_score do not.
4. Full mode must validate pitch ids exactly {1,2,3} and require P2a to select a real slate member.
5. Do not use _script_view() as the P4/P5 draft. It omits opening, inter, and closing MUSIC lines at _otr_scifi_fable2.py:1809-1826.

### Required P5 contract

P5 should accept the normalized complete draft, not _script_view(), and run the same parser as P3. Add a pure validate_revision_contract() that makes a revision ineligible unless it preserves title, cast, scene sequence, skeleton, and explicit ownership of scene 0 plus every inter-scene MUSIC cue. Unnoted scenes must match the normalized draft byte-for-byte. A failed/ineligible or tied revision retains draft 1 with a visible reason; it must never silently turn a good draft into a hard failure.

The future whole-play defect score must be new and deterministic. Do not reuse legacy line_quality_defect_score, which is line/request-specific and fail-opens to zero on exceptions at nodes/_otr_line_composer.py:2375-2388. A valid winning revision must atomically update parsed artifact, P6 input, P7 proof artifact/map, P8 input, draft1_sha256, final_sha256, and better_draft_choice.

### S2 acceptance tests

1. Matrix test: 30/119 low mode; 120/350/900 full mode after S2; 901 rejects before fetch/LLM calls. Check pass receipts and order.
2. Full fake-LLM 350-word run: three unique pitches/cards, valid selection, critic rejects unknown scene/speaker, valid revision updates all artifacts, and invalid/truncated/worse revision preserves draft 1 with a reason.
3. Revision-law tests: unnoted-scene mutation, title/cast/scene/music/coda drift, and ambiguous scene-0/inter-cue ownership all make draft 2 ineligible.
4. Cross-media fixture: c02/c03 portraits and scene targets exist; HuMo/IA2V TALKING choose portraits; LTX non-talking, Wan, and generic cloud I2V choose scene stills; announcer sentinel resolves correctly.

## 5. Legacy and source-bank containment

The nested structured-call clamp is shared behavior, not fable2-only. It first validates normally, then deep-clamps only string_too_long paths and re-validates at nodes/_otr_structured_call.py:322-370 and :420-474. Valid legacy output remains untouched, but an overlong nested legacy response now succeeds after bounded truncation rather than retry/failure. Add science_news tests for a byte-identical valid golden response and the intended nested-overlong behavior.

The reviewer sentinel guard only rejects a role flip when the row carries a non-character sentinel at nodes/_otr_ledger_reviewer.py:1086-1117; sentinel definition is nodes/production_ledger.py:90-96. The branch still permits a normal real-character repair. Add one legacy sentinel test and one real-character role-repair test; no new legacy behavior break was verified in this inspection.

## 6. Defect B acceptance sequence after fixes

After the P0 fixes, repeat the prescribed LTX smoke with the real canonical workflow and source_bank=scifi_fable2, target_words=30, and character_visual=ltx_audio_in. Reset the resident server selectively before booting. A green result requires all of the following:

1. The fable2 structural freeze verdict contains no legacy-reroll attempt against line_composer_system.
2. Every frozen voiced row still agrees with its proof map.
3. CastLock resolves every character voice and announcer alias correctly.
4. Every character IA2V beat has a portrait, own audio, and final video clip; no fallback is taken.
5. The final asset exists under its canonical otr\episodes\<episode> location and the published file exists under otr\obs.

Only that run proves the complex media path. The current code inspection proves character portrait/register joins are compatible, but cannot substitute for an end-to-end LTX render.
