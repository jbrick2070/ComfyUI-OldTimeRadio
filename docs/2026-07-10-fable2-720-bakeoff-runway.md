# 720-word Story-Engine Bake-off -- runway v5 FINAL (kibitz r1-r4 CONVERGED)

**Arc:** kibitz 2026-07-10 evening, `kibitz-runs/2026-07-10-720-bakeoff/`
(r1-r4; panel codex gpt-5.5 x4 + antigravity r3 manual + Claude anchor x4,
judge Claude). r4 = precision pins only -> CONVERGED. $0 spend (local panel).

**Baseline:** HEAD `b1cf0085` / `v2.0-alpha`. Already shipped, NOT
re-implemented: r2-QA P0 fold @ `47bf50f2`. Spec shapes:
`docs/2026-07-10-fable2-s2-QA-ANALYSIS-r2.md`; every graph/line literal is
re-derived LIVE at build.

**Goal:** blind-judged 4-contender bake-off at 720 words: (A) scifi_fable2,
(B) original_radio, (C) science_news, (D) GPT-authored fable2-pipeline voice
pack. PART 1 = fable2 720-capable; PART 2 = qualification + event. 720w is
inside S2's 120-900 band -- no act-chunking.

## PART 1

### C1 -- durable-field identity (BEFORE C3; blocking)
Authored music fields (cue_id, description, generation_prompt,
anchor_line_id, placement, target_duration_s, cue_spec_sha256) OWNED by
memory; set_music extended to carry them (drops them today,
production_ledger.py:1083-94). Durable render fields copy from disk ONLY on
identity match; changed hash invalidates wav/cache/timing. **Hash pins
(sorted-key JSON, no transient render fields):** `cue_spec_sha256` =
sha256(sorted-key JSON of {generation_prompt, target_duration_s, placement,
anchor_line_id}); lines `text_for_tts_source_sha256` = sha256(canonical
text); clips render-spec hash = sha256(sorted-key JSON of source/render
request fields, excluding outputs/timing). Supersede the blind-preserve
text_for_tts test (test_production_ledger.py:273-292) with same-hash-retain
+ changed-hash-invalidate; keep a legacy read-compat migration test.

### C2 -- text_for_tts delivery routing
(1) science_news byte-parity fixture FIRST (must pass before AND after
C2/C3). (2) Fable2 Phase 7 stamps EVERY non-skipped voiced line -- even when
delivery == canonical -- with text_for_tts + source sha + normalization
receipt; canonical text/counts/proof untouched (numbers->words + Dr.->Doctor
move here, restoring the pronunciation switched off in the P0 fold).
(3) ONE resolver returning (canonical, delivery); policy/lane gate INSIDE the
resolver (content-owned: absent/empty/stale sha = terminal BEFORE
generation; legacy: passthrough, zero behavior change). Used for line
filtering, neutral prep, adapter prep, delivery vectors, request hashing in
_otr_voice_node_common.py. (4) **Two-bus clip contract:** per-bus expected
line_id arrays (character bus node 81, announcer bus node 82); post-loop
terminal check that consumed count == provided count on BOTH buses --
shortfall AND surplus fail (SceneSequencer only catches shortfall today,
scene_sequencer.py:838-868). (5) Sidecar audit: concrete list (indextts2 /
chatterbox / dia), one-line uses-voice-common verdict each, recorded in the
C2 commit message.

### C3 -- cue manifest + canonical wiring (code + JSON + tests, ONE commit)
- StableAudioTheme node 83 -> cue_audio_clips + cue_manifest_json +
  render_log + done; every ledger.music[] prompt rendered straight to the
  episode audio dir; legacy opening/closing SLICED from the batch via the
  manifest.
- **Link 243 disposition EXPLICIT:** remove it. SignalLostVideo.closing_audio
  is declared AUDIO but unused (video_engine.py:~2081 declaration, ~2260
  unused -- confirm live); retire the input in the same commit. Re-enumerate
  ALL node-83 outbound links live before editing.
- **Exact input spec:** SceneSequencer node 3 + EpisodeAssembler node 7 each
  gain OPTIONAL forceInput sockets `music_cue_audio` (AUDIO) +
  `music_cue_manifest_json` (STRING), appended AFTER all existing inputs, no
  widget key (widgets_values untouched -- BUG-LOCAL-097). Canonical links
  added by input NAME after live re-derivation.
- **Manifest = versioned schema** (`manifest_version: 1`): rows carry cue_id
  (unique), batch_index (in-bounds), sample_count (>0), sample_rate
  (== AUDIO), prompt + prompt sha, seed, requested/actual duration, canonical
  output path. ONE shared parse/validate helper used by SceneSequencer,
  EpisodeAssembler, and tests. Slicing by manifest sample counts ONLY
  (pack_audio_batch right-pads, base.py:104+); batch count == row count or
  terminal. Keying = cue_id + batch_index, never positional.
- SceneSequencer inserts inter-scene cues at authored boundaries (net-new
  wiring: music rows are passthrough today) + stamps scene-relative timing;
  EpisodeAssembler extracts opening/closing + shifts inter-scene to master
  time; science_news legacy manifest = exact 3-slot parity.
- SAME COMMIT: update tests/test_full_workflow_v2_audio_wiring.py +
  tests/test_stable_audio_theme.py; ADD workflow audit failing on unlinked
  cue/manifest outputs or stale legacy cue links; OTR_WorkflowValidator +
  JSON round-trip + link referential + input-name + widgets_values-count
  audits.

### C4a -- S2 pure contracts (LLM-free, ships first)
Scene-count TABLE (word-band -> scene count), pinned + tested; exact
quotient/remainder allocation VECTOR (kills the scalar per_scene_words
drift). **Boundary verdicts explicit:** 30 + 119 = compact mode retained
(P2a/P4/P5 stamped skipped); 120/350/720/900 = full mode accepted (720 row
asserts the table's scene count + vector summing to exactly 720); 901 =
terminal reject BEFORE any creative call. FinalDraft frozen dataclass (raw
source, normalized source, ParsedScript, proof_map, P3/P5 receipts, score
tuple, sha set) with a PURE constructor -- parse/proof/score extracted from
the mutating _assemble path (today proof_map is built inside assembly while
it saves incrementally, _otr_scifi_fable2.py:1647-1802). Pure
validate_revision_contract (protected fields, unnoted edits, budgets;
malformed rules fail loud). Deterministic lexicographic keep-better judge
(draft 2 only if eligible AND strictly lower; tie keeps draft 1 + reason).
Full negative matrix.

### C4b -- S2 loop wiring
P2a 3-pitch select (ids {1,2,3} validated), P4 critic, P5 revision consuming
normalized markup + protected artifacts; P3 receipts record ACTUAL max
budget + attempts. Judge selects a FinalDraft; _assemble runs ONCE on the
winner (losing drafts never touch ledger/proof). **SAME CHANGE, all four
gate surfaces:** runner `_ONE_DRAFT_THRESHOLD` gate
(_otr_scifi_fable2.py:~236-264), writer run()-entry
assert_supported_target_words (OTR_LedgerScriptWriter.py:~3220),
pipelines.json pass rows for P2a/P4/P5 (~:52-80), and the gate assertions in
tests/test_fable2_artifacts.py:~676-679. Mocked full-loop test before live.

### C5 -- caption/credits alias + HuMo guard (before proof rolls)
Captions: guarded import of _otr_ledger_consumers.cast_lookup with
flat-import fallback (module is stdlib-only with a CLI path); test package
import AND CLI execution. Credits: resolve the VOICE row through cast_lookup
(ref_by_char is raw char_id today). HuMo stale guard:
role/source-family/shot-lock predicate, not literal char_id.

### C5.5 -- proof rolls (lock scope: 350w + 720w default lane only)
350w fable2 smoke -> triage: targeted tests -> fix at root (kibitz every
failure per operator directive) -> rerun targeted tests -> rerun smoke ->
proceed only when green -> 720w default-lane verification roll. Green bar:
frozen_clean + content_mutations=0 receipt + assets Test-Path'd at canonical
paths. (LTX roll cut from lock scope -- separate media-lane validation.)

## PART 2

### C6 -- qualification gate (all four; SAME run-sheet JSON as the event)
Run-sheet fields: contender id; FULL seed envelope with RESOLVED values
(OTR_FABLE2_SEED A/D, OTR_ORIGINAL_SEED B, OTR_CAST_SEED, OTR_STYLE_SEED);
provider/slot handle/resolved model slug/auth status/actual cost receipt --
**missing slug, auth status, or cost receipt = qualification incomplete;
contender excluded until rerun**; source id; ACTUAL word count (reported,
not pass/fail); freeze verdict + receipt per-policy (A/D frozen_clean +
content_mutations=0; B/C any non-terminal + doctor-edit/skip disclosure);
output paths; rerun log. **Env hygiene:** the resident ComfyUI process
bleeds env -- the runner SETS every envelope seed and CLEARS non-applicable
ones before each contender. Contender B runs its no-source spark lane
(news_close_brief hardwired empty + rejected if set,
_otr_original_radio.py:515-516) -- scored WITHOUT the news-seam axis,
stated. One 720w calibration roll each for B/C; tuning frozen after.

### C7 -- event
Pinned story for A/C/D. Contender D: reuses the scifi_fable2 frame deck
(_DECK_PATH hardcoded + sidecar-whitelisted); authors pack SEAMS + bank row
(`scifi_gpt_pack`, same pipeline); frozen committed authoring prompt;
recorded model slug; JSON-shape repairs only, diffs logged, zero wording
edits; passes the same lint (fable2-specific lint fixture = pipeline bug to
root-fix, never a D exemption). 4 x 720w; SOFT-blind labels (stated); ONE
logged rerun per failed render; scorecard: coherence, character
distinctness, dialogue, ending earned, news-seam (A/C/D only),
would-listen-again. Operator judgment final; no automated content gate.

## VERIFY-AT-BUILD checklist (r4)
1. science_news byte-parity fixture green before AND after C2/C3.
2. Every run loads workflows/otr_canonical.json, never a copy.
3. Node-83 outbound links re-enumerated live; link 243 disposition explicit.
4. Validator + round-trip + link referential + input-name + widgets-count
   audits green on the C3 commit.
5. No unlinked cue/manifest outputs; no stale legacy cue links.
6. Per-bus line_id order + exact consumed==provided clip counts (surplus
   fails too).
7. Captions package import AND CLI both pass.
8. Boundary tests 30/119/120/350/720/900/901 with the stated verdicts.
9. Runner sets + clears every envelope seed per contender.
10. Proof rolls: frozen_clean, content_mutations=0, canonical-path assets.
11. SignalLostVideo closing_audio confirmed unused live before retiring.
12. C2 sidecar audit list in the commit message.

## Deferred
P2.1 doctor-skip retirement (disclosure instead), P2.3 soak cohorts, >900w
act-chunk, cloud OpenRouter pins, source-consuming original_radio variant,
LTX-lane validation roll.
