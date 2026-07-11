# OTR Handoff Log

Append-only session log, newest at top. What each session actually did;
GO_FORWARD_PLAN.md stays lean and forward-only.

## 2026-07-11 -- HEAD 2f335c28 (v2.0-alpha) [720-bakeoff C1/C2 coder window]

Did:
- C1 SHIPPED @ 9949bb6e: durable-field identity in production_ledger --
  _row_identity gates the disk merge so durable render fields (wav/timing)
  copy forward ONLY on unchanged content identity (lines=sha of text,
  music=cue_spec_sha256, clips=render-spec); empty-source -> no gate (skip/
  clear preserves durable per the ownership contract). set_music now carries
  anchor_line_id/placement/target_duration_s + stamps cue_spec_sha256. 5 new
  tests; golden fable2 fixture regenerated. Suite 7468/31/1 + Bible green.
- C2 SHIPPED @ 2f335c28: text_for_tts delivery routing. _otr_readiness
  stamps text_for_tts + source sha + receipt on fable2 voiced lines (canonical
  untouched -- restores the pronunciation the P0 fold switched off). NEW
  _otr_text_delivery resolver (LEGACY passthrough = byte-identical spine;
  CONTENT_OWNED = verified stamp, absent/stale = terminal before gen). Voice
  node routes prep/vector/hash through it. scene_sequencer two-bus surplus+
  shortfall terminal check. 26 new tests incl. science_news byte-parity fixture.
  Suite 7494/31/1 + Bible green.
- C3 wiring kibitz'd (r3, Codex + Claude Code grounded; Antigravity timed out).
  HARDENED spec = docs/2026-07-11-c3-cue-manifest-wiring/FINAL_HARDENED_PLAN.md.
  Surfaced real build-breakers before touching the canonical JSON: legacy
  ledger.music[] is empty (node 83 must synthesize legacy cues; inter_NN
  KeyErrors compose_music_prompt), sentinel lines have no cue_id (use C1's
  anchor_line_id), node-7 input deletion = widget-slot drift (keep declared),
  music must be a 3rd bus, slice by sample_count (no silence-trim) + resample.
Current step: 720-bakeoff C3 (cue manifest + canonical workflow wiring) --
CODE-READY per the hardened spec; canonical-JSON rewire, one atomic commit.
Next: build C3 in a fresh window from FINAL_HARDENED_PLAN.md (re-derive live
literals per the VERIFY-AT-BUILD list); STOP after C3 green+pushed.
Commits: 9949bb6e (C1), 2f335c28 (C2) -- both pushed. C3 docs this commit.

## 2026-07-10 ~14:20 -- HEAD af378aad (v2.0-alpha) [scifi_fable2 S1b coder window, QA fold]

Did:
- External QA analysis (docs/2026-07-10-fable2-s1b-QA-ANALYSIS.md) folded: it
  OVERTURNED the 5C-mutator theory -- real chain = doctor 'skip' clears text ->
  Ledger.save() stale-disk merge resurrects old text -> Phase 10 gap. P0 fixes
  shipped @ af378aad: ownership-aware merge (_MERGE_OWNED_ROW_FIELDS), doctor
  skip stamps tts_skip_reason, 5B/5C lane capability gate
  (_legacy_line_compose_applicable; fable2 pack has no line_composer_system).
  QA regression file tests/test_ledger_merge_ownership.py. Suite 7451/31/1.
- LTX MEDIA PATH GREEN: "The Butterfly's Gambit" published to obs (1787s,
  41.8 MB) -- character lane ltx_audio_in + stills; capability gate fired live;
  freeze passed; canonical no-diff.
Current step: fable2 S2 (full loop, 350w) with the QA runway items folded in:
proof-provenance (doctor/Phase-7 rewrite after proof seal -> text_for_tts),
inter-scene music wiring, caption/credits sentinel alias, HuMo stale guard,
per-scene band allocation (all pinned w/ file:line in the QA analysis doc).
Next: S2 in a fresh coder window; operator eyeball on both fable2 episodes.
Commits: af378aad (+ this docs commit) -- pushed.

## 2026-07-10 ~13:15 -- HEAD 8e3d9228 (v2.0-alpha) [scifi_fable2 S1b coder window]

Did:
- S1b SHIPPED: runner + dispatch + registry flips + 80+ tests @ a24b75c4;
  25-roll live-smoke hardening (kibitz r2/r3/r4 + sonnet/opus fan-out per the
  new kibitz-every-failure directive) @ ff4c226d + 8e3d9228. FIRST GREEN
  EPISODE: "Einstein's Echo" in obs (570s); canonical no-diff + validator OK.
- ROOT-CAUSE fix: reviewer role_mismatch flipped sentinel announcer rows to
  character breadcrumb-lessly (sonnet+opus converged on reviewer.py role
  branch); symmetric guard + breadcrumb + regression tests shipped.
- OPEN BLOCKER: cascade 5C-reroll failure path stamps skip=True on target
  rows when fable2's pack (correctly) lacks line_composer_system -> Phase 10
  needs_full_rerun. LTX media roll (stills+ltx_audio_in via _tmp probe,
  16gb_full + character_visual override) got 25 min deep; blocked on this.
- External-QA brief written per operator: docs/2026-07-10-fable2-s1b-QA-
  PROBLEM-STATEMENT.md (big problems + full downstream landmine audit ask).
Current step: resolve the skip-mutator blocker (QA brief) -> green LTX-lane
fable2 roll -> then fable2 S2 (full loop, 350w).
Next: operator runs the QA brief through the external analyst; fold findings.
Commits: a24b75c4, ff4c226d, 8e3d9228 (+ this docs commit) -- all pushed.

## 2026-07-10 ~08:00 -- HEAD c932880f (v2.0-alpha) [scifi_fable2 coder window]

Did:
- scifi_fable2 S1a SHIPPED: writer tail (J.5 -> M save) extracted into
  `_run_writer_tail(ctx)` + 17-field WriterTailContext (doc s11 pins);
  moved body verified character-identical vs pre-extraction modulo the 2
  pinned gates (title override precedence + run_story_spine gate, s14/8);
  late _OTRC/_PL imports followed the tail. 11 new tests
  (test_fable2_tail_context.py: ctx contract, no-closure, delegation,
  same-run byte identity, spine gate both ways, title precedence x3,
  refine stash x2). 3 AST pin modules updated to follow the move
  (story_brief_c5a2 fixture, announcer title-regen pin, title scratchpad).
  ROOT-CAUSE find: my byte-identity test leaked production_ledger._CURRENT
  (singleton) -> broke lfc C4 tests downstream; autouse save/restore
  fixture added. Commit `948c5a0a`.
- ONE legacy science_news 30w live smoke on the extracted tail: RESULT
  SUCCESS 555s (baseline band), "Etna's Secret" published to obs (60.7 MB,
  Test-Path confirmed); J.5 regen fired live (title_source=
  llm_post_composition). Ledger scrubbed (paths anonymized, article text
  truncated, all keys/rows kept) -> tests/fixtures/fable2/
  legacy_reference_ledger.json + README. Commit `c932880f`.
- Gates: suite 7332/31/1 + Bug Bible 17/7/3 green at 948c5a0a (+ post-
  fixture full-suite re-run green); BOM/AST/0-byte/HEAD==origin verified.
  Also committed a leftover ENGINE_MATRIX docs hunk from the prior
  session (`5f5820a7`).
Current step: scifi_fable2 S1b -- spine, live (runner P0/P1-one-pitch/
P2b/P3/P6/P7 + P8 audit-only; flip runnable+executable SAME change; doc
s13 S1b test set; 30w live smoke; validator no-diff record).
Next: S1b in a coder window (doc sections 5/8/11/13; re-pin splice lines
in the S1b commit).
Commits: 5f5820a7, 948c5a0a, c932880f (+ this docs commit) -- all pushed.

## 2026-07-10 ~06:45 -- HEAD d7379920 (v2.0-alpha) [scifi_fable2 coder window]

Did:
- scifi_fable2 S0 SHIPPED (all inert, doc = 2026-07-10-scifi-fable2-architecture.md):
  banks.json row before custom_source_bank + fable2_multipass pipeline row
  (registry-legal slots); 9-seam pack scifi_fable2_v1.json (FORMAT block
  byte-identical script/revision); frame_deck.json 14 cards + 6 stances +
  sidecar registration; detection-only story_rules (empty replacements);
  _otr_fable2_markup.py parser (full defect enum, collected defects, split
  word counters, per-constituent lines); 66 new tests incl. rss-not-spark,
  slot-enum rejection, deck lint, science_news pinned row. Doc s14 pins
  1/5/10 resolved in-doc. science_news untouched; NO workflow diff.
- COMMIT NOTE: my staged S0 files were swept into the freeze-cascade
  window's commit d7379920 mid-session (one bundled commit, pushed). Content
  verified file-by-file; full suite re-certified at that HEAD.
- Gates at HEAD: suite 7321 passed/31 skipped/1 xfailed; Bug Bible 17/7/3;
  BOM/AST/JSON verify clean; HEAD == origin.
Current step: scifi_fable2 S1a -- tail extraction ALONE (writer
_run_writer_tail(ctx) + WriterTailContext, byte-identity pin
test_fable2_tail_context.py, ONE legacy science live smoke, then scrub the
ledger into tests/fixtures/fable2/legacy_reference_ledger.json). Nothing
fable2-visible ships in S1a.
Next: fresh coder window claims the slot, reads doc sections 11+13+14, does
S1a only, then S1b (spine + runnable flip same change).
Commits: none under my own SHA (work rode d7379920); this docs commit.

## 2026-07-10 ~02:45 -- HEAD 636d78cf (v2.0-alpha) [original_radio window]

Did (operator overnight directive: "run two more 420w, analyze, optimize
the original path, prompts not py"):
- 420w night batch, 4 rolls total. PUBLISHED: "Ashes of the Pawn"
  (otr\obs\signal_lost_ashes_of_the_pawn_20260710_014548_..._final.mp4,
  18 min e2e). Roll A died at QA: the confirm judge "proved"
  news_source_framing by quoting the CLEAN intro verbatim -- fixed at
  root (3d32b265: news_source_framing + machine_attribution join
  weapons as lexicon-only kill classes; suite 7153 green then). Roll C
  died HONESTLY: writer armed a climax ("holding his revolver") --
  correct lexicon kill. Roll D died at concept: empty cast name x2
  (archetype "The Stenographer").
- ANALYSIS (leg 1): 239/420 words (thin brief -> thin outline);
  key_terms landed 1/5 (story diverged from concept); intro
  ventriloquized a character quote; ZERO quote-wrapped lines and ZERO
  stage directions at 420w (30w observations did not recur); no audible
  name drift (visual portrait prompt invented "Ferrywoman Edith" --
  eyeball item); outro button landed well.
- OPTIMIZED (prompt/data only, 636d78cf, pack JSON): concept demands
  non-empty CAPS personal names w/ example; script_brief demands
  episode-shape (opening/two turns/closing image) + key_term weaving +
  no-arms menace rule; both intro seams forbid quoting characters.
- NOT re-verified live: the portability coder window claimed the repo
  mid-session (S1 in flight, 9 py files dirty + llm_policy.py
  untracked); full suite red from ITS tree, my lane tests 42/42 green.
  NEXT lane action = one 420w verification roll AFTER the portability
  window settles, then eyeball all published episodes.
Current step: original_radio pre-ship -- operator eyeball (now 2
episodes in obs: page_in_the_tempest 30w, ashes_of_the_pawn 420w) +
one post-tune 420w verification roll.
Next: eyeball; verification roll; source-bank e2e sweep.
Commits: 3d32b265, 636d78cf (+ this docs commit) -- pushed. Suite was
7153 green pre-portability-dirt; Bug Bible 17/7/3.

## 2026-07-10 ~01:30 -- HEAD 1c735c2d + docs (v2.0-alpha)

Did:
- LIVE 30w original_radio OBS smoke: GREEN on roll 6 -- "Page in the
  Tempest" published (otr\obs\...20260710_010652...final.mp4, 48 MB,
  RESULT SUCCESS, 548s). Five real production bugs found+fixed at root
  across the failed rolls, each with tests, suite+bible green, pushed:
  7f459e21 (A2 verbatim grounding: ws-normalized match + typed repair +
  deterministic key_term prune -- the prune FIRED live on a later roll),
  75173fc4 (original_qa evidence bar: hard kills need lexicon
  corroboration or a confirm-pass verbatim quote; discards stamped LOUD),
  a61ab2ed (kill authority per class: weapons/anachronism lexicon-only
  -- a grounded quote proves the line, not the class), 6fdf3f6e (ladder
  logs raw-output head on every failure -- exposed gemma truncation),
  d526c8b7 (creative slot -> nemo in canonical: gemma-4 Q8 cannot hold
  n_ctx 4096 on 16GB, the silent 2048 downgrade truncated concept JSON;
  enforces the standing bake-off rejection), 1c735c2d (epilogue_missing
  deterministically refuted when the outro row exists + slot pins
  retargeted).
- Bug Bible +BUG-11.26 (verbatim-grounding gates) + static tripwire +
  kebab fix, pushed (survival guide @ 1a01037).
- Validator record: OTR_WorkflowValidator OK in the green run (23/55,
  drift=0); the lane itself = NO workflow diff.
Current step: original_radio pre-ship -- smoke + validator gates GREEN;
OPERATOR EYEBALL is the only remaining gate (content notes in
GO_FORWARD section 0: name drift, stage-direction leak, quote-wrapped
lines, sci-fi premise tension).
Next: operator eyeballs the published mp4; then source-bank e2e sweep.
Commits: 7f459e21, 75173fc4, a61ab2ed, 6fdf3f6e, d526c8b7, 1c735c2d
(+ this docs commit) -- all pushed. Operator's own windows added
b288d8b6, bff86af9 (portability docs, benign).

## 2026-07-09 ~night -- HEAD 604ccdd3 (v2.0-alpha)

Did:
- /kibitz r2 (coding plan) on ARCHITECTURE_V4 + INTRO_REWRITE_SPEC:
  anchor-first, Codex auto green, agy auto timed out -> operator pasted
  the manual prompt, its review judged. 3-way convergence; shape A
  locked; synthesis = R2_CODING_PLAN.md. Operator left ("do r3-r4 and
  start coding") -> full autonomy.
- /kibitz r3 (wiring): 5 codex must-fixes verified+folded (seam-accessor
  wall, briefs return shape, dual source_meta restamp, title-regen
  staleness root-cause, QA-before-aggregates order) = R3_WIRING_DELTAS.md.
  /kibitz r4: converged, pins P1-P8 (agy auto dead 3x; codex + anchor).
- BUILT + PUSHED CHUNK A `181506e8` (intro rewrite all banks + title fix;
  c5a2 pin retargeted to the script_text L-opener per its own docstring).
- BUILT + PUSHED CHUNK B `604ccdd3` (the whole original_radio
  SAME-COMMIT set, runnable:true). Mid-build catches fixed at root:
  spark deck needed the routing pack-SIDECAR registration; the
  bank-shape dispatch needed the runnable conjunct (custom keeps its
  pinned LOUD SourceContractMissingError path).
- Suite 7136/31/1 + Bug Bible 16/7/3 green after each chunk; AST/BOM/
  0-byte verify clean; HEAD == origin. No workflow JSON diff.
- Note: `3060fd3a` (portability brief) is the operator's own docs commit
  from his other window -- audited, benign.
Current step: original_radio campaign -- BUILD SHIPPED; remaining gates =
live 30w original_radio smoke + OTR_WorkflowValidator no-diff record +
OPERATOR EYEBALL (queued).
Next: run the live 30w smoke (selective reset first), then eyeball, then
the source-bank end-to-end sweep.
Commits: 181506e8, 604ccdd3 (+ this docs commit) -- all pushed.

## 2026-07-09 ~evening -- HEAD 5a09984c (v2.0-alpha)

Did:
- 5-agent Sonnet QA fan-out on all 4 source-bank routes + ledger contract
  (operator skipped further live smokes). Synthesis:
  docs/2026-07-09-source-route-qa/QA_SYNTHESIS.md (local; dated dirs are
  gitignored).
- FIXED+PUSHED closing-seam bank routing (QA F1) -- coda/announcer
  seams pack-route; PD+Shakespeare coda re-authored to bridge contract;
  title_form_label wired; 30 tests. SHA CORRECTION (codex fan-out catch):
  the CODE+TESTS live in `40535ddc` (the operator's Codex loop committed
  the in-flight tree bundled with its dia hardening); `321bcc9c` on top
  carries only docs (dated doc dirs gitignored). Cite 40535ddc for the
  closing-seam code.
- FIXED+PUSHED 5a09984c: produced-story meta split -- K.5.6 summary pass
  stamps meta["produced_story"]; credits/HUD/treatment/music repointed.
- Seated tencent/hy3:free on the roundtable panel until 2026-07-21
  (62962121) + CLAUDE.md section 8 arc routing (R1 cloud, r2-r4 kibitz).
- original_radio R1 COMPLETE: ARCHITECTURE_V1 + anchor review -> live
  4-model roundtable (GPT-5.6-sol / Gemini-3.1-pro / DeepSeek-v4-pro /
  hy3:free; ~$0.13) -> pass01_judgment.md -> ARCHITECTURE_V2.md. Key
  redesigns: creative front (concept/select/brief) runs INSIDE
  build_original_briefs at D.2 BEFORE structure; v2-plan naming adopted
  (original_multi_pass + original_*_system seams); whole-script
  original_qa gate; disclosure must EXPLICITLY say machine-generated;
  cast pass collapsed; num_characters widget feeds the concept pass.

- R1 pass02 run on ARCHITECTURE_V3 (operator overrides: Hitchcock ironic
  epilogue instead of spoken disclosure; NO era frame / raw timeless
  story; RUNNABLE ON BUILD, no staged flips, no fallbacks, HARD FAILS
  ACCEPTED; north star = max story complexity / max code elegance).
  Panel 4x"no" -> judged -> **ARCHITECTURE_V4.md = BUILD SPINE**. Key:
  the epilogue is the ANNOUNCER OUTRO line (empty news_close_brief
  routes there; outro already knows the produced ending) -- zero new
  passes; disclosure lives in the printed layer (news_used + bank-aware
  HUD label replacing hardcoded "NEWS SEED" + unconditional credits
  line); anachronism defense is prompt-side + lexicon only.

- Local read-only fan-out QA (operator request) on the two shipped chunks:
  Antigravity returned NO blockers/majors; 2 verified MINORs FIXED same
  session (stopword bypass in produced-story cast grounding; off-by-one
  dropping the closing excerpt window at exact cap boundary -- also fixed
  in the older reflection builder it was copied from). Codex CLI not on
  system PATH from this session; operator pasting the brief into Codex
  manually -- its report landed at docs/2026-07-09-source-route-qa/
  local_fanout/codex_review_manual.md and was judged SAME SESSION: one
  real BLOCKER-class bookkeeping catch (the 321bcc9c/40535ddc SHA mixup,
  corrected in these docs); all its code checks CLEARED the current tree.
  Fan-out verdict overall: architecture sound, 3 real minors total, all
  fixed and pushed.

- NEW OPERATOR FEATURE (late): post-composition INTRO REWRITE -- once the
  story is done, rewrite the announcer intro from the PRODUCED first
  scene + cast, spoiler-safe by input starvation (scene-1 rows only).
  Spec: docs/2026-07-09-original-radio/INTRO_REWRITE_SPEC.md (shape A =
  derive ProducedOpenBrief -> existing safe-open composer, anchor lean;
  shape B = new rewrite seam). Runs BEFORE outro compose so the
  tone-echo reads the final intro. Joins kibitz r2 scope.

Current step: original_radio campaign -- R1 CONVERGED (2 passes,
~$0.26 total). Next: /kibitz r2 (coding plan) on
docs/2026-07-09-original-radio/ARCHITECTURE_V4.md + INTRO_REWRITE_SPEC.md,
then r3 wiring, r4 convergence, then build: tests first, SAME-COMMIT
registry set SHIPPING runnable:true, pre-ship gates = suite + Bug Bible
+ mocked pipeline + live 30w smoke + operator eyeball.
Commits: 62962121, (40535ddc co-authored), 321bcc9c, 5a09984c -- all pushed.
