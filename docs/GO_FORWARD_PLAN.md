# OTR Go-Forward Plan

**ACTIVE STEP (2026-07-10 evening): 720-word STORY-ENGINE BAKE-OFF RUNWAY.**
Plan of record: `docs/2026-07-10-fable2-720-bakeoff-runway.md` (kibitz'd before
C4 coding). Chunks C1-C7: P1.1 ownership/revision merge contract -> P1.3
text_for_tts (science_news byte-parity fixture FIRST) -> P1.4 cue manifest +
canonical workflow delta (links 241-243 out, 280-283 in, code+JSON same commit)
-> P1.5 S2 full loop (3-pitch/P4/P5/keep-better judge, 120-900w) -> 350w + 720w
live rolls -> P2.2 caption/credits alias + HuMo guard -> bake-off: fable2 vs
original_radio vs science_news vs a GPT-authored from-scratch fable2-pipeline
pack (operator-ratified contender D), one pinned news story, 720w each, blinded
operator judging. 720w needs NO act-chunking (inside S2's 120-900 band).
Precondition SHIPPED @ `47bf50f2`: r2-QA P0 fold -- FreezePolicy
content_owned_readonly freeze boundary (fable2 skips reviewer/doctor/5B/A3/5C/
escalation/A2/D3-mutation/Phase-7-normalization; read-only proof verification;
tagged-unresolvable bank TERMINAL; freeze_capability_receipt w/ sha256 +
content_mutations=0), live-root veto after reviewer save-rebind (P0.3), 19-field
absent-key merge ownership (P0.2). Suite 7463/31/1 + Bug Bible green. Governing
spec: `docs/2026-07-10-fable2-s2-QA-ANALYSIS-r2.md`. Deferred: P2.1
retire-doctor-skip, P2.3 soak cohorts, >900w act-chunk, cloud pins.

**scifi_fable2 S1b SHIPPED + FIRST GREEN EPISODE (2026-07-10 midday, this
coder window; slot released).** Spine live per doc s13 + the s13.5
deviation record: runner @ `a24b75c4`, 25-roll live-smoke hardening +
kibitz r2/r3/r4 folded @ `ff4c226d` (+ follow-up chunk this session) --
**"Einstein's Echo" published to obs (RESULT SUCCESS 570s, 56.9 MB,
Test-Path'd), canonical JSON NO-DIFF + OTR_WorkflowValidator OK
recorded.** Key S1b deltas (details: doc s13.5 +
`docs/2026-07-10-fable2-s1b-smoke-hardening.md`): P2c read-split
(10th seam `fable2_news_read_system`), parser delete-only decoration
normalization, 4-rung P3 ladder + few-shot + micro-episode line cap,
one-word cast labels, sentinel announcer char_id (freeze-cascade
mutator dodge -- ROOT MUTATOR STILL UNIDENTIFIED, open question in the
hardening doc), source-as-legality-authority subset gates, delete-only
dossier entity filtering, nested-path clamp in _otr_structured_call.
Next fable2 step = **S2: full loop** (P1 three-pitch + P2a select + P4
critic + P5 revision + keep-better-draft; 350w live smoke; doc s13 S2
test set). **BLOCKER RESOLVED (external QA fold @ af378aad):** real
chain was doctor-skip + stale-disk merge resurrection, NOT the 5C
reroll -- ownership-aware merge + skip contract + 5B/5C lane capability
gate shipped; **LTX media path GREEN: "The Butterfly's Gambit" in obs
(1787s, character lane ltx_audio_in + stills)**. S2 must fold the QA
runway items (proof-provenance/text_for_tts, inter-scene music wiring,
caption/credits sentinel alias, HuMo stale guard, per-scene band
allocation): `docs/2026-07-10-fable2-s1b-QA-ANALYSIS.md` (file:line
pins) + the brief `docs/2026-07-10-fable2-s1b-QA-PROBLEM-STATEMENT.md`.

**Updated:** 2026-07-10 midday -- HEAD `ff4c226d` (+ this session's
follow-up commit)
**Branch:** `v2.0-alpha`
**Status:** original_radio LIVE SMOKE GREEN 2026-07-10 ("Page in the
Tempest" published to obs). Remaining gate = OPERATOR EYEBALL only ->
then source-bank sweep -> portability.

**PLATFORM-PORTABILITY BUILD: S0-S6 SHIPPED overnight 2026-07-10 (this
coder window; claim RELEASED -- ratify decisions + tier smokes remain).**
Spec docs/2026-07-09-platform-portability-final.md executed end to end:

- S0 `c32d7cbe`+`390e5015`: 7 committed-state defects (loader guard,
  vram-node accel helpers, _detect_host mps/vendor + lying-CUDA fix,
  dispatcher gen_fn raise, bark honest row + cpu_floor kokoro, HuMo
  script/label alignment, indextts2 installer pair now SHIPPED).
- S1 `af60c09e`: LLMRuntimePolicy end to end (FA2 probe + tag auto-quant
  DELETED; lane backstop; policy-keyed LLM cache; GGUF artifact table;
  n_ctx downgrade + preflight-tolerance now raises).
- S2 `dd79f5b2`: profile schema v2 (+linux/+mps/+gpu_vendor/llm/video/
  image/audio/render/preflight) + ALL 8 profiles migrated + widget_mapping
  v2 (writer out of exempt_node_types; 20 creative names exempt).
- S3 `e1f692e7`: registry CAPABILITIES v2 ATOMIC (device_backends
  supersedes cpu_ok; vendor pins table-visible; humo fp8 flags). FABLE
  GATE ran pre-commit: NO MUST-FIX.
- S4 `2e31efb9`: policy v2 consumers -- REAL host_caps at every adapter
  boundary (was {} everywhere), adapter-level image assert_usable (was
  never called), CastLock meta.voice_device stamp, kokoro/musicgen/
  chatterbox waterfalls DELETED, frame budget static + MotionBudgetError
  (never resizes).
- S5 `6fb5cc29`/`b60e9df3`/`b8bbadd3`/`6f0131da`: widgets (writer 28->34
  slots 28-33; director 14/8; castlock 6) ATOMIC with canonical + mapping;
  gate_in link 279 gates the WRITER; scripts/build_variants.py (+--check,
  ratify_before_emit refusal); semantic master_hash ASSERTED by the
  validator; otr_api soft-skip RETIRED (hard fail).
- S6 `2bd212ec`: profiles otr_nv40_12gb / otr_amd16_rocm / otr_amd8_rocm
  / otr_mac_mps (draft) added; EMITTED 5 variants + launch recipes
  (8gb_lite, cpu_floor, nv40, amd16, amd8; --check green). REFUSED
  pending ratification: 16gb_full, cloud_all, otr_mac_mps.
- SMOKE GATE: 30w canonical live smoke on the FULL rebuilt stack =
  RESULT SUCCESS in 543s (baseline 548s) -- published
  `otr\obs\signal_lost_whispers_of_deception_20260710_043218_..._final.mp4`
  (57.2 MB). Suite 7251/31/1 + Bug Bible 17/7/3 green at ship.

RATIFIED (operator, 2026-07-10 morning, all four): (1) 16gb_full
REGENERATED from canonical (viz lanes + z_image_turbo) -- nv50 identity
variant EMITTED; (2) cloud tier RENAMED cloud_all -> otr_cloud_lanes;
OpenRouter slot pins = NEXT SESSION (the one remaining ratify gate --
cloud variant still refused); (3) mac ceiling 10.0 ratified -- otr_mac_mps
draft variant EMITTED; (4) AMD/Mac ship draft-UNVERIFIED behind the
acceptance gates in their launch recipes. Post-ratification emission:
7 variants + recipes committed (--check green).
NEXT SMOKES: cpu tier on this box (--cpu; NOTE cpu_floor inherits
canonical z_image_turbo images -- cuda-only, so the cpu smoke needs the
google image lane + OTR_GOOGLE_API_KEY or pre-staged stills; surfaced by
the new adapter-level gate, decide before running), then the nv50
identity re-soak after ratify (1). Bug Bible candidates queued:
lying-CUDA is_available pattern; registry-row-vs-generation-path honesty;
waterfall-deletion class. Longer wan-beat lanes now FAIL LOUD instead of
shrinking (dial: frame_count widget / OTR_VIDEO_BUDGET_MARGIN).

**Shipped 2026-07-09 evening (both pushed):**

- Closing-seam routing fix (QA F1): coda + announcer seams now
  pack-route per bank; PD/Shakespeare coda seams re-authored to the bridge
  contract; title pass reads banks.json `title_form_label`. 30 new tests.
  CODE COMMIT = `40535ddc` (the operator's Codex loop committed the
  in-flight tree, bundled with its dia hardening); `321bcc9c` carries
  only docs. SHA corrected per the codex fan-out catch.
- Produced-story meta split (`5a09984c`): K.5.6 `run_produced_story_summary`
  stamps `meta["produced_story"]` (logline/subject of the ACTUAL episode);
  credits/HUD/treatment premise + music last-ditch mood repointed. Operator
  ruling: pre-gen source digest and post-gen story brief NEVER share a name.
- QA synthesis (local, gitignored dir): docs/2026-07-09-source-route-qa/.
- original_radio ARCHITECTURE_V1 + R1 roundtable artifacts (local):
  docs/2026-07-09-original-radio/.

This file is for short-term coordination only. Longer runway lives in
`ROADMAP.md`; old sprint logs belong in `docs/GO_FORWARD_ARCHIVE.md`,
`docs/HANDOFF_LOG.md`, or dated docs.

## Current Status

Recent green code chunks on `v2.0-alpha`:

- Media archive seed deck is green and pushed.
- Visual-style bleed guard is green and pushed.
- `recur_frac` is now the concise `recursive fractal light field` pack; LTX
  audio-in talking prompts keep a face/mouth/lip-sync cue without cartoon
  wording. Commit: `ac919d99`.
- Model-slot audit + pre-smoke contract inspection is green locally:
  `docs/2026-07-09-model-slot-audit.md` now records the canonical kept local
  matrix, retired/non-invocable ids, and the requested Chatterbox/Dia/Qwen/Wan
  plus Comfy Cloud still/video readiness queue. `tests/test_model_slot_audit.py`
  pins the canonical contracts and the newly inspected candidate surfaces.
- All-Chatterbox 30-word OBS live smoke completed from a real `science_news`
  MIT News source. Output landed in
  `C:\Users\jeffr\Documents\ComfyUI\output\otr\obs\signal_lost_the_allocation_key_20260709_162019_silent_procgen_blended_captioned_with_credits_final.mp4`.
  The smoke exposed a real no-reuse gap: distinct logical Chatterbox ids could
  share the same underlying WAV. The fix blocks same-asset/provider collisions
  when `allow_voice_reuse=False`.
- Dia is now in the announcer on-deck lane: `dia` serves `announcer_voice`,
  `announcer_dia_v1` is in the profile resolver, the announcer dropdown exposes
  it explicitly, and the default voice bank has a British-leaning male/female
  preferred announcer pair with seeded 50/50 selection and no-reuse separation
  from character refs.

Current runnable source banks in `nodes/story_packs/banks.json`:

- `science_news`
- `media_archive`
- `public_domain_story`
- `shakespeare`

Known non-runnable bank:

- `custom_source_bank` stays listed but intentionally fails loud.

Unrelated local file present before this plan update and left untouched:

- `docs/2026-07-08-source-banks-v2-plan.md`

## Next Action

### 0. ACTIVE: `original_radio` campaign (operator 2026-07-09)

Build spine: **ARCHITECTURE_V4.md** (R1 converged x2, ~$0.26). Operator
locks: Hitchcock epilogue AS the announcer outro; printed-layer provenance;
no era frame; RUNNABLE ON BUILD (no flips/switches/fallbacks, hard fails
accepted); north star = max story complexity / max code elegance.
R2 DONE (2026-07-09 late): /kibitz r2 -- Codex gpt-5.5 (auto) + Antigravity
gemini-3.5-pro (operator-pasted manual after one auto timeout) + Claude
anchor; 3-way convergence, no architecture change. Coding plan =
**docs/2026-07-09-original-radio/R2_CODING_PLAN.md** (mirror:
kibitz-runs/2026-07-09-original-radio/r2/final.md). Key r2 locks:
intro-rewrite SHAPE A (derive ProducedOpenBrief -> existing safe-open
composer; derive helper lives in _otr_story_brief.py); runtime dispatch on
BANK SHAPE (empty fetcher+interpreter), NEVER pipe.requires_source_contract
(routing law 88-91); original_multi_pass ships executable:true (sweep 4b
elif demands it -- no sweep patch); build_original_briefs returns the
validate_interpreter_result contract with news_close_brief="";
story_rules/original_radio.json is load-bearing (outro rules resolve);
provenance data-driven (bank defaults hud_origin_label + credits_source_line;
news_used gets final title + origin_label via builder signature); KEEP
news_coda_no_brief flag; keep-intro failure posture upheld (agy hard-fail
logged as operator dial); no workflow JSON diff expected (registry-derived
dropdown; source_bank already headless-whitelisted).
R3 (wiring) + R4 (convergence) DONE same night (Codex + anchor; agy auto
lane timed out 3x -- its r2 manual review is on the record). Artifacts:
R3_WIRING_DELTAS.md + kibitz-runs/.../r4/final.md (r4 pins P1-P8).
BUILD SHIPPED (operator away, autonomy directive):
- CHUNK A `181506e8`: intro rewrite ALL banks (derive_produced_open_brief
  in _otr_story_brief + writer I.4.9 block, keep-intro posture, flags
  read-extend) + J.5 title-regen root-cause fix (ledger-assembled).
- CHUNK B `604ccdd3`: the original_radio SAME-COMMIT set, runnable:true
  -- registry rows (pipeline executable:true; spark deck registered as a
  pack SIDECAR in routing), 13-stage pack w/ Hitchcock outro seam,
  story_rules, _otr_original_radio.py (concept->select->brief +
  whole-script QA w/ coalesced outro repair), writer bank-shape dispatch
  (runnable+empty-empty), interpreter-shaped adapter + dual source_meta
  restamp, provenance surfaces (news_used labels/final title, HUD origin
  label x2, credits_source_line stamp+render).
Both pushed; suite 7136 passed/31 skipped/1 xfailed + Bug Bible 16/7/3
green; AST/BOM/0-byte verify clean. NO workflow JSON diff (source_bank
menu derives from the registry).
SMOKE GATE GREEN (2026-07-10 early): live 30w original_radio OBS smoke
PUBLISHED -- `otr\obs\signal_lost_page_in_the_tempest_20260710_010652_
silent_procgen_blended_captioned_with_credits_final.mp4` (48 MB, RESULT
SUCCESS, 548s; disclosure line explicit machine-generated; QA meta
clean_after_discard w/ discards stamped). Validator record:
OTR_WorkflowValidator OK in the green run -- 23 nodes / 55 links /
widget_vector_drift=0; the LANE produced NO workflow diff (registry-
derived dropdown confirmed); the only canonical diff since 604ccdd3 is
the deliberate creative-slot fix (d526c8b7). It took SIX live-smoke
hardening commits to get here -- 5 judge/ladder root fixes + 1 workflow
config fix (7f459e21, 75173fc4, a61ab2ed, 6fdf3f6e, d526c8b7, 1c735c2d;
details in HANDOFF_LOG 2026-07-10). Suite 7151/31/1 + Bug Bible 17/7/3
green; bible +BUG-11.26.
REMAINING GATE: OPERATOR EYEBALL of the published episode (consider the
3-episode batch from V2 s7). Content items FOR the eyeball, not gates:
(a) outro/dialogue name drift (outro names 'Harold', cast-external;
run-2 had 'Eliza'), (b) stage directions leaked into a spoken line one
run ('(smashes a nearby console)'), (c) script lines arrive quote-
wrapped, (d) nemo premise drifted sci-fi in 2 of 4 rolls (period-legit
Dimension-X flavor; timeless-rule tension). Then the source-bank
end-to-end sweep (section 3 below).
Queued design item for the arc: demote `meta["news"]` to provenance-only
+ distinct-name migration (operator: "we can't just throw around meta").

QUEUED CAMPAIGN (2026-07-10, analysis DONE + VERIFIED, code NOT started):
**lean-mean rip** (de-slop / ship-shape). Plan =
`docs/2026-07-10-lean-mean-rip-final.md` @ `b9219478` (committed; the dated
folder holds local round artifacts). r1-r4 converged same night (3 mechanical
audits -> 3-Fable architect fan-out -> grounding -> Fable gate: CONVERGED after
7 must-fixes, folded), THEN a 2-agent code-readiness sweep re-verified every
claim against the REAL post-portability HEAD `20185542` (48 confirmed; 9
amendments folded, incl. the quantified W5 obligation: remove node-1 inputs[9]
+ renumber link 279 dst_slot 34->33). ~32-33k LOC deletion in waves W0-W8 +
consolidations C1-C7 + giants split SW1-SW4. Portability precondition is now
MET (S0-S6 shipped + ratified). D-1..D-6 OPERATOR RATIFIED 2026-07-10 (plan @
`e569880d`), D-2 codicil: rip RTXUpscale now; a FUTURE system-agnostic
multi-GPU upscale campaign rebuilds against the `upscale_stage` profile
reservation, honest-switch law (widgets land WITH a working engine). CLEARED
TO EXECUTE post-portability-settle; execution-time gates = R-4 seam re-survey
+ R-7 re-grep only.

NEXT-CODER-SESSION ITEM (operator 2026-07-10): **ENGINE_MATRIX.md** --
device_backends must be END-USER VISIBLE, not hidden in registry rows.
Spec: extend scripts/build_variants.py with an emit step rendering
docs/ENGINE_MATRIX.md from the THREE registries' CAPABILITIES v2 rows
(engine id, role, device_backends, requires_vendor, needs_fp8_te/fp4,
practical_without_gpu, sidecar_conditional) + a reason-code legend
(vendor pin / fp8 / NVML gate / cu128 / no-install-path). GENERATED,
never hand-edited; `--check` regenerates + diffs it like the variants;
README links it. Tests: pin emitted file == live registries (rides
--check). NO workflow JSON / widget changes -- zero positional risk.
Operator chose matrix-doc ONLY (recipes/validator/tooltip surfaces
declined for now). Small chunk, lands BEFORE the lean-mean campaign.

### 1. Model-Slot Audit And End-To-End Smokes (live smokes PARKED 2026-07-09)

Inventory every local model/engine exposed through the production slots:

- music
- audio/TTS
- still image
- video

For each candidate, document and test:

- required inputs and produced outputs
- supported slot/family/role
- required model files and expected VRAM class
- canonical workflow compatibility
- whether it can complete a tiny end-to-end smoke from `workflows/otr_canonical.json`

Decision rule:

- If a model fits the slot contract and finishes an end-to-end smoke, keep it
  in the tested path.
- If it cannot fit, cannot run, silently downgrades, OOMs outside its claimed
  tier, or produces the wrong artifact shape, remove it from the tested path or
  mark it non-invocable. No silent fallback.

Deliverables:

- Done: compact tested/retired model matrix.
- Done: focused tests that prove unsupported engines fail loud.
- Done: pre-live-smoke input/output contract inspection for `chatterbox`, `dia`,
  `qwen_image`, `wan_ti2v`, `wan_i2v`, `cloud_nano_banana_2`,
  `cloud_seedream_2`, `cloud_krea_2_turbo`, `cloud_luma_photon_flash`,
  `cloud_vidu_q2_pro_fast_720p`, and its SFX sibling.
- Done: canonical offline API dry-run from `workflows/otr_canonical.json`.
- Remaining: live sidecar/GPU/provider smokes after the selective headless reset
  and any required auth/asset preflight.

Recommended live-smoke order:

1. Done: full all-Chatterbox 30-word OBS smoke.
2. Done: Dia cast/announcer contract is green in tests. Remaining: Dia one
   character line plus Dia announcer line live sidecar smoke, then a full
   all-Dia smoke if the sidecar path is healthy.
3. Comfy Cloud stills: Luma Photon Flash, Krea 2 Turbo, Seedream 2, Nano Banana
   2.
4. Cheap Comfy Cloud video: Vidu Q2 Pro Fast 720p.
5. Local heavy visuals: Wan TI2V first; Qwen Image after CLIP/VAE preflight is
   strengthened; Wan I2V after the 5B Wan path unless the 14B target is needed.

### 2. `original_radio` Source Bank

Design and implement a no-source original-drama lane where the LLM creates a
random original old-time-radio premise, cast, outline, and filled ledger.

Hard requirements:

- No news/source attribution.
- No franchise or modern IP wording.
- No guns, knives, smoking, or source-seed leakage.
- Small cast, clear radio conflict, coherent ending.
- Fail loud after bounded repair attempts if the LLM cannot produce a valid
  brief/ledger.

Architecture direction:

- Add `original_radio` as its own source bank, not as a fake RSS/source lane.
- Add an original multi-pass runner only when its tests exist.
- Let the original branch generate compatibility `meta["news"]` fields for
  current downstream code, while stamping honest provenance under sidecars.
- Deep-think the creative route before coding; if the shape is still ambiguous,
  use the repo roundtable/Fable rules for a grounded design pass.

Dynamic visual-style direction:

- Explore an improvising visual pack for original episodes.
- It must validate through the same visual-style schema, never write ad-hoc
  disk packs during a render, and must not leak the style name into story
  premise, title, narration, or dialogue.

### 3. Source-Bank End-To-End Sweep

After the model-slot audit and `original_radio` work, prove every runnable bank
works end to end or fix it before moving on:

- `science_news`
- `media_archive`
- `public_domain_story`
- `shakespeare`
- `original_radio` once runnable

Use 30-word/random smokes first. Look specifically for:

- source-bank drift
- story/source leakage
- title or premise pollution from visual style
- weak cast separation
- stale sci-fi wording in non-sci-fi banks
- bad coda/source-note behavior
- forbidden content
- broken still/video/audio routing

Stop when each lane is good enough to proceed; do not polish forever.

### 4. Portability

Only after the above is green, continue portability work:

- no-GPU / procedural
- all-cloud
- RTX 8 GB, 12-16 GB, and 24 GB+
- Mac
- AMD where practical
- RunPod/cloud GPU

Canonical workflow remains `workflows/otr_canonical.json`; exported workflows
must be generated from canonical, not hand-maintained.

## Last Validation

Post live-smoke hardening (2026-07-10):

```text
pytest -q -p no:cacheprovider

7151 passed, 31 skipped, 1 xfailed, 5 warnings
```

Bug Bible (bible now 156 entries, +BUG-11.26):

```text
cd C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide
pytest -q -p no:cacheprovider tests\bug_bible_regression.py

17 passed, 7 skipped, 3 xfailed
```

Focused voice/cast subset:

```text
pytest -q -p no:cacheprovider tests\test_model_slot_audit.py tests\test_voice_bank.py tests\test_audio_engine_adapters.py tests\test_engine_profiles.py tests\test_tts_engine_sidecars.py tests\test_announcer_voice.py

114 passed
```

Focused contract subset:

```text
pytest -q -p no:cacheprovider tests\test_model_slot_audit.py tests\test_cloud_image_adapters.py tests\test_cloud_video_adapters.py

83 passed
```

No workflow JSON edit was needed for the Dia announcer on-deck chunk: the
canonical workflow already carries the `announcer_voice_engine` widget and the
announcer node engine dropdown derives from the profile/registry menu. No live
Dia sidecar smoke was run in this chunk.

## Standing Rules

- `workflows/otr_canonical.json` is the canonical workflow.
- Any node/widget/wiring change must update that workflow in the same change.
- Every headless/API smoke must load the canonical workflow.
- Reset selectively before headless runs; never blanket-kill Python.
- Render assets go straight to `otr\episodes\<ep>\`, final to `otr\obs\`.
- Do not revert unrelated/user changes.
- Fix root causes, not shims.
- No silent fallback.
- JSON owns content/config.
- Python owns validation/routing/execution.
- Commit and push every green chunk to `origin/v2.0-alpha`.

## Pointers

- `ROADMAP.md`
- `AGENTS.md`
- `CLAUDE.md`
- `docs/BUG_LOG.md`
- `docs/GO_FORWARD_ARCHIVE.md`
- `docs/2026-07-08-source-banks-v2-plan.md`
- `docs/google_tts_ideas.md`
- `docs/multimodal-story-schema/MEDIA_ARCHIVE_QA_HANDOFF.md`
- `workflows/otr_canonical.json`
