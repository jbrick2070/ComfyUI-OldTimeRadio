# OTR GO-FORWARD PLAN -- SINGLE SOURCE OF TRUTH (what's LEFT)

> Last updated 2026-07-02 late evening | branch v2.0-alpha | prod/main + tags operator-GATED.
> NIGHT QUEUE SCORED (see PROOF9_VERDICT.md in docs/2026-07-02-canonical-ia2v):
> proof9c FAILED (VRAM ceiling ops breach at 768x416 -- desktop squatter; no clips);
> 120w soak COMPLETED + scored as the interim S4x verdict (chars ~3x relative
> articulation lift; the 2.0 bar is CANVAS-SPECIFIC -- metric NOT scale-invariant);
> 30w soak was still rendering at wrap (~55 min in) -- QA IT FIRST next window
> (final in otr\obs + 6/6 clips + no breach), then proof9d clean 832x448 re-run
> (kill the desktop squatter; baseline <=~2.5GB) for the final S4x GO/NO-GO.
>
> **LEAN + FORWARD-ONLY.** This doc holds the PLAN: current step, forward order, open items,
> hard rules, and POINTERS to sprint specs. It does NOT record what got done -- that lives in
> `docs/HANDOFF_LOG.md` (recent sessions) + `docs/GO_FORWARD_ARCHIVE.md` (deep history). If this
> doc starts growing a change-log of shipped work or inlining sprint detail, TRIM it: move history
> to the log, replace detail with a pointer. Keep it short.

---

## 1. CURRENT STEP

**NEXT CODE = S1 STILLS LANE** (canonicalize_image + recraft/flux_pro/nano_banana_2
adapters + portrait-mint gates incl. `portrait_mint_3d` + the generic profile->schema
CONFORMANCE TEST). Order for the next window: (1) QA the 30w soak, (2) proof9d clean
832x448 re-run + score, (3) S1. S5 GPU exit gate (silent-vs-audio-in VRAM/time A/B on
the first live clip) rides whichever render window comes first.

**OPERATOR DIRECTIVE 2026-07-02 evening (SHIPPED @ cc349c1d): NO hidden cloud enable
switch.** `OTR_ENABLE_COMFY_CLOUD_MEDIA` is REMOVED (same clean break as OpenRouter C6):
the dropdown pick IS the enable; missing credentials fail LOUD at invoke-time auth
(naming OTR_COMFY_API_KEY / logged-in hidden inputs); budget unset = $10 DEFAULT_BUDGET_USD
safety cap (explicit 0 = deliberate spend-off). This SUPERSEDES every "flag OFF/ON" test/
acceptance line in pass04 sec 8 -- S1 builds on the no-flag reality. The S0 live smokes
now need only auth (+ OTR_RUN_CLOUD_SMOKE=1, the paid-run gate for the script).

**S5 SHIPPED @ fb23d82d (2026-07-02 late evening):** silent two-stage HQ recipe in
eng_ltx_video (OTR_LTX_VIDEO_RECIPE auto|hq_two_stage|single_pass; dev-unet auto default;
init still required fail-LOUD; single_pass frozen byte-identical; 2 LTX rows, NO
ltx_lowvram). REMAINING = the GPU A/B exit gate above.

**CLOUD ENGINE LANES S0 (context; operator "build" 2026-07-02).** Build doc =
`docs/2026-07-02-cloud-engines/roundtable/pass04_plan.md` @ 29b11e77 (4-round roundtable converged
+ operator amendment: audio reactivity DEFAULT-ON all video roles, mute = audited opt-down).
S0 PROGRESS (2026-07-02): chunks 1-3 SHIPPED + PUSHED, suite 5988/0 + Bug Bible green each chunk:
- c1 @ 5a79a926 `nodes/_otr_shared/cloud_media_backend.py` (error taxonomy, auth broker, prompt_id
  session table w/ leak sweep, budget state machine, provider semaphores, billing JSONL, mute-opt-
  down knob) + 27 tests.
- c2 @ 7d8c490f `cloud_media_cache.py` (RequestCacheKey pre-submit-only, atomic store, quarantine,
  per-key locks) + `cloud_media_canonical.py` (PartnerResult/CanonicalAsset + S1/S2/S3 contracts)
  + 14 tests.
- c3 @ 44f36fdc+f3a97ea6 `scripts/otr_pin_partner_nodes.py` + CHECKED-IN
  `nodes/_otr_shared/partner_nodes.yaml` -- 11/11 rows pinned from the LIVE core
  (`C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI`, override `OTR_COMFY_CORE_ROOT`); all async
  EXECUTE_NORMALIZED_ASYNC, all expose both auth hidden inputs; drift test = SUBPROCESS `--check`
  (in-process comfy import corrupts pytest teardown -- do not inline it) + 6 tests.
- c4 @ f9eed360 `nodes/_otr_shared/cloud_media_invoke.py` -- `invoke_partner_node(node_key,
  inputs, *, timeout_s, estimated_usd) -> PartnerResult` (backend-owned loop thread; session from
  prompt context / `bind_prompt_id()` headless; hidden-auth per row; 5s-tick watchdog w/ interrupt
  cancel + 20s ProgressBar heartbeat; streamed temp downloads; release-vs-bill settlement) + 23
  tests + GATED smoke harness `scripts/otr_cloud_s0_smoke.py` (leg1 recraft auth proof, leg2 Kling
  avatar conditioned by `tests/fixtures/baseline_v1.5.wav`).
S0 REMAINING: ONLY the live smoke RUNS -- operator must set `OTR_RUN_CLOUD_SMOKE=1` and
`OTR_COMFY_API_KEY` (budget optional -- unset = the $10 default cap; the enable flag is
REMOVED per the 2026-07-02 directive above), then
`python scripts/otr_cloud_s0_smoke.py --leg 1` / `--leg 2`. NO existing render-path changes in S0;
workflow JSON untouched until S4 (operator-gated). NEXT CODE = S1 (stills lane: canonicalize_image
+ recraft/flux/nano adapters + portrait-mint gates).

**CODE BATON 2026-07-02 (later):** back with the LTX-fixes window; cloud S0 remainder BUILT there
(c4 above). c5 (docs window, 2026-07-02 pm): roster expanded to 14 pinned rows
(+cloud_ideogram_v4, +cloud_seedream_2, +cloud_elevenlabs_voice_selector AUX -- the TTS row
requires its ELEVENLABS_VOICE output) + versioned pricing stamp `docs/2026-07-02-cloud-engines/
PRICING.md` (211cr=$1; voice ~$1.10/ep FLAT across tiers; per-line Kling lipsync ~$0.25-1.00 = the
dominant cost) + prompt/param profiles doc (see below).

**VIDEO TEAM warning (2026-07-02 pm, codex-verified) -- RESOLVED at a9440980 (same day, night):**
the audited eng_cloud_video.py draft emitted unpinned kwargs; the SHIPPED commit folds every catch:
cloud_wan_i2v sends EXACTLY `first_frame`/`model`/`prompt_extend`/`seed`/`watermark` (no prompt;
audio not sent -- mute row); cloud_seedance_2 is an HONEST DARK ROW (raises loud) until the S1
V3-expansion pin names its dynamic inputs; the kling pair's kwargs are all pinned; per-row
conformance is test-locked in tests/test_cloud_video_adapters.py. STILL OWED AT S1: the generic
profile->schema CONFORMANCE TEST (every emitted kwarg declared in the yaml) as the permanent guard.
`canonicalize_video` is REAL now (strip + post-strip proof) -- see HANDOFF_LOG.

**S3 hold SUPERSEDED (operator, 2026-07-02 evening, voice: "code the cloud video plan... tagged to
go forward"):** S3 CORE shipped @ a9440980 -- 4 rows registered dark + fail-closed, EMPTY
default_roles (selectable picks only; never automatic). S3 FULL (reactive auto-defaults + ShotLock
audit stamps + fallback chains + seedance/wan V3 expansion + live provider proof) still rides the
original order: live smokes (operator env) -> S1 STILLS -> S3 full. One window at a time,
coordinated HERE.

QUEUED BEHIND cloud S1+S3: creative formats F1 Living Evidence Board + F2
Tin-Toy Theatre -- plan `docs/2026-07-02-creative-formats/CREATIVE_FORMATS_PLAN.md`
(kibitz-hardened; ideation record in docs/2026-07-02-cloud-engines/roundtable/ideas_synthesis.md).

**Previously current (unchanged, operator-gated, NOT code):** All-engines x all-slots soak RUN +
talking-radio (C) morning-eyeball GO/NO-GO -- see below.

**All-engines x all-slots: CODE SHIPPED (slot-audit C0-C5) -- remaining = the live-GPU soak RUN.**
Boot headless ComfyUI, load `otr_scifi_16gb_full.json`, apply the all-role profile
(`slot_matrix.build_all_role_profile` -- 3 roles post rip-sfx-broll), render a leg per engine, run
`content_oracle.check_manifest` on the per-beat manifest. GPU-operator-gated (not code).
Sprint spec: `docs/2026-06-30-slot-audit/SPRINT_PLAN.md`. Acceptance met in code; the RUN proves
it empirically. Accelerator = S-F visual smoke fixture (shipped).

**rip-sfx-broll SHIPPED 2026-07-01 (see HANDOFF_LOG.md):** the role model is now
speaker = {character, announcer, music_open, music_close, music_inter} and
video = {announcer_visual, music_visual, character_video}; NO FALLBACKS -- an unmapped role or an
old `speaker_role:"sfx"` ledger FAILS LOUD everywhere. Build plan + kibitz judgments:
`docs/2026-07-01-rip-sfx-broll/`. Old on-disk episode ledgers predating the rip must be
regenerated before reuse.

**Opt-in feature SHIPPED (not part of the forward order):** brief-driven HuMo radio-host
+ `OTR_LTX_RADIO_FACE` A/B (default OFF, byte-identical). See HANDOFF_LOG.md.

**TALKING-RADIO: RECIPE_IA2V is the dev-default lane; S4/S4b/S4c SHIPPED (2026-07-02).**
The canonical comfy.org IA2V transplant reversed the C NO-GO (see HANDOFF_LOG + PROOF7_VERDICT.md
in docs/2026-07-02-canonical-ia2v). Shipped since: talking prompt register, S4 portrait init, S4b
face-forward portrait mint, S4c radio-face DEFAULT-ON for ia2v bookends (env A/B = single-pass
only). VERDICT (2026-07-02 late evening, PROOF9_VERDICT.md): PROVISIONAL PASS on direction --
the 120w soak (S4 fired, log-proven) lifted characters ~3x relative to the announcer anchor,
but the 2.0 bar is CANVAS-SPECIFIC; final GO/NO-GO = proof9d clean 832x448 re-run. **S5 SHIPPED
@ fb23d82d** (silent HQ two-stage in eng_ltx_video; GPU A/B exit gate owed). Story-writer
fixes are PARKED in the transplant repo (UpstreamStoryLab GO_FORWARD "DEFERRED STORY-LLM FIXES")
-- NO production story-LLM changes until the refactor. Source-bank visual-style transplant stays
OUT (research mode; docs in `ComfyUI-OTR-UpstreamStoryLab\docs`).

---

## 1A. OPEN ITEMS (post-soak, priority order -- detail in the sprint specs, do not inline here)

Coverage-soak sprint spec (kibitz r1-r4 converged): `docs/2026-06-29-coverage-soak/SPRINT_PLAN.md`.
The load-bearing OPEN items (7 of 11 sub-items already shipped -- see HANDOFF_LOG.md):

- **E1 -- no-fallback scaffolding migration (NOT STARTED, load-bearing).** `make_fallback_of(` still
  live (`render_driver.py:1800/2288`); `FLOOR_NAMES`/`UNIVERSAL_FLOOR`/`SYNTH_FALLBACKS`/
  `EXPECTED_OOM_TRAIL` still used; `eng_character_3d.py` refs the chain. No-fallback is a standing
  operator directive.
- **E2 -- deprecate `allow_auto_fallback` in place (NOT STARTED).** `otr_video_director.py:228` still a
  plain BOOLEAN passed through (:354); force false + relabel.
- **E3-doc -- edit THIS doc only.** station_card + abstract retired (C0). `still_motion` is NOT retired
  (it is UNIVERSAL_FLOOR + mesh_stage's fallback target) -- do not unregister it.
- **E4 -- "which model" dropdown spell-out (DEFERRED, low priority).** Audio-reactive + VRAM-tier
  suffixes shipped; spelling out LTX/Wan/image recipe per label needs a no-drift design decision.
- **S-C C1 -- shared `audio_motion_profile` (NOT STARTED).** Per-beat rms/peak/onset/silence/brightness/
  dynamic-range/speech-vs-music/duration driving every engine. C2 (per-engine consumers + HuMo
  phrase-chunking, the real clip-underrun fix) deferred per the plan.
- **Writer (non-blocking):** long-target freeze-gate flakiness -- target_words=800 tripped
  BUG-LOCAL-276 on 2/3 attempts + under-delivered length. Writer-side look. See
  `docs/2026-07-01-overnight/MORNING_REPORT.md`.
- **Force-map optimization (non-blocking):** `OTR_FORCE_ENGINE_MAP=*=<engine>` still mints Flux stills
  for the pre-override plan; short-circuit still-gen when a role is forced to an `accepts_still=False`
  engine.

Invariants for all: single resident heavy <= 14.5 GB; audio byte-identical; no-fallback (hard-fail
LOUD); UTF-8 no BOM; SFW; workflow-JSON edited in the SAME change as code; suite + Bug Bible + B7
green + push per green chunk.

---

## 2. HARD RULES (invariants -- apply every session)

- **WORKFLOW SOURCE OF TRUTH (hard):** `workflows/otr_scifi_16gb_full.json` IS production. ANY node/
  wiring/widget change goes IN that file in the SAME change as the code (unwired code is dead). Every
  API/headless/soak run LOADS this real JSON. After editing, re-validate: `OTR_WorkflowValidator` +
  JSON round-trip + link/widget audit. `widgets_values` is POSITIONAL (BUG-LOCAL-097) -- append at
  END; a mid-list removal shifts every later value (re-audit by name).
- Do ONLY the forward order (section 3). Everything else is PARKED (section 8).
- Audio SPINE is SHIPPED + FROZEN: byte-identical master + mux-LAST (no `-shortest`);
  `test_audio_byte_identical` stays GREEN. Only sanctioned audio work = the upstream "whiny" voice fix.
- Invariants: single resident heavy <= 14.5 GB (host NVML); 100% local/offline; determinism seed-keyed;
  every in-render fallback LOUD; UTF-8 no BOM; SFW; V-12 dep isolation; no new widgets in the static
  shell (V-11).
- GIT: ONE branch `v2.0-alpha`; commit AND push together per green chunk; operator eyeball gates TAGS/
  promotions only; after a push verify HEAD==origin / no 0-byte / no BOM / AST parse on touched .py.
  prod/`main` GATED (a `v2.0-alpha-stable` tag on `v2.0-alpha` is fine).
- EVERY session updates THIS doc (lean, forward-only) + appends one entry to `docs/HANDOFF_LOG.md`.
  (The old otr-build-tracker dashboard is retired -- the small log replaces it.)
- C7 seed pins (`OTR_CAST_SEED`/`OTR_STYLE_SEED`) only behind `OTR_C7=1`; normal runs log
  `cast RNG seed=... (OS entropy)`.

---

## 3. FORWARD ORDER (do in sequence within a track)

> Two tracks. Item 1 (punch-list) is OPERATOR-GATED look-QA; the ENGINE track (items 3-4) proceeds.

1. **Punch list (GATE A) -- operator-approved.** Captions DONE. REMAINING: node-audit that LTX
   radio-open + procgen rolling credits are in the SAVED JSON (not just the headless path); prove a
   render FROM the JSON has them, then operator look-QA.
2. **latentsync -- REMOVED** (not a live lane; dropped from the order).
3. **Wan 2.2 video -- operator-approved.** Both engines BUILT + validated (wan_i2v 14B + wan_ti2v 5B).
   REMAINING = operator WEBM eyeball (14B vs 5B) + optional formal `--acceptance` GREEN (slow
   wan-music-bed leg, attended) + M9 CS-3 proof. Detail: section 4.
4. **Coverage sweep GREEN (GATE-A acceptance).** Re-run the permutation matrix after the soak fixes;
   RED until Wan lands (Wan is core/blocking). Visual-engine set is wired; writer-LLM + voice leg-sets
   still need a runnable harness. Hardening M1-M9 shipped (see HANDOFF_LOG.md); exact `--acceptance`
   invocation in `docs/2026-06-13-goforward-wan-hardening/`.
5. **3D sprints.** S-3D-0 spike + T1 template + T2a wrap smoke -> `character_3d` family. Detail:
   `docs/2026-06-09-3d-toolkit/3D_TOOLKIT_PLAN.md`.
6. **Switchable distribution S3-S6** -- generator + `.gen.json` tiers + wizard + README (closing).
   Detail: `docs/2026-06-10-switchable-workflow-architecture__decision-and-plan.md`.

**0-E parallel track:** CPU side shipped, GPU-green; Phase B (E-1..E-7) HELD on
`scripts/_otr_0e_gpu_go.txt`. **Audio parallel track:** the "whiny" voice fix (upstream TTS only;
may have self-resolved -- verify first).

---

## 4. WAN 2.2 VIDEO -- REMAINING (active build)

Two selectable Wan 2.2 engines, eyeball-gated, b-roll/camera motion only (lip-sync stays on HuMo).
Core Comfy Wan nodes (NOT the KJ wrapper). Phase 1 + the 5 code-gap fixes DONE (`2fbc2f3`).

- **Phase 2 -- 16GB leg.** Drive `eng_wan_i2v.render_clip` via the real path; ASSERT wan_i2v is the
  final_engine (FAIL LOUD on fallback) + render-phase NVML <= 14.5 GB + byte-identical mux + silent
  mp4. Reset the box first.
- **8GB tier -- TI2V-5B as a SEPARATE engine.** Fetch the TI2V-5B GGUF + wan2.2 VAE (record repo/sha/
  license, fail-closed); own flag/model/VAE env + registration + tests. Do NOT alias WanI2VEngine.
- **Eyeball gate.** Present both webms (I2V-14B vs TI2V-5B) in `docs/2026-06-12-ltx23-motion/wan_clips/`.
  Bar = real camera motion, still preserved, no warp. If motion too subtle, the Path B two-expert
  HIGH/LOW handoff is the mitigation (not a knob tweak).
- **CS-3:** sequential residency (Wan ~14GB + HuMo ~7GB cannot co-reside) -- prove per-beat NVML
  <= 14.5 GB + inter-beat reclaim (`wrapper_bridge.reclaim_idle_models`) drains the prior engine. A
  mixed Wan+HuMo episode is the test.

---

## 5. OPEN THREADS / BACK-BURNER (pointers, not plans)

- **LTX motion amount (recommended next opener thread).** LTX holds composition; open = MOTION amount.
  Env-testable A/B first (no code): at 832x480 set `OTR_LTX_SAMPLER=distilled` +
  `OTR_LTX_SAMPLER_NAME=euler_cfg_pp` + `OTR_LTX_I2V_STRENGTH=0.75`, A/B vs the good 5/09 `l001` /
  5/28 `b001` bookends; if it matches, bake those + boomerang + audio-length into `eng_ltx_video.py`.
  Forensic in `BUG_LOG_2026-06.md` (BUG-LOCAL-412).
- **CS-2** phase attribution (~16 GB machine pin vs 14.5 render-phase). **CS-4-open** (deprioritized)
  14B HuMo umt5-TE detach; default char tier is `humo_1.7B`.
- **README "what to expect per video model"** (newbie audience; folds into S6) once the opener bake-off
  settles.
- **Ship defaults (release):** announcer + character = flux_still, music = viz_green; HuMo/3D
  selectable-not-default until verified. Operator eyeballs 2-3 finals/slot.
- **Harness polish:** output-tree resolver should prefer the live server's `OTR_OUTPUT_DIR` (fail LOUD
  on mismatch) -- NOTE: bit the 2026-07-01 visual soak (the capstone default tree disagreed with the
  launcher's `--output-directory`; the driver now pins `OTR_SOAK_SERVER_OUTPUT`). OH-3 janitor at boot.
- **OH-4** live->attic migration STAGED (`docs/2026-06-11-output-tree-consolidation/`), awaits "go OH-4".
- **0-E Phase B** tickets E-1..E-7, gated on the sweep GO file.
- **Operator gates:** ComfyUI Desktop relaunch, fresh-render acceptance, whiny-voice reel, S-3D-0 green
  light, `v2.0-alpha-stable` tag decision.

---

## 6. RUNWAY (remaining sprints to "done")

"Done" = platform wired into real episodes (real per-beat video + byte-identical mux + legacy procgen
gone) + all video models verified live + the first 1-2 3D models rendering. ~s2-s9: S-3D-0 spike ->
T2b keystone GO/NO-GO -> T4 driver + LOOK gate -> W7 production wiring + soak ("v1-usable") -> S3-S6
distribution. SHORTCUT FORK: keystone NO-GO -> `character_3d` defers (HuMo-2D stays) -> ~2-3 sprints.

---

## 7. POINTERS (evidence + tooling)

- Done history: `docs/HANDOFF_LOG.md` (recent) + `docs/GO_FORWARD_ARCHIVE.md` (deep).
- 3D spec (item 5): `docs/2026-06-09-3d-toolkit/3D_TOOLKIT_PLAN.md`.
- Switchable spec (items 3 + 6): `docs/2026-06-10-switchable-workflow-architecture__decision-and-plan.md`.
- Wan/sweep hardening: `docs/2026-06-13-goforward-wan-hardening/`.
- Bug logs: ACTIVE `BUG_LOG_2026-06.md` (BUG-LOCAL-400+); ARCHIVE `BUG_LOG.md` (001..~305).
- Bug Bible: `C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide` (`BUG_BIBLE.yaml` +
  `tests/bug_bible_regression.py`; cd-to-root + venv python + RELATIVE path).
- Smoke harness: `scripts/queue_smoke.py` + `scripts/otr_api.py`. Overnight sweep launch +
  GO file: `scripts/_otr_0e_gpu_go.txt`.

---

## 8. PARKED -- not now

Story-spine; story-pipeline; broader audio stack; MuseTalk; RTXUpscale; switchable S3-S6 (closing,
after 3D); 3D GPU lanes until S-3D-0 + operator green light; the STORY-ENGINE quality roundtable
side-campaign (`docs/2026-06-21-allnight-864-frontier/SPRINT_READY_PLAN.md` -- resume only on explicit
operator go).
