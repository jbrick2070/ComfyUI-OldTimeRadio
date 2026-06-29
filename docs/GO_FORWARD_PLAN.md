# OTR GO-FORWARD PLAN -- SINGLE SOURCE OF TRUTH (what's LEFT)

> **>>> CURRENT STEP -- 2026-06-29 DELETE ALL CODE-GATING (registry IS the menu). HEAD `cf4487a6` ==
> origin/v2.0-alpha (pushed). prod/main + tags GATED.** OPERATOR DIRECTIVE (2026-06-29, supersedes the
> opt-in/validated model): NO opt-in / validation / promotion gates ANYWHERE (video, image, voice, LLM).
> The registry IS the menu -- a REGISTERED engine is SELECTABLE and renders (may hard-fail LOUD, that's OK);
> validation is the operator's MANUAL process, never a code gate. No `OTR_ENABLE_*`, no `VALIDATED_ENGINES`
> filter, no production guard, no behind-the-scenes "model waiting to be promoted".
>
> CODE-READY PLAN (kibitz-hardened, 2 grounded rounds Codex+Antigravity -> the AUTHORITATIVE spec is
> `kibitz-runs/2026-06-29-delete-optin-v2/r2/final.md`; problem framing in `docs/2026-06-29-dropdown-optin/`).
> Sequenced, suite+BugBible+B7 green AND push per green chunk to v2.0-alpha:
> - **C2** remove the `requires_flag` GATE: base `engine_registry_base.assert_usable` L222-228 + each
>   RENDER-READY video adapter check (humo / wan_i2v / wan_ti2v / still_parallax / mesh_stage / ltx_video /
>   ltx_av / visualizer). Image adapters rely on the base gate only (flux2_klein/z_image just disk-check).
>   KEEP the disk MISSING_MODEL checks. KEEP `requires_flag` field as vestigial (set None on survivors) +
>   `GATED_BY_FLAG` enum member -- the audio protocol-parity tests iterate AudioEngine.__annotations__ and
>   audio is FROZEN. Flip the ~video/image tests that assert gated_by_flag. (C2 = the actual smoke unblock.)
> - **C3** UNREGISTER the dark scaffolds (NotImplementedError render path): `triposr`, `triposg_talk`,
>   `hunyuan3d_talk`, `trellis_talk` (video) + `hidream_i1`, `sd35_large` (image) -- drop @register + package
>   import + CAPABILITIES row; FIX the ripple IN THE SAME CHUNK: render_driver `SYNTH_FALLBACKS`/`ENGINE_FAMILY`/
>   `OOM_ENGINES`/`EXPECTED_OOM_TRAIL`, `otr_image_director.three_d_locked_slots`, and the fixtures/tests that
>   expect them registered (use a test-registered stub).
> - **C4** drop `VALIDATED_ENGINES` + `validated_engine_names()` (both registries) -> directors use
>   `all_engine_names()`; flip test_tested_only_dropdown_gate + test_still_aspect_and_labels +
>   test_video_triposr + test_ltx_audio_in_engine + test_video_cheap_render.
> - **C5** harness decouple from the flag: gpu-smoke drop the `flag_set` ready-assert; coverage drop the
>   `OTR_ENABLE_WAN_*` acceptance_preflight; dep-pilot KEEP its probe manifest (module/class/forward metadata
>   not in CAPABILITIES) but rename OPT_IN_ENGINES->probe + drop `flag` keys; fix test_video/audio_dep_pilot.
> - **C6** voice+LLM: remove the requires_flag GATE from the audio voice adapters + audio registry base +
>   any LLM opt-in gate (writer/openrouter as a GATE) -- KEEP creds/file-presence checks. HARD CONSTRAINT:
>   `test_audio_byte_identical` MUST stay green (defaults + master-mux UNCHANGED; only selectability changes).
> - **C7** docstring/comment cleanup (engine_registry_base GATED_BY_FLAG now dead for video/image) + a guard
>   test (no REGISTERED engine carries a live flag gate / renders NotImplementedError).
>
> **C2 SHIPPED (2026-06-29 this session):** removed the `requires_flag` GATE -- the base
> `engine_registry_base.assert_usable` block is gone + the 8 render-ready video adapters
> (humo / wan_i2v / wan_ti2v / still_parallax / mesh_stage / ltx_video / ltx_av / visualizer) and the 4 image
> survivors (flux2_klein / z_image_turbo / qwen_image / lumina_image) all set `requires_flag=None` (vestigial;
> the `GATED_BY_FLAG` enum member + the field annotation are KEPT for audio protocol-parity, audio FROZEN).
> Flipped ~30 video/image gate tests to the "registry IS the menu" contract. The dep-pilot/gpu-smoke RIPPLE
> (3 tests) was handled test-only -- the dep-pilot no-drift contract dropped the flag coupling + gpu-smoke
> dropped the gated_by_flag detail; the harness CODE refactor (OPT_IN_ENGINES->probe rename, drop flag keys,
> drop the flag_set ready-assert, coverage acceptance_preflight) STAYS for C5. Full suite 5750/0, Bug Bible
> 16/3xfail, B7 green. The per-model SMOKE PASS resumes now (selecting humo/wan/ltx_av/still_parallax/
> flux2_klein/z_image just renders -- no launch flag). NEXT = C4.
>
> **C3 SHIPPED (2026-06-29 this session):** UNREGISTERED the dark NotImplementedError scaffolds -- video
> triposg_talk / hunyuan3d_talk / trellis_talk (eng_character_3d) + triposr; image hidream_i1 + sd35_large
> (dropped @register + package import + CAPABILITIES row; the source files are KEPT -- they "return when
> built"). Fixed the ripple in the SAME chunk: render_driver + scripts/otr_video_soak SYNTH_FALLBACKS /
> ENGINE_FAMILY / OOM_ENGINES / EXPECTED_OOM_TRAIL -- the soak's SYNTHETIC OOM head renamed triposg_talk ->
> the explicit `soak_oom_3d` stub (same 3-hop chain shape, no dead real-engine names); the video + image
> dep-pilot probe manifests dropped the unregistered entries. Tests: rewrote test_video_character_3d +
> test_video_triposr to the unregistered/source-still-dark contract (instantiate the classes directly),
> retargeted the soak + dropdown-gate + dep-pilot + still-aspect tests, test_image_platform_c1's 3D-lock now
> uses a test-registered stub, and DELETED the now-empty test_image_engine_matrix_peers (both subjects
> unregistered). Full suite 5723/0, Bug Bible 16/3xfail, B7 green. NEXT = C5.
>
> **C4 SHIPPED (2026-06-29 this session):** DELETED the validated-subset dropdown filter -- `VALIDATED_ENGINES`
> + `validated_engine_names()` removed from BOTH the video and image registries; the 3 director COMBOs
> (`otr_video_director._video_model_combo` / `_image_model_combo` + `otr_image_director._image_model_combo`)
> now build from `all_engine_names()`, so the dropdown == the full registry (every registered engine is
> selectable; manual validation only). Tests: rewrote test_tested_only_dropdown_gate to the dropdown==registry
> contract + retargeted test_still_aspect_and_labels / test_ltx_audio_in_engine / test_video_cheap_render off
> the deleted filter. Full suite 5722/0, Bug Bible 16/3xfail, B7 green. NEXT = C6.
>
> **C5 SHIPPED (2026-06-29 this session):** DECOUPLED the harness from the flag. gpu-smoke
> (`otr_video_gpu_smoke`) dropped the `flag_set` ready-assert + the ENGINES "flag" key + the "set the flag"
> next-step (NOT READY now means deps/forward absent, never an opt-in flag). coverage-sweep
> (`otr_coverage_sweep`) dropped the `OTR_ENABLE_WAN_*` acceptance preflight + `WAN_ENABLE_FLAGS` (only
> OTR_TEST_MODE + --exclude stay gated). dep-pilot (`otr_video_dep_pilot` + `otr_audio_dep_pilot`): renamed
> the probe manifest `OPT_IN_ENGINES` -> `PROBE_ENGINES` and deleted the vestigial "flag" keys + the verdict
> "flag" field. Tests: retargeted test_video_gpu_smoke (not-ready-without-deps), test_coverage_sweep_acceptance
> (no enable-flag preflight), test_video_dep_pilot + test_audio_dep_pilot (PROBE_ENGINES + curated no-drift
> contract, not flag-derived). Full suite 5722/0, Bug Bible 16/3xfail, B7 green. NEXT = C7.
>
> **C6 SHIPPED (2026-06-29 this session):** removed the VOICE + LLM opt-in gates. Audio: deleted the
> `GATED_BY_FLAG` block from the audio registry `assert_usable` (registry IS the menu) + set
> `requires_flag=None` on the 3 flagged voice/music adapters (chatterbox / dia / stable_audio_music). The
> `requires_flag` field annotation + the `GATED_BY_FLAG` enum are KEPT (audio protocol-parity frozen). LLM:
> `openrouter_enabled()` now returns `bool(OPENROUTER_API_KEY)` only -- the separate `OTR_ENABLE_OPENROUTER`
> opt-in flag is GONE as a gate (any CONFIGURED LLM with its creds present is selectable; the API-key/creds
> check stays). **HARD CONSTRAINT HELD: test_audio_byte_identical GREEN** -- the default voice/music engines
> (already flag-free) + the master-mux path are UNCHANGED; only the selectability of non-default engines
> changed. Flipped ~9 audio gate tests + 1 OpenRouter gate test to the selectable contract. Full suite
> 5722/0, Bug Bible 16/3xfail, B7 green. NEXT = C7 (docstring cleanup + the guard test).
>
> WHERE WE ARE (2026-06-29 late session): the interim "drive the flag from the dropdown selection" approach
> (option B, 1c73aec) was REVERTED at `cf4487a6` (pushed) -- the operator wants the gate DELETED, not driven.
> Suite 5750/0, Bug Bible 16/3xfail at HEAD. The per-model SMOKE PASS is PAUSED: it resumes AFTER C2 lands
> (once the gate is gone, selecting humo/wan/ltx_av/still_parallax/flux2_klein/z_image just renders -- no
> launch flag). Smoke list (post-C2): video = visualizer / ltx_video / ltx_av / humo_14B_169 / wan_ti2v /
> wan_i2v / still_parallax(3D); image = flux_gen1 / flux2_klein / z_image_turbo. ComfyUI Desktop needs a
> RESTART to load each code chunk (Python module cache).
>
> WHERE WE ARE (2026-06-29 session -- a big operator-directed UX + robustness pass on top of C1-C6, all
> pushed to v2.0-alpha):
> - **Node 87 truthful video dropdowns** (`c6183500`+`11afbd64`): the 3 hidden Route-A per-role slots carried
>   saved engines (humo_14B_169/wan_ti2v/ltx_video) that SILENTLY beat the visible other_beats dropdown ->
>   humo ran unasked + flux minted stills for it. Exposed the 3 slots on node 87 + defaulted them to the
>   `(use Other Beats default)` sentinel; 3 visible video slots -> `visualizer` (clean default, no red);
>   `allow_auto_fallback=False` (strict, no silent swap). Synced config/profiles/16gb_full.json
>   (other_beats_visual=visualizer; per-role keys OMITTED=inherit) + test_workflow_live_passes_validator +
>   test_two_heavy_roles. Proven by a 449s all-visualizer render (zero FLUX/HuMo, obs_publish OK, audio
>   byte-identical). Suite 5750/0.
> - **Story-quality v2 BAKED IN / de-forked** (`73b32af3`): on/off lever removed, the C1-C6 spine runs
>   unconditionally; ~16 files; suite green.
> - **Kokoro pack (54 voices) installed** (1038lab/KokoroTTS -> models\TTS\KokoroTTS\voices) so the announcer
>   voices (bm_george etc.) resolve, not just bm_fable. On disk (not git).
> - **Soak hardening** (`dd273614`): the overnight soak failed SILENTLY (337 legs / 0 episodes) on a config
>   bug -- `act_count=3` vs 80-word episodes -> the budget validator rejected EVERY leg. ROOT FIX -- WORDS
>   DYNAMICALLY STEER THE ACTS: `_otr_overnight_420_soak.py` derives per-leg `act_count=max(1,min(6,words//120))`,
>   words floored at 60 (legacy 420w->3 preserved) so the budget node can never reject a leg. +
>   `docs/2026-06-29-soak-runbook/SOAK_RUNBOOK.md`: the verify-a-shipped-obs-episode-BEFORE-trusting-the-loop
>   gate + full procedure. Re-validated: leg 0 success, 1 ep -> obs. (Harness .py is .gitignored -- the fix
>   lives on disk; force-add if you want it git-tracked.)
> OPEN: **HUMO "dropdown is the only switch" flag-free refactor REVERTED** (kept the tree green) -- removing
> humo's OTR_ENABLE_HUMO gate ripples into the dep-pilot + gpu-smoke infra (they conflate "flag-gated" with
> "needs-dep-verification"); needs a design call (do flag-free engines still get dep-verified?) before redoing
> across ~16 engines + ~20 tests. **char_voice=indextts2** as the saved node default = 1-value follow-up.
> **/soak skill** (interactive preflight checker) requested -- can't install a Cowork skill from a session
> (-> Settings>Capabilities); substance = the runbook + the dynamic-acts fix. **Kibitz r1** done
> (codex+antigravity reviews in kibitz-runs/2026-06-29-soak-runbook/r1; not synthesized -- operator pivoted).
> node-87 other_beats SETTLED = keep `humo_1.7B` (operator 2026-06-28; both stay selectable).
> - **C3 shipped** (`5198d8fe`): S2 news-coda arc bridge floor -- v2 system examples +
>   `_NEWS_CODA_ARC_BRIDGES` (arc_shape-keyed, sha256(cast_seed)-selected, each validator-checked);
>   unknown arc / v2-OFF keep the legacy NEWS_CODA_POOL byte-identical. Lock: `test_story_quality_coda.py`.
> - **C4 SHIPPED `041d28d8` (S3 body-gate text-scored accept; was PINPOINTED -- `OTR_LedgerScriptWriter.run`
>   L4515-4589):** new module helper `_otr_body_score` scores original vs reroll on the SHIPPED text
>   (`10*grounding_failed + 3*hard_leak + 2*trunc + 2*run_on + 1*roster_caps`, lower wins, ORIGINAL on tie);
>   hard_leak = `verify_and_repair_line(text, policy=...).needs_recompose`; run_on uses C2's
>   `derive_one_breath_cap` + `max_clause_markers=max(3,cap//8)`; roster_caps matches the locked cast full
>   names from `req.allowed_roster`; a MID-CLAUSE roster-caps hit TRIGGERS a reroll (never an in-place strip);
>   v2-OFF keeps `_use_rr=_bg_res_ok` byte-identical. Lock: `tests/test_story_quality_body_gate.py` (9). The
>   original C4 design follows for history: today the gate
>   rerolls ONLY on grounding failure (`validate_composed_grounding`) and accepts the reroll ONLY if it
>   re-validates grounding (L4548 `if _bg_res_ok and _bg_res.text.strip()`). S3 (v2-gated) replaces the
>   ACCEPT test with a total-order score on the SHIPPED TEXT for BOTH `cleaned` (orig) and `_bg_res.text`:
>   `score = 10*grounding_failed + 3*hard_leak + 2*trunc + 2*run_on + 1*roster_caps` (lower wins, ORIGINAL
>   on tie). Add a module helper `_otr_body_score(text, bg_entry, grounded_nouns, entity_policy, req)`:
>   grounding_failed = not validate_composed_grounding(...)[0]; hard_leak from verify_and_repair_line(text,
>   _episode_entity_policy) [check its return shape first]; trunc = is_truncated; run_on = flag_one_breath(
>   text, max_words=derive_one_breath_cap(req.words_per_beat_range))[0]; roster_caps = an ALL-CAPS token-run
>   matching an episode CAST FULL NAME (NOT any caps token -- NASA/UCLA safe; cast names live on the locked
>   cast rows). v2-OFF keeps `_use_rr = _bg_res_ok` (byte-identical). roster_caps detail: a MID-CLAUSE hit
>   (grammatical subject/object, "...when CLARISSE GORDON claim...") => set the reroll TRIGGER (needs_recompose,
>   alongside `not _bg_ok`), NEVER an in-place strip (that yields "...when claim..."); only a LEADING/TRAILING
>   vocative is scrub-safe via scrub_roster_vocative. Add `tests/test_story_quality_body_gate.py`. Imports
>   needed in the writer: derive_one_breath_cap + is_truncated (from _otr_line_hygiene). Reuse the existing
>   reroll loop -- do NOT add a second compose path. **C5 (S4) SHIPPED `5afde8fb`** = find_cliche_phrase exact-span replacement
>   before EVERY quality-gate return path (kept-reroll AND kept-original) in _otr_line_composer L~2515, curated
>   safe-replacement map (case-match) + the _quality_repair_attempted guard; else accept 2nd-best + stamp
>   `cliche_shipped_after_reroll`. **C6 (S5) SHIPPED `343e0868`** = story_quality_scan two principals = top-2 by dialogue-line
>   count (wants are verb phrases, no name parse); update test_story_quality_scan_r2 expected register_overlap.
>   Then the 1-local + 1-grok acceptance render (report voiced-word counts; the length experiment).
> - **C2 shipped** (`75930608`): G1 gate de-compression -- `_QUALITY_COLLAPSE_HINT_V2` (non-compressing,
>   v2-selected), `line_quality_defect_score` v2 keep-better, and the budget-derived one-breath cap
>   (`derive_one_breath_cap` + relaxed clause `max(3,cap//8)`) threaded at first-pass / reroll / scan from
>   one helper + `meta["words_per_beat_range"]` stamped v2-only. v2-OFF byte-identical. Suite 5736/0,
>   Bug Bible green. Lock: `tests/test_story_quality_g1.py`.
> - **BASELINE FIX shipped** (`6d30cec2`): HEAD had 6 PRE-EXISTING reds (b7 sweep, capability_profiles,
>   force_input_sockets, 2x workflow_apply, structure-pin) from a ComfyUI Desktop UI-save that polluted
>   `otr_scifi_16gb_full.json`. Fixed: node 87 widgets_values display-labels -> bare engine ids
>   (humo_14B_169 (16:9)->humo_14B_169 @0/1/16; visualizer (16:9)->humo_1.7B @2); dropped converted-input
>   widget keys on nodes 80-83; renamed `alias`->`node_key` in build_humo_bakeoff_workflow.py; updated the
>   stale ltx_audio_in pin. **OPERATOR FLAG:** node-87 other_beats set to humo_1.7B (committed test intent),
>   NOT the UI-save's `visualizer` -- flip if visualizer was the intended dropdown choice (1-value change).
> - **C1 shipped** (`738b3b63`): shared leaf helpers (`derive_one_breath_cap`/`_hard_clauses`/
>   `find_cliche_phrase` in `_otr_line_hygiene.py`) + golden fixtures + `test_story_quality_golden.py`
>   (kibitz option C: hybrid raw-failing + corrected-target rows). Additive; v2-OFF byte-identical.
>   Full suite 5732/0, Bug Bible 16/3xfail, B7 green.
> - **C2 GROUNDED MUST-DO (from the fixtures kibitz):** `flag_one_breath`'s SOFT clause tripwire (>22 words
>   AND >=3 [,;]+conj) fires even under a raised `max_words` -- so C2's v2 path MUST thread BOTH
>   `derive_one_breath_cap(range)` for max_words AND a relaxed `max_clause_markers` (the gate at
>   `_otr_line_composer` L2319 calls `flag_one_breath(cleaned)` with DEFAULTS today). Else fuller multi-clause
>   lines keep getting rerolled -> defeats G1. See `kibitz-runs/2026-06-28-story-quality-fixtures/final.md`.
> - **REMAINING:** C2 (G1) -> C3 (S2 coda) -> C4 (S3 body-gate) -> C5 (S4 cliche) -> C6 (S5 scan); each
>   suite+BugBible+B7 green, commit+push per chunk; then the 1-local+1-grok acceptance render.
>
> **>>> PRIOR CONTEXT -- 2026-06-28 3-WAY STORY-QUALITY KIBITZ CONVERGED -> build-ready final.md.** The operator-directed
> story-quality kibitz (Codex gpt-5.5/high + Antigravity gemini-3.5-pro + Claude as code-grounded panelist AND
> judge; 8 agent calls, $0 local) converged at r4 -> `kibitz-runs/2026-06-28-story-quality/final.md` (r1->r4
> plans + judgment logs alongside; anchors in `docs/2026-06-28-story-quality-kibitz/`).
> - **LEAD FINDING (G1):** the 2026-06-27 anti-overstuffing gates now OVER-correct. `_QUALITY_COLLAPSE_HINT`
>   (`_otr_line_composer` L2293) literally says "rewrite ... under ~20 words, at most one concrete detail" -> it
>   FORCES the compression that turns rich lines into noun-salad, AND (the 14-beat skeleton x the ~22-28-word
>   `one_breath` cap) hard-bounds every episode at ~210-310 voiced words regardless of `target_words`. LENGTH +
>   CRAFT share ONE root cause -- the fix is to TUNE the gates (v2-gated), not remove them.
> - **Evidence:** the 27-ep overnight all-visualizer soak + 3 operator-directed enrichment renders (mistral-720,
>   grok-720, grok-1340; all-visualizer, kokoro). Grounded census: the 2026-06-27 gate-seam cluster (3.1-3.7)
>   HOLDS (anchor-stuffing/dignity/cost-boilerplate/near-dup/coda-trunc all ~0). NEW defects: G1 (lead), S2 (coda
>   fallback 18/29, WEAK-LOCAL -- grok PASSES the bridge), S3 (gemma reroll roster-caps + run-ons), S4 (cliche
>   still ships), S5 (interchangeable voices -> measurement-only), S1 (seed drift -- DEFERRED), S6 (CUT).
> - **Length probe:** `target_words` is a near-no-op (grok @720w=213 voiced words, @1340w=263, @2000w=ERROR over
>   the ~1363-word ceiling = `BEAT_WORD_HARD_MAX(80) x 14 beats`). Addressed via G1 as a SIDE EFFECT; NEVER padded.
> - **Panel corrected 3 draft errors (all code-grounded):** S1 dramatic binding, S5 `speech_signature` threading,
>   and the G1 keep-better re-score ALL already EXIST -> reframed "add" -> "fix/enforce". `final.md` = a
>   v2-gated, flag-OFF-byte-identical, **6-green-commit** plan (shared leaf helpers -> G1 -> S2 -> S3 -> S4 -> S5)
>   with one shared `derive_one_breath_cap` helper, a concrete golden-ledger acceptance set, and a
>   verify-at-build checklist.
> - **NEXT:** operator DECISION GATE (final.md last section: confirm `story_quality_v2` is the gate flag; accept
>   "length as a side effect, per-line cap up to ~60w, no padding"; keep the ~1363 ceiling; S1 deferred / S6 cut)
>   -> then a CODER window builds the 6-commit sequence (CPU/content only, suite+BugBible+B7 per chunk, push per
>   green chunk to `v2.0-alpha`).
> - **BOX STATE:** a headless ComfyUI on :8000 (booted FLOOR + `OTR_ENABLE_OPENROUTER=1`, PID 27224) is left
>   RESIDENT idle from the enrichment renders; the operator's Comfy Desktop :8001 was untouched. Selective-reset
>   before the next headless run (CLAUDE.md S4). NOTE: the soak launcher hydrates the OpenRouter KEY but does NOT
>   set `OTR_ENABLE_OPENROUTER=1` -- that gap is why the overnight soak ran 0 frontier legs.
>
## 1. CURRENT STEP

See the CURRENT STEP block at the TOP of this file -- 2026-06-29: DELETE ALL CODE-GATING
(registry IS the menu; no opt-in/validation/promotion gates in video/image/voice/LLM).
Code-ready kibitz-hardened plan = `kibitz-runs/2026-06-29-delete-optin-v2/r2/final.md`,
sequenced C2-C7; interim option-B reverted at `cf4487a6` (pushed); suite 5750/0. The
per-model smoke pass resumes after C2 (the gate removal). C2 + C3 + C4 + C5 + C6 SHIPPED this
session (requires_flag GATE removed; dark scaffolds unregistered; VALIDATED_ENGINES filter
deleted; harness decoupled from the flag; voice + LLM gates removed with test_audio_byte_identical
GREEN; suite 5722/0). NEXT = build C7 (docstring/comment cleanup + the guard test).

Prior step (2026-06-28, story-quality): C1-C6 SHIPPED (HEAD `343e0868`); the 1-local +
1-grok acceptance render was still pending when the opt-in-gating work took priority.

Older ACTIVE/SUPERSEDED step history -> `docs/GO_FORWARD_ARCHIVE.md`.

## 2. HARD RULES (invariants -- apply every session)

- **WORKFLOW SOURCE OF TRUTH (operator, hard):** `workflows/otr_scifi_16gb_full.json` IS the
  production workflow. (1) ANY node / wiring / widget change MUST be made IN that file in the SAME
  change as the code -- code that is not wired into this JSON is DORMANT and does nothing (the §4D
  miss, 2026-06-13: node + blend input shipped + tested but unwired -> ran dead in production). After
  editing, re-validate via `OTR_WorkflowValidator` + a JSON round-trip + the link/widget audit.
  (2) EVERY API / headless / soak run MUST LOAD this real JSON -- never a stale copy, a generated
  `.gen.json`, an ad-hoc graph, or the Linux-mount snapshot (the sandbox mount lags file writes; always
  read/write the Windows path + verify via Desktop Commander).
- Do ONLY the forward order (section 3). Everything else is PARKED (section 8) -- not story-spine, not
  story-pipeline, not the broader audio stack, not other ROADMAP items.
- Audio SPINE is SHIPPED + FROZEN: byte-identical master + mux-LAST (no `-shortest`);
  `test_audio_byte_identical` stays GREEN. Only sanctioned audio work = the upstream character-voice
  "whiny" fix.
- Invariants: single resident heavy engine <= 14.5 GB (host NVML); 100% local/offline; determinism
  seed-keyed (per-seed within a render, NOT run-to-run); every in-render fallback LOUD; UTF-8 no BOM;
  SFW; V-12 dep isolation; no new widgets in the static workflow shell (V-11).
- GIT (operator 2026-06-10): ONE branch `v2.0-alpha`; commit AND push together per green chunk; the
  operator eyeball gates TAGS/promotions only; after a push verify HEAD==origin / no 0-byte / no BOM /
  AST parse on touched .py. prod/`main` is GATED until operator work is done (a `v2.0-alpha-stable`
  tag on `v2.0-alpha` is fine).
- EVERY session updates this doc + the `otr-build-tracker` dashboard (content; keep the gauge+lanes
  styling).
- C7 seed pins (`OTR_CAST_SEED`/`OTR_STYLE_SEED`) only behind `OTR_C7=1`; normal runs must log
  `cast RNG seed=... (OS entropy)`. Do NOT set `OTR_C7` for normal runs.

---

## 3. FORWARD ORDER (do in sequence)

> **Two tracks, parallel.** Item 1 (punch-list audit) is OPERATOR-GATED (look-QA -- section 5); the
> ENGINE track (items 3-4, Wan + sweep GREEN) proceeds NOW. "In sequence" applies WITHIN a track, not
> across the operator gate.

1. **Punch list (GATE A) -- OPERATOR APPROVED 2026-06-21, proceed.** Captions DONE (node 86
   `OTR_CaptionBurn` in `otr_scifi_16gb_full.json`, profile resolves `burn_captions=True`). REMAINING:
   node-level audit of LTX radio-open + procgen rolling credits -- baked into the headless path but maybe
   NOT into the saved JSON; prove a render FROM the JSON has them, then operator look-QA.
2. **latentsync -- REMOVED 2026-06-21 (operator: "we ripped it out").** Verified: NO engine file under
   `nodes/_otr_video_engines/`, 0 references in `otr_scifi_16gb_full.json`, only a few stray comment/env
   strings remain (`OTR_LSYNC_BASE_ENGINE`). Not a live lane -- dropped from the forward order. (A trivial
   code-comment scrub of the stray strings can ride any future cleanup; not a roadmap item.)
3. **Wan 2.2 video engine (section 4) -- OPERATOR APPROVED 2026-06-21 ("100% approved"): proceed with the
   eyeball + acceptance.** BOTH engines BUILT + validated live (2026-06-14, `bcbe05a`):
   wan_i2v (14B, post mixin-refactor) + the new wan_ti2v (5B/GGUF 8GB tier). REMAINING = the operator
   WEBM EYEBALL (14B vs 5B) + the optional formal full-episode `--acceptance` GREEN exit (slow
   wan-music-bed leg, run attended) + the M9 CS-3 instrumented proof. Code-complete; gates are the
   operator's.
4. **Coverage sweep GREEN (GATE A acceptance).** Re-run the permutation matrix after the soak fixes.
   Matrix (additive, not cross-product): a visual-engine leg-set (varies each of music/announcer/
   other_beats), a writer-LLM leg-set (varies node-1 `creative_writing_model`/`technical_model`), and a
   curated voice-variation leg-set (2-3 refs per voice engine). Unique story per leg (OS entropy, no
   seed pins). **Wan is a CORE/BLOCKING engine** -- the sweep is NOT green until `wan_i2v` (and
   `wan_ti2v`) pass, so it stays RED until item 3 lands; that is expected. This re-run also answers the
   one open R2 question: whether `humo_1.7B` renders NATIVE char beats at 70w once its enable flag is on
   (the soak floored it only because the flag was off). **GATE-A precondition: harden the
   sweep FIRST (section 4A M1-M4) -- DONE 2026-06-13: the M1-M5 acceptance gate landed
   (`scripts/otr_coverage_sweep.py --acceptance`), so a silent fallback / empty-results
   run / missing VRAM measurement now scores RED, not GREEN.**
   **S6 harness reality:** `otr_coverage_sweep.py` enumerates ONLY the visual-engine
   leg-set today (the dropdown rotation). The writer-LLM leg-set (node-1
   `creative_writing_model`/`technical_model`) and the curated voice-variation leg-set
   are NOT yet wired into a runnable harness -- TODO: point them at a real driver
   (e.g. a `run_combo_matrix.py`) or run them as separate parametrized soak legs.
   "Coverage sweep GREEN" today means the visual-engine set only.

   **SOAK READINESS AUDIT (2026-06-13).** Walked the registry + harness. Conclusion:
   **clear to run a wan_i2v-only soak today** (no wan_ti2v hard prereq for validation).
   Verified live: `wan_i2v` enumerates `ok`/runnable under `16gb_full` (legs
   `music_visual=wan_i2v` + `other_beats_visual=wan_i2v`) -- the old "add wan_i2v to the
   enable-set" note is STALE/resolved. 27 legs enumerate; the only skips are
   `hunyuan3d_talk`/`trellis_talk` (missing cu128 toolchain, expected darks). Wan models
   on disk + `OTR_ENABLE_WAN_I2V=1` env known. **Two limitations to know:**
   (i) `--acceptance` exit is RED-by-construction until `wan_ti2v` is built (M2 requires
   BOTH Wan engines) -- expected; read the per-leg verdicts in `coverage_sweep_summary.json`,
   the wan_i2v leg PASS/FAIL is the meaningful signal.
   (ii) **The M1 no-fallback (CS-1) gate is bound to `--acceptance`** (`forbid_fallback=
   args.acceptance`); the capstone CLI does not expose it. So re-running the NON-Wan
   permutation soak (the set that originally false-greened) WITH the M1 fix active and a
   clean GREEN/RED exit needs either `wan_ti2v` built OR a small **`--strict-fallback`**
   flag that decouples M1 from the Wan-engine requirement (~10 lines; RECOMMENDED, optional
   -- operator's call). Until then: `--acceptance --only wan` exercises M1 on the wan_i2v
   legs (overall RED expected), and a non-acceptance sweep runs but with M1 OFF
   (informational). No half-built code, no missing capability rows beyond the deferred
   `wan_ti2v`, no broken tests (the 2 `test_model_catalog_scan` reds are pre-existing /
   environmental, tracked separately).
5. **3D sprints.** s2 = S-3D-0 spike + T1 template + T2a wrap smoke; then the `character_3d` family
   (image-routing must-fixes already landed). Detail in the 3D plan (pointers).
6. **Switchable distribution S3-S6** -- generator + `.gen.json` tiers + wizard + README (closing phase).

**0-E parallel track:** `ltx_orbit`/`still_parallax`/`mesh_stage` CPU side shipped + all three GPU-green;
Phase B (E-1 probe, E-6 renders, per-engine sweep legs) HELD on the `scripts/_otr_0e_gpu_go.txt` GO file.

**Audio parallel track (own window, never blocks video):** the character-voice "whiny" fix (upstream TTS
only; frozen spine untouched). Operator note: may have self-resolved -- verify before scheduling work.

---

## 4. WAN 2.2 VIDEO -- REMAINING (active build)

Two selectable Wan 2.2 video engines, eyeball-gated, b-roll/camera motion only (lip-sync stays SEPARATE
on LatentSync/HuMo). Core Comfy Wan nodes, NOT the KJ wrapper (KJ drags in SageAttention + a numpy<2 pin
this box violates). Phase 1 + the 5 code-gap fixes are DONE (`2fbc2f3`); the full grounded spec is in
that commit + git history of this file.

- **Phase 2 -- 16GB engine leg.** Drive `eng_wan_i2v.render_clip` via the real path
  (`scripts/otr_run_leg.ps1` / `coverage_sweep --only ...`). ASSERT `wan_i2v` is the final_engine in the
  trace (FAIL LOUD on fallback, CS-1) + render-phase NVML <= 14.5 GB + byte-identical audio mux + silent
  mp4 (h264/yuv420p/bt709, fps 25, `has_audio` False). Kill/reset the Phase-1 server first.
- **8GB tier -- TI2V-5B as a SEPARATE engine.** Fetch the TI2V-5B GGUF (Q6/Q5_K_M) + the wan2.2 VAE into
  `C:\ComfyUI-Models\` (record HF repo + sha256 + license, fail-closed). Define a NEW `wan_ti2v` engine
  (own flag/model/VAE env, registry registration, `_node_candidates` incl. the 5B latent node, loader
  mode, `canonicalize`, profile hook + tests) -- do NOT alias `WanI2VEngine`.
- **Eyeball gate.** Present both webms (I2V-14B vs TI2V-5B, same still + prompt) in
  `docs/2026-06-12-ltx23-motion/wan_clips/`. Bar = real camera motion, still preserved, no warp.
  **S3 motion risk to watch:** the wired I2V-14B fp8 is a SINGLE low-noise expert (the
  two-expert HIGH/LOW MoE handoff, Path B, is NOT wired -- see `eng_wan_i2v` header). If
  the "real camera motion" bar FAILS (motion too subtle / static), the Path B two-expert
  HIGH/LOW handoff is the mitigation, not a knob tweak. Call this out at the eyeball.
- **Risk CS-3 (reframed):** sequential-residency, NOT co-residency -- see section 4A M9
  and the section-5 CS-3 entry. The supervised Wan batch proves the inter-beat reclaim,
  it does not "decide if they co-stage."

---

## 4A. WAN + GATE-A SWEEP HARDENING (roundtable 2026-06-13, grounded vs HEAD 134f8e2)

Folded from a 3-model roundtable (GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4) + Claude's
grounding against the real code. Full judgment + raw reviews:
`docs/2026-06-13-goforward-wan-hardening/`. These gate item 3 (Wan) and item 4 (sweep
GREEN). MUST-FIX -- until M1-M4 land, a GREEN sweep is meaningless:

> **STATUS 2026-06-13 (autonomous build) -- LANDING LEDGER:**
> - **M1 + M4** `9b2294b` -- no-runtime-fallback gate + VRAM fail-closed (12 tests).
> - **M2 + M3 + M5** `0ab55bc` -- sweep `--acceptance`: empty/required-engine exit
>   code + Wan enable-flag / OTR_TEST_MODE / --exclude preflight (17 tests).
> - **M6** `ec91a3c` -- `assert_usable` preflights UNET + umt5 CLIP + VAE (8 tests).
> - **M7** `f71edaa` -- render_clip ffprobe-PROVES the silent-clip contract (13 tests).
> - **S1 + S5** `dfe9ab5` -- wan_i2v vram_estimate 14500 + real wan2.2-i2v asset id.
> - **S7 + S10** `f3a529f` -- per-shot/seed init staging + Pillow-required fail-loud.
> - **S3 / S6 / S8** -- folded into this doc (MoE eyeball risk, sweep-harness reality,
>   the exact acceptance invocation below).
>
> **M8 + S2 -- LANDED 2026-06-14 (`bcbe05a`).** The `wan_ti2v` engine is built: its 5B core
> node class (`Wan22ImageToVideoLatent`) was captured from a live `/object_info` first; M8 raises
> `EngineUnusable` when the resolved VAE basename is empty or is the 2.1 VAE; S2 added the
> `medium`/8000 CAPABILITIES row (registry-consistency invariant holds -- the row + the registered
> engine landed together). Validated live (5B bare-graph smoke PASS). **STILL OPEN:** **M9** (CS-3
> sequential residency) + **S4** (leg isolation/reclaim) + **S9** (post-reset verify) are live-GPU
> proof obligations -- partial evidence only. A full multi-leg `--acceptance` GREEN exit is gated on
> the slow wan-music-bed leg (run it attended/selectively), not on missing code.
>
> **S8 -- exact acceptance invocation** (ComfyUI venv python; live server on :8000;
> `OTR_TEST_MODE` UNSET; `OTR_ENABLE_WAN_I2V=1` (+ `OTR_ENABLE_WAN_TI2V=1` once built);
> Wan UNET + umt5 CLIP + VAE on disk):
> `python scripts\otr_coverage_sweep.py --acceptance --only wan`
> (`--only wan` matches the `sweep_<slot>_wan_i2v` / `_wan_ti2v` legs; drop `--only`
> for the full visual set. `--exclude` of a core Wan engine is REFUSED in acceptance.)

- **M1 -- the sweep is BLIND to silent fallback.** `otr_coverage_sweep.py` runs every
  leg with `expect_engine=""`, which `_otr_soak_capstone.py:464` treats as
  informational (no assert), so a leg that silently falls back to `still_kenburns`
  scores PASS (this is exactly CS-1). FIX (NOT per-leg `expect_engine=engine` -- that
  false-fails a slot that gets 0 beats at 30w): in acceptance mode assert ZERO runtime
  fallbacks across the whole trace -- fail any shot where `final_engine != attempts[0]`
  -- with an opt-out only for known-degrade experiment legs. (Verify the trace field is
  a stable requested-id, not an alias.)
- **M2 -- the sweep returns GREEN on EMPTY results.** `return 0 if passed ==
  len(results)` makes `0 == 0` pass when `--only`/`--exclude` filter everything out or
  `wan_ti2v` is unregistered. FIX: fail on empty results; for GATE-A, fail unless BOTH
  `wan_i2v` AND `wan_ti2v` are present in results with PASS.
- **M3 -- acceptance preflight (closes the R2 trap).** `availability()` is pure
  profile-fit and never reads `OTR_ENABLE_WAN_I2V`, so a gated-off Wan leg enumerates
  "run", `assert_usable` fails it closed, it falls back, and (pre-M1) passes -- the same
  `gated_by_flag` mechanism that floored HuMo-1.7B (commit 5231d31). FIX: the acceptance
  run preflights `OTR_ENABLE_WAN_I2V=1` (+ future `OTR_ENABLE_WAN_TI2V=1`) and the model
  files, and FORBIDS `--exclude` of the core Wan engines.
- **M4 -- the V-3 VRAM gate fails OPEN.** `driver_peak = int(report.get("vram_peak_mb")
  or -1)` then fails only if `> ceiling`, so a missing/0/negative measurement (`-1`)
  PASSES -- the `<=14.5GB` invariant can read GREEN with no measurement. FIX: fail
  closed when `vram_peak_mb` is absent or `<= 0`.
- **M5 -- the Wan render-phase VRAM assert is skipped under `OTR_TEST_MODE`** (`if not
  os.environ.get("OTR_TEST_MODE"): ... assert_peak_within_ceiling`). Phase-2 acceptance
  MUST run with `OTR_TEST_MODE` UNSET; the harness preflight fails if it is set.
- **M6 -- `assert_usable` preflights only the ckpt.** The umt5 CLIP + the VAE are
  required graph loaders. FIX: verify UNET+CLIP+VAE present + matching the sha/license
  manifest before any forward (offline / no-runtime-fetch invariant).
- **M7 -- the Phase-2 clip contract is SELF-DECLARED, not asserted.** `_clip_from_raw`
  hardcodes `has_audio=False`/h264/yuv420p/bt709/fps25 in a dict; the soak only inspects
  the obs final's audio. FIX: ffprobe the emitted silent Wan mp4 (or a real-path test)
  to PROVE those fields before mux.
- **M8 -- `wan_ti2v` VAE fail-closed.** `eng_wan_i2v` defaults the VAE to
  `wan_2.1_vae.safetensors`; the 5B needs the Wan2.2 VAE. Give `wan_ti2v` its own VAE
  env; raise `EngineUnusable` if the resolved VAE basename is empty OR equals the 2.1
  basename. Do NOT inherit `_loader_names()` unchanged.
- **M9 -- CS-3 = sequential residency (see section 5).** Prove per-beat peak <= 14.5GB +
  the inter-beat reclaim drains the prior heavy engine (incl. the retained Wan unet
  patcher) before the next loads; that is the real risk, not co-residency. Unblocks
  Phase-2 scoping.

SHOULD-FIX: **S1** raise `CAPABILITIES["wan_i2v"].vram_estimate_mb` 14000 -> the measured
Phase-2 peak (or 14500); the 14499 smoke figure was WITHOUT `free_after_use`, which is
load-bearing -- document it as mandatory. **S2** add a concrete `wan_ti2v` CAPABILITIES
row (`medium` / ~8000 DRAFT -- the 5B VAE decode may push higher, verify on the 8GB
probe / `["wan2.2-ti2v-5b"]`). **S3** surface the single-expert (low-noise) MoE motion
risk on the eyeball gate -- Path B two-expert HIGH/LOW handoff is the mitigation if the
"real camera motion" bar fails. **S4** sweep leg isolation -- reclaim/restart between
legs that swap heavy engines (one resident server, no teardown -> residue corrupts the
next leg's peak; ties to CS-2 + the CLAUDE.md reset directive). **S5** fix the stale
`["wan2.1-i2v"]` label -> the real Wan2.2 I2V asset id. **S6** point item-4's writer-LLM
+ voice-variation leg-sets at their real harness (run_combo_matrix.py?) or mark TODO --
`otr_coverage_sweep.py` enumerates ONLY the visual-engine set today. **S7** stage the
init image under a shot/seed/uuid name (`otr_wan_init_WxH.png` is fixed -> same-dim
renders overwrite; low risk, driver is sequential). **S8** spell `scripts/otr_coverage_sweep.py`
+ the exact `--only` Wan substring + required env. **S9** Phase-2 post-reset verify
(PID/start-time changed, Sage NOT active, `OTR_TEST_MODE` unset, env visible) before
submitting. **S10** `_materialize_init_image`: require Pillow + fail loud (the no-Pillow
path leans on `WanImageToVideo` cover-resize -- N9 risk).

CUTS (panel consensus -- do NOT over-engineer): no broad VRAM-budget-aware scheduler to
close CS-3 (the reclaim assertion suffices; wait for a measured failure); do NOT subclass
all of `WanI2VEngine` for `wan_ti2v` (share only pure dims/aspect/materialize/canonicalize
helpers; keep loaders + node candidates + graph SEPARATE); keep the GATE-A sweep ADDITIVE,
not a visual x writer x voice cross-product. VERIFY-AT-BUILD: capture TI2V-5B's exact core
node class from `/object_info` before coding (the "5B latent node" is underspecified).

---

## 4B. WAN PHASE 1 -- DONE (pointer)

Phase 1 PROVEN: a real Wan b-roll clip (wan_i2v 14B fp8 in-process, ~14.5 GB; commits `2fbc2f3` +
`8eaf058`). Phase 2 is the ACTIVE next step (section 1); remaining Wan work = sections 4 + 4A. The
overnight-soak companion findings (R1 GPU-proven, R2 harness fix unexercised, R3 landed) live in git +
`scripts/FABLE_SOAK_REVIEW.md`; the not-done remainder (R2 verify) is in section 5.

---

## 5. OPEN TICKETS

- **SCHEMA-ADHERENCE (2026-06-25 -- LEVER-1 LOAD-BEARING SHIPPED; see the CURRENT STEP block at the top):**
  LEVER 1 tolerance (`pass04_plan.md` C0-C6, refined by the nested-fork + c4-scope roundtables) SHIPPED in 2
  green chunks `516644eb` (C0+C1+C2+C5+C6: `apply_field_aliases`/`__otr_field_aliases__` before-validator +
  `validate_tolerant_data` core; proven nested Opus `normalize_length` failure fixed) + `d4ca6cd4` (C3:
  JSON-syntax-only structural rung). C4 (schema-in-repair) DEFERRED -- proven failure already fixed, would test
  dead code; OPTIONAL `_build_schema_snippet`-shim recipe ready in c4-scope/, reopen on a real captured drift.
  LEVER 2 binary lane `docs/2026-06-25-schema-adherence/binary/pass01_plan.md` still GATED on **G1** (offline
  abstain-residual count -- the cheap first move; may DROP the lane) + **G2** (byte-identity of abstain).
  **G1 DONE -> Lever 2 (binary lane) DROPPED (genuine residual ~0; `binary/G1_RESULTS.md`); SCHEMA-ADHERENCE
  SPRINT COMPLETE.** NO workflow-JSON change.
- **LOOK-QA BUGS (NEW 2026-06-14 eve — operator look-QA pass; all in `BUG_LOG_2026-06.md`):**
  - **BUG-408 default MUSIC sounds non-musical (SA3).** **IMPLEMENTED 2026-06-14 (`3a4f71d`).** Path B:
    SA3-shaped prompt + real negative + per-cue `seconds_start` within a 30s `seconds_total` context (latent
    stays `dur` → length+determinism unchanged), env-overridable sampler knobs. Suite 4261/0. **OPERATOR-GATED:**
    restart Desktop, A/B listen (tune `OTR_SA3_CFG/STEPS/CONTEXT_S`), then RE-BASELINE the `test_audio_byte_identical`
    golden (intended music-bytes change). Plan: `docs/2026-06-14-sa3-music-improvement/roundtable/pass01_plan.md`.
  - **BUG-409 title card scrambles the WHOLE window** — **FIXED 2026-06-14 (`9e0b658`).** New
    `_title_reveal_progress` resolves the reveal in the first ~40% of the window then holds solid (env
    `OTR_TITLE_REVEAL_FRACTION`); close card stays bounded to the main video (no credits overlap). Suite 4259/0.
  - **BUG-410 closing ROLLING CREDITS** — **CLOSED 2026-06-14 (operator-verified on flux_still).** Credits
    scroll over the held last clip to the end again (silent after the theme). Detail in `BUG_LOG_2026-06.md`
    + `docs/2026-06-14-credits-tail-fix/`. (HuMo backdrop not yet eyeballed — low risk, engine-agnostic path.)
  - **BUG-411 flux BOOKEND / image lost its "lush" cinematic tint (NEXT — HANDOFF FOCUS).** The 6/5 image
    pipeline (`visual/batch_flux_render.py` + `flux_prompt_extractor`) was WHOLLY REWRITTEN into
    `_otr_image_engines/flux_gen1.py` + `otr_meta_brief_image_prompt.py` (pure insertions after `e4cb3ac`).
    Model/steps/cfg/sampler IDENTICAL (flux1-dev-fp8, 20, 1.0, euler/simple), but the rewrite DROPPED the look
    levers: **(1) FluxGuidance = 3.5** (flux_gen1 has NO FluxGuidance node — biggest factor), **(2) the
    cinematic style suffix** `"cinematic, 35mm film, anamorphic lens, volumetric lighting, heavy vignette,
    muted color grade, sharp focus"`, **(3) the radio broadcast-distress suffix** + retrofuturistic radio
    fallback (`35mm film grain ... dim amber and cyan rim lighting`), **(4) bookend seed 4242**, **(5) portrait
    style line**. 6/5 workflow widgets inspected + confirmed (no other hidden hardcodes). FIX = restore those in
    the new pipeline (FluxGuidance node @ ~3.5 + the suffixes + seed). Full forensic in `BUG_LOG_2026-06.md`
    BUG-411. CODER-READY (the next window's task).
  These are GATE-A look-QA items (operator-gated track), parallel to the engine forward order — NOT a
  reordering of section 3.

- **IMPROVED 3D INPUT -- BLOCKED on a PATH DECISION (operator look-QA 2026-06-14; GROUNDED this session).**
  The 3D rotating output looked like a "blobby plaster-of-paris" block. GROUNDING (checked logs + disk):
  the ONLY 3D system actually installed/active is **HunyuanWorld-Mirror / WorldMirror 2.0**
  (`custom_nodes/ComfyUI-HunyuanWorld-Mirror`, model `C:\ComfyUI-Models\WorldMirror-V2\HY-WorldMirror-2.0`)
  -- NOT Blender, NOT OTR's deferred character_3d/TripoSG. Recent episode ledgers used NO 3D engine; the
  server log only shows HWM loading (no episode rendered 3D). **WorldMirror is a MULTI-VIEW SCENE
  reconstructor** (image SEQUENCE -> point cloud / Gaussian splat): per its docs 1 frame = "depth/normals
  only"; good 3D needs **8-24 FEATURE-RICH frames, orbital/forward parallax, well-lit, 50-70% overlap**. A
  single flat/low-feature image -> the plaster blob. So the earlier "clean / object-free single image"
  idea is the OPPOSITE of what WorldMirror needs -- object-free helps only single-image-to-OBJECT-mesh
  tools (TripoSG / Hunyuan3D-2 / TRELLIS), which are NOT installed. **OPEN DECISION (operator, next window)
  -- the prompt strategy is opposite per path:** (A) WorldMirror scene/world -> improved input = GENERATE
  an orbit/multi-view sequence + rich-textured scene prompts (NOT a plain bg); or (B) single-image ->
  object mesh -> INSTALL TripoSG/Hunyuan3D-2 + clean isolated-subject prompts. Do NOT draft the improved
  3D prompts (and do NOT wire character_3d) until the path is picked. A roundtable can harden the chosen
  path's prompt set. (Note: the live roundtable launcher stalled this session -- the panel blocked with no
  output; budget a retry or a smaller panel.) Example obs finals to eyeball the EPISODE look (these do NOT
  contain 3D): `output\otr\obs\signal_lost_plunging_depths_20260614_185229_silent_procgen_blended_final.mp4`
  (a pre-fix render -- shows the closing FREEZE + the skinny flux_still portrait, both now fixed at HEAD).
- **HuMo full-frame TEST (operator 2026-06-14 -- future experiment, NOT now).** Operator wants to
  eventually SEE HuMo rendered full-frame (not the 480x832 portrait pillarbox). For now portrait stays
  HuMo's REQUIREMENT -- BUG-407 shipped "full frame everything EXCEPT HuMo". Future: a HuMo full-frame /
  16:9 smoke to evaluate whether the talking-head holds at a wider aspect before changing the default.
- **Look-QA the 5 overnight 120-word episodes (NEW 2026-06-14).** The default-lane soak ran 5/5 SUCCESS
  (LTX + humo_1.7B); the episode outputs (`...\output\otr\episodes` + obs finals) are NOT yet eyeballed.
  Check audio sync, burned captions, procgen scopes/credits, character look. This is the operator's
  "analyze the soak" item; verdicts in `scripts/_otr_120word_soak_summary.json`.
- **Wan WEBM EYEBALL -- DONE 2026-06-14 (operator + Claude live smoke).** RESULT: **Wan i2v 14B
  DRIFTS** -- holds the input still ~1 frame then re-interprets the scene into its own subject (a
  generic tube close-up). NOT fixable by easy input knobs: cfg3.5->2.0 + a locked-tripod prompt STILL
  drifts; cfg1.5 COLLAPSES into incoherent abstraction. **LTX (2B v0.9) HOLDS** the composition with
  subtle motion in all 3 modes tested (ksampler 30-step, distilled, AND 1216x704 hires -- hires
  answers the "low-res" note). => **RECOMMEND: Wan i2v 14B -> BACK-BURNER for the music/announcer
  OPENER role** (keep selectable; revisit only with Path B two-expert handoff, GO_FORWARD 4A S3); LTX
  stays the opener engine; **PROMOTE LTX-REGR (below) to the active thread.** Evidence:
  `docs/2026-06-14-wan-ti2v/EYEBALL_FINDINGS.md` + `eyeball_frames/COMPARISON_montage.png`. AWAITS
  operator confirm on the re-prioritization.

- **Non-Wan soak = ENOUGH (operator call 2026-06-13).** The non-Wan permutation coverage sweep
  (`--strict-fallback --exclude wan/latentsync/triposg`) has run sufficiently; do NOT keep grinding it.
  The non-lip-sync FLOORS (`still_kenburns` / `still_parallax` / Ken-Burns / `station_card`) render fine
  and are acceptable for the 8GB tier, but they are NOT the target experience -- the operator wants real
  audio-driven lip-sync, not a still with motion. Focus the remaining runway on **getting the Wan lane
  bug-free** (section 1 + 4 + 4A). A new sweep, if ever needed, should add `--exclude-engine humo` (the
  exact-match flag added `ca10b63`: skips the 14B `humo` that TIMES OUT per CS-4, KEEPS `humo_1.7B`).
- **LTX-REGR — SUPERSEDED 2026-06-15 by the LTX 22B-GGUF splice** (`docs/2026-06-15-ltx-splice/SPLICE_PLAN.md`).
  LTX-REGR's recommended fix was to bake the **2B** v0_9 recipe into `eng_ltx_video.py`; the splice instead swaps
  `LtxVideoEngine` to the **22B GGUF** mini recipe (verified-working). **Do NOT do the 2B bake.** Original entry kept
  below for history only:
  **LTX-REGR (operator 2026-06-13; PROMOTED to active 2026-06-14 pending operator confirm)** -- LTX
  clips no longer animate like the **2026-05-30..06-05** era (motion lost / too static). `BUG-LOCAL-113b`
  (`8115c72`: ksampler 30-step euler cfg3.0 as the LTX default, distilled 8-step = the
  `OTR_LTX_SAMPLER=distilled` rollback) was the prior fix, but the operator STILL sees the regression.
  **2026-06-14 eyeball update:** the Wan-vs-LTX smoke proved LTX HOLDS the still composition cleanly
  (good) -- so the open question is narrowed to **MOTION AMOUNT** (5/30-6/5 read as more dynamic; the
  current ksampler/distilled holds are subtle). With Wan i2v back-burnered for openers, this is the
  recommended NEXT thread. Probe = an LTX **--strength / sampler-mode / step-count / cfg / frame-cap**
  sweep (otr_ltx_motion_smoke.py exposes all of them; --strength is the prime motion lever, 1.0=max
  freedom) against the 5/30 baseline + the 169 decode floor from look-QA round 5.
  **FORENSIC DONE 2026-06-14 (BUG-LOCAL-412, `BUG_LOG_2026-06.md`):** diffed the GOOD 5/09 `l001` + 5/28
  `b001` LTX bookends vs the current engine (ledgers + the DELETED `batch_ltx_render.py` @ `70d379b^` + the
  old workflow JSON widgets). The good recipe = **v0_9 / sampler `euler_cfg_pp` / 8 distilled steps / cfg
  1.0 / 832×480 / I2V strength 0.75 / `loop_via_reverse` boomerang / audio-length**; the cleanbreak
  `70d379b` DELETED that node and `eng_ltx_video.py` shipped **ksampler / `euler` / 30-step / cfg 3.0 /
  768×512-or-1472×832 / strength 1.0 / 169-cap / no boomerang** (the code comment itself admits `euler_cfg_pp`
  is the documented dynamic-motion sampler but the default was left on `euler`). The old WORKFLOW JSON baked
  in NOTHING but seed/method/cap — the recipe lived in code. **ENV-TESTABLE A/B FIRST (no code change):** at
  832×480 set `OTR_LTX_SAMPLER=distilled` + `OTR_LTX_SAMPLER_NAME=euler_cfg_pp` + `OTR_LTX_I2V_STRENGTH=0.75`,
  re-render a bookend, A/B vs `l001`/`b001`; if it matches, bake those defaults + the boomerang + audio-length
  back into `eng_ltx_video.py` (coder chunk; no JSON change implicated).
- **CS-1** -- the latentsync legs must show latentsync IN THE TRACE (a prior "PASS" was fallback-only);
  re-verify in the sweep. (Non-Wan -> deprioritized per the operator's "non-Wan soak = enough" call.)
- **CS-2** -- machine NVML pins ~16 GB per leg vs the 14.5 ceiling while driver-phase attribution reads
  ~3 GB; needs phase attribution (the 1.7B leg's 10,305 MB render-phase peak is a partial answer).
- **CS-3 (reframed 2026-06-13)** -- NOT a co-residency budget: wan_i2v (~14GB) +
  humo_1.7B (~7GB) cannot co-reside under 14.5GB by construction, so they must render
  SEQUENTIALLY. The real proof obligation = per-beat NVML peak <= 14.5GB AND the
  inter-beat reclaim (`wrapper_bridge.reclaim_idle_models`, BUG-291) fully drains the
  prior heavy engine -- incl. the retained Wan unet patcher -- before the next beat
  loads. A mixed Wan+HuMo episode is the test. This UNBLOCKS Phase-2 scoping (no
  open "decision" needed). See section 4A M9.
- **CS-4-open** (deprioritized) -- targeted post-encode umt5-TE detach for the OPT-IN 14B HuMo lane so it
  fits 14.5 GB. The default char tier is `humo_1.7B` (`955f134`); the 14B is opt-in.
- **R2 verify** -- confirm `humo_1.7B` renders native char beats at 70w with its enable flag ON (the
  soak floored it only via `gated_by_flag`); answered by the item-4 re-run.
- **README "what to expect per video model" (operator 2026-06-14).** Once the opener model bake-off
  settles (interactive render bench artifact `otr-render-bench` + `docs/2026-06-14-wan-ti2v/
  EYEBALL_FINDINGS.md`), add a user-facing "what to expect from each video engine" section to the
  README (newbie audience -- folds into S6/closing): Wan i2v 14B = drifts off the still (b-roll only,
  NOT openers); LTX = holds composition + subtle motion (opener default); TI2V-5B = 8GB tier, lower-res.
  Source the verdicts from the operator's bench ratings (export button).
- **Ship defaults (release)** -- proposed: announcer + character = `flux_still`, music = `visualizer`
  (selectable: station_card, still_parallax, abstract — `ltx_orbit` ripped 2026-06-15 in the LTX splice Phase 0). Keep HuMo/latentsync/3D
  selectable-not-default until verified. Operator eyeballs 2-3 finals/slot.
- **Harness polish** (minor) -- output-tree resolver should prefer the live server's `OTR_OUTPUT_DIR`
  (fail LOUD on mismatch); run the OH-3 janitor sweep at server boot; widen the heartbeat cadence.
- **OH-4** -- the 14-entry / ~8.2 GB live->attic migration STAGED, awaits operator "go OH-4"
  (`docs/2026-06-11-output-tree-consolidation/OUTPUT_TREE_CONTRACT.md`).
- **0-E Phase B** -- tickets E-1..E-7, gated on the sweep GO file; coder-window ready.
- **Operator gates** -- ComfyUI Desktop relaunch (look-QA), fresh-render acceptance, whiny-voice P0 matrix
  + reel, S-3D-0 green light, `v2.0-alpha-stable` tag decision. (latentsync demos REMOVED 2026-06-21.)

---

## 6. RUNWAY (remaining sprints to "done")

"Done" = platform wired into real episodes (real per-beat video + byte-identical mux + legacy procgen
path gone) + all video models verified live + the first 1-2 3D models rendering. ~s2-s9:
S-3D-0 spike -> T2b keystone GO/NO-GO (timeboxed ~1wk) -> T4 driver + LOOK gate -> W7 production wiring +
soak ("v1-usable") -> S3-S6 distribution. SHORTCUT FORK: S-3D-0 or keystone NO-GO -> `character_3d`
defers (HuMo-2D stays) -> collapses to ~2-3 sprints (0-E + closing). Done splits: "v1-usable" (one
engine, one real episode) vs "B-parity ship" (>=2 engines bind at SHIP).

---

## 7. POINTERS (evidence + tooling -- not plans)

- Tracker dashboard: `otr-build-tracker` artifact (OneDrive\Documents\Claude\Artifacts).
- Soak review (R1/R2/R3 detail + roundtable): `scripts/FABLE_SOAK_REVIEW.md`.
- Wan/sweep hardening (grounded QA + 3-model roundtable judgment, 2026-06-13):
  `docs/2026-06-13-goforward-wan-hardening/` (pass00 plan+QA, pass01/pass01b raw
  reviews, pass01_judgment.md).
- Overnight sweep: `scripts/otr_overnight_sweep_launch.ps1`; tasks `otr-overnight-sweep` +
  `otr-sweep-monitor`; digest `scripts/sweep_monitor_digest.md`; GO file `scripts/_otr_0e_gpu_go.txt`.
- 3D spec (forward item 5): `docs/2026-06-09-3d-toolkit/3D_TOOLKIT_PLAN.md`.
- Switchable spec (items 3 + 6): `docs/2026-06-10-switchable-workflow-architecture__decision-and-plan.md`.
- Bug log (this repo): ACTIVE = `BUG_LOG_2026-06.md` (epoch BUG-LOCAL-400+, started 2026-06-14);
  ARCHIVE = `BUG_LOG.md` (BUG-LOCAL-001..~305, through 2026-06-12, reference only).
- Bug Bible: `C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide` (`BUG_BIBLE.yaml` +
  `tests/bug_bible_regression.py`; run cd-to-root + venv python + RELATIVE path).
- Full smoke harness: `scripts/queue_smoke.py` + `scripts/otr_api.py`.

---

## 8. PARKED -- not now

Story-spine; story-pipeline; broader audio stack; MuseTalk; RTXUpscale;
switchable S3-S6 (closing phase, AFTER 3D); 3D GPU lanes until S-3D-0 + the operator green light.
(**LTX-AV audio-input lane MOVED OUT of PARKED 2026-06-17** -- operator revived it as the CURRENT STEP
(section 1): M1+M3 are shipped, the remaining work is recipe-align + M4 GPU smoke. It already uses the good
Q3_K_M GGUF unet; do NOT rebuild from scratch.)

(**STORY-ENGINE quality roundtable (2026-06-21) -- PARKED side campaign.** A 4-pass live roundtable converged a
sprint-ready plan for 8 content-only story-engine fixes (length tail / costly-choice binding / ending-aware outro
/ gender-pronouns / speech register / narration hygiene / arc-shape variety; F9 reorder + F10 anti-repeat list
deferred). Docs: `docs/2026-06-21-allnight-864-frontier/` -- `SPRINT_READY_PLAN.md` + `STORY_ENGINE_KICKOFF.md` +
`roundtable/pass0{1,2,3,4}_judgment.md`. All content-only inside the FIXED ledger, ZERO workflow-JSON edits
(verified vs the real consumers). NOT active -- the visual fixes (section 1) + the forward order win. Resume only
on an explicit operator green light.)
