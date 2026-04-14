# OTR v2.0 Bug Log

Active bug log for the v2.0 Visual Sidecar build.
Every bug gets logged the moment it is found. Entries are never deleted.

---

### BUG-LOCAL-001: v2_preview.py placeholder nodes flagged as output nodes [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** Bug Bible regression BUG-01.02 fails: "Output nodes without folder_paths usage: v2_preview.py"
- **Cause:** All four v2 placeholder nodes (CharacterForge, ScenePainter, VisualCompositor, ProductionBus) had `OUTPUT_NODE = True` despite not writing any files to disk. They only return in-memory tensors or strings.
- **Fix:** Removed `OUTPUT_NODE = True` from all four placeholder classes. These nodes are data-flow nodes, not file-output nodes.
- **Verify:** `python -m pytest bug_bible_regression.py -v --pack-dir .` passes BUG-01.02
- **Tags:** widget-drift, registration, bug-bible

### BUG-LOCAL-002: Stale TestWorkflowJSONLite references deleted workflow [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0 | **Bible candidate:** no
- **Symptom:** 8 test errors + 1 failure in test_core.py: FileNotFoundError for otr_scifi_16gb_lite.json
- **Cause:** The lite workflow was removed in commit 44cbdec ("chore: remove lite workflow") but the TestWorkflowJSONLite test class was not cleaned up.
- **Fix:** Removed the entire TestWorkflowJSONLite class from test_core.py with a comment noting the removal reason.
- **Verify:** `pytest tests/test_core.py -v` shows 83 passed, 0 errors
- **Tags:** stale-test, cleanup

### BUG-LOCAL-003: Widget-value drift in workflow-to-API prompt conversion [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** ComfyUI rejects POST /prompt with HTTP 400. Node #15 (BatchAudioGenGenerator) gets `episode_seed: 3.0, model_id: 3.0` instead of `episode_seed: "", model_id: "facebook/audiogen-medium"`.
- **Cause:** Workflow-to-API conversion mapped `widgets_values` positionally to ALL widget-capable params. But ComfyUI's workflow JSON excludes linked inputs from `widgets_values`, so linked params (script_json, production_plan_json) consumed slots 0-1, shifting all downstream values by 2 positions.
- **Fix:** Filter widget-capable params to only UNLINKED ones before positional mapping. `unlinked_widgets = [p for p in widget_capable if p not in linked]`.
- **Verify:** Regenerate debug_prompt.json and check node #15 values are correct.
- **Tags:** widget-drift, api, baseline-capture

### BUG-LOCAL-004: v2 placeholder nodes cause API 400 from missing required inputs [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0 | **Bible candidate:** no
- **Symptom:** ComfyUI rejects prompt because CharacterForge, ScenePainter require MODEL/CLIP/VAE inputs that are not connected in the audio-only workflow.
- **Fix:** Strip v2 placeholder nodes from the API prompt before submission. They are not part of the audio pipeline.
- **Verify:** Prompt submits successfully with only audio-pipeline nodes + PreviewAudio capture node.
- **Tags:** api, baseline-capture, placeholder

### BUG-LOCAL-005: Emoji vs [EMOJI] placeholder mismatch in dropdown values [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** ComfyUI 400 error: `runtime_preset: '\ud83d\udcfb standard (12 min)' not in ['[EMOJI] standard (12 min)', ...]`
- **Cause:** Workflow JSON stored real Unicode emoji (e.g. U+1F4FB) in dropdown values, but the running ComfyUI node code uses `[EMOJI]` as a text placeholder. The API prompt validation does exact string matching.
- **Fix:** Added `_dropdown_text_match()` that strips leading emoji or `[TAG]` prefixes before comparing, and remaps to the schema's expected value.
- **Verify:** Regenerate debug_prompt.json and check node #1 runtime_preset matches schema.
- **Tags:** encoding, widget-drift, api, baseline-capture

### BUG-LOCAL-007: PARSE_FATAL when target_length=short (3 acts) + runtime_preset=[FAST] quick (5 min) [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** ScriptWriter generates 4 scenes, 48 lines, but 0 parseable dialogue lines. PARSE_FATAL fires, execution aborts. Episode never reaches TTS stage.
- **Cause:** `short (3 acts)` compresses the arc so aggressively that Mistral-Nemo produces narration/outline-style content instead of `CHARACTER: dialogue` format. The parser finds no dialogue tags and hard-aborts.
- **Fix:** Keep `[FAST] quick (5 min)` runtime target but use `medium (5 acts)` for `target_length`. Five acts requires 45 minimum dialogue lines, forcing proper dialogue structure. workflow updated: `target_length` = `medium (5 acts)`.
- **Verify:** Run `test_audio_byte_identical.py --capture-baseline` and confirm ScriptWriter log shows `dialogue lines > 0`.
- **Tags:** script-writer, parse-fatal, episode-length

### BUG-LOCAL-008: Node 15 (OTR_BatchAudioGenGenerator) widget drift recurrence [FIXED-WORKAROUND]
- **Date:** 2026-04-12 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** API prompt has `episode_seed: 3.0, model_id: 3.0` (both float) instead of `episode_seed: "", model_id: "facebook/audiogen-medium"`. Positional mapping shifted by 2.
- **Cause:** ComfyUI `/object_info` schema returns `optional` params in a different order than `INPUT_TYPES` defines them. The `_workflow_to_api_prompt` positional mapper uses schema order for `params_with_wv_slot`, but `widgets_values` are stored in `INPUT_TYPES` order. When the schema omits or reorders optional params, the wv indices are wrong. Root cause: schema ordering vs INPUT_TYPES ordering mismatch for this node specifically. `debug_audiogen_schema.json` is dumped on each baseline run for diagnosis.
- **Fix (workaround):** `_fix_known_widget_drift()` in `_run_baseline.py` hardcodes correct values for `OTR_BatchAudioGenGenerator` after prompt conversion. Real fix requires aligning schema ordering — see `debug_audiogen_schema.json` output.
- **Verify:** Check `debug_prompt.json` after run — node #15 should show `episode_seed: "", model_id: "facebook/audiogen-medium"`.
- **Tags:** widget-drift, api, baseline-capture, schema-ordering

### BUG-LOCAL-009: Preset/target_length mismatch causes wrong dialogue line targets
- **Date:** 2026-04-12 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** `runtime_preset=[FAST] quick (5 min)` paired with `target_length=medium (5 acts)` tells the LLM "Target 8-minute runtime, MINIMUM 45 dialogue lines" even though the actual runtime is 5 minutes. LLM overshoots or gets confused by conflicting length signals.
- **Cause:** `length_instruction` dict was hardcoded per `target_length` with fixed runtime targets and dialogue line minimums that did not scale with `target_minutes`. Also, the 1-min test preset was prone to PARSE_FATAL (see BUG-LOCAL-007).
- **Fix:** (1) Removed 1-min test preset, set minimum to 3 minutes. (2) Added `_safe_length_for_preset` auto-clamp: each runtime_preset forces the safe `target_length` (e.g. quick->medium, long->long 7-8 acts, epic->epic 10+ acts). (3) Made `length_instruction` dynamic: dialogue line floor = `max(18, target_minutes * 8)`, act label from `target_length`, runtime target from actual `target_minutes`.
- **Verify:** Run with each preset. Check runtime log for "PREFLIGHT: Auto-clamped target_length" when mismatch detected. Verify `length_instruction` shows correct minute target and proportional line count.
- **Tags:** preset, length-scaling, parse-fatal-prevention

### BUG-LOCAL-011: Obsidian profile string mismatch - all guardrails dead [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** Selecting "Obsidian (UNSTABLE/4GB)" in ComfyUI had zero effect - no one-shot mode, no token cap, no runtime clamp. Obsidian users got full Pro behavior, then OOM on 4GB cards.
- **Cause:** Code checked for `"Obsidian (Low VRAM/Fast)"` (6 locations in story_orchestrator.py) but INPUT_TYPES dropdown value is `"Obsidian (UNSTABLE/4GB)"`. String never matched. Likely a rename in the UI that was never propagated to the runtime code.
- **Fix:** Replace all 6 occurrences of `"Obsidian (Low VRAM/Fast)"` with `"Obsidian (UNSTABLE/4GB)"` to match INPUT_TYPES. Caught by new `test_dropdown_guardrails.py` regression suite (59 tests).
- **Verify:** Run `pytest tests/test_dropdown_guardrails.py -v` — TestGuardrails::test_obsidian_disables_multipass and test_obsidian_caps_runtime must pass.
- **Tags:** string-mismatch, obsidian, guardrails, dead-code

### BUG-LOCAL-010: Full pre-flight guardrail sweep [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0 | **Bible candidate:** no
- **Symptom:** Multiple unguarded parameter combos could cause silent failures or PARSE_FATAL: (a) 2 characters + 7+ acts = dialogue starvation, (b) 8 characters + 5 min = too many voices for runtime, (c) "maximum chaos" + chunked outline pushes temp above model max, (d) Obsidian + 20 min = 2500 token cap truncates 60% of script, (e) `news_headlines` widget has zero effect, (f) `temperature` widget silently overridden by `creativity`.
- **Cause:** Pre-flight validation only checked 1-min edge case. No guardrails for character count vs episode length, no profile-aware runtime cap, no temp ceiling in outline gen.
- **Fix:** (a) Clamp chars to 4 if <=5 min, to 3 if <=3 min. Floor chars to 3 if >=7 acts. (b) Obsidian profile caps target_minutes at 10. (c) Outline gen temp no longer adds +0.1 when already >= 1.0. (d) Deprecated tooltips on news_headlines and temperature widgets.
- **Verify:** AST parse clean. Check PREFLIGHT log lines for each clamp scenario.
- **Tags:** guardrails, pre-flight, parameter-validation

### BUG-LOCAL-006: Converted widget alignment in widgets_values mapping [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0 | **Bible candidate:** yes
- **Symptom:** Node #2 (Gemma4Director) gets `tts_engine: 0.4` (should be dropdown string). Widget values shifted by 1.
- **Cause:** The BUG-LOCAL-003 fix skipped ALL linked inputs from positional mapping, but linked inputs with a `"widget"` flag in the workflow JSON ("converted widgets") still keep their slot in `widgets_values`. Only linked inputs WITHOUT the widget flag should be skipped.
- **Fix:** Check `inp.get("widget")` on each linked input. Include converted widgets in the positional mapping, skip non-widget links.
- **Verify:** Regenerate debug_prompt.json and check node #2 values: temperature=0.4, tts_engine='bark (standard 8GB)', vintage_intensity='subtle'.
- **Tags:** widget-drift, api, baseline-capture

### BUG-LOCAL-012: Episode duration significantly undershoots target_minutes
- **Date:** 2026-04-12 | **Phase:** 0-1 | **Bible candidate:** yes
- **Symptom:** Test run "The Last Frequency" with target_minutes=3, 2 characters, 3 acts, Standard profile generated a 2-minute episode (vs 3-minute target). ~33% duration shortfall.
- **Cause:** Dialogue scaling formula enforces **line count minimum** (floor = max(18, target_minutes * 8)) but not **dialogue density**. For 3 min with 2 chars: floor = 24 lines total (12 per char). LLM hit the minimum and stopped, natural pacing resulted in ~1 min audio runtime. The 41 total generated lines (39 dialogue + 2 ANNOUNCER) meet the **count** requirement but not the **duration** requirement.
- **Fix:** (Phase 0.5) Relabel target_minutes dropdown to reflect realistic output range: "Target 3 (actual 2-3 min)" instead of exact promise. No code change — UI expectation mismatch only.
- **Verify:** Added UI warning labels to INPUT_TYPES. User sees "2-3 min" as the expected range when they select "3 min".
- **Tags:** duration, dialogue-scaling, episode-length, ui-expectation

### BUG-LOCAL-014: Maximum chaos creativity produces unparseable dialogue format
- **Date:** 2026-04-12 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** 8 chars + 3 min + maximum chaos: ScriptWriter generated 1065 tokens across 3 scenes but parser found 0 dialogue lines. LLM used `*NAME*(emotion): dialogue` format with single asterisks. Characters detected as garbage: "ENVSIRENS WAIL", "OPENING THEME". Episode proceeded with near-silent output.
- **Cause:** (1) Maximum chaos (temperature=1.35) pushes Mistral-Nemo into non-standard formatting. (2) Parser Pass 1 regex only accepted 0 or 2 asterisks around names, not 1. (3) Permissive fallback matched structural tags as "characters", so dialogue_count > 0 and PARSE_FATAL never fired.
- **Fix:** Four-layer defense: (a) Clamped maximum chaos temp from 1.35 to 0.95, wild & rough from 1.1 to 0.92 - LLM stays creative but follows structural rules. (b) Hardened Pass 1 regex to accept 0-2 asterisks and filter structural tag names. (c) Added Format Normalizer pass (Creative-to-Strict): same LLM, low temperature, rewrites any dialogue format into strict Canonical 1.0 BEFORE parser runs. (d) Structural name blocklist prevents ENV/SFX/MUSIC tags from being misidentified as characters.
- **Verify:** Run 8 chars + 3 min + maximum chaos again. Check runtime log for "CREATIVITY maximum chaos - temp=0.95" and "FORMAT_NORM: Success". Verify dialogue_count > 0 in ScriptWriter DONE line.
- **Tags:** parse-fatal, creativity, format-drift, dialogue-parser, temperature, phase-0.5

### BUG-LOCAL-015: System cascades to Director crash on 0-dialogue script
- **Date:** 2026-04-12 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** When ScriptWriter produced a script with garbage "dialogue" (structural tags misidentified as characters), the run continued to Director and Bark. Director attempted to generate voice assignments for "ENVSIRENS WAIL" and "OPENING THEME". Bark generated near-silent audio.
- **Cause:** Permissive fallback matched structural tags, returning dialogue_count > 0, which bypassed PARSE_FATAL. No quality gate between ScriptWriter output and downstream nodes.
- **Fix:** (a) Structural name blocklist in permissive fallback prevents false positive matches. (b) Format Normalizer pass gives the parser clean input. (c) PARSE_FATAL still fires as last resort if both normalizer and fallback fail.
- **Verify:** Same test as BUG-014. If normalizer fails gracefully, PARSE_FATAL should fire with clear error instead of silent garbage propagation.
- **Tags:** cascade-failure, parse-fatal, quality-gate, phase-0.5

### BUG-LOCAL-013: UI doesn't warn user when guardrails clamp parameters [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0.5 | **Bible candidate:** no
- **Symptom:** User selected "8 characters + 3 minutes" but the guardrail silently clamped to 3 chars in the logs. No warning visible in ComfyUI UI — user had no idea the setting was changed.
- **Cause:** Guardrail warnings were logged internally but not returned to the UI. ComfyUI only shows what the nodes return.
- **Fix:** ~~(v1) Prepend as comment block to script_json~~ Caused BUG-016. **(v2)** Guardrail clamp warnings logged to otr_runtime.log as `GUARDRAIL_UI:` lines alongside existing `PREFLIGHT:` lines. script_json stays pure valid JSON.
- **Verify:** ✅ VALIDATED 2026-04-12 19:50:39: Test ran with 8 chars + 3 min. PREFLIGHT fired and logged clamps. ✅ REVISED: BUG-016 fix confirmed guardrail_warnings log via `_runtime_log()` without corrupting JSON.
- **Tags:** ui, guardrails, feedback, phase-0.5

### PHASE 0.5 QA SUMMARY [VALIDATED 2026-04-12]
- **All fixes deployed and tested together**
- **Test case:** 8 characters + 3 minutes + maximum chaos creativity
- **Run 1 result (old code, temp=1.35):**
  - PREFLIGHT guardrails fired: clamped 8→3 chars, disabled act breaks
  - FORMAT_NORM activated but reported "No improvement" (both counts 0)
  - Parser recovered 6 dialogue lines via permissive fallback
  - QA_REPAIR auto-injected ANNOUNCER bookends (generic canned text)
  - **KokoroAnnouncer crashed (BUG-016):** JSON comment prefix broke `json.loads()`
- **Post-crash fixes applied:**
  - BUG-016: ✅ Guardrail warnings now log-only, script_json stays pure JSON
  - BUG-014 (updated): ✅ Temperature clamped: maximum chaos 1.35→0.95, wild & rough 1.1→0.92
  - BUG-017: ✅ Story-aware ANNOUNCER via LLM micro-pass replaces canned placeholders
  - BUG-018: ✅ Test suite updated for runtime_preset removal
- **Test suite status:**
  - test_core.py: 83 passed, 21 skipped
  - test_dropdown_guardrails.py: 133 passed, 0 failed
  - AST parse: ✅ Clean
- **Code changes validated:**
  - No BOM: ✅ Confirmed
  - Obsidian strings: ✅ All 8 updated correctly
  - runtime_preset dropdown: ✅ Removed entirely, target_minutes is now sole control
  - Workflow JSON: ✅ Updated to remove runtime_preset widget index
- **Next phase:** Reload ComfyUI and retest with all Phase 0.5 changes live

### BUG-LOCAL-016: Guardrail warning comments break downstream JSON parsing [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** Node #13 (OTR_KokoroAnnouncer) crashes with `json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)`. The `script_json` string starts with `// GUARDRAIL WARNINGS:` instead of valid JSON.
- **Cause:** BUG-013 fix prepended `// comment` lines to the `script_json` output from ScriptWriter. JSON does not support comments. KokoroAnnouncer (and any other downstream node) calls `json.loads(script_json)` which fails immediately on the `//` prefix.
- **Fix:** Remove comment-prefix injection from script_json. Guardrail warnings are already logged via PREFLIGHT log lines visible in otr_runtime.log. Instead, store warnings in a separate `guardrail_warnings` string and log them, but keep script_json as pure valid JSON.
- **Verify:** Run 8 chars + 3 min + maximum chaos. KokoroAnnouncer should receive valid JSON and not crash. Check otr_runtime.log for PREFLIGHT warnings still present.
- **Tags:** json-parse, guardrails, downstream-crash, phase-0.5

### BUG-LOCAL-017: QA_REPAIR ANNOUNCER bookends are generic canned text [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0.5 | **Bible candidate:** no
- **Symptom:** When the LLM fails to generate ANNOUNCER bookends (e.g. at high creativity), QA_REPAIR auto-injects canned placeholder text: "Welcome to Signal Lost. Tonight's broadcast takes us into the unknown." and "And so the transmission ends. This has been Signal Lost. Stay safe." These are completely generic with no story context - no date, no location, no character names, no science hook.
- **Cause:** QA_REPAIR in `_parse_script()` had no access to episode context (title, news, characters). It could only insert hardcoded strings.
- **Fix:** (a) QA_REPAIR now flags missing ANNOUNCER with `__NEEDS_LLM_OPENING/CLOSING__` sentinels. (b) New `_generate_announcer_bookends()` method does a quick LLM micro-pass (temp 0.4, max 200 tokens, ~3-5s) at the `write_script` call site where full context is available. The LLM reads episode_title, genre, news headline, character names, and a dialogue preview to generate story-specific bookends. (c) Falls back to canned text if LLM call fails.
- **Verify:** Run any episode where ANNOUNCER is missing from LLM output. Check otr_runtime.log for "ANNOUNCER_GEN: Generated opening (N chars) + closing (N chars)". ANNOUNCER lines should reference actual story content.
- **Tags:** announcer, qa-repair, llm-micro-pass, story-context, phase-0.5

### BUG-LOCAL-019: FORMAT_NORM times out generating runaway filler tokens [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** FORMAT_NORM LLM pass exceeds 120s wall-clock budget. The LLM generates 1700+ tokens at ~14 tok/s but dialogue count plateaus at 22 lines around token 700 — the remaining 1000+ tokens are stage-direction prose, scene descriptions, and padding that the streaming counter never recognizes as dialogue. Fires on every 8-min target run tested so far. Pipeline falls back to original script text and relies on permissive 2B-fallback parser.
- **Cause:** FORMAT_NORM has no early-stop heuristic. The `max_new_tokens` budget is too generous relative to the input script length, and the LLM drifts into narrative prose after exhausting the dialogue content. The 120s timeout is a blunt wall-clock kill, not a quality gate.
- **Fix:** (1) Token budget reduced from `min(2048, len//3+500)` to `min(1024, len//4)` — prevents runaway filler. (2) Timeout reduced from 120s to 75s. For a 10k-char script: old budget=2048 tokens, new budget=1024.
- **Verify:** Run 8-min target with maximum chaos. FORMAT_NORM should complete in <75s or bail faster, not generate 1700+ filler tokens.
- **Tags:** format-norm, timeout, runaway-tokens, early-stop, phase-0.5

### BUG-LOCAL-020: Episode duration significantly undershoots target_minutes (systemic)
- **Date:** 2026-04-12 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** 8-minute target produces 3.7-minute output (46%). 3-minute target produces ~2-minute output (67%). The LLM prompt says "MINIMUM 64 dialogue lines" but only 25 are generated. At maximum chaos, Mistral-Nemo ignores word/line count instructions. The word-per-minute estimator used 130 wpm but Bark TTS actually paces at ~67 wpm, making estimates doubly wrong.
- **Cause:** Three compounding issues: (a) LLM instructions used minutes (not measurable) instead of words (countable). (b) No post-generation enforcement — the pipeline accepted whatever the LLM produced. (c) Duration estimator used 130 wpm instead of the measured 67 wpm.
- **Fix:** Word-count enforcement system with raw-text-first pipeline reorder: (1) Convert `target_minutes` to `target_words` using measured Bark rate of 67 wpm. (2) LLM prompt now asks for specific word count ("write at least 536 words of dialogue") instead of minutes. (3) Post-generation pipeline reordered to: **WORD_EXTEND → ANNOUNCER → FORMAT_NORM → Parse**. All four stages operate on raw text before a single final parse. (4) `_extend_script_dialogue()` counts dialogue words via regex on raw text, generates additional dialogue lines via LLM if under 70% target, appends to raw text. (5) ANNOUNCER bookends generated on raw text (sees full extended script). (6) FORMAT_NORM normalizes the complete text (original + extensions + announcer) in one pass. (7) Parser runs once on clean text. (8) Duration estimator fixed to use 67 wpm.
- **Verify:** Run 8-min target. Check runtime log for `WORD_ENFORCEMENT:` lines showing word count vs target, and `WORD_EXTEND:` if extension fires. Final output should be closer to 8 min than 3.7 min.
- **Tags:** duration, word-count, enforcement, extension-pass, bark-wpm, pipeline-reorder, phase-0.5

### BUG-LOCAL-019: Gender assignment inversion in LLMDirector procedural cast
- **Date:** 2026-04-13 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** LLMDirector JSON plan specifies correct genders (e.g. COMMANDER_MC: "Male, 50s") but procedural cast assigns the opposite gender voice preset. In soak Run 77 (maximum chaos, post_apocalyptic, 2100w): COMMANDER_MC (male) got FLETCHER HUDSON (female, 60s), TARKON_TS (male) got GULLIVER KAPOOR (female, 50s), PALMER_PR (female) got RASHIDA CORBEN (male, 20s). All 4 non-announcer characters had inverted gender assignments.
- **Cause:** Pending investigation. The Director JSON `gender_hints` parse returned 0 hints (`Parsed 0 gender hints from script: {}`), causing procedural cast to ignore the Director's own voice_assignments and assign randomly from the pool. Likely the gender hint regex does not match the maximum-chaos script format.
- **Fix:** pending
- **Verify:** pending
- **Tags:** gender, llm-director, procedural-cast, maximum-chaos, soak

### BUG-LOCAL-020: Name squish and character drift under maximum chaos
- **Date:** 2026-04-13 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** Character "Nemo Sirikit" appears in LLMDirector plan as `NEIMO_NEMEO_SIRIKIT` (hallucinated spelling). In the script body the same character appears as `NS`, `NEMO`, and `NEMO SIRIKIT`. BatchBark cannot map `NS` or `NEMO` to the Director plan, so both fall back to `v2/en_speaker_9` (already assigned to COMMANDER_MC). Result: 3 characters share one voice, 2 voices unused.
- **Cause:** Maximum chaos creativity (highest temperature/top_p) causes the LLM to hallucinate variant spellings of character names. The Director name-matching is exact-match only and cannot reconcile `NEIMO_NEMEO_SIRIKIT` with `NEMO` or `NS`.
- **Fix:** pending
- **Verify:** pending
- **Tags:** name-squish, character-drift, maximum-chaos, batch-bark, voice-collapse, soak

### BUG-LOCAL-021: Act count exceeds target_length ceiling under maximum chaos
- **Date:** 2026-04-13 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** Config specified `medium (5 acts)` but the generated script contains 8 acts (ACT 1 through ACT 8). The act-by-act generation loop did not enforce the target ceiling.
- **Cause:** Pending investigation. The act-by-act chunked generation may not hard-cap the number of iterations, relying on the LLM to self-terminate. Under maximum chaos temperature the LLM keeps generating new acts instead of concluding.
- **Fix:** pending
- **Verify:** pending
- **Tags:** act-count, target-length, maximum-chaos, chunked-generation, soak

### BUG-LOCAL-028: BUG-LOCAL-027 shipfix regressed Node 15: linked converted widgets eat widgets_values slots [FIXED]
- **Date:** 2026-04-14 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** HTTP 400 value_not_in_list on Node 15 (OTR_BatchAudioGenGenerator) immediately after deploying 2b52ebe. ComfyUI rejected the API submission because `model_id` received `3.0` instead of `"facebook/audiogen-medium"`. Same positional-shift class of bug that BUG-LOCAL-003 and BUG-LOCAL-027 addressed, surfaced on a different param by the new mapper.
- **Cause:** The rewritten `_workflow_to_api_prompt` in 2b52ebe kept a "consume-and-skip" branch for linked params that carried a `"widget": {"name": ...}` metadata block (a converted widget). The reasoning was that a converted widget still reserves its widgets_values slot. In practice ComfyUI's web UI does NOT keep widgets_values slots for inputs that have been converted to sockets — it saves slots only for inputs still displayed as widgets. Node 15's `script_json` and `production_plan_json` are linked + carry the `widget` metadata, but have no slots in `widgets_values`. The mapper consumed `wv[0]` (episode_seed's slot) and `wv[1]` (model_id's slot) for nothing, shifting every subsequent value down by two. `model_id` ended up with `wv[3] = 3.0`.
- **Fix:** `_workflow_to_api_prompt` now treats any linked param as consuming zero widgets_values slots, regardless of whether the input has converted-widget metadata. The walk is: start with linked names already populated from the link map, then iterate declared params and only the widget-backed + not-linked ones consume a slot. This is the original BUG-LOCAL-003 contract, restored.
- **Verify:** `pytest tests/test_widget_drift_guard.py::TestLinkedConvertedWidgetSlots -v` (4 tests) locks down the Node 15 case explicitly: `model_id` must stay a string, `episode_seed` must be empty, `guidance_scale` + `default_duration` must land as 3.0 each, and the link tables for `script_json` / `production_plan_json` must survive intact.
- **Tags:** widget-drift, socket-only, linked-converted-widget, api, hotfix, bug-bible

### BUG-LOCAL-027: Widget-drift in soak API mapper emits project_state as string, drops optimization_profile [FIXED]
- **Date:** 2026-04-14 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** Wire-level capture of `soak_target_api.json` showed node #1 (Gemma4ScriptWriter) and node #2 (Gemma4Director) were being submitted with `"project_state": "Standard"` and `"project_state": "Pro (Ultra Quality)"` respectively, while `optimization_profile` was missing entirely. ProjectState loader silently failed to parse the string as a dict, producing an empty preamble; optimization_profile silently defaulted to "Standard" (only by luck matching the intended widget default). Ghost runs (empty CAST / SCENE ARC / FULL SCRIPT) traced back to this corrupted input.
- **Cause:** `_workflow_to_api_prompt` in `scripts/soak_operator.py` walked `widgets_values` positionally against `INPUT_TYPES` order without filtering socket-only params (types like `PROJECT_STATE` that have no widget in the UI). Because `project_state` was declared between `arc_enhancer` and `optimization_profile` in the optional block, every widget after it shifted up by one slot. `optimization_profile`'s value landed in the `project_state` key, and the true `optimization_profile` key was never emitted at all. Same bug class as BUG-LOCAL-003; the fix there addressed linked inputs but not socket-only inputs.
- **Fix:** Added `_is_widget_backed(spec)` helper that returns True for `STRING/INT/FLOAT/BOOLEAN` primitives and for dropdowns (list-typed specs), and False for socket-only custom types. Mapper now walks params in declaration order but only widget-backed params consume a `widgets_values` slot. Socket-only params are either filled via the link map or omitted from `inputs`. Defense in depth: moved `project_state` to the LAST entry in `optional` for both `Gemma4ScriptWriter` (`nodes/story_orchestrator.py`:2484-2534) and `LLMDirector` (`nodes/story_orchestrator.py`:6649-6670) so any future mapper regression cannot shift widget slots. Also stripped the `"3"/"3.0"/3/3.0` back-compat hack from `BatchAudioGenGenerator.model_id` (`nodes/batch_audiogen_generator.py`:102) — scar tissue from widget drift that's no longer needed. Added `API_PAYLOAD` and `DRIFT_DETECTED` instrumentation lines in the soak operator just before the POST. Tightened `_RE_SCENE_MARKER` to numeric-only and added a `_RE_SCENE_TERMINATOR` for `=== SCENE FINAL ===` (kills BUG-LOCAL-026 confound).
- **Verify:** `pytest tests/test_widget_drift_guard.py -v` (18 tests) passes. Assertions lock down: (1) `project_state` is never emitted as a string, (2) `optimization_profile` always survives with its correct string value, (3) mapper stays correct even if `project_state` is interleaved before `optimization_profile` in INPUT_TYPES, (4) scene regex no longer captures `FINAL` as a scene number. On next live soak run, runtime log must show `API_PAYLOAD node=1 ... optimization_profile='Standard' project_state=None` and no `DRIFT_DETECTED` lines.
- **Tags:** widget-drift, socket-only, api, soak, ghost-run, input-types, regression-test

### BUG-LOCAL-026: Scene regex matches "FINAL" as a scene number, inflates scene counts [FIXED]
- **Date:** 2026-04-14 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** `SCENE_TRACK: count=6 | tokens=['1', '2', 'FINAL', '3', '4', 'FINAL']`. FORMAT_NORM's `has_scenes` signal was fooled by pseudo-scenes, and the chunked FORMAT_NORM split treated `SCENE FINAL` blocks as real scene boundaries.
- **Cause:** `_RE_SCENE_MARKER` used `\S+?` for the scene token, which matched any non-whitespace including the literal "FINAL" that the creative LLM emits as a closing-scene marker.
- **Fix:** Tightened `_RE_SCENE_MARKER` to `===\s*SCENE\s+(\d+)(?:\s*:\s*[^=]*?)?\s*===` (numeric only). Added separate `_RE_SCENE_TERMINATOR` for `=== SCENE FINAL ===`. `_scene_inventory` returns numeric tokens followed by `'END'` when a terminator is present, so downstream counts are honest.
- **Verify:** `pytest tests/test_widget_drift_guard.py::TestSceneRegex -v` (5 tests) passes.
- **Tags:** regex, scene-marker, parser, soak

### BUG-LOCAL-024: FORMAT_NORM ghost-run bypass and silent bailout on long scripts [FIXED]
- **Date:** 2026-04-13 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** Runs 011 and 012 under maximum chaos produced scripts with no CAST section, no `=== SCENE N ===` markers, and TITLE_STUCK downstream. Runtime log showed FORMAT_NORM was skipped entirely, despite the script clearly being malformed. On longer scripts FORMAT_NORM also logged `Output too short - keeping original` and silently bailed out.
- **Cause:** Two independent blindspots in `_normalize_script_format`: (1) pre-flight skip heuristic only checked dialogue line count (`voice_tag_count >= 3 OR canonical_count >= 5`), so a ghost run with voice tags but no CAST or scene markers bypassed normalization entirely; (2) token budget was capped at 1024 regardless of script length, so a reformatted output of a 10k-char script could not fit in the budget and triggered the `< 0.3 * input` bailout.
- **Fix:** (1) Tightened skip heuristic to require ALL THREE signals present: `has_dialogue AND has_scenes AND has_cast` (scene marker count >= 1 AND unique character count >= 2). Missing any signal forces FORMAT_NORM to run. (2) Added `_normalize_chunked` following the `_grammarian_chunked` pattern: split by `=== SCENE N ===` markers, reformat each scene independently with a full per-chunk 1024-token budget, 75s per-chunk timeout, reassemble with 80% dialogue-count floor. Single-pass retained for scripts with <=50 dialogue lines or <2 scenes. Also hoisted class constant `_FORMAT_NORM_NON_CHARS` so canonical-name regex excludes SCENE/ACT/SFX/ENV/NARRATOR etc. from the skip count.
- **Verify:** Run soak with a creative-pass script that lacks CAST + scene markers. Runtime log should show `FORMAT_NORM: Running (dialogue=X+YV, scenes=0, cast=0) - missing: scenes cast`, followed by single-pass or chunked flow. For 50+ line scripts, log should show `FORMAT_NORM: Chunked mode` with per-chunk progress.
- **Tags:** format-norm, ghost-run, skip-heuristic, chunked-generation, token-budget, maximum-chaos, soak

### BUG-LOCAL-023: Grammarian timeout on long scripts (60+ dialogue lines) [FIXED]
- **Date:** 2026-04-13 | **Phase:** 0.5 | **Bible candidate:** yes
- **Symptom:** `GRAMMARIAN: Failed (Grammarian exceeded 75s) - keeping original` on a 67-line space opera script. Token budget of 2048 at ~15 tok/s needs ~136s, but timeout was 75s. Grammarian silently falls back to original script, losing all grammar polish.
- **Cause:** Single-pass grammarian with fixed 75s timeout cannot handle scripts with 50+ dialogue lines. The prompt + full script exceeds what the LLM can process within the timeout window.
- **Fix:** Implemented chunked grammarian in `_grammarian_pass()`. Scripts with >50 dialogue lines are split by `=== SCENE N ===` markers, each scene polished independently (90s timeout per chunk, 1024 token budget), then reassembled. Falls back to 40-line raw chunking if no scene markers exist. Single-pass timeout increased from 75s to 150s as safety net. Each chunk has its own dialogue-line safety check; failed chunks keep original text without blocking the rest.
- **Verify:** Run soak with 60+ line episode config. Runtime log should show `GRAMMARIAN: Chunked mode` followed by per-chunk progress, ending with `GRAMMARIAN: Chunked complete`.
- **Tags:** grammarian, timeout, chunked-generation, long-scripts, soak

### BUG-LOCAL-018: test_dropdown_guardrails.py references removed runtime_preset [FIXED]
- **Date:** 2026-04-12 | **Phase:** 0.5 | **Bible candidate:** no
- **Symptom:** `pytest tests/test_dropdown_guardrails.py` fails with `KeyError: 'runtime_preset'` during collection. 1 additional NameError at runtime for `RUNTIME_PRESETS` variable.
- **Cause:** runtime_preset was removed from INPUT_TYPES but the test file still extracted it from `_REQUIRED` and used it in 12 test locations.
- **Fix:** Removed all runtime_preset references from tests. Replaced `runtime_preset="[FAST] quick (5 min)"` with `target_minutes=5`, etc. Added `test_runtime_preset_removed` assertion alongside existing dead-param checks. Removed obsolete `test_no_1min_test_preset` and `test_runtime_presets_produce_different_target_minutes`.
- **Verify:** `pytest tests/test_dropdown_guardrails.py -v` shows 133 passed, 0 failed.
- **Tags:** test-suite, runtime-preset, cleanup, phase-0.5

