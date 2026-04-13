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
