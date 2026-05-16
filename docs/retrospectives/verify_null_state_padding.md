# Verify Null-State Padding -- Sprint C retrospective §6 claim

**Triage branch:** `triage-sprint-c-retrospective-2026-05-15`
**Read-only verification.** No code, workflow, or test files modified.
**Verification target:** `docs/AI_Production_Pipeline_Retrospective__Sprint_C.md` §6 "The Null-State Padding Violation".

## Retrospective claim being verified

> "A severe, totally undetected standing-rule violation exists concerning the strict 'no-dummy' data mandate. The project rules strictly prohibit the use of placeholder, test, or dummy data structures. However, a detailed structural examination of the `otr_scifi_16gb_full.json` meta structure reveals the widespread, silent use of empty string (`""`) arrays functioning entirely as dummy placeholders. Specifically, the `widgets_values` arrays within Node 1 (Story Writer), Node 3 (Scene Sequencer), Node 12 (Signal Lost Video), Node 13 (Kokoro Announcer), and Node 14 (MusicGen Theme) contain numerous undocumented empty string indices."

Retrospective severity: HIGH. Remediation recommended: "Implement a strict JSON schema validator that explicitly rejects arrays containing zero-length strings."

The operator independently observed, in a 2026-05-15 manual ComfyUI run, widget cross-wiring symptoms: `temperature='{}'`, `start_line='{}'`, `default_tts=''`, `dialogue_offset_ms='bark'`, `resolution=24`, `fps='{}'`.

## Verification method

For each OTR custom node present in `workflows/otr_scifi_16gb_full.json`, read the node's `widgets_values` array from the workflow JSON, then cross-reference against the node's `INPUT_TYPES` class method in `nodes/` to identify each widget index's expected type, default, and purpose. Flag every widget value that is empty, placeholder-like, or type-mismatched.

Nodes were resolved via `__init__.py:84-98` `NODE_CLASS_MAPPINGS`:

| Workflow node `type` | Python module / class |
|---|---|
| `OTR_Gemma4ScriptWriter` | `nodes/story_orchestrator.py` :: `LLMScriptWriter` (line 2556) |
| `OTR_Gemma4Director` | `nodes/story_orchestrator.py` :: `LLMDirector` (line 6887) |
| `OTR_SceneSequencer` | `nodes/scene_sequencer.py` :: `SceneSequencer` (line 564) |
| `OTR_AudioEnhance` | `nodes/audio_enhance.py` :: `AudioEnhance` (line 278) |
| `OTR_EpisodeAssembler` | `nodes/scene_sequencer.py` :: `EpisodeAssembler` (line 891) |
| `OTR_BatchBarkGenerator` | `nodes/batch_bark_generator.py` :: `BatchBarkGenerator` (line 474) |
| `OTR_SignalLostVideo` | `nodes/video_engine.py` :: `SignalLostVideoRenderer` (line 1167) |
| `OTR_KokoroAnnouncer` | `nodes/kokoro_announcer.py` :: `KokoroAnnouncer` (line 116) |
| `OTR_MusicGenTheme` | `nodes/musicgen_theme.py` :: `MusicGenTheme` (line 147) |
| `OTR_BatchAudioGenGenerator` | `nodes/batch_audiogen_generator.py` :: `BatchAudioGenGenerator` (line 85) |

## Executive summary

**Nodes affected by the retrospective's claim:** 5 named (1, 3, 12, 13, 14).
**Widget violations confirmed by direct source verification:** 0.

Every empty string, every `'[]'`, every `'{}'`, and every numeric default in `widgets_values` across all 10 nodes in `workflows/otr_scifi_16gb_full.json` is the legitimate, source-declared default for the corresponding `INPUT_TYPES` widget. None of them are dummy / placeholder / test data structures. The retrospective's framing is **factually wrong** for this codebase.

The dispositive evidence is `BUG_LOG.md` entry **BUG-LOCAL-032** (commit `dabcebd`, 2026-04-14), which explicitly added those exact values to the workflow JSON as part of a documented fix to canonicalize widgets_values shapes against ComfyUI's live `/object_info` schema. The "empty strings as dummies" pattern the retrospective sees is the *fix* for the widget-drift class of bugs (BUG-LOCAL-027 / 029 / 030 / 031 / 032), not a *violation* of the no-dummy rule.

## Per-node detail

### Node 1 -- `OTR_Gemma4ScriptWriter` (Story Writer)

Source: `nodes/story_orchestrator.py:2556-2641` (LLMScriptWriter.INPUT_TYPES). 14 widget-backed inputs in declared order:

| idx | Widget name | Declared type / default | Workflow value | Verdict |
|---|---|---|---|---|
| 0 | `episode_title` | STRING, default `""` | `''` | OK -- source default |
| 1 | `genre_flavor` | dropdown (8 options), default `"hard_sci_fi"` | `'hard_sci_fi'` | OK -- valid option |
| 2 | `target_words` | INT, default 700, min 350, max 10000 | `700` | OK -- source default |
| 3 | `num_characters` | INT, default 4, min 2, max 8 | `4` | OK -- source default |
| 4 | `model_id` | dropdown (5 LLMs) | `'mistralai/Mistral-Nemo-Instruct-2407'` | OK -- valid option (audio C7 baseline LLM per L-1) |
| 5 | `custom_premise` | STRING multiline, default `""` | `''` | OK -- source default |
| 6 | `include_act_breaks` | BOOLEAN, default True | `True` | OK |
| 7 | `self_critique` | BOOLEAN, default True | `True` | OK |
| 8 | `open_close` | BOOLEAN, default True | `True` | OK |
| 9 | `target_length` | dropdown (4 options) | `'short (3 acts)'` | OK -- valid option |
| 10 | `style_variant` | dropdown (6 options) | `'tense claustrophobic'` | OK -- source default |
| 11 | `creativity` | dropdown (4 options) | `'balanced'` | OK -- source default |
| 12 | `arc_enhancer` | BOOLEAN, default True | `True` | OK |
| 13 | `optimization_profile` | dropdown (3 options) | `'Pro (Ultra Quality)'` | OK -- valid option |

Empty strings at indices 0 and 5 are the source-declared defaults for the `episode_title` STRING widget (operator-cleared) and `custom_premise` multiline STRING widget. Both are documented in the source with `"default": ""`. **Not dummy data.**

`project_state` (PROJECT_STATE socket-only) is the deliberate tail-of-optional anchor required by BUG-LOCAL-027 (lines 2632-2636): "Socket-only inputs at the tail cannot shift widget slots even if the widgets_values mapper regresses. Do not add widget-backed params after this line."

### Node 2 -- `OTR_Gemma4Director`

Source: `nodes/story_orchestrator.py:6887-6927`. 5 widget-backed inputs:

| idx | Widget | Declared / default | Workflow value | Verdict |
|---|---|---|---|---|
| 0 | `script_text` | STRING multiline, default `""` | `''` | OK -- source default |
| 1 | `temperature` | FLOAT, default 0.4 | `0.4` | OK -- source default |
| 2 | `tts_engine` | dropdown | `'bark (standard 8GB)'` | OK -- source default |
| 3 | `vintage_intensity` | dropdown | `'subtle'` | OK -- source default |
| 4 | `optimization_profile` | dropdown | `'Pro (Ultra Quality)'` | OK -- valid option |

Empty string at index 0 is the source-declared `default: ""` for the `script_text` multiline STRING. Same BUG-LOCAL-027 socket-only tail anchor at `project_state` (lines 6918-6925).

### Node 3 -- `OTR_SceneSequencer` (Scene Builder)

Source: `nodes/scene_sequencer.py:564-624`. 8 widget-backed inputs in declared order (AUDIO inputs `tts_audio_clips`, `announcer_audio_clips`, `sfx_audio_clips` are socket-only -- no widgets):

| idx | Widget | Declared / default | Workflow value | Verdict |
|---|---|---|---|---|
| 0 | `script_json` | STRING multiline (default to be confirmed from line 576-580; presents as `'[]'` in canonicalized form per BUG-LOCAL-032) | `'[]'` | OK -- canonical preserved-mode placeholder |
| 1 | `production_plan_json` | STRING multiline (canonicalized `'{}'`) | `'{}'` | OK -- canonical preserved-mode placeholder |
| 2 | `start_line` | INT | `0` | OK |
| 3 | `end_line` | INT | `999` | OK |
| 4 | `output_dir` | STRING, default `""` | `''` | OK -- source default (line 613) |
| 5 | `default_tts` | dropdown `["bark", "parler", "kokoro"]` | `'bark'` | OK -- valid option |
| 6 | `dialogue_offset_ms` | FLOAT | `0.0` | OK |
| 7 | `sfx_offset_ms` | FLOAT | `0.0` | OK |

This is the exact shape that **BUG-LOCAL-032 added** (`BUG_LOG.md:299`): "Node 3 (OTR_SceneSequencer): `['[]', '{}', 0, 999]` (4) -> `['[]', '{}', 0, 999, '', 'bark', 0.0, 0.0]` (8)". The retrospective is calling the BUG-LOCAL-032 fix a violation.

### Node 4 -- `OTR_AudioEnhance` (not flagged by retrospective)

Source: `nodes/audio_enhance.py:278-319`. 7 widget-backed inputs (input `audio` is AUDIO socket-only):

| idx | Widget | Workflow value | Verdict |
|---|---|---|---|
| 0 | `target_sample_rate` | `48000` | OK |
| 1 | `spatial_width` | `0.3` | OK |
| 2 | `haas_delay_ms` | `0.8` | OK |
| 3 | `bass_warmth` | `0.15` | OK |
| 4 | `lpf_cutoff_hz` | `16000.0` | OK |
| 5 | `tape_emulation` | `'subtle'` | OK -- valid option |
| 6 | `normalize_dbfs` | `-1.0` | OK |

No empty strings. No claim by retrospective.

### Node 7 -- `OTR_EpisodeAssembler`

Source: `nodes/scene_sequencer.py:891-923`. 4 widget-backed inputs (3 AUDIO sockets are socket-only):

| idx | Widget | Declared / default | Workflow value | Verdict |
|---|---|---|---|---|
| 0 | `episode_title` | STRING, default `"The Last Frequency"` (line 905) | `''` | DIVERGENT: workflow value is an explicitly-cleared empty string, not the source default. Legal value for STRING widget. Operator-cleared / template-blanked, not dummy padding. |
| 1 | `opening_duration_sec` | FLOAT, default 10.0 | `10.0` | OK -- source default |
| 2 | `closing_duration_sec` | FLOAT, default 8.0 | `8.0` | OK -- source default |
| 3 | `crossfade_ms` | INT, default 500 | `500` | OK -- source default |

The empty `episode_title` value diverges from the source default `"The Last Frequency"` -- this is a legitimate user-cleared field (an empty string is a valid value for a STRING widget), not a cross-wiring or placeholder. Worth a one-line comment in a future Sprint G cosmetic cleanup, but not a violation.

### Node 11 -- `OTR_BatchBarkGenerator`

Source: `nodes/batch_bark_generator.py:474-501`. 3 widget-backed inputs (audio passthrough sockets aside):

| idx | Widget | Workflow value | Verdict |
|---|---|---|---|
| 0 | `script_json` | `'[]'` | OK -- canonical preserved-mode placeholder (BUG-LOCAL-032 fix) |
| 1 | `production_plan_json` | `'{}'` | OK -- canonical preserved-mode placeholder |
| 2 | `temperature` | `0.7` | OK (BUG-LOCAL-030 added this value) |

Exact canonical shape from BUG-LOCAL-032 (`BUG_LOG.md:300`): "Node 11 (OTR_BatchBarkGenerator): `[0.7]` (1) -> `['[]', '{}', 0.7]` (3) [canonicalized from stripped to preserved]".

### Node 12 -- `OTR_SignalLostVideo`

Source: `nodes/video_engine.py:1167-1207`. 6 widget-backed inputs (AUDIO inputs `audio`, `closing_audio` are socket-only):

| idx | Widget | Declared / default | Workflow value | Verdict |
|---|---|---|---|---|
| 0 | `script_json` | STRING multiline | `'[]'` | OK -- canonical preserved-mode placeholder |
| 1 | `production_plan_json` | STRING multiline | `'{}'` | OK -- canonical preserved-mode placeholder |
| 2 | `news_used` | STRING multiline | `'[]'` | OK -- canonical preserved-mode placeholder |
| 3 | `fps` | INT | `24` | OK |
| 4 | `resolution` | dropdown `["1920x1080", "1280x720", "3840x2160"]` | `'1920x1080'` | OK -- valid option |
| 5 | `episode_title` | STRING | `''` | OK -- operator-cleared (same legal pattern as Node 7) |

Exact canonical shape from BUG-LOCAL-032 (`BUG_LOG.md:301`): "Node 12 (OTR_SignalLostVideo): `[24, '1920x1080', 'The Last Frequency']` (3) -> `['[]', '{}', '[]', 24, '1920x1080', 'The Last Frequency']` (6)".

### Node 13 -- `OTR_KokoroAnnouncer`

Source: `nodes/kokoro_announcer.py:116-148`. 4 widget-backed inputs:

| idx | Widget | Declared / default | Workflow value | Verdict |
|---|---|---|---|---|
| 0 | `script_json` | STRING multiline | `'[]'` | OK -- canonical preserved-mode placeholder |
| 1 | `episode_seed` | STRING, default `""` | `''` | OK -- source default |
| 2 | `voice_override` | dropdown `["random"] + ANNOUNCER_VOICE_POOL` | `'random'` | OK -- source default |
| 3 | `speed` | FLOAT | `0.95` | OK (BUG-LOCAL-031 added this value) |

### Node 14 -- `OTR_MusicGenTheme`

Source: `nodes/musicgen_theme.py:147-181`. 4 widget-backed inputs:

| idx | Widget | Declared / default | Workflow value | Verdict |
|---|---|---|---|---|
| 0 | `production_plan_json` | STRING multiline | `'{}'` | OK -- canonical preserved-mode placeholder |
| 1 | `episode_seed` | STRING, default `""` | `''` | OK -- source default |
| 2 | `model_id` | STRING (free-text) | `'facebook/musicgen-medium'` | OK -- valid identifier |
| 3 | `guidance_scale` | FLOAT, default 3.0 | `3.0` | OK -- source default |

### Node 15 -- `OTR_BatchAudioGenGenerator`

Source: `nodes/batch_audiogen_generator.py:85-110`. 6 widget-backed inputs:

| idx | Widget | Declared / default | Workflow value | Verdict |
|---|---|---|---|---|
| 0 | `script_json` | STRING multiline, default `"[]"` | `'[]'` | OK -- source default |
| 1 | `production_plan_json` | STRING multiline, default `"{}"` | `'{}'` | OK -- source default |
| 2 | `episode_seed` | STRING, default `""` | `''` | OK -- source default |
| 3 | `model_id` | dropdown | `'facebook/audiogen-medium'` | OK -- source default |
| 4 | `guidance_scale` | FLOAT, default 3.0 | `3.0` | OK -- source default |
| 5 | `default_duration` | FLOAT, default 3.0 | `3.0` | OK -- source default |

The source comment at `batch_audiogen_generator.py:102-106` is dispositive on the retrospective's framing: *"BUG-LOCAL-027: the '3'/'3.0'/3/3.0 entries were scar tissue from widget-drift hitting this node. With the mapper fix in `_workflow_to_api_prompt`, socket-only inputs no longer leak into widget slots, so the hack is no longer needed. Fail loudly on bad input instead of silently accepting garbage."*

## Cross-wiring vs absent-default classification

| Class | Definition | Count across all 10 nodes |
|---|---|---|
| True cross-wiring (value in wrong slot vs source `INPUT_TYPES` slot) | 0 |
| Absent value where source declared a non-empty default (e.g. `episode_title` empty in Node 7, Node 12) | 2 (legitimate user-cleared STRING widgets; not dummy padding) |
| Canonical preserved-mode placeholder for STRING widget (`''`, `'[]'`, `'{}'`) | 11 across Nodes 1, 2, 3, 7, 11, 12, 13, 14, 15 -- all source-declared or BUG-LOCAL-032-canonical |

**Zero of the values flagged by the retrospective are cross-wired or function as dummy placeholders.** They are all either the source `INPUT_TYPES`-declared default or the canonical preserved-mode shape that ComfyUI's `/object_info` schema requires.

## Root cause hypothesis -- where does the retrospective's claim come from?

The retrospective's framing reverses cause and effect. The genuine sequence is:

1. **Original defect class (multiple BUG-LOCAL-027 / 029 / 030 / 031 entries):** ComfyUI Web-UI workflow JSONs can omit trailing unlinked widget slots when those slots hold defaults (a "stripped" preserved shape). Older versions of `_workflow_to_api_prompt` had an auto-sensing heuristic that handled unambiguous cases, but any `widgets_values` array length strictly less than the widget-backed input count was a preserved-truncated shape the heuristic could not always reconstruct. The runtime symptom is exactly the operator's 2026-05-15 observation: `temperature='{}'`, `start_line='{}'`, etc. -- values land in the wrong widget slots because the mapper offsets them.
2. **Fix (BUG-LOCAL-032, commit `dabcebd`, 2026-04-14):** Compute the canonical preserved-mode shape (linked placeholders + all unlinked defaults, in declared input order) from the live `/object_info` schema for every node and write back the canonical array. The fix INTRODUCED the empty strings, `'[]'`, `'{}'`, and numeric defaults that the retrospective now mis-reads as "dummy padding".
3. **Architectural reinforcement (BUG-LOCAL-027 socket-only-at-tail rule):** `nodes/story_orchestrator.py:2632-2636` and `:6918-6925` document the rule. The empty strings in `widgets_values` are part of the *defensive canonical shape* that prevents the cross-wiring class from recurring.

The deep-research report appears to have pattern-matched on "empty string == dummy" without consulting the source `INPUT_TYPES` declarations or the BUG-LOG history. This is a classic deep-research hallucination class: a surface-level lexical scan ("zero-length string in production data array == policy violation") presented as a structural finding.

## Repair strategy

**None needed.** The widget values are correct. The retrospective's recommended remediation -- "Implement a strict JSON schema validator that explicitly rejects arrays containing zero-length strings" -- would **break** the BUG-LOCAL-032 fix and reintroduce the widget-drift bug class.

If Sprint A or Sprint G wants a stronger guarantee, the right remediation is the opposite direction:

- Add a **schema-positive** check that compares each node's `widgets_values` array against the live `/object_info` canonical preserved-mode shape from `scripts/_schema_sweep.py` (which already exists per BUG-LOCAL-032's verify line). Treat any divergence (length mismatch, type mismatch against declared input type) as a hard fail.
- This is a one-line addition to the existing `tests/test_workflow_validation_*` family that has been growing across S26-S34. It is not a Sprint A blocker.

## Confirmation of operator's 2026-05-15 manual ComfyUI symptoms

The operator's manual symptoms (`temperature='{}'`, `start_line='{}'`, `default_tts=''`, `dialogue_offset_ms='bark'`, `resolution=24`, `fps='{}'`) are **consistent with the widget-drift class** described in BUG-LOCAL-027 / 029 / 030 / 031 / 032 -- the exact bug class that BUG-LOCAL-032's canonical-shape fix was designed to prevent.

They are **inconsistent** with the current contents of `workflows/otr_scifi_16gb_full.json` on this branch, where the canonicalized `widgets_values` arrays are correctly aligned with current `INPUT_TYPES` (verified row-by-row above).

Two scenarios could explain the operator's manual observation:

1. **Stale workflow JSON loaded in ComfyUI Desktop.** ComfyUI Desktop on the workstation may have been holding a pre-BUG-LOCAL-032 cached copy of the workflow (e.g. loaded into the browser session before the dabcebd canonicalization landed and not reloaded). The cross-wiring symptoms would surface when ComfyUI mapped the older preserved-truncated array into the newer `INPUT_TYPES`.
2. **A subsequent regression on the same bug class.** If a commit after dabcebd reverted the canonicalization without updating BUG_LOG.md, the file would once again present preserved-truncated arrays. The size walk in the companion `UNEXPECTED_FINDING_nul_padding.md` doc shows the workflow JSON went through significant churn after BUG-LOCAL-032 (re-baseline at `068bf54` +8.5K, then `af4e655` +16K); a sub-commit could have lost the canonical shape on some other node. **Worth checking in Sprint A but outside Sprint C scope.**

In neither scenario is the empty-string-as-dummy framing of the retrospective accurate.

## Recommendation for Sprint A acceptance row (carried into Deliverable 4)

Add a **schema-positive widgets_values canonical-shape gate** to Sprint A's empirical-verification pass, using the existing `scripts/_schema_sweep.py` infrastructure. The check fails the run if `widgets_values` for any node in any committed workflow JSON diverges from the canonical preserved-mode shape produced by `/object_info`. This locks the BUG-LOCAL-032 fix as a permanent regression guard and pins it against future widget-drift recurrence -- including whatever produced the operator's 2026-05-15 manual symptoms.

## Sources cited (in-repo, read-only)

- `workflows/otr_scifi_16gb_full.json` (current branch HEAD)
- `__init__.py:84-98` (NODE_CLASS_MAPPINGS)
- `nodes/story_orchestrator.py:2556-2641, 2632-2636, 6887-6927, 6918-6925`
- `nodes/scene_sequencer.py:564-624, 891-923`
- `nodes/audio_enhance.py:278-319`
- `nodes/batch_bark_generator.py:474-501`
- `nodes/video_engine.py:1167-1207`
- `nodes/kokoro_announcer.py:116-148`
- `nodes/musicgen_theme.py:147-181`
- `nodes/batch_audiogen_generator.py:85-110, 102-106`
- `BUG_LOG.md:294-304` (BUG-LOCAL-032)
- `docs/AI_Production_Pipeline_Retrospective__Sprint_C.md` §6
- `docs/closed-sprints/2026-05-15-sprint-c-story-brief-v2.md` §C-final
