# Antigravity Re-verification Report

## VERDICT
STILL-VALID (with one minor STALE line number cite update). The recent changes (credits tail-chain node 95 integration and no-fallback cleanup) have landed cleanly without regressing or invalidating the core audit plan logic. The test suite is currently running green at HEAD.

## STILL-VALID
- `nodes/_otr_workflow_validator.py:175-179` (Expected slot length check)
- `tests/test_cast_lock.py:62-64` (CastLock serialized slots check)
- `tests/test_announcer_voice.py:87` (AnnouncerVoice serialized slots check)
- `tests/test_batch_character_voices.py:94-95` (BatchCharacterVoices serialized slots check)
- `tests/test_stable_audio_theme.py:80` (StableAudioTheme serialized slots check)
- `nodes/cast_lock.py:18-20` (E.4 docstring details)
- `nodes/OTR_LedgerScriptWriter.py:2196-2213` (Story scaffold widget definition)
- `nodes/OTR_LedgerScriptWriter.py:1662-1682` (`_apply_story_scaffold_env` resolution method)
- `nodes/otr_caption_burn.py:70-86` (`_resolve_ledger_path` suffix handling)
- `nodes/otr_caption_burn.py:98-108` (`burn_captions_on_video` method signature)
- `nodes/otr_caption_burn.py:160-198` (`INPUT_TYPES` optional section and early burn return)
- `nodes/otr_caption_burn.py:183-192` (`_default_out` path constructor)
- `nodes/otr_post_upscale_procgen_blend.py:923-930` (Intermediate blending directory logic)
- `config/profiles/widget_mapping.json:95-112` (Features mapping definition)
- `tests/test_workflow_live_passes_validator.py:56-85` (Saved workflow visual structure pins)
- `tests/test_post_upscale_procgen_blend.py:150-163` (ASS basename path parsing tests and widget defaults)
- `__init__.py:299-302` (Main `OTR_CaptionBurn` registration under `_NODE_MODULES`)
- `tests/test_caption_burn_cw4.py:19-20` (Imports checking)

## STALE
- `nodes/_otr_voice_node_common.py:183-188`: The docstring comment for `voice_input_types` has shifted due to the recent "no-fallback rip" changes. It is now located at **`nodes/_otr_voice_node_common.py:177-182`** (shifted up by 6 lines).

## NEW MUST-FIX
- None.

## MISREADS
- None.

## TAIL ORDER QUESTION
If node 86 `OTR_CaptionBurn` becomes the caption owner, the correct stage order wiring is:
`84 -> 93 -> 86 -> 95 -> 85` (meaning `OTR_CaptionBurn` runs **BEFORE** `OTR_CreditsRoll`, so credits stay caption-free).

### Justification:
1. **Ledger Resolution Path (`_resolve_ledger_path`):** The ledger resolution in `nodes/otr_caption_burn.py:70-86` relies on matching the file stem and stripping standard suffixes (`_silent`, `_captioned`, `_final`, `_blend`). If `OTR_CreditsRoll` runs first, it appends the credits and outputs a file ending in `_with_credits.mp4`. The stem then ends with `_with_credits`, which is NOT in the stripped list. This causes ledger resolution to fail, so `OTR_CaptionBurn` would silently fall back to passthrough (burning no captions).
2. **Avoiding Redundant Re-encoding and Loss of Stream Copy:** `OTR_CreditsRoll.roll` generates a silent credits clip and appends it to the body video using the concat demuxer with `-c copy` (stream copy), which is fast and lossless. If `OTR_CaptionBurn` runs after `OTR_CreditsRoll`, it has to run an ffmpeg re-encode (`-c:v libx264`) on the *entire* concatenated video (body + credits), losing the performance and quality benefits of the concat stream copy and wasting cycles re-encoding the credits roll.
3. **Mux Duration Assertion and `declared_credits_tail_s` Wiring:** The duration guard in `OTR_MasterAudioMux` compares the silent video's duration `v_dur` against the master audio's duration `a_dur` plus the `declared_credits_tail_s` float (`v_dur <= a_dur + declared_credits_tail_s + tol`). If `OTR_CreditsRoll` runs last, it directly connects to both `MasterAudioMux` slot 1 (video path) and slot 6 (declared duration), ensuring the duration verification matches the concatenated file's structural boundary perfectly. If `CaptionBurn` runs after `CreditsRoll`, it produces the terminal video path without returning a declared duration float, causing messy indirect wiring from Node 95 for the duration float while Node 86 feeds the video path.
4. **Semantic Appropriateness:** Since SDH captions caption dialogue, and there is no dialogue during the credits roll, the credits segment should logically remain clean and caption-free, which is naturally achieved by appending it after the caption burn step.

## PROFILE TARGETING ANALYSIS
- **widget_mapping.json:** In `config/profiles/widget_mapping.json:95-112`, `features.burn_captions` and `features.caption_style` still map targets directly to `OTR_PostUpscaleProcgenBlend`.
- **Profile JSONs:** The three profile files (`16gb_full.json`, `8gb_lite.json`, `cpu_floor.json`) define the `"burn_captions"` and `"caption_style"` features abstractly. Since these profile definitions are decoupled from the raw node classes, they correctly target `OTR_PostUpscaleProcgenBlend` via `widget_mapping.json`.

## VALIDATOR CLI --strict-types CONTEXT
- **Verdict:** This is a **CLI-context artifact**, not a real registration gap.
- **Justification:** 
  In `__init__.py`, the node loader dynamically imports `new_node_modules_table()` from `nodes/_otr_class_registry.py` and merges these spec definitions into `_NODE_MODULES` at runtime if their backing files exist on disk. They are then dynamically added to ComfyUI's global `NODE_CLASS_MAPPINGS`.
  However, the validator CLI in `tools/validate_workflow_links.py:61-81` parses `__init__.py` *statically* using Python's `ast` module (without executing it) and extracts keys only from dict literals assigned directly to `_NODE_MODULES` or `NODE_CLASS_MAPPINGS` in that file. Because nodes 80-83 are registered dynamically via `_otr_class_registry.py`, the static AST parser cannot see them, leading to the false positive report.
