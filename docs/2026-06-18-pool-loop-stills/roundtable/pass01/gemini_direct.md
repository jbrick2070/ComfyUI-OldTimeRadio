<!-- model: gemini-pro-latest (direct API) -->

VERDICT: yes-with-fixes. The threading plan is solid, but optimizing away scene stills for face engines will break the universal fallback floor, and the role definitions contradict themselves.

MUST-FIX BEFORE BUILD:
1. [Section 3] The fallback trap. You ask if the still phase should skip emitting scene stills when `other_beats=HuMo` because HuMo uses portraits. NO. `render_driver.py` explicitly defines `EXPECTED_OOM_TRAIL` where `humo_1.7B` falls back to `still_kenburns`. `still_kenburns` is `static_motion` and strictly requires a scene still. If you skip the scene still, a runtime OOM on HuMo will cause the fallback to crash with `MISSING-STILL`, violating the "never aborts" invariant. Fix: Always generate scene stills for roles where `accepts_still=True`, regardless of the primary engine's family.
2. [Section 1 & 2] Graph-order impossibility. The still phase runs *before* `ShotLock` freezes the audio budget. It cannot map N stills to final `beat_id`s because the final beats don't exist yet. Fix: `derive_scene_still_targets` must emit exactly N targets using generic keys (e.g., `other_pool_0` to `other_pool_N-1`). `otr_shot_lock.py` must then stamp `shot["_still_index"] = f"other_pool_{i % pool_n}"` onto the final shots during its existing clip-budget pass. `render_driver` remains unchanged and just reads `_still_index`.
3. [Section 2] Role set contamination. Under "Threading path", you state `ImageDirector other_beats_image_model role set = {character_video, scene_broll, background_abstract}`. This is fatal. `otr_shot_lock.py` explicitly defines `CHARACTER_VIDEO` as a `CHARACTER_BEARING_ROLE`. If you pool `character_video` as an "other beat", characters will lose their unique per-beat prompts and sync. Fix: Remove `character_video` from the `other_beats` role set. It must strictly be `{scene_broll, background_abstract}`.

SHOULD-FIX:
1. [Section 5] Determinism in prompt selection. When `derive_scene_still_targets` reduces M lines to N pool stills, it must deterministically choose which lines provide the text prompts. Fix: Explicitly slice the first N lines that map to `other_beats` roles in sequential order to generate the N pool stills.
2. [Section 4] Widget defaults vs JSON. Having the widget default to `unique_per_beat` + 8 while the JSON defaults to `pool_n_loop` + 4 creates a desync if the node is ever reset or rebuilt. Fix: Update the `OTR_VideoDirector` widget defaults in code to match the JSON (`pool_n_loop` and `4`).

CUT THESE (over-engineering):
1. [Section 1] "at dispatch (the dispatcher dedups)". Safe to cut. If `derive_scene_still_targets` only emits N targets with generic pool keys, the dispatcher requires zero new deduplication logic. It just processes what it is handed.