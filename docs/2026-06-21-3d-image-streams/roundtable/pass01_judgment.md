# pass01 judgment (Claude = judge + grounded panelist)

## Panel
GPT-5.5, Gemini-3.1-pro, Claude-Opus-4.8, DeepSeek-v4-pro, Grok-4.3 (Claude-Sonnet errored).
Claude added a grounded panelist pass (verified the linchpin claims against the real code).

## ACCEPTED (CONFIRMED against render_driver.py / eng_mesh_stage.py)
- **The big one (3-model + Claude-verified):** mesh init is the SCENE STILL, not the portrait
  (`_SCENE_INIT_FAMILIES` override at build_request_from_shot:703-708; mesh_stage family=image_to_video).
  pass00's premise was false; the whole design re-based on the real seam.
- `requires_mesh_portrait` absent on MeshStageEngine (verified) -> add explicit `requires_mesh_fodder`.
- New kinds invisible: `_still_index` only matches `scene_` prefix -> plate must be `scene_background_plate`.
- `_portrait_index` has no kind filter -> would pick up fodder rows -> add a kind filter.
- Fork belongs in OTR_MetaBriefImagePromptGen (LLM authors prompts; dispatcher only consumes).
- Capability must be resolved AFTER `apply_engine_override`/`OTR_FORCE_ENGINE_MAP` (GPT).
- Mesh cache keyed on the still hash -> cache miss/rebuild per beat -> stable per-subject fodder +
  generalize to `mesh_subject_id` (char_id | object_id; today writes misleading "uncast").
- Subject policy for announcer/music (no char_id): object or reroute, never `uncast`-on-environment.
- _still_index last-write-wins race if both scene_* and plate exist (Gemini) -> 3D beats mint ONLY
  fodder+plate, or prioritize the plate.
- Opaque source-over composite (kills the ghost) + a regression check; keep blend as opt-in style.
- Prompt scaffolds (positive+negative) for character fodder / object fodder / plate are a functional
  dependency, not polish.

## CUT (panel over-engineering)
- Extra IMAGE-cache-key logic for fodder: `request_cache_key` already includes `kind`, so
  `kind="mesh_fodder"` already isolates the image cache (Gemini). The MESH-cache (mesh_cache_key) is the
  real one to fix -- kept.

## DEFERRED to v1.5 (panel SHOULD/NICE)
- Cycles + 3-point lighting + multi-view texture bake. Separate sprint AFTER clean fodder + opaque
  composite (clean fodder is the higher-leverage fix; lighting on a blob is lipstick).

## VERIFY-AT-BUILD (UNVERIFIABLE from the excerpts)
- That the engine capability actually reaches the prompt-gen seam at prompt time.
- The final engine map (incl. force-map) is resolvable before image-prompt mint.
- `_still_index` priority when a scene_* still and a scene_background_plate co-exist.

## Convergence
One grounded pass produced a complete re-base + a closed must-fix set with no internal contradictions.
The design is build-ready pending the three verify-at-build checks. No second live pass needed (the
remaining items are build-time verifications, not open design forks).
