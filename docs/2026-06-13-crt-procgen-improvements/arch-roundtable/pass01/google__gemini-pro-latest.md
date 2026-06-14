<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The plan removes scopes from the floor but suppresses them during gaps, leaving the "Signal Lost" gaps completely dead, and hallucinates a modular drawing API that does not exist in the codebase.

MUST-FIX BEFORE BUILD:
1. [Proposed v2 design] Dead Gaps: The plan removes scopes from the floor (`OTR_SignalLostVideo`) but states "Gap beats (no clip) -> treat as the floor -> suppress." If the floor no longer has scopes, and the overlay suppresses them during gaps, the radio-floor gaps will have zero audio-reactive elements.
   - Fix: Do NOT suppress scopes during gap beats. Draw them full-frame (or centered) during gaps to replace the scopes removed from the floor.
2. [Proposed v2 design] Crash in Sequential Mode: The plan maps `fi` to `[start_s, start_s+dur_s)`. Grounding (`otr_silent_composite.py`) explicitly shows `start_s` is optional; if missing, the manifest falls back to SEQUENTIAL mode. Relying on `start_s` will fail.
   - Fix: Call `plan_timeline_segments()` to get the frame-accurate segment list, and map `fi` to the active beat by accumulating `n_frames` sequentially. 
3. [Proposed v2 design] Hallucinated Scope Module: The plan claims "The same `_draw_fft_scope`/`_draw_scope` module" exists. Grounding (`grounding_crt_code.py`) shows all drawing (title, grid, scopes, CRT vignette) is tightly coupled inside a single monolithic `_CRTRenderer.render()` method.
   - Fix: Explicitly refactor `_CRTRenderer` to extract the scope components (frequency ring, orbiting particles, mirrored waveform, frequency bars) into standalone methods so the new node can call them without redrawing the title, grid, and background.

SHOULD-FIX:
1. [Open architecture questions - 4] Aspect Ratio Detection: Relying on `engine_id` for aspect ratio is brittle and assumes a static registry.
   - Fix: Probe the clip's native dimensions from the manifest `path` (via ffprobe or similar) to determine portrait vs landscape definitively.
2. [Open architecture questions - 6] Backwards Compatibility: Hard-switching the floor to stop drawing scopes will break legacy v1 pipelines that still rely on it.
   - Fix: Add a `draw_scopes` boolean flag to `OTR_SignalLostVideo`'s `INPUT_TYPES` (default `True` for v1 compat, set to `False` for v2).
3. [Open architecture questions - 3] Beat-boundary flicker: Hard cuts of scope geometry at beat seams will look like glitches if the audio is continuous.
   - Fix: Implement a brief (e.g., 2-4 frame) alpha crossfade or size interpolation when transitioning between portrait gutters and landscape suppression.

CUT THESE (over-engineering):
1. [Proposed v2 design & Open architecture questions - 1] Python-based Compositing: The plan proposes `OTR_SceneAwareScopes` will output an mp4 "composited green-only over the input". Decoding, PIL-compositing, and re-encoding the 1920x1080 master video in Python is slow and risks generation loss.
   - Why it is safe to cut: ffmpeg already handles the green blend perfectly. Have `OTR_SceneAwareScopes` generate a standalone `scopes_only.mp4` (black background, green scopes) at 25fps. Extend `OTR_PostUpscaleProcgenBlend` to accept this as a 3rd input and do a double-blend (`[main][pgn]blend...[tmp]; [tmp][scopes]blend...[out]`) entirely in ffmpeg.