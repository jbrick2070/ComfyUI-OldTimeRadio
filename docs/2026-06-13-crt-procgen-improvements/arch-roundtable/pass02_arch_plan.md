# v2 SCENE-AWARE SCOPES -- buildable architecture (arch round 2 hardened)

3/3 panels (gpt-5.5, gemini-3.1-pro, deepseek-v4-pro) agree on the shape; round-2 nailed
the wiring. Invariants: no audio-spine touch; green-only; deterministic; Pillow-only/CPU;
no new model; floor's gap-fill/credits/green-blend roles UNCHANGED; title+CRT stay in floor.

## Part 1 -- scope-module refactor (PREREQUISITE = v1 §4C S2; "land" = merged + full-pipeline-green)
The scopes do NOT exist as reusable green helpers -- the drawing is monolithic in
`_CRTRenderer.render()` and `_waveform_mirror(draw,wave,x,y,w,h,vol,t)` /
`_freq_bars_wide(draw,freq,x,y,w,h,vol)` are RECTANGULAR + draw NON-green (CRT_CYAN/RED/
AMBER). v2 needs CIRCULAR GREEN-ONLY scopes, so this is a REWRITE not a wrap: new helpers
`draw_fft_scope(draw, cx, cy, r, freq_window, env)` / `draw_scope(draw, cx, cy, r, wave,
env)` -- green-only palette hardcoded (CRT_GREEN/CRT_DIM brightness only), geometry by
`(cx,cy,r)` params (no `self`). Both the floor (v1) and the new node (v2) call them.

## Part 2 -- `OTR_SceneAwareScopes` (NEW node) -> `scopes_only.mp4`
Full ComfyUI skeleton (mirror the existing nodes):
`CATEGORY="OldTimeRadio/v2/video"`, `FUNCTION="render_scopes"`,
`RETURN_TYPES=("STRING",)`, `RETURN_NAMES=("scopes_mp4_path",)`.
INPUT_TYPES.required: `clip_manifest_json` (STRING, from the manifest PRODUCER --
`OTR_VideoRenderBatch` -- NOT `OTR_SilentComposite`, which only returns
`silent_video_path/report`). optional: `audio` (AUDIO, analysis-only; if absent ->
silent black overlay, no fail), `fps` (25), `out_w/out_h` (1920/1080 delivery size, NOT
the 1472x832 composite canvas), scope params, `ffmpeg`.
Output: BLACK frame + GREEN-ONLY scopes, drawn on BLACK only (NO master decode -> no
generation loss), `out_w x out_h` @ 25fps, SILENT, bt709/yuv420p/CFR (explicit encode
flags), length == the assembled timeline. PIL-frames -> ffmpeg stdin (the floor's pattern).

## Part 3 -- `OTR_PostUpscaleProcgenBlend` EXTENDED (3rd optional `scopes_mp4`)
Build a NEW 3-input filtergraph when `scopes_mp4` is present (do NOT append to the
2-input graph -- it converts to yuv420p before the 1st blend):
`[0:v]format=gbrp[main]; [1:v]<scale+crop+setpts+crush+zeroRB>format=gbrp[pgn];
 [2:v]<scale>colorchannelmixer(zero R+B)format=gbrp[scp];
 [main][pgn]blend=all_mode=screen[tmp]; [tmp][scp]blend=all_mode=lighten[out];
 [out]format=yuv420p[v]`.
**2nd blend = `lighten` (max), NOT `screen`** -- two screens COMPOUND brightness where the
scopes overlap the procgen; lighten avoids the double-bright. `-map [v] -map 0:a? -c:a
copy` (source is the silent intermediate; audio is muxed later -- unchanged). Absent
`scopes_mp4` -> the existing single-blend path (v1 compat). `bypass=True` -> copy source,
no procgen, no scopes.

## Part 4 -- `OTR_SignalLostVideo` floor: `draw_scopes` flag
Add `"draw_scopes": ("BOOLEAN", {"default": True})` to the OPTIONAL INPUT_TYPES (default
True = v1 compat). When False (v2), SKIP render() sections 2 (centre ring) + 3 (orbiting
particles) + 5 (mirrored waveform) + 6 (freq bars) -- and guard their dependent vars --
leaving section 1 (title), 4 (grid), 7 (bottom bar), 8 (CRT post) intact. Floor roles
UNCHANGED.

## Beat map + eligibility (scene-aware core)
- `segs, total = plan_timeline_segments(manifest, floor_available=True,
  target_total_frames=manifest.get("total_target_frames"), fps=25)` -> integer frame
  ranges `[cursor, cursor+n_frames)`. **`total` is the frame count** (no source-video probe
  needed). Walk segments by accumulated `n_frames` (handles missing `start_s`/SEQUENTIAL).
- Eligibility by `segment["source"]`:
  - `"clip"` + PORTRAIT -> scopes in the real gutters.
  - `"clip"` + LANDSCAPE -> SUPPRESS (b-roll is the subject).
  - `"floor"`/`"black"` HEAD or INTER-BEAT gap -> DRAW scopes (centre ok -- no portrait)
    so the signal-lost gaps STAY ALIVE.
  - the TAIL gap (rolling-credits post-roll) -> SUPPRESS (or gutter-only) -- do NOT cover
    the credits. Distinguish the tail = the final segment(s) past the last clip.
- Aspect: parse the manifest; for each `source=="clip"` segment `ffprobe segment["path"]`
  native dims -> PORTRAIT if `h>w`. CUT the engine_id->aspect registry (ffprobe covers
  pre-composite assets); un-probeable -> SUPPRESS + log the reason. VERIFY: `path` is a
  pre-composite asset with real dims (not an already-pillarboxed canvas).
- Gutter geometry from the DELIVERY size: `portrait_w=round(480*out_w/1472)`;
  `gutter=(out_w-portrait_w)//2`; `left_cx=gutter//2`, `right_cx=out_w-gutter//2`,
  `cy=out_h//2`, `r=int(min(gutter*0.36, out_h*0.30))`; `amp<=r*0.35`.

## Audio re-analysis (frame-identical, invariant-safe)
`_analyze_audio(audio_np, sr, total, fps=25)` with the EXISTING `spf = sr//fps` chunking
(exact at 25fps: 48000//25=1920) -> frame-identical to a 25fps floor, no new helper, no
spine touch. `total` from the planner above. Zero-pad chunks past audio end; stable-hash
seeded RNG for any noise (same as v1; also fixes the floor's unseeded `np.random`).

## Pipeline + wiring
`SilentComposite -> RTXUpscale -> PostUpscaleProcgenBlend(source, procgen, scopes_mp4) ->
MasterAudioMux`, with `OTR_VideoRenderBatch.manifest -> OTR_SceneAwareScopes -> scopes_mp4`
(SceneAwareScopes runs parallel; it needs only manifest + audio). Frame-count assertions
before/after the blend (all three inputs == `total`).

## CUTS (carried + arch) / OPEN
CUT: beat-boundary crossfade (hard on/off for v1, 2-4f fade later); ultra-dim landscape
edge; the engine_id->aspect registry; two user-facing scope paths long-term (hard-switch
after the compat flag). OPEN/VERIFY-AT-BUILD: clip `path` native-dims are pre-composite;
the `lighten` 2nd-blend visibility over the floor CRT; golden-frame tests (1 portrait /
1 landscape / 1 head gap / 1 credits tail).
