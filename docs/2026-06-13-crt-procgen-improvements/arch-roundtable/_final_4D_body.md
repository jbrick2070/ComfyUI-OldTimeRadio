## 4D. v2 SCENE-AWARE SCOPES -- architecture (round-robin CONVERGED, the proper landscape fix)

> Architecture round robin 2026-06-13 (gpt-5.5 + gemini-3.1-pro + deepseek-v4-pro; 3
> passes; Claude judge/grounder; ~$0.54; artifacts in
> `docs/2026-06-13-crt-procgen-improvements/arch-roundtable/`). The §4C landscape-gutter
> OPEN decision, resolved. The procgen FLOOR is NOT relocated -- only the SCOPES split out
> into an ADDITIVE late node, so the floor's gap-fill/credits/green-blend roles stay
> intact (answers the operator's "don't wire the procgen out" worry). Builds AFTER §4C.

### Why a late node (grounded)
The floor (`render_video`) cannot be scene-aware: it receives no manifest and no per-beat
engine plan, and `meta.visual_plan` carries `characters/scenes/style` only. The per-beat
portrait(HuMo)-vs-landscape(LTX/Wan) aspect first exists at the **clip manifest**
(produced by `OTRVideoRenderBatch`, whose `RETURN_NAMES = ("render_report_json",
"clip_manifest_json")` -- CONFIRMED an output) and at the **post-upscale blend**. So the
scene-aware draw must live late, where the manifest is.

### The 4 parts
1. **Circular green-only scope helpers (= §4C-v1 §S2; a REWRITE, not a wrap).** The
   existing `_waveform_mirror`/`_freq_bars_wide` are rectangular + draw non-green
   (CRT_CYAN/RED/AMBER). Build `draw_fft_scope(draw, cx, cy, r, freq_window, env)` /
   `draw_scope(draw, cx, cy, r, wave, env)` -- GREEN-ONLY (CRT_GREEN/CRT_DIM/CRT_DARK
   only; forbid the colored constants), geometry by params (no `self`). SHARED by the
   floor (v1) and the late node (v2).
2. **`OTR_SceneAwareScopes` (NEW node) -> `scopes_only.mp4`.** Draws BLACK + green-only
   scopes on black (NO master decode -> no generation loss). Skeleton:
   `CATEGORY="OldTimeRadio/v2/video"`, `FUNCTION="render_scopes"`,
   `RETURN_TYPES=("STRING",)`, `RETURN_NAMES=("scopes_mp4_path",)`; register in
   `NODE_CLASS_MAPPINGS`. INPUT required: `clip_manifest_json` (wired from
   `OTRVideoRenderBatch`); optional: `audio` (analysis-only; absent -> synthesize
   `volume=[0.0]*total`, `freqs=[zeros(32)]*total`, `waves=[zeros(200)]*total`, do NOT call
   `_analyze_audio`), `out_w/out_h` (1920/1080 delivery size), `ffmpeg`. Output:
   `out_w x out_h` @ **25fps HARD-LOCK**, silent, bt709/yuv420p/CFR; PIL frames -> ffmpeg
   stdin (the floor's pattern).
3. **`OTR_PostUpscaleProcgenBlend` EXTENDED (3rd optional `scopes_mp4`).** When present,
   build a NEW 3-input filtergraph (do NOT append to the 2-input one -- it converts to
   yuv420p before blend 1): `[0:v]format=gbrp[main]; [1:v]<existing procgen
   scale/crush/zeroRB>format=gbrp[pgn]; [2:v]<scale,fps,setpts,setsar>colorchannelmixer
   (zero R+B)format=gbrp[scp]; [main][pgn]blend=all_mode=screen[tmp]; [tmp][scp]blend=
   all_mode=lighten[out]; [out]format=yuv420p[v]`. **2nd blend = `lighten` (max)** -- two
   `screen`s compound brightness where scopes overlap the procgen. `-map [v] -map 0:a?
   -c:a copy`, keep the existing bt709/CFR flags. Absent -> the current single-blend path.
4. **`OTR_SignalLostVideo` floor: `draw_scopes` flag.** Add `("BOOLEAN", {"default":
   True})` to the OPTIONAL INPUT_TYPES; thread `render_video -> _CRTRenderer -> render()`.
   False (v2) -> SKIP render() sections 2/3/5/6 (ring/particles/waveform/bars) + guard
   their dependent vars; keep 1/4/7/8 (title/grid/bottom/CRT). Floor roles UNCHANGED.

### Beat map + eligibility (the scene-aware core)
`segs, total = plan_timeline_segments(manifest, floor_available=True,
target_total_frames=manifest.get("total_target_frames"), fps=25)` -> integer ranges
`[cursor, cursor+n_frames)`; **`total` is the frame count** (no source-video probe). By
`segment["source"]`: `clip`+PORTRAIT -> gutters; `clip`+LANDSCAPE -> suppress; head/inter
`floor`/`black` gap -> DRAW (centre: `cx=out_w//2, cy=out_h//2, r=int(min(out_w*0.16,
out_h*0.30))`) so the signal-lost gaps STAY ALIVE; the TAIL gap (credits post-roll, i.e.
the gap segment(s) at `cursor >= last-beat-end` computed from the beat extents) ->
SUPPRESS (don't cover the credits). Aspect: `ffprobe segment["path"]` (h>w=portrait),
memoized per path; un-probeable -> suppress + log; registry CUT. Gutter geometry from the
delivery size: `portrait_w=round(480*out_w/1472)`, `gutter=(out_w-portrait_w)//2`,
`left_cx=gutter//2`, `right_cx=out_w-gutter//2`, `cy=out_h//2`,
`r=int(min(gutter*0.36, out_h*0.30))`, `amp<=r*0.35`.

### fps + audio
HARD-LOCK 25 across the planner / `_analyze_audio` / scopes encode (the floor defaults 24,
so force `OTR_SignalLostVideo.fps=25` in this workflow OR relax the blend assert: the
24fps procgen is framesync'd by the existing blend, so assert only `scopes==source` frame
count). `_analyze_audio` keeps the EXISTING `sr//fps` chunking (exact at 25fps ->
frame-identical, no spine touch). Deterministic stable-hash RNG (also fixes the floor's
unseeded `np.random`).

### v1/v2 sharing + the lower-risk option
The circular green helpers are shared: §4C-v1 calls them from the floor (beat-agnostic),
v2 from the late node (scene-aware); `draw_scopes=False` turns the floor scopes off for v2.
LOWER-RISK option: leave the floor's drawing untouched and build the helpers in the NODE
ONLY -- i.e. skip §4C-v1's floor-scope placement and go straight to v2. Operator's build call.

### v2 sprints + VERIFY-AT-BUILD
S-v2a circular green helpers (= §4C S2). S-v2b `OTR_SceneAwareScopes` + registration +
the planner-driven eligibility/aspect. S-v2c extend the blend (3-input gbrp double-blend,
`lighten`) + `draw_scopes`. S-v2d wiring (`OTRVideoRenderBatch.clip_manifest_json ->
SceneAwareScopes -> blend`) + golden-frame tests. VERIFY: clip `path` is pre-composite
with real dims; `lighten` stays visible over the floor CRT; empty manifest -> fail early;
golden frames for portrait / landscape / head-gap / credits-tail.
