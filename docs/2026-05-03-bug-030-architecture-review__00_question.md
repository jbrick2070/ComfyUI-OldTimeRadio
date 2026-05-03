# Question -- 2026-05-03

# Code review: BUG-LOCAL-030 Phase A + Phase B end-to-end visual stack

## Project context

ComfyUI custom-node radio-drama generator (OTR "SIGNAL LOST"). Windows / RTX 5080 Laptop / 16 GB VRAM. The previous soak produced a "mostly black" final video — root cause: HuMo's native 480x832 portrait got pillarboxed into a 832x480 landscape canvas, leaving 67% of every frame as pure black bars; AND the per-clip-mux composite mode never blended the procgen visual layer.

**Just shipped (commits f1467a2 → 6164b86, 6 commits over the last hour, 214 tests green):** BUG-030 Phase A + Phase B end-to-end visual stack pivot. Rendered + composited + upscaled + procgen-blended pipeline. Round-robin code review SKIPPED at user directive during shipping; user now wants peace-of-mind retrospective consult before queuing the soak. **NO new fixes from you — verdict only on whether the architecture will produce visible output and what (if anything) was missed.**

---

## Architecture (full pipeline left → right)

```
HuMo render (480x832 portrait, native, 25 fps, length=97 frames=3.88s)
                                    │
                                    ↓
LTX render (832x480 landscape, native, 25 fps, 8n+1 frames)
                                    │
                                    ↓
Per-clip pillarbox into 1472x832 landscape canvas (BUG-030 Phase A)
  HuMo: scale=-2:832:force_original_aspect_ratio=decrease,
        pad=1472:832:(ow-iw)/2:(oh-ih)/2:color=black,fps=25
        → 480x832 native + 496px black bars per side
  LTX:  same formula
        → scale to height=832 = 1442x832 + 15px black bars per side
                                    │
                                    ↓
Concat per-line clips (ffmpeg concat-demuxer, -c copy, no re-encode)
                                    │
                                    ↓
Mux master_mix audio (ffmpeg, -c:v copy -c:a copy -shortest)
                                    │
                                    ↓
composited/<ep>.mp4 (1472x832 with HuMo native + LTX native + audio)
                                    │
                                    ↓
OTR_RTXUpscale (NVIDIA RTX VSR, 1472x832 → 1920x1080, audio passthrough)
                                    │
                                    ↓
obs/<ep>_1080p.mp4 (1920x1080 deliverable, RTXUpscaled)
                                    │
                                    ↓
NEW: OTR_PostUpscaleProcgenBlend (BUG-030 Phase B)
  Inputs: source mp4 (RTXUpscale output, 1920x1080) + procgen mp4 (1920x1080 native)
  ffmpeg filter:
    [1:v]scale=-2:ih:force_original_aspect_ratio=decrease,crop=iw:ih,setpts=PTS-STARTPTS[pgn];
    [0:v][pgn]blend=all_mode=lighten:all_opacity=0.500[v]
  Map: [v] for video, 0:a? for audio (-c:a copy -- C7-safe)
  Encode: libx264 yuv420p crf 18 preset fast, -shortest
                                    │
                                    ↓
obs/<ep>_1080p_procgen_blended.mp4 (FINAL deliverable)
```

Procgen renders at NATIVE 1920x1080 (was 832x480) — this is BUG-030 Phase B's central architectural decision. Reasoning: synthetic patterns (CRT scanlines, audio-reactive flicker) get smeared/ringed when fed through RTX VSR (designed for natural / AI-rendered content). Native-rendering procgen at 1080p keeps it crisp; the post-RTXUpscale blend preserves it; the blend's `lighten` mode + 0.5 opacity fills the visible HuMo black pillarbox bars (the 496px black each side from the per-clip composite) with audio-reactive CRT scanlines — the SIGNAL LOST visual signature.

C7 audio byte-identity: per-clip-mux audio path stays intact (HuMo + LTX video gets re-encoded but audio is `-c:a copy` throughout). RTXUpscale uses `-c:a copy`. Post-upscale blend uses `-c:a copy`. So audio never re-encodes downstream of `OTR_SignalLostVideo`'s initial CreateVideo step.

---

## Workflow JSON wiring (just shipped)

- New node id 58: `OTR_PostUpscaleProcgenBlend` at pos `[4900, 1100]`
- Link 95: node 56 (RTXUpscale) slot 0 (`upscaled_mp4_path`) → node 58 slot 0 (`source_mp4_path`)
- Link 96: node 12 (SignalLostVideo) slot 0 (`video_path`) → node 58 slot 1 (`procgen_mp4_path`)
- node 58 widget defaults: `blend_mode=lighten, blend_opacity=0.5, ffmpeg=ffmpeg, bypass=False, out_suffix=_procgen_blended`
- node 12 (SignalLostVideo) widget resolution flipped 832x480 → 1920x1080
- node 52 (VideoComposite) widgets: canvas_width=1472, canvas_height=832, humo_target_height=832, humo_pillar_width=512 (reserved for future layered-mode), audio_source=master_mix_per_clip_mux
- last_node_id 57→58, last_link_id 94→96

---

## Asks (please answer all)

### 1. Will the per-clip-mux composite produce VISIBLE video?

The simple-pillarbox formula:
```
scale=-2:832:force_original_aspect_ratio=decrease,
pad=1472:832:(ow-iw)/2:(oh-ih)/2:color=black,
fps=25
```

For HuMo input 480x832: `scale=-2:832:force_original_aspect_ratio=decrease` — does this correctly leave 480x832 unchanged (since height is already 832 and the "decrease" flag means "don't enlarge")? Or does it actually resize down because something about `force_original_aspect_ratio=decrease` triggers when the input matches the target dim?

For LTX input 832x480: scale to height=832 = ?? what does ffmpeg actually compute? Is it 1442x832 (preserves aspect ratio: 832 × 832/480) or does the `force_original_aspect_ratio=decrease` flag cause it to clamp width somewhere?

Then `pad=1472:832:(ow-iw)/2:(oh-ih)/2:color=black` — does this center the scaled output in a 1472x832 black canvas? Any concern about odd-pixel widths breaking yuv420p encoding (libx264 requires even pixel dims)?

Net visible output per frame:
- HuMo character clips: 480-pixel-wide center strip + 496px black bars per side
- LTX clips: 1442-pixel-wide near-full-canvas + 15px black bars per side

Will viewers actually see HuMo lipsync content in those character clip frames, or is something else going to make them invisible (encoding alpha, color space, fps mismatch, etc)?

### 2. Will the post-upscale procgen blend produce VISIBLE blended output?

The blend ffmpeg command:
```
ffmpeg -y -loglevel error
  -i <source_1920x1080.mp4>
  -i <procgen_1920x1080.mp4>
  -filter_complex
    "[1:v]scale=-2:ih:force_original_aspect_ratio=decrease,crop=iw:ih,setpts=PTS-STARTPTS[pgn];
     [0:v][pgn]blend=all_mode=lighten:all_opacity=0.500[v]"
  -map "[v]" -map "0:a?"
  -c:v libx264 -pix_fmt yuv420p -crf 18 -preset fast
  -c:a copy
  -shortest
  <out>.mp4
```

Specific concerns:
- The `[1:v]scale=-2:ih:force_original_aspect_ratio=decrease,crop=iw:ih,setpts=PTS-STARTPTS[pgn]` chain on procgen input — what does this actually do? `scale=-2:ih` says "scale to current height, width auto". `crop=iw:ih` says "crop to current width × current height" (no-op). Combined: it's effectively `setpts=PTS-STARTPTS` only? Is this enough to handle fps / pts mismatch between source and procgen if they were rendered at different fps (RTXUpscale output likely 25fps from upstream; procgen renders at 24fps default)?
- `blend=all_mode=lighten:all_opacity=0.500` — `lighten` = max(source, overlay) per channel. Overlaid procgen with 0.5 opacity. Does this actually let HuMo lipsync content (potentially dark noir scenes) shine through where it's brighter than procgen? Or will procgen's flicker/scanlines obliterate the HuMo content if procgen has bright peaks?
- `-shortest` — RTXUpscale source duration is exactly the composite duration. Procgen duration is the FULL EPISODE audio duration (master_mix). They should be IDENTICAL because procgen was rendered FROM the same audio that was muxed into the per-clip composite. Confirm this assumption — if procgen is even 100ms longer than source, `-shortest` would clip it cleanly. If procgen is SHORTER, the source video's tail would have NO procgen overlay (just the source content). Is this the right safety direction?
- Audio: `-c:a copy` from source (RTXUpscale output, which got `-c:a copy` from VideoComposite, which got `-c:a copy` master_mix from per-clip-mux step). Confirm this is the right behavior — procgen's own audio gets DISCARDED via no `-map` for `1:a`, which is correct because the source already has master_mix audio.

### 3. Are there any latent ffmpeg pitfalls in this stack?

Specifically:
- Color space mismatch (rec601 vs rec709)? Source is libx264 yuv420p; procgen is whatever `OTR_SignalLostVideo` writes (also yuv420p typically). Risk of color shift on blend?
- pix_fmt compatibility — both source and procgen are yuv420p. Should be safe but flagging.
- Frame rate mismatch — composite is fps=25, RTXUpscale preserves; procgen widget default is fps=24. Will the blend filter resample procgen to 25? Or will it produce dropped/duped frames?
- Audio alignment — `-c:a copy` from source carries master_mix audio. Procgen renders FROM master_mix, so they SHOULD be sample-aligned. But if there's any duration drift between procgen-render and master_mix-mux, the audio could feel slightly off-sync visually.

### 4. Is the simple-pillarbox composite the right choice for Phase A, or is there a known-better pattern I should have used?

Per Jeffrey's spec: HuMo native portrait + LTX native landscape, both pillarboxed into 1472x832 with black bars. The visible black bars get filled by procgen at the post-upscale stage.

Alternative architectures considered + rejected:
- Layered overlay (HuMo as 512x832 pillar over FLUX env still bg): more complex, shipped as utility code but not used in current Phase A.
- Crop-to-fill instead of pillarbox: would crop HuMo content (lose head/feet of speaker). Rejected.
- Smaller canvas (832x832 square): would force LTX into pillarbox too. Rejected for being not landscape.

Did we miss a cleaner pattern? Specifically: is there a way to AVOID having 496px of pure black on character clips and still keep HuMo at native quality without forcing it through OOD dims?

### 5. Per-element follow-up-fix probability estimate

For each of these architecture elements, give a percentage estimate of how likely YOU think a follow-up fix will be needed in the next 2 weeks of soak runs (0% = bulletproof, 100% = will definitely break in production):

- **Element 1:** simple-pillarbox formula in `_layered_per_clip_silent` ELSE branch (scale-fit + pad-with-black for both HuMo + LTX)
- **Element 2:** post-RTXUpscale procgen blend ffmpeg chain
- **Element 3:** procgen rendered at 1920x1080 native (was 832x480) — VRAM impact, render time, framework correctness
- **Element 4:** workflow JSON wiring (new node 58 + 2 new links + bumped IDs)
- **Element 5:** the `lighten` blend mode at 0.5 opacity choice for the procgen overlay (visual aesthetic AND the assumption that lighten will let HuMo content shine through)

Give your reasoning briefly.

### 6. What did I miss?

Open-ended. Anything about the architecture that surprises you, looks suspicious, or seems undertested? Specifically:
- Other workflow JSON wiring concerns (node 58 sequence order = 9300, sandwiched between RTXUpscale at 9200 and CLIPLoader at 9050 — does this run in the right place?)
- Any concern about hot-reload / sys.modules caching now that 5+ custom node files have been edited in this session?
- C7 byte-identity claim — is `-c:a copy` end-to-end actually sufficient, or does any of these ffmpeg passes have a hidden audio re-encode trigger I missed?

Per CLAUDE.md, this fix touches the audio C7 path (multiple ffmpeg muxes) → would normally require this round-robin pre-implementation, but the user explicitly directed to ship first then consult. Reviewing AFTER landing.
