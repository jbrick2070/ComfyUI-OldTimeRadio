# Round A -- ChatGPT (gpt-5.5) elapsed=102.9s

## Verdict

**Green for the BUG-LOCAL-030 soak, with one caveat:** the shipped fix is correct for the active `_layered_per_clip_silent` simple-pillarbox path as described. The even-snap formula addresses the yuv420p chroma-alignment crash class for the LTX 832×480 → 1442×832-in-1472×832 case.

I would **not block the soak** on this commit.

Caveat: I cannot verify exact line numbers from the prompt and I will not invent them. References below are by file/function:

- `nodes/otr_video_composite.py`, `_layered_per_clip_silent`
- `nodes/otr_video_composite.py`, legacy `_pillarbox_humo_silent`
- `nodes/otr_video_composite.py`, legacy/non-per-clip `_build_filter_complex`
- `nodes/otr_post_upscale_procgen_blend.py`, `PostUpscaleProcgenBlend.blend()` / `_build_blend_cmd`
- `tests/test_video_composite_layered.py`

---

# 1. Is `trunc((ow-iw)/4)*2` correct + idiomatic?

**Yes, correct for this yuv420p hardening purpose.**

For a pad gap:

```text
gap = ow - iw
wanted center offset = gap / 2
needed yuv420p-safe offset = even integer
```

Your expression:

```text
trunc((ow-iw)/4)*2
```

means:

```text
2 * trunc(gap / 4)
```

That gives the largest even offset that does not exceed the centered offset when `gap` is non-negative.

For the reported failure:

```text
ow = 1472
iw = 1442
gap = 30
naive = 15
new = trunc(30/4)*2 = trunc(7.5)*2 = 14
```

So yes: it converts the unsafe odd offset to an even one.

FFmpeg expression evaluation supports `trunc()`, arithmetic precedence is normal, and the expression should parse as:

```text
trunc(((ow-iw)/4))*2
```

So the syntax is fine.

### `trunc` vs `floor`

For your real case, `ow - iw >= 0`, so:

```text
trunc((ow-iw)/4)
```

and:

```text
floor((ow-iw)/4)
```

are equivalent.

They only differ for negative values. But negative pad gaps would already mean “input wider/taller than output canvas,” which is a separate invalid/edge condition for this pad operation. So I do **not** consider `trunc()` a problem here.

### Idiom note

A more format-general FFmpeg idiom sometimes uses chroma subsampling constants such as `hsub` / `vsub`, where available, but for your actual output target of yuv420p/yuv420p-like 4:2:0, hard-snapping to multiples of 2 is exactly the right practical move.

So: **not the most abstract possible expression, but correct, readable, and appropriate for this codebase.**

---

# 2. Does the 1-pixel off-center matter visually?

**No, not meaningfully.**

For the important LTX case:

```text
canvas: 1472
content: 1442
gap: 30

old ideal visual center: 15 left / 15 right
new safe placement:      14 left / 16 right
```

That is a **1-pixel content-center shift** at the 1472-wide intermediate resolution. If later upscaled to 1920-wide, it becomes roughly:

```text
1 * 1920 / 1472 ≈ 1.3 px
```

Still visually negligible.

### Procgen/lighten overlay concern

I do **not** expect the post-RTXUpscale procgen blend to reveal this as a visible seam.

Reasons:

- The procgen blend is full-frame, not aligned to the pillarbox edge as a hard mask.
- The content is merely shifted one pixel relative to the mathematical center.
- `lighten` blending will not create a new vertical discontinuity at the old 15px centerline.
- Any asymmetry would be between a 14px and 16px pillar region before upscale, which is below practical visibility in this context.

The only way it might become detectable is if you had a highly symmetric full-frame overlay/vignette and someone pixel-peeped the left/right border widths. For an OTR noir radio-drama visual, this is effectively invisible.

---

# 3. Did you miss other odd-pixel pad/overlay sites?

Based on the inventory you gave: **production active path looks covered for this specific bug.**

But there are two residual codebase risks depending on whether legacy paths are truly unreachable.

## A. `_layered_per_clip_silent` simple-pillarbox path

This is the actual BUG-LOCAL-030 production fix. The replacement:

```text
pad=...:trunc((ow-iw)/4)*2:trunc((oh-ih)/4)*2
```

is correct.

## B. Layered branch overlay X

The defensive change:

```text
overlay=x=trunc((W-w)/4)*2:y=0
```

is also correct for yuv420p alignment.

For current values:

```text
1472 - 512 = 960
960 / 2 = 480
```

Already even. But your future-widget example is valid:

```text
1472 - 510 = 962
naive = 481
snapped = trunc(962/4)*2 = 480
```

So the defensive change is justified.

## C. Legacy `_pillarbox_humo_silent`

You said this still uses the naive form and is **not used by the active per-clip-mux renderer**.

Verdict:

- If truly unreachable in the production path: **not a soak blocker.**
- If any UI branch, fallback, test helper, or back-compat mode can call it with yuv420p output: **same bug class remains there.**

So I would classify it as a **known non-production residual risk**, not a missed active-path bug.

## D. `OTR_PostUpscaleProcgenBlend._build_blend_cmd`

Your description sounds safe for this bug class.

If it only uses:

```text
crop=iw:ih
setpts=PTS-STARTPTS
```

and the source/procgen are both 1920×1080, then there is no odd pad/overlay offset problem there.

Residual risk there is dimension/FPS/duration mismatch, not chroma-offset crash.

## E. Legacy `OTR_VideoComposite._build_filter_complex`

This one is the other meaningful caveat.

You said it uses:

```text
overlay=x=offset_x:y=0
```

with `offset_x` computed Python-side per clip.

If that path can produce odd `offset_x` and the output is yuv420p or another subsampled format requiring even chroma placement, then **yes, it has the same class of risk**.

Verdict:

- If non-per-clip-mux mode is dead/legacy/unreachable for soak: **not a blocker.**
- If it remains selectable: **residual odd-overlay risk exists there.**

---

# 4. Residual risk after this fix

## A. Concat demuxer with `-c copy`

Risk: **low to medium**, depending on how tightly the per-clip pillarbox outputs are normalized.

The even-snap fix does not address concat stream compatibility.

For `concat demuxer + -c copy`, all per-clip video streams need to be practically identical:

- codec
- profile/level
- resolution
- pixel format
- SAR/DAR
- time base
- frame rate mode
- color metadata
- field order
- audio presence/absence if muxed

If `_layered_per_clip_silent` emits all clips through the same ffmpeg encoding command with the same `fps`, canvas size, pix_fmt, SAR, and encoder settings, this should be fine.

The specific BUG-LOCAL-030 fix makes inconsistent pad offsets less likely to crash during clip creation, but it does not itself prove concat homogeneity.

## B. `-shortest` in procgen blend

Risk: **low for crashes, medium for subtle truncation.**

`-shortest` usually will not crash. The risk is that if procgen is a little shorter than the source, the blended video can stop early.

If the final mux preserves audio byte-identically and video is allowed to end at the shorter stream, this may be acceptable. But if source/procgen duration mismatch is common, `-shortest` can silently cut the tail visually.

This is not related to the even-snap fix.

## C. FPS mismatch: 24fps procgen vs 25fps source

Risk: **medium for visible judder**, low for hard failure.

If the blend graph does not normalize FPS/timebase before blending, FFmpeg framesync will make choices about which frames correspond at each timestamp. That can produce:

- duplicate frames
- dropped frames
- uneven motion
- subtle cadence judder

For static/noir/procgen texture overlays, this may be tolerable. For motion-heavy procgen, it may be visible.

Again: not a BUG-LOCAL-030 regression, but still a soak item to watch.

## D. `lighten` at 0.5 washing out dark noir HuMo scenes

Risk: **medium artistic risk, low technical risk.**

`lighten` blending can absolutely raise blacks if the procgen layer has brighter pixels in dark regions. At 0.5 opacity, this can reduce noir contrast.

Failure mode is not crash/invisible video. It is “the shot no longer looks as dark as intended.”

The even-snap asymmetry itself should not materially worsen this.

---

# 5. Follow-up-fix probability estimate

For only the three touched elements in commit `df317ac`:

| Element | Follow-up fix probability | Reasoning |
|---|---:|---|
| **Element 1: even-snap pad formula in `_layered_per_clip_silent` ELSE branch** | **6%** | The formula is correct for the reported yuv420p offset bug. Remaining risk is not the expression itself, but adjacent assumptions: non-negative pad gap, all inputs fitting the canvas after scale, and no other branch using naive centering. |
| **Element 2: even-snap overlay X in layered branch** | **4%** | Defensive change is correct and current widget values were already even. Low chance of needing follow-up unless future pillar sizes or formats expose a different alignment constraint. |
| **Element 3: rename `out` → `output_path` in `PostUpscaleProcgenBlend.blend()`** | **2%** | Pure local rename to satisfy audit. Only realistic risk is a missed reference or brittle test/string audit behavior. Assuming tests pass, this is very low risk. |

Overall: **this commit looks low-risk and appropriately surgical.**

---

# 6. What did you miss?

The main thing I would keep in mind is that this commit fixes the **active reported crash vector**, not every possible odd-center expression in the repository.

My residual watchlist:

1. **Legacy `_pillarbox_humo_silent` still has the same bug class if callable.**  
   Not a soak blocker if truly out of production.

2. **Legacy/non-per-clip `_build_filter_complex` may still be vulnerable if `offset_x` can be odd.**  
   Again, only relevant if that mode is selectable.

3. **The simple-pillarbox scale path may still assume the scaled clip never exceeds the canvas width.**  
   Your LTX case is fine. But if a future source is wider than the canvas aspect after `scale=-2:{canvas_h}`, the pad stage could fail for a different reason. Not caused by this fix, but worth remembering during soak triage.

4. **Concat `-c copy` remains a separate homogeneity risk.**  
   The even-snap fix prevents per-clip creation crashes; it does not guarantee concat safety.

5. **Blend duration/FPS/artistic behavior are still soak concerns.**  
   Especially `-shortest`, 24/25fps cadence, and `lighten` washing out dark scenes.

Final verdict: **Proceed to soak. BUG-LOCAL-030’s active-path fix is sound.**
