# Question -- 2026-05-03

# Code review: BUG-LOCAL-030 even-snap hardening fix (commit df317ac)

## Context

Previous round-robin (commit `1b81545` transcripts at `docs/2026-05-03-bug-030-architecture-review__01_chatgpt.md` + `__02_gemini.md`) flagged a yuv420p odd-pixel pad bug in the simple-pillarbox formula. Gemini caught it, ChatGPT missed it. Naive `pad=W:H:(ow-iw)/2:(oh-ih)/2` produces `x=15` for LTX 832x480 → scaled 1442x832 → padded into 1472x832 (odd offset → ffmpeg may hard-crash on yuv420p chroma alignment, or auto-snap with warning on newer builds).

Just shipped fix in commit `df317ac`. Want a peace-of-mind round-robin verifying the fix is correct + flagging any residual risk before the soak. **NO new fixes from you — verdict only.**

---

## The fix (4 surgical edits)

### Edit 1: simple-pillarbox formula in `_layered_per_clip_silent` ELSE branch

**Before:**
```python
vf = (
    f"scale=-2:{canvas_h}:force_original_aspect_ratio=decrease,"
    f"pad={canvas_w}:{canvas_h}:(ow-iw)/2:(oh-ih)/2:color=black,"
    f"fps={canvas_fps}"
)
```

**After:**
```python
vf = (
    f"scale=-2:{canvas_h}:force_original_aspect_ratio=decrease,"
    f"pad={canvas_w}:{canvas_h}:"
    f"trunc((ow-iw)/4)*2:trunc((oh-ih)/4)*2:color=black,"
    f"fps={canvas_fps}"
)
```

### Edit 2: layered-branch overlay X (defensive)

**Before:**
```python
overlay_chain = "[bg][pillar]overlay=x=(W-w)/2:y=0[v]"
```

**After:**
```python
overlay_chain = "[bg][pillar]overlay=x=trunc((W-w)/4)*2:y=0[v]"
```

Same change applied to the `extend_tail_s > 0` variant of `overlay_chain`. Pillar 512 in canvas 1472 currently gives 480 (already even) — change is purely defensive against future widget tweaks (e.g. `humo_pillar_width=510` would otherwise crash).

### Edit 3: rename `out` → `output_path`

In `nodes/otr_post_upscale_procgen_blend.py` `PostUpscaleProcgenBlend.blend()` method. Pure rename, no behavior change. Reason: Bug Bible BUG-01.02 requires `OUTPUT_NODE = True` files to contain one of: `get_output_directory | folder_paths | _REPO_ROOT | output_dir | output_path`. The rename satisfies the audit while improving readability.

### Edit 4: 3 new tests

`tests/test_video_composite_layered.py`:
- `test_pad_offset_math_produces_even_for_ltx_dim_mismatch` — pure-Python repro of the ffmpeg expression; locks in the math (LTX 1442 → 14, HuMo 480 → 496, edge cases for 0/1/2/3/4-pixel gaps).
- `test_overlay_offset_in_layered_branch_is_even_snapped` — asserts `overlay=x=trunc((W-w)/4)*2:y=0` is in the cmd AND naive `(W-w)/2` is NOT.
- 2 existing tests updated with `assert "trunc((ow-iw)/4)*2" in vf` + `assert "(ow-iw)/2:(oh-ih)/2" not in vf` sentinels.

---

## Math sanity check

For canvas_w=1472, canvas_h=832:

| Input dims | Naive `(ow-iw)/2` | New `trunc((ow-iw)/4)*2` | Even? | Δ from naive |
|---|---|---|---|---|
| HuMo 480x832 | x=496, y=0 | x=496, y=0 | ✓ | unchanged |
| LTX 832x480 → scaled 1442x832 | x=15 (odd!), y=0 | x=14, y=0 | ✓ | -1 px (off-center) |
| Equal dims (1472x832) | x=0, y=0 | x=0, y=0 | ✓ | unchanged |
| 1px gap | x=0 | x=0 | ✓ | unchanged |
| 2px gap | x=1 (odd!) | x=0 | ✓ | -1 px |
| 3px gap | x=1 (odd!) | x=0 | ✓ | -1 px |
| 4px gap | x=2 | x=2 | ✓ | unchanged |

Worst case off-center: 1 pixel. For 1442-px-wide LTX content in 1472 canvas, the 1-pixel asymmetry (14px black left, 16px black right vs symmetric 15+15) is imperceptible.

---

## Asks (please answer all)

### 1. Is `trunc((ow-iw)/4)*2` correct + idiomatic for ffmpeg expressions?

The expression syntax: ffmpeg's filter expression eval supports `trunc()`, `floor()`, `ceil()`, `round()`. Multiplication / division work as expected. Order of operations: `trunc((ow-iw)/4)*2` should evaluate as `trunc(((ow-iw)/4))*2` — confirm.

Alternative formulas to consider:
- `floor((ow-iw)/4)*2` — same as trunc for positive values; may differ for hypothetical negative (input bigger than canvas — pad filter doesn't accept negative anyway, but mention if relevant)
- `((ow-iw)/2 - mod((ow-iw)/2, 2))` — explicit "subtract odd remainder"
- `2*trunc((ow-iw)/4)` — algebraically equivalent

Is there a more idiomatic / safer ffmpeg pattern I should know about?

### 2. Does the 1-pixel off-center matter visually?

For LTX content (1442px wide centered in 1472 canvas):
- Old: 15px black left + 15px black right (symmetric, but odd offset = crash risk)
- New: 14px black left + 16px black right (asymmetric, 1px off-center)

Visually 14 vs 16 is invisible at viewing distance. But the post-RTXUpscale procgen blend will overlay this. Will the procgen overlay reveal the 1-pixel asymmetry as a visible seam, or is it lost in the lighten blend?

### 3. Did I miss any other odd-pixel pad sites in the codebase?

I only fixed the simple-pillarbox formula in `_layered_per_clip_silent` + the layered-branch overlay X. Other places in the codebase that might have similar `(W-w)/2` or `(ow-iw)/2` patterns:
- The legacy `_pillarbox_humo_silent` helper (kept for back-compat, used by tests, NOT used by the active per-clip-mux renderer) still uses the naive form. Should I fix it too even though it's not in the production path?
- The post-upscale procgen blend (`OTR_PostUpscaleProcgenBlend._build_blend_cmd`) — uses `crop=iw:ih` and `setpts=PTS-STARTPTS` only; no explicit pad/overlay offsets. Procgen and source are both 1920x1080 so no scale/crop math should produce odd offsets.
- The OTR_VideoComposite legacy filter chain (`_build_filter_complex` for the non-per-clip-mux mode) uses `overlay=x=offset_x:y=0` where `offset_x` is computed Python-side per-clip — does this path have the same risk?

### 4. Residual risk after this fix

What's left that could still crash or produce invisible video on the next soak? Specifically:
- Concat demuxer with `-c copy` if the per-clip pillarbox output streams have any inconsistent SAR/DAR/pix_fmt
- `-shortest` behavior with the procgen blend if procgen and source durations differ slightly
- FPS mismatch (24fps procgen vs 25fps source) producing judder in the blend
- `lighten` mode at 0.5 opacity washing out dark noir HuMo scenes

### 5. Per-element follow-up-fix probability estimate

For each of these THREE elements (the only things touched in commit `df317ac`):

- **Element 1:** even-snap pad formula in `_layered_per_clip_silent` ELSE branch (the actual crash-prevention fix)
- **Element 2:** even-snap overlay X in layered branch (defensive — current widgets give even result)
- **Element 3:** rename `out` → `output_path` in PostUpscaleProcgenBlend (pure rename to satisfy Bug Bible audit)

Give a percentage estimate of how likely YOU think a follow-up fix will be needed in the next 2 weeks of soak runs (0% = bulletproof, 100% = will definitely break in production). Brief reasoning.

### 6. What did I miss?

Open-ended.
