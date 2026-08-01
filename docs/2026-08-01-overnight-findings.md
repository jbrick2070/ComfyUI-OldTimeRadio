# Overnight run, 2026-08-01 -- findings and open items

Written while the 45-word gemma-4 campaign runs. Everything below is measured on
this box, committed and pushed. Read this before picking the work back up.

---

## 1. WHAT SHIPPED (7 commits, suite 8228 green)

| # | Change |
|---|---|
| `40963c1b` | DMD restart sampler moved into the pack; `FALLBACK-flow` -> named refusal; false `FRAME_MOTION_FLOOR` comment corrected |
| `57ce052d` | The `wan_ti2v` recipe seam -- five class-level owners so a subclass can own its recipe |
| `66846467` | `fastwan_8gb` adapter + registration (three silent traps pinned) |
| `b07c2de0` | Tier profile, asset manifest, honest licence gate |
| `0e9bbffe` | Profile shape fix + the test gap that hid it |
| `859b1a1c` | Live gate: 81 frames @ 832x480, real DMD transition; cap 17 -> 81; two identity leaks |
| `4ccfe3a8` / `265c4beb` | `otr_w45_fastwan`; `otr_g4_*` canonical-gemma campaign profiles |
| `15c530a0` | Story-writer transport-decoration canonicalizer |

## 2. PROVEN LIVE

* **`fastwan_8gb` renders.** 81 native frames, 832x480, h264, 25 fps, **69.9 s**,
  `noise_scaling=ModelSamplingAdvanced` (NOT the fallback). Bench predicted 65.3 s.
* **Six local video engines render** through the executor thread: `wan_ti2v`
  (72.7 s), `fastwan_8gb` (**36.7 s**), `ltx_8gb` (20.3 s), `ltx_video` (83.8 s),
  `humo` (80.5 s), `ltx_audio_in` (177.3 s). At 25 frames on the same still,
  **FastWan is 1.98x the incumbent at VRAM identical to the megabyte (7270 MB)**.
* **`wan_i2v` is NOT INSTALLED** -- checkpoint absent, refuses by name. Not a failure.
* **A complete episode published** end to end (`ltx_8gb`, 35-word campaign):
  1920x1080, 2973 frames, 118.9 s, AAC stereo, 51.3 MB, in `otr/obs/`.

## 3. OPEN DEFECTS -- OPERATOR DECISIONS

### 3A. The 8 GB tier cannot load its own configured model
`otr_8gb_wan`, `otr_8gb_ltx`, `otr_8gb_fastwan` all pin `gemma-4-12b` against a
**6.8 GB** ceiling. Correctly estimated it is 8.03 GB (Q4_K_M @ 2048), so even
with the estimator fixed it does not fit. Either the ceiling is wrong for a 12B
or the tier wants a smaller model (`gemma-4-E4B-it` = 4.50 GB, PASSES).
**Not fixed: changing three shipped tiers is an operator call.**

### 3B. The VRAM fit gate ignores quant AND ctx
`_otr_model_catalog._estimate_resident_gb` (`:1476-1525`) prices a `gguf_native`
row from ONE pinned `approx_safetensors_gb` and `_row.context_window` --
**never the caller's `ctx`, and there is no quant parameter at all.**

    measured: identical 14.60 GB estimate at ctx 2048, 4096 AND 8192
    on disk:  Q8_0 = 11.80 GB (== the pinned value), Q4_K_M = 6.63 GB
    so:       a tier pinning Q4_K_M is priced at Q8_0 -- a 1.8x over-estimate

Both panels confirmed it; codex correctly ruled it belongs in its own plan.

### 3C. `wan_ti2v` still has no declared canvas
Proven live this session, and it is not theoretical. On a server booted without
`OTR_VIDEO_LANDSCAPE_CANVAS` the incumbent fell through to 1472x832 and the cost
model allowed **23 frames** where a 161-frame segment was planned:

    MotionBudgetError: static frame budget 161 ... affordable 23 frames
    (free=13359 MB)   <- 23 is EXACTLY the 1472x832 arithmetic

`fastwan_8gb` declares its canvas and would have been priced at 832x480 (72
frames). r4 cut the incumbent declaration from that build, correctly at the time;
the case is now much stronger.

### 3D. The cost model refuses renders that demonstrably fit
Measured peak for a whole 81-frame render at 832x480 is **6563 MiB** -- LESS than
the model's fixed **7000 MB overhead alone**. Bench also measured VRAM FLAT across
17/49/81 frames (6563.1 / 6531.1 / 6563.1), while the model prices frames
linearly. `ltx_8gb` DOES scale, so this is per-engine, not universal.
The standing ruling forbids refitting from the bench; this session produced the
**production** evidence that a refit is owed.

### 3E. The repair ladder is stateless
No assistant turn carries the rejected draft, so "repair the defects, keep the
same wording" is addressed to a model that cannot see the draft. Four rungs, four
full regenerations, same habit. **Fix implemented then REVERTED:**
`ProviderCapacityMessages` reserves REMAINING capacity for output and
`prompt_no_room` refuses before the call, so at `n_ctx=2048` injecting a draft
risks converting a recoverable defect into a deterministic one. Needs a capacity
budget first.

## 4. CORRECTIONS TO EARLIER CLAIMS IN THIS SESSION

Recorded so they are not repeated:

* "The incumbent renders production at 1472x832" -- **refuted by ffprobe** on the
  2026-07-23 clips (all 832x480). The live-vs-bench VRAM spread is a
  MEASUREMENT-SCOPE change (`2b095143`, 2026-07-20), not canvas.
* "Ship `fastwan_8gb` at 17 frames" (r4) -- **revised.** 17 is the motion FLOOR
  used as a ceiling and cannot serve a CHAINABLE engine; the canonical run refused
  a 177-frame planned segment. Now 81, the highest MEASURED rung.
* "The writer fails after an hour of stills+audio" -- **false.** Writer legs died
  in 1.5-11 min; the 82-minute leg was `wan_ti2v` failing at VIDEO.
* "The ladder retries with the same instruction" -- **false.** It appends the
  defect list and a stage-direction repair note.
* "ctx inflates the VRAM estimate" -- **false.** Identical at 2048/4096/8192; the
  driver is quant.

## 5. STILL OWED

1. The 45-word gemma-4 campaign result (running).
2. Seed stability: two cold FastWan renders compared by DECODED-FRAME hash.
3. FastVideo upstream revision pinned in the manifest (sigma digest is pinned).
4. The Kijai extraction's licence notice -- `commercial_clean` stays **False**.
5. Repeated-session teardown across alternating `wan_ti2v` / `fastwan_8gb`.
