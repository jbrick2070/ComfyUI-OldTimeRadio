# Next move: a 120-word all-engine test, or fix the known bugs first?

**Decision needed.** Branch `v2.0-alpha`, HEAD `b15af058`, suite 8231 green.
Everything below is MEASURED on this box in the last 12 hours, with file:line or a
log quote. Nothing is projected.

---

## 1. WHERE THE PIPELINE ACTUALLY IS

### Works, proven by published episodes (45-word canonical, gemma-4-12b, both randomizers)

| leg | result | wall | episode |
|---|---|---|---|
| `fastwan_8gb` | **SUCCESS** | 39 min | 1920x1080, 138 s |
| `ltx_8gb` | **SUCCESS** | 34 min | 1920x1080, 119 s |
| `humo` | **SUCCESS** | 58 min | 1920x1080, 104 s |

3 of 6 legs shipped a complete episode through `obs_publish`. The prior 35-word
Mistral campaign shipped 1 of 6.

### Engine-level: 6 of 7 local video engines render (single-engine smoke, 25 frames)

`wan_ti2v` 72.7 s · `fastwan_8gb` **36.7 s** · `ltx_8gb` 20.3 s · `ltx_video` 83.8 s ·
`humo` 80.5 s · `ltx_audio_in` 177.3 s. `wan_i2v` is NOT INSTALLED (checkpoint
absent; refuses by name). FastWan is **1.98x** the incumbent at VRAM identical to
the megabyte (7270 MB both).

### The 3 campaign failures

1. **`wan_ti2v` TIMEOUT (90 min).** Rendered several clips (peaks 12181 / 12614 /
   10774 MB), then refused: `static frame budget 133 ... affordable 24 frames
   (free=13575 MB)`. **24 is EXACTLY the 1472x832 arithmetic**; at a declared
   832x480 it would be 75. Two stacked defects -- section 2A and 2B.
2. **`ltx_video` FAIL** at `OTR_VideoRenderBatch`, shot `music_opening`. Message
   truncated in the captured log; **not yet root-caused.**
3. **`ltx_audio_in` FAIL** -- "Native GGUF generation exhausted the provider output
   capacity". An OUTPUT-ROOM failure caused by `n_ctx=2048`. **Already fixed**
   (profiles now 4096; Q4_K_M @ 4096 = 9.43 GB, still under the 14.5 ceiling).
   Unverified by a re-run.

## 2. THE KNOWN, UNFIXED DEFECTS

### 2A. `wan_ti2v` declares no render canvas
`render_driver.declared_render_canvas` reads `engine.render_canvas`; `ltx_8gb` and
now `fastwan_8gb` declare one, `wan_ti2v` does not, so it falls through to the
1472x832 shared landscape default whenever the server was not booted with
`OTR_VIDEO_LANDSCAPE_CANVAS`. **Demonstrated live three times.** The campaign
profiles carry an empty `launch.env`, so the incumbent renders 3.07x the pixels.

### 2B. The frame-cost model over-refuses
`FRAME_COST_MODEL["wan_ti2v"] = (7000.0, 185.0)` prices frames linearly. The
four-arm bench measured VRAM **FLAT** across 17 / 49 / 81 frames at 832x480
(6563.1 / 6531.1 / 6563.1 MiB) -- **less than the model's fixed 7000 MB overhead
alone**. `ltx_8gb` DOES scale (6467 -> 6819), so this is per-engine. The standing
ruling forbids refitting from the bench; this campaign is the first PRODUCTION
evidence that a refit is owed.

### 2C. The VRAM fit gate ignores quant AND ctx
`_otr_model_catalog._estimate_resident_gb` prices a `gguf_native` row from ONE
pinned `approx_safetensors_gb` and `_row.context_window` -- never the caller's
`ctx`, and there is no quant parameter. Measured: identical 14.60 GB at ctx 2048 /
4096 / 8192; on disk Q8_0 = 11.80 GB (== the pinned value), Q4_K_M = 6.63 GB. A
1.8x over-estimate that FAILS the 8 GB tier on the model that tier configures.

### 2D. The 8 GB tier cannot load its own writer model
`otr_8gb_wan` / `otr_8gb_ltx` / `otr_8gb_fastwan` pin gemma-4-12b against a
**6.8 GB** ceiling. Correctly estimated it is 8.03 GB (Q4_K_M @ 2048). Even with
2C fixed it does not fit. `gemma-4-E4B-it` (4.5 GB) does.

### 2E. The markup repair ladder is stateless
No assistant turn carries the rejected draft, so "repair the defects, keep the same
wording" is addressed to a model that cannot see the draft -- four rungs, four full
regenerations. The obvious fix (inject the draft) was implemented and **REVERTED**:
`ProviderCapacityMessages` reserves REMAINING context for output, and the
`ltx_audio_in` leg proves that room is already thin. Needs a capacity budget first.

### 2F. `ltx_video` campaign failure -- not root-caused
The only failure with no diagnosis attached.

## 3. THE FORK

### Option A -- run a 120-word all-engine campaign now
* 6 legs at ~2-4x the current beat count. Rough estimate 6-10 h of GPU.
* **For:** longer content is a different regime -- more beats, more
  coverage-planned (multi-clip) segments, more writer output. It may surface
  failure classes a 45-word episode never reaches, and coverage planning is
  exactly where `wan_ti2v` already broke.
* **Against:** 2A/2B are UNFIXED and already demonstrated. A longer run makes
  coverage-planned segments MORE common, so `wan_ti2v` would very likely fail
  again for a reason already understood -- paying hours to re-learn it.

### Option B -- fix the known bugs first, then test
* **For:** 3 of 6 failures are diagnosed; two share one root (2A + 2B). Fixing
  first makes the next campaign a real test rather than a re-demonstration.
* **Against:** 2B is a change to a shipped engine's admission math and 2A changes
  the incumbent's render resolution -- both touch a proven path. A fix without a
  long-form test may itself be under-validated.

### Option C -- something else the panel names.

## 4. WHAT THIS ROUND MUST RULE ON

1. **A or B or C**, with the reasoning, not a preference. If B, the ORDER, and
   which of 2A-2F are in scope versus deferred.
2. **2A specifically:** is declaring `wan_ti2v.render_canvas` safe on a proven
   engine, and what re-proving does it owe? It changes render resolution for any
   boot without the env, which is most of them.
3. **2B specifically:** does production evidence (measured 6563 MiB flat vs a 7000
   MB fixed overhead) license a refit that a standing ruling forbids from bench
   data? If yes, what is the correct shape -- refit the row, or make the overhead
   canvas-scaled, or something else?
4. **Is a 120-word test the right instrument at all** for what we still do not
   know, or is there a cheaper probe that answers the same question? Name the
   specific unknown each proposed run would close.
5. **2F:** how to root-cause `ltx_video` without burning another full leg.

## 5. CONSTRAINTS

- `wan_ti2v`'s frozen RECIPE does not move. Its canvas and cost row are the
  subject of 2A/2B and move only if this round says so, with re-proving costed.
- No fallbacks, no silent degrade. A refusal must name what to change.
- Fail loud and EARLY beats fail loud and late.
- The randomizers stay on; a fix that only works on unrolled defaults is not a fix.
- 16 GB RTX 5080 laptop. A 16 GB success is never physical-8GB qualification.
