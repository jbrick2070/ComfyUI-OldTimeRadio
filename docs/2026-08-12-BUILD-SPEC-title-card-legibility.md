# BUILD SPEC -- make the hero title card legible (start AND end)

**Operator, 2026-08-12:** *"the credits at the start are hard to see ... the
title is big on the screen and does the matrix thing"*, then *"title fix is
first please do"*, then *"and also the title fix card at start and end"*.

**Status: SPECIFIED, NOT BUILT.** Every fact below is verified against the real
Windows files tonight. Nothing here is inferred from a panel's prose.

## 1. The defect

The hero title is phosphor green drawn into the procgen CRT frame, then
composited by `nodes/otr_post_upscale_procgen_blend.py` with the shipped
defaults `_DEFAULT_BLEND_MODE = "screen"`, `_DEFAULT_GREEN_ONLY = True`.

`screen` is `out = A + B - A*B` and can only LIGHTEN. `green_only` zeroes
procgen R and B, so only G moves. Computed contrast for a phosphor glyph
(procgen G=255) over real source pixels:

| source region | source RGB | glyph out | contrast |
|---|---|---|---|
| dark ceiling | (14, 22, 30) | (14, 255, 30) | 8.96:1 readable |
| mid console | (70, 130, 150) | (70, 255, 150) | 1.75:1 |
| **lit monitor** | (100, 220, 230) | (100, 255, 230) | **1.13:1 invisible** |

Observed on `otr/obs/signal_lost_the_ai_acceleration_dilemma_20260812_181530_
..._final.mp4` at t=3s over a control room of lit cyan monitors.

**A black outline drawn into the procgen layer is a mathematical no-op:**
`screen(A, 0) = A`, verified for all three regions. Any darkening -- outline,
shadow, scrim -- is unreachable through a lighten-only blend. This is why the
operator's own suggestion cannot be implemented where the title is drawn today.

**CAVEAT (GPT, R1):** the table above is PRE-ENCODE arithmetic. yuv420p chroma
subsampling, limited/full-range conversion and compression move edge contrast.
The defect is real and was also seen by eye, but the NUMBER must be re-derived
on the final artifact.

## 2. Where everything actually is (verified)

| thing | location |
|---|---|
| title rasterised | `nodes/video_engine.py` `_CRTRenderer._draw_title_card()`, ~line 673, `fill=hero_col` (`CRT_GREEN`) |
| drawn into | the same PIL frame as the whole CRT signature -- scanlines, waveform, ident |
| card schedule | `_CRTRenderer._resolve_card_windows()` ~line 453 -> `self._cards`, each `{start_f, music_end_f, dock_frames, end_f, kind}` |
| **both cards exist** | OPEN (docks into the section-1 ident) and optional CLOSE -- the operator's "start and end" needs no new timing work |
| active-frame test | `for c in self._cards: if c["start_f"] <= fi < c["end_f"]` ~line 523 |
| blend | `nodes/otr_post_upscale_procgen_blend.py`, consumes a FLATTENED `procgen_mp4_path` |

Canonical chain, read from `workflows/otr_canonical.json` links:

```
OTR_SignalLostVideo(12) -> SilentComposite(84) -> PostUpscaleProcgenBlend(93)
                        -> CaptionBurn(86) -> CreditsRoll(95) -> MasterAudioMux(85)
```

## 3. The direction (roundtable R1, driver-verified)

**Emit the title through the text layer that already renders legibly over
anything, downstream of the blend.** `OTR_CaptionBurn` (node 86) runs AFTER the
blend, and `nodes/_otr_captions.py:85-91,210` already builds ASS V4+ styles
carrying `OutlineColour`, `BorderStyle`, `Outline`, `Shadow` -- its own comment
describes *"White text, ~55%-opaque black box (BorderStyle=3)"*. That is the
operator's requested outline, already implemented, already in the right place.

**Rejected, with reasons:**
* **RGBA overlay between nodes** -- dead on transport. The pipeline encodes
  `libx264` / `yuv420p`, which carries no alpha. Would need a codec change or a
  sidecar.
* **Adaptive glyph colour** -- cut by both panel models: luminance sampling plus
  temporal hysteresis, and it still does not deliver the requested outline.
* **Dimming the source under the title** -- cut by both: damages picture content
  to fix an overlay, and risks banding. NOTE: the driver considers a
  time-gated scrim still worth testing as a FALLBACK if the ASS route stalls,
  because the pipeline already uses a 55%-black caption box and the measured
  dark-background case is 8.96:1. It is a fallback, not the plan.

## 4. Build steps

1. **Emit title events.** Add a title-card ASS emitter fed by `self._cards` and
   the existing decode logic (`_DECODE_GLYPHS`, `_decode_line`, `_fit_hero`,
   `_wrap_words`). One event per decode step preserves the Matrix scramble;
   `\pos` + `\t()` express the POP -> dock move. Emit for BOTH `kind` values.
2. **Add a TITLE style** to `_ass_header` in `nodes/_otr_captions.py`: large
   size, phosphor primary, black `OutlineColour`, `BorderStyle=1` with a real
   `Outline` width (start at 3). Keep the SDH style byte-identical.
3. **Suppress the procgen draw** so the title is not rendered twice. This is
   the risky half: `_draw_title_card` also suppresses the section-1
   ident/subtitle while a card is active (`card_active`), so removing the draw
   must NOT change that gating.
4. **Wiring.** Preferred: fold title events into the SAME `.ass` the captions
   already burn -- no node/widget/workflow change at all. Only if that proves
   impossible does this become a canonical JSON change, which lands in the same
   commit per the build law.

## 5. Acceptance -- measured, not eyeballed

GPT's R1 correction stands: `>= 4.5:1` must name WHICH pixels, because an
outline can separate the glyph edge while the fill still measures 1.13:1
against the original monitor. Report three surfaces distinctly:
**original scene**, **outline/matte**, **final adjacent background**.

* a 35-word `viz_green` leg (fastest lane, no `boot_contract`) rendered end to
  end on the real canonical workflow
* contrast measured on the FINAL artifact after yuv420p, not from blend math
* **both** the opening and closing cards, including scramble frames, and the
  BRIGHTEST frame of each interval
* a title-disabled control render proving the CRT signature, scanlines and
  waveform are unchanged outside the title matte and title time ranges
* the standing gates: full suite, Bug Bible, `build_variants --check`, AST, BOM

## 6. Do not lose

* The Matrix decode animation is KEPT. The recipes are not on the table.
* `_OPENING_MUSIC_SUFFIX` and the still-spine work from earlier tonight are
  unrelated and already shipped (`4e49ee4b`, `c9c8e5c0`, `a2a85bcc`, `f70df546`).
* The end-credits SCROLL is a DIFFERENT surface and is already fixed
  (`f70df546`). Do not conflate them -- that conflation cost the operator an
  evening of thinking the title was done.
