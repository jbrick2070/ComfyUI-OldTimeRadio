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

## 3A. DE-RISK PASS (Opus, 2026-08-12 late) -- READ BEFORE BUILDING

The driver flagged "suppressing the procgen draw without disturbing the
`card_active` gating" as the risky half. **That framing was wrong**, and the
real risk is elsewhere. Findings, each cited:

**The gating is trivial.** `card_active` is assigned once (`video_engine.py:527`)
and read once (`:655`) -- a single if/else, no other consumer repo-wide.

**But `_draw_title_card` draws FIVE things, not one**, and two exist ONLY on
card frames:

| element | non-card frame | card frame |
|---|---|---|
| ident line at `_ident_xy` | solid `=== SIGNAL LOST ===` (`:665`) | the same line, DECODE-SCRAMBLED (`:764-768`) |
| subtitle `"<title>"` | drawn (`:667`) | not drawn |
| mm:ss timestamp | drawn (`:669-671`) | not drawn |
| carrier meter | not drawn | drawn (`:769`) |
| hero title block | -- | centred -> docked (`:771-828`) |

So **an early return from `_draw_title_card` is WRONG** (it also kills the
carrier decode and meter -- the carrier scramble IS part of the kept Matrix
look), and **forcing `card_active` False is WRONG** (kills those two AND adds
the subtitle + timestamp from frame 0).

**THE MINIMAL DIFF: suppress AFTER the carrier block.** Insert between
`video_engine.py:769` and `:771` -- after the carrier line and meter, before the
hero block -- guarded by a helper defaulting to OFF with an
`OTR_HERO_TITLE_IN_PROCGEN=1` escape hatch for the acceptance control render.
Two lines of behaviour plus the helper; `card_active`, `_cards`,
`_resolve_card_windows`, `_draw_ident` and `_draw_carrier_meter` all untouched.
No widget, no `INPUT_TYPES` change, therefore NO canonical JSON change.

**THE DOCK: keep it in ASS, option (a).** At the end of the dock the hero font
shrinks to exactly `self._title_size` (`:386,:388`) and its position lerps to
`ident_x, ident_y` (`:806,:812-813`) -- it lands ON the carrier line and is then
hard-cut back to the ident at `end_f`. It is a ~0.4 s handoff, not a state.
Express it as an ASS `\move` from centre to the `_ident_xy` anchor: `_ident_xy`
is (40, 20) at 1920x1080 and `_ass_header` pins `PlayResX/Y` to 1920x1080
(`_otr_captions.py:222-223`), so no new math.

### THE ACTUAL DANGER -- fix this BEFORE deleting a single glyph

**`OTRCaptionBurn` has FOUR no-error passthrough exits, and `burn_captions`
DEFAULTS TO FALSE** (`otr_caption_burn.py:172-174`). It returns the input video
untouched when: captions are off (`:220-221`), no timed ledger resolves
(`:224-226`), or the burn raises `ValueError` (`:232-235`) -- which includes
`build_ass_from_ledger` returning `None` (`:131`), itself returning
`(None, ...)` when the ledger will not load (`_otr_captions.py:251-252`) **or
when there are no speech lines** (`:335-336`).

Today that is harmless because the title is baked into pixels. The moment the
title lives only in ASS, every one of those paths ships an episode with **NO
TITLE** and a cheerful log line. **Ruling: title events must NOT be gated behind
`burn_captions`, and the title emission must be a REQUIRED save that refuses
rather than passes through** -- the same discipline as `61ae356c`. The
`if not events: return (None, ...)` at `_otr_captions.py:335` must also be
reordered so title-only output is still valid ASS.

### Two more, verified

* **The card windows cannot be recomputed caption-side.** `self._cards` comes
  from `_resolve_title_timing(led, volume, fps, total_frames)`
  (`video_engine.py:2392`), which needs the per-frame VOLUME ENVELOPE for its
  fallback (`:295-297`) and `total` for the synthesized close (`:323-334`).
  `build_ass_from_ledger` takes a ledger path and works in seconds
  (`_otr_captions.py:241`). So the emitter MUST live in `video_engine` and hand
  CaptionBurn a sidecar. Step 1's "fed by `self._cards`" hid a real cross-node
  dependency; reimplementing the window math caption-side will drift silently.
* **Clamp the closing card to the MAIN video, not the encoded file.** The
  synthesized close sets `music_close_end_f = total` (`:333`) but the encode
  appends the HUD post-roll (`total_encode_frames = total_frames + _hud_frames`,
  `:2395`). An ASS end time taken from the final mp4 hangs the closing title over
  the post-roll and into the credits.

### Struck from this spec

The "time-gated scrim" fallback in section 3 is **unreachable and is withdrawn**.
The blend is a single global `blend=all_mode=` over the whole frame
(`otr_post_upscale_procgen_blend.py:390-400,:437-439`) with no per-region and no
time gating, and `_BLEND_MODE_CHOICES` (`:135-142`) offers no mode where procgen
darkens source while `screen`+`green_only` is in force. Any scrim means changing
the blend node itself -- MORE work than the ASS route. **The ASS route is the
only plan.** Recolouring the hero white also fails: over the measured lit
monitor it screens to (255,255,255) against a ~200-luminance source, still under
2:1.

### The regression test that does not exist today

**No test constructs `_CRTRenderer` or calls its `render()`.** Existing title
coverage is `_title_reveal_progress` (`tests/test_video_render_path_cw4.py:645`,
pure function), `_draw_crt_overstrike_text` on a standalone canvas (`:662`), and
`_resolve_title_timing` window arithmetic (`test_video_ledger.py:495,522`).
Nothing anywhere asserts the ident is on screen, so a silent ident loss ships
green. Add two headless PIL tests (no ffmpeg, no GPU): one asserting the ident
slot has phosphor ink DURING a card (carrier decode) and on the first frames
AFTER it, and one asserting the centre has zero hero ink by default and non-zero
under `OTR_HERO_TITLE_IN_PROCGEN=1`. Use `draw_scopes=False` so the centre
assertion is not polluted by the ring/particles/waveform; the grid's green is
capped at `15 + vol*25` (`:596`), far below a 90 threshold.

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
