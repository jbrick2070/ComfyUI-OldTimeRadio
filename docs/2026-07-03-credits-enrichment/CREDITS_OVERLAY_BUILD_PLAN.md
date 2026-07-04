# Credits Overlay — CODE-READY BUILD PLAN (v1)

**Date:** 2026-07-03  **Branch:** v2.0-alpha  **Status:** build-ready (operator-approved; "editing this later" expected).
Design source: `docs/Credits Overlay - plain.html` + `docs/2026-07-03-credits-enrichment/CREDITS_OVERLAY_DESIGN.md`.
Grounding: 2 Fable passes + a delta, all against the real Windows files. This REPLACES OTR_CreditsRoll's
single rolling roll with the operator's 3-column console. Node I/O + workflow JSON UNCHANGED (the S3
wiring 93->95->85 stands) -> zero JSON delta. No-fallback contract preserved: a missing RECEIPT raises;
only probe fields ([SYSTEM]/VRAM) + pure flavor text may be soft.

## 0. Presentation model (operator lock)
- **Columns 1 + 2 = STATIC dashboard**, held the whole time.
- **Column 3 = SCROLLS** the full narrative (spine + full transcript + intel). NOTHING dropped.
- Rendered as: one STATIC base PNG (bg + cols 1-2 + col-3 header/footer frame) + one TALL col-3
  content PNG that ffmpeg scrolls. 0.6s fade-in / 0.8s fade-out; video ends at credit-end.
- **Duration (declared tail):** `dur = LEAD_HOLD(3) + scroll_px/pps + TAIL_HOLD(4)`, where
  `scroll_px = max(0, transcript_h - col3_view_h)`. If `dur` would exceed a sane ceiling, SPEED UP
  (`pps = scroll_px/(CEIL - holds)`) — never truncate. `dur` is the existing
  `declared_credits_tail_s` FLOAT output; mux guard already credits-aware (no mux change).

## 1. Layout (block order, data source, tier). RAISES=receipt; soft=probe/flavor; omit=frozen fact omitted if absent.

### COL 1 (STATIC) x 56-654, rule x 698
1. HERO = `meta.episode_title` upper — tier HERO, auto-shrink 76->48px to fit 598px, wrap to 2 lines below floor, never truncate. RAISES.
2. Subtitle `SIGNAL LOST` at 50% resolved-hero, teal — tier SUBTITLE; beside it dim `EPISODE TREATMENT` — TAG.
3. Meta strip `<style> · <WxH> · <date>` — FINE. style=`meta.style` (NEW STAMP, S-A below); WxH from `_probe_video`; date=meta timestamp else render clock (soft).
4. `MODELS` H1 + `GENERATIVE STACK · THIS EPISODE` TAG.
   - `IMAGE` SUBHEAD + `REV {image_engines.image_revision}` MICRO; rows `role -> engine` from `meta.image_engines.by_role` (RAISES; empty by_role -> existing `(no stills dispatched…)`).
   - `VIDEO` SUBHEAD + `{n} RENDER ROLES · REV {render_engines.video_revision}` MICRO; rows `role -> engine` from `meta.render_engines.by_role`; MICRO suffix per row = `recipe · quant · lora@x · canvas · family` from `render_engines.by_engine[eng]` (only non-None printed; all-None = no suffix). RAISES.
   - `MUSIC` SUBHEAD; `theme -> meta.music_engine` + `· closing cue looped` MICRO. RAISES.
5. `[ PRODUCTION LEDGER ]` H2 grid: `VRAM:` `render_engines.vram_peak_mb` (+ `of {gpu_total}` soft); `FRAMES:` `Σclips.frames @ fps · {clip_count}` (manifest input, omit); `SEED:` `cast_contract.cast_seed` (+ `seed_source` omit); `COMMIT:` `_git_short_sha()` RAISES; `REV:` `img{image_revision}·vid{video_revision}`.
6. `[ SYSTEM ]` H2 grid Host/CPU/RAM/GPU/CUDA from `collect_system_specs()` — all soft ("(unknown)").
7. spacer; footer `Made with OTR v2.0-alpha — 100% generated` teal FOOTER.

### COL 2 (STATIC) x 742-1166, rule x 1210
1. `CAST & VOICES` H1 + `DELIVERED VOICE · PERSISTENT` TAG.
2. Cast entries from `ledger.cast` (RAISES): NAME (bold); line2 `engine · voice_ref_id` LABEL + `"speech_signature"` from `meta.cast_voice_slots[char_id]` (omit if absent). 1-char grace = extra gap, no filler.
3. rule; `[ WRITER / LLM CONFIG ]` H2 grid: Creative(A)/Technical(B) RAISES; Slot routing/Creativity/Temp-top_p from `gen_params_initial` (omit); `Words: target N/actual M (char x/ann y)` (target=`gen_params_initial.target_words` omit; actuals computed from `led.lines`).
4. spacer; footer `>> voices from the CastLock final stamp — delivered, not planned.`

### COL 3 (SCROLLS) x 1256-1864 — tall content canvas
1. `[ STORY SPINE ]` H1 — premise `meta.news.script_brief` + `meta.dramatic_state` (dramatic_question / character_a_wants / character_b_wants / ending_change). Same source as old `_build_hud_dossier` STORY SPINE; each line omit-if-absent (frozen facts).
2. `[ CLASSIFIED TRANSCRIPT ]` H1 + `EPISODE // {TITLE} · SCENE 1` TAG — then FULL dialogue from `led.lines` (order via `_otr_ledger_consumers.iter_lines` + `speaker_name`): `SPEAKER [voice_ref]` SPEAKER-tier + wrapped text BODY. Announcer `[voice]` from cast stamp (omit if unresolvable).
3. `>> SOURCE INTERCEPT: {meta.news.script_brief}` FOOTER (omit if news None).
4. `>> DIAGNOSTIC:` one line chosen by `cast_seed % len(pool)` (deterministic; pool may echo REAL stamps — vram/cast count — never invent a number).

## 2. PIL constants (1080p)
Geometry: margins L/R 56, top 48, bottom 40; col1 56-654, col2 742-1166, col3 1256-1864; rules 1px #4ade80@alpha51.
Fonts (JetBrains Mono via `_load_font` chain; bold=700 else stroke_width=1): HERO 76->48 #5eead4; SUBTITLE 50%hero #5eead4; H1 28 #5eead4; H2 22 #5eead4; SUBHEAD 18 #a7f3d0; NAME 22 #bbf7d0; SPEAKER 19 #5eead4; VALUE/BODY 20 #bbf7d0; LABEL 20(19) #4ade80@140; GRID 16 (label#4ade80@140 / value#bbf7d0); TAG 15 #4ade80@140; MICRO 14 #4ade80@110; FOOTER 15-16 #4ade80@115. line-gap 1.28x.
Compositing (one static base PNG; col3 separate tall PNG): (a) radial bg #241017 center (1574,454)->#0a0d0a->#050705, gen 480x270 upscale, alpha 225 so backdrop ghosts (BUG-410 look); (b) neon bar 30x520 @x1832 y54, 4-stop pink->cyan, GaussianBlur(26) alpha128; (c) glow: text layer + GaussianBlur(6)@45% under sharp; hero/subtitle extra GaussianBlur(12)@35%; (d) sharp text; (e) scanlines every 4px 1px black@66. Flicker skipped (held card + live backdrop).
ffmpeg: `-stream_loop -1` backdrop, scale/pad, `eq=brightness=-0.25`, overlay base@0,0, overlay col3 via `crop=COL3_W:COL3_VIEW_H:0:'clip((t-LEAD)*pps,0,scroll_px)'` @COL3_X,COL3_Y, `-t dur`, `fade in 0.6 / out 0.8`, `-an`; concat unchanged.

## 3. Functions in nodes/otr_credits_roll.py
- `build_credits_sections` -> `build_credits_layout(led, *, w, h, manifest)` returning `{hero, subtitle, col1_blocks, col2_blocks, col3_flow}`; keep every `_require`/raise; NEW required `meta.episode_title`, `meta.style`; lines as (label,value,tier) tuples.
- `render_roll_canvas` -> `render_static_base(layout,w,h)` (RGBA full frame: bg+neon+cols1-2+col3 header/footer+scanlines) and `render_scroll_canvas(col3_flow,col_w)` (tall RGBA).
- `compute_roll_duration_s` -> `compute_credits_duration_s(scroll_px, pps)` per S0.
- `render_credits_clip` keep signature/contract; internals = 3 inputs (backdrop/base/scroll) + the crop-scroll filter + fades.
- KEEP `plan_backdrop`, `append_credits`, `_git_short_sha`, `_gpu_name`, node I/O (2 forceInput, 3 outputs, zero widgets). Extend `_probe_video` to also return `duration`. Add helpers `_load_font`(exists), `_fw`,`_fh`,`_draw_wrapped`,`_two_tone_row`,`_autoshrink_pt`.

## 4. Stamp additions (land in the SAME atomic change; else `_require` raises at render)
- **S-A style tag:** stamp `meta.style` durably (it IS stamped at `OTR_LedgerScriptWriter.py:5417` on the writer singleton — CONFIRM it reaches `get_ledger().data`; if only on the wire ledger, add a `stamp_durable` copy like the S2 credits work). Operator: ADD IT.
- **S-B per-engine recipe/family:** add `by_engine[eng] = {recipe,quant,use_lora,render_canvas,family}` to `_build_render_engines_payload` (`otr_video_render_batch.py:26-58`) from the manifest clip rows (`clip["family"]` at `render_driver.py:2314`; recipe/quant/lora already available there). Image side: add `family_by_engine`/`family` in the image dispatcher stamp via the image-registry adapter `.family`. Additive sibling keys — existing `by_role` consumers (`video_engine.py:1832/1904`) untouched.

## 5. Tests (tests/test_credits_roll_spec.py)
- Rewrite: duration (scroll model), canvas (static base fixed WxH + tall scroll canvas grows), clip-render duration math, hero-autoshrink (>=1 line, floor), title/style missing -> raise.
- Update call-shape (receipt logic preserved): the motion/images/cast/music/seed/planned-voice tests; `_led()` fixture gains `episode_title`, `style`, `render_engines.by_engine{recipe,family}`, `dramatic_state`.
- New: recipe/family suffix rendered from by_engine; STORY SPINE block from dramatic_state; SOURCE INTERCEPT omit-if-news-None; diagnostic deterministic per cast_seed + never prints a fabricated number; scroll-never-drops (full transcript present in the tall canvas).
- Stamp unit tests: `_build_render_engines_payload` by_engine shape (extend the render-batch test); style durable-stamp presence.
- `tests/test_hud_dossier_bug3.py` untouched (node 12 keeps its early dossier). Full suite + Bug Bible + B7 after; commit+push per green chunk.

## 6. Risks (top 3)
1. **Recipe/family + episode_title/style now required** — any pre-existing soak/test ledger lacking them raises at the terminal node. Sweep fixtures + land stamps + reader + tests in ONE commit.
2. **ffmpeg crop-scroll y-expression** off-by-one at clamp bounds; duration frame-count test must cover both scrolling and `scroll_px==0` (short script) branches.
3. **Fontless fallback** (`load_default()` ignores pt) would render a tiny hero — raise CreditsDataError if no truetype resolves (consistent with no-fallback).

## 7. Gate + verify
Land atomic (renderer + 2 stamps + tests + fixtures). Suite + Bug Bible + B7 green, push per chunk. Then a Fable final grounded gate (CLAUDE.md §9 — production render-path). Live smoke: render a SHORT + a LONG (long transcript) episode, confirm the held dashboard + full col-3 scroll land in otr/obs, last frame is the scroll tail (not black), body audio byte-identical, no mux ValueError. Operator eyeball ("editing this later").
