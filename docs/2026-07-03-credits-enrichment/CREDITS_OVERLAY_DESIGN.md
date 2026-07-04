# Credits Overlay — design spec (operator prototype "Credits Overlay.dc.html", 2026-07-03)

Source: claude.ai design prototype (a962a063), captured visually (raw HTML is cross-origin,
so this is the faithful transcription). This REPLACES OTR_CreditsRoll's single rolling roll
with the operator's THREE-COLUMN console layout — the old two-panel dossier look, restored,
LATE, with the new model receipts added. Target canvas 1920x1080 (matches the cloud/1080p
delivery). Green/teal CRT monospace; DIM labels + BRIGHT values; bracketed console headers.

## Operator directives (this design)
- **TITLE HIERARCHY (operator tweak 2026-07-03):** the **EPISODE TITLE** (e.g. "NEON TRUTH")
  is the HERO — the single biggest element. **"SIGNAL LOST"** drops BELOW it as a subtitle at
  **50% the episode-title size, same light green** (the series/anthology label, no longer the hero).
- Restore the multi-column STATIC dossier (NOT a single small rolling column).
- Type ~1.5x the old HUD on all columns (readability was the complaint).
- **Add more useful telemetry to COLUMN 1** (operator: "feel free to add more useful telemetry
  in the first column") — pull whatever is durably stamped + genuinely useful (see the telemetry
  menu below; Fable confirms availability).
- Keep the green CRT look over the looped-last-clip backdrop (green-safe: hierarchy by intensity).
- **Palette (from the standalone export):** background `#0a0d0a` (near-black), hero/headers teal
  `#5eead4` (light green), bold monospace; body = mid-green, labels = dim green.

## COLUMN-1 telemetry menu (add the durably-available, useful ones — Fable to confirm)
Beyond MODELS + [SYSTEM]: per-video-engine RECIPE / quant / LoRA (render_engines stamp),
VRAM peak (vram_used_mb / VramPeakProbe), total frames + fps, per-role render time, cast seed +
style seed, git commit, news headline/seed, word counts (target/actual, char/ann), episode duration,
image revision, OpenRouter resolved models + cost (if used). Keep it dense but readable; no field
that isn't a real stamp (no-fallback -> a required-but-missing field raises).

## Layout — 3 columns

### Column 1 (LEFT) — treatment / models / system
- **Hero (line 1, BIGGEST):** the EPISODE TITLE, e.g. `NEON TRUTH` — teal `#5eead4`, bold mono, the
  single largest element on the card.
- **Subtitle (line 2, 50% of hero, same light green):** `SIGNAL LOST` (series label) + dim
  `EPISODE TREATMENT`.
- Sub line (bright/dim mix): `<style> · <runtime> · <resolution> · <date>`
  e.g. `silent_scientific_protest · 1.1 min · 1920x1080 · 2026-07-02`
- **MODELS** (teal header) / dim subhead `GENERATIVE STACK · THIS EPISODE`
  - `IMAGE  stills` .......... `<image engine>`  (bright, right-aligned) — flux2_klein
  - `VIDEO  3 RENDER ROLES`
    - `announcer` ... `<eng> · <family>`  (humo · audio-driven face)
    - `music` ...... `<eng> · <family>`  (ltx_video · text-to-video)
    - `character` .. `<eng> · <family>`  (wan_i2v · image-to-video)
  - `MUSIC  theme` .......... `<music engine> · <cue note>`  (musicgen · closing cue looped)
- **[ SYSTEM ]** bracketed block (dim label : bright value):
  `Host:` iDream OS Windows 11 (AMD64) · `CPU:` ... (24 physical / 24 logical) ·
  `RAM:` 63.4 GiB (peak 27.4 GiB) · `GPU:` NVIDIA RTX 5080 Laptop (15.9 GiB VRAM) ·
  `CUDA:` 13.0 · torch 2.10.0+cu130 · Python 3.12.11

### Column 2 (MIDDLE) — cast & writer
- **CAST & VOICES** (teal header) / dim subhead `DELIVERED VOICE · PERSISTENT`
  - Per cast member: NAME (bright), then `<engine> · <voice_ref> "<signature>"` (dim/green)
    ANNOUNCER — kokoro · bm_fable "documentary relaxed"
    KANE SIRIKIT — indextts2 · vz_bill_boerst "baritone"
    ALICE MALONE — indextts2 · vz_caro_davy "warm alto"
- **[ WRITER / LLM CONFIG ]** bracketed block (dim label : bright value):
  `Creative (A):` <creative model> · `Technical (B):` <technical model> ·
  `Slot routing:` <N> A<->B transitions · `Creativity:` <creativity> ·
  `Temp / top_p:` <temp> / <top_p> · `Words:` target <t> / actual <a> (char <c> / ann <n>)

### Presentation model (OPERATOR LOCK 2026-07-03) — cols 1-2 STATIC, col 3 SCROLLS
Columns 1 and 2 (the dashboard: TITLE/MODELS/LEDGER/SYSTEM + CAST/WRITER) are a STATIC panel,
held the whole time. **Column 3 SCROLLS vertically** — the FULL script cannot fit statically, so
it rolls like the old `_build_right`. NO dropping / no "archive" truncation — the whole script
plays. The credits-tail DURATION is driven by the col-3 scroll length (lead hold + scroll_px/pps +
tail hold), declared to the credits-aware mux; speed the pps up if very long, never cut content.
(This reverts Fable's held-card idea — operator: "the script has to be scrolling.")

### Column 3 (RIGHT, SCROLLING) — story spine + full transcript + intel
Scrolls, in order:
- **[ STORY SPINE ]** — premise + dramatic question + A wants / B wants / ending, from
  `meta.news.script_brief` + `meta.dramatic_state` (same source as the old `_build_hud_dossier`
  STORY SPINE block). Include it here so the "and any other info" rides the scroll.
- **[ CLASSIFIED TRANSCRIPT ]** / dim `EPISODE // <TITLE> · SCENE 1` — then the FULL dialogue:
  per line `SPEAKER [voice]` (bright teal) + wrapped line text (bright green), ledger order.
- **`>> SOURCE INTERCEPT:`** the news seed (optional, omit if absent).
- **`>> DIAGNOSTIC:`** one seeded in-world flavor line (no fabricated numbers).
- Any other narrative/telemetry the operator wants folded into the scroll rides here (col 3 is the
  long lane; cols 1-2 stay the fixed dashboard).

## Data map (all from the DURABLE ledger the late node already reads)
- title/style/runtime/res/date -> meta (episode title, style, duration_s, WxH, date).
- IMAGE/VIDEO/MUSIC engines -> `meta.image_engines.by_role`, `meta.render_engines.by_role`
  (+ the family label per engine), `meta.music_engine`.
- [SYSTEM] -> `_otr_sys_specs.collect_system_specs()`.
- CAST & VOICES -> `cast[].name / voice_engine / voice_ref_id` + `meta.cast_voice_slots[].speech_signature`.
- [WRITER/LLM CONFIG] -> `meta.gen_params_initial` (creative_writing_model, technical_model,
  creativity, temperature, top_p, target_words, slot_transitions) + meta word counts.
- CLASSIFIED TRANSCRIPT -> the ledger dialogue lines (speaker + text), same source the old HUD used.

## Implementation note (OTR renders credits as PIL frames, not HTML)
The `.dc.html` is the VISUAL TARGET. OTR_CreditsRoll renders the credits as video frames via
PIL/ffmpeg. The existing `video_engine._TelemetryHUDRenderer` (`_build_left` + `_build_right`)
already draws the old two-panel CRT dossier in PIL — this design extends it to 3 columns + the
model receipts + a MUCH bigger title, driven by the durable receipts, composited over the
looped backdrop, held for the credits-aware duration (static columns; transcript scrolls if long).
No-fallback: a missing receipt RAISES (unchanged contract).
