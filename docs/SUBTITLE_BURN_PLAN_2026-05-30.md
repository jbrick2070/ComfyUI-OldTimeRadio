# OTR v2.0-alpha -- Burned-In Open Captions (SDH) -- Coding & Proposal Plan (2026-05-30)

**Branch:** `v2.0-alpha` | **HEAD at authoring:** `bfc761a` (LoRA consolidated, stable)
**Goal:** Burn always-on **open captions / SDH** (Subtitles for the Deaf and Hard of hearing)
into the final 1920x1080 deliverable, synced to the dialogue audio, including speaker
labels and sound/music cues.
**Stance:** additive, toggleable, audio-safe. Honors all Prime Directives.

---

## 0. Feasibility (verified 2026-05-30)

- **Timing already exists.** `nodes/_otr_ledger.py` (schema `l3-2026-05-14`) writes
  `lines[]` with `text`, `speaker_role`, `char_id`, `start_s`, `dur_s`. Caption timecodes
  are `(start_s, start_s + dur_s)` -- **no derivation needed**. Display names come from
  `cast[]` cross-referenced by `char_id`.
- **ffmpeg has libass.** Build reports `--enable-libass`; `subtitles`, `ass`, and `drawtext`
  filters all present. ASS burn is the recommended path.
- **Audio is safe.** All three final passes (VideoComposite mux, RTXUpscale, PostUpscaleProcgenBlend)
  already use `-c:a copy`. Captions touch the **video stream only**, so audio stays
  byte-identical (Prime Directive 1).

---

## 1. Method decision -- ASS (libass), not SRT or drawtext

| Option | Verdict |
|---|---|
| **SRT + `subtitles` filter** | Simple, but styling is global and crude. No reliable per-speaker color, no box control. |
| **`drawtext` per line** | Full control but we'd hand-build timing `enable='between(t,a,b)'` per line, manual wrapping, no SDH conventions. Brittle. |
| **ASS (.ass) + `ass`/`subtitles` filter** | **CHOSEN.** Per-style + inline overrides: opaque box, exact position, per-speaker color, italics for sound cues, auto-wrap. libass renders it crisply at 1080p. |

**Burn location: fold into `PostUpscaleProcgenBlend`'s existing `filter_complex`.**
That pass already does the only `libx264` video encode in the tail. Appending the subtitle
filter to the `[v]` chain burns captions **in the same encode** -- no extra re-encode pass,
no added quality loss. Audio remains `-c:a copy`.

```
# current (paraphrased):
[1:v]scale...blend=all_mode={mode}:all_opacity={opacity}:shortest=1[v]
# proposed when captions on:
[1:v]scale...blend=...:shortest=1[vb];[vb]ass='<id>_captions.ass'[v]
-map "[v]" -map "0:a?" -c:v libx264 -c:a copy
```

Burning at the **final** stage (post-upscale, native 1080p) keeps text sharp -- captions are
never upscaled/softened, and they sit on top of the procgen overlay so nothing covers them.

---

## 2. Pipeline -- new pieces

1. **`scripts/otr_captions.py` (new module, tracked): `build_ass_from_ledger(ledger_path, style)`**
   - Resolve paths via `nodes/_otr_paths.py` (NOT a hardcoded glob).
   - Read `lines[]`; skip/translate non-dialogue roles:
     - `character` / `announcer` -> spoken caption, prefixed with speaker label.
     - `music_*` -> `[MUSIC]` or a music-note cue, only if no concurrent dialogue.
     - `sfx` -> bracketed sound cue (e.g. `[STATIC HISS]`) from the line text/tag.
   - Cross-ref `cast[]` by `char_id` for the display NAME and per-speaker color slot.
   - Apply SDH line rules (section 4); emit V4+ ASS with one `[V4+ Styles]` + dialogue events.
   - Write `<id>_captions.ass` into the episode tree.
   - Return the path + a lint report (CPS/line-length warnings).

2. **`PostUpscaleProcgenBlend` -- minimal change**
   - When captions enabled: call `build_ass_from_ledger(...)`, append `ass=` to `filter_complex`.
   - When disabled: identical to today (zero behavioral change).

3. **Toggle (two options -- pick at QA):**
   - **(A) Widget (recommended, discoverable):** add `burn_captions` BOOL + `caption_style`
     COMBO to the blend node. **Cost:** node 12 widget vector grows by 2 -> must update
     `OTR_WorkflowValidator`'s expected vector and re-run the validator POST (it HARD-RAISES
     on drift). Wire defaults into the workflow JSON (Prime Directive 3).
   - **(B) Env toggle (zero-drift, fastest to smoke):** `OTR_BURN_CAPTIONS=1` +
     `OTR_CAPTION_STYLE=sdh_standard`, read via `winreg`/`os.environ` exactly like the
     existing `OTR_CAST_SEED`/`OTR_STYLE_SEED` pattern. No widget, no validator change.
   - **Plan: ship P1 on (B) to QA fast, then promote to (A) with the validator update.**

---

## 3. Font & color spec -- **QA THIS SECTION**

Two presets so you can compare. Default is the accessibility-standards build; the themed
variant matches the OTR green-CRT look. Values are for **1920x1080**.

### Preset 1 -- `sdh_standard` (recommended default; BBC/Netflix-aligned)

| Attribute | Value | Rationale |
|---|---|---|
| Font family | **Arial** (fallback Helvetica/Liberation Sans) | Ubiquitous on Windows, high legibility, no licensing issue. |
| Font size | **52 px** | ~1/21 of frame height; BBC floor is ~1/20. Readable, not huge. |
| Weight | Regular (Bold off) | Bold + box can smear on the busy overlay. |
| Primary text color | **white `#FFFFFF`** | Max contrast on dark box. |
| Background | **opaque black box, ~70% alpha** (`BorderStyle=3`, `BackColour=&H59000000`) | Standard SDH legibility on any background. |
| Outline/shadow | Outline 0 (box handles contrast), Shadow 0 | Box preset; no double treatment. |
| Position | **bottom-center, MarginV 70** | Lands in the pillarbox letterbox bar -- never covers the picture. |
| Max lines | **2** | SDH standard. |
| Per-speaker color | Announcer **white**; characters cycle **yellow `#FFFF00`**, **cyan `#00FFFF`**, **green `#00FF00`** | Classic SDH speaker differentiation (CEA-608 palette). |
| Speaker label | `NAME: ` prefix on first line of each speaker's turn | SDH identifies who speaks. |
| Sound cues | **italic, bracketed**, e.g. `[STATIC]`, in the speaker's color or neutral white | Distinguishes non-speech audio. |
| Music cue | `♪ Theme ♪` (music notes), italic | Standard music indicator. |

**ASS style line (sdh_standard):**
```
Style: SDH,Arial,52,&H00FFFFFF,&H000000FF,&H00000000,&H59000000,0,0,0,0,100,100,0,0,3,0,0,2,60,60,70,1
```
(ASS color = `&HAABBGGRR`; `&H59000000` = ~65% opaque black box. Per-speaker color via inline
`{\c&H00FFFF&}` overrides: yellow=`&H0000FFFF`, cyan=`&H00FFFF00`, green=`&H0000FF00`.)

### Preset 2 -- `otr_crt` (themed variant for QA comparison)

| Attribute | Value |
|---|---|
| Font family | **Consolas** (or bundled mono) -- matches the terminal/console card aesthetic |
| Font size | 50 px |
| Primary color | OTR green **`#33FF66`** |
| Background | semi-transparent black box ~50% (`&H7F000000`) |
| Outline | 2 px black (`BorderStyle=1`) for scanline-friendly edge glow |
| Position | bottom-center, MarginV 70 |
| Per-speaker color | green shades / amber `#FFBF00` for announcer |

**Trade-off to QA:** `otr_crt` looks on-brand but green-on-busy-green-overlay can lose contrast
during the oscilloscope segments; `sdh_standard` always reads. Recommendation: **ship
`sdh_standard` as default, offer `otr_crt` as the style option.**

---

## 4. SDH line rules (the captions builder enforces)

- **Reading speed:** target <= 17 CPS, hard cap 20 CPS. If a line's `dur_s` is too short for
  its text, log a lint warning (don't silently truncate -- Prime Directive 1 is about audio,
  but captions should still carry the full words; prefer splitting into 2 events).
- **Line length:** <= ~37 chars/line, <= 2 lines; wrap on word boundaries.
- **Min duration:** >= 1.0 s on screen even for short lines (extend end if the gap allows).
- **No overlap:** if two lines' `[start, end]` overlap, clamp the earlier end to the later start.
- **Speaker label** only when the speaker changes (don't repeat `NAME:` for consecutive same-speaker lines).
- **ASS escaping:** escape `{`, `}`, and backslashes in dialogue text.
- **Music/SFX-only gaps:** caption `[MUSIC]` / sound cue only when no speech is concurrent.

---

## 5. Wiring & verification (every step gated)

1. **P0 -- offline builder.** Write `scripts/otr_captions.py`; generate `<id>_captions.ass`
   from the Generator's Grasp ledger. **QA the .ass file directly** (text, timing, speaker
   labels, CPS lint) before any video work.
2. **P1 -- env-gated burn.** Add the `ass=` append behind `OTR_BURN_CAPTIONS` (zero widget
   drift). Run one smoke. Extract frames at known caption timestamps -> **QA legibility,
   position, color, box opacity** against this spec.
3. **Audio regression (mandatory):** assert the final `otr/obs` **audio stream is byte-identical**
   to a no-caption render (`ffprobe`/stream md5). Prime Directive 1.
4. **P2 -- promote to widget.** Add `burn_captions` + `caption_style` to the blend node;
   update `OTR_WorkflowValidator` expected vector; **validator POST -> drift=0, no raise**;
   wire defaults into `workflows/otr_scifi_16gb_full.json` (Prime Directive 3).
5. **Standing gates:** Bug Bible + core + `test_audio_byte_identical.py` after every code
   change. Log any new bug to `BUG_LOG.md` immediately.
6. **One change per commit;** validator after every JSON edit; smoke after any ffmpeg-arg change.

---

## 6. Risks / edge cases

- **Font availability:** Arial ships with Windows; if libass can't find it via fontconfig,
  pass an explicit `fontsdir` or `force_style`. Verify the resolved font in the smoke.
- **Boomerang/looped fill clips** (BUG-117d, BUG-135): caption text comes from the line, timing
  from `start_s`/`dur_s` -- unaffected by the video doubling/looping. Confirm in the smoke.
- **Duration contract (BUG-084):** video may be ~0.3 s longer than audio; the last caption's
  end is clamped to audio end so no caption hangs on trimmed-off video.
- **Pillarbox geometry (BUG-030):** MarginV places captions in the black bar; verify they don't
  collide with the bottom procgen waveform/HUD -- nudge MarginV if needed (QA item).
- **CPS overflow on fast/urgent lines:** split into two events rather than truncate.
- **Special chars / non-ASCII** in dialogue: ASS-escape; keep UTF-8 no BOM.

---

## 7. Open QA questions for Jeffrey

1. **Default style:** `sdh_standard` (white-on-black, standards) vs `otr_crt` (green, themed)?
   Recommend standard as default, themed as an option.
2. **Per-speaker colors:** keep the CEA-608 palette (white/yellow/cyan/green), or map colors
   from the cast (e.g., announcer always amber)?
3. **Sound cues:** how literal? `[STATIC]`/`[ALARM]` from sfx tags, plus `♪` for music --
   or dialogue-only captions to start?
4. **Toggle default:** captions ON by default for every render, or opt-in per episode?
```
