# Radio Play Studio UI: the design anchor (driver's code-grounded proposal)

Operator, 2026-10-23: build the Studio page to the design canvas
(https://claude.ai/artifact/QZnPyw4vgo1jVG6PYiFUuG) before the regression work; "the UI needs to
be creative and rock solid". This is the proposal the design round tries to break before any code.

## What exists (read from the files, not remembered)

- One file: `contest/radio-play-studio/radio_play_studio/static/index.html` (1,578 lines): a
  `<style>` block (lines 7-163), static markup (165-223), one inline `<script>` (225-1576).
- Layout today: a single 760 px column (`main { max-width: 760px }`), Georgia serif, light theme by
  default with a dark theme under `prefers-color-scheme: dark` (tokens `--bg --panel --ink --muted
  --line --accent --accent-ink --ok --bad --bar --kept`).
- Views, toggled by `hidden` from the script: the landing form (`#form`, built by `build()` and
  `modeSwitch()` from `/api/options`; `#gobar` with `#go` Make the episode and `#direct` Direct the
  shots first; `#projects`), the single-run view (`#run`: `#bar #fill #status #error #result`), and
  the board (`folded(true)` hides `#form #gobar`, shows `#summary #board`).
- The board's static markup: `.stages` (three `.stage` with `<b>` + `<span>`), `.bar #pfill`,
  `#pstatus`, `fieldset#castbox` (`#castrow`), `#scenes`, `.actions` (sticky; `#b_stills #b_video
  #b_publish #b_look #b_forget #hint #perror #saved`), `#published`.
- Script-built: `personCard()` (`.person` > `.face` + `b` + `.badge.changed` + Paint again),
  `sceneBox()` (`details.scene` > `summary` with `b`, `.count`, `.flag.changed`, `.flag.failed`;
  `.scenebody` > optional `.scenenote` + `.shots`), `shotCard()` (`.shot` > `.still` (img, `.kind`,
  `.play`) + body: `.who` x2, `.when`, `.badge`, `.line`, labelled Picture / Motion textareas or the
  lane-fixed help line, `.why`, `.tools` with Paint again / Film again / Keep). `refreshBoard()` sets
  `.shot` state classes (`changed`, `failed`, `kept`, ...), badges, `.why`, the stage lines via
  `showStage()` (`.on` / `.done`), the button labels and `disabled`, and `#hint`.
- Strings: every visible word comes from `static/i18n/<code>.json` (nine languages) through
  `data-i18n` / `tx()`; `tests/test_rps_i18n.py` requires every key the page names to exist in
  English and every language to carry exactly English's keys and placeholders.

## The proposal

### Invariants (what "rock solid" means here)

1. **Behaviour does not change.** No change to the script's logic, its event wiring, its fetches or
   its draft rules. Every element id the script reads stays, with the same element type. Class
   names the script sets (`shot`, `changed`, `failed`, `kept`, `on`, `done`, `badge ...`, `flag
   ...`, `quiet`, `small`, `dirty`) keep their meaning; CSS may restyle them, never repurpose them.
2. **Markup moves only where a wrapper is needed for layout**, and only in the static block: the
   board's stages/bar/pstatus/castbox go into an `<aside class="rail">`, `#scenes` into a
   `<div class="stage-floor">`, the two inside a `.board-grid`. The script never queries by DOM
   position (verified: it uses `$()` by id and keeps element references), so wrapping is safe.
   Script-built cards change no markup; their look is CSS only, plus at most two class names added
   at creation time if a layout cannot be reached otherwise (each listed in the commit).
3. **Offline-first.** No web font, no CDN, no image request the page does not already make. The
   look comes from a local font stack and CSS.
4. **Both themes, both widths.** Light and dark keep working (the dark theme becomes the house
   look, light stays complete), and the page works at 360 px wide with no horizontal scroll, touch
   targets at least 44 px, focus rings visible.
5. **Nine languages.** Longer strings (German-length French, Hindi, Japanese) must not break a
   layout: no fixed-width text boxes, buttons wrap their label, the three stage cells shrink.
6. **Accessibility unchanged or better**: real buttons and labels as today, contrast 4.5:1 for
   text in both themes, state never shown by colour alone (the badge text stays).

### The look: "the studio at night" (creative brief)

An old-time radio studio after hours, drawn with restraint, never a costume:

- **Palette** (the page's own dark tokens, kept): ink `#17130f`, panel `#211b15`, paper
  `#f0e6d6`, muted `#a89a86`, line `#3a3026`, clay accent `#e0703f`; plus one new token, an
  **on-air amber** `#ffb347` used ONLY for the live state (a stage running).
- **Type**: titles in a local serif stack (`"Iowan Old Style", "Palatino Linotype", Palatino,
  Georgia, serif`), body in the system sans (`system-ui, "Segoe UI", ...`) for legibility at small
  sizes, mono (`ui-monospace, Consolas, ...`) for counts and timings. Today everything is Georgia.
- **The ON AIR stage rail**: the three stages as numbered lamps in a vertical rail (desktop) or a
  row (phone). A stage that is running gets an amber lamp with a soft glow and a slow pulse
  (`prefers-reduced-motion` turns the pulse off); done is a filled clay disc with a check; waiting
  is an outlined disc. The progress bar becomes a **VU meter**: the existing `#pfill` width,
  drawn over a row of tick marks.
- **The cast as portrait frames**: 3:4 frames with a thin inner mat, the name on a small label
  plate under each.
- **Shot cards as cue cards**: the still on the left with a film-strip edge (two rows of sprocket
  holes drawn with a CSS gradient on `.still`), the speaker and line set like a script page (the
  name in small caps, the line in the serif italic), the Picture and Motion boxes, and the three
  buttons in a right-hand column on wide screens. The state stripe stays on the left edge.
- **Scene headers as tape labels**: a slightly off-white strip with the scene title in small
  caps, the shot count, and the changed / failed flags.
- **The sticky bar as the console**: a panel-coloured strip across the full width with the
  primary button solid clay and the two others outlined; the reason line beside them.
- **The landing page** keeps its order (title, tagline, mode cards, form, buttons, projects) but
  becomes a two-column hero on wide screens: the title, tagline, the two mode cards and the
  buttons on the left; the form ("The show") on the right as a panel. One column under 900 px.

### Layout

- `main` widens from 760 px to `min(1360px, 100%)` when the board is open (a `board-open` class
  on `<body>`, set and cleared in `folded()` — the one script line this adds) and to 1200 px for the
  landing hero; the single-run view stays a readable 760 px column inside it.
- Board: `.board-grid` = rail (300 px) + scenes (flex) at >= 1000 px; one column below, the rail
  becoming a compact horizontal strip at the top.
- Shot card: grid `200px 1fr 150px` at >= 1100 px (still | text | buttons), `176px 1fr` with the
  buttons under the text between 560 and 1100, one column under 560 (as today).

## What could break, and how it is checked

| Risk | Check |
|---|---|
| a wrapper changes an id's parent the script relies on | grepped: the script never walks the DOM by position except `showStage()` (`el.querySelector("span")`, line 1360), which reads a stage cell's FIRST `<span>`: a stage cell keeps its state line as its first span, and any lamp is drawn with CSS (`::before`) or a non-span element |
| the sticky bar covers the last card | bottom padding on `.stage-floor` equal to the bar height; checked at 360 px with nine languages |
| long translated labels overflow | live check with `?lang=fr`, `hi`, `ja` at 360 px and 1440 px |
| the light theme left half-styled | live check with `data-theme="light"` and the OS in light |
| the landing form's script-built rows assume the 760 px column | live check that every `build()` row renders in the right-hand panel |
| reduced motion | the pulse is inside `@media (prefers-reduced-motion: no-preference)` |
| tests | `contest/radio-play-studio/tests` (242) and the pack's guard set; `test_rps_i18n` for any new key |

## The proof

The live page in the browser pane, against the real project `c80906f6f093` and the real server,
at 1440 px and 375 px, dark and light, in English, French, Hindi and Japanese: screenshots of the
landing hero, the board, a card being edited (dirty), a stage running (amber lamp), and the
published state. No behaviour check is skipped because "only CSS changed": a Paint again on one
card, the draft save, and the button state change are exercised by clicking.
