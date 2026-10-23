# Radio Play Studio UI: the design anchor, v2 (after the design round)

Operator, 2026-10-23: build the Studio page to the design before the regression work; "the UI
needs to be creative and rock solid"; "ask Fable for design as well".

**Design round, 2026-10-23, four seats in parallel, each briefed to refute v1:** Codex (file-grounded:
every DOM claim against `index.html`), Opus (judgment: the first-time user and the judge), Sonnet
(layout, CSS feasibility, themes, nine languages, accessibility; standing in the Cursor seat, which
this window does not log into), Fable (taste and identity). Verdicts: Opus AMEND, Sonnet AMEND,
Codex "not code-ready", Fable "right room, wrong furniture". They converged; v2 folds every finding
that grounds against the code and names what is deferred. No second round: the four agree on the
direction and every open point collapsed to a concrete rule below.

## The concept: Three lamps (Fable)

A dark room, paper cards, and exactly three things that can glow: Stills, Video, Publish, each a
round lamp. The lamp is the only thing on the page that moves; everything else holds still like a
printed call sheet. By day (light OS theme) the same room with the blinds open: the lamps are
filled discs, no glow.

**Cut, by agreement of two or more seats:** the film-strip sprockets (cover the picture, collide
with the play button and the video controls, and lie about the object), the VU meter (a monotonic,
stepped bar drawn as a level meter lies, and its ticks make a stall look broken), the tape-label
scene headers (an off-white strip fails contrast with the clay and red flags), the two-column
landing (the mode cards live inside `#form`, and putting the button left of the fields breaks tab
order), 3:4 portrait frames (crop a 16:9 fallback still), small caps and italic as identity (do
nothing in Devanagari, kana and han), and any printed "ON AIR" (the page invites editing while a
stage runs; the lamp is decorative and `aria-hidden`).

### Palette

| Token | Dark (night) | Light (day) | Role |
|---|---|---|---|
| `--bg` | `#17130f` | `#f4efe6` | the room |
| `--panel` | `#211b15` | `#fffdf8` | desks: cards, rail, console |
| `--strip` | `#2a221b` | `#ece4d6` | scene header strip, one step up |
| `--ink` | `#f0e6d6` | `#2a2118` | text |
| `--muted` | `#a89a86` | `#6f6254` | help, counts |
| `--line` | `#3a3026` | `#d9ccb8` | rules, frames |
| `--field-line` | `#867560` | `#9a8a72` | input borders (3:1 non-text) |
| `--accent` | `#e0703f` | `#b4491f` | clay: the enabled press, the changed state |
| `--lamp` | `#ffd98a` | `#8a5200` | a lit lamp (reads as light, apart from clay) |
| `--lamp-glow` | `rgba(255,217,138,.35)` | none | the halo, night only |
| `--ok` / `--bad` | `#7ec28a` / `#f08a7e` | `#2f6b3a` / `#a3261a` | filmed / failed |
| `--kept` | `#8f826f` | `#70644f` | kept (light fixed from 3.5:1 to 5:1) |
| `--focus` | `#ffd98a` | `#8a5200` | the focus ring |

Theme mechanism unchanged: light on `:root`, dark under `prefers-color-scheme: dark` guarded by
`:not([data-theme="light"])`. Add `color-scheme: light dark` (and `dark` in the dark block) so
native selects, scrollbars and video controls follow.

### Type (all local)

- Body: `Georgia, "Iowan Old Style", "Palatino Linotype", "Noto Serif", serif`, 16 px / 1.5. Georgia
  is the page's voice today and on both Windows and Mac.
- Chrome (labels, badges, flags, buttons, `.help`): `system-ui, "Segoe UI", Roboto, "Noto Sans",
  "Helvetica Neue", Arial, "Noto Sans Devanagari", "Hiragino Sans", "Yu Gothic", "Microsoft YaHei",
  sans-serif`.
- Counts and timings: `ui-monospace, "Cascadia Mono", Consolas, Menlo, monospace` with
  `font-variant-numeric: tabular-nums`.
- The spoken line `.line`: upright serif with a left rule (it is in the play's language, which may
  differ from the page's, so no italic and no `:lang` trick). Letter-spacing only under
  `:lang(en), :lang(fr), :lang(es), :lang(it), :lang(pt)`.

### Signature moments (all on hooks the script already sets)

1. **A lamp comes on**: `.stage.on::before` is a lit disc with a halo and a slow pulse (inside
   `prefers-reduced-motion: no-preference`; reduced motion keeps a steady lit disc). Shape carries
   state as well as colour: running = ring plus lit core; done = filled clay disc with a check;
   waiting = empty ring. `aria-current="step"` on the running stage (one script line).
2. **Held**: when the last run is held (`#b_look` shown), the running lamp is steady, not pulsing:
   `#board:has(#b_look:not([hidden])) .stage.on::before { animation: none }`.
3. **A shot fails**: the card's still goes grey (as today), the left edge goes red, the `.why` note
   reads as a note in body serif with a red left rule; the badge word stays the state.
4. **Published**: the episode's video sits at the top of the scenes column in the page's one
   paper mat; every lamp is done.
5. **The single run** (`Make the episode`, the path a judge presses first): the same lamp beside its
   status line while it runs, pure CSS
   (`#run:has(#result[hidden]):has(#error[hidden])`).

## Layout

- `main` stays 760 px for the landing and the single run; it widens to `min(1360px, 100%)` only
  while the board is shown, by CSS: `main:has(#board:not([hidden]))`. No body class, no script.
- `#run`, `#summary` and `#result` keep a 760 px measure inside the wider main.
- **Landing: one column** (no change to order or script): masthead (title, tagline, language), the
  two mode cards as the hero, the form, the buttons, the project list, then `#run` directly under
  the buttons. The form gets the panel look.
- **Board skeleton** (static markup; ids, types and the script's owners unchanged):

  ```text
  section#board
    div.board-grid
      div.rail            (not <aside>: no unnamed landmark inside a live region)
        div.stages        (three .stage, each: <b> then the status <span> as its FIRST span)
        div.bar > #pfill
        p#pstatus
        fieldset#castbox > #castrow
      div.stage-floor     (min-width: 0)
        div#published     (moved here: at the top of the scenes, before the console)
        div#scenes
    div.actions           (the console: LAST direct child of #board, never a grid item)
  ```

  `grid-template-columns: 300px minmax(0, 1fr)` at >= 1000 px; one column below, the rail first.
  The rail is `position: sticky; top: 16px` at the two-column width so the lamps stay in view.
  No ancestor of `.actions` gets `overflow`. The cast in the rail is a grid
  `repeat(auto-fill, minmax(104px, 1fr))` of square frames with an inner mat.
- **Shot card**: `.tools` becomes the third direct child of `.shot` (one script change: `body.append(why);
  el.append(still, body, tools)`; nothing reads `tools` after creation). Layout by container query
  on `.scenebody` (`container-type: inline-size`): >= 700 px `200px minmax(0,1fr) 150px`;
  480-700 px `176px minmax(0,1fr)` with the tools under the text; under 480 px one column. Card
  buttons full width in their column, labels wrap.
- **The console** (`.actions`): panel-coloured, full board width, sticky to the viewport bottom.
  The ENABLED stage button is the primary (`.actions button:enabled` minus Look again / Forget gets
  the clay fill), not a fixed one. The reason line stays beside the buttons at every width. Under
  560 px: disabled stage buttons collapse and the hint clamps to two lines; under 520 px of height
  the console stops sticking. `html { scroll-padding-bottom }` keeps a focused field clear of it.

## The named script changes ("logic frozen, state exposed")

Each is a single statement, exposes state the script already holds, and is listed in the commit:

1. `shotCard()`: `.tools` becomes the third child of `.shot` (layout).
2. `showStage()`: `aria-current="step"` on the running stage (a11y).
3. Live regions: `aria-live` leaves `#board`; `#pstatus` and `#hint` get `role="status"`, `#perror`
   `role="alert"` (static markup) -- `refreshBoard()` rewrites every badge on each keystroke, and a
   live board re-announced all of it.
4. The draft save error writes `#saved` with an `error` class, and success clears it (a failed save
   looked the same as "Saved").
5. `drawBoard()`: a scene the person had open stays open after a redraw (keyed by `scene_id`).

## Contracts the build keeps (Codex inventory)

- Every id keeps its element type: `#form` a `<form>`; `#lang`, `#f_bank` selects (and `#f_bank`'s
  first option stays first); `#mode_*` radios; `#go #direct #b_* #change_story` buttons; `#fill`
  `#pfill` plain elements taking inline widths.
- Script-owned containers are never given static children: `#form`, `#cast_list`, `#result`,
  `#published`, `#projects`, `#summary_text`, `#castrow`, `#scenes`, `.still` (Play clears it),
  `#hint`, and every translated leaf. Decoration is backgrounds and pseudo-elements only.
- `refreshBoard()` reassigns `.shot` and `.badge` `className`: no extra class on them; hooks go on
  the body div, `.tools` or `.still`.
- Every new rule is scoped (`.person .face`, `.shot .who`, `#board .small`): `.face`, `.who`,
  `.badge`, `.small`, `.quiet` are shared with the landing's cast editor.
- Touch targets: `.small`, `.actions button`, `#lang`, `.still .play` at least 44 px (the play
  button grows itself). `:focus-visible` ring on every control. `forced-colors` gives the lamps a
  border.

## Tests added with the build

- A static DOM contract test: every id the script reads exists once in the static markup or is
  created by the script, with its tag type; `.actions` is the last element child of `#board`;
  `#published` precedes it.
- The dynamic mode keys `mode.{classic,own}.{title,text}` exist in every language (the existing
  test only finds literal keys).

## The proof

Live, in the browser pane, against the real project `c80906f6f093` and the real server: the
landing and the board at 1440 px and 375 px, dark and light (emulated `prefers-color-scheme`), in
all nine languages with a script check that `document.documentElement.scrollWidth <=
clientWidth` and every visible button is at least 44 px tall; screenshots of the landing, the
board, a dirty card, and the published state; a Paint again click, the draft save, and the button
state change exercised by clicking. Then a refute pass on the pushed diff.

## Deferred (real, not in this build)

- "New take" badges on cards whose still or clip changed since the last view (Opus 3).
- The film-time estimate under Film the shots, and `minutes_left` on the board (Opus 4).
- A designed empty first board while the stills are being made, a "no signal" card, a form
  placeholder while ComfyUI answers (Opus 6, 7).
- Fable's microcopy: `hint.held`, `confirm.forget`, `status.finished` (nine-language rewrites).
- A held lamp for another project's run, `data-state` on `#board` (Opus 7).
