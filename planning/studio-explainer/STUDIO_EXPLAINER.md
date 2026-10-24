# DIRECT THE SHOTS
### Radio Play Studio: the page that lets you change any picture before a frame is filmed

**Estimated runtime:** ~5:45 (≈145 wpm, dry narration)
**Visual style:** 3D wireframe / isometric vector motion graphics
**Audio:** dry narration, no music (one non-musical cue — see AUDIO SYSTEM)
**Pipeline:** NotebookLM-compatible scene blocks
**Palette:** locked — indigo (#12092f), panel (#1c1240), lavender (#f4edfb), purple (#7c3aed), waveform green (#50f898) — the page's own colours, read off the pack's registry icon
**Data convention:** SOLID purple = documented fact (what the page shows and the code does) · DASHED lavender = inference (what a user is likely to do next) · green = a lamp that is lit
**Scene count:** 13 (+ end card)
**Narration word count:** ~840

**v2 changes from v1 (2026-10-24):**
- Palette moved to the page's new colours (the registry icon's indigo, purple and waveform green).
- Scene 02 shows the page as it opens now: the page-language menu top right, apart from the language of the play.
- New Scene 03 (your machine, then how it's filmed) and Scene 04 (the writer and the keys).
- Old Scenes 03-11 are now 05-13; their facts are unchanged.

Facts in this script come from the Radio Play Studio page as served (its English strings), the
"Direct the shots" section of its README, and the live proofs recorded in
`planning/scene-producer/PLAN.md` on 2026-10-23. Every number on screen is one of those receipts, except the filming-card times in Scene 03, which come from the regression round of 2026-10-23 (`planning/regression/ROUNDS.md`). This explainer is about the Studio, the full-screen web page the pack ships; the App
Mode form inside ComfyUI has its own explainer.

---

## AUDIO SYSTEM

Dry narration over silence. One cue: a slow film-projector shutter flutter, heard only on the
scene transitions and once more when the Run button is pressed; it never settles into a
rhythm. It cuts to silence at the end card.

---

## SCENE 01 — COLD OPEN: ONE PICTURE

**VISUAL:** Isometric wireframe of a single shot card, floating over the indigo grid: a still
(a wireframe radio console), a line of dialogue under it, two text boxes labelled PICTURE and
MOTION, three buttons. The PICTURE box lights solid purple; the still above it redraws itself,
once. Nothing else on the card moves.

**NARRATION:** A radio play is twenty-two shots. You changed one picture. The Studio repaints that
one still, films that one clip, and reuses the other twenty-one. That is the whole idea.

**ON-SCREEN TEXT:** ONE PICTURE · ONE STILL · ONE CLIP · 21 REUSED

**TRANSITION:** The card shrinks into a wall of twenty-two cards; the camera pulls back to show the
page they sit on.

---

## SCENE 02 — THE STUDIO

**VISUAL:** The full Radio Play Studio page in wireframe: the title in heavy capitals with a green
rule under it, a page-language menu at the top right with nine entries, two rows of cards (machine,
then how it's filmed), two mode cards — A CLASSIC, OR SURPRISE ME and TELL MY OWN STORY — and one big button,
MAKE THE EPISODE. Beside it, quieter, DIRECT THE SHOTS FIRST. A dashed lavender arrow points at the
quiet button.

**NARRATION:** Radio Play Studio is the pack's own web page. It runs on your machine, talks to your
ComfyUI, and speaks nine languages; the menu at the top right sets the page's own, apart from the
language of the play. Press Make the episode and the whole play is written, cast,
voiced, painted and filmed in one run. On the Tell my own story path there is a second, quieter
button: Direct the shots first.

**ON-SCREEN TEXT:** RADIO PLAY STUDIO · 127.0.0.1:8765 · MAKE THE EPISODE · DIRECT THE SHOTS FIRST

**TRANSITION:** The quiet button depresses; the form folds to one line and a board slides up.

---

## SCENE 03 — YOUR MACHINE, THEN HOW IT'S FILMED

**VISUAL:** Two rows of wireframe cards above the form. Top row: NVIDIA 8 GB · NVIDIA 16 GB OR MORE ·
MAC 16 GB · AMD · COMFY CLOUD · GOOGLE CLOUD; NVIDIA 16 GB OR MORE lights solid purple with a small
green tag, THIS MACHINE. Second row: three cards, TTS + FOLEY VIDEO, TTS W/ VIDEO, TTS W/ STILLS,
each with one line of model names (VIDEO · STILLS · VOICES · MUSIC · WRITER) and a time line. The top
card switches to MAC 16 GB: the foley card dissolves, two remain, and the still-model dropdown below
loses entries. The two cloud cards carry a dashed lavender tag, NOT YET PROVEN.

**NARRATION:** The page starts with two questions. Your machine: it starts on the card your ComfyUI
reports. Then how it is filmed: TTS plus Foley video, TTS with video, or TTS with stills, and on
a graphics card, only the ways proven on it. Each card names its video model, still model, voices, music and
writer, read from the workflow itself, and how long your acts took on our test machine. Pick a Mac
and the foley card goes, and so do the still models a Mac cannot run. The cloud ways are offered,
and marked not yet proven.

**ON-SCREEN TEXT:** YOUR MACHINE · HOW IT'S FILMED · PROVEN ON THIS MACHINE · CLOUD: NOT YET PROVEN

**TRANSITION:** The chosen way card drops into the form, which fills itself in.

---

## SCENE 04 — THE WRITER AND THE KEYS

**VISUAL:** A closed panel labelled "Writer: gemma4-12b-int8-convrot (runs here, free)". It opens:
the local models, then OpenRouter, Google Gemini and Comfy, each tagged (cloud, paid). Picking
OpenRouter reveals one model row. Below the form, just above the press bar, a key box draws itself:
COMFY KEY with a masked field (for a cloud still); OPENROUTER KEY with a file path. The press bar shows "Needs a key: Comfy
key" in a warm tone.

**NARRATION:** The writer sits in a closed panel, and its title always says who will write and
whether it costs anything. Left alone, the workflow's own writer runs. Open it to pick OpenRouter,
Google or Comfy, each a paid cloud model with its own model choice. A pick that needs a key brings
up one box above the button. A Comfy key can be typed in, and the app holds it in memory only. A
Google or OpenRouter key goes in a file the page names.

**ON-SCREEN TEXT:** WRITER · RUNS HERE, FREE · CLOUD, PAID · ONLY THE KEYS YOU NEED

**TRANSITION:** The key box slides away; the quiet button, DIRECT THE SHOTS FIRST, lights.

---

## SCENE 05 — THE BOARD

**VISUAL:** The board, in three bands. Top: three stage markers in a row — STILLS, VIDEO, PUBLISH —
with a thin progress bar. Left: a CAST strip of three portrait frames. Centre: SCENE 1 and SCENE 2
as collapsible rows. Bottom: one sticky bar with three buttons. Each band lights solid purple as it is
named.

**NARRATION:** The play is written, cast and voiced as usual, then it stops after the stills. The
page becomes a board at once and fills in when the stills are done. Three stages across the top:
Stills, Video, Publish. The cast on the left. The scenes in the middle, every shot a card. One bar
at the foot with the three things you can press.

**ON-SCREEN TEXT:** STILLS → VIDEO → PUBLISH · CAST · SCENES · ONE BAR

**TRANSITION:** The camera pushes into the stage markers.

---

## SCENE 06 — THREE STAGES, SAID PLAINLY

**VISUAL:** The three markers large. Under each, its state line in lavender monospace, exactly as the
page prints it: STILLS "22 painted, 1 to paint"; VIDEO "21 of 22 filmed"; PUBLISH "after every shot
is filmed". The first marker fills solid purple, the second is outlined, the third is dim.

**NARRATION:** Each stage says where it stands in plain words. Stills: how many are painted and how
many are waiting. Video: how many of the shots are filmed. Publish: after every shot is filmed.
You never have to guess what a press will do.

**ON-SCREEN TEXT:** "22 painted, 1 to paint" · "21 of 22 filmed" · "after every shot is filmed"

**TRANSITION:** Pan left to the cast strip.

---

## SCENE 07 — THE CAST

**VISUAL:** Three portrait frames: ANNOUNCER, JEFFREY, MAGNOLIAREX, each with a PAINT AGAIN button.
A dashed lavender line runs from Jeffrey's portrait to every card in the scenes that has his name on
it; the ones marked KEPT stay dim.

**NARRATION:** One picture per character, with Paint again. A new portrait repaints every shot that
character is in, except the kept ones. The page says so right under the strip: so try the shots
first.

**ON-SCREEN TEXT:** CAST · PAINT AGAIN · "except the kept ones, so try the shots first"

**TRANSITION:** Pan to Scene 2 and open it.

---

## SCENE 08 — A CARD

**VISUAL:** One card, full frame. Left: the still and a badge. Centre: JEFFREY · "Goodnight, old
friend. You kept us honest for twenty years." Then the PICTURE box and the MOTION box. Right: PAINT
AGAIN, FILM AGAIN, KEEP. Each element lights solid purple as it is named. A second, smaller card
beside it shows a MOTION box replaced by the line "This shot's motion is set by its lane."

**NARRATION:** A card is one shot. The still. Who speaks, and the line. The Picture box: the words
the still is painted from. The Motion box: the words the clip is filmed from, shown only where the
shot's lane lets you set it; elsewhere the card says the motion is set by its lane. Paint again is
another draw of the same picture. Film again keeps the still and films a new take. Keep freezes the
card as it is.

**ON-SCREEN TEXT:** PICTURE · MOTION · PAINT AGAIN · FILM AGAIN · KEEP

**TRANSITION:** The badge on the card flips through its states.

---

## SCENE 09 — BADGES: THE CARD'S STATE

**VISUAL:** Seven badges in a column, each in the page's own words, each lighting in turn: CHANGED,
NOT PAINTED YET · ACTION CHANGED, NOT FILMED YET · PAINTED, NOT FILMED · TO FILM AGAIN · FILMED ·
KEPT · FAILED. FAILED is the only one drawn in a warmer tone.

**NARRATION:** A badge names where the card stands. Changed and not painted yet. The action changed
and not filmed yet. Painted and not filmed. To film again. Filmed. Kept. Failed. A failed shot
blocks Publish until you paint or film it again, or keep it as it is.

**ON-SCREEN TEXT:** 7 STATES · FAILED BLOCKS PUBLISH

**TRANSITION:** Drop to the bar at the foot.

---

## SCENE 10 — ONE BAR, THREE PRESSES

**VISUAL:** The sticky bar: PAINT THE STILLS (1 CHANGED) solid purple; FILM THE SHOTS and PUBLISH THE
EPISODE outlined and dim. One line of lavender text beside them: "Only the changed shots are painted;
everything else is kept. Paint them before filming."

**NARRATION:** Paint the stills, with the count of what changed. Film the shots, with the count of
what is left to film. Publish the episode. One line beside them says why a button is off. Only what
changed is made again: a Stills run paints the changed stills and keeps the rest; a Video run films
the shots that are not filmed yet, or that you changed, and takes every other clip from what the
project already filmed; Publish mixes the voices and music under the filmed clips and loads no
generation weights.

**ON-SCREEN TEXT:** PAINT THE STILLS (1 CHANGED) · FILM THE SHOTS · PUBLISH THE EPISODE

**TRANSITION:** The bar's purple button depresses; the projector shutter flutters once.

---

## SCENE 11 — MEASURED, NOT PROMISED

**VISUAL:** A wireframe ledger with four rows, each a date-stamped receipt drawn in solid purple:
"one picture edit → 1 still repainted"; "then Video → 1 clip filmed, 21 reused";
"one motion edit → 1 clip filmed, 21 reused"; "no edit → 22 reused". A fifth row in dashed lavender:
"your edit → ?".

**NARRATION:** These are receipts, not claims. On the author's sixteen-gigabyte laptop, on a
twenty-two-shot play, on the twenty-third of October: one picture edit repainted one still, and
the Video run after it filmed one clip and reused twenty-one. One motion edit filmed one clip and
reused twenty-one. A run with no edits reused all twenty-two.

**ON-SCREEN TEXT:** 2026-10-23 · 22 SHOTS · 1 / 21 · 1 / 21 · 0 / 22

**TRANSITION:** The ledger folds shut; a wireframe power switch appears and flips off.

---

## SCENE 12 — IT SURVIVES A RESTART

**VISUAL:** The Studio window and a ComfyUI window side by side. The ComfyUI window goes dark
mid-render and comes back. The Studio window shows its stage bar pause, then a line: "ComfyUI no
longer has this run." The Video marker lights again; one card re-films; the rest stay dim with
small REUSED tags.

**NARRATION:** A directed project survives a restart. Its draft and its one pending run live in a
project file. Start the app again and it looks at each project it left pending: re-attaches to a
run that is still going, finds a submit whose answer was lost, saves a result that finished while
it was away, and, when ComfyUI no longer has the run, gives it up so you can press again. A Video
run after a crash reuses every shot that finished before it. Only the rest are filmed.

**ON-SCREEN TEXT:** PROJECT FILE · RE-ATTACH · LOOK AGAIN · ONLY THE REST ARE FILMED

**TRANSITION:** Both windows slide away; a folder tree draws itself.

---

## SCENE 13 — WHERE IT ALL LANDS

**VISUAL:** An isometric folder tree in lavender wireframe: output → otr → episodes → _replay → p →
‹project› → 01-stills, 02-stills, 03-video … and a clips/ store beside them, each bundle a small
box. A second branch: the app's own episodes/projects/‹project›/. A third, solid purple: otr → obs →
one file with a play triangle.

**NARRATION:** Each Stills or Video run writes a numbered bundle under your ComfyUI output folder,
with the clip store beside it. The page keeps its own copies under its episodes folder. And the
published episode lands in otr, obs, like any other — the place you already look.

**ON-SCREEN TEXT:** otr/episodes/_replay/p/‹project›/NN-stage · clips/ · otr/obs/

**TRANSITION:** Everything but the obs file cuts to black.

---

## END CARD / SOURCES

**VISUAL:** Black. Citation list scrolls in purple monospace.
**NARRATION:** (silent — 3 seconds of black before end card)
**ON-SCREEN TEXT:**
SOURCES
• contest/radio-play-studio/README.md, "Your machine, and how it's filmed", "Keys" and "Direct the shots" — github.com/jbrick2070/ComfyUI-OldTimeRadio
• planning/regression/ROUNDS.md — the times on the filming cards
• radio_play_studio/static/i18n/en.json — the page's own words
• planning/scene-producer/PLAN.md — the live proofs, 2026-10-23
• Registry: comfyui-old-time-radio (publisher fluxus)

---

**Runtime estimate:** ~5:45 at ≈145 wpm
**Scene count:** 13 (+ end card)
**Narration word count:** ~840

**Production notes:**
- Scene 01 is the hook and the whole promise; hold the single redraw a beat longer than feels natural.
- Scene 11 is the one a sceptic needs; keep the receipts in the page's own numbers and do not round.
- Scenes 06 and 09 quote the page verbatim; if the strings change, these scenes change.
- If runtime is tight, fold Scene 13 into Scene 12's tail.
- This is the Studio's explainer. The App Mode form has its own (planning/app-view-explainer/).
