# Radio Play Studio for Stream Deck

Make a radio play without touching the page. Dials (or keys) pick the language, the story
bank, the visual style, the number of acts and the machine; one key makes the episode and
fills in as it goes. It drives [Radio Play Studio](../README.md) on this computer, over the
same local HTTP the page uses; the Studio did not change for it.

It looks like the two things it joins. The colours are the Studio's own -- deep purple, the
faint grid, the green lamp, the OTR icon's heavy lettering with its hot-pink shadow -- and
every choice is drawn as a **ComfyUI node**: a title bar, typed sockets, a wire. On a Stream
Deck + the four touch-strip cells sit edge to edge, so their wires join and the strip reads as
one graph, Language → Story bank → Visual style → Acts. While an episode is being made every
node is outlined green, the way ComfyUI marks the node it is running, and the Make key's ring
fills with the lamp's green.

## What is on the deck

| Control | On a Stream Deck + dial | On a key (any Stream Deck) | Face |
|---|---|---|---|
| **Choice** (one per field; pick the field in its settings) | turn: next or previous choice · press: back to the default | press: next choice · hold: back to the default | a node: the field's name, where the pick sits in its list (`3/11`), the pick, a green bar while an episode is made |
| **Machine** | turn: rest on a machine and how it films · press: switch to it | press: rest on the next one · hold: switch to it | the card, how it films (`unproven` where it has not been proven); the resting row is outlined white |
| **Make** | — | press: make the episode; when done, play it; after a failure, clear it · hold: open the Studio page on the run | `MAKE` over the waveform; then a ring that fills, the percent, the phase and minutes left; a check when done; the reason when it failed |

So a deck with no dials (Stream Deck MK.2, XL, Mini, Neo) can run the whole thing from keys:
put one Choice key per field, a Machine key and the Make key on a page.

The story bank skips *my story*: that bank needs typed text, which stays on the page. A roll
row shows as *Surprise me*. While the page is making an episode the Make key follows that one;
during a Scene Producer stage the deck shows busy and the choices wait.

Every word on every face is fitted to its face: widths are measured from the real Segoe UI
fonts (`src/gfx/font-metrics.ts`), text shrinks, then wraps, and is cut only when a server
error is longer than any key. `tests/faces-fit.test.ts` measures every face in every state
against its box -- a node's body, or the Make key's clear disc at that line's height.

## Install

Stream Deck 7.1 or newer, Node 24, and Radio Play Studio running on this computer.

```bash
cd contest/radio-play-studio/streamdeck
npm install
npm run build
npm run link
```

`npm run link` registers `com.jbrick2070.otr.sdPlugin` with the Stream Deck app (the Elgato
CLI's `streamdeck link`). Drag the actions onto a page. If the Studio runs on another port,
set it in any action's settings; the host is always `127.0.0.1`, because the Studio answers
no other. The port and the picks are saved and come back after a restart.

## Develop

```bash
npm test          # vitest: the faces fit, dial math, the event stream, the store against a fake Studio
npm run typecheck
npm run validate  # the Elgato CLI's manifest check
npm run images    # redraw the default key images from the live renderer
```

`API-CONTRACT.md` lists every route the deck uses and what it expects back.
`tests/helpers/fake-studio-server.ts` is a stand-in Studio that enforces the server's two
rules (loopback Host, JSON posts); `../tests/test_rps_streamdeck_contract.py` runs the same
calls against the real server.

MIT, like the rest of this repository. The shape (SDK 3, one store, rate-limited SVG faces)
follows [streamdeck-fadester](https://github.com/jbrick2070/streamdeck-fadester). ComfyUI is
the open-source project this pack runs on; the node look is a nod to its graph editor, and no
ComfyUI or Comfy Org logo is used.
