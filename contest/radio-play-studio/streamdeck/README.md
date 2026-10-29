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

Stream Deck 7.1 or newer on Windows 10 or 11, and Radio Play Studio running on this computer.

1. Get the packed plugin,
   [`dist/com.jbrick2070.otr.streamDeckPlugin`](dist/com.jbrick2070.otr.streamDeckPlugin): from a
   clone it is already on disk, or download it straight from GitHub at
   <https://github.com/jbrick2070/ComfyUI-OldTimeRadio/raw/main/contest/radio-play-studio/streamdeck/dist/com.jbrick2070.otr.streamDeckPlugin>.
2. Double-click it. The Stream Deck app opens and installs **Radio Play Studio**.
3. Its three actions are under **Radio Play Studio** in the action list. Drag them onto a page:
   on a Stream Deck +, a Choice on each of the four dials (set each one's field in its
   settings: Language, Story bank, Visual style, Acts), with the Make key and a Machine key on
   the keys above; on a deck with no dials, one Choice key per field, a Machine key and the Make
   key.
4. Start the Studio ([its README](../README.md)). Until it answers, the faces read *Off air*.

If the Studio runs on another port, set it in any action's settings; the host is always
`127.0.0.1`, because the Studio answers no other. The port and the picks are saved and come back
after a restart. A newer package installs over an older one by double-clicking it the same way.

### Or run it from the source

Node 24, then:

```bash
cd contest/radio-play-studio/streamdeck
npm install
npm run build
npm run link
```

`npm run link` registers this folder, `com.jbrick2070.otr.sdPlugin`, with the Stream Deck app (the
Elgato CLI's `streamdeck link`), so a rebuild shows after the plugin restarts. Use the package or
the link, not both: they install the same plugin id.

## Develop

```bash
npm test          # vitest: the faces fit, dial math, the event stream, the store against a fake Studio
npm run typecheck
npm run validate  # the Elgato CLI's manifest check
npm run images    # redraw the default key images from the live renderer
npm run pack      # build, then write dist/com.jbrick2070.otr.streamDeckPlugin
```

The package in `dist/` is the one people install, so it is committed, and `tests/package.test.ts`
fails when it no longer matches the source: every file in the plugin folder byte for byte, and
`bin/plugin.js` against a fresh build. After a change, run `npm run pack` and commit the package
with it. The packer writes `manifest.json` back without its final newline; that is how it is kept.
Raise `Version` in the manifest for a release, so an installed copy can tell it is older.

`API-CONTRACT.md` lists every route the deck uses and what it expects back.
`tests/helpers/fake-studio-server.ts` is a stand-in Studio that enforces the server's two
rules (loopback Host, JSON posts); `../tests/test_rps_streamdeck_contract.py` runs the same
calls against the real server.

MIT, like the rest of this repository. The shape (SDK 3, one store, rate-limited SVG faces)
follows [streamdeck-fadester](https://github.com/jbrick2070/streamdeck-fadester). ComfyUI is
the open-source project this pack runs on; the node look is a nod to its graph editor, and no
ComfyUI or Comfy Org logo is used.
