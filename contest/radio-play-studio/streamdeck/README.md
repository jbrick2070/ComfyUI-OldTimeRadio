# Radio Play Studio for Stream Deck +

Make a radio play without touching the page: four dials pick the language, the story bank,
the visual style and the number of acts, a fifth picks the machine and how it films, and one
key makes the episode and fills in as it goes. It drives [Radio Play Studio](../README.md)
on this computer, over the same local HTTP the page uses; the Studio did not change for it.

## What is on the deck

| Control | Turn | Push | Face |
|---|---|---|---|
| **Choice** dial (one per field; pick the field in its settings) | next or previous choice | back to the form's default | the field's name, the pick, a bar while an episode is made |
| **Machine** dial | rest on a machine and a way of filming (`16 GB · foley`, `8 GB · stills`, ...; `?` marks an unproven row) | switch the Studio to it: the form is read again for that graph | the row the next episode runs on |
| **Make** key | — | make the episode from what the dials show; when done, play it; after a failure, clear it. **Hold** to open the Studio page on the run. | a ring that fills, the percent, the phase and minutes left; a check when done; the reason in red when it failed |

The story bank dial skips *my story*: that bank needs typed text, which stays on the page.
While the page is making an episode the key follows that one; during a Scene Producer stage
the deck shows busy and the dials wait.

## Install

Stream Deck 7.1 or newer, Node 24, and Radio Play Studio running on this computer.

```bash
cd contest/radio-play-studio/streamdeck
npm install
npm run build
npm run link
```

`npm run link` registers `com.jbrick2070.otr.sdPlugin` with the Stream Deck app (the Elgato
CLI's `streamdeck link`). Drag the three actions onto a Stream Deck + page. If the Studio
runs on another port, set it in any action's settings; the host is always `127.0.0.1`,
because the Studio answers no other.

## Develop

```bash
npm test          # vitest: dial math, the event stream, the store against a fake Studio
npm run typecheck
npm run validate  # the Elgato CLI's manifest check
```

`API-CONTRACT.md` lists every route the deck uses and what it expects back.
`tests/helpers/fake-studio-server.ts` is a stand-in Studio that enforces the server's two
rules (loopback Host, JSON posts), so the tests prove the plugin works within them.

MIT, like the rest of this repository. The shape (SDK 3, one store, rate-limited SVG faces)
follows [streamdeck-fadester](https://github.com/jbrick2070/streamdeck-fadester).
