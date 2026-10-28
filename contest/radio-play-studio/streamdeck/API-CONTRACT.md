# What the deck asks the Studio (2026-10-28)

The plugin is a client of Radio Play Studio's own server, `../radio_play_studio/server.py`,
on `http://127.0.0.1:8765`. It uses the routes the page uses and nothing else; the server
did not change for it. Two of the server's rules shape every call:

* **A loopback `Host` only** (`_host_ok`). A Node `http.request` to `127.0.0.1:<port>` sends
  `Host: 127.0.0.1:<port>`, which passes. Nothing is loosened.
* **JSON posts only** (`Content-Type: application/json`, a declared `Content-Length`).

`src/types/contract.ts` is the typed form of this file; `tests/helpers/fake-studio-server.ts`
is a stand-in that enforces the two rules above. On the Studio side,
`tests/test_rps_streamdeck_contract.py` pins the shapes below.

| Route | Used for | Answer |
|---|---|---|
| `GET /api/options[?workflow=<file>]` | The form and the machine menu for one workflow (the server's own without the query). Read on start, when the Studio comes back, and when the machine dial switches. Takes ~40 s while ComfyUI renders. | `{fields: [{key, label, group, help, kind, default, choices?, min?, max?}], photo_stills, menu?: {machines: [{key, ways: [{key, workflow, proven}]}], detected, current: {machine, way, workflow}}}`. `menu` is absent when the server was started on a graph that is not on the menu; the machine dial then has nothing to switch and no `workflow` is posted. `503 {error}` while ComfyUI cannot be read; `400` for a workflow not offered. |
| `GET /api/episodes` | Who is making what: the runs this server has started. Polled every 10 s while idle (5 s while the Studio is off), so a run the page starts is followed, and the GPU is seen as busy during a Scene Producer stage. | `{episodes: [{id, finished, started, project_id?, stage?}]}` newest first. A row with `project_id` is a stage run: busy, never followed. |
| `POST /api/episodes` | Make the episode. Body `{"choices": {language, bank, style, acts, workflow?}}`: what the four dials show, in `form.FIELDS` keys, plus the menu workflow (`runner.WORKFLOW_CHOICE`). Fields not posted keep the workflow's own values. | `202 {id}`. `409 {error}` while anything is being made (an unfinished episode is followed; anything else is shown as busy with the server's words). `503` ComfyUI unreachable. `400` a refused choice. |
| `GET /api/episodes/<id>/events` | Progress, as `text/event-stream` with `id:` and `data:` lines and a `: keep-alive` every 15 s. `Last-Event-ID` resumes a cut stream. | `submitted`, `progress {value, message?, phase?, minutes_left?, nodes_done?, nodes_total?}`, `step {value, step, steps, phase?}`, `status`, `output`, `note {message}`, then `done {path, run, file}` or `error {message}`. The share only moves forward and stays under 1 until `done`; the key shows 100% on `done` only. |
| `GET /episodes/<run>/<file>` | Play the finished episode: opened in the browser from `done.run` and `done.file`, each URL-encoded. | The file. |
| `GET /#run=<id>` | Hold the key: the page opens on the run the deck is following. | The page. |

Not used on purpose:

* `POST /api/presence`. The page's heartbeat keeps the GPU's models loaded while a tab is
  open; a deck that never closes would keep them loaded for good. Without it the Studio frees
  them `GPU_IDLE_S` (150 s) after the deck's run ends and no page is open, which is what a
  closed page gets. A run that is going is never freed: `should_free_gpu` checks for one.
* `GET /api/keys`. Whether a cloud lane needs a key depends on the stills and writer picks,
  which the deck does not make; the page says so where those are chosen.
* The Scene Producer routes (`/api/projects/...`). Directing shots is the page's work.

The deck's picks and the page's form are separate: the page remembers its own values in the
browser, the deck keeps its in Stream Deck global settings. What runs is whichever pressed
Make. Picks the next form does not offer (another machine, another menu) fall back to that
form's defaults.
