# CreatorOS concepts -> OTR: coding plan

Baseline `main` 90413ac3, ComfyUI core 0.42.1, frontend 1.55.18, 2026-10-24. Every seam below was
read in the real files; re-read them at the commit you start from. Plan only: nothing is built.

**Where the ideas come from.** [thePloggy/CreatorOS-Studio](https://github.com/thePloggy/CreatorOS-Studio),
Apache-2.0, a Node/React studio that drives a local ComfyUI (built for Comfy's #OpenSourceHiggsfield
challenge). No CreatorOS code is copied: every row re-implements an idea in OTR's own Python/JS. If
code is ever copied verbatim it keeps its Apache-2.0 header and a change note, which is compatible
with OTR's MIT licence.

## Scope rules (hold for every row)

1. **No Ollama, no llama.cpp.** Nothing here adds, calls or depends on either. CreatorOS's Ollama
   features (GPU hand-off between Ollama and ComfyUI, prompt rewriter, vision checks, CPU fallback)
   are out.
2. **`otr/obs/` is untouched.** Every finished episode still publishes there (CLAUDE.md 2026-08-17;
   mux ruling 2026-10-15 at `nodes/otr_master_audio_mux.py:2411`). No row moves, hides, sorts or
   withholds an obs file.
3. **No fallbacks.** A value the code cannot read is reported as unknown, with the reason. A
   display-only feature (live view, notification, hint) never changes or gates a render.
4. **Story and music are done** (CLAUDE.md 2026-08-04, 2026-09-17). No row touches prompts or prose.
5. **Wired in the same change.** Grep for a non-test caller of every new helper before pushing.
6. **No new widgets planned.** If a row ends up needing one: append-only, then
   `build_variants.py --all` and the four checks in CLAUDE.md section 0.
7. **Two boxes.** `ListAgents` + `SendMessage` the peer window before touching a file it owns. State
   the 5080 / 4060 blast radius in every commit message (CLAUDE.md 0B).

## What we are not taking, and why

| CreatorOS feature | Why not |
|---|---|
| Setup wizard, model manifest, per-recipe missing node/model check | Already stronger here: `scripts/otr_provision.py` (installs, prints a receipt), `scripts/otr_asset_index.py` (generated "to use X, download Z"), `js/lane_node_packs_core.js` (missing pack named when the workflow opens) |
| Render watchdog, ComfyUI link state | `scripts/otr_render_watchdog.ps1` (heartbeat stall, queue down) |
| Per-asset provenance, replayable recipe | The ledger plus replay bundles (`scripts/otr_freeze_replay_bundle.py`, `scripts/otr_verify_replay.py`) |
| Redo one shot, keep the rest, re-cut | Scene Producer: kept clips, takes, reuse by causal key (`nodes/otr_video_render_batch.py:429` `_producer_clips`, `:663`) |
| ETA learned from past runs | Radio Play Studio already calibrates from saved timings (`contest/radio-play-studio/radio_play_studio/runner.py:536`, `save_timing` at `:560`) |
| Re-attach to a job after a restart | Studio looks for a lost submit in ComfyUI's queue and history (`runner.py:58` `HISTORY_LOOKBACK`, `:762`) |
| Prompt files re-read on every job | Its value is prompt iteration, and story quality is done |
| Ollama GPU hand-off, CPU fallback | Rule 1; a silent CPU fallback also breaks rule 3 |
| Reject = move the file out of the library | Rule 2 |
| Per-recipe wiring sidecars (`.pilotage.json`) | One canonical graph |
| Public-demo owner key, visitor stats, React/mobile shell | Not OTR's shape |

## The plan at a glance

| Sprint | Rows | Touches | Review before code |
|---|---|---|---|
| **S1 Headless ops** | 1.1 live view, 1.2 failure hints, 1.3 phone ping | `scripts/`, new `config/node_phase_labels.json` | None: each row has one verifiable answer. 1.1's preview half is a measurement first |
| **S2 Environment fingerprint** | 2.1 stamp, 2.2 known-good + drift report | `nodes/` (ledger), `scripts/` | Full `kibitz-plugin:kibitz` arc: it is a ledger change |
| **S3 Studio proposals** | 3.1 to 3.5 | `contest/radio-play-studio/` | Fold into `planning/studio-ui/DESIGN_ANCHOR.md` through its owner first |
| **S4 Broadcast feed** | 4.1 | publish boundary, routes, `js/` | Full arc (D1 decided: A, air by default, pull from air) |

Every chunk, every sprint: scoped suite + Bug Bible regression green, push to `main`, then
Composer/Sonnet QA plus one driver-aware CLI lane on the pushed diff (CLAUDE.md 2026-09-19,
2026-09-07). Live proof is a 30-word canonical leg that reaches `otr/obs/`.

## S1 Headless ops

### 1.1 Live view in the headless runner

**Gap.** `scripts/otr_api.py` only polls `/history` (`poll_history`, `:879`). Its `on_tick` heartbeat
tells the leg log "pending" for the whole of a video walk that runs for most of an episode: never
which node is running or how far along it is. Yet the nodes already report:
`nodes/_otr_shared/node_progress.py` drives ComfyUI's progress bar for the writer's beat loop, the
local voice render, the scene sequencer and the video walk (`WeightedProgress`). ComfyUI sends those
frames on the websocket of the `client_id` that submitted the prompt, and nobody listens.

**Build.**
- `otr_api`: open `ws://<host>/ws?clientId=<id>` on a background thread with `aiohttp` (a ComfyUI
  dependency, already in the venv: confirm the import first) and pass the SAME id to
  `submit_prompt` (`:733` already takes `client_id`). The thread ends when `poll_history` returns.
- Text frames: `executing` (node id -> `class_type` from the submitted API prompt -> plain label) and
  `progress` (value/max). The listener keeps the latest `{node, label, value, max}` and hands it to
  the existing `on_tick` status dict under a `live` key, so the heartbeat line callers already print
  reads `node 7 <label> 34/96` instead of `pending`. One line format, no second log stream.
- Labels: `config/node_phase_labels.json`, one plain label per class in `node_list.json` (22 today)
  plus a few core patterns (sampler, decode, load, save). A test fails when a class in
  `node_list.json` has no label, so a new node cannot run unnamed.
- Previews, measured first. Core honours a per-prompt `extra_data.preview_method`
  (`ComfyUI/execution.py:734`). Probe: one 30-word leg with `"preview_method": "latent2rgb"` merged
  into the existing `extra_data` (otr_api sets the Comfy key there, `:755`); count binary frames
  during the video walk. The walk quiets its inner samplers' bars (`QuietInnerBars`); whether that
  also silences previews is exactly what the probe answers.
  Frames arrive -> write the latest to `<leg log stem>_preview.jpg` (a diagnostic, overwritten, not
  a deliverable). None -> drop previews and record the measurement in this file.

**Done when.** Unit tests for the frame parser and the label table are green; a live 30-word leg's
heartbeat names the node and its progress through writer, voices and video walk, and the leg
reaches `otr/obs/`.
**Out of scope.** Changing the watchdog's stall rule to read progress (a follow-up once the
heartbeat carries it).

### 1.2 Failure hints

**Gap.** `describe_execution_error` (`scripts/otr_api.py:794`, called at `:969`) already names the
node, the exception and the traceback tail. OTR's own refusals already say what to do. Platform
failures do not: CUDA out of memory, a model file ComfyUI does not list ("Value not in list"), a
truncated weights file, a missing Python module, CUDA gone, a VAE or LoRA from another model family,
the server gone.

**Build.**
- `scripts/otr_known_failures.py`, pure: rows of `(pattern, hint, refs)`, first match wins,
  `hint_for(text) -> str | None`.
- Hints point at OTR's own tools: "Value not in list" -> `otr_asset_index.py` /
  `otr_fetch_lane_weights.py` / `otr_provision.py`; missing module -> `otr_venv_audit.py`; server
  gone -> the watchdog verdict and the server log; out of memory -> close other GPU programs or use
  the smaller workflow JSON for this card (the tier JSONs are the protection, 2026-07-03).
- `refs` are Bug Bible ids or `apple/PROD_BUG_LOG.md` PBUG ids found by grepping both at build time.
  A row with no matching entry has empty refs. Never invent an id. The Bible stays read-only here:
  entries change only in the operator's fan-out.
- Wire: `describe_execution_error` appends ` -- hint: ...`; the `node_errors` refusal in
  `submit_prompt` does the same.

**Done when.** Each row matches its fixture string; each ref resolves (PBUG in
`apple/PROD_BUG_LOG.md`, Bible id in the survival-guide `BUG_BIBLE.yaml`; a missing Bible repo fails
the test with a named message, not a skip); a real refused prompt (a bad combo value sent through
otr_api) prints its hint.

### 1.3 Phone ping

**Build.**
- `scripts/otr_notify.py`: one function, `notify(title, body)`, POSTs to `OTR_NOTIFY_URL` (an ntfy
  topic URL: ntfy.sh, free, or self-hosted). The runner prints its state at start:
  `notify: on -> <host>` or `notify: off (OTR_NOTIFY_URL unset)`, so it is never a hidden switch.
- `scripts/otr_canonical_api_run.py`, after the `RESULT` line (`:657`): SUCCESS -> episode title and
  obs filename, sent only after the obs file is confirmed on disk; failure -> node, exception, and the
  1.2 hint.
- `scripts/otr_render_watchdog.ps1`, on DEAD (exit 2): "leg dead: <stall | queue down>".
- A failed POST is logged and changes nothing about the run.
- Privacy: anyone who knows an ntfy.sh topic name can read it. Use a long random topic or self-host.
  Messages carry no local paths beyond the obs filename.

**Done when.** A test against a local HTTP stub; one live leg pings the phone.

## S2 Environment fingerprint (full kibitz arc before code)

**Gap.** A ledger records only OTR's own commit (top-level `"commit"`,
`nodes/production_ledger.py:1178`). When an episode breaks after a ComfyUI or node-pack update,
nothing on disk says what moved, and with two boxes there is no quick way to see how they differ.

### 2.1 Stamp the environment into every ledger

- `nodes/_otr_shared/env_fingerprint.py`: stdlib only, computed once per process (it is a boot
  constant), bounded in time. Fields:
  - ComfyUI: `__version__` from `comfyui_version.py`, plus the commit when the core tree has `.git`.
  - Every custom node folder: commit, or the `pyproject.toml` version when there is no `.git`, or
    `unknown: <reason>`. Read `.git/HEAD`, `refs/` and `packed-refs` directly, not 40 `git`
    subprocesses; handle a `.git` file that points elsewhere (worktrees).
  - Python, torch, the CUDA build, GPU name, driver (one bounded `nvidia-smi` call; on AMD or Mac,
    `unknown: no nvidia-smi`), OS.
- Stamp it under `meta.env` (+ `meta.env_version`) where the ledger is created (`Ledger.__init__`,
  `nodes/production_ledger.py:1175`).
- Replay: add `"env"` to `_REPLAY_RUN_VOLATILE_META` (`nodes/production_ledger.py:584`). A replay runs
  on today's machine, so it records today's environment, never its source's.
- Arc questions: bump `_otr_ledger.CURRENT_SCHEMA_VERSION` or not; a dirty flag per repo (worth the
  cost?); compute at prestartup or at the first ledger; which machines' behaviour changes (none: the
  stamp is read-only metadata; prove it per CLAUDE.md 0B).

### 2.2 Known-good snapshot and drift report

- When `otr_canonical_api_run.py` ends SUCCESS and the obs file is on disk, copy that episode's
  `meta.env` to `<otr_state_dir()>/known_good_env.json` with the episode id and time
  (`nodes/_otr_paths.py:744`, per machine, under the output tree, never committed). Also
  `--freeze <episode>` by hand.
- `scripts/otr_env_drift.py`, read-only:
  - no arguments: this machine now vs its known-good;
  - `<episode A> <episode B>`: diff the two ledgers' `meta.env` ("worked yesterday, broken today", or
    "works on the 4060, not the 5080").
  - Output: Python, torch and driver changes first, then packs changed (old -> new), added, removed.

**Done when.** Suite + Bible green; a live 30-word leg reaches `otr/obs/` with `meta.env` in its
ledger; a replay of it stamps the replay machine's env; `otr_env_drift.py` run on two real ledgers
names a real difference (or none).

## S3 Radio Play Studio proposals (through the design anchor first)

Studio ships inside the pack (`contest/radio-play-studio/`), its page is mid-build to
`planning/studio-ui/DESIGN_ANCHOR.md` (v2, 2026-10-23), and that anchor's rules govern: nine
languages, contrast tokens, keyboard access, and "the lamp is the only thing on the page that
moves". These rows go to the window that owns the anchor as proposals to fold in, not as code to
drop on the page.

| Row | Proposal | Grounding |
|---|---|---|
| 3.1 | **Tab title shows the run.** While a stage runs: `(42%) Radio Play Studio`. A stage that finishes while the tab is hidden: `(1 ready)` until the tab is looked at. Strings through i18n. | The page sets no `document.title` and has no `visibilitychange` handler today |
| 3.2 | **The running step in plain words** in the status line ("voicing the cast", "filming shot 12"), from the 1.1 label table. Labels become i18n keys, not English strings. | `runner.py:153` `stage_note` already carries `current_node` |
| 3.3 | **Say when a step runs long.** When the running node passes its calibrated time, say "this step is taking longer than last time" instead of letting minutes-left sit still. | `runner.py:1022` `seconds_left`: an overrunning node contributes `max(0, history - elapsed)` = 0, so the estimate goes quiet without saying why |
| 3.4 | **Before / after on a repainted card.** On a card with a new still take, flip or slide between the kept still and the new one, so keep vs new is a visual choice. Needs the previous take's file to still exist: confirm in the Scene Producer store first. | The anchor already plans "New take" badges (`DESIGN_ANCHOR.md:168`); takes live in `draft.takes` (`static/index.html:1576`) |
| 3.5 | **Phone ping when a stage finishes**, reusing 1.3's `notify`. Where that function lives so both the scripts and Studio can import it is settled when this row is folded in. | Studio stages run for many minutes |

Not proposed: a live preview on the cards. It would be a second moving thing and breaks the
anchor's lamp concept; the anchor's owner can revisit it.

## S4 Broadcast feed (D1 decided: A; full arc before code)

**Gap.** The README's 24/7 broadcast points OBS at `otr/obs/`, so every leg airs, including probe
legs whose harness label becomes the title card. `otr/obs/` must stay the dailies and the success
signal, so the fix cannot touch it.

**D1, decided by the operator 2026-10-24: A, air by default, pull from air.** A separate broadcast
folder that OBS watches. Every publish also adds a hardlink there (same volume, no extra disk). A
review tab can pull an episode from air (removes only the link) or put it back. The unattended 24/7
feed keeps working. Not chosen: B (curated, nothing airs until sent) and C (skip S4).

**Constraints the arc must honour.**
- The output-tree contract allows exactly `episodes` + `obs` at the `otr/` top level
  (`nodes/_otr_paths.py:751`). The feed goes under `otr/episodes/_shared/broadcast/`, unless the
  operator amends that contract.
- `OTR_OBS_DIR` may be pinned elsewhere (`nodes/_otr_paths.py:785`). A hardlink across volumes fails:
  report it loudly in the mux report and never touch the obs copy.
- The link is made inside `_publish_to_obs` (`nodes/otr_master_audio_mux.py:1945`) after the obs copy
  succeeds, so all three publish paths that call it (`:1347`, `:2073`, `:2421`) get it.
- New routes sit beside `/otr/latest_ledger` (`__init__.py:418`): `GET /otr/episodes` (obs MP4s with
  a ledger summary: title, bank, lane, length, rights receipt, on air or not) and
  `POST /otr/broadcast` `{episode, action}`. Episode ids only, never paths; confine with
  `nodes/_otr_paths.py` (`is_readable_in_roots`, `confine_to_output_tree`); the routes can only add or
  remove links in the broadcast folder. ComfyUI routes have no login, so on a `--listen 0.0.0.0`
  server anyone on the LAN can use them; the worst they can do is pull an episode from air.
- UI: a ComfyUI sidebar tab (`app.extensionManager.registerSidebarTab`, present in the installed
  frontend 1.55.18), in the repo's pattern: `js/episodes_tab_core.js` (pure, tested with
  `node --test` under `tests/js/`) + `js/episodes_tab.js` glue.
- README section "Continuous 24/7 broadcast" points OBS at the feed folder.
- Later, not built here: the planned obs-playlister plugin can read the same on-air list.

**Done when.** A live leg lands in `otr/obs/` and in the feed; pull and put back work from the tab;
`otr/obs/` is byte-identical before and after a pull.

## Order and owners

1. **S1 now.** Scripts only, no graph change. Whichever coder window takes it messages the peer first.
2. **S2 after its arc.** `nodes/` is 5080-owned. The arc settles the schema questions in 2.1.
3. **S3 when the Studio page reaches the anchor's build.** Hand 3.1 to 3.5 to the anchor's owner.
4. **S4 after its arc.** D1 is decided (A, 2026-10-24). The publish boundary and `nodes/` are
   5080-owned; `js/` and the routes go with them.

Delete each row from this file when it ships. Its receipt goes in the commit message, not here.
