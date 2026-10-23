# OTR Scene Producer - go-forward plan

Baseline `main` 3d77b76f, ComfyUI core 0.42.1, frontend 1.55.18, comfy-sdk 0.5.0, comfy-api-proxy 0.1.6, 2026-10-22. Since release 2.3.37 Radio Play Studio ships inside the pack, so Studio changes reach registry users too. Every seam below was read in the real files; re-read them at the commit you start from. The approved page design is `mockup.html` in this folder. Implementation and live proof are outstanding.

## What it is

An optional shot editor for Radio Play Studio, built and proven first on the My Story bank. The first producer run writes, voices and paints the episode exactly as today, then stops before video. The user sees one card per shot (about one per spoken line), grouped by scene, each with its still. They paint again, edit or keep a few cards or none, film, then publish. Only what changed is re-rendered. Audio, cast, voices and timing never move on a visual edit. A partial result never publishes. ComfyUI runs everything; Studio edits and submits. With the one new widget blank, every workflow, bank and Studio form behaves exactly as today, in the same node order.

## What we reuse

| Already built | Where | What it gives Producer |
|---|---|---|
| Replay bundle and importer | `production_ledger.py` `import_replay_bundle`, `load_replay_manifest`, `_confine_replay` | The checkpoint: immutable, hashed, master verified, fresh workspace per run; extra members verify and materialize; a relative `replay_from` anchors at ComfyUI's output folder |
| Freezer logic | `scripts/otr_freeze_replay_bundle.py` | Moves into `nodes/` (below) |
| Audio pass-throughs on replay | writer, Freeze, CastLock, TTS, music, SceneSequencer; EpisodeAssembler re-hashes the master | No LLM, TTS or music after the first run, with no change to those nodes |
| Replay branches in the queue-time gates | `_otr_visual_assets.plan_prompt`, `cloud_slug_preflight`, `cloud_balance_preflight`; `_otr_lane_rolls` | Where the resolver plugs in; lane rolls already keep frozen lanes on replay and need no change |
| Workflow Validator, node 63 | `_otr_workflow_validator.py`; gates the writer, voices and director | The one place that can refuse a whole prompt before anything runs |
| Shot Lock replay branch | `otr_shot_lock.py` ~3577: reuses frozen `video.shots`, no LLM | The overlay seam |
| Shot rows | role, char_id, start_s, dur_s, `creative`, `source_line_ids` | Card content; scenes via beat and shot `scene_id`; ungrouped is valid |
| Still cache | dispatcher `request_cache_key` | Unchanged cards and portraits reuse |
| Request builder and receipts | `render_driver.build_request_from_shot`; `_RECEIPT_CAUSAL_KEYS` | The basis of the clip reuse key; receipts stay as they are |
| `ExecutionBlocker(None)` | core `graph_utils.py` ~140, `execution.py` ~265 | Silently skips everything downstream; the prompt still ends as success |
| `/view` | core `server.py` ~596 | Studio's server reads bundle files; no new route |
| SDK and proxy | `comfy_sdk` jobs, outputs; proxy `app.py` | A blocked stage is a succeeded job; the job id is the ComfyUI prompt id; `client.jobs.get(id)` re-attaches after a Studio restart; clips surface under the `gifs` kind |

## Graph contract

**One widget, in ComfyUI's advanced mode.** Shot Lock gains one optional input, its only widget, appended last:

```python
"direction_json": ("STRING", {"default": "", "multiline": True, "advanced": True,
                              "tooltip": "Written by Radio Play Studio's shot editor. Leave empty."}),
```

- **`advanced: True`** is ComfyUI's flag for a power-user input. Frontend 1.55.18 hides it behind the node's advanced toggle and lists it under Advanced inputs in the side panel. It is visibility only: the widget still saves its value, so positions in `widgets_values` never shift.
- **Not `socketless`.** The frontend creates no input entry for a socketless widget. That breaks the repo's rule that every widget value has an input descriptor (`tests/test_canonical_widget_input_parity.py`), and the server does not enforce it anyway. The Workflow Validator's literal-only check is the guard instead.
- **Off the app form.** `config/app_mode.json` lists its rows by name, and this one is not added.
- **No node modes.** Every node stays at Always; no mute or bypass is used, so a saved graph never changes shape.
- **Saved graphs.** Canonical serializes `[""]` plus one input descriptor; every generated workflow regenerates. Old `[]` graphs already convert on all three converters. The writer gains nothing: Studio sets its existing `replay_from`.

**No pruning, no new output nodes.** Studio submits the whole shipped graph. Core runs output nodes first, so turning Shot Lock or the dispatcher into one would reorder ordinary runs; neither changes. A stage ends where its last node, after freezing, returns `ExecutionBlocker(None)` on every output.

| Stage | Run | Ends at |
|---|---|---|
| Stills, first | Fresh My Story run, as today | Dispatcher #91: freeze, block |
| Stills, again | Replay of the latest bundle, no LLM | Dispatcher #91: freeze, block |
| Video | Replay of the latest bundle | Batch #92: freeze, block |
| Publish | Replay of the latest Video bundle | Mux #85, unchanged; no freeze |

**Direction JSON**, versioned: `project_id`, `stage`, `out` (the bundle this run writes), `source` (previous bundle's manifest digest; absent on the first run), scene text, per-card picture text and motion text, take numbers per portrait, still and clip, and kept ids. No selection in V1: each stage acts on everything changed.

**The stage lives only on the wire.** Shot Lock sets it on the ledger it passes downstream, as a runtime key, on every run, and clears it when the widget is blank. The durable save and the freeze strip it. A blank-widget replay of a producer bundle therefore reaches the mux.

**Validation is in the Workflow Validator.** `VALIDATE_INPUTS` cannot refuse a prompt: core drops only the outputs that depend on the failing node and still runs the writer. Node 63's resolver checks shape and version, refuses a linked or non-literal direction, confines `project_id` and `out` to short safe names, refuses an `out` that exists, and matches `source` against the bundle in `replay_from`. These checks run before both `_queue_time_readiness_gates` calls, which otherwise roll lanes, preflight clouds and fetch assets. A raise there halts the prompt before any asset loads. Execution is sequential, so a duplicate queued behind its twin fails at node 63 once the first has frozen `out`.

## Stages

**Stills.** On replay Shot Lock writes card motion text and clip takes onto the shot rows. On the first run the dispatcher stores the image prompt payload it received, before its own merge, as `images.producer_prompt_payload`. It must be added inside the dispatcher's own rebuild of the images section, which replaces the whole section with six named keys. On replay the image prompt node rebuilds from that payload. A card edit replaces only that object's prompt text, with the scene text appended; identity, co-references and every other field are kept, and no template or LLM runs. The dispatcher, in producer mode, runs its cached path instead of verify-only and mints only misses, portraits first.

Take numbers above zero enter every seed branch: bookend, fixed mode, scene character and portrait. A character's take enters the identity seed of every scene still of that character, whether or not the lane mints a portrait. Studio's My Story graphs mint none without photos: scene stills take their identity from a hashed identity prompt. Take 0 is today's seed exactly. Kept objects keep their row whatever their key. Rows are one per object, newest wins, and the render driver's portrait indexes become newest-wins. Paint again raises the still's take; Film again raises only the clip's take and keeps the still. Node 12 renames the workspace on every run; Shot Lock already adopts the renamed id and rebases every path before the dispatcher, which prefers that wired id, so minting on replay writes into the right folder with no change.

**Video.** Each shot's reuse record lives in `video.producer_clip_reuse[shot_id]`, a map beside the shot rows, never on them, because `ShotRow` forbids extra fields. Paths are episode-relative. Shot Lock's replay branch re-stamps the imported video section, so the map survives; Batch writes it when it stamps its video section. It holds the full clip manifest row, receipts, segments and foley stem. Its key is a hash of every segment's request with image and audio paths replaced by file hashes, plus recipe id, model file path, size and mtime, engine id and version, and clip take. The clip's own sha is verified on use. A hit rehydrates Batch's clips and receipts, so render engines, the manifest, credits and foley rebuild exactly as if rendered. A miss re-renders the whole shot. The clip take re-derives the seed. Reuse is decided before any engine or cloud call. `_RECEIPT_CAUSAL_KEYS` is untouched, so old bundles still verify.

**Publish.** Batch, in the publish stage, raises before any engine if a shot lacks a verified reusable clip. Otherwise it rehydrates every shot, and composite, captions, credits and mux run as today. No generation weights load.

**Freezer.** It moves to a helper in `nodes/` with relative imports. It raises `ReplayBundleError`, never `SystemExit`, which would kill ComfyUI's prompt worker, and takes an explicit destination. The CLI becomes a thin wrapper. It carries every member present: ledger, master, canon, stills, portraits, clips and `audio/foley`. Bundles live at `otr/episodes/_replay/p/<project>/<nn>-<stage>`, short for Windows path limits. Nothing is deleted automatically in V1: each stage run adds one bundle and one workspace, and the operator removes old ones by hand.

**Readiness.** One pure resolver, called by node 63, `_otr_visual_assets` and the two cloud preflights. It finds Shot Lock through all links and rejects an ambiguous graph. A Stills replay needs image-engine assets, a Video replay clip assets, Publish none, and no replay needs writer assets.

**Known cost.** Node 12's CRT base video renders and renames on every stage, measured at about 53 frames a second, a few minutes for a typical episode. V1 accepts it.

## Studio

The approved design is `mockup.html`.

- **Entry.** A quiet second button, "Direct the shots first", sits beside "Make the episode" on the My Story path. The form stays first and folds to one line once a project exists; the board takes the run section.
- **Runs.** Every run submits the shipped graph through the existing busy guard. The first takes the user's My Story choices, and the project records which workflow it used: the reference twin when photos were given. Later runs reuse that recorded workflow and the first run's recorded form choices, because `runner.workflow_for` picks the twin only when the bank is my_story and the image engines come from the video director's widgets. Node 63 refuses a replay whose image engines differ from the bundle's. They set `replay_from` to the previous bundle, Source to `roll (any eligible bank)`, and `source_ref` and every My Story box to empty, which the writer's replay check allows. A small internal patcher writes those values and `direction_json` by node type; the form's `apply_choices` rejects unknown fields and stays unchanged.
- **Results.** A succeeded job ends a stage; only Publish picks the mux output. Studio's server reads `cards.json` and media from the bundle through ComfyUI's `/view`, copies them by sha into `out_dir/projects/<project_id>/`, and serves them itself, so the page never calls ComfyUI and pruned bundles never blank it. `cards.json` is a small versioned view the freeze writes: shots grouped by scene, each with its still or clip, takes and state.
- **Page.**
  - **Cast:** one picture per character with Paint again; it repaints that character's shots unless kept. The picture is the portrait when the lane mints one, otherwise the character's first scene still.
  - **Scenes:** collapsed, opening when one holds a changed or failed card.
  - **Card row:** thumb, who, the spoken line, Picture and Motion text, Paint again, Film again, Keep. Paint again repaints the still; Film again keeps the still and films a new take.
  - **Buttons:** Paint the stills (N changed), Film the shots (N to film), Publish the episode, each with a one-line reason when disabled.
- **Progress.** In producer mode the dispatcher and Batch each drive one node-level bar over their items, and the page marks the k-th card of the work order.
- **Project file.** One atomic file holds the draft direction, the bundle names in order, and the pending `out` with its job id. After a restart, `client.jobs.get` re-attaches a running job; a succeeded job with `cards.json` finishes the stage; a failed or unknown job with no `out` offers the run again.
- **Languages.** New strings go into every language file (the parity test enforces it).
- **README.** The README gains a Producer section only after the proof passes. When it mentions the feature relative to contest judging is the operator's call.

## Not in V1

| Cut | What you give up |
|---|---|
| A separate Prepare stage | Cards before the first stills; an edited card costs one extra still |
| Graph pruning in Studio, output-node changes, canvas stage files, `VALIDATE_INPUTS` checks, `socketless` | Nothing |
| Automatic deletion of old bundles and workspaces | Disk grows by one bundle and one workspace per stage run |
| Second checkpoint format, writer widget, revision store, journals, receipt transport, separate prompt or direction files, per-clip sidecars | Nothing; the ledger carries them |
| Selection of individual cards | Each stage acts on everything changed |
| Template prompt rebuilds, per-field structured edits | One picture text and one motion text per card, plus scene text |
| Multiple takes per card, history beyond the previous bundle | Rollback to older takes |
| Banks other than My Story, auto-freezing ordinary episodes, reordering or splitting shots, bbox or location-image conditioning | Later; the backend is bank-agnostic |

## Build order

Each chunk: scoped suite green on the Windows venv, named files only, pushed to `main`, then Composer and Sonnet QA on the pushed diff. Every chunk touches shared code, so each commit states that blank runs are unchanged and how that was measured.

1. Freezer helper in `nodes/`; new members; `direction_json`; node 63 validation and resolver; canonical, generated workflows and parity tests.
2. Shot Lock runtime stage and overlay; stage blockers in the dispatcher and Batch; producer-only UI outputs and bars.
3. Prompt payload in the ledger; replay rebuild; dispatcher minting, takes, kept, newest-wins rows; Stills freeze and `cards.json`.
4. Clip reuse map, takes, publish refusal; Video freeze.
5. Studio page, recorded workflow, internal patcher, project file, restart, translations.
6. Live proof on the real My Story workflow, then record it.

## Implementation traps

- `ExecutionBlocker` is imported lazily from core, never redefined, and stubbed in tests. It is always given `None`; a message turns the skip into an execution error.
- UI values are lists, and clips go under `gifs`.
- Shot Lock's `IS_CHANGED(**kwargs)` stays as is; a test calls it with no arguments.
- `prompt_objects` is node 89's payload as the dispatcher received it; the dispatcher's own merge raises on a duplicate id.
- The freeze reads the post-rename workspace; node 12 renames before Shot Lock runs.
- Every bundle is frozen before the mux ever runs, so its master is the assembler's, never the foley mix, and Publish always replays a Video bundle.
- An edited My Story still records its source receipt as `stale_source_receipt`, the dispatcher's existing label for a changed prompt. Nothing gates on it; it is the truthful record of a user edit.

## Proof

- **Blank widget:** on the 5080, node order, VRAM peak and the published file match a baseline. Old `[]` vectors, the app form and mux discovery are unchanged. A blank-widget replay of a producer bundle publishes.
- **Zero supplied images on My Story:** Stills, Video and Publish produce a playable final in `otr/obs/`. Master, cast and timing hashes equal bundle 1's. Only the first run calls an LLM. With photos supplied, the reference twin works the same way.
- **One card edit:** exactly that still and that clip re-render; the reports show the rest reused.
- **Paint again:** gives a different image, and take 0 is byte-identical to today. A portrait's Paint again repaints only that character's unkept shots.
- **Film again:** reuses the still and re-renders the clip.
- **Video, then Stills again, then Video:** re-renders only the changed shots, with foley intact.
- **Duplicates and bad input:** a duplicate queued behind its twin, a tampered bundle, a linked direction, a missing still or changed audio is refused at node 63, before any model loads.
- **Publish:** with a shot missing, it refuses before any engine. Complete, it publishes and loads no generation weights.
- **Studio restart:** a running job re-attaches, a finished one completes, and a failed one can run again.
- **Widget:** `direction_json` is hidden behind Shot Lock's advanced toggle, absent from the app form, and its value survives a save and reload on the canvas.
- **Registry zip:** `planning/` stays out of it.
