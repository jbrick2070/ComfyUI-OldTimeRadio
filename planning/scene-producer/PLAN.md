# OTR Scene Producer - go-forward plan

Baseline `main` 0c4a509d, ComfyUI core 0.42.1, comfy-sdk 0.5.0, comfy-api-proxy 0.1.6, 2026-10-22. Every seam below was read in the real files; re-read them at the commit you start from. The approved page design is `mockup.html` in this folder. Implementation and live proof are outstanding.

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

**One widget.** Append optional multiline STRING `direction_json`, default `""`, as Shot Lock's only widget. Canonical serializes `[""]`; every generated workflow regenerates. Old `[]` graphs already convert on all three converters. The writer gains nothing: Studio sets its existing `replay_from`.

**No pruning, no new output nodes.** Studio submits the whole shipped graph. Core runs output nodes first, so turning Shot Lock or the dispatcher into one would reorder ordinary runs; neither changes. A stage ends where its last node, after freezing, returns `ExecutionBlocker(None)` on every output.

| Stage | Run | Ends at |
|---|---|---|
| Stills, first | Fresh My Story run, as today | Dispatcher #91: freeze, block |
| Stills, again | Replay of the latest bundle, no LLM | Dispatcher #91: freeze, block |
| Video | Replay of the latest bundle | Batch #92: freeze, block |
| Publish | Replay of the latest Video bundle | Mux #85, unchanged; no freeze |

**Direction JSON**, versioned: `project_id`, `stage`, `out` (the bundle this run writes), `source` (previous bundle's manifest digest; absent on the first run), scene text, per-card picture text and motion text, take numbers per portrait, still and clip, and kept ids. No selection in V1: each stage acts on everything changed.

**The stage lives only on the wire.** Shot Lock sets it on the ledger it passes downstream, as a runtime key, on every run, and clears it when the widget is blank. The durable save and the freeze strip it. A blank-widget replay of a producer bundle therefore reaches the mux.

**Validation is in the Workflow Validator.** `VALIDATE_INPUTS` cannot refuse a prompt: core drops only the outputs that depend on the failing node and still runs the writer. Node 63's resolver checks shape and version, refuses a linked or non-literal direction, confines `project_id` and `out` to short safe names, refuses an `out` that exists, and matches `source` against the bundle in `replay_from`. A raise there halts the prompt before any asset loads. Execution is sequential, so a duplicate queued behind its twin fails at node 63 once the first has frozen `out`.

## Stages

**Stills.** On replay Shot Lock writes card motion text and clip takes onto the shot rows. On the first run the dispatcher stores the image prompt payload it received, before its own merge, in the ledger's images section. On replay the image prompt node rebuilds from that payload. A card edit replaces only that object's prompt text, with the scene text appended; identity, co-references and every other field are kept, and no template or LLM runs. The dispatcher, in producer mode, runs its cached path instead of verify-only and mints only misses, portraits first.

Take numbers above zero enter every seed branch: bookend, fixed mode, scene character and portrait. A portrait's take also enters its dependents' seeds. Take 0 is today's seed exactly. Kept objects keep their row whatever their key. Rows are one per object, newest wins, and the render driver's portrait indexes become newest-wins. Node 12 renames the workspace on every run, so the dispatcher's stills-folder re-resolution must follow a renamed replay workspace, not only a `pending_` one.

**Video.** Each shot's reuse record lives in the ledger's video section, with episode-relative paths. It holds the full clip manifest row, receipts, segments and foley stem. Its key is a hash of every segment's request with image and audio paths replaced by file hashes, plus recipe id, model file path, size and mtime, engine id and version, and clip take. The clip's own sha is verified on use. A hit rehydrates Batch's clips and receipts, so render engines, the manifest, credits and foley rebuild exactly as if rendered. A miss re-renders the whole shot. The clip take re-derives the seed. Reuse is decided before any engine or cloud call. `_RECEIPT_CAUSAL_KEYS` is untouched, so old bundles still verify.

**Publish.** Batch, in the publish stage, raises before any engine if a shot lacks a verified reusable clip. Otherwise it rehydrates every shot, and composite, captions, credits and mux run as today. No generation weights load.

**Freezer.** It moves to a helper in `nodes/` with relative imports. It raises `ReplayBundleError`, never `SystemExit`, which would kill ComfyUI's prompt worker, and takes an explicit destination. The CLI becomes a thin wrapper. It carries every member present: ledger, master, canon, stills, portraits, clips and `audio/foley`. Bundles live at `otr/episodes/_replay/p/<project>/<nn>-<stage>`, short for Windows path limits. After a new bundle verifies, it deletes this project's bundles older than the previous one and this project's earlier workspaces, found by a `producer_project_id` stamped durably on the first run. It never touches the running workspace.

**Readiness.** One pure resolver, called by node 63, `_otr_visual_assets` and the two cloud preflights. It finds Shot Lock through all links and rejects an ambiguous graph. A Stills replay needs image-engine assets, a Video replay clip assets, Publish none, and no replay needs writer assets.

**Known cost.** Node 12's CRT base video renders and renames on every stage, measured at about 53 frames a second, a few minutes for a typical episode. V1 accepts it.

## Studio

The approved design is `mockup.html`.

- **Entry.** A quiet second button, "Direct the shots first", sits beside "Make the episode" on the My Story path. The form stays first and folds to one line once a project exists; the board takes the run section.
- **Runs.** Every run submits the shipped graph through the existing busy guard. The first takes the user's My Story choices. Later runs set `replay_from` to the previous bundle, clear the My Story boxes and set Source off my_story, as the writer's own replay check requires.
- **Results.** A succeeded job ends a stage; only Publish picks the mux output. Studio's server reads `cards.json` and media from the bundle through ComfyUI's `/view`, copies them by sha into `out_dir/projects/<project_id>/`, and serves them itself, so the page never calls ComfyUI and pruned bundles never blank it. `cards.json` is a small versioned view the freeze writes: shots grouped by scene, each with its still or clip, takes and state.
- **Page.**
  - **Cast:** one portrait per character with Paint again; it repaints that character's shots unless kept.
  - **Scenes:** collapsed, opening when one holds a changed or failed card.
  - **Card row:** thumb, who, the spoken line, Picture and Motion text, Paint again, Film again, Keep.
  - **Buttons:** Paint the stills (N changed), Film the shots (N to film), Publish the episode, each with a one-line reason when disabled.
- **Progress.** In producer mode the dispatcher and Batch each drive one node-level bar over their items, and the page marks the k-th card of the work order.
- **Project file.** One atomic file holds the draft direction, the bundle names in order, and the pending `out` with its job id. After a restart, `client.jobs.get` re-attaches a running job; a succeeded job with `cards.json` finishes the stage; a failed or unknown job with no `out` offers the run again.
- **Languages.** New strings go into every language file (the parity test enforces it).
- **README.** The README gains a Producer section only after the proof passes. When it mentions the feature relative to contest judging is the operator's call.

## Not in V1

| Cut | What you give up |
|---|---|
| A separate Prepare stage | Cards before the first stills; an edited card costs one extra still |
| Studio pruning, output-node changes, canvas stage files, `VALIDATE_INPUTS` checks | Nothing |
| Second checkpoint format, writer widget, revision store, journals, receipt transport, separate prompt or direction files, per-clip sidecars | Nothing; the ledger carries them |
| Selection of individual cards | Each stage acts on everything changed |
| Template prompt rebuilds, per-field structured edits | One picture text and one motion text per card, plus scene text |
| Multiple takes per card, history beyond the previous bundle | Rollback to older takes |
| Banks other than My Story, auto-freezing ordinary episodes, reordering or splitting shots, bbox or location-image conditioning | Later; the backend is bank-agnostic |

## Build order

Each chunk: scoped suite green on the Windows venv, named files only, pushed to `main`, then Composer and Sonnet QA on the pushed diff. Every chunk touches shared code, so each commit states that blank runs are unchanged and how that was measured.

1. Freezer helper in `nodes/`; new members; `direction_json`; node 63 validation and resolver; canonical, generated workflows and parity tests.
2. Shot Lock runtime stage and overlay; stage blockers in the dispatcher and Batch; producer-only UI outputs and bars.
3. Prompt payload in the ledger; replay rebuild; dispatcher minting, takes, kept, newest-wins rows; renamed-workspace re-resolution; Stills freeze and `cards.json`.
4. Clip reuse records, takes, publish refusal; Video freeze; pruning.
5. Studio page, project file, restart, translations.
6. Live proof on the real My Story workflow, then record it.

## Implementation traps

- `ExecutionBlocker` is imported lazily from core, never redefined, and stubbed in tests. It is always given `None`; a message turns the skip into an execution error.
- UI values are lists, and clips go under `gifs`.
- Shot Lock's `IS_CHANGED(**kwargs)` stays as is; a test calls it with no arguments.
- `prompt_objects` is node 89's payload as the dispatcher received it; the dispatcher's own merge raises on a duplicate id.
- The freeze reads the post-rename workspace; node 12 renames before Shot Lock runs.

## Proof

- **Blank widget:** on the 5080, node order, VRAM peak and the published file match a baseline. Old `[]` vectors, the app form and mux discovery are unchanged. A blank-widget replay of a producer bundle publishes.
- **Zero supplied images on My Story:** Stills, Video and Publish produce a playable final in `otr/obs/`. Master, cast and timing hashes equal bundle 1's. Only the first run calls an LLM. With photos supplied, the reference twin works the same way.
- **One card edit:** exactly that still and that clip re-render; the reports show the rest reused.
- **Paint again:** gives a different image, and take 0 is byte-identical to today. A portrait's Paint again repaints only that character's unkept shots.
- **Film again:** reuses the still and re-renders the clip.
- **Video, then Stills again, then Video:** re-renders only the changed shots, with foley intact.
- **Duplicates and bad input:** a duplicate queued behind its twin, a tampered bundle, a linked direction, a missing still or changed audio is refused at node 63, before any model loads.
- **Publish:** with a shot missing, it refuses before any engine. Complete, it publishes and loads no generation weights.
- **Studio restart:** a running job re-attaches, a finished one completes, and a failed one can run again. Older bundles and workspaces are pruned.
- **Registry zip:** `planning/` stays out of it.
