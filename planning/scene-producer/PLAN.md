# OTR Scene Producer - go-forward plan

Baseline `main` f3b1921e, ComfyUI core 0.42.1, 2026-10-22. Every seam below was read in the real files at that commit; re-read them at the commit you start from. Implementation and live proof are outstanding.

## What it is

An optional card editor for Radio Play Studio. The first producer run writes, voices and paints the episode and stops before video. The user then sees one card per shot, grouped by scene, each with its still. They can re-roll or edit one card, a few, or none, then generate video, then publish. Only what changed is re-rendered. Audio, cast, voices and timing never move on a visual edit. A partial result can never publish as a finished episode. ComfyUI runs everything; Studio only edits and submits. With the one new widget blank, every workflow, bank and Studio form behaves exactly as today, in the same node order.

## What we reuse

| Already built | Where | What it gives Producer |
|---|---|---|
| Replay bundle, freezer, importer | `scripts/otr_freeze_replay_bundle.py`; `production_ledger.py` `import_replay_bundle`, `_confine_replay` | The checkpoint: immutable, hashed, master verified, fresh workspace per run. A relative `replay_from` is anchored at ComfyUI's output folder |
| Audio pass-throughs on replay | writer, Freeze, CastLock, TTS, music, SceneSequencer; EpisodeAssembler re-hashes the master | No LLM, TTS or music on any later stage, with no changes to those nodes |
| Replay branches in the queue-time gates | `_otr_visual_assets.plan_prompt`, `_otr_lane_rolls`, `cloud_slug_preflight`, `cloud_balance_preflight` | Where one shared resolver plugs in |
| Shot Lock replay branch and `VALIDATE_INPUTS` | `otr_shot_lock.py` ~3577 reuses frozen `video.shots`; `VALIDATE_INPUTS` returns True today | The overlay seam and a queue-time check |
| Shot rows | role, char_id, engine_id, start_s, dur_s, `creative`, `source_line_ids`; one row per line | Card content. Scenes come from beat and shot `scene_id` via `source_line_ids`; ungrouped is valid |
| Still cache | dispatcher `request_cache_key`: role, object, prompt hash, seed, engine, version, size, anchor | Unchanged cards and portraits reuse; a re-rolled portrait changes the anchor of its dependent stills |
| Request builder | `render_driver.build_request_from_shot` | A clip reuse key computed before any engine loads |
| ComfyUI `ExecutionBlocker(None)` | core `comfy_execution/graph_utils.py` ~140, `execution.py` ~265 | Stops every node downstream of a stage's last node, silently. Core's docstring names this use |
| ComfyUI `/view` | core `server.py` ~596: any file under output, subfolder confined | Studio reads cards, stills and clips from the bundle with no new route |
| Studio | SDK submission, busy guard, SSE progress, direct calls to ComfyUI for `/object_info`, `/upload/image`, `/free` | No job framework |

## Graph contract

**One widget.** Append optional multiline STRING `direction_json`, default `""`, as Shot Lock's only widget. Canonical serializes `[""]`; variants regenerate. Old `[]` graphs already convert on all three converters. The writer gains nothing: Studio sets its existing `replay_from`.

**No pruning, no new output nodes.** Studio submits the whole shipped graph it already loads, with two values set. Core runs output nodes before other ready nodes, so turning Shot Lock or the dispatcher into an output node would reorder every ordinary run; neither changes. A stage ends where its last node, after freezing, returns `ExecutionBlocker(None)` on every output. Canvas users get the same staging by setting the same two values and queueing normally.

| Stage | Run | Ends at | Then |
|---|---|---|---|
| Stills, first | Fresh, no `replay_from`. Writer, voices, music, Shot Lock, image prompt LLM, all stills, exactly as today | Dispatcher #91 | Freeze bundle 1, block |
| Stills, again | Replay of the latest bundle; no LLM | Dispatcher #91 | Freeze, block |
| Video | Replay of the latest bundle | Batch #92 | Freeze, block |
| Publish | Replay of the latest Video bundle | Mux #85, unchanged | Published to `otr/obs/` |

The first run is the only one with an LLM, and it is a normal fresh run, so every existing writer preflight covers it. Image and video previews return as standard UI outputs in producer mode only; blank runs return what they return today.

**Direction JSON**, versioned: `project_id`, `stage`, `out` (the bundle name this run will write), `source` (the previous bundle's manifest digest, absent on the first run), per-scene text, per-card still text and motion text, per-object take numbers for portraits, stills and clips, `selection` (`all_dirty` or ids), accepted ids.

**Validation.** `VALIDATE_INPUTS` checks shape, version, name confinement, and that `out` does not already exist, at `/prompt` time, before anything runs. That last check is what makes a duplicate submission impossible. Shot Lock repeats the checks at run time, because a canvas user may wire the widget as an input, and compares `source` with the imported bundle.

## Stages

**Stills.** Shot Lock stamps the direction into the ledger meta that flows downstream and, on replay, overlays card motion onto the frozen plan. On replay the image prompt node rebuilds every object from the bundle's saved prompt objects. A card edit changes only that object's prompt text, appending the scene text, and recomputes its hash; every other field, including identity and co-references, is kept. No template rebuild. The dispatcher runs its cached path instead of verify-only and mints only misses, portraits first. A take number above zero enters the seed derivation and the cache key; take 0 is today's seed exactly. The freeze adds stills, portraits, `prompt_objects.json`, `direction.json` and `cards.json`: shot rows grouped by scene, each with its still, takes and status.

**Video.** Each shot's clip is reused when a sidecar key next to the clip still matches: a hash of the `build_request_from_shot` output, engine id and version, still sha, audio slice sha and clip take, plus the clip's own sha on disk. Otherwise the whole shot re-renders. A clip take above zero re-derives the seed from the request hash with the take mixed in. `_RECEIPT_CAUSAL_KEYS` stays untouched, so replay verification of old bundles keeps working. Reuse is decided before any engine or cloud call. In producer mode the manifest carries `complete`; the compositor refuses `complete: false`, and blank runs never set the key. The freeze adds clips and sidecars and rewrites `cards.json`.

**Publish.** Replay of the latest Video bundle, no block. The compositor's refusal guards partial work. With every clip reused, no generation weights load. Credits read the singleton the importer rebuilt; the mux publishes as today.

**Readiness.** One pure resolver reads the direction and the bundle manifest from the queued prompt. It finds Shot Lock through all links, since the gate-in walker never reaches it, and rejects an ambiguous graph. A Stills replay needs image-engine assets, a Video replay needs clip assets, Publish needs neither, and no replay needs writer assets. The four existing replay branches call it.

**Disk.** The freeze helper, after its new bundle verifies, deletes this project's bundles older than the previous one and its older replay workspaces. Nothing outside `_replay/producer/<project_id>/` and this project's workspaces is touched.

## Studio

One page beside the automatic form. Each run submits the shipped graph with `replay_from` set to `otr/episodes/_replay/producer/<project_id>/<previous>` and Shot Lock's direction filled. When the job ends, Studio reads `<out>/cards.json`, stills and clips through ComfyUI's `/view`, using the same relative subfolder. Expect up to about a hundred cards on a long episode, so scenes collapse.

The page has three parts:

- **Cast:** each portrait with a Try again button.
- **Cards:** grouped by scene. Each shows who, where and action, its still or clip, an editable still text and motion text, and Try again for the still and for the clip.
- **Buttons:** Generate stills, Generate video, Publish. Each acts on the selection or on everything dirty.

One atomically saved project file under `out_dir/projects/<project_id>.json` holds the draft direction, the bundle names in order, and the pending `out`. After a restart, if `<out>/cards.json` exists the run finished; otherwise the page says it has not finished and offers no resubmit. The busy guard is shared.

## Not in V1

| Cut | What you give up |
|---|---|
| A separate Prepare stage before stills | Seeing cards before the first stills exist; an edited card costs one extra still |
| Studio graph pruning, output-node changes, canvas stage workflow files | Nothing |
| Second checkpoint format, writer widget, revision store, compare-and-swap, journals, receipt transport, job API, bundle pointer inside the direction | Nothing |
| Template prompt rebuilds, structured per-field edits | One still text and one motion text per card, plus scene text |
| Multiple takes per card, history beyond the previous bundle | Rollback to older takes |
| Cancellation, idle-watcher changes | The watcher already waits for idle |
| Auto-freezing ordinary episodes, scene view for other routes, reorder or split shots, bbox or location-image conditioning | Later |

## Build order

Each chunk: scoped suite green on the Windows venv, named files only, pushed to `main`, then Composer and Sonnet QA on the pushed diff. Every chunk touches shared code, so each commit message states that blank runs are unchanged and how that was measured.

1. Freezer helper with an explicit output path, shared by the CLI and the stage nodes. New bundle members. `direction_json` and `VALIDATE_INPUTS` on Shot Lock. Canonical, variants and parity tests.
2. Shot Lock meta stamp and overlay. The stage blocker in the dispatcher and Batch. Producer-only UI outputs. The resolver in the four branches.
3. Image prompt replay rebuild, dispatcher minting and takes, Stills freeze and `cards.json`.
4. Clip sidecar reuse and takes, the `complete` flag and compositor refusal, Video freeze, disk cleanup.
5. Studio page and project file.
6. Live proof on the real workflow, then record it.

## Proof

- Blank widget: on the 5080, node execution order, VRAM peak and published output match a baseline run. Old `[]` vectors, the app form and mux discovery are unchanged.
- Zero supplied images: Stills, Video, Publish give a playable final in `otr/obs/`. Master, cast and timing hashes equal bundle 1's. Only the first run calls an LLM.
- One card edit: exactly that still and that clip re-render; the reports show the rest reused.
- Still Try again gives a different image; take 0 is byte-identical to today. A portrait Try again re-renders only that character's stills.
- Clip Try again: still reused, clip re-rendered.
- A duplicate submission, a tampered bundle, a missing still or changed audio is refused before any model loads.
- Partial Video then Publish is refused. Complete Video then Publish succeeds, and cached Publish loads no generation weights.
- A restart mid-job shows not finished, with no resubmission. Older bundles and workspaces are pruned and the latest two remain.
