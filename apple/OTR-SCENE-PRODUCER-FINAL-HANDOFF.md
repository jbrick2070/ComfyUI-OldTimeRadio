# OTR Scene Producer - go-forward plan

Baseline `main` 7ca363e9, 2026-10-22. Every seam below was read in the real files at that commit; re-read them at the commit you start from. Implementation and live proof are outstanding.

## What it is

An optional card editor for Radio Play Studio. After the episode is written and voiced, the user sees one card per shot, grouped by scene, with the audio already frozen. They edit one card, a few, or none, then generate stills, approve them, generate video, and publish. Only what changed is re-rendered. Audio, cast, voices and timing never move on a visual edit. A partial result can never publish as a finished episode. ComfyUI runs everything; Studio only edits and submits. With the one new widget left blank, every existing workflow, bank and Studio form behaves exactly as today.

## What we reuse

| Already built | Where | What it gives Producer |
|---|---|---|
| Replay bundle, freezer, importer | `scripts/otr_freeze_replay_bundle.py`, `production_ledger.py` (`import_replay_bundle`, `replay_descriptor`) | The checkpoint: immutable, hashed, master-audio verified, fresh workspace per run, zero-still bundles already valid |
| Audio pass-throughs on replay | writer, Freeze, CastLock, TTS, music, SceneSequencer; EpisodeAssembler copies and re-hashes the master | No LLM, TTS or music on any resume, with zero changes to those nodes |
| Replay branches in the queue-time gates | `_otr_visual_assets.plan_prompt`, `_otr_lane_rolls`, `cloud_slug_preflight`, `cloud_balance_preflight` | The place one shared resolver plugs in |
| Shot Lock replay branch | `otr_shot_lock.py` ~3577: reuses the frozen `video.shots`, no LLM | The overlay seam |
| Shot rows | `ShotRow`: role, char_id, engine_id, start_s, dur_s, `creative` (expression, motion, camera) | Everything a card shows, with no new ledger field |
| Scene ids | Writers stamp `scene_id` on beat and shot rows; lines carry beat and shot ids | Scene grouping by lookup; ungrouped is valid |
| Still cache | `otr_image_gen_dispatcher.py` ~2076: key = role, object, prompt hash, seed, engine, version, size, anchor; seeded from the incoming ledger | Unchanged cards reuse their stills for free |
| Template prompt builder | `compose_image_prompt_fallback` | Deterministic recompile of a changed card, no LLM |
| Clip receipt per segment | `render_driver.py` `_RECEIPT_CAUSAL_KEYS`: prompt, seed, engine, recipe, frames, canvas, still hash, sampler, model artifacts | Most of the clip reuse key |
| Studio | SDK submission, busy guard, SSE progress, output pick by node type, idle watcher that only fires when nothing runs | No job framework needed |

## Graph contract

**One widget.** Append optional multiline STRING `direction_json`, default `""`, as Shot Lock's only widget. Canonical serializes `[""]`; variants regenerate. Old graphs with `[]` already convert on all three converters (each has an empty-vector branch), so no migration code. The writer gains nothing: its existing `replay_from` is the checkpoint pointer.

**Four terminals.** Studio prunes the shipped workflow it already loads to the chosen terminal's transitive input-link closure (validator and credential nodes come along through the writer's links).

| Stage | Terminal | Change to the node |
|---|---|---|
| Prepare | `OTR_ShotLock` #90 | `OUTPUT_NODE=True`; in producer mode, freeze the bundle at the end |
| Stills | `OTR_ImageGenDispatcher` #91 | `OUTPUT_NODE=True`; return stills as standard image outputs; freeze |
| Video | `OTR_VideoRenderBatch` #92 | Return clips as video outputs; manifest carries `complete`; freeze |
| Publish | `OTR_MasterAudioMux` #85 | Unchanged |

Normal Queue Prompt: blank widget, whole graph, identical to today. Producer: Studio fills the widget and prunes. Each stage freezes into `<episodes>/_replay/producer/<project_id>/<stage>-<n>/`, so the next stage's `replay_from` is known by name and no receipt has to travel back.

**Direction JSON** (versioned): `project_id`, `stage`, `bundle` (relative path, required after Prepare), per-scene text, per-card text, per-card seed, `selection` (`all_dirty` or explicit ids), accepted ids. Shot Lock validates it before anything runs: unknown ids, a bundle whose identity does not match, or malformed JSON fail loud before any model loads.

## Stages

**Prepare.** The audio episode exactly as today, ending at Shot Lock. No image prompts, no stills. Bundle = ledger, master WAV, canon. The base CRT video re-renders on CPU on every later stage; V1 accepts that cost.

**Stills.** Replay from the latest bundle. Shot Lock overlays the direction onto the frozen plan. The image prompt node runs its LLM once, on the first Stills run, as it does today; the prompt objects it produced are saved into the Stills bundle as one file. Every later run rebuilds unchanged objects from that file and template-compiles changed cards from their row plus card text, with no LLM. The dispatcher runs its normal cached path instead of verify-only: portraits and references first, then scene stills, minting only misses. Bundle adds stills and the prompt objects.

**Video.** Replay from the latest bundle. A shot's clip is reused only when its saved receipt equals the rebuilt request, the clip bytes verify (sha256, size, frame count), and the audio slice identity matches; the current receipt covers neither clip bytes nor audio, so both are added to the key. Whole shots only: a changed shot re-renders all its segments. Seed override wires the existing `request_seed` row field, today stamped zero and never read, into the request builder; absent means derive from hash as today. Reuse is classified before any engine or cloud call. Bundle adds the referenced clips and receipts.

**Publish.** Replay from the latest Video bundle, closure to the mux. The compositor refuses a manifest with `complete=false`. With everything cached, only composite, captions, credits and mux run; no generation weights load. Credits read the singleton the importer rebuilt. The mux publishes to `otr/obs/` as today.

**Readiness.** One pure resolver reads the direction and the bundle manifest from the queued prompt and returns needs-stills, needs-video, or errors. The four existing replay branches call it. The gate-in walker never reaches Shot Lock, so the resolver associates writer and Shot Lock through all links and rejects an ambiguous graph. Stills needs image-engine assets, Video needs clip assets, Publish needs neither.

## Studio

One page beside the automatic form. Scene list, then cards showing who, where, action and camera from the shot rows, with an editable text per scene, an editable text and seed per card, a frozen-audio player, and four buttons: Prepare, Generate stills, Generate video, Publish, each on the selection or on everything dirty. Previews come from the stills and clips the SDK already discovers as outputs. One atomically saved project file under `out_dir/projects/<project_id>.json` holds the direction, the latest bundle per stage, accepted asset ids, and the pending submission (direction snapshot plus SDK job id). The busy guard is shared. After a restart an uncertain job shows unknown and is never resubmitted.

## Not in V1

| Cut | What you give up |
|---|---|
| Second checkpoint format, writer widget, revision store, compare-and-swap, journal service, stage receipt JSON | Nothing |
| Image prompt node as the Prepare terminal | Seeing the compiled prompt before Stills |
| Canvas stage workflow files | Canvas users queue the full graph with direction set; caches make unchanged shots free |
| Structured per-field edits (setting and light, wardrobe, framing, camera end) | One text per scene and per card instead |
| Multiple takes per card | One accepted plus one pending |
| Keeping every stage snapshot | Prepare bundle plus the latest per stage; no rollback to older takes |
| Cancellation, restart reconnect, idle-watcher changes | Unknown state after a restart; the watcher already waits for idle |
| Auto-freezing ordinary episodes, scene view for other routes, reorder or split shots, bbox or location-image conditioning | Later |

## Build order

Each chunk: scoped suite green on the Windows venv, named files only, pushed to `main`, then Composer and Sonnet QA on the pushed diff.

1. Freezer helper shared by the CLI and terminals; bundle gains clips, prompt-objects file and direction hash as verified members; `direction_json` on Shot Lock; canonical, variants, parity tests.
2. Shot Lock: validation, overlay, terminal freeze; the readiness resolver in the four branches.
3. Image prompt node producer branch; dispatcher producer branch and image outputs; Stills freeze.
4. Clip reuse, seed wiring, `complete` flag, compositor refusal; Video freeze.
5. Studio page and project file.
6. Live proof on the real workflow, then record it.

## Proof

- Blank widget: automatic workflows, old `[]` vectors, the app form and mux discovery unchanged.
- Zero supplied images: Prepare, Stills, approve, Video, Publish; a playable final in `otr/obs/`; master, cast and timing hashes equal the Prepare bundle's.
- One card edit: exactly that still and that clip re-render; the dispatcher report and render trace show the rest reused.
- Seed only: still reused, clip re-rendered. Text only on a scene: every card in it re-renders.
- Tampered bundle, missing still, changed audio: refused before any model loads.
- Partial Video then Publish: refused. Complete: published. Cached Publish loads no generation weights.
- Restart mid-job: unknown, no resubmission.
