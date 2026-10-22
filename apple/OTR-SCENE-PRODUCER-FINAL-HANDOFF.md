# OTR Scene Producer — implementation plan

Repository: `ComfyUI-OldTimeRadio` (this checkout). Design grounding baseline: `5f810ee9`, 2026-10-22; verify seams against current source before coding. Status: ready to implement; code and hardware acceptance remain outstanding. Paths are repository-relative unless stated otherwise; Windows runtime commands describe local implementation/live acceptance.

## 1. Scope and build order

Build an **optional** Scene Producer inside OTR and the existing Radio Play Studio with four repeatable ComfyUI job types: **prepare → stills → video → publish**. Existing automatic ComfyUI/Studio routes remain usable without Producer, scene checkpoints or manual scene work. Keep the frozen EpisodeAssembler master authoritative. Reuse existing nodes and group timed line-based shots into editable scenes, retaining bookend/music rows; add no checkpoint node class.

V1 edits shared setting/light, visible cast, props/wardrobe, framing, concrete action, one camera movement/end composition, references, seeds and approved takes. Use existing models/banks. Draft all cards/defaults automatically from the ledger and existing prompts; users may leave everything unchanged or edit one shot. Require no manual scene breakdown or new mandatory story fields. Offer batch approval without requiring individual card edits. New audio preparation starts a new direction set; preserve take history without automatically remapping approvals across checkpoints.

Build contracts/compiler and canonical integration first, then admission/resume, selected generation, Studio and live proof. Empty `production_request_json` preserves automatic mode. Producer jobs derive from the canonical graph plus the chosen effective profile; keep one graph authority.

Optional future enhancement: project existing ledger/progress from other routes into the same scene view as data becomes available. This is not an acceptance requirement. It must add no model call, blocking stage, required checkpoint or failure condition to existing runs; defer it if no safe read-only seam exists.

## 2. Stages and graph

| Stage | Terminal class / observed ID | Mode / selection | Work |
|---|---|---|---|
| prepare | `OTR_MetaBriefImagePromptGen` / 89 | `fresh_prepare` / `all_dirty` | Author audio, shot plan and real image prompt objects; render #12's CPU base video once; commit verified checkpoint. No generated stills or shot clips. |
| stills | `OTR_ImageGenDispatcher` / 91 | `resume_audio_checkpoint` / `all_dirty` or `explicit` | Compile this revision deterministically; generate requested images and reference dependencies; record completed/pending takes. |
| video | `OTR_VideoRenderBatch` / 92 | `resume_audio_checkpoint` / `all_dirty` or `explicit` | Verify selected shots' current stills; render selected whole shots and their ordered continuation segments. No image generation. |
| publish | `OTR_MasterAudioMux` / 85 | `resume_audio_checkpoint` / `verify_only` | Verify complete accepted takes, composite, caption, credit and mux to `otr/obs/`. No image or clip generation. |

Reject other combinations on both sides. Empty explicit selection requests no media/reference work. Unselected changed media remains pending/stale. Missing or stale required stills block Video before acquisition/spend; Publish applies this to its complete selection.

Studio loads **`workflows/otr_canonical.json`**, resolves classes using `node_id_for_type(litegraph, class_type)`, converts through the existing live-schema converter, then retains the selected terminal's transitive input-link closure. Resolve IDs before API conversion; retain validator/credential dependencies and valid links/slots.

Observed wiring: Freeze62 → ShotLock90 via252; Assembler7 → ShotLock `audio_done` via253; SignalLostVideo12 → ShotLock `gate_in` via284. ShotLock feeds MetaBrief89 via255 and Dispatcher91 via256; prompts reach Dispatcher via258. Dispatcher feeds Batch92 via260/267; Batch manifest feeds SilentComposite84 via261. Credits95 feeds mux85 via274. Derive closures from links rather than a static pruning list.

Make MetaBrief/Dispatcher terminal-capable. `wrap_stage_result` preserves result tuples/UI and adds sanitized `ui.otr_stage_receipt` only at the matching terminal; upstream terminals must not finalize/release another stage. Batch keeps its two STRING outputs. Partial Video returns its report and empty nonpublishable manifest; only Publish builds the complete manifest. Partial graphs contain no composite/mux. Automatic output discovery selects mux explicitly; `master_mux_gap_for_prompt` applies only when mux exists. Verify every generated variant.

## 3. Contracts and direction compiler

Use strict versioned types for revisions, direction, requests, checkpoints, work plans and takes. Store authoring in `ledger.video.scene_plan`; `Ledger.set_scenes` cannot retain arbitrary fields. Extend strict `ShotRow`/`VideoRequest` only where consumed.

An immutable revision identifies parent, checkpoint/digest, direction, selected takes and route/compiler fingerprints. Cards identify checkpoint, existing video shot and source line IDs; narrative scene/beat IDs remain distinct. Scenes own continuity; cards own framing, visible IDs, action and movement. Every exposed field has a compiler consumer.

`ProductionStageRequest` carries `project_id`, `revision`, `job_id`, `stage`, `mode`, checkpoint ID/digest on resume, inline canonical JSON `direction_payload` plus SHA-256, `selection_mode` and selected IDs in their proper domains. Studio persists a UUID `job_id` before submission; it is also the idempotency key/journal basename. Same ID/digest reconciles to the existing job; different bytes under that ID fail. Deliberate new attempts get new IDs.

Backend recomputes the payload hash. Enforce **1 MiB UTF-8 for the entire request** in Studio/backend; retain rejected drafts without truncation or another upload route. Payload contains direction, take IDs and saved seeds, never host paths/full ledgers. Reroll changes a seed in a new revision. Approval binds take ID, effective input hash and checkpoint.

Append optional STRING widget **`production_request_json`**, default `""`, `multiline=False`, after writer `bank_roll_pool`. Append its canonical saved value; update input-order, both converters/parity paths, API allowlists and variants together. Migrate only graphs missing this one trailing default; retain strict refusal of other positional drift. Older servers reject producer jobs. Ordinary app/My Story forms must not expose this internal field.

Compiler precedence: **location defaults → scene → shot → manual prompt body**. Preserve required engine formatting/style suffix and explicit visible/reference IDs. Manual overrides invoke no LLM; contradictions become editable diagnostics. Hash resolved inherited values, body, references and settings. Use framing-aware identity/detail and `flux3_klein_ref`'s three-reference cap; report unanchored visible subjects. V1 shares location text/state; bbox/3D and location-image conditioning require a future adapter contract.

Prepare performs bounded authoring. Every visual resume uses deterministic ShotLock overlay and PromptGen compilation from saved real prompts plus changed cards, with zero LLM calls. Legacy replay's empty prompt objects or nonempty-file image check cannot prove producer validity.

## 4. Checkpoint and audio resume

Backend `_otr_paths.otr_producer_root(project_id)` resolves `<ComfyUI output>/otr/episodes/<project_id>/producer/`, with UUID checkpoints at `checkpoints/<checkpoint_id>/manifest.json`. This fixed control root is never an execution ledger/rename target. Resolve validated IDs internally; accept no client directory. Require declared membership, full digests/sizes, normalized relative descriptors, no duplicates and approved-root/junction/symlink confinement. Verify before binding a usable workspace; invalid checkpoints fail explicitly.

`otr_producer_checkpoint_v1` binds source ledger, exact assembler master bytes/hash/flavour, sample metadata, lines/timing/cast/voices, freeze receipt, source IDs, initial plan/prompts/routes, cue manifest and base video/title-card plan. Manifest master digest must equal `ledger.audio.master_audio_sha256`. Verify ledger/master/cues and canonical timing/content hashes on every import; use source IDs/values rather than rebased paths. Freeze assembler audio, never the final foley mix.

Every job gets a fresh execution episode ID/workspace UUID. Prepare uses normal pending-to-title finalization. Before checkpoint commit, verify #12's returned base-video bytes/frame/duration coverage, title plan, resolved identity, ownership and durable paths; logged finalization failure cannot pass.

Resume import allocates its final execution ID up front and preserves source freeze identity. #12 verifies/materializes the saved video as `<execution_episode_id>.mp4`, returns its real path/title plan and updates durable paths **without rendering or renaming**. ShotLock/Dispatcher IDs must match. Rebase through existing path authority; master source/destination must differ because assembler replay copies unconditionally. Extend ShotLock's strict timing join with producer workspace UUID plus freeze identity. Pass real decoded AUDIO and exact duration downstream.

Keep `replay_descriptor` unchanged. Add validated `frozen_audio_descriptor(meta)` for legacy replay or producer checkpoints, reusing current audio pass-through bodies. Wire writer import **before My Story validation, rolls or model loads**; update Freeze, CastLock, shared voice/TTS, stable-audio/music-cue consumers, SceneSequencer and EpisodeAssembler. Producer visual compilation has its own context; do not spoof `meta.replay_from`.

Restore immutable audio/provenance plus the requested revision's verified `images`, `visual` and video/take state once into the active ledger. Generic `Ledger.save()` cannot recover discarded sections or replace stale nonempty metadata. Never read mutable latest-project state mid-job. Visual overlays cannot change frozen audio/cast or execution identity. Pin checkpoints, accepted/reference takes and their owning episodes against cleanup.

## 5. Admission and selective generation

At first gated `WorkflowValidator.validate()` execution, verify producer context before ordinary story admission. Build one pure **`StageWorkPlan`** from checkpoint/revision, effective policy and verified inventory. Admission/rendering share authoring, reference, image, shot/segment, reuse and publication requirements. Bind context to actual prompt ID plus project/revision/request digest; clear on success/failure and reconstruct after restart from verified records.

Restore frozen effective routes/language/style and director roll/skip receipts. Validator, lane rolls, visual assets, cloud-slug and cloud-balance preflights consume this plan. Node presence is not inference. Prepare needs authoring assets; resumes skip writer/TTS/music inference; cached Publish needs no generation weights/balance. Revalidate bytes at use. Do not classify producer resume as ordinary replay in `_otr_visual_assets`, which skips its entire asset plan.

Retain complete timelines; partial selection is not a gap and `meta.render_plan` does not select work. Compute **CPU-only effective fingerprints before engine preparation**: compiled prompts/inherited values; actual seeds; engine/recipe/compiler versions; pinned model digests; sampler/profile/canvas/FPS/frame coverage; consumed image/reference bytes; audio content, slice sample bounds and slicer version where consumed. Existing render receipt keys omit audio. Unrelated revision fields are provenance, not per-take invalidators.

Use a static whole-shot key for initial inputs/continuation policy; resolve segment keys when predecessor terminal frames exist. Derive segment seeds from saved base seed, stable index and versioned rule. Verify full chains, hashes/frame counts and matching foley/audio receipts. Approval never rerenders; reference changes invalidate consumers; movement-only edits preserve stills. Failed/partial/corrupt outputs never become hits.

Classify reuse before **both cloud fan-out and local scope entry**; acquire models, `assert_usable`, prepare scopes and spend only for misses. Cached Publish verifies recorded pinned model/recipe identity even without installed weights; changed requested version or changed chosen installed artifact is a miss. Publish returns incomplete without generating. Require complete current accepted takes, including deliberately accepted existing floor outcomes.

Add producer-aware `IS_CHANGED` to MetaBrief/Dispatcher and extend Batch with job/revision, checkpoint, contract version and consumed-inventory **byte-content digest**, not only manifest or size/mtime. Resolve confined validated IDs; force producer execution/revalidation if linked data/inventory is unavailable. Preserve automatic caching semantics.

## 6. Publication ledger

Checkpoint provenance owns source bank, title, language/style, writer identities, cast seed, delivered voice IDs/engines, music and frozen audio. Accepted manifests own visual/generation receipts. Rebuild from the **complete ordered accepted set**, including reused references/clips and explicit floors; exclude replaced/unaccepted takes. Keep original generation provenance separate from current publish telemetry.

| Owner at Publish | Required state |
|---|---|
| Dispatcher91 | Rebuild `images`, `visual`, `meta.image_engines`; `stamp_durable` and return matching `patched_ledger_json` to Batch. |
| Batch92 | Build complete manifest; replace `meta.render_engines`, revisions and ordered `meta.render_trace` through `stamp_durable`. No ledger output wire. |
| Credits95 | Read bound singleton; validate required receipts before compositing; retain existing checks. |
| Writer → mux85 | Carry delivery intent; visual/engine consumers read durable state. |

Propagate save failures. Resume preserves source bank, draft digest/publication requirement but allocates a fresh delivery token, bound identically to writer output and active ledger for `_assert_delivery_binding`. Preserve freeze/workspace proof; discard prior execution publication/cleanup status.

## 7. ComfyUI canvas and Studio interface

| Node | Inputs/widgets | Outputs and visible result |
|---|---|---|
| Writer1 | Append only `production_request_json`; empty preserves automatic mode. Studio fills it; ordinary forms hide it. | Existing sockets and names remain. |
| SignalLostVideo12 | Preserve inputs/widgets; producer context selects verified resume. | Preserve `video_path`, `title_card_plan_json` and standard video preview. |
| ShotLock90 | Preserve socket/widget order and strict `audio_done` join. | Existing outputs; deterministic revised shot plan. |
| MetaBrief89 | Preserve inputs/widgets; `OUTPUT_NODE=True`. | Preserve `image_prompts_json`, `report`; stage summary/receipt at Prepare only. |
| Dispatcher91 | Preserve inputs/widgets; `OUTPUT_NODE=True`. | Preserve `patched_ledger_json`, `image_done`, `report`; still previews/receipt at Stills only. |
| Batch92 | Preserve inputs/widgets and two outputs. | Preserve `render_report_json`, `clip_manifest_json`; per-take clip previews and partial/complete stage status. |
| Mux85 | Preserve sockets/widgets. | Preserve `final_video_path`, `report`; final video/Publish receipt. |

Do not duplicate stage/selection controls across nodes. Scene authoring/job selection live in Studio; the canvas keeps model/profile controls and wiring. Studio overrides the request in its API submission; the saved canonical request stays empty so normal Queue Prompt remains automatic. Use existing media previews plus `ui.otr_stage_receipt`; stage text is displayed in Studio/history without requiring a custom canvas extension. Upstream previews are not final deliverables.

Studio offers a **Scene Producer** workspace alongside the automatic form:

- **Prepare:** source/profile controls and Prepare action; after success show checkpoint identity, frozen-audio player and read-only cast/voice/timing. Changing story/audio requires a new Prepare.
- **Direction:** scene list and timed shot cards; shared setting/light/props/wardrobe, explicit visible/reference selection, framing/action, camera movement/end composition and optional manual prompt body. Show inherited versus overridden values, reference-cap diagnostics and compiled prompts tagged with their actual revision; edits mark any previously compiled preview stale.
- **Stills/Video:** multi-select, Generate selected or Generate all dirty, completed/pending/stale/failed states, dependency explanations, image/clip previews, seed reroll, take history and individual/batch approval. Saving direction/approval never starts inference. Video identifies missing still work instead of creating it silently.
- **Publish:** readiness/missing-take list, accepted selection and Publish action; show final video plus OBS publication receipt. A completed partial job remains distinct from a publish-ready project.

Controls reflect saved revision/checkpoint readiness and the shared busy state; backend admission remains authoritative. Show actual job/node/step progress without presenting a completed upstream stage as 100% of the episode. Provide owned cancellation, actionable errors, retained drafts and reconnect/resume status. Do not expose JSON, ledger paths or implementation details in ordinary user flow.

Studio stays a separate stdlib/SDK client; import neither `nodes` nor `folder_paths`. Store revisions under `settings.out_dir/projects/<project_id>/` and submit immutable requests through the SDK. ComfyUI owns verified state/artifacts. Add no unauthenticated OTR POST render/file-write route.

Return sanitized `ui.otr_stage_receipt` with schema, project/revision/job/checkpoint IDs, status, takes and standard `{filename, subfolder, type}` descriptors. Backend journals at `{filename: "<job_id>.json", subfolder: "otr/episodes/<project_id>/producer/jobs", type: "output"}`, independent of execution rename. Record actual prompt ID, running/failed/interrupted/completed status and final descriptors. Retrieve through `/view` after history/event loss; verify identities. Journal maps job to prompt; SDK job ID is separate. Missing journal means unknown submission, never automatic resubmission.

Persist intent before sending; reconcile uncertain jobs. Use `expected_revision` compare-and-swap; edits during a job create later revisions and results attach to their original revision. Back the Run store durably. All stages share automatic `_busy`; cancel only a proven owned prompt and retain finished verified takes. Producer jobs must not arm the legacy browser-idle `/free` watcher; prevent an existing watcher racing a newly reserved stage.

Keep validator `release_models_after_this_prompt()`. After persisted Prepare, its matching terminal frees authoring references with `free_otr_pipeline_residue`; do not repeat broad cleanup at other terminals. Private video scopes close in `finally`/`_end_engine_scope`; producer guards own failure/cancel journal/context/residue cleanup. Write media directly into canonical execution directories with local `.part` files and atomic verified-take commits.

## 8. Files and acceptance

| Area | Files |
|---|---|
| Contracts/compiler/coordinator | New `nodes/_otr_scene_plan.py`, `_otr_scene_compile.py`, `_otr_scene_producer.py`; keep pure compilation separate from persistence. |
| Paths/resume | `_otr_paths.py`, `production_ledger.py`, writer, Freeze/CastLock/shared voice/music, both classes in `scene_sequencer.py`, `video_engine.py`. |
| Visual/publish | ShotLock, MetaBrief, Dispatcher, Batch, `_otr_video_engines/render_driver.py`, `schemas.py`, credits and mux. |
| Admission | Validator, lane rolls, visual assets, cloud-slug/cloud-balance preflights. |
| Studio | New `radio_play_studio/project.py`; existing `graph.py`, `runner.py`, `server.py`, `form.py`, `static/`. |
| Wiring/ingress | `workflows/otr_canonical.json`, generated variants/manifests, `config/app_mode.json`, both converters, `_otr_workflow_apply.py`, `scripts/otr_api.py`. |

Every executable chunk wires real callers/canonical inputs, passes scoped and required full OTR/Bug Bible regressions, commits named files and pushes `main` under operator rules. Hardware proof follows suite-green pushes; required independent code QA still applies. Tags/releases require operator authorization.

Use Windows venv `C:/Users/jeffr/Documents/ComfyUI/.venv/Scripts/python.exe`, `PYTHONUTF8=1`, `pytest -q -p no:cacheprovider`. Run sibling Bug Bible from its root with relative `tests/bug_bible_regression.py`. Baseline existing failures.

Required proofs:

1. **Workflow/UI:** WorkflowValidator, JSON round-trip, live widgets/links, both converters, variants `--all`/`--check`, correct output in all variants. Update input-order/source-ref/language/selective-roll/app-mode ratchets; verify one trailing-default migration, unsupported servers, request hash/size rejection, near-limit SDK submission and the canvas/Studio states above. Prove existing automatic bank/profile paths complete without producer requests/checkpoints or manual scene work; an untouched generated direction set and a single-card edit both work in Producer.
2. **Resume:** wrong/tampered/escaped master, cue/timing corruption, unique workspaces, real AUDIO, matching #12/ShotLock/Dispatcher identity, exactly one base-video render, failed finalization preventing checkpoint commit, stable lookup after rename and zero authoring/TTS/music calls on visual resume.
3. **Cache:** empty explicit selection, missing inventory membership, stale still blocking Video before acquisition/spend, partial job never reaching composite, same-path/same-length changed bytes, model/audio/settings/reference changes, continuation invalidation, movement edit preserving stills and zero-generation cached Publish.
4. **Publication:** two partial still/clip batches → replace one take → restart → cached Publish. Assert unchanged audio/cast/voice/writer/music, current visual state, complete credits/trace without double counts, Dispatcher-owned wire/singleton equality, Batch-owned singleton/disk equality, loud save failure, real `build_credits_layout`, delivery binding and foley/frame alignment. Extend durable-stamp, canonical-replay, credits, Batch and recipe-stamp tests.
5. **Recovery/lifecycle:** SDK/prompt distinction, journal JSON via `/view` after history loss, duplicate/uncertain submissions, revision conflicts, edits during jobs, owned cancellation, cleanup, shared busy and legacy watcher races.

Live acceptance loads the actual canonical workflow after confirming no active prompt and resetting selectively. Prove two scenes/two characters, approved stills/full video, one still edit and a separate movement edit, expected misses, byte-identical frozen master/cast/timing, fresh-process resume, accurate credits and playable final in `otr/obs/`. Inspect later frames as well as starting stills.

Before coding read `AGENTS.md`, `CLAUDE.md`, `apple/GO_FORWARD_PLAN.md` and relevant production-log/Bug Bible entries. Inspect current branch/status, preserve unrelated edits and coordinate overlapping files. Start implementation from current main under operator rules.
