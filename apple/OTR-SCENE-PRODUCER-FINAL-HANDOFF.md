# OTR Scene Producer — implementation plan

Repository: this checkout. Grounded against the replay/freezer and visual pipeline at `7f1f4626`, 2026-10-22. Implementation and live proof remain outstanding.

## Product

Add an **optional** scene/card editor to Radio Play Studio. Existing ComfyUI, Studio, bank and profile flows keep working without Producer or manual scene breakdown. Generate cards/defaults from the ledger; users may edit one card or none. Keep frozen audio, cast, voices and timing while regenerating selected visuals.

**The system generates the stills.** Stills creates needed character portraits/reference images and scene stills through the existing ComfyUI image dispatcher. Bringing images is never required; My Story may retain its optional photo references. Procedural lanes keep their existing no-still behavior. Studio is the editor, not a second image-generation backend.

Use one shared backend and canonical graph with derived stage projections. Use the existing replay bundle/importer and still cache. Add no second checkpoint format, writer request widget, revision database or backend job queue. Auto-freezing ordinary episodes and incremental scene views for other routes are off by default and outside V1 acceptance.

## Four stages

| Stage | Terminal | Behavior |
|---|---|---|
| Prepare | `OTR_MetaBriefImagePromptGen` (#89) | Existing authoring/audio/ShotLock plus real prompt objects; freeze audio/plan/prompts into a replay bundle. No generated stills/clips yet. |
| Stills | `OTR_ImageGenDispatcher` (#91) | Replay audio; compile direction without an LLM; generate selected dirty images and their portrait/reference dependencies. Freeze the resulting bundle. |
| Video | `OTR_VideoRenderBatch` (#92) | Replay audio/stills; verify required images, render selected dirty shots, reuse valid clips. Freeze the resulting bundle, including partial results. |
| Publish | `OTR_MasterAudioMux` (#85) | Verify complete accepted media; composite, captions, credits and mux to `otr/obs/`. Generate no missing stills/clips. |

Studio resolves classes on its loaded canonical/profile LiteGraph before API conversion, then keeps the selected terminal's transitive input-link closure. Preserve validator/credential dependencies and all slots. Keep #12: it gates ShotLock90 through link284. MetaBrief/Dispatcher become `OUTPUT_NODE=True`; Batch/mux already are. Emit stage completion only at the matching terminal, not every upstream output node.

V1 keeps #12's existing CPU base-video/title rendering and rename on each resume. This costs CPU but needs no new pass-through branch. Use real decoded replay AUDIO; freeze from the actual post-rename workspace and verify paths/identity before reporting success. No automatic episode freezes or pauses are introduced.

## Direction and native canvas

Append one optional multiline STRING **`direction_json`**, default `""`, after ShotLock's existing inputs. Canonical #90 currently has no widgets; serialize `[""]`, preserve all existing links/output indices and regenerate variants. Blank means existing automatic/ordinary replay behavior. Writer retains its existing **`replay_from`** bundle pointer and gains no widget.

Versioned direction JSON contains producer/stage, bundle identity, scene/card overrides, selection (`all_dirty` or `explicit`), per-object/per-shot seed overrides and accepted takes. Prepare uses no replay pointer; subsequent stages require a verified bundle and matching source identity. Empty explicit selection means no work; Publish is verify-only. Reject unknown IDs, ambiguous ownership and malformed direction before inference.

Group lines by `line.beat_id → beats[].scene_id → scenes[]`, with existing shot/source-line links. Lines do not directly carry scene IDs. Missing scenes are valid: show ungrouped cards without adding a required ledger field or asking users to create scenes.

Compile location/scene defaults → card overrides → manual prompt body, retaining engine formatting/style and explicit visible/reference IDs. Expose setting/light, wardrobe/props, framing/action, one camera movement/end composition and seeds; each field must affect its consumer. Save real prompt objects; no LLM rewrite on visual resumes. Respect `flux3_klein_ref`'s three-reference limit and preserve portrait identity. No new bbox/3D or location-image conditioning promise.

Canvas users can load optional stage workflow JSONs generated from the same canonical/profile graph, set `replay_from`, paste direction and queue. Generate these projections with the existing workflow builder; never maintain independent hand-wired copies. Validate stage/graph compatibility. The full automatic canonical workflow still queues normally with blank direction; changing a JSON stage label alone does not prune a graph.

## Replay bundle extensions

Refactor the existing freezer into a reusable helper shared by its CLI and stage terminals. Keep `otr_replay_bundle_v1`, immutable export/atomic completion, manifest membership/digests/path confinement, assembler-master hash verification, fresh replay workspaces and existing audio pass-through nodes.

Zero-still bundles **already work** when no ledger image rows name missing assets. Do not weaken missing-file checks. Extend producer bundles with a versioned producer marker, hashed direction, saved image prompt objects and a durable episode-local clip/take index. Freeze the actual referenced clips or frame directories, segment receipts and consumed foley/audio sidecars. The shared `node_episode_manifest.json` is not durable per-episode state. Import/materialize/rebase through the existing importer, without a parallel audio-resume system.

Export Prepare/Stills/Video bundles under the existing replay root, outside deletable execution directories, before cleanup. Each is immutable and names the exact source state; never edit an old bundle or substitute the final foley mix for assembler audio. Keep the latest successful bundle if a later job fails. Producer bundles/accepted assets are retained explicitly; disk grows with stage snapshots. Publish need not duplicate the video bundle.

## Five visual changes and early admission

1. **ShotLock:** overlay direction on the imported frozen plan; preserve source IDs, frame coverage and strict audio/workspace join.
2. **MetaBrief:** on producer resume return saved real prompt objects plus deterministic changed-card compilation. Preserve ordinary replay behavior.
3. **Dispatcher:** bypass verify-only replay only for validated Stills work. Reuse its incoming cache/index, generate misses plus reference dependencies, retain unselected pending state. Video/Publish verify their required image set without minting.
4. **Render driver/Batch:** add verified per-shot clip reuse and seed overrides. Wire overrides to the actual request builder, which currently derives seeds from hashes instead of reading ShotLock's zero seed. Distinguish absent override from deliberately chosen zero; derive continuation segment seeds deterministically. Wire still seed overrides to its actual resolver too.
5. **Readiness:** one small pure classifier reads the literal ShotLock direction and verified replay manifest from the queued prompt **before** validator rolls/assets/cloud gates. Associate the writer/ShotLock through all dependency links; the gate-only walker does not reach ShotLock. Reject unresolved linked direction/ambiguous graphs. Restore frozen effective routes/language/receipts, and classify actual image/video misses: Stills needs image generation assets, Video needs clip assets, cached Publish needs neither. Node presence alone cannot decide this.

Existing audio replay branches stay in use. Producer My Story resumes must not be rejected by fresh-authoring selection checks: narrowly admit verified producer replay through existing validator/writer admission, while retaining ordinary My Story/replay conflict checks. Clear/ignore live authoring choices only in this validated mode; no new writer field or changes to every audio node.

## Reuse and publication truth

Strengthen the **existing** still cache where necessary for edited prompts/seeds/settings; do not build another cache. Verify current bytes, required membership and reference identity, not just file existence. Clip keys cover resolved prompts, actual seeds, engine/recipe/model identity, sampler/canvas/FPS/frame coverage, still/reference hashes and consumed audio/slice identity. Compute keys before local/cloud engine preparation. Resolve chained segment keys after predecessor frames exist. Unselected dirty shots remain pending; movement-only edits preserve stills; approval alone never renders.

Batch keeps its two outputs, `render_report_json` and `clip_manifest_json`. Producer manifests carry an explicit complete flag; the compositor refuses an incomplete producer manifest. Partial stages never reach publish. Publish fails before generation spend if required media is missing/stale; deliberately accepted existing floor outcomes remain explicit. Preserve automatic manifest compatibility.

Replay clears render telemetry. Rebuild aggregates from the **complete accepted set**, including reused takes: Dispatcher stamps `images`, `visual`, `meta.image_engines` durably and in its returned ledger; Batch replaces `meta.render_engines`/ordered trace from accepted clip receipts. Keep frozen cast/voice/music/writer provenance, delivery binding and loud save failures. Credits reads the singleton; passing a wire-only or latest-partial-job summary is insufficient.

Update visual-node `IS_CHANGED` for direction and consumed inventory bytes; force execution if that inventory cannot be verified. Keep unchanged automatic seeds/cache behavior. Acquire models/paid calls only for validated misses; cached Publish can verify pinned recipe/model identities without installed generation weights.

## One-page Studio

Add an optional Producer page: generated scene/card grid, frozen audio player, source/profile Prepare controls, separate Stills/Video buttons, selected/all-dirty actions, previews, seed reroll, individual/batch approval and Publish readiness. Show inherited/edited values and pending/stale/failed state. No required manual card edits or image uploads; keep the existing automatic form.

Use one atomically saved project file under Studio's output folder: direction, latest bundle/receipt, selected takes and one pending submission. Persist an immutable submitted direction snapshot and SDK job ID when known. Disable editing/submission while busy. On restart/uncertain submission, reconnect to the known job or show status unknown; never silently resubmit paid work. No ancestry/CAS/job-journal service is needed.

Return a small stage receipt with bundle manifest descriptor, direction/cards, take status and safe media descriptors. Derive `replay_from` as its bundle directory relative to ComfyUI output, not a Windows host path. **Prove JSON receipt transport first:** existing proxy output discovery recognizes images/gifs/audio, so an arbitrary UI JSON key is insufficient. Prefer supported SDK metadata/downloads; permit a small confined read-only history/`/view` adapter if needed. No new render/file-write HTTP route; all inference remains in ComfyUI's executor.

Share the existing busy guard and private-engine `finally` cleanup. Add cancellation only for a confirmed owned job; otherwise retain busy/unknown status. Producer jobs must not arm the legacy browser-idle `/free` watcher. Free authoring residue after persisted Prepare, not midway through downstream work. Media goes directly into canonical episode storage; bundle export precedes cleanup.

## Build and acceptance

Build reusable freezer/producer helper and receipt transport → direction/early classification → still generation and clip reuse → one-page Studio/canvas projections. Touch the existing freezer/importer, visual nodes/render driver, readiness helpers, narrowly scoped story admission, workflow builder/converters and Studio. One small producer helper can own direction validation/compilation and stage receipts; avoid a new orchestration framework.

Prove:

- Legacy automatic banks/profiles and ordinary replay still work; old empty ShotLock widget vectors get the optional default, and unsupported servers fail clearly. All three converters already tolerate the old empty vector: test parity rather than relaxing unrelated drift. Validate canonical links/widgets, generated projections/variants and automatic mux output discovery.
- **Zero supplied images:** Prepare → generate portraits/scene stills → batch approve → Video → Publish. No manual scene breakdown; also prove untouched cards, one-card edits and optional My Story photos.
- Tampered/escaped bundles, missing referenced stills, wrong assembler master, unique replay workspaces, real AUDIO/duration and post-rename freeze correctness. No authoring/TTS/music inference on resume.
- Same-path changed bytes, reference/seed/model/audio/settings edits, selected/empty work, continuation invalidation, partial manifest refusal and zero generation calls/loads on cached Publish.
- Partial still/video batches → replace a take → restart → publish: unchanged audio/cast/timing, complete durable credits/trace and foley/frame alignment, loud save failure, unknown-job handling and watcher/cancel ownership.

Run scoped and required OTR/Bug Bible suites on the Windows venv, canonical validator/JSON/widget/link audits and variants checks. Commit named files and push suite-green chunks to `main` under operator rules. Live proof loads the canonical workflow/derived closures, resets only with no active prompt, and verifies a playable final in `otr/obs/`. Preserve unrelated edits. Tags/releases remain operator decisions.
