# Lean/Mean Cleanup - Current Coding Plan

**Status:** current plan authority. **Cleanup HAS landed from this document** -- orders 1-6 complete (`3f727a43`), and individual rows carry their own resolved status (e.g. `nsfw_frame_qc.py`, deleted 2026-08-28). The header said "planning only, no cleanup code has landed" until 2026-08-28, which told every auditor who read this file first exactly the wrong thing. Trust the per-row status, not a global claim. Original wording preserved in git.
<!-- superseded: planning only, no cleanup code has landed from
this document. -->

**Grounding baseline:** committed `v2.0-alpha` HEAD
`ed3ae2c708b32e988dc15d47666c891ec68ad74c` on 2026-08-22. The worktree also
contained unrelated operator work, so every execution chunk must re-check its
targets against the then-current committed HEAD.

`ROADMAP.md` decides when this campaign starts. `docs/GO_FORWARD_PLAN.md` owns
the work before it and only points here. This file is the sole current authority
for cleanup scope, order, blast radius, and verification. It intentionally has
no running changelog; git history and the dated evidence documents are the
archive.

Supporting evidence:

- `docs/2026-07-25-dormant-3d-rip-brief.md`
- `docs/2026-07-25-dormant-3d-rip-judgment.md`
- `docs/ENGINE_MATRIX.md`
- `docs/PRODUCTION_SPRINT_LESSONS.md`

## 1. Goal and decision rule

Make the repository look deliberately designed: remove code with no useful
current consumer, consolidate exact duplicate boundaries, preserve useful
manual/public surfaces until their compatibility cost is explicit, and leave
the canonical production workflow simpler rather than merely smaller.

Line count is not evidence. A target is removable only after all five checks:

1. It is absent from `workflows/otr_canonical.json`, including dynamic engine
   selection through profiles, registries, dispatchers, and route freeze.
2. It has no production importer or caller.
3. Any registered public node, saved-workflow/API contract, supported manual
   tool, or required startup side effect has been migrated or explicitly
   retired; none is silently treated as unreachable.
4. Tests are classified as contract coverage, relocation candidates, or tests
   of dead code; a green private test is not production reachability.
5. The matrix states what disappears and the migration, tombstone, or explicit
   acceptance needed before deletion.

Use these verdicts:

- **REMOVE-SAFE:** no current production or public consumer; delete subject and
  dead private tests together.
- **REMOVE-AFTER-MIGRATION:** the implementation is not canonical, but a live
  guard, startup side effect, public node ID, saved artifact, or useful manual
  surface must move or be retired first.
- **CONSOLIDATE:** keep caller-specific behavior while centralizing only the
  exact common mechanism.
- **KEEP:** current value is verified or the old removal premise is false.
- **RE-GROUND:** no deletion until the named uncertainty is resolved at current
  HEAD.

## 2. Current removal and loss matrix

### 2.1 Small, zero-reachability candidates

| Target | Verdict | What it does now | What is lost by removal | Required handling |
|---|---|---|---|---|
| All `_truncate_at_sentence_boundary` and `_tail_at_sentence_boundary` definitions in `nodes/story_orchestrator.py` | REMOVE-SAFE | Two earlier definitions are rebound, and the two later private definitions also have zero callers. | No current runtime behavior; no supported external owner was found. | Add the AST duplicate-top-level-definition guard first, then delete all four definitions. |
| `nodes/_otr_shared/sidecar.py` | REMOVE-SAFE | Unadopted generic sidecar helper. | Only unused future scaffolding. | Update the source-scan assertion in `tests/test_video_dep_pilot.py`; do not touch the live TTS helper in `nodes/_otr_audio_engines/_otr_sidecar.py`. |
| `nodes/_otr_probe.py` | REMOVE-SAFE | Unconsumed probe abstraction. | An unused helper surface. | Prove zero consumers again, then delete the module; it has no private test coverage to unwind. |
| `nodes/otr_shot_duration_calculator.py` | REMOVE-SAFE | Contains an already-unregistered duration stub superseded by `OTR_ShotLock`. | Direct imports of the retired helper implementation. | Preserve the two deleted node IDs in workflow validation; replace broad implementation tests with the existing retirement guard. |
| Legacy cast prototypes | REMOVE-SAFE | `_build_cast_rows` in `OTR_LedgerScriptWriter.py` and the test-only `_otr_cast_contract.py` / `_otr_cast_repair.py` prototype stack have no production caller and are not in the live CastLock path. | Manual/test-only `CastContract`, lock/load, alias detection, director-plan builder, and repair experiments. | Keep live `_otr_casting.lock_cast`; delete the writer orphan, both prototype modules, `test_cast_contract.py`, `test_cast_contract_helpers.py`, and `test_cast_repair.py` together after a final import scan. Rehome any assertion R4 judges to protect the live CastLock contract before deleting its old host. |
| `scripts/normalize_workflow_widgets.py` | REMOVE-AFTER-MIGRATION | Self-described over-aggressive normalizer; no caller. Its dry-run mode can still report drift. | That standalone dry-run report. | Confirm `OTR_WorkflowValidator` plus the link/widget audit cover the useful report, or have R4 explicitly accept the report loss, then delete; never retain an unsafe apply path merely for history. |
| ~~`nodes/_otr_shared/nsfw_frame_qc.py`~~ | **RESOLVED -- DELETED 2026-08-28** | Default-off, test-only offline frame sampler; zero production consumers confirmed by audit. | The safety-tool loss WAS accepted, explicitly, by the operator this session and with his reasoning stated: visual-content style is now owned by `_otr_banana_route`, and the 2026-08-03 no-content-guardrails directive stands. That is the R4-equivalent acceptance this row required. | DONE. Module + `test_video_nsfw_frame_qc.py` + `test_video_nsfw_frame_qc_additive.py` removed in one commit, and `test_video_survival_guide_vectors.py` de-referenced ATOMICALLY in the same change -- its line-27 import would otherwise have failed collection of the whole file and silently taken the ghost-node, VRAM-leak, widget-serialization and pipe-deadlock vectors with it. `retry_taxonomy.FailureKind.NSFW` was deliberately NOT touched: the dependency ran one way only, and that WARN-class value is independent and separately tested. |
| `nodes/_otr_shared/slot_matrix.py` and `content_oracle.py` | REMOVE-AFTER-MIGRATION | Test-only matrix and media/QC assertions. | Direct deletion would weaken live engine tests. | Move useful helpers into `tests/support/`, update consumers, then remove the production-looking modules. |

No tracked `_tmp_video_art_*.json` cleanup remains. Generated `scripts/_*.json`
and `scripts/_*.jsonl` artifacts are already ignored; do not invent a cleanup
chunk for them.

### 2.2 Dormant engine and 3D contract retirement

| Target | Verdict | What it does now | What is lost by removal | Required handling |
|---|---|---|---|---|
| `eng_character_3d.py` and the dormant `character_3d` contract | REMOVE-AFTER-MIGRATION | Three unregistered, unimplemented talker adapters plus preflight/licensing and family-lock scaffolding. | Dormant resurrection scaffolding and its private tests. | First move unknown-engine rejection to the post-freeze `OTR_VideoDirector` boundary. Then remove the adapters, zero-declarer capability/schema, 3D granularity lock, synthetic render/soak branches, and tests as one re-grounded family retirement. |
| `eng_still_parallax.py` | REMOVE-AFTER-MIGRATION | Real DepthAnything 2.5D renderer, but unregistered and absent from canonical routes. | A reusable implementation with test-only direct instantiation; no known saved/manual consumer was found. | Explicitly accept the reusable-code loss, remove its cross-imports and tests, and do not confuse it with `mesh_stage`. |
| `eng_triposr.py` | REMOVE-AFTER-MIGRATION | Unregistered dark adapter whose execution methods are unimplemented. | Dormant scaffold and its tests. | Remove after the guard migration and test-reference cleanup. |
| `hidream_i1.py` and `sd35_large.py` | RE-GROUND | Unregistered/unimplemented adapters also store later operator-authored prompt-style research. | Deletion would discard that research and break a directive-map test. | R4 must first select a concrete surviving prompt-policy authority and owning test. Until that destination exists and the still-valid directives are migrated, keep both files. |

The first code change in this lane is not a deletion. Add one registry-membership
validation immediately after `freeze_role_engines` in `OTR_VideoDirector`, and
validate every effective non-empty engine ID, including custom picks and
force-map values. Keep route freeze pure. Push that migration green before
removing the old hidden check.

All seven dark IDs were historically registered/selectable in alpha builds
before their unregister commits. R4 must explicitly decide whether that prior
public-menu exposure requires named `RETIRED_ENGINE_IDS`/validator compatibility
or whether generic unknown-engine rejection is sufficient; do not assume either
answer. Protect all of `mesh_stage`, `requires_mesh_fodder`, `directory_clip`,
`OTR_SilentComposite`, and the portrait ledger.

### 2.3 Public/manual nodes and the visual proof of concept

| Target | Verdict | What it does now | What is lost by removal | Required handling |
|---|---|---|---|---|
| `OTR_ProjectStateLoader` / `nodes/project_state.py` | REMOVE-AFTER-MIGRATION | Registered manual series-bible loader; absent from canonical. | The public/manual node ID, helper API, and potential external saved-workflow compatibility; no known saved artifact was found. | Remove the unused StoryOrchestrator import, unregister, add a validator tombstone, and document the replacement path. |
| `OTR_SaveToEpisodeWorkspace` | REMOVE-AFTER-MIGRATION | Registered manual image-batch episode sink with UI preview; absent from canonical but used at `C:\Users\jeffr\Documents\ComfyUI\_otr_full_api.json:261`. | Manual arbitrary-image save/preview workflows and that known local artifact. | Migrate or explicitly retire the known artifact, unregister, tombstone, and remove its tests. |
| `OTR_VideoProbe` | REMOVE-AFTER-MIGRATION | Registered manual per-role host/engine report; canonical code gets host facts elsewhere. | The public diagnostic ID, its user-facing report, and potential external saved-workflow compatibility; no known saved artifact was found. | Build and name a user-facing replacement or have R4 explicitly accept diagnostic loss; then unregister, tombstone, and update platform tests. |
| Five registered `visual/` POC nodes and legacy backends | REMOVE-AFTER-MIGRATION | Public manual POC surfaces, absent from canonical. | Five public IDs and potential external saved-workflow compatibility, plus worker CLI, wedge probe, and legacy backends; no known saved artifact was found. | Move `visual/_hf_token.py` startup behavior to surviving infrastructure first; then unregister all five IDs, add tombstones, clean scanners/tests, and delete the tree. |
| `OTR_VRAMGuardian` | RE-GROUND | Manual blanket unload node; conflicts with the targeted-lever policy. | Removing it loses explicit manual flush; keeping it preserves a policy exception. | The execution round must choose retire+tombstone, debug-gate, or rewrite onto targeted levers. Do not silently keep or rip it. |

### 2.4 Consolidation work that must preserve behavior

| Seam | Verdict | Safe boundary | Forbidden shortcut |
|---|---|---|---|
| Visualization helper duplication | **STRUCK -- DO NOT DO (operator, 2026-08-23)** | Was CONSOLIDATE. Operator: *"every video lane is independent"* / *"dont consolidate"*. The duplication across video lanes is DELIBERATE, not debt. | Do not centralize `_ref_path`, `_canvas_dims`, `_build_render_request`, `_clip_from_raw`, or anything else across video engines. |
| Six local image `_role_of` helpers | **STRUCK -- DO NOT DO (operator, 2026-08-23)** | Was CONSOLIDATE, and was BUILT AND REVERTED the same night under the ruling above. | Leave the six copies alone. |
| TTS sidecar lifecycle | CONSOLIDATE | Extend the live `_otr_audio_engines/_otr_sidecar.py`; migrate exact WAV loading first, then qualify IndexTTS2 lifecycle differences separately. | Do not introduce another sidecar abstraction or erase timeout/cleanup differences. |
| ffprobe callers | CONSOLIDATE | For node/runtime callers, one binary resolver/raw JSON probe, with caller-owned wrappers preserving GraphExecutionError, `-1`, best-effort, CreditsDataError, and CORRUPT_OUTPUT policies. Standalone script callers wait for the per-file audit and may adopt only a cold-import-safe helper. | Do not force one failure policy on every caller or silently pull standalone tools into a node import graph. |
| Writer | CONSOLIDATE IN SLICES | Move `_resolve_inputs` plus its source-helper closure to `nodes/_otr_writer_inputs.py`; then move the already-existing `WriterTailContext` and `_run_writer_tail` together to `nodes/_otr_writer_tail.py`, with hashes and focused tests at each seam. The ordered `INPUT_TYPES` descriptor belongs to the later schema chunk, not this split. | Do not use old line ranges, pretend the tail seams are not already isolated, or replace live `_otr_casting.lock_cast`. |
| StoryOrchestrator | CONSOLIDATE IN SLICES | Migrate runtime log/timeout first; then move the complete news/history/rank/RSS/recording closure with all consumers; only then delete definition-only orphan clusters. | Do not call the whole module dead or delete apparent clusters without closure reachability. |
| Writer widget schema | CONSOLIDATE, WORKFLOW-ATOMIC | Ordered descriptor manifest with lazy choices/defaults, preserving required/optional/hidden/forceInput and positional widget order. | Do not insert widgets, update code without canonical+variants, or treat a schema refactor as code-only. |
| Dict/object shapes | SEAM-SPECIFIC | Choose one shape per named boundary and migrate producer first, only inside the ordered chunk that owns that seam. This is not a standalone campaign wave. | No repository-wide attribute conversion; ComfyUI AUDIO and ledger formats are intentionally dicts. |
| Test path/stubs | **RE-GROUNDED 2026-08-23 -- the premise is FALSE at current HEAD** | Was: remove redundant root inserts. MEASURED: there is NO repo-root conftest.py and NO `pythonpath` key in pyproject.toml, so the 175 per-file `sys.path.insert` calls across the test tree are the LOAD-BEARING import mechanism, not redundancy -- removing any of them breaks collection. The runner-root proof the row demanded cannot be given because no runner root exists. A future chunk may ADD `pythonpath = ["."]` first and then sweep, as its own change with its own suite run. | No universal LLM stub and no global `sys.path` rewrite -- and now also: no insert removal before a runner root EXISTS. |
| `scripts/` | RE-GROUND PER FILE | Build a current owner/caller/test/operational-value table before each deletion group. | Do not reuse the old bulk kill list; active bakeoff, doctor, render, soak, and recovery tools are protected until proven otherwise. |
| Tombstone tests | RE-GROUND PER ASSERTION | Consolidate only pure duplicate absence assertions after checking they no longer guard live behavior. | No blanket deletion and no single catch-all file copied from the old plan. |
| OpenRouter surface | GATED RE-GROUND | Cloud/OpenRouter routes already exist. Build a current coverage matrix only after draft `config/profiles/otr_cloud_lanes.json` is ratified, its matching workflow variant/recipe is emitted, and a current canonical cloud smoke passes. | Do not execute the old file-count diet, mistake a missing emitted variant for a missing cloud boundary, or cut provider behavior before qualification. |

### 2.5 Runtime slop to repair rather than delete

| Surface | Current behavior and risk | Required change | Preserved behavior |
|---|---|---|---|
| README CRT claims | Three places describe SignalLost as a guaranteed fallback, but it is an explicit policy route. | Rewrite all three claims to describe an explicit/policy signal-lost route. | No runtime change. |
| Legacy audio output fallback in `nodes/video_engine.py` | A ledger-resolution failure can write to a hard-coded user output path. | First route headless/test recovery through explicit output configuration; then make production ledger failure loud. | Supported recovery remains explicit and testable; no hidden path shim. |
| Optional origin-label JSON in `nodes/video_engine.py` | Parse failure is silently ignored before deliberate raw-seed fallback. | Parse once and log-continue. Hard-fail only if a later design makes this optional metadata a production contract. | Raw-seed treatment fallback remains. |
| Two bare exceptions | Heartbeat write and model-loader eviction failures hide their exception type. | Catch `Exception`, log context, and preserve non-fatal heartbeat behavior; name loader eviction failure without swallowing it silently. | Heartbeat failure does not kill an episode; loader cleanup policy remains caller-owned. |
| `OTR_VRAMContextTest.optimization_profile` | The widget appears to control optimization but currently labels output only. | Either wire real behavior or deprecate it through an explicit saved-workflow/schema migration that preserves positional safety; do not silently delete, reorder, or leave an inert switch. | The diagnostic node itself remains. |

## 3. Protected surfaces and rejected old premises

These are not cleanup targets unless a later, separately grounded decision
changes their contract:

- `nodes/_otr_audio_cache.py`: live Google-TTS cache with integrity and billing
  value.
- `nodes/_otr_audio_engines/_otr_sidecar.py`: live Chatterbox/Dia lifecycle.
- `mesh_stage` and its directory-clip/SilentComposite lane.
- `OTR_VRAMContextTest`: real in-ComfyUI VRAM diagnostic. Its apparently inert
  `optimization_profile` widget is a separate honesty fix, not grounds to delete
  the node.
- `perfect_run_spacesaver`: intentionally preserved positional widget slot and
  tested deprecated no-op. The operator still wants a functional space-saver as
  separate deferred product work; cleanup must not delete its slot or use that
  request to revive the closed vendor-specific RTX node.
- SFX boundary: current runtime strips video audio and terminally muxes only the
  frozen upstream master mix. The operator-selected future direction is to
  retain and mix selected video-generation audio as inexpensive ambience, not
  build a separate provider layer whose technology moves too quickly. This
  cleanup campaign does not implement that feature; it must neither revive the
  retired provider/bed stack nor misstate the future path as live.
- Passage selector: parked future capability; redesign before any attempted
  wiring, but do not delete it as accidental slop.
- The frozen audio registry split: deliberate architecture, not unfinished base
  consolidation.
- Parenthesized sentinel enum values: intentional compatibility/reset behavior.
- Live gender-pool fallback: keep until bounded repair/pool-capacity guarantees
  exist and the policy test is intentionally inverted.
- Active bakeoff, doctor, render, soak, recovery, engine-matrix, and workflow
  validation tools.

## 4. Coding order

Each numbered item is an ordering boundary, not permission to batch unrelated
deletions into one commit. Within a boundary, land the smallest independently
green commit and push it before continuing.

| Order | Work | Why it is here | Exit condition |
|---:|---|---|---|
| 0 | Finish the executable rows in `docs/GO_FORWARD_PLAN.md`; its handoff row is only a pointer. At the campaign start, run the operator-pinned full `r2 -> r3 -> r4` arc against committed current HEAD and produce an exact file/symbol/test manifest. | Cleanup inventories decay fastest; the final pre-code pass must see the tree that will actually be cut. | R4 has no unresolved must-fix, dirty operator paths are excluded, and the manifest names every consumer and loss. |
| 1 | Truth and prevention: correct the README fallback claims; replace the hard-coded legacy audio path only after explicit recovery configuration exists; narrow/log the two bare exceptions; log-continue optional origin-label JSON failure; add the duplicate-definition guard; and move any still-valid HiDream/SD3.5 prompt directives into their real authority. | Make current contracts honest and prevent the same dead-code pattern before cutting files. | Focused contract tests plus full gates are green; production fails loudly where required and no supported recovery becomes stricter accidentally. |
| 2 | Zero-reachability removals: shadowed functions, unused sidecar/probe, retired shot-duration implementation, narrow legacy cast prototypes, and the unsafe normalizer after validator-report equivalence. Relocate test-only slot/QC helpers separately. | Smallest blast radius; validates the deletion discipline and shrinks later surveys. | Every removed symbol has zero production/public consumers; useful test assertions survive in test support; tombstones remain where required. |
| 3 | Move unknown-engine rejection to the post-freeze VideoDirector boundary. | This live fail-closed behavior is hidden inside the dormant family slated for removal. | Custom picks and force-map unknown IDs fail at the new boundary; known IDs behave unchanged; old guard still remains until this commit is pushed. |
| 4 | Retire the full dormant 3D/dark family scope approved by the current R4, preserving `mesh_stage`; handle HiDream/SD3.5 only after a concrete directive authority is selected and migration lands. Rebase the synthetic character-3D OOM/no-fallback proof in `render_driver.py` and `scripts/otr_video_soak.py` onto a live heavy engine before deleting those branches. | The guard dependency is now satisfied, and cutting the full zero-consumer contract avoids leaving misleading half-scaffolds without losing a useful failure contract. | No dark adapter/family consumer remains; OOM/no-fallback coverage survives on a live route; canonical dynamic routes and live mesh receipts stay valid; retirement handling for the historically alpha-public IDs matches the explicit R4 decision. |
| 5 | Retire selected public/manual nodes: ProjectStateLoader, SaveToEpisodeWorkspace, and VideoProbe. Resolve VRAMGuardian separately. | These are not canonical, but their saved-workflow compatibility cost requires explicit tombstones and replacement documentation. | Registrations/imports/tests are clean, known artifacts are migrated or explicitly retired, validator tombstones exist, and canonical validation passes. |
| 6 | Move Hugging Face token startup behavior, then retire the five-node `visual/` POC and backends. | Startup token discovery is the one live dependency that makes a blind tree deletion unsafe. | Token export works from surviving infrastructure before the POC commit; public IDs are tombstoned and scanner/test references are clean. |
| 7 | **CANCELLED BY THE OPERATOR, 2026-08-23.** Was: land exact small consolidations across engines. Operator: *"every video lane is independent"*, *"dont consolidate"*. Engine-to-engine duplication is an ARCHITECTURAL CHOICE here, not debt to be paid down -- a shared helper is a coupling point between lanes that are meant to move independently. The `_role_of` consolidation was actually built and proven green that night, then REVERTED unbuilt under this ruling. | -- | Nothing to do. Do not re-propose without an explicit operator reversal. |
| 8 | **APPROVED (operator, 2026-08-23, after a Fable explainer): "do it with everything else that is safe in a new window and lets regression test."** Consolidate ffprobe resolution/raw probing for node/runtime callers while retaining caller-specific wrappers -- the failure POLICIES stay per-caller, which is what distinguishes this from the cancelled order 7. Leave standalone script callers unchanged until their order-11 owner audit proves a cold-import-safe adoption. Known wins recorded during the consult: only otr_credits_roll honors OTR_FFPROBE today; the "25/1" fps-parse bug has been independently re-fixed at least three times; wan_shared trusts PATH blindly. | It is broad but mechanically bounded once dead/public surfaces are gone, without pre-empting script reachability work. | Every migrated caller retains its named error/fallback contract and focused failure tests; standalone tools still run independently; the full suite plus a live canonical render proof gate the chunk. |
| 9 | Split Writer and StoryOrchestrator in symbol/closure-sized green commits. Delete proven orphan clusters only after their live closure has moved. | Highest semantic blast radius; it benefits from all earlier surface reduction. | Replay/hash/contract tests prove behavior at each seam; no source, news, history, timeout, runtime-log, or CastLock path is stranded. |
| 10 | Replace imperative Writer `INPUT_TYPES` construction with an ordered descriptor manifest. Update canonical plus every variant/recipe atomically. | Widget position is persisted production data, so this is a separate workflow epoch after code seams stabilize. | Widget counts/names/order match live `INPUT_TYPES`; JSON round-trip, referential links, variant checks, `OTR_WorkflowValidator`, and a real canonical smoke are green. |
| 11 | Audit `scripts/` per owner, including standalone ffprobe callers; then do scoped redundant test-root/stub cleanup and assertion-level tombstone/test consolidation. | Operational scripts, import setup, protocol fixtures, and live guards cannot be inferred from names or age. | Each deletion has an owner/caller/test record and accepted loss; no active bench, doctor, recovery, render path, or protocol-specific fixture disappears. |
| 12 | Re-ground the OpenRouter diet only after draft `otr_cloud_lanes` is ratified, a matching workflow variant/recipe is emitted, and a current canonical cloud smoke passes. | Cloud routes exist, but that specific profile-to-workflow qualification boundary is incomplete, so pruning before it would be guesswork. | Current coverage matrix proves what is redundant; the emitted variant validates and all retained cloud routes have current live receipts. |

## 5. Verification contract for every code chunk

1. Re-run source/import/profile/registration reachability for the exact symbols.
2. Run focused tests, then the full Windows suite with
   `pytest -q -p no:cacheprovider`, plus the Bug Bible regression. A static audit
   finding never creates a production PBUG or Bible entry.
3. For any node, widget, schema, wiring, or profile impact, load and validate the
   real `workflows/otr_canonical.json`. Reflect graph-affecting node/wiring/widget
   changes in canonical and variants in the same change; for intentionally
   unwired public-node retirement, prove and record that absence. Audit widget
   counts/names/order and every link endpoint.
4. Add a canonical headless/live smoke when a reachable runtime or graph
   contract changes. Reset ComfyUI selectively first and verify the final asset,
   not merely resident VRAM.
5. Check touched Python with AST parse; check touched files for UTF-8 without
   BOM and for zero-byte accidents.
6. Commit and push the smallest green chunk to `v2.0-alpha`; verify
   `HEAD == origin/v2.0-alpha` before starting the next chunk.

Stop the chunk if reachability, loss, or replacement differs from this matrix.
Update this current plan first, then re-run the affected round; do not preserve
a stale kill list for the sake of momentum.
