# OTR bounded cleanup: implementation contract for Opus

## 0. Authority, scope and current state

Owner: Opus implements; Astra is the code-grounded reviewer and sole judge of this plan. This is a bounded cleanup sample, not a repository audit. User requests a full Kibitz R1-R4 with two reviewers, followed by a handoff. Do not implement the three runtime changes during plan review.

Verified Windows base: `1f27e4123a9567400ab39ef698d2a8a394601284`, branch `v2.0-alpha`. Earlier reports used `719edac0f0fe0c859bce5dd40cb9c45d98d3c4db`; the intervening fast-forward did not change these audio/freeze sources or canonical graph. Revalidate at implementation HEAD. Paths below are relative to `C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio`.

Read AGENTS.md, CLAUDE.md, docs/PRODUCTION_SPRINT_LESSONS.md, applicable PROD_BUG_LOG/Bug Bible entries, docs/OTR_STANDING_RULINGS.md, docs/DEAD_CODE_HUNT_PROMPT_V5.md and docs/DEAD_CODE_EXECUTION_PLAN.md. Newer closure receipts outrank historical Lean Mean observations. The Sep 4 shared ffmpeg resolver closure supersedes the old three-resolver KEEP; no ffmpeg work is proposed here. Preserve independent engine lanes and operator KEEP rulings. Static findings do not create PBUG/Bible entries.

Current preparatory changes: new `scripts/otr_canonical_audio_check.py`, new `tests/test_canonical_audio_check.py`, removal of the 228-line copied-loop `tests/test_episode_assembler_offset_shift.py`. No production edits. Unrelated untracked `diff.txt` and `diff_utf8.txt` belong to the user and must remain untouched.

## 1. Ranked eligible work and exact edits

### C1. Remove seven write-only SceneSequencer assignments (confirmed)

File/symbol: `nodes/scene_sequencer.py:888`, `SceneSequencer.sequence`.

Delete lines 928-932: `breath_ms`, `beat_pause_ms`, `pause_ms`, `scene_transition_ms`, `act_break_ms`; delete the obsolete two-line pacing comment immediately above. Delete `current_character_name = None` at 988 and `current_character_name = character_name` at 1124. Verify symbol references before deleting; numbers are navigation aids, not patch coordinates.

Correct the stale pacing comment at 922 and module header at 6-15 in this same file. Describe SceneSequencer as sequencing supplied character/announcer audio against ledger lines with roomtone/DSP and durable scene positions; describe EpisodeAssembler as combining the supplied scene/music bus and writing the master WAV. The header must not claim this node generates Bark/Parler speech or implements removed BEAT/PAUSE buffers.

Chain: registered `OTR_SceneSequencer` (`__init__.py:171`) -> canonical node 3 -> `sequence` -> scene audio and durable line positions -> canonical AudioEnhance -> EpisodeAssembler/master WAV. These six local names have seven stores and no reads, dynamic lookup or exported identity. No config, widget, port, public signature or platform adapter changes. Payoff: seven irrelevant assignments and misleading prose removed; no measured speed claim. Risk: accidentally deleting neighboring live pacing/DSP/state code. Preserve `current_env`, roomtone, character delivery processing, all tensor device/dtype logic and the complete assembly loop.

### C2. Give G8 sole ownership of freeze line-ID collision diagnostics (confirmed)

File/symbol: `nodes/_otr_ledger_freeze.py:262`, `_check_per_line_invariants`; second owner `_check_g8_line_id_uniqueness:893`.

Delete `seen_line_ids: set[str] = set()` at 299 and the four-line collision `elif`/`else` block at 310-313. Keep the missing/non-string/empty ID check at 308-309. Keep G8 unmodified. Update this function's docstring at 270 to describe ID shape validation and name G8 as the uniqueness owner.

Chain: FreezeCascade -> `run_gap_audit:706` -> per-line invariants at 719 and G8 at 728 -> Phase 0 audit receipt at 1017 (collects), Phase 10 at 1045 (refuses invalid freeze) -> structured `.errors`, logs and downstream verdict. G8 retains one capped summary, first five repeated occurrences plus `(+N more)`.

Payoff: five implementation lines and a second diagnostic owner removed. This intentionally changes error messages/counts for duplicate IDs; it does not change acceptance. The reduction depends on the number of repeated occurrences, not a constant one error. One in-repo test pins the old word: `tests/test_lfc_phase_0_10_gap_audit.py:242`, `test_duplicate_line_id_raises`, asserts `"duplicated"` at 247. Change only that assertion to the G8 summary/duplicate ID in the SAME chunk; the char-ID test at 417-422 must keep its distinct `"duplicated"` expectation. External log scrapers are unverified. Do not promise byte-identical error receipts. Public functions, schemas, phase ownership, missing-ID rejection, role checks and platform behavior stay intact. The similar duplicate check in `nodes/_otr_ledger_cleanup.py:490` is the writer's separate validation stage and stays. Cast char-ID uniqueness also stays.

### C3. Stop acquiring a model that the freeze cascade never calls (confirmed; highest behavioral risk)

File/symbol: `nodes/OTR_LedgerFreezeCascade.py:191`, `OTR_LedgerFreezeCascade.run`.

1. Keep `_OTRML` import: final teardown still uses it. Keep replay/no-ledger early returns at 215-263.
2. Replace 265-269 with a short accurate comment and `_OTRMI.require_model(technical_model, slot="technical")`, retaining nonempty wired-input validation but removing the unused `resolved_technical_id` binding.
3. Remove 270-287: imports of `policy_from_meta` and `load_config_from_meta`; `lfc_data`, `lfc_meta`, `lfc_policy`, `lfc_load_config`; `request_slot`; `make_generate_fn`.
4. Pass `None` as the existing first positional argument to `_LFC_ORCH.run_freeze_cascade` at 314. Preserve the public orchestrator signature and node schema.
5. Correct adjacent comments and the `technical_model` tooltip at 144-152. Suggested tooltip: "Technical model ID from the writer. Required for compatibility on a normal current-ledger run; freeze does not acquire or generate with this model." It validates a nonempty value, not graph connectivity or catalog membership. Preserve input name, type, forceInput, widget order and links. Say serialization precedes the unload receipt, not that no model can be resident. Do not refactor the two serialization blocks.

Also replace the freeze node's stale module docstring at 1-5 and the nearby obsolete reviewer-phase comment at 135-140 with current deterministic validation/readiness/freeze wording. These are prose corrections within the already-owned node file.

Chain: canonical node 1 output slot 4 -> link 115 -> FreezeCascade node 62 `technical_model` input 4 -> require_model -> currently request_slot/make_generate_fn -> `run_freeze_cascade:714` -> `_run_inline_safety_cleanup:676` -> `del generate_fn:692`. The cleanup implementation was retired by operator directive on Aug 5; it only stamps `same_story_safety_cleanup`. Other policies bypass that stub. No callback invocation survives. Freeze outputs feed the canonical ledger/audio/video consumers; the returned tuple remains seven items and script_json equals v2_ledger_json.

Payoff: about 18 setup lines plus no unnecessary acquisition/config plumbing on this path. `request_slot` can validate catalog/policy, choose a provider, admit local VRAM, reuse cache or load a model. Remote paths do not necessarily load local weights or evict residency; no GB/seconds savings are measured. Removing the call intentionally removes its catalog, policy/config, provider and VRAM failure paths. `require_model` validates a nonempty string only, not catalog membership. Blank wired input still fails on a normal current-ledger run. Replay/no-ledger behavior still returns before validation. No provider/platform lane is merged or modified.

C3 is a removal of unused model acquisition, with a deliberate failure-path change; it is a separate commit from C1/C2. Freeze success will no longer imply that the technical ID is loadable. Actual generating consumers, including shot-lock's `request_slot("technical", ...)` at `nodes/otr_shot_lock.py:1721`, own their own admission. A nonempty invalid ID can fail there later, or never be admitted if no generating consumer runs. Do not retain unused acquisition to satisfy old diagnostics. In `nodes/_otr_freeze_cascade.py`, update only `run_freeze_cascade`'s docstring at 721-727 to describe deterministic validation/readiness/freeze and the retired receipt, not optional content cleanup; its implementation and signature remain untouched.

Keep `_otr_freeze_cascade` receipts/policy selection/readiness/freeze behavior, capability telemetry, `vram_at_cascade_entry_gb`, public callback parameter, and final `unload_llm_if_local_resident` at 376. Writer teardown can be skipped by `OTR_WRITER_UNLOAD_AFTER_SCRIPT=0`; residual local residency therefore still needs the final gate. Keep unload failure stamping, exception propagation, script-text fallback, first JSON fallback, and post-finally JSON serialization/fallback. The telemetry value may change because the unused model is no longer acquired; its field stays.

Branch contract for C3:

| Trigger | Required behavior after edit |
|---|---|
| Canonical replay descriptor | Return seven-item replay pass-through before validation/acquisition; no new cleanup or freeze receipt |
| No current ledger / peek returns None | Existing `needs_full_rerun` result and error JSON; no placeholder ledger or acquisition |
| Blank technical value on current-ledger path | Existing MissingModelInputError before entering the cascade; no load/unload added to this early exit |
| Nonempty but unavailable/unknown model or rejected LLM load policy | Do not request_slot; deterministic freeze proceeds without model catalog/provider/VRAM admission |
| Existing local cached model | Do not reuse it for generation; retain final conditional unload, including residual writer ownership |
| Remote/empty cache | Final conditional unload skips local teardown; do not invent an always-load or always-unload guarantee |
| Policy resolution / structural freeze rejection | Real orchestrator returns its existing terminal disposition and receipts, node still runs finally |
| Unexpected cascade exception | Run finally once, then propagate the original exception |
| Script-text assembly exception | Keep incoming script text; still finalize the ledger JSON and unload receipt |
| First serialization fails | Initially retain incoming JSON; second serialization may subsequently succeed |
| Final serialization fails | Retain the first serialization (or incoming JSON if both failed); no invented guarantee that the latest stamp reached the wire |
| Unload raises | Log failure, stamp freeze_unload_ok=False, retain otherwise successful verdict |
| Rerun/recovery | No automatic retry is added. Ordinary node re-execution and the existing replay route keep their separate semantics; provider/model retries of other generating callers are untouched |

User clarification: never retain obsolete runtime code solely to satisfy diagnostic/source-string tests. Retained checks/receipts above have stated interface, freeze acceptance, recovery or downstream ownership; obsolete test expectations change with the retired behavior. Broader retirement of the public technical socket is a separate schema/consumer decision, not secretly folded into this no-wiring-change chunk.

Test files requiring edits in the SAME C3 chunk: `tests/test_llm_runtime_policy.py`, `tests/test_lfc_b1_cascade_unload_in_finally.py`, `tests/test_cascade_freeze_unload_visible.py`, `tests/test_lfc_c4_news_used_passthrough.py`, and `tests/test_freeze_policy_readonly.py`. The first file's `test_downstream_llm_consumers_thread_the_ledger_policy:339` still asserts two freeze source strings at 346-347. Remove the freeze-source read/assertions, rename it `test_shot_lock_threads_the_ledger_policy`, reword its docstring around that live consumer, and preserve its `policy=policy_from_meta(meta)` expectation. The two unload test files and C4 passthrough helper must replace success stubs with fail-fast sentinels; retain all news_used assertions. The policy file owns the real-cascade proof below. Never add dead text to satisfy obsolete source assertions.

## 2. Ownership and extraction decision

SceneSequencer owns scene-relative line timing; EpisodeAssembler owns promotion to master-mix coordinates; G8 owns freeze collision summaries; the writer owns its earlier duplicate validation; the model loader owns actual acquisition for callers that generate. Preserve those boundaries. Do not extract an offset helper just to make a copied harness convenient. The live assembler loop at `scene_sequencer.py:1896-1921` covers persisted legacy clips and remains in production. Music-row marker promotion at 1923+ has different zero-offset behavior and stays separate. No new wrapper or common engine abstraction is needed for C1-C3.

Expected runtime reduction is approximately 30 lines plus adjacent comments, not an audited repository total. New validation adds code. Retiring 228 lines of mirrored test code is reported separately from production cleanup; the new script/test currently total 331 lines, so the preparatory replacement is a net increase of 103 test/check lines.

## 3. Fresh evidence contract

The new audio check loads `workflows/otr_canonical.json` on each case, real NODE_CLASS_MAPPINGS, live INPUT_TYPES, FUNCTION and positional widgets; it verifies the selected edges, link fan-out/types and declared boundary producers. It calls real SceneSequencer -> AudioEnhance -> EpisodeAssembler, saves/loads a real ProductionLedger and writes/decodes a real master WAV. An independent correlation measurement locates the enhanced signal in the WAV; ledger positions must agree within one sample. Opening and no-opening cases check measured offsets, prior unmarked clips, already-master clips, missing clip time and repeated real assembly. Fresh output directories, canonical/source hashes and start/end HEAD prevent silent reuse of old receipts.

Boundaries are synthetic upstream voice/music assets and ledger, blank video policy and replay descriptor. This proves the non-foley/non-replay CPU audio segment only. It does not execute ComfyUI scheduling/IS_CHANGED, model generation, model cache behavior, foley, a full episode, GPU, or publishing. Full graph qualification is not inferred. The old 12-method offset test copied a loop and did not call the production implementation; removal does not mean every historical fixture has been requalified by two new cases.

At this base: focused fresh launcher passed (1 test, 9.75 s); measured offsets 1.50 s and 0.00 s. Full existing-suite run before retirement and full run after replacement both exited 2 with the exact SAME 54 failing node IDs; no additional failures. This is a relative comparison, not a green suite. Bug Bible: 22 passed, 27 skipped, 3 xfailed (exit 0). No reason to classify all old tests as stale. Failure baseline must be recaptured at implementation HEAD, not waived forever.

R2 preparation correction: real roomtone (`scene_sequencer.py:692`) and tape hiss (`audio_enhance.py:278`) consume NumPy randomness, so the unseeded check's WAVs were not deterministic. The fresh check now seeds NumPy and Torch to 20260910 per case and records that seed; the launcher checks byte-identical WAVs from TWO independent fresh processes. Latest focused result: 1 passed in 12.11 s. Bug Bible repeated: 22 passed, 27 skipped, 3 xfailed. The full suite is being rechecked after this test-only refinement; read the final qualification receipt for its completed result. Compare hashes only with the same seed, saved canonical settings and environment, never against old unseeded receipts or across libraries/platforms.

## 4. Implementation sequence and validation recipes

### P0. Intake and qualification (before runtime edits)

Read current rules/queue and this final plan. Record Windows branch/HEAD/status and canonical SHA. Coordinate one coder owner; preserve unrelated changes. Inspect actual files by symbol and ensure C1 stores remain unread, G8 still runs, and the cascade still never calls its callback. If any of these preconditions changed, stop that candidate and report the new consumer. Use the current Windows venv; never the lagging Linux mount. Inspect fresh preparatory harness changes/receipt and confirm they have landed before building on them.

P0.5 is the preparatory commit: the new audio script/launcher and old copied-loop test removal must be committed and pushed together with the qualification receipt BEFORE C1. If Astra's closing receipt already supplies that pushed commit, verify it instead of repeating the work. If those three paths are still uncommitted, finish their focused/full/Bible qualification and land them as their own chunk. C1 cannot rely on an untracked local harness.

No ComfyUI server, GPU run, external generation or paid panel is needed for this cleanup. Do not run old soak/cold-drive scripts as qualification. Do not edit canonical JSON: there is no intended node/widget/link change. Record byte-identical canonical before/after; run the existing validator/guardrails against that real file. Stop on schema drift instead of manufacturing a replacement graph.

### P1. C1 chunk

Edit only `nodes/scene_sequencer.py` as specified. Run the fresh canonical launcher test and input-type/signature parity tests. Compare deterministic WAV hashes and measured offsets against a newly captured pre-edit run under the same environment. Timing values, coordinate markers and repeated-assembly behavior must agree. No new test that merely asserts names are absent.

### P2. C2 chunk

Edit `nodes/_otr_ledger_freeze.py`, the line-ID assertion in `tests/test_lfc_phase_0_10_gap_audit.py`, and strengthen `tests/test_g8_line_id_uniqueness.py` with real function calls, not source-string copies:

- For two equal IDs, Phase 0 must collect exactly one collision diagnostic, the G8 summary. Phase 10 must still refuse and carry that summary in `.errors`; unrelated minimal-fixture errors may coexist.
- In `test_duplicate_line_id_raises`, collect errors starting with `"G8:"`; assert exactly one and that it contains the duplicated fixture ID plus `"duplicate line_id(s)"`. Do not weaken the old assertion to `"duplicate" OR "G8"`, which could accept an unrelated message.
- Missing, empty and non-string IDs must still produce per-line invalid-ID errors and no G8 collisions; include integer input as well as None.
- Eight occurrences of one ID -> exactly one G8 summary with `+2 more`; other independent invalid-role errors remain.
- Exercise the writer duplicate check through its existing test coverage; its diagnostic and behavior are unchanged.

Do not replace all freeze fixtures or count the whole error list as one. Expected changes are collision diagnostics only; no acceptance relaxation.

### P3. C3 chunk

Edit the freeze node and runtime-policy test exactly as C3. Extend BOTH `tests/test_lfc_b1_cascade_unload_in_finally.py` and `tests/test_cascade_freeze_unload_visible.py` in place for boundary-focused behavior:

- Replace successful `request_slot`/`make_generate_fn` stubs with fail-fast sentinels and assert zero calls on normal run, cascade exception and script assembly fallback. Record `run_freeze_cascade` receives None plus the same ledger and readiness toggles. Mocking the cascade here is acceptable only for fault injection/argument capture, not evidence that the real cascade never generates.
- Successful unload and failed unload: preserve verdict and seven outputs; returned JSON copies must match and expose the correct `freeze_unload_ok` boolean. Cascade exception must propagate while unload runs once; first/second serialization failures must retain documented fallback behavior.
- Add blank technical-model normal-run refusal plus replay/no-ledger no-acquisition checks, retaining their existing return contracts.
- Put these three early-exit cases in the B1 test file with their own patches, not its helper that forces a current ledger: replay JSON is `{"meta":{"replay_from":"bundle"}}`, result verdict is `replay`; no-ledger uses `has_current_ledger=False` and no peek patch, result verdict is `needs_full_rerun`; blank-model uses current-ledger/peek patches with `technical_model=""` and `"   "`, expecting actual MissingModelInputError. Acquisition sentinels always remain zero. Assert no cascade or final unload on these pre-try exits; retain exact seven-output contracts where the function returns.
- Put serialization fault cases in the B1 file with the orchestrator stubbed to a successful disposition. Avoid changing the global json module: patch the node module's `json` NAME to a SimpleNamespace containing real json.loads and a counter-based dumps callable that delegates to the saved real json.dumps except at the chosen call number. Test first-only, second-only and both dumps failing. First-only must return the real second JSON with the unload stamp; second-only keeps the first JSON, which lacks the newly added stamp in a fresh fixture; both keep incoming JSON. Assert two serialization attempts, one final unload and equal JSON output ports. Do not run these patches on replay/no-ledger cases, whose JSON paths differ. Do not patch json.loads or use a precomputed 'ok' string as proof that serialization saw the correct ledger state.
- In `tests/test_freeze_policy_readonly.py`, add `test_canonical_freeze_without_acquisition_in_fresh_process`, parametrized on `original` and `scifi_news_pro`. The exact fresh-process implementation contract below is the decisive proof; the older mocked node tests are supplemental. Do not mock the real cascade, cleanup, policy resolution, phase dispatch or ledger persistence.
- Bind the node invocation's readiness values/input ports to the real canonical node 62; assert the unchanged link 115 and actual NODE_CLASS_MAPPINGS class. Old mocked node tests remain supplemental. Do not create a separate fake workflow.

Fresh fixture recipe (Astra executed this against the real package at the base HEAD, both routes returned `frozen_with_warns`, zero post-audit errors): use `production_ledger.new_ledger(episode_id, str(tmp_path / "otr" / "episodes" / episode_id / "audio"))`; create `meta` with `setdefault` (new ledgers do not initially have it); stamp `source_bank=bank`, `episode_title="Freeze fixture"`, `style="radio_drama"`. Cast is one row `{char_id:"c01", name:"Mira", traits:"calm"}`. Lines is one row `{line_id:"l001", char_id:"c01", speaker_role:"character", text:"The signal is clear.", beat_id:"b001"}`; beats is `[{beat_id:"b001"}]`. Leave other constructor defaults. For `scifi_news_pro`, call the real `_otr_content_authorship.stamp_receipt(led.data, owner_bank=bank, accepted_artifacts={"probe":{"text":"The signal is clear."}})` before saving; this exercises real voice coverage and authorship validation. Call `led.save()`, then real cascade with canonical readiness values (currently True/True). This fixture's authored asset is declared synthetic; do not fake accepted production provenance. Receipt: `docs/2026-09-10-cleanup-opus/freeze_fixture_receipt.txt`.

Permanent fresh-process proof (locked choice; no new framework/script file):

1. The new pytest test launches `[sys.executable, "-", str(ROOT), str(tmp_path), bank]` with a small literal Python program on `input=`, `text=True`, UTF-8, captured stdout/stderr and timeout 120 s. No shell quoting or external CLI. The child must not import pytest, tests or old fixture builders. CPU/UTF-8/no-bytecode environment is explicit; set OTR_OUTPUT_DIR and OTR_EXTRA_OUTPUT_ROOTS to the fresh child output root and remove OTR_OBS_DIR.
2. Import `scripts/otr_canonical_audio_check.py` by file path and call its `load_package()` to load actual registrations in the fresh interpreter. Import production ledger, model loader, cascade and authorship modules under that package namespace. This uses the fresh production loader, not collection-time module replacements. `_CURRENT` cannot leak back to the pytest parent because the entire real invocation is in the child.
3. Read/hash the actual canonical file, find exactly one active OTR_LedgerFreezeCascade by type, resolve its class from actual NODE_CLASS_MAPPINGS, inspect live INPUT_TYPES/FUNCTION/RETURN_TYPES. Bind widgets by their input widget names in saved order (currently the two readiness booleans). Validate declared names/count and all incoming link endpoints/source output fan-out. In particular technical_model comes from writer output 4 via link 115 at this base; follow actual names/links and fail if the producer boundary changes, rather than hardcoding a replacement graph.
4. Build the fixture above. The five linked writer values are declared synthetic boundaries: script_text="The signal is clear.", script_json=json.dumps(led.data), news_used="fixture", estimated_minutes=1, technical_model="unused-model-boundary". Use that intentionally non-catalog nonempty ID to prove freeze no longer attempts admission. Replace only loader.request_slot and loader.make_generate_fn with counter-incrementing functions that raise AssertionError. Invoke the actual registered node FUNCTION with bound widgets and these boundary values; do NOT substitute run_freeze_cascade or unload_llm_if_local_resident. Empty cache makes the real conditional unload a no-op on CPU.
5. Assert zero acquisition calls; seven outputs; output[1] == output[6]; successful frozen verdict; preserved news_used; returned JSON freeze_unload_ok=True; unchanged canonical text; expected retired/not-applicable cleanup receipt; persisted ledger exists. Then invoke real run_freeze_cascade with a separate counter-incrementing poison callback and the same valid ledger/readiness values; assert zero callback calls, successful disposition, zero post-audit errors and unchanged text. This intentionally also exercises deterministic re-entry; it does not claim scheduler cache/replay qualification.
6. Emit a compact JSON receipt under the child output root with bank, canonical hash, production source hashes, invoked class/function, node/input links/widget values, declared boundary values, verdict, counters and ledger path. Check HEAD/canonical/source hashes before/after. Parent requires child exit 0 and matching receipt/counters; after a fresh checkout with C3 not applied, the acquisition sentinel should fail, establishing a meaningful regression check.

The new fresh-process check is implementation work for C3, not already-passed evidence. Astra's existing probe proved the real orchestrator/fixture routes only; it did not run the current acquiring node with poison acquisition. The audio check and this proposed freeze test remain separate bounded routes sharing only package loading; no generic workflow-execution framework is introduced.

Exact owned file list: C1 one production file; C2 one production file plus the two G8/phase tests above; C3 the freeze node, one docstring-only edit in `_otr_freeze_cascade.py`, and five tests above. Do not change `_otr_freeze_cascade.py` implementation/signature, `_otr_model_loader.py`, shared policy modules, `__init__.py` or the canonical JSON. Correcting tooltip prose changes descriptive INPUT_TYPES metadata only, not the input/widget contract.

Focused command arguments (append to the Windows venv python; Opus runs them, never the operator):

| Stage | Arguments |
|---|---|
| C1 | `-m pytest tests/test_canonical_audio_check.py tests/test_input_types_signature_parity.py -q -p no:cacheprovider` |
| C2 | `-m pytest tests/test_g8_line_id_uniqueness.py tests/test_lfc_phase_0_10_gap_audit.py tests/test_ledger_cleanup_pass.py tests/test_ledger_cleanup_contracts.py -q -p no:cacheprovider` |
| C3 | `-m pytest tests/test_freeze_policy_readonly.py tests/test_lfc_b1_cascade_unload_in_finally.py tests/test_cascade_freeze_unload_visible.py tests/test_lfc_b14_unload_on_exit.py tests/test_lfc_c4_news_used_passthrough.py tests/test_freeze_cascade_v2_ports.py tests/test_canonical_replay.py -q -p no:cacheprovider` |
| C3 policy | `-m pytest tests/test_llm_runtime_policy.py::test_shot_lock_threads_the_ledger_policy -q -p no:cacheprovider`, plus the full `tests/test_llm_runtime_policy.py` module for before/after failure comparison. Seven existing failures in that module are NOT permission to skip new failures. |
| Workflow | `-m pytest tests/test_workflow_live_passes_validator.py tests/test_workflow_json_guardrails.py tests/test_workflow_validator_widget_vector.py tests/test_widget_drift_gate_no_silent_exemption.py -q -p no:cacheprovider` (compare the known writer-widget failure separately) |
| Drift | `-m pytest tests/test_dropdown_matrix_drift.py tests/test_machine_matrix_drift.py tests/test_model_asset_index_drift.py -q -p no:cacheprovider` |
| Variants | `scripts/build_variants.py --check` (current result: no committed variants; soft-skip guard OK; this is not 54-variant qualification) |

For the direct structural gate, load the actual registered `OTR_WorkflowValidator` class in a fresh package process and call `validate(str(CANONICAL), True, True, prompt=None, unique_id=None)` with empty optional stamp fields. `None` prompt means there are no live engine selections to prepare. Pair it with JSON round-trip, canonical byte hash, and the workflow tests; no generated API prompt, network or assets are required. If the current implementation changes asset preparation semantics, stop and use the pure `validate_workflow_contract` and `widget_vector_drift` checks with actual mappings instead, reporting that narrower scope.

### P4. Every runtime chunk: acceptance, git and stop/rollback

Use `C:/Users/jeffr/Documents/ComfyUI/.venv/Scripts/python.exe`, UTF-8, CPU mask, pytest `-q -p no:cacheprovider`. Run focused tests, full repository suite and Bug Bible from its own root with relative `tests/bug_bible_regression.py`. Include dead-code plan's drift battery and `scripts/build_variants.py --check` where applicable; use exact current available commands, never guessed script names. Save commands, exits and failing-node-ID sets. Require focused checks green and zero NEW unexpected full-suite failures relative to an untouched same-HEAD baseline. Do not add failures to quarantine/EXPECTED_FAILED_NODEIDS to pass this wave. Existing unrelated failures remain separately disclosed; do not claim the full suite green.

Verify canonical SHA unchanged, AST parse and no empty/BOM touched files, `git diff --check`, exact scoped diff. Commit and push each qualified chunk together to `v2.0-alpha`, verify HEAD equals origin. Never edit pyproject/release metadata, tag/promote/publish, blanket reset or revert unrelated work. If HEAD changes, revalidate changed prerequisites. On a new regression, fix within the chunk or revert only this chunk's patch; after a pushed chunk use a scoped revert commit and push, never reset shared history. Stop if removing C3 acquisition leaves a real callback consumer, changes freeze acceptance/receipts beyond the declared removals, or requires a new architectural decision. Deliver exact net line count from the finished diff, not this estimate.

## 5. Review ownership, limits and coverage

For the remaining rounds, review residual defects in this CURRENT plan, not abandoned wording from earlier reports. Read only source sections necessary to verify a concrete concern. Fifteen primary files is a hard maximum, not a reading quota. Do not re-report a requirement as missing when this plan already states it. Focus R3 on the fresh child-process binding, receipts, failure injection and exact edit ownership; focus R4 on any remaining must-fix. A concise zero-defect review is valid.

Exactly two local Kibitz lanes: Antigravity and Cursor; driver Codex/Astra has no duplicate CLI lane. Run R1 approach -> R2 coding -> R3 wiring -> R4 convergence sequentially. Driver writes an anchor before each fan-out, verifies every claim, records judgment and feeds each final.md byte-for-byte into the next input. No nested delegation, extra workers, model upgrades, paid panels, code edits, tests, servers or GPU activity by reviewers. Antigravity may write only its designated review file; Cursor returns stdout. Read current real Windows files; a profile snapshot is context, not current truth. Cite file:line and exact symbols, consumer/output chains, compatibility, payoff/risk and confirmed/likely/unverified status. Zero defects is valid. Keep each review concise (under 1000 words), no file dumps. Review no more than 15 primary files; use targeted consumer searches and list uncovered work if this is insufficient. Existing applicable operator rules remain authoritative; this document adds no authority to override them.

Antigravity primary ownership (15 files): this plan; AGENTS.md; CLAUDE.md; docs/OTR_STANDING_RULINGS.md; docs/DEAD_CODE_HUNT_PROMPT_V5.md; docs/DEAD_CODE_EXECUTION_PLAN.md; nodes/scene_sequencer.py; nodes/audio_enhance.py; nodes/production_ledger.py; scripts/otr_canonical_audio_check.py; tests/test_canonical_audio_check.py; nodes/_otr_ledger_freeze.py; tests/test_g8_line_id_uniqueness.py; workflows/otr_canonical.json; __init__.py. Focus C1/C2 ownership and fresh audio evidence. Targeted search writer collision consumers; do not expand to other engines.

Cursor primary ownership (15 files): this plan; AGENTS.md; CLAUDE.md; docs/OTR_STANDING_RULINGS.md; docs/DEAD_CODE_EXECUTION_PLAN.md; workflows/otr_canonical.json; __init__.py; nodes/OTR_LedgerFreezeCascade.py; nodes/_otr_freeze_cascade.py; nodes/_otr_model_loader.py; nodes/_otr_model_inputs.py; tests/test_llm_runtime_policy.py; tests/test_lfc_b1_cascade_unload_in_finally.py; tests/test_cascade_freeze_unload_visible.py; tests/test_lfc_phase_7_8_readiness.py. Trace C3 through acquisition/cache, failure, final unload and recovery; focus exact implementation/test instructions. Targeted searches may locate callback consumers/fixture ownership but list larger unreviewed bodies.

Astra owns adjudication, required project lessons/newer receipts and closing remaining fixture/command details. Reviewed runtime sample: audio sequencing/enhancement/assembly, freeze diagnostics and unused acquisition route. Unreviewed: full model generation, scheduler cache/retry, all engine lanes, all legacy graphs, full publishing, broad audio cache teardown, other harness retirement. Preserve existing KEEP decisions without re-litigating them. Next bounded wave after these three chunks: trace one canonical artifact from generation to durable reuse and ask which work is repeated or discarded, with proof of all consumer/compatibility paths before proposing removal. No additional hunting in this implementation wave.
