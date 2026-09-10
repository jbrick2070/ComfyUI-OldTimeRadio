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

Chain: registered `OTR_SceneSequencer` (`__init__.py:171`) -> canonical node 3 -> `sequence` -> scene audio and durable line positions -> canonical AudioEnhance -> EpisodeAssembler/master WAV. These six local names have seven stores and no reads, dynamic lookup or exported identity. No config, widget, port, public signature or platform adapter changes. Payoff: seven irrelevant assignments and misleading prose removed; no measured speed claim. Risk: accidentally deleting neighboring live pacing/DSP/state code. Preserve `current_env`, roomtone, character delivery processing, all tensor device/dtype logic and the complete assembly loop.

### C2. Give G8 sole ownership of freeze line-ID collision diagnostics (confirmed)

File/symbol: `nodes/_otr_ledger_freeze.py:262`, `_check_per_line_invariants`; second owner `_check_g8_line_id_uniqueness:893`.

Delete `seen_line_ids: set[str] = set()` at 299 and the four-line collision `elif`/`else` block at 310-313. Keep the missing/non-string/empty ID check at 308-309. Keep G8 unmodified.

Chain: FreezeCascade -> `run_gap_audit:706` -> per-line invariants at 719 and G8 at 728 -> Phase 0 audit receipt at 1017 (collects), Phase 10 at 1045 (refuses invalid freeze) -> structured `.errors`, logs and downstream verdict. G8 retains one capped summary, first five repeated occurrences plus `(+N more)`.

Payoff: five implementation lines and a second diagnostic owner removed. This intentionally changes error messages/counts for duplicate IDs; it does not change acceptance. The reduction depends on the number of repeated occurrences, not a constant one error. No in-repo exact-message consumers require the old freeze message; external log scrapers are unverified. Do not promise byte-identical error receipts. Public functions, schemas, phase ownership, missing-ID rejection, role checks and platform behavior stay intact. The similar duplicate check in `nodes/_otr_ledger_cleanup.py:490` is the writer's separate validation stage and stays. Cast char-ID uniqueness also stays.

### C3. Stop acquiring a model that the freeze cascade never calls (confirmed; highest behavioral risk)

File/symbol: `nodes/OTR_LedgerFreezeCascade.py:191`, `OTR_LedgerFreezeCascade.run`.

1. Keep `_OTRML` import: final teardown still uses it. Keep replay/no-ledger early returns at 215-263.
2. Replace 265-269 with a short accurate comment and `_OTRMI.require_model(technical_model, slot="technical")`, retaining nonempty wired-input validation but removing the unused `resolved_technical_id` binding.
3. Remove 270-287: imports of `policy_from_meta` and `load_config_from_meta`; `lfc_data`, `lfc_meta`, `lfc_policy`, `lfc_load_config`; `request_slot`; `make_generate_fn`.
4. Pass `None` as the existing first positional argument to `_LFC_ORCH.run_freeze_cascade` at 314. Preserve the public orchestrator signature and node schema.
5. Correct only adjacent comments that still claim an inline model mutation or that serialization must happen while this model is loaded. Do not refactor the two serialization blocks.

Chain: canonical node 1 output slot 4 -> link 115 -> FreezeCascade node 62 `technical_model` input 4 -> require_model -> currently request_slot/make_generate_fn -> `run_freeze_cascade:714` -> `_run_inline_safety_cleanup:676` -> `del generate_fn:692`. The cleanup implementation was retired by operator directive on Aug 5; it only stamps `same_story_safety_cleanup`. Other policies bypass that stub. No callback invocation survives. Freeze outputs feed the canonical ledger/audio/video consumers; the returned tuple remains seven items and script_json equals v2_ledger_json.

Payoff: about 18 setup lines plus no unnecessary acquisition/config plumbing on this path. `request_slot` can validate catalog/policy, choose a provider, admit local VRAM, reuse cache or load a model. Remote paths do not necessarily load local weights or evict residency; no GB/seconds savings are measured. Removing the call intentionally removes its catalog, policy/config, provider and VRAM failure paths. `require_model` validates a nonempty string only, not catalog membership. Blank wired input still fails on a normal current-ledger run. Replay/no-ledger behavior still returns before validation. No provider/platform lane is merged or modified.

Keep `_otr_freeze_cascade` receipts/policy selection/readiness/freeze behavior, capability telemetry, `vram_at_cascade_entry_gb`, public callback parameter, and final `unload_llm_if_local_resident` at 376. Writer teardown can be skipped by `OTR_WRITER_UNLOAD_AFTER_SCRIPT=0`; residual local residency therefore still needs the final gate. Keep unload failure stamping, exception propagation, script-text fallback, first JSON fallback, and post-finally JSON serialization/fallback. The telemetry value may change because the unused model is no longer acquired; its field stays.

Test coupling requiring an edit in the SAME C3 chunk: `tests/test_llm_runtime_policy.py:339`, `test_downstream_llm_consumers_thread_the_ledger_policy`, still asserts two freeze source strings at 346-347. Remove the freeze-source read/assertions, rename/reword the remaining check around the real shot-lock consumer, and preserve its `policy=policy_from_meta(meta)` expectation. Add the behavioral no-acquisition proof below. Never add dead text to satisfy obsolete source assertions.

## 2. Ownership and extraction decision

SceneSequencer owns scene-relative line timing; EpisodeAssembler owns promotion to master-mix coordinates; G8 owns freeze collision summaries; the writer owns its earlier duplicate validation; the model loader owns actual acquisition for callers that generate. Preserve those boundaries. Do not extract an offset helper just to make a copied harness convenient. The live assembler loop at `scene_sequencer.py:1896-1921` covers persisted legacy clips and remains in production. Music-row marker promotion at 1923+ has different zero-offset behavior and stays separate. No new wrapper or common engine abstraction is needed for C1-C3.

Expected runtime reduction is approximately 30 lines plus adjacent comments, not an audited repository total. New validation adds code. Retiring 228 lines of mirrored test code is reported separately from production cleanup; the new script/test currently total 312 lines.

## 3. Fresh evidence contract

The new audio check loads `workflows/otr_canonical.json` on each case, real NODE_CLASS_MAPPINGS, live INPUT_TYPES, FUNCTION and positional widgets; it verifies the selected edges, link fan-out/types and declared boundary producers. It calls real SceneSequencer -> AudioEnhance -> EpisodeAssembler, saves/loads a real ProductionLedger and writes/decodes a real master WAV. An independent correlation measurement locates the enhanced signal in the WAV; ledger positions must agree within one sample. Opening and no-opening cases check measured offsets, prior unmarked clips, already-master clips, missing clip time and repeated real assembly. Fresh output directories, canonical/source hashes and start/end HEAD prevent silent reuse of old receipts.

Boundaries are synthetic upstream voice/music assets and ledger, blank video policy and replay descriptor. This proves the non-foley/non-replay CPU audio segment only. It does not execute ComfyUI scheduling/IS_CHANGED, model generation, model cache behavior, foley, a full episode, GPU, or publishing. Full graph qualification is not inferred. The old 12-method offset test copied a loop and did not call the production implementation; removal does not mean every historical fixture has been requalified by two new cases.

At this base: focused fresh launcher passed (1 test, 9.75 s); measured offsets 1.50 s and 0.00 s. Full existing-suite run before retirement and full run after replacement both exited 2 with the exact SAME 54 failing node IDs; no additional failures. This is a relative comparison, not a green suite. Bug Bible: 22 passed, 27 skipped, 3 xfailed (exit 0). No reason to classify all old tests as stale. Failure baseline must be recaptured at implementation HEAD, not waived forever.

## 4. Implementation sequence and validation recipes

### P0. Intake and qualification (before runtime edits)

Read current rules/queue and this final plan. Record Windows branch/HEAD/status and canonical SHA. Coordinate one coder owner; preserve unrelated changes. Inspect actual files by symbol and ensure C1 stores remain unread, G8 still runs, and the cascade still never calls its callback. If any of these preconditions changed, stop that candidate and report the new consumer. Use the current Windows venv; never the lagging Linux mount. Inspect fresh preparatory harness changes/receipt and confirm they have landed before building on them.

No ComfyUI server, GPU run, external generation or paid panel is needed for this cleanup. Do not run old soak/cold-drive scripts as qualification. Do not edit canonical JSON: there is no intended node/widget/link change. Record byte-identical canonical before/after; run the existing validator/guardrails against that real file. Stop on schema drift instead of manufacturing a replacement graph.

### P1. C1 chunk

Edit only `nodes/scene_sequencer.py` as specified. Run the fresh canonical launcher test and input-type/signature parity tests. Compare deterministic WAV hashes and measured offsets against a newly captured pre-edit run under the same environment. Timing values, coordinate markers and repeated-assembly behavior must agree. No new test that merely asserts names are absent.

### P2. C2 chunk

Edit `nodes/_otr_ledger_freeze.py` and strengthen `tests/test_g8_line_id_uniqueness.py` with real function calls, not source-string copies:

- For two equal IDs, Phase 0 must collect exactly one collision diagnostic, the G8 summary. Phase 10 must still refuse and carry that summary in `.errors`; unrelated minimal-fixture errors may coexist.
- Missing, empty and non-string IDs must still produce per-line invalid-ID errors and no G8 collisions; include integer input as well as None.
- Eight occurrences of one ID -> exactly one G8 summary with `+2 more`; other independent invalid-role errors remain.
- Exercise the writer duplicate check through its existing test coverage; its diagnostic and behavior are unchanged.

Do not replace all freeze fixtures or count the whole error list as one. Expected changes are collision diagnostics only; no acceptance relaxation.

### P3. C3 chunk

Edit the freeze node and runtime-policy test exactly as C3. Extend `tests/test_lfc_b1_cascade_unload_in_finally.py` and/or `tests/test_cascade_freeze_unload_visible.py` in place for boundary-focused behavior:

- Replace successful `request_slot`/`make_generate_fn` stubs with fail-fast sentinels and assert zero calls on normal run, cascade exception and script assembly fallback. Record `run_freeze_cascade` receives None plus the same ledger and readiness toggles. Mocking the cascade here is acceptable only for fault injection/argument capture, not evidence that the real cascade never generates.
- Successful unload and failed unload: preserve verdict and seven outputs; returned JSON copies must match and expose the correct `freeze_unload_ok` boolean. Cascade exception must propagate while unload runs once; first/second serialization failures must retain documented fallback behavior.
- Add blank technical-model normal-run refusal plus replay/no-ledger no-acquisition checks, retaining their existing return contracts.
- Separately call the REAL `run_freeze_cascade` with a poison callback on inline and producer-owned policy routes using current valid fixtures. Do not mock `_run_inline_safety_cleanup` or phase dispatch to manufacture this proof. Assert callback untouched, same text, expected retired/not-applicable receipt and actual disposition. Reuse current fixture builders only after inspecting their schema against current production; do not trust their age or success label. This is CPU validation of the real deterministic cascade, not full canonical rendering.
- Bind the node invocation's readiness values/input ports to the real canonical node 62; assert the unchanged link 115 and actual NODE_CLASS_MAPPINGS class. Old mocked node tests remain supplemental. Do not create a separate fake workflow.

### P4. Every runtime chunk: acceptance, git and stop/rollback

Use `C:/Users/jeffr/Documents/ComfyUI/.venv/Scripts/python.exe`, UTF-8, CPU mask, pytest `-q -p no:cacheprovider`. Run focused tests, full repository suite and Bug Bible from its own root with relative `tests/bug_bible_regression.py`. Include dead-code plan's drift battery and `scripts/build_variants.py --check` where applicable; use exact current available commands, never guessed script names. Save commands, exits and failing-node-ID sets. Require focused checks green and zero NEW unexpected full-suite failures relative to an untouched same-HEAD baseline. Do not add failures to quarantine/EXPECTED_FAILED_NODEIDS to pass this wave. Existing unrelated failures remain separately disclosed; do not claim the full suite green.

Verify canonical SHA unchanged, AST parse and no empty/BOM touched files, `git diff --check`, exact scoped diff. Commit and push each qualified chunk together to `v2.0-alpha`, verify HEAD equals origin. Never edit pyproject/release metadata, tag/promote/publish, blanket reset or revert unrelated work. If HEAD changes, revalidate changed prerequisites. On a new regression, fix within the chunk or revert only this chunk's patch; after a pushed chunk use a scoped revert commit and push, never reset shared history. Stop if removing C3 acquisition leaves a real callback consumer, changes freeze acceptance/receipts beyond the declared removals, or requires a new architectural decision. Deliver exact net line count from the finished diff, not this estimate.

## 5. Review ownership, limits and coverage

Exactly two local Kibitz lanes: Antigravity and Cursor; driver Codex/Astra has no duplicate CLI lane. Run R1 approach -> R2 coding -> R3 wiring -> R4 convergence sequentially. Driver writes an anchor before each fan-out, verifies every claim, records judgment and feeds each final.md byte-for-byte into the next input. No nested delegation, extra workers, model upgrades, paid panels, code edits, tests, servers or GPU activity by reviewers. Antigravity may write only its designated review file; Cursor returns stdout. Read current real Windows files; a profile snapshot is context, not current truth. Cite file:line and exact symbols, consumer/output chains, compatibility, payoff/risk and confirmed/likely/unverified status. Zero defects is valid. Keep each review concise (under 1000 words), no file dumps. Review no more than 15 primary files; use targeted consumer searches and list uncovered work if this is insufficient. Existing applicable operator rules remain authoritative; this document adds no authority to override them.

Antigravity primary ownership (15 files): this plan; AGENTS.md; CLAUDE.md; docs/OTR_STANDING_RULINGS.md; docs/DEAD_CODE_HUNT_PROMPT_V5.md; docs/DEAD_CODE_EXECUTION_PLAN.md; nodes/scene_sequencer.py; nodes/audio_enhance.py; nodes/production_ledger.py; scripts/otr_canonical_audio_check.py; tests/test_canonical_audio_check.py; nodes/_otr_ledger_freeze.py; tests/test_g8_line_id_uniqueness.py; workflows/otr_canonical.json; __init__.py. Focus C1/C2 ownership and fresh audio evidence. Targeted search writer collision consumers; do not expand to other engines.

Cursor primary ownership (15 files): this plan; AGENTS.md; CLAUDE.md; docs/OTR_STANDING_RULINGS.md; docs/DEAD_CODE_EXECUTION_PLAN.md; workflows/otr_canonical.json; __init__.py; nodes/OTR_LedgerFreezeCascade.py; nodes/_otr_freeze_cascade.py; nodes/_otr_model_loader.py; nodes/_otr_model_inputs.py; tests/test_llm_runtime_policy.py; tests/test_lfc_b1_cascade_unload_in_finally.py; tests/test_cascade_freeze_unload_visible.py; tests/test_lfc_phase_7_8_readiness.py. Trace C3 through acquisition/cache, failure, final unload and recovery; focus exact implementation/test instructions. Targeted searches may locate callback consumers/fixture ownership but list larger unreviewed bodies.

Astra owns adjudication, required project lessons/newer receipts and closing remaining fixture/command details. Reviewed runtime sample: audio sequencing/enhancement/assembly, freeze diagnostics and unused acquisition route. Unreviewed: full model generation, scheduler cache/retry, all engine lanes, all legacy graphs, full publishing, broad audio cache teardown, other harness retirement. Preserve existing KEEP decisions without re-litigating them. Next bounded wave after these three chunks: trace one canonical artifact from generation to durable reuse and ask which work is repeated or discarded, with proof of all consumer/compatibility paths before proposing removal. No additional hunting in this implementation wave.
