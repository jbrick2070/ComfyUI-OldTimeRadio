# Opus handoff: clean audio and bounded cleanup

Status: DONE 2026-09-10. All three chunks are implemented, reviewed, committed and pushed to origin/v2.0-alpha: P1 `cc09b54a`, P2 `7aa46655`, P3 `e562e146`. Codex owns sprints 3-5 from HEAD `e562e146`. The result section below is the closing receipt; the sections after it are the pre-implementation contract kept for the record. The preparation shipped in "test: qualify fresh canonical audio check and cleanup handoff" (a3551ff5).

Qualified Windows base: `1f27e4123a9567400ab39ef698d2a8a394601284`; closing pre-commit HEAD `b9d7be0863788e2507de5e68f9273e61c27268ba` adds only an unrelated My Story scope document. Branch `v2.0-alpha`. Recheck HEAD and ownership before editing. This is a bounded sample, not a complete repository audit.

## Implementation result (2026-09-10)

| Chunk | Commit | Runtime lines (nodes/) | Evidence |
|---|---|---|---|
| P1 / C4+C1 clean audio | `cc09b54a` | +73/-199 | Three-case fresh canonical CPU check passed: chirp opening 1.50 s, chirp no-opening 0.00 s (both unchanged from pre-edit evidence), silence 0/0/0 nonzero samples at scene/enhanced/master; clean enhancement equals the mono scene duplicated; asymmetric stereo through function defaults is bit-identical; all four tape modes leave silence at zero (subtle alone produced 95,998 nonzero samples before). Canonical sha 9ab0abe6 -> b24221b6, only node 4 widgets and one description word; validator 23 nodes / 62 links / drift 0. |
| P2 / C2 G8 ownership | `7aa46655` | +7/-10 | 115 focused tests passed; two equal ids yield exactly one G8 summary at Phase 0 and Phase 10 (freeze still refused); None/""/7 ids stay per-line errors; eight occurrences yield one "+2 more" summary. Canonical bytes unchanged. |
| P3 / C3 freeze no-acquisition | `e562e146` | +65/-67 | Fresh-process test on both banks runs the real registered node against canonical node 62 (link 115, widgets True/True) with the package-qualified loader poisoned: zero acquisition calls, frozen_with_warns, freeze_unload_ok true, cleanup status retired_no_content_policy / not_applicable_content_owned, real run_freeze_cascade with a poisoned callback never invokes it. Negative control: the same child against the pre-P3 commit aborts with "freeze must not call request_slot". Slot sweep 43 -> 42 sites, floor 12. |

Runtime total across the three chunks: +145/-276 (net -131) lines in `nodes/`; test/check lines are reported separately in `implementation_receipt.json`. Full suite: a fresh same-HEAD baseline (14166 collected, exit 2, 54 unexpected failing node ids) and every post-chunk run (14166 / 14171 / 14180 collected) carry zero new failures (P1 and P2 the identical 54 ids; the P3 run had one order-dependent baseline failure, the catalog disk-space precheck, pass -- it passes in isolation and no catalog file was touched); the suite is not green and nothing was quarantined. Bug Bible after each chunk: 22 passed, 27 skipped, 3 xfailed. `scripts/build_variants.py --check`: no committed variants.

Reviews (one finished-diff reader per chunk, CLAUDE.md 2026-09-07): P1 cursor-agent, no ship blocker (two stale-prose findings folded in); P2 cursor-agent, no ship blocker; P3 internal Claude reviewer subagent grounded on the real files (substitute: the cursor-agent lane returned a one-line preamble with exit 0 on its first run) plus a cursor-agent retry, both no ship blocker; their module-identity hardening, stale docstring, comment wording and peek-None findings were folded in before commit.

Remaining production listening limitation: only the CPU synthetic-boundary segment was executed. No ComfyUI server, GPU, model generation, foley mux, video render, published episode or production listening happened in this wave. Existing masters, cached node outputs and replay bundles keep their original sound; the first normal fresh non-replay assembly is where the clean path is heard. The next bounded review named below is unchanged.

## Implementation chunks as planned (completed; kept for the record)

| Chunk | Confirmed eligible work | Intended result |
|---|---|---|
| P1 / C4+C1 | Retire automatic roomtone and tape hiss; clean canonical enhancement defaults; seven unused local stores | Supplied voice/music without added noise or coloration, preserving episode levels and timing |
| P2 / C2 | Remove the second freeze line-ID collision reporter | G8 owns collision summaries; invalid input still fails |
| P3 / C3 | Remove freeze's unused model acquisition | Deterministic freeze no longer loads/adopts a model it never calls; final conditional unload and recovery remain |

Each candidate's exact files, symbols, consumer chain, public/platform implications, edits, tests and stop conditions are in PLAN.md. C4 retires active behavior by operator choice; it is not claimed dead code. Approximately 158 gross runtime lines are candidates for deletion across this sample, before prose/test additions. Actual net removal is measured after implementation.

The main audible change is confirmed in source and a fresh CPU diagnostic: exact zero voice inputs became nonzero at SceneSequencer, and subtle tape independently added noise to zero input. All-off AudioEnhance preserved zero. Existing episode WAVs and replays retain their original sound; a normal fresh assembly is required. This does not promise denoising of noise already in source assets.

## Preparation and evidence

- Fresh `scripts/otr_canonical_audio_check.py` calls actual registered canonical SceneSequencer -> AudioEnhance -> EpisodeAssembler, uses real ledger persistence and written WAVs, and independently measures opening placement.
- `tests/test_canonical_audio_check.py` launches it in independent fresh processes. The copied 228-line `tests/test_episode_assembler_offset_shift.py` is retired. New preparation totals 331 lines, a net increase of 103 test/check lines; do not count this as runtime removal.
- Pre-clean-audio focused result: 1 passed in 12.11 s. Full suite before/after: same 54 unexpected failing node IDs, zero new failures; latest collection 14166, exit 2. Bug Bible: 22 passed, 27 skipped, 3 xfailed, exit 0. This is not a green full suite.
- Actual canonical validator: 23 nodes, 62 links, zero widget-vector drift. Variant check reports no committed variants, not a multi-variant render proof.
- `qualification.json`, `canonical_audio_receipt.json`, `audio_noise_before.json`, `freeze_fixture_receipt.txt` and `slot_sweep_accounting.json` hold the compact evidence. Clean-audio and no-acquisition node regression tests in PLAN.md are planned work, not already-passed results.
- `campaign_receipt.json` records the complete arc: C1-C3 R1-R3, late clean-audio C4 R1-R3 and shared integrated R4. Two fixed local lanes, 14 actual review calls, no additional workers/model upgrades/paid panel. Astra verified claims, corrected the stale canonical prohibition and rejected misreads. The final preparation also received its one finished-code CLI review. Reviewers expose no usage/cost receipt, so no spend number is claimed.

## Copyable Opus instruction (executed 2026-09-10)

Read AGENTS.md, CLAUDE.md, the current GO_FORWARD ownership/queue, and docs/2026-09-10-cleanup-opus/PLAN.md plus qualification.json. Implement P1, P2 and P3 as the three scoped chunks specified there. Begin by verifying the preparatory commit is pushed and revalidate candidate preconditions at current Windows HEAD. The operator wants clean audio: remove the automatic noise sources and make the real canonical enhancement defaults dry; preserve useful rate/channel conversion, levels, timing, public node contracts and independent engine lanes. Extend the fresh canonical check in place; do not preserve obsolete runtime work to satisfy stale diagnostics. Run the declared focused/full/Bug Bible/structural checks and compare unexpected failures with a fresh same-HEAD baseline; do not quarantine new failures. Obtain one grounded finished-diff CLI review per code chunk, commit and push each qualified chunk to v2.0-alpha, and verify HEAD equals origin. Preserve unrelated files. No model upgrades, nested workers, paid panels, GPU/server runs, release changes or artifact/cache deletion. Stop after these chunks and their receipts; do not expand the hunt. Report exact runtime/test line deltas, evidence, and any remaining production listening limitation.

## Coverage limits and next bounded wave

Reviewed: the supplied-audio CPU route and its helper/test consumers; freeze duplicate ownership; freeze acquisition, cache-admission consequences, failure/finally and replay boundaries. Static compatibility checks cover public registration and both roomtone CPU/CUDA branches. No GPU, model generation, full Comfy scheduler/cache, foley render, video appearance, published episode, legacy graph fleet or repository-wide dead-code audit was performed.

After this cleanup, the next bounded review should follow one canonical generated artifact into durable reuse, looking for work repeated or discarded and retiring only what has no useful consumer. The wider cache migration and remaining harness inventory are separate work.

Optional speaker placement is also a separate future design/listening topic. Support more than two speakers with no five-speaker cap. The operator is considering bounded randomness or presets; no algorithm is approved. Astra proposes subtle balanced positions assigned by stable speaker identity once per episode, centered narration, and no position changes between lines or when later speakers enter. Define persistence/replay and mono compatibility before implementation. Do not include panning in P1 or label it coding/wiring-reviewed.
