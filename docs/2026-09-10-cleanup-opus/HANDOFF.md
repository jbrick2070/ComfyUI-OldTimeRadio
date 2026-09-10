# Opus handoff: clean audio and bounded cleanup

Status: R1-R4 converged; ready for the three bounded Opus implementation chunks. Runtime cleanup has not been applied. The preparation and this handoff ship together in the commit named "test: qualify fresh canonical audio check and cleanup handoff"; verify that commit is present on origin/v2.0-alpha before P1.

Qualified Windows base: `1f27e4123a9567400ab39ef698d2a8a394601284`; closing pre-commit HEAD `b9d7be0863788e2507de5e68f9273e61c27268ba` adds only an unrelated My Story scope document. Branch `v2.0-alpha`. Recheck HEAD and ownership before editing. This is a bounded sample, not a complete repository audit.

## Pending implementation, in order

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

## Copyable Opus instruction

Read AGENTS.md, CLAUDE.md, the current GO_FORWARD ownership/queue, and docs/2026-09-10-cleanup-opus/PLAN.md plus qualification.json. Implement P1, P2 and P3 as the three scoped chunks specified there. Begin by verifying the preparatory commit is pushed and revalidate candidate preconditions at current Windows HEAD. The operator wants clean audio: remove the automatic noise sources and make the real canonical enhancement defaults dry; preserve useful rate/channel conversion, levels, timing, public node contracts and independent engine lanes. Extend the fresh canonical check in place; do not preserve obsolete runtime work to satisfy stale diagnostics. Run the declared focused/full/Bug Bible/structural checks and compare unexpected failures with a fresh same-HEAD baseline; do not quarantine new failures. Obtain one grounded finished-diff CLI review per code chunk, commit and push each qualified chunk to v2.0-alpha, and verify HEAD equals origin. Preserve unrelated files. No model upgrades, nested workers, paid panels, GPU/server runs, release changes or artifact/cache deletion. Stop after these chunks and their receipts; do not expand the hunt. Report exact runtime/test line deltas, evidence, and any remaining production listening limitation.

## Coverage limits and next bounded wave

Reviewed: the supplied-audio CPU route and its helper/test consumers; freeze duplicate ownership; freeze acquisition, cache-admission consequences, failure/finally and replay boundaries. Static compatibility checks cover public registration and both roomtone CPU/CUDA branches. No GPU, model generation, full Comfy scheduler/cache, foley render, video appearance, published episode, legacy graph fleet or repository-wide dead-code audit was performed.

After this cleanup, the next bounded review should follow one canonical generated artifact into durable reuse, looking for work repeated or discarded and retiring only what has no useful consumer. The wider cache migration and remaining harness inventory are separate work.

Optional speaker placement is also a separate future design/listening topic. Support more than two speakers with no five-speaker cap. The operator is considering bounded randomness or presets; no algorithm is approved. Astra proposes subtle balanced positions assigned by stable speaker identity once per episode, centered narration, and no position changes between lines or when later speakers enter. Define persistence/replay and mono compatibility before implementation. Do not include panning in P1 or label it coding/wiring-reviewed.
