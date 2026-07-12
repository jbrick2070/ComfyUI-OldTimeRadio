# Problem statement: training-fidelity telemetry capture for codex56sol LLM calls

2026-07-12. Operator-approved direction ("add telemetry if it works and is
not a big performance hit"). Successor to the do-now items in
kibitz-runs/2026-07-12-contract-adapter/r1/final.md (M3) after the
contract-adapter idea was PARKED (contracts never freeze; revive only on a
stable defect class). This capture is what makes "stable defect class"
measurable at all.

## Why

`_call` in `nodes/_otr_original_codex56sol.py` (:661-770) journals
raw-response hashes plus pass/slot, but NOT: the exact messages sent, the
raw rejected output, the exact validator error, which deterministic
projection fired and what it changed, or the accepted artifact at the
moment of acceptance (accepted artifacts land only after lane completion,
:1575-1587). Consequences today: prod-bug forensics reconstruct prompts by
hand; seam tunes are argued from memory instead of counted defects; and no
future adapter/constrained-decoding census is possible. This telemetry is
useful for bug forensics and seam tuning EVEN IF no adapter is ever
trained.

## What to capture (one record per call ATTEMPT)

- identity: invocation_id, episode/lineage id, bank id, pack version +
  pack hash, pass_id, seam, slot, attempt number (base call vs repair N).
- model identity: repo_id + resolved commit sha (NOT the mtime-picked
  snapshot alone -- record the sha `resolve_snapshot_dir` landed on,
  `nodes/_otr_hf_env.py:132-150`), quant policy, backend
  (transformers/gguf/openrouter), temperature, max_new_tokens, and the
  OTR_CAST_SEED / OTR_STYLE_SEED / OTR_ORIGINAL_SEED env pins if set.
- payload (content-addressed): sha256 + blob for (a) exact messages
  (system+user as sent), (b) raw model output. Blobs go to a dedupe store
  keyed by hash -- seam prefixes repeat massively across calls, so
  dedupe collapses most of the volume.
- outcome: parse ok / json error; pydantic schema errors (exact);
  post_validator error string (exact); deterministic projection applied
  (which repair fn + a minimal field-level diff); accepted bool; final
  acceptance hash when the lane completes (backfilled once, by
  invocation_id).
- timing: wall latency, prompt/output token counts if available.

## Hard constraints

1. FAIL-OPEN. Every telemetry write wrapped so an exception logs one
   warning and continues the render. Telemetry can be lossy; renders
   cannot. A telemetry-caused lane abort is a P0 self-inflicted bug.
2. SIDECAR PATH, never the episode ledger. Default root
   `otr/telemetry/<bank>/<YYYY-MM-DD>/` (JSONL append + blobs/ subdir).
   No reads or writes to episode ledger files from the telemetry path
   (WinError 5 per-beat-save contention class). Append-only, buffered,
   no fsync-per-record.
3. PERF BUDGET: < 0.5% of call wall time. Given 100-200 s NF4 decodes and
   millisecond-scale hash+append, expected real cost is < 0.1%. Assert
   order-of-magnitude in a test with a stub generate fn (telemetry
   overhead per call < 50 ms).
4. OTR_TEST_MODE: capture OFF by default under tests; explicit opt-in env
   (OTR_TELEMETRY=1 forces on; =0 forces off; unset = on in production
   lanes only).
5. UTF-8, no BOM. SFW. Records are self-describing JSON with a
   schema_version field from day one.
6. Wiring surface: the capture hooks live in/around `_call` (single choke
   point for all nine passes) plus one backfill call at lane completion.
   No new node, no new widget, NO workflow JSON change expected. If any
   config lever is exposed as a widget after all, it appends at the END of
   widgets_values with the full section-0 re-validation ritual.
7. Scope: codex56sol lane first (it has the bakeoff traffic). The hook
   shape should be liftable to other lanes later, but do NOT generalize
   in v1.

## Acceptance

- 30-word live smoke: episode renders normally, telemetry dir contains
  one record per LLM attempt with resolvable blob hashes, and a forced
  telemetry-write failure (e.g. read-only dir) still yields a green
  render with one warning line.
- Regression suite + Bug Bible green. New tests: fail-open behavior,
  OTR_TEST_MODE default-off, record schema round-trip, dedupe store
  hit path, perf budget stub test.
- Retention note in the README of the telemetry dir (safe to delete any
  day-folder; blobs are re-derivable only from future runs).

## Non-goals

- No adapter training, no census analytics, no dashboards in this change.
  Those consume this data later.
- No capture of operator-personal data; payloads are model prompts and
  outputs only.
