# B1b post-code QA brief -- the beat-scoped checkpoint hoist

Review target: the UNCOMMITTED change at HEAD `b214481b` to
`nodes/_otr_video_engines/eng_ltx_8gb.py` and the tests added to
`tests/test_ltx_8gb_graph_and_loads.py`. Green: focused 26/26, 29 mutants
proven (27 defect + 2 CONTROL), full suite pending.

## What shipped

`Ltx8gbEngine.load()` only resolved node CLASSES, and the segment graph carried
its own `CheckpointLoaderSimple`, so every segment of a multi-segment beat
re-read a 6.34 GiB checkpoint. "One model load per beat" was the design and not
the behaviour.

1. **`_assert_checkpoint_integrity(ckpt)`** -- the 4 GiB floor extracted from
   `assert_usable` into a shared helper. Same reason code, same message, same
   ordering. It matters because `assert_usable` runs PER SEGMENT inside
   `render_driver._render_one`, which is AFTER `BeatSession` opens, so hoisting
   the load into `prepare()` moved the real load ahead of the only size check.
   `resolve_session_config` proves the file EXISTS and takes its receipt, never
   its size.
2. **`prepare()` override.** Resolve the frozen config, check the floor, THEN
   `super().prepare()` (which takes the cross-process GPU lease and resolves
   classes), then execute a loader-only mini-graph through
   `run_graph(..., on_result=_register)` whose results become
   `prepared["external_results"]`. Refuse a checkpoint returning fewer than 3
   outputs. On any failure: `teardown(prepared)` wrapped in its own
   try/except, then re-raise.
3. **`_build_graph(..., external_results=None)`** -- omits the definitions of
   ids the caller supplied, keeps every wire.
4. **`render_clip`** forwards `prepared["external_results"]` to both
   `_build_graph` and `run_graph`.
5. **`teardown()`** pops `external_results` before delegating to the base.

## Decisions already made -- do NOT propose alternatives

- CHECKPOINT ONLY. The T5 `CLIPLoader` is deliberately NOT hoisted: the
  pos/neg encodes are per-segment either way, so hoisting it buys wall-clock
  while pinning ~9 GB for the whole beat -- a guaranteed OOM on an 8 GB tier.
  `ModelSamplingLTXV` is a cheap per-segment clone and stays.
- `_build_graph` stays CONDITIONAL. A caller that prepared nothing (the
  single-clip path, which is what production runs today) renders exactly as
  before.
- The per-render patcher harvest in `render_clip` still names `ckpt`. It is a
  no-op on the hoisted path (the id-dedupe skips it) and load-bearing on the
  unprepared one.

## What to review

Assume every new test is decorative until a mutation kills it. This project
has repeatedly shipped green, well-named tests that proved nothing.

1. **Leaks and ownership.** Is there any path where a loaded MODEL is not
   registered for teardown, or where the GPU lease is taken and not released?
   Walk every exception path in `prepare()` and `teardown()`, including a
   raising `unload()`, a raising `on_result`, and a failure between `prepare()`
   and the first `render_clip`.
2. **The unprepared path.** Does anything here change behaviour for a caller
   that passes `prepared={"patchers": []}`? That path is production today.
3. **Correctness of the omission.** Can a graph end up with a wire whose source
   is neither a node nor an external? Can an id be both?
4. **The floor.** Is the ordering claim true -- is there any way the real load
   now precedes the size check? Does `assert_usable` still behave identically?
5. **Test fidelity.** The new fixture patches `_resolve_model_file`,
   `wrapper_bridge.resolve_graph_classes`, three lease functions and lowers
   `_LTX8_CKPT_MIN_BYTES`. Does any of that hide the behaviour under test?
6. **Anything the mutation harness cannot reach** -- state changes with no
   observable assertion.

Report findings with file:line and a concrete one-line fix. Do not rewrite the
files.
