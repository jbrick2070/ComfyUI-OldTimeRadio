# B1b -- the beat-scoped checkpoint hoist: what shipped and what QA found

HEAD at entry `b214481b`. Pre-code authority: `docs/2026-07-26-8gb-1080p-arc-judgment.md`
(kibitz r1-r4, 8 agent calls) -- the design was already panel-decided, so this
chunk ran the fan-out AFTER the code, not before. Post-code panel: two Sonnet
lenses (resource ownership; decorative-assertion hunt) and agy
`Gemini 3.6 Flash (High)` via kibitz. Brief:
`docs/2026-07-26-b1b-hoist-qa-brief.md`. Run:
`kibitz-runs/2026-07-26-b1b-agy/r4/`. $0 external.

## The defect

`Ltx8gbEngine.load()` resolved node CLASSES only, and the segment graph carried
its own `CheckpointLoaderSimple`, so every segment of a multi-segment beat
re-read a 6.34 GiB checkpoint. `BeatSession` has said "ONE prepare/load per
beat" since chunk 5; for this adapter that was the design and not the
behaviour.

## What shipped

1. **`_assert_checkpoint_integrity(ckpt)`** -- the 4 GiB floor lifted out of
   `assert_usable` into a shared helper, same reason code, same message, same
   ordering. **This was the blocker the pre-code panel found.** `assert_usable`
   runs PER SEGMENT inside `render_driver._render_one`, which is AFTER
   `BeatSession` opens, so moving the real load into `prepare()` put it ahead
   of the only size check in the adapter. `resolve_session_config` does not
   close the gap: it proves the file EXISTS and takes its receipt, never its
   size.
2. **`prepare()` override.** Frozen config -> floor -> `super().prepare()`
   (lease + classes) -> a loader-only mini-graph through
   `run_graph(..., on_result=_register)` whose results become
   `prepared["external_results"]`. Refuses a checkpoint returning fewer than
   three outputs, by name. On any failure: `teardown(prepared)` wrapped in its
   own handler, then re-raise.
3. **`_build_graph(..., external_results=None)`** -- omits the definitions of
   ids the caller supplied, keeps every wire. Omitting is not an optimisation:
   `run_graph` REFUSES a graph that also defines an id the caller supplied.
4. **`render_clip`** forwards the externals to both `_build_graph` and the
   executor.
5. **`teardown()`** drops `external_results` before delegating.

**The floor check runs BEFORE `super().prepare()`, not after.** That is the
C-3 lesson from two sessions ago made structural: a raise after the
cross-process lease is taken strands it for the life of the server, and the
owner is the live ComfyUI process, so the PID-liveness reclaim cannot help.
Two mutants prove the position, not just the presence.

## What the panel found

**No must-fixes.** Three independent seats, no disagreement on the design.
Everything below is real and was fixed in this chunk.

| finding | source | fix |
|---|---|---|
| `except Exception` around the failure-path `teardown()` call -- a `KeyboardInterrupt` landing inside teardown would REPLACE the original cause, and `beat_session.py` uses `except BaseException` for exactly this | Sonnet lens 1 | now `except BaseException` |
| The `teardown` docstring claimed the cleared dict covered "the MODEL and the embedded VAE", which reads as though teardown reclaims both. It does not: slot 2 has never been handed to `_detach_patchers`, here or in any sibling | Sonnet lens 1 | docstring now states the scope and names it a family-wide separate ticket |
| `_register` harvests slot 0 with no stated reason -- ambiguous for the next maintainer | agy | documented: the MODEL is the `ModelPatcher` teardown knows how to unwind, and the duck-typed `detach` check is what keeps that honest rather than a positional assumption |
| The harvest test's `teardown()` sat OUTSIDE its own `finally`, so a mid-loop failure skipped the four teardown assertions | Sonnet lens 2 | moved inside |
| No assertion that the harvested patcher IS the object the segments render through -- a copy would detach something no render ever used | agy | `assert ckpt_model is prepared["external_results"]["ckpt"][0]` |

**Verified clean and worth not re-litigating:** every raise point in
`prepare()`/`teardown()` (floor, `super().prepare()`, the mini-graph, a raising
`on_result`, the arity refusal, a raising `unload()`) releases the lease --
the base teardown's release sits in a `finally`. `on_result` fires in the same
loop iteration that produces the handle, so there is no window between "loaded"
and "owned". `teardown` is idempotent (`_detach_patchers` empties the list;
`release()` self-guards). Nothing retains the externals after the pop --
`seen` holds `id()` integers, not references.

**Raised, out of scope, recorded here so it is not lost:** `MotionEngineBase`
has no re-entrancy guard, so a second `prepare()` on one engine instance with
no teardown between blocks the full 120s lease timeout rather than failing
fast -- the owner PID is this same live process, so the stale-lock reclaim
never fires. Pre-existing and shared by every engine; `BeatSession.open()`
already guards its own path. It belongs with the untracked-VAE item in a
family-wide ticket, not in an `ltx_8gb` chunk.

## Mutation proof: 29 mutants, 27 defect + 2 CONTROL, all PROVEN

`tmp/_kbA_b1b0_mutate.py`. Both baselines asserted `failed=0` first; failure
ids parsed from BOTH the `FAILED path::name` and `  - path::name` shapes.

Ten mutants are new for the hoist. Two are worth naming because they prove the
corrected control contract from B1b-0 actually has teeth:
`HOIST_omits_the_loader_even_when_nothing_was_prepared` turns
`test_the_graph_carries_ITS_OWN_loader_nodes_today` red -- the test the
previous draft wrongly declared would FLIP is now doing its real job, catching
a hoist that breaks callers who prepared nothing. And
`FLOOR_runs_AFTER_the_lease_is_taken` / `FLOOR_dropped_from_prepare_entirely`
pin the floor's POSITION rather than its presence.

`OWNERSHIP_taken_after_the_graph_instead_of_during` is the sharpest: it drops
`on_result=_register`, the arity refusal still fires, and the only thing that
changes is that a loaded 6.34 GiB MODEL is never detached. One assertion
catches it.

The two CONTROLs -- the recipe moving its own step count, the default
checkpoint being renamed -- broke nothing, which is what keeps the new
assertions from being literal pins.

## Gate

Suite **7097 passed / 27 skipped / 1 xfailed** (7087 -> 7097: ten new tests).
Bug Bible 17 passed / 24 skipped / 3 xfailed. AST, no BOM, no zero-byte, ASCII
on all eleven touched paths. Canonical `9872624A` byte-identical -- no node,
widget, link or schema moved.

## What B1b does NOT prove

No GPU leg has run. The one-load count is proven against fakes through the
real `run_graph`, the real `_topo_order`, the real `prepare`/`teardown` and the
real graph builder -- but the 6.34 GiB file has never been loaded once per beat
on the 5080. That is 7d's job, and the VRAM probe must start BEFORE
`BeatSession` opens or it will miss the load peak entirely now that the load
happens in `prepare()`.

The T5 remains per-segment by decision, so a beat still pays its ~9 GB
`CLIPLoader` cost once per segment. The CPU-mode T5 hoist is a later chunk,
gated on a measured peak-RAM number for a 4-segment beat.
