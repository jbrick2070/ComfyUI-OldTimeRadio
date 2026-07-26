# B1b-0 post-code QA -- findings and what shipped

Target: `tests/test_ltx_8gb_graph_and_loads.py`, the regression net `ltx_8gb`
never had. Written the previous session, held back from origin because the QA
fan-out had not run. HEAD at entry `477f771f`. **No production code changed.**

Panel: three Sonnet lenses (decorative assertions / fidelity / over-pinning +
hygiene) and agy `Gemini 3.6 Flash (High)` via kibitz, all grounded through
Desktop Commander against the real Windows files. $0 external. Run:
`kibitz-runs/2026-07-26-b1b0-agy/r4/`; brief:
`docs/2026-07-26-b1b0-net-qa-brief.md`.

## The finding that mattered: the net could not observe the thing it was for

The file declared that B1b would FLIP two named assertions --
`test_the_graph_carries_ITS_OWN_loader_nodes_today` and
`test_THE_LOAD_COUNT_every_render_reloads_the_checkpoint_today` -- and that
everything else was a control. **Both declarations were wrong, and the Sonnet
over-pinning lens and agy reached that independently, without seeing each
other.** Grounded and confirmed against the decided design:

`_build_graph` stays CONDITIONAL under B1b -- it omits `ckpt` ONLY when the
caller supplied one through `external_results`, and still emits it when the
caller prepared nothing. Every test in the file either calls `_build_graph`
directly or hands `render_clip` a HAND-BUILT `prepared = {"patchers": []}`.
Neither carries externals, so both stay on the UNSUPPLIED branch forever. The
graph keeps its loader; the count stays 3.

So the file's flagship test -- the defect stated as a number -- was
structurally incapable of observing the hoist, and **no assertion anywhere in
it would have gone red against a hoist that silently did nothing.** Editing
the literal `3` to `1` when B1b lands would have produced a red that looked
like a broken hoist and was actually a harness gap.

That is the same failure class this project has now hit three times: a test
that is green, well-named, and describes work it does not do. What is new is
the scale -- previously one assertion, here the file's entire stated contract.

### The corrected contract, now written into the module docstring

- Every existing test is a CONTROL on the hoist. The load count STAYS 3. If it
  ever reads 1, `render_clip` has begun hoisting behind the caller's back --
  a worse bug than the one B1b fixes.
- **EXACTLY ONE assertion is expected to flip:**
  `test_the_executor_is_called_with_the_current_keep_contract`'s
  `"external_results" not in seen["kwargs"]`, because `render_clip` will
  forward the caller's externals on every call. The `keep` set itself must not
  move -- the executor unions the externals in.
- The 1-load proof is **owed by the hoist chunk**, which must call `prepare()`
  once and reuse the returned dict across several `render_clip` calls.

### And the mechanism is now proven ahead of the wiring

`test_the_forward_runs_with_the_checkpoint_supplied_EXTERNALLY` drives THIS
engine's real graph with the `ckpt` definition deleted and an already-produced
handle supplied through the landed B1a seam: the loader class never runs, the
downstream nodes receive the SUPPLIED handles, and the handle survives a
`free_after_use` pass. Mutating `keep |= set(ext)` out of `wrapper_bridge`
turns it red -- which is agy's r3 B1a finding, now pinned against a real
adapter graph rather than a synthetic fixture.

## The coverage hole: prompt polarity was never pinned

The Sonnet decorative-assertion lens found it and it is real at HEAD. The net
pinned every wire that reads the LOADERS and stopped there. Unpinned:
`cond.negative -> img2vid(1)`, `sample.positive -> cond(0)`,
`sample.negative -> cond(1)`, `decode.samples -> sample(0)`.

A positive/negative swap renders the negative prompt. It does not crash, does
not shorten the clip, and **no forward test in this file could ever see it** --
the fakes return canned tuples and never inspect what they were handed. That is
precisely the wrong-graph-shape defect the file's own docstring cites as the
reason it exists. Closed by `test_the_PROMPT_POLARITY_is_pinned_on_every_hop`;
three mutants prove it.

## Everything else that shipped

| finding | source | fix |
|---|---|---|
| Module docstring said "Five of its six sibling adapters have graph-shape tests" -- **all six do**, with file:line for each | Sonnet fidelity | corrected |
| `_ltx8_frame_length` had **ZERO coverage anywhere in `tests/`** -- and B3/B4 rest on it: once ping-pong is deleted, a non-`8n+1` segment is a hard RenderError | agy + independent grep | `test_the_frame_length_ladder_floors_caps_and_snaps_to_8n_plus_1` (floor, cap, snap-DOWN, plus a 199-case invariant sweep) |
| `clip["frame_count"] == 9` was decorative: the decode fake returned a fixed-size array, so `_ltx8_frame_length` could return anything | Sonnet decorative | the decode fake now sizes its batch from the `length` the graph actually asked `LTXVImgToVideo` for -- the count is an OBSERVATION |
| No test pinned that the resolver's values REACH the nodes (`steps`, `cfg`, shifts, `terminal`, `sampler`, seed, geometry) -- the build's recurring severed-channel class | agy | `test_the_RESOLVED_RECIPE_VALUES_reach_the_nodes_that_consume_them`, compared against the resolver, not literals |
| `_node_candidates` assertions skipped `pos` / `neg` / `loadimage` | Sonnet decorative | added |
| The `sample` fake was a 1-tuple; `SamplerCustom` returns `(output, denoised_output)`, and this repo's own sibling test models it as 2 | Sonnet decorative + fidelity | corrected |
| Every mp4-creating call sat OUTSIDE its own `try/finally`, so a raise on iteration 2 or 3 leaked the files from iterations 1-2 into the in-tree tmp tier | Sonnet hygiene | creation moved inside the `try` in all four forward tests |
| `test_the_harvest_registers_a_REUSED_handle_exactly_once` claimed to "drive the POST-HOIST condition directly" -- it simulates it through today's path | Sonnet decorative | docstring corrected |

Verified TRUE and left alone: the 19-entry `_ENVS` list matches every
`OTR_LTX_8GB_*` variable production reads, exactly; `device == "cpu"` is the
real default; the `keep` / `free_after_use` contract matches the call site;
`Wire("ckpt", 2)` is the embedded VAE; every other fake's arity matches both
the real node class and the slots the graph reads; the file is pure ASCII with
no BOM; monkeypatch scoping and ffmpeg skips are clean.

## Mutation proof: 19 mutants, 17 defect + 2 CONTROL, all PROVEN

`tmp/_kbA_b1b0_mutate.py`, rewritten this chunk. Both baselines asserted
`failed=0` first, and failure ids are parsed from BOTH the `FAILED path::name`
and `  - path::name` shapes -- a harness blind to the second reported
`failed=0` for four live mutants this week.

The two CONTROL mutants are the new half. They change values the recipe is
ENTITLED to change -- the default step count, the default checkpoint basename --
and must break NOTHING. They are what proves the new
`test_the_RESOLVED_RECIPE_VALUES_...` compares against the resolver rather than
secretly pinning literals: a tightening that also refuses honest input is not a
fix. Both broke nothing.

New mutants: `POLARITY_sampler_reads_the_negative_as_positive`,
`POLARITY_cond_negative_reads_the_positive_slot`,
`DECODE_reads_the_denoised_latent`, `RECIPE_steps_crossed_with_max_shift`,
`RECIPE_length_hardcoded_at_the_node`,
`RECIPE_seed_never_reaches_the_sampler`, `LADDER_snaps_UP_instead_of_down`,
`LADDER_floor_removed`, `CANDIDATE_pos_class_renamed`,
`EXTERNALS_not_added_to_keep` (in `wrapper_bridge`, the first cross-file mutant
this harness has carried).

## Gate

Full Windows suite **7087 passed / 27 skipped / 1 xfailed** (7083 -> 7087: the
four new tests). Bug Bible 17 passed / 24 skipped / 3 xfailed. AST parse, no
BOM, no zero-byte, ASCII on all eleven touched paths. Canonical
`9872624A` byte-identical -- this chunk touches no node, widget, link or
schema.

## Still open, deliberately

- Nothing here exercises `Ltx8gbEngine.prepare()`, because it does not exist
  yet. The 1-load proof is B1b's to write and is named in the module docstring
  so the next window cannot miss it.
- The fakes still ignore their kwargs everywhere except `img2vid`'s `length`
  and the three spied nodes in the external-supply test. That is a deliberate
  split of labour with the static graph-dict assertions -- but it means any
  NEW wire must be pinned statically, never assumed covered by a forward test.
- `SamplerCustom`'s real arity could not be read off this box (no
  `comfy_extras` under the ComfyUI root); the 2-tuple is corroborated by this
  repo's own `sampleradv` fake in `tests/test_video_motion_forward.py`.
