# B1b-0 post-code QA brief -- the `ltx_8gb` graph-and-loads regression net

Review target: `tests/test_ltx_8gb_graph_and_loads.py` (419 lines, UNCOMMITTED
at HEAD `477f771f`). Production under review with it:
`nodes/_otr_video_engines/eng_ltx_8gb.py` and
`nodes/_otr_video_engines/wrapper_bridge.py`.

This chunk changed NO production code. It is a test file only.

## What the net is for

`ltx_8gb` is the 8 GB video tier. It had no graph-shape test and no test that
drove `render_clip`: a wrong graph shape turned nothing red and surfaced on a
live GPU render. The next chunk (B1b) hoists `CheckpointLoaderSimple` out of
the per-segment graph so a multi-segment beat loads the 6.34 GiB checkpoint
ONCE. Landing that on top of no coverage would put the engine's first graph
tests and the codebase's first `prepare()` override in one diff.

## The B1b design, already decided -- do not propose alternatives

1. `Ltx8gbEngine.prepare()` overrides the base: `super().prepare()` first, then
   a LOADER-ONLY mini-graph through `wrapper_bridge.run_graph(..., terminal=None)`;
   the returned results dict IS `prepared["external_results"]`.
2. The CHECKPOINT ONLY is hoisted. The T5 / `CLIPLoader` is deliberately NOT
   hoisted (the pos/neg encodes stay per-segment either way, so hoisting the
   loader buys wall-clock only while pinning ~9 GB for the whole beat).
   `modelsampling` is NOT hoisted -- it is a cheap per-segment clone+patch.
3. `_build_graph` stays CONDITIONAL: it omits the `ckpt` node ONLY when the
   caller supplied one through `external_results`, and still emits it when the
   caller prepared nothing. Sibling tests hand-build `prepared={"patchers": []}`
   and call `render_clip` directly.
4. The checkpoint patcher is harvested into `prepared["patchers"]` INSIDE
   `prepare()`. `prepared["external_results"]` is cleared before delegating to
   the base teardown.

## What to review

Adversarial review of the TEST FILE. Assume every test is decorative until a
mutation kills it. This project has twice shipped a green, well-named test that
proved nothing -- one claimed to detect a branch swap and would have passed
under the exact edit it claimed to detect.

1. **Decorative assertions.** For each test, name the single-line production
   mutation it claims to catch, and say whether the test would actually go red.
   Call out assertions that are true by construction of the fixture.
2. **Fidelity.** Every claim in a docstring, comment or constant must be TRUE
   against production at HEAD. Quote the contradicting production line.
3. **Coverage holes.** Which parts of `_build_graph`'s wiring are NOT pinned by
   any assertion in this file? Name them.
4. **Over- and under-pinning against the hoist.** Given the decided design
   above, which assertions would go red under a minimal correct hoist -- and,
   just as important, which assertions CLAIM they will flip but structurally
   cannot, because the test never exercises the mechanism that would change
   them?
5. **Hygiene.** Temp-file leakage, global state, cross-test pollution, env
   isolation, clean skips without ffmpeg.

Report findings with file:line and a concrete one-line fix each. Do not rewrite
the file.
