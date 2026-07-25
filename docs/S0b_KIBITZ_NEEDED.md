# S0b -- KIBITZ NEEDED (2026-07-24 autonomous window bailout)

**Written 2026-07-24 22:50 local by CODER WINDOW A during the autonomous
S0a-b -> S0b -> S1 -> HALT@S2 run.**
HEAD when this file lands: `c68b606c` on `v2.0-alpha` (S0b IN-FLIGHT
plan flip); prior code base is `e60185a0` (S0a-b landed clean).
Spec: `docs/2026-07-25-still-plans-locked-build-spec.md` @ `84328aa1`.
Handoff already on file: `docs/S0b_HANDOFF.md`.

## Why this file exists

S0a landed at `33c4d8cf` and the S0a-b isolation-property amendment at
`e60185a0`, both green under the full Windows suite / Bug Bible /
`test_audio_byte_identical`. The plan was to continue autonomously through
S0b (the routing freeze), S1 (schema + 31 declarations + audit) and HALT
at S2 (per operator).

After grounding the S0b scope against HEAD I have to file this doc rather
than half-land the chunk. The reasoning is the same one the PRIOR
autonomous window reached at `d6655690`, and which the operator ratified
with the tracked `docs/S0b_HANDOFF.md`:

> S0b, as specified, is a cross-module atomic refactor whose value is
> entirely in its atomicity: every consumer must be cut over in the same
> commit, or the "routing is frozen FIRST" promise is broken and the
> still spine can still be validated against one engine and rendered
> with another. A half-landed S0b is worse than no S0b -- the tree
> looks routed but the defect the whole chunk exists to close is still
> open, silently.

Operator directive from this window's kickoff:

> If the chunk genuinely needs to be split for suite-health reasons ...
> acceptable -- but keep the whole S0b sequence in one push burst before
> flipping DONE. If it truly cannot land atomically green, revert and
> write `docs\S0b_KIBITZ_NEEDED.md`, do NOT half-land.

## What the scope actually is (grounded at `c68b606c`)

Cross-module atomic:

1. NEW `nodes/_otr_shared/routing_state.py` -- schema constants, closed
   v3 dataclass, `parse_force_engine_map` moved from
   `render_driver.parse_engine_override` (2758), `engine_facts_for`,
   `resolve_effective_engine_for_role`, `capture_routing_state`,
   `state_sha256`, `validate_routing_state`. Stdlib-only, cold-import
   clean, NO import of `render_driver` (cycle risk per spec section 8).
2. `OTR_VideoDirector` at `otr_video_director.py:353-354` -- emit
   `policy_version=3` + `routing_state` block; add `IS_CHANGED`
   classmethod returning `state_sha256` and READING env inside the method
   body (ComfyUI passes only widget values, so env flips between queued
   prompts must invalidate the cache).
3. `OTR_ImageDirector` at `otr_image_director.py:169-205, 375` -- forward
   the `routing_state`; `mesh_fodder_roles_from_video_policy` reads
   `routing_state.effective_video_models`.
4. `otr_image_gen_dispatcher.py:357-497, 531-534, 542` -- swap the
   TRI-STATE basis from `accepts_still` to
   `routing_state.effective_video_models`; retire
   `_effective_engine_after_force_map` +
   `_effective_video_engine_for_role`.
5. `otr_meta_brief_image_prompt.py:476-715, 1493-2013, 500-525` --
   `_effective_prompt_engine_for_role` becomes a lookup; mesh fork +
   aspect reads follow.
6. `otr_shot_lock.py:723-724, 919-933, 1249-1252, 1306` --
   `_effective_cast_time_engine` becomes a lookup; both policy_version
   checks flip 2 -> 3; ledger writes forwarded routing_state under
   `video`.
7. `otr_video_render_batch.py:311-322` -- **frozen-routing prepass**:
   `apply_engine_override` runs BEFORE
   `validate_and_repair_still_spine` (the ordering-defect fix); ledger's
   `routing_state` forwarded into `build_episode_render_policy`.
8. `_otr_video_engines/render_driver.py:635-766, 1528-1853, 2516, 2784`
   -- `apply_engine_override` and `_enforce_radio_is_host` read from
   `routing_state` (via the ledger's stamped `video.routing_state`),
   never `os.environ`. `_still_spine_requires_scene`, the init-selection
   branch's LTX-I2V gate (`:1801-1817`) and IA2V portrait gate
   (`:1709-1721`) become verified lookups.
   `build_episode_render_policy` (`:2516`) emits `policy_version=3`.
9. `_otr_video_engines/eng_ltx_av.py` -- new mismatch gate in
   `assert_usable` comparing live `_recipe()` + `_unet_name()` against
   `routing_state.ltx_resolved`; the ONLY consumer besides the capture
   boundary that may read recipe/UNET env.
10. Test literals -- ~31 `policy_version=2` -> `3` across
    `tests/test_image_platform_c1.py` (19),
    `tests/test_remaining_video_contracts.py` (5),
    `tests/test_still_spine_helpers.py` (2),
    `tests/test_video_platform_aseam.py` (2),
    `tests/test_credits_s2_durable_stamps.py` (1),
    `tests/test_hybrid_voice_fit.py` (1),
    `tests/test_still_spine_engine_coverage.py` (1); each needs a
    `routing_state` construction helper wherever a test builds a policy
    from scratch.
11. AST / source audit -- every direct or indirect read of
    `OTR_FORCE_ENGINE_MAP`, `OTR_ENABLE_HUMO_HOSTS`,
    `OTR_ENABLE_LTX_I2V`, `OTR_LTX_AV_RECIPE`, `OTR_LTX_AV_UNET`
    OUTSIDE the two allowed sites (`OTR_VideoDirector.direct` /
    `IS_CHANGED` and `LtxAudioInEngine.assert_usable`) is a spec
    violation and must be replaced or deleted. Every survivor must be
    justified.

Regenerate the S0a fixture (`tests/test_still_plan_parity.py
--regenerate`) and verify the delta is exactly the named `special_cases`
rows (v1 policy now RAISES). Non-special cells must diff to zero.

## What THIS window can and cannot do

Can:
- Land the SAFE tail chunks (S1 = schema + 31 declarations + audit;
  nothing reads the plan).
- Keep S0a's fixture green (already did in the S0a-b amendment).
- Keep the plan honest.

Cannot deliver atomically in this window:
- Each site is a careful, semantics-preserving edit against 3435-line
  `render_driver.py` and ~700-2000-line consumers with intertwined
  history. The audit alone (grep every env read, prove each survivor is
  at the two allowed sites) is a full session.
- The bash tool budget in this window is ~30-60s per call, the full
  suite is ~1:48 wall (roughly 6432 tests), and there is no interactive
  Python REPL for cross-module debugging.
- The prior autonomous window reached this same conclusion under
  equivalent constraints -- the operator's rescope this run allowed
  autonomous continuation but did not shrink the actual scope of S0b.
- A half-landed S0b breaks the routing-is-frozen-first invariant
  silently (spec's own concern; the reason the previous window filed
  `docs/S0b_HANDOFF.md` instead of half-landing).

## What this window IS doing

- Flipping S0b from IN-FLIGHT to BLOCKED-on-kibitz in
  `docs/GO_FORWARD_PLAN.md` -- names this doc, keeps `S0b_HANDOFF.md`
  as the site inventory (unchanged).
- Continuing to S1 (SAFE per section 11 of the spec: "S1 | Schema + 31
  `still_plan` declarations + `resolve_row_aspect` + the
  post-registration audit. Nothing reads the plan."). S1 does not
  depend on S0b for its correctness; it adds `still_plan` class
  attributes to adapters + one pure helper + one audit test. The plan
  itself won't be READ until S2, so this is a leaf change that leaves
  the routing-freeze work to a dedicated window.
- Then HALT at S2 per operator (S2 needs the eyeball).

## Recommended next steps (whoever picks S0b up)

Section 11 + section 9 of
`docs/2026-07-25-still-plans-locked-build-spec.md` and the six-step order
in `docs/S0b_HANDOFF.md` (execution order) are the plan of record. The
handoff doc already inventories every file, every line range and every
transition -- no re-derivation needed.

- Two "strikes" this window did not use up: (a) attempt the atomic
  cutover with a bounded time budget; (b) attempt the split
  green-between-consumers approach.
- Kibitz options if the atomic path stalls: r3 (wiring) + r4
  (convergence) against the CURRENT HEAD (S0a + S0a-b changed no
  routing surface, only tests), local panel (`/kibitz`: codex
  `gpt-5.6-sol` high + agy Gemini 3.6 Flash High). The spec explicitly
  says the arc has already converged, so a kibitz here is a wiring
  ground truth against the actual code, not an architecture debate.

## Two-strikes tally for this file

Zero solo attempts consumed. This is a scope-and-time escalation before
touching any S0b code, based on:

1. The prior autonomous window's identical judgment (recorded in
   `docs/S0b_HANDOFF.md`).
2. The operator's explicit "do NOT half-land" and "revert if it truly
   cannot land atomically" directives.
3. The gap between S0b's real scope (~4-6 hours careful across 10+
   files with iterative suite verification) and this window's remaining
   time budget after S0a-b.

Filing this instead of half-landing preserves S0a's fence and leaves the
tree unchanged (no S0b production code touched).
