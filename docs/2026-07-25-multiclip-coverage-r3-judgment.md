# MULTI-CLIP COVERAGE -- r3 judgment (wiring)

**Runs:** `kibitz-runs/2026-07-25-multiclip-coverage/r3/` (codex `gpt-5.6-sol`
high) and `kibitz-runs/2026-07-25-multiclip-coverage-agy/r3/` (agy
`Gemini 3.6 Flash (High)`), pins verified, run independently.
Code baseline `a1d810f1`. Claude is grounded panelist and sole judge.

**Both: VERDICT no.** Both accept the architecture; both find the WIRING
under-specified in the same place first. That agreement is the round's value.

## 1. THE HEADLINE -- the route lock I landed today is ONE NODE TOO LATE

Both seats, independently, and I confirmed the node order myself by reading
the canonical JSON:

```
87 OTR_VideoDirector -> 88 OTR_ImageDirector -> 89 OTR_MetaBriefImagePromptGen
-> 90 OTR_ShotLock -> 91 OTR_ImageGenDispatcher -> 92 OTR_VideoRenderBatch
```

`resolve_final_shot_engines` (landed `57f4983a`) runs in **node 92**. Stills
are minted in **node 91**. Image PROMPTS are generated in **node 89**. So the
route lock closed the gap it was aimed at -- spine validation vs render -- but
it sits DOWNSTREAM of both prompt generation and still minting.

**That is exactly why `otr_meta_brief_image_prompt._effective_prompt_engine_for_role`
exists and says, verbatim, that it "mirrors the image dispatcher's
effective-engine seam: selected per-role engine, then force-map, then the
radio-is-host HuMo redirect."** The mirror is not sloppiness -- it is the
image phase compensating for an authority that runs after it.

**JUDGE CALL: hoist the freeze into `OTRShotLock.lock`, after policy
validation and before `build_execution_plan` (`otr_shot_lock.py:1091-1142`).**
Then:
- `build_execution_plan`, cast-time validation, the prompt hooks and the
  `CoveragePlan` all consume ONE snapshot;
- render-time `resolve_final_shot_engines` becomes an **equality ASSERTION**,
  not a mutator (codex's framing, adopted);
- the MetaBrief and dispatcher mirrors become deletable -- a real
  simplification, not just a fix.

**codex's caveat, adopted:** the snapshot reads environment state, so
`OTRShotLock` needs `IS_CHANGED` over every captured variable (or they get
threaded through `video_policy_json`), or ComfyUI will serve a cached lock
across an env change.

**Honest note:** node 89 runs BEFORE node 90, so hoisting to ShotLock still
does not put MetaBrief downstream of the authority. Eliminating THAT mirror
means resolving effective route at VideoDirector (87) time. Out of scope here;
recorded so a later window does not think ShotLock finished the job.

## 2. codex-only findings that change the plan (all confirmed)

1. **JUMP CUT has no image-phase consumer at all.** `OTR_MetaBriefImagePromptGen`
   runs independently of ShotLock and the dispatcher only materializes objects
   from `image_prompts_json`; nothing turns coverage segments into extra still
   objects (`otr_image_gen_dispatcher.py:557-567`, `:906-947`, `:1067-1119`).
   **The first JUMP successor would reach render with no minted init image.**
   Fix: ShotLock patches jump-still requests into `script_json`; the dispatcher
   merges them into `objects` + `required_scene_targets` and stamps path/hash
   before `image_done`; the spine validates every jump segment, not one
   beat-level scene row. No new node/link needed -- the existing
   ShotLock -> Dispatcher -> RenderBatch wire already carries it.
2. **The proposed step order cannot execute CHAIN at all.** Terminal-frame
   persistence currently runs only after `run_real_episode` has rendered every
   shot (`otr_video_render_batch.py:346-360`,
   `render_driver.py:3024-3035`), but segment N+1 needs segment N's terminal
   frame synchronously. **The terminal transaction moves into step 3**, not
   step 4: render -> apply drop_head/trim_tail -> extract last VISIBLE frame
   (pinned bt709) -> validate/hash -> atomic rename -> resolve successor token.
   Beat assembly may stay in step 4.
3. **The existing ffprobe helper cannot make the assertion we specified** --
   it collects neither frame count nor dimensions (`wan_shared.py:52-123`).
   Needs `-count_frames` / `nb_read_frames` plus width, height, both frame-rate
   fields, pixel format and colour tags.
4. **Roster-audit location:** bottom of `_otr_video_engines/__init__.py` AFTER
   all guarded adapter imports -- not inside `registry.py`, which would report
   every not-yet-imported adapter missing.
5. **Do not delete the ping-pong tests.** `tests/test_clip_fill.py:120-156`
   stays green because the helper remains legal for `still_*`/decorative
   lanes; instead ASSERT the `ltx_8gb` coverage path never calls it. Deleting
   them would drop valid coverage while proving nothing.

## 3. The one real split -- how much schema becomes authoritative

- **agy:** add `role`, `char_id`, `start_s`, `dur_s` and `coverage_plan` to
  `ShotRow` (`schemas.py:302`) before enabling validation, or `extra="forbid"`
  raises on real production rows.
- **codex:** do NOT make the legacy shot schema authoritative in this slice.
  Validate `CoveragePlan` DIRECTLY at both boundaries (ShotLock before
  serialization, RenderBatch before execution).

**JUDGE CALL: codex.** agy is right about the crash and wrong about the
remedy's scope -- widening `ShotRow` to match today's dicts is a separate
change with unknown breakage across every consumer, and it buys nothing this
slice needs. Validating the new nested contract directly closes the new
contract without dragging the legacy one along. agy's finding survives as the
REASON not to naively `model_validate` a production shot dict.

**codex's addition, adopted:** a wire-only plan is useless -- ShotLock must
durably stamp `sections={"video": video_section}`, and after jump-still
resolution the dispatcher must stamp BOTH updated `video` and `images`, not
only `images` (`production_ledger.py:527-564`,
`otr_image_gen_dispatcher.py:962-966`), or the plan vanishes from the durable
ledger and cannot support replay.

## 4. Convergence (adopted)

- **`max_render_frames == 0` stays the shipped "unpinned" transport value.**
  Both seats reject a global rejection -- it would break the 8-GB WAN contract
  (`f914f0a4`). Scope it: for `coverage_mode="multi_clip"`, resolve 0 at the
  ShotLock freeze to the adapter's positive static maximum and reject a
  missing/invalid resulting `FrameContract.max_frames`. **W4 answered: the
  8-GB contract is safe.**
- **Deferred token:** one exact form, `otr-deferred://terminal/<shot_id>/<prior_segment_index>`.
  Resolved to a hashed, existing episode-local PNG BEFORE
  `VideoRequest.model_validate`, `_assert_family_inputs_satisfiable`, or any
  adapter call; unresolved at that boundary is terminal. The spine skips it;
  `validate_and_repair_still_spine` learns the prefix. **Do NOT add it to
  `role_compat.INPUT_TOKENS`** (codex) -- it is a VALUE of the existing
  `init_image` capability, not a new capability.
- **"Prepare once" still is not "load once."** Both. `Ltx8gbEngine.load`
  resolves classes only; `CheckpointLoaderSimple` / `CLIPLoader` live inside
  every graph execution (`eng_ltx_8gb.py:316-423`). Needs a beat-session
  contract with reusable MODEL/CLIP/VAE handles, the loader prefix split from
  the segment graph, and `wrapper_bridge.run_graph` taught to accept prepared
  handles (`wrapper_bridge.py:301-377`). Test loader-node call counts.
- **Teardown in one outer `finally`**; no new ComfyUI node, widget, input or
  link -- canonical JSON unchanged for this slice.

## 5. Adopted acceptance number (codex should-fix 1)

**First live target = a 169-frame beat**: `161 + (9 - 1)` -- the LTX-8GB cap
plus one legal minimum segment less the chained duplicate head frame. It
proves a two-segment chain with NO tail trim, and unlike "over 161" it is
exactly reproducible. Add a separate **162-frame CPU case** to exercise legal
tail trimming.

## 6. Named test files (adopted)

- `tests/test_multiclip_coverage_plan.py` -- contract purity, property sweeps,
  seam ownership.
- `tests/test_ltx_8gb_multiclip.py` -- prompt hooks, deferred resolution,
  loader-call count, teardown-on-failure, no ping-pong.
- `tests/test_multiclip_transactional_assembly.py` -- terminal ordering,
  rollback, CFR/stream rejection, exact ffprobe count.
- Extend `tests/test_workflow_json_wiring_invariants.py` (pin the unchanged
  ShotLock -> Dispatcher -> RenderBatch path) and
  `tests/test_capability_profiles.py:384` (swallowed-import detection).
- `tests/test_clip_fill.py` stays; add the "coverage path never calls it" pin.

## 7. Revised chunk order

1. **Hoist the route freeze into ShotLock** + `IS_CHANGED` + render-time
   equality assertion. Independently shippable, and it retires two mirrors.
2. Declaration surface (`FrameContract`, continuity token, Protocol
   signatures) + roster audit at the bottom of `__init__.py`. All adapters
   `single_only`.
3. Partitioner + `CoveragePlan`, durably stamped, validated at both
   boundaries.
4. Jump-still image-phase consumer (ShotLock patch -> dispatcher merge ->
   spine validates every jump segment).
5. Beat-session lifecycle (reusable handles, loader-count asserted, `finally`
   teardown).
6. Terminal transaction IN the render loop + transactional assembly + the new
   ffprobe.
7. `ltx_8gb` live slice at 169 frames.
8. Later: pause map; then further adapters; audio lanes last.

## 8. Open for r4

- Whether chunk 1's mirror retirement can safely delete
  `_effective_prompt_engine_for_role` given node 89 precedes node 90, or
  whether MetaBrief must keep a mirror until a VideoDirector-time freeze
  exists.
- Whether `wrapper_bridge.run_graph` gains a prepared-handles parameter or the
  adapter's segment graph takes them -- codex offered both; pick one.
- Confirm no chunk in 1-7 touches `workflows/otr_canonical.json`.
