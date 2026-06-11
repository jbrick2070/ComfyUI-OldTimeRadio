# Pass 02 judgment -- still-image spine CONVERGED (campaign ~$0.09)

Verdicts: gpt-5.5 no, gemini-3.1-pro no, deepseek no -- but ZERO architecture
challenges; every item is implementation plumbing on the pass01 plan. Same
convergence profile as look-qa-r5 pass03. The items below are FOLDED into the
build spec as hard preconditions/details (the builder treats them as part of
pass01_plan.md):

1. **One versioned prompt-object schema** (GPT-1): `derive_image_prompts`
   returns `{"objects":[...]}` (portrait rows migrated to the same object
   schema in the SAME patch); `dispatch_images` consumes objects, never bare
   char_id maps. No dual-schema compatibility shims -- migrate once, with
   tests.
2. **Guard branching by kind** (GPT-2): person guard + gear scrub run ONLY on
   kind=portrait; scene stills get the no-text clause + scene-safe checks.
   The derivation paths split BEFORE the guards.
3. **w/h plumbing end-to-end** (Gem-1): pinfo -> request dict -> engine call
   -> `request_cache_key` gains kind/w/h. Landscape scene stills are real,
   not defaults.
4. **Cache-hit materialization** (Gem-2): a hit copies the cached file into
   the CURRENT episode stills/ + appends a fresh ledger row (today's code
   `continue`s -- episode folders would silently miss reused stills).
5. **Trace stamps via the request** (Gem-3): `_init_source`/`_init_image`
   stamped in `build_request_from_shot`, copied to trace rows in
   `run_episode` (the established `_prompt_*` pattern).
6. **Two preconditions become STEP 0 probes** (DS-1/2): (a) still_kenburns
   external-init support -- verify, ADD if missing (the 6/5 look depends on
   it); (b) the render node's image_done gate input -- verify, ADD if
   missing. Both are small, both block ST-5/ST-6, both land FIRST.
7. **episode_id into INPUT_TYPES + the saved json wire** (DS-3): explicit;
   the json edit is part of the slice (in place, per the operator directive).

Standing rejections from pass01 hold (LTX img2vid cut from v1; role-only open
detection rejected; per-beat stills for every beat cut; no new model
abstraction). The operator north star (6/5 restoration; grain of salt on
invention) governed both passes.

**CONVERGED.** Build spec = pass01_plan.md + this judgment's seven folded
items. Suggested build order: STEP 0 probes (item 6) -> ST-1 helpers -> ST-2
schema/objects -> ST-3 dispatcher -> ST-4 driver -> ST-5 kenburns/wan wiring
-> ST-6 json + gate -> ST-7 tests -> ONE 30w acceptance render -> operator
eyeball. Commit+push per green chunk (operator git policy 2026-06-10).
