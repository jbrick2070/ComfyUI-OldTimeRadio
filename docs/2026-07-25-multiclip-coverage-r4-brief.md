# MULTI-CLIP COVERAGE -- r4 brief (convergence)

**Round:** r4, the final convergence round of the arc. Code baseline HEAD
`99d0cf0c` on `v2.0-alpha` (suite 6454 passed / 27 skipped / 1 xfailed; Bug
Bible 17; canonical `otr_canonical.json` SHA-256 prefix `5377914B`).
Prior judgments of record, all three ALREADY ADOPTED and NOT reopened here:
`docs/2026-07-25-multiclip-coverage-r1-judgment.md` (architecture),
`-r2-judgment.md` (coding plan), `-r3-judgment.md` (wiring).

**THE ONLY QUESTION THIS ROUND:** is there any remaining **MUST-FIX** before
**chunk 1** executes? Chunks 2-8 are in scope only where they constrain
chunk 1's shape.

**OUT OF SCOPE -- do not re-argue (settled r1-r3, operator-ratified):**
the requirement itself (enough REAL rendered clips per beat for MOVING video;
chain preferred, jump cut acceptable, reuse only if loop-closed, `still_*`
lanes are one still, audio lanes cut at PHRASE boundaries); containment of
multi-clip inside `render_shot` (one `ShotRow`, one manifest row per beat);
the ExecutionGroup/DAG expansion (CUT three rounds running); the pause map as
a RANKER over quantum-legal cut points, deferred to a later chunk; `ltx_8gb`
as the first vertical slice at a 169-frame beat; keeping
`tests/test_clip_fill.py`. A repeat of any of these is not a must-fix.

## The chunk order of record (r3 section 7)

1. **Hoist the route freeze into `OTRShotLock`** + `IS_CHANGED` over captured
   env state + render-time `resolve_final_shot_engines` demoted to an
   EQUALITY ASSERTION. Independently shippable; retires effective-engine
   mirrors.
2. Declaration surface (`FrameContract` = min/max/quantum/discrete/
   allow_tail_trim, continuity token on the `VideoEngine` Protocol,
   `registry.py:51-98`) + roster audit at the BOTTOM of
   `_otr_video_engines/__init__.py` after all guarded imports. All adapters
   `single_only`.
3. Partitioner + `CoveragePlan`, durably stamped, validated at BOTH
   boundaries; `CoveragePlan` validated directly, legacy `ShotRow` NOT made
   authoritative.
4. Jump-still image-phase consumer (ShotLock patches requests -> dispatcher
   merges into `objects` + `required_scene_targets` -> spine validates every
   jump segment).
5. Beat-session lifecycle: reusable MODEL/CLIP/VAE handles, teardown in one
   outer `finally`, assert LOADER-call count (not `prepare` count).
6. Terminal transaction INSIDE the render loop + transactional assembly + a
   new ffprobe helper with `-count_frames`.
7. `ltx_8gb` live slice at a 169-frame beat (`161 + (9-1)`) + a 162-frame CPU
   tail-trim case.
8. Later: pause map; then further adapters; audio lanes last.

## Chunk 1 as it will actually be built

Freeze each shot's FINAL effective engine ONCE, inside `OTR_ShotLock` (node
90), so `build_execution_plan`, cast-time validation, the still spine, the
prompt hooks and the future `CoveragePlan` all consume ONE snapshot. The two
mutations to freeze are the ONLY two that matter today:
`apply_engine_override` (`OTR_FORCE_ENGINE_MAP`, `render_driver.py:2825`) and
`_enforce_radio_is_host` (`render_driver.py:1413`, gated by
`OTR_ENABLE_HUMO_HOSTS`). Both are already unified in
`resolve_final_shot_engines` (`render_driver.py:2788`, landed `57f4983a`).
After the hoist, the render-time call at `otr_video_render_batch.py:314`
becomes an equality assertion rather than a mutator.

Canonical node order, confirmed against `workflows/otr_canonical.json`:
`87 OTR_VideoDirector -> 88 OTR_ImageDirector -> 89 OTR_MetaBriefImagePromptGen
-> 90 OTR_ShotLock -> 91 OTR_ImageGenDispatcher -> 92 OTR_VideoRenderBatch`.

## Claude's anchor findings, NEW since r3 -- confirm or refute each

Every one of these was read at HEAD `99d0cf0c` on the real Windows tree.
Attack them; a refutation with a file+line is worth more than agreement.

**A1 -- r3's placement cite is wrong, and the naive hoist is a NO-OP.**
r3 said "after policy validation, before `build_execution_plan`
(`otr_shot_lock.py:1091-1142`)". But `build_execution_plan` is DEFINED at
`otr_shot_lock.py:1059`, so 1091-1142 is INSIDE it, and no shot rows exist
before it returns. Worse: `resolve_final_shot_engines(ledger)` reads
`ledger["video"]["shots"]`, and inside `OTRShotLock.lock` the ledger's
`led["video"] = video_section` assignment does not happen until AFTER
`groups, shots = build_execution_plan(...)` at `:1297`. So calling the
existing function at the r3-specified point iterates an empty/stale
`video` section and silently freezes NOTHING. The freeze must either (a) run
inside `engine_for(role)` so shots are minted already-effective, or (b) run on
the returned `shots`/`groups` lists before `led["video"]` is stamped, via a
list-level helper that `resolve_final_shot_engines` then delegates to. Which?

**A2 -- `execution_groups` carry their own `engine_id` and nobody mentioned
them.** `build_execution_plan` builds `groups` with
`"engine_id": engine_for(role)` (`otr_shot_lock.py:1069-1082`) and passes them
through `_resolver.validate_execution_groups`. A shot-only freeze leaves
`group["engine_id"]` on the PICKED engine while `shot["engine_id"]` holds the
EFFECTIVE one -- a new divergence introduced by the fix itself. Does chunk 1
freeze groups too, and does `validate_execution_groups` accept the redirected
engine for an announcer/music group?

**A3 -- the cast-time preflight validates the WRONG engine after the fix.**
`_assert_family_inputs_satisfiable_cast_time(engine_id, b, ledger, policy)`
runs inside `build_execution_plan` (`otr_shot_lock.py:~1090`) on
`engine_for(b["role"])` -- the PICKED engine. If the freeze runs after it, the
same defect class the route lock exists to kill survives one level up: cast
time proves inputs satisfiable for an engine that will not render the beat.
This argues for placement (a) in A1. Confirm or refute.

**A4 -- the MetaBrief mirror is ROLE-map derived, and a transport for its
retirement ALREADY EXISTS.** r3 left open whether chunk 1 can delete
`otr_meta_brief_image_prompt._effective_prompt_engine_for_role`. Grounding it:
that helper takes `video_models` (the per-role video slot map from the
director policy) and calls
`otr_image_gen_dispatcher._effective_video_engine_for_role(role, eng_id)`
(`otr_meta_brief_image_prompt.py:501-531`). It never touches a shot row. So
the authority it mirrors is a ROLE -> EFFECTIVE ENGINE map, which is fully
computable at `OTR_VideoDirector` (node 87) time -- upstream of MetaBrief.
And the transport is already wired and shipping: VideoDirector resolves each
per-role engine to its `render_aspect`, ImageDirector forwards that map into
`image_policy_json`, and MetaBrief reads it in `_still_aspects_from_policy`
(`otr_meta_brief_image_prompt.py:~575`). Therefore retiring the MetaBrief
mirror needs no new node and no new link -- it needs VideoDirector to stamp a
frozen role->effective-engine map into a policy it already emits.
**Question:** does that belong in chunk 1, or is it a separate chunk 1b, and
does adding a key to an existing policy JSON payload leave
`workflows/otr_canonical.json` byte-identical?

**A5 -- `OTRShotLock` has no `IS_CHANGED` today.** It ships only
`VALIDATE_INPUTS` returning True (`otr_shot_lock.py:1222-1224`). Adding
`IS_CHANGED` is a pure Python classmethod -- no widget, input, link or schema
change -- so canonical stays byte-identical. The captured set for chunk 1 is
`OTR_FORCE_ENGINE_MAP` + `OTR_ENABLE_HUMO_HOSTS`. **Deliberately NOT captured:
`OTR_LTX_AV_RECIPE` / `OTR_LTX_AV_SHARP` / `OTR_LTX_AV_UNET`.**
`eng_ltx_av.py:402-432` documents per-call re-reads verbatim ("Read fresh
every call -- an operator flips daily<->hero per beat"), so freezing them
would silently delete an advertised per-beat capability; that is an OPERATOR
decision, not a coder call, and it is flagged and unruled. Chunk 1 therefore
freezes ROUTING ONLY. **Does leaving the recipe reads live break the
render-time equality assertion?** They do not change `engine_id`, only
`wants_talking_prompt()` -- state the failure mode with a file+line if you
disagree.

**A6 -- `provider_side` is a three-part rule, not an attribute.**
`_is_cloud_video_engine` (`render_driver.py:1275`) accepts a `cloud_` id
prefix OR the attribute OR a `cloud_`-prefixed node key.
`cloud_kling_avatar` has NO `provider_side` attribute and is caught by the id
prefix alone. Any freeze-time classification must call that function, never a
bare `getattr`, or the radio-host redirect can send a cloud avatar to local
LTX. Chunk 1 will carry a regression on both a PICKED and a FORCED
`cloud_kling_avatar`.

## r3's three explicitly open items -- rule on each

1. Can chunk 1 safely delete `_effective_prompt_engine_for_role` given node 89
   precedes node 90 -- or must MetaBrief keep a mirror until a
   VideoDirector-time freeze exists? (See A4: a third answer may be available.)
2. Does `wrapper_bridge.run_graph` gain a prepared-handles parameter, or does
   the adapter's segment graph take them (`wrapper_bridge.py:301-377`)? codex
   offered both in r3; pick ONE and say why. (Chunk 5, but it fixes chunk 2's
   Protocol shape, so it is decided here.)
3. Confirm that NO chunk in 1-7 touches `workflows/otr_canonical.json`.
   Name any node/widget/input/link/schema surface either of you believes is
   forced, with the file and line that forces it.

## Answer format

For each item: **MUST-FIX**, **SHOULD-FIX**, or **AGREE**, with a file+line
citation for every claim. Close with a single **VERDICT: yes** (chunk 1 may
execute as specified, with any must-fixes listed) or **VERDICT: no** (a
must-fix invalidates the chunk's shape). Unsupported claims are discarded by
the judge; a confident cite to a line that does not say what you claim costs
more than silence.
