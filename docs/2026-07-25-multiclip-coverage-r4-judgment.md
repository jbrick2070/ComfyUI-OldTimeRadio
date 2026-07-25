# MULTI-CLIP COVERAGE -- r4 judgment (convergence)

**Runs:** `kibitz-runs/2026-07-25-multiclip-coverage/r4/` (codex `gpt-5.6-sol`
high, pin verified in `codex_model_selected.txt`) and
`kibitz-runs/2026-07-25-multiclip-coverage-agy/r4/` (agy
`Gemini 3.6 Flash (High)`, pin verified in `antigravity.log`), run
independently. Code baseline HEAD `99d0cf0c`. Brief:
`docs/2026-07-25-multiclip-coverage-r4-brief.md`. Claude is the grounded
panelist and sole judge.

**BOTH SEATS: VERDICT yes-with-fixes.** The arc converges. The architecture,
the chunk order and the acceptance numbers stand. Chunk 1's SHAPE changes --
it moves upstream -- and that change is proven, not argued.

## 1. THE DECISIVE FINDING -- node IDs are not execution order, and I verified it

codex's must-fix 1, which neither r3 nor my own brief had: **there is no
`89 -> 90` edge.** I confirmed it by walking `workflows/otr_canonical.json`'s
link list myself rather than taking the claim:

```
87 OTR_VideoDirector  <- 63 OTR_WorkflowValidator
88 OTR_ImageDirector  <- 87 (slot0 -> in0)
89 OTR_MetaBrief...   <- 62 OTR_LedgerFreezeCascade, 88 (slot0 -> in1)
90 OTR_ShotLock       <- 62, 7, 12, 87 (slot0 -> in2)
91 OTR_ImageGenDisp.  <- 7, 88, 89 (slot0 -> in2), 90 (slot0 -> in0, slot4 -> in4)
92 OTR_VideoRenderBatch <- 7, 91, 91

89 in ancestors(90): False        90 in ancestors(89): False
87 in ancestors(89): True         87 in ancestors(90): True
```

**89 and 90 are INDEPENDENT branches that reconverge at 91.** The r3 judgment
(and my own brief) read the ascending node ids as a pipeline; they are not one.
So the r3 plan's premise -- "hoist to ShotLock, node 90, and MetaBrief at 89 is
merely upstream" -- is wrong in a way that matters: a ShotLock freeze cannot
inform MetaBrief on ANY execution order, not just today's.

**JUDGE CALL: the freeze originates at `OTR_VideoDirector` (node 87), the
UNIQUE common ancestor of both branches.** codex's must-fix 1 is adopted in
full, and its "cut a separate chunk 1b" is adopted with it -- the VideoDirector
transport is not an optimization, it is chunk 1's correctness condition.

**Canonical JSON stays byte-identical, and the topology above is the proof.**
87 already wires its policy STRING to BOTH 88 and 90; 88 already wires to BOTH
89 and 91. Every one of the four consumers is reachable by adding KEYS to
payloads on wires that already exist. No node, widget, input or link changes.
That settles r3 open item 3 with evidence rather than assertion.

## 2. Where both seats agree, adopted without argument

- **A1 option (a).** Resolve the role map ONCE before group construction;
  `engine_for(role)` returns only the EFFECTIVE engine, so groups, the
  cast-time preflight and shots are minted effective from birth. Both seats,
  independently. **CUT option (b)**, the post-build list mutator -- it is too
  late for cast-time validation and invites exactly the divergence it means to
  fix.
- **A2 groups.** `validate_execution_groups` (`_otr_shared/resolver.py:125`)
  checks structure and ordering only -- unique ids, acyclic `depends_on`,
  provider-before-consumer -- so it accepts a redirected engine. Input
  satisfiability stays with the cast-time preflight. Add the invariant: every
  shot's engine equals its group's engine for that role.
- **A3.** The preflight must receive the effective engine. Option (a) delivers
  that by construction.
- **A6.** All freeze-time classification calls
  `render_driver._is_cloud_video_engine` / `_radio_is_host_redirect_applies`
  (`render_driver.py:1275`, `:1376`), never a bare `getattr`.
  `cloud_kling_avatar` has a `cloud_` id and NO `provider_side` attribute.
  Regression on PICKED and FORCED `cloud_kling_avatar`; both stay cloud.
- **r3 open item 2 -- RESOLVED, unanimous.** `wrapper_bridge.run_graph` does
  NOT gain a prepared-handles parameter (`wrapper_bridge.py:301-377` is a
  generic executor). Adapter `prepare` owns reusable MODEL/CLIP/VAE handles;
  each segment graph receives them as literals and omits its loader nodes; one
  outer `finally` calls adapter teardown. Matches the shipped
  `prepare`/`render_clip`/`teardown` protocol (`registry.py:92-97`).
- **A5 recipe.** Route equality covers `engine_id`/`family` ONLY. The
  `OTR_LTX_AV_RECIPE`/`_SHARP`/`_UNET` per-call re-reads
  (`eng_ltx_av.py:391-432`) do not change `engine_id`, so they cannot break the
  equality assertion, and chunk 1 does NOT freeze them -- the advertised
  per-beat recipe capability survives untouched and stays an operator question.
  **codex's addition, adopted:** chunk 5's session continuity token MUST
  include recipe + weight identity, so prepared handles cannot survive a
  recipe change.

## 3. THE ONE PLACE I OVERRULE A SEAT

**agy's must-fix 1 prescribes the wrong authority.** It says `engine_for`
should delegate to
`otr_image_gen_dispatcher._effective_video_engine_for_role`. **Rejected.**
That helper's force-map half, `_effective_engine_after_force_map`
(`otr_image_gen_dispatcher.py:377-397`), SWALLOWS a malformed map --
`except Exception: return eng_id`, docstring "never block dispatch on a bad
override". Routing that through the LOCK would import a silent-unforced path
into the authority and regress the fail-closed contract landed this morning at
`57f4983a`, where a malformed `OTR_FORCE_ENGINE_MAP` is terminal
(`render_driver.py:2843-2856`). The frozen map is computed from
`render_driver`'s own parse, fail-closed, and the dispatcher mirror is DELETED
rather than promoted. agy's direction (mint effective at birth) is right; its
choice of source is not.

## 4. What I found that neither seat had in full

**There are THREE effective-engine mirrors at HEAD, not two, and TWO of them
hardcode the redirect target.**

1. `otr_meta_brief_image_prompt._effective_prompt_engine_for_role` (`:501-526`)
2. `otr_image_gen_dispatcher._effective_video_engine_for_role` /
   `_effective_engine_after_force_map` (`:377-425`)
3. **`otr_shot_lock._effective_cast_time_engine` (`:919-934`)** -- a third
   mirror living INSIDE the lock's own preflight, which independently re-reads
   both env vars AND swallows a malformed force map with a warning
   (`"OTR_FORCE_ENGINE_MAP ignored in cast-time preflight"`). codex found this
   one independently; agy did not.

And the drift neither seat named: `_NEVER_HUMO_REDIRECT_ENGINE` is defined once
(`render_driver.py:1234`) but the redirect target is written as the bare
literal `"ltx_audio_in"` in TWO mirrors -- `otr_shot_lock.py:933` and
`otr_image_gen_dispatcher.py:422`. Three copies of one routing constant, two of
them uncoupled from its definition. All three mirrors die in chunk 1.

## 5. codex must-fix 3, adopted -- the freeze must reach the DERIVED maps too

Freezing `engine_id` is not enough: route-derived policy is still computed from
PICKED engines in several places. VideoDirector derives `aspects` and `talking`
from `resolved_video` (`otr_video_director.py:363-372`, `:399-439`);
ImageDirector's 3D lock reads picked `video_models`
(`otr_image_director.py:134-160`); only mesh-fodder routing already mirrors the
effective route (`:183-205`). Aspects, talking flags, 3D locks, mesh-fodder
roles, MetaBrief families and dispatcher still-capabilities all consume the
frozen effective map, or a forced engine still receives picked-engine geometry
and asset contracts -- the exact defect class this build exists to close.

## 6. codex must-fix 4, adopted -- demoting ONE call site is insufficient

Three late mutation paths exist, not one: `otr_video_render_batch.py:314`,
`run_real_episode` (`render_driver.py:2755`), and the per-request
`_enforce_radio_is_host` call in `build_request_from_shot`
(`render_driver.py:1495-1513`). All three become ONE pure equality assertion
over snapshot + shot engine/family + group parity, mutating nothing.

## 7. Chunk 1, as it will now be built (supersedes r3 section 7 item 1)

1. `OTR_VideoDirector` computes the frozen role -> EFFECTIVE engine map with a
   fail-closed parse, plus a normalized `routing_env_snapshot`
   (`OTR_FORCE_ENGINE_MAP`, `OTR_ENABLE_HUMO_HOSTS`), and stamps both into the
   policy payload it already emits. Add `OTRVideoDirector.IS_CHANGED` over the
   normalized snapshot.
2. `OTR_ImageDirector` explicitly forwards both (its payload is constructed
   key-by-key, `otr_image_director.py:370-410` -- a new key is NOT forwarded
   automatically).
3. Derived maps -- aspects, talking, 3D locks, mesh-fodder -- move onto the
   effective map.
4. `OTR_ShotLock` REJECTS a snapshot/current-env mismatch; `engine_for` returns
   the effective engine; groups + preflight + shots all mint from it; stamp
   BOTH picked and effective role maps plus the snapshot into `video` (codex
   should-fix 2, adopted -- `video.roles` alone cannot replay an equality
   failure); add `IS_CHANGED`.
5. Delete all three mirrors and `_effective_cast_time_engine`.
6. The three render-time mutation paths become one pure equality assertion.
7. Acceptance: canonical byte-identical `5377914B...` stated as an explicit
   criterion; full suite; Bug Bible; the panel's verify checklist.

## 8. Chunks 2-8 unchanged

r3 section 7 items 2-8 stand as judged, with chunk 5 taking codex's continuity
-token addition from section 2 above. No seat reopened them.

## 9. Convergence statement

Neither seat raised a new must-fix against the ARCHITECTURE -- every must-fix
this round is about chunk 1's placement and the completeness of the mirror
retirement. That is convergence. **The arc is closed; chunk 1 executes.**

---

# ADDENDUM -- final all-Sonnet grounded fan-out (operator-directed, pre-code)

Six independent Sonnet subagents, each told to read the REAL Windows files via
Desktop Commander (never the lagging Linux mount), audited chunk 1 before any
code was written. Dimensions: (1) VideoDirector/ImageDirector transport,
(2) ShotLock freeze placement, (3) mirror-retirement blast radius,
(4) render-time equality assertion, (5) route-derived policy maps,
(6) test + canonical-JSON safety. Every finding below was re-grounded by the
judge before adoption. **The fan-out changed the plan in four material ways.**

## FO-1 -- the freeze cannot be COMPUTED at VideoDirector as specified

Agents 1 and 2 independently: `nodes/otr_video_director.py` has **no `import
os`** and reads neither env var. The force-map + redirect logic exists ONLY in
`render_driver.py` (3489 lines) and in the mirrors. So "VideoDirector computes
the frozen map" as r4 specified would force node 87 to import from the render
driver -- against that file's own cold-import contract (docstring lines 20-22,
"stdlib + the dep-free registry/role_compat") -- or to become a FOURTH copy of
the redirect. Both are wrong.

**JUDGE CALL -- the root fix the panel did not reach: extract the authority.**
A new dep-free `nodes/_otr_shared/route_freeze.py` owns force-map parsing +
application (fail-closed) and the never-humo redirect, exporting
`freeze_role_engines(video_models, env)` and `routing_env_snapshot(env)`.
`render_driver` delegates to it (its public functions keep byte-identical
behaviour), VideoDirector imports it cheaply, and all mirrors collapse onto it.
This removes the duplication instead of relocating it -- CLAUDE.md's root-cause
rule, applied to the fix itself. Guard: `tests/test_video_platform_aseam.py:71-82`
asserts these modules import no torch/transformers/diffusers at import time, so
the new module stays stdlib-only.

## FO-2 -- codex must-fix 3 is 1/6 urgent, not 6/6 (agent 5, grounded)

Of the six derived values codex said must consume the frozen map:

- **`aspects` -- CONFIRMED, and it is a LIVE DEFAULT-ENV BUG, not a latent
  one.** `otr_video_director._role_aspects` (`:399-419`) derives from the PICKED
  engine. Pick `humo_1.7B` (portrait, `eng_humo.py:225/626`) for
  `announcer_visual`; with `OTR_ENABLE_HUMO_HOSTS` unset the effective engine is
  `ltx_audio_in`, which is **wide** (`eng_ltx_av.py:345-347`). The still is
  minted PORTRAIT and the wide render centre-crops it -- and `eng_ltx_av.py:345-347`
  documents that exact outcome verbatim: *"the director defaulted to a 832x1216
  PORTRAIT still that the wide render then centre-cropped, lopping the subject's
  head off."* **This is the decapitation bug, and chunk 1 fixes it.**
- **3D lock -- CONFIRMED as code, DORMANT in practice.** No registered engine
  declares `requires_mesh_portrait`; the three that do were unregistered
  2026-06-29 (`_otr_video_engines/__init__.py:122-125`). Fix it, but it earns no
  urgency and no live proof is possible today.
- **`talking` -- PARTIALLY ALREADY PATCHED.** `_effective_talking_roles`
  (`otr_meta_brief_image_prompt.py:552-574`) already upgrades via the effective
  engine. One-directional (never downgrades), which covers the dangerous way.
- **REFUTED as open work, all three already effective-aware:** mesh-fodder
  roles (`otr_image_director.py:183-205`), MetaBrief families (`:475-494`), and
  dispatcher still-capabilities (which hard-fails rather than guessing).

Scope shrinks on evidence. codex's must-fix stands, but only `aspects` is
urgent and only `aspects` + the dormant 3D lock need new work.

## FO-3 -- the equality assertion breaks more than the panel counted (agent 4)

- **14 assertion sites across 3 files assert the MUTATION** and need inverting:
  `test_video_render_driver_additive.py:615,625,658,673,676,690`,
  `test_brief_radio_host.py:479,486`,
  `test_video_render_driver_perbeat_audio.py:315,324,353,362,373`.
- **Node-level integration tests drive the real `OTRVideoRenderBatch` with
  hand-built ledgers** (`test_video_render_driver_additive.py:565,580`,
  `test_video_render_driver_perbeat_audio.py:696`) that carry no snapshot. The
  `OTR_TEST_MODE` escape hatch does NOT cover them: it guards only
  `validate_and_repair_still_spine` at `otr_video_render_batch.py:320-331`,
  while `resolve_final_shot_engines` at `:314` runs unconditionally.
- **TWO SHIPPED HTTP ENTRY POINTS BYPASS THE WHOLE CHAIN:**
  `POST /otr/video_render_single` -> `render_driver.render_single()` (`:3432`)
  and `POST /otr/video_render_soak` -> `run_gpu_soak()` (`:3387`), both
  registered unconditionally in `__init__.py:511-534`. Neither builds a ledger
  through any director; neither ever calls the three mutation paths.
- **`_enforce_radio_is_host(shot)` takes a shot and no ledger**, so tests
  calling it directly have no snapshot to assert against.

**JUDGE CALL on the honest rule, stated loudly rather than buried:** the
assertion is over the frozen map WHEN THE LEDGER CARRIES ONE. A ledger with no
frozen map is a non-director path -- soak, single-render, legacy fixture -- and
keeps today's mutating behaviour, logged explicitly at INFO. This is NOT a
silent fallback: a ledger that HAS a snapshot and disagrees with it is
terminal, and the absent-snapshot branch is named, logged and tested. The
alternative -- a strict assertion -- would break two shipped production
endpoints to satisfy a contract they were never in.

## FO-4 -- decomposition: chunk 1 is THREE pushed chunks, not one

Both the blast radius (agent 3: 6 call sites for mirror B, 5 direct test
asserts for mirror C, and `test_still_plan_parity.py`'s shared helpers
`_materialized_row`/`_render_decisions_row` at `:201,:227` -- deleting B breaks
that entire module) and the test inversions make a single atomic commit
reckless. Order, each green and pushed on its own:

- **1a -- the shared authority.** Add `_otr_shared/route_freeze.py`; delegate
  `render_driver`, the three mirrors and `_effective_cast_time_engine` to it.
  ZERO behaviour change, zero test inversions. Kills the duplication and both
  hardcoded `"ltx_audio_in"` literals (`otr_shot_lock.py:933`,
  `otr_image_gen_dispatcher.py:422`) that drift from
  `_NEVER_HUMO_REDIRECT_ENGINE` (`render_driver.py:1234`).
- **1b -- freeze and consume.** VideoDirector stamps the frozen map +
  snapshot; ImageDirector forwards (its payload is key-by-key,
  `otr_image_director.py:370-414`, so a new key is NOT auto-forwarded);
  ShotLock's `engine_for` returns effective so groups + preflight + shots mint
  effective from birth; `aspects` moves onto the effective map -- **the
  decapitation fix**; mirrors deleted.
- **1c -- the equality assertion** at all three late paths, with the 14 test
  inversions and the named absent-snapshot branch.

## FO-5 -- confirmed safe, no action needed (agent 6)

No test pins an exact key set, dict equality or golden fixture of
`video_policy_json` / `image_policy_json`, so additive keys are safe. Every
canonical-JSON pin is topology- or widget-vector-based
(`test_workflow_json_wiring_invariants.py`,
`test_workflow_graph_integrity_guards.py`), and `widget_vector_drift`
(`_otr_workflow_validator.py:158-181`) reads `INPUT_TYPES()` only -- it never
introspects `IS_CHANGED`. Precedent exists: `WorkflowValidator.IS_CHANGED`
already ships (`_otr_workflow_validator.py:225-238`). **Canonical
`5377914B...` stays byte-identical across 1a/1b/1c.**

## FO-6 -- the exhaustive env-read inventory (agent 3), which the freeze must respect

Nine code reads of the two vars, all in `nodes/`: `render_driver.py:1202`
(ambient-audio predicate), `:1439` (the redirect authority), `:1832`
(radio_host_portrait init), `:2843` (force-map authority);
`otr_shot_lock.py:922,930`; `otr_meta_brief_image_prompt.py:300`;
`otr_image_gen_dispatcher.py:387,415`. The last five collapse onto
`route_freeze`. The three render-time reads at `:1202`, `:1832` and the
redirect itself stay live by design -- they run AFTER mutation and are the
ground truth the image phase mirrors (agent 5's "MUST STAY LIVE").
