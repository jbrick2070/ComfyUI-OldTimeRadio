# S0b handoff -- the routing freeze (still-plans build)

**Written 2026-07-24 by CODER WINDOW A after landing S0a @ `33c4d8cf`.**
Spec of record: `docs/2026-07-25-still-plans-locked-build-spec.md` @
`84328aa1`. Base HEAD when this file lands: `9fe8b626` on `v2.0-alpha`.

## Why this file exists

S0a shipped clean (6432 passed / 27 skipped / 1 xfailed; Bible 17;
audio-byte-identical PASS; canonical byte-identical) and the parity fixture
at `tests/fixtures/still_plan_head_parity.json` locks HEAD's routing
behaviour across 8 configurations x 31 engines. The autonomous coder run
that landed S0a stopped here because S0b, as specified, is a
**cross-module atomic refactor** whose value is entirely in its atomicity:
every consumer must be cut over in the same commit, or the "routing is
frozen FIRST" promise is broken and the still spine can still be validated
against one engine and rendered with another. A half-landed S0b is worse
than no S0b -- the tree looks routed but the defect the whole chunk exists
to close is still open, silently.

## What S0b IS

Per section 11 of the spec:

> S0b | The routing freeze: closed v3 state; `IS_CHANGED`; forwarding
> VideoDirector -> ImageDirector -> MetaBrief/Dispatcher and VideoDirector
> -> ShotLock -> ledger -> VideoRenderBatch -> render policy; ONE resolver
> + engine facts; shared force-map parser; all five resolvers reduced to
> verified lookups; `apply_engine_override` and `_enforce_radio_is_host`
> on frozen state; the frozen-routing prepass before
> `otr_video_render_batch.py:322`; the LTX adapter mismatch gate; ~31 test
> literals migrated; AST/source audit for stray env reads.

## Site inventory (grounded at `9fe8b626`)

### New shared module -- ONE file, stdlib-only, cold-import clean

`nodes/_otr_shared/routing_state.py` (new). NO import of `render_driver`
(cycle risk, spec section 8). Owns:

- The closed v3 schema constants (accepted keys, defaults, hash boundary).
- `parse_force_engine_map(spec: str) -> dict[str, str]` -- moved from
  `render_driver.parse_engine_override:2757`. Same grammar
  (`role=engine,role=engine`, `*=engine`), same public/legacy
  resolution via `_otr_shared.public_engines.resolve_engine_id`, same
  fail-closed on unknown engines. Called ONCE at the capture boundary.
- `engine_facts_for(engine_id) -> {"engine_id", "family", "provider_side"}`
  -- the ONE registry helper the spec requires. Because
  `_radio_is_host_redirect_applies` reads `family` and `provider_side`
  (`render_driver.py:1376-1389`), those two attributes cannot be
  re-derived downstream; every consumer takes a facts dict.
- `resolve_effective_engine_for_role(role, picked_id, routing_state,
  engine_facts) -> internal_engine_id`. Pure. Mirrors today's ordering:
  force map -> radio-host redirect (skipped when
  `enable_humo_hosts=True`, skipped for cloud engines, skipped for roles
  other than `announcer_visual` / `music_visual`).
- `capture_routing_state(picked_video_models) -> routing_state dict`. The
  SOLE env-reading function outside the adapter mismatch gate. Reads
  `OTR_FORCE_ENGINE_MAP`, `OTR_ENABLE_HUMO_HOSTS`, `OTR_ENABLE_LTX_I2V`,
  `OTR_LTX_AV_RECIPE`, `OTR_LTX_AV_UNET`. Parses the force map ONCE;
  malformed -> `effective_video_models = picked_video_models` plus one
  error receipt (preserving `render_driver.py:2792-2799` and
  `otr_image_gen_dispatcher.py:387-396`). Populates `ltx_resolved` only
  when `ltx_audio_in` is effective on any role, via one shared
  `_unet_identity()` normalisation.
- `state_sha256(routing_state) -> str` -- SHA-256 over canonical sorted
  JSON of `routing_state` EXCLUDING `state_sha256`; returns lowercase hex.
- `validate_routing_state(routing_state) -> None` -- raises
  `ValueError` on missing keys, unknown keys, mismatched hash, or
  `policy_version != 3`. Every consumer calls this before it reads
  anything.

### 7 still-plan / routing consumers (spec section 9, table 1)

All cut over ATOMICALLY -- no site is migrated ahead of the others.

| # | File:lines | Change |
|---|---|---|
| 1 | `nodes/otr_video_director.py:353-354` | Emit `policy_version=3` + a `routing_state` block. `_role_aspects` / `_role_talking` continue to read the picked engines (they are display authorities; S2 retires them as routing authorities). Add `IS_CHANGED` classmethod returning `capture_routing_state(picked).state_sha256`. Env reads happen INSIDE the classmethod body -- ComfyUI passes widget values only, so an env flip between queued prompts must invalidate the cache. |
| 2 | `nodes/otr_image_director.py:169-205, 375` | Forward the `routing_state` block on to `image_policy`. `mesh_fodder_roles_from_video_policy` becomes a read of `routing_state.effective_video_models` (still returns the same list). |
| 3 | `nodes/otr_image_gen_dispatcher.py:357-497, 531-534, 542` | Keep the TRI-STATE; swap its basis from `accepts_still` to a read of `routing_state.effective_video_models`. Retire the module's own `_effective_engine_after_force_map` + `_effective_video_engine_for_role` (they are the second env-reading resolver in section 2). `dispatch_images` policy_version check flips 2 -> 3. |
| 4 | `nodes/otr_meta_brief_image_prompt.py:476-715, 1493-2013, 500-525` | `_effective_prompt_engine_for_role` becomes a lookup into `routing_state.effective_video_models`. Mesh fork + aspect reads follow suit. |
| 5 | `nodes/otr_shot_lock.py:723-724, 919-933, 1249-1252, 1306` | `_effective_cast_time_engine` becomes a lookup. Both policy_version checks flip 2 -> 3. Ledger writes the forwarded `routing_state` under `video`. |
| 6 | `nodes/otr_video_render_batch.py:311-322` | The FROZEN-ROUTING PREPASS lands here: call an ordering-fix wrapper that applies `apply_engine_override` BEFORE `validate_and_repair_still_spine`. Also forwards the ledger's `routing_state` into `build_episode_render_policy`. |
| 7 | `nodes/_otr_video_engines/render_driver.py:635-766, 1528-1853, 2516, 2784` | `apply_engine_override` and `_enforce_radio_is_host` read from `routing_state` (via the ledger's stamped `video.routing_state`), never `os.environ`. `_still_spine_requires_scene`, the init-selection branch's LTX-I2V gate (`:1801-1817`) and IA2V portrait gate (`:1709-1721`) become verified lookups. `build_episode_render_policy` (`:2516`) emits `policy_version=3`. |

### The six `_SCENE_INIT_FAMILIES` call sites

Per spec section 9: `render_driver.py:639, 1570, 1623, 1637, 1641, 1660`.
S0b does **not delete** the frozenset -- S3 does. S0b just ensures the
five sites at 1570 / 1623 / 1637 / 1641 / 1660 that live inside
`build_request_from_shot` consume the frozen-state answer (via the
routing-state lookup) rather than re-doing the family test on live env.

### The LTX adapter mismatch gate

`nodes/_otr_video_engines/eng_ltx_av.py` -- add an early gate in
`assert_usable` (called before model load / GPU alloc) that compares the
LIVE `_recipe()` + `_unet_name()` against the ledger's frozen
`routing_state.ltx_resolved` and fails LOUD when they disagree. The gate
is the ONE place besides the capture boundary that may read recipe/UNET
env; every other consumer reads `routing_state.ltx_resolved` (spec
section 8, indirect-live-reads rule).

### The five UNRELATED POLICY_VERSION constants -- DO NOT TOUCH

`nodes/_otr_casting.py:1652`, `nodes/_otr_voice_bank.py:43`,
`nodes/cast_lock.py:494`. They version the casting and voice-bank
policies, not the video policy. Global search/replace on `POLICY_VERSION`
is FORBIDDEN.

### Test literals to migrate (~31)

Per spec section 9:

- `tests/test_image_platform_c1.py` (19)
- `tests/test_remaining_video_contracts.py` (5)
- `tests/test_still_spine_helpers.py` (2)
- `tests/test_video_platform_aseam.py` (2)
- `tests/test_credits_s2_durable_stamps.py` (1)
- `tests/test_hybrid_voice_fit.py` (1)
- `tests/test_still_spine_engine_coverage.py` (1)

Each is a literal `2` on a `policy_version` field; flip to `3`, add a
`routing_state` construction helper if the test builds a policy from
scratch. Bulk of the diff.

## Parity impact on the S0a fixture (named deltas)

Per spec section 10 (transition rule) and the S0a fixture at
`tests/fixtures/still_plan_head_parity.json`:

- **Materialized values**: `effective_engine` should stay IDENTICAL for
  every configuration -- the new resolver is spec'd to mirror the old
  behaviour exactly. Verify per-cell before regenerating.
- **`special_cases.policy_version_v1_empty_models`**: after S0b, v1
  policy is rejected fail-closed (`raises ValueError`). The current
  fixture records the return values today. Update to record the raise
  instead (structured record: `{"raises": "ValueError: ..."}`).
- **`special_cases.incomplete_policy_no_video_models`**: unchanged (a v3
  policy with `video_models=None` is still an incomplete policy the
  dispatcher's tri-state handles). Reconfirm.
- **`special_cases.empty_video_models_dict`** and
  **`empty_engine_id_per_slot`**: reconfirm; no change expected.
- **`authored` and `render_decisions`**: MUST stay byte-identical for
  every cell. The spec is explicit: "S0b does not change any still
  dimension." Any drift is a bug.

Regenerate with:

    .venv\Scripts\python.exe tests\test_still_plan_parity.py --regenerate

then diff the JSON; every non-`special_cases` cell must diff to zero.

## AST / source audit for stray env reads

Every direct `os.environ.get("OTR_FORCE_ENGINE_MAP" ...)`,
`OTR_ENABLE_HUMO_HOSTS`, `OTR_ENABLE_LTX_I2V`, `OTR_LTX_AV_RECIPE`,
`OTR_LTX_AV_UNET`, and every indirect read (a helper that reads them and
returns a routing decision) OUTSIDE:

- `OTR_VideoDirector.direct` / `OTR_VideoDirector.IS_CHANGED` (the
  capture boundary), and
- `LtxAudioInEngine.assert_usable`'s new mismatch gate

is a spec violation and must be replaced with a `routing_state` read or
deleted. Grep starting points:

    grep -rn "OTR_FORCE_ENGINE_MAP\|OTR_ENABLE_HUMO_HOSTS\|OTR_ENABLE_LTX_I2V\|OTR_LTX_AV_RECIPE\|OTR_LTX_AV_UNET" nodes/

Every survivor must be justified.

## Verifications the chunk owes

Per spec section 13:

- `IS_CHANGED` changes for each routing env input and returns the exact
  forwarded `state_sha256`; identical state is hash-stable.
- Every consumer rehashes and fails closed on
  missing/malformed/unknown-key/mismatched state.
- No direct OR indirect routing/activation env read survives outside the
  capture boundary and the adapter mismatch gate.
- Changing recipe/UNET after capture leaves upstream decisions frozen and
  fails the adapter before GPU work.
- Full Windows suite green, Bug Bible green,
  `test_audio_byte_identical` green, canonical byte-identical at
  `5377914B14911B7362D2516BAD3008BB6EF6ACB87C6E13C77C3D4C0D9D8A8C39`.

## Recommended execution order

One coder window, one commit per step, suite green after each, push after
each:

1. Land `nodes/_otr_shared/routing_state.py` + focused tests (schema,
   hash, resolver, capture, force-map parser, engine facts). No other
   consumer touched. Suite must stay green (it will -- nothing reads the
   new module yet).
2. Land the atomic cutover: OTR_VideoDirector emits v3 with
   routing_state and IS_CHANGED; all 7 consumers switch to reading
   routing_state.effective_video_models; the five resolvers become
   lookups; policy_version literals flip 2 -> 3 across production and
   tests; render_driver's `apply_engine_override` and
   `_enforce_radio_is_host` read frozen state.
3. Land the prepass reorder in `otr_video_render_batch.py` so
   `apply_engine_override` runs before `validate_and_repair_still_spine`.
4. Land the LTX adapter mismatch gate in `eng_ltx_av.py:assert_usable`.
5. Regenerate the S0a fixture and verify the delta is exactly the
   named `special_cases` rows (v1 policy now raises). Non-special cells
   must diff to zero.
6. AST / source audit: grep every routing env read; every survivor must
   be at the capture boundary or the mismatch gate.

Estimated wall time for an experienced coder window: 4-6 hours of
careful work with suite runs between steps.

## What S0a already provides for S0b

The parity fixture is the S0b executioner: after the routing freeze
lands, regenerate and diff. Any cell whose `authored` or
`render_decisions` differs from HEAD is a bug S0b introduced; the fixture
tells the coder exactly which (configuration, engine) pair to chase.
