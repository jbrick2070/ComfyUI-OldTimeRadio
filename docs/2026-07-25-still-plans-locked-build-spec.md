# Still plans -- LOCKED BUILD SPEC

**Locked 2026-07-25 at HEAD `aa2d4a15`, branch `v2.0-alpha`.**
Self-contained: everything a builder needs is here. The r1-r5 kibitz
judgments live in `kibitz-runs/2026-07-24-still-plans{,-r5}/` (gitignored,
local) and are review history, not build input.

Arc: r1 -> r5, local panel (codex `gpt-5.6-sol` high + agy
`Gemini 3.6 Flash (High)`), Cowork Claude as grounded panelist and judge.
Converged at r5: agy "architecture and chunking have fully converged"; codex
raised no architectural objection, only spec precision (folded in below).

## 1. THE OPERATOR'S DECISION (authoritative, do not re-derive)

> "each video engine needs a separate set of instructions and prompts about
> what kind of images it needs, and the actual image gen is specified
> separately in the image gen dropdown"
>
> "we don't need complex lip logic -- I feel it built an overly complicated
> architecture around this when I should have just built a completely
> independent still plan for each video model"

Every video model owns ONE independent still plan: how many images, what
kind, what shape, required or not, plus the framing text for its own kinds.
Roles are a consequence of the plan, never its key. A plan never references a
role.

An earlier cross-cutting "engine image contract" framing is SUPERSEDED and
must not be revived: no `ResolvedBeat`/`TargetKey`, no three-reader exact-set
equality protocol, no contract hash in the image cache key, no
routing-snapshot subsystem, no model x role or model x beat-class registry,
no plan-per-recipe, no predicate DSL, no inheritance or "shared defaults", no
new ComfyUI node or widget.

## 2. THE ROOT CAUSE, MEASURED

The block was framed as "five role-indexed places disagree about what images a
model needs". Grounding at HEAD shows something simpler and worse:

> **FIVE MODULES INDEPENDENTLY RE-DERIVE "WHICH ENGINE IS EFFECTIVE", FROM
> LIVE ENVIRONMENT, AT FIVE DIFFERENT MOMENTS -- AND THE STILL-SPINE
> VALIDATOR RUNS BEFORE ANY OF THEM.**

| # | Module | What it resolves |
|---|---|---|
| 1 | `otr_video_director.py:399-440` (`_role_aspects`, `_role_talking`) | the **PICKED** engine only -- no force map, no redirect |
| 2 | `otr_image_gen_dispatcher.py:399-425` (`_effective_video_engine_for_role`) | force map -> radio redirect, hand-rolled |
| 3 | `otr_meta_brief_image_prompt.py:500-525` (`_effective_prompt_engine_for_role`) | delegates to 2, but is its own seam |
| 4 | `otr_shot_lock.py:919-933` (`_effective_cast_time_engine`) | reads `os.environ` and re-implements both steps |
| 5 | `render_driver.py:1439` (`_enforce_radio_is_host`) + `:2751` (`apply_engine_override`) | the authoritative render-time answer |

And the ordering defect: `otr_video_render_batch.py:322` calls
`validate_and_repair_still_spine` **before** `run_real_episode` reaches
`apply_engine_override` at `render_driver.py:2751`. **With a force map set,
the still spine is validated against the picked engine and rendered with the
forced one.**

Two facts that explain why this survived:

- `otr_video_render_batch.py:311-321` SKIPS the validator entirely when
  `OTR_TEST_MODE=1` and no target receipt is present. The suite never
  exercised the ordering.
- `apply_engine_override` is idempotent per shot (`if not engine or
  shot.get("engine_id") == engine: continue`), so hoisting it earlier is safe
  -- but only while the environment cannot change between the two calls.
  Reading frozen state is what makes that unconditional.

**Therefore routing is frozen FIRST, and the plan table is wired to it.**

## 3. WHAT EACH ENGINE ACTUALLY NEEDS (three shapes, 31 engines)

Measured by driving the real producer per engine; full record in
`docs/STILL_PLAN_SEED_INVENTORY.md` (mechanism map + parity matrix).

**Shape A -- scene spine (26 engines).** `scene_open` per episode +
`scene_beat` / `scene_character` per beat, WIDE 1472x832, plus a `portrait`
per subject at the engine's aspect.

`wan_ti2v`, `wan_i2v`, `ltx_8gb`, `ltx_video`, `ltx_audio_in`, `still_pan`,
`still_flat`, `still_word`, `still_motion`, `word_razzle`,
`cloud_kling_avatar`, `cloud_seedance_2`, `cloud_vidu_q2_pro_fast_720p`,
`cloud_vidu_q2_pro_fast_720p_sfx`, `cloud_wan_i2v`, `cloud_wan_i2v_audio`,
`google_veo_video`, `google_omni_video`, `google_vid_sfx_omni`,
`google_vid_sfx_veo_lite`, `google_vid_sfx_veo_fast`,
`google_vid_sfx_veo_pro`, `humo`, `humo_1.7B`, `humo_1.7B_169`,
`humo_14B_169`.

**Shape B -- mesh fork (`mesh_stage`).** `mesh_fodder` per beat
(`target_class=mesh`, `style_tail_policy=minimal_clean`) +
`scene_background_plate` per beat (`target_class=scene`), both
`required=always`, plus the portrait. No cinematic scene still.

**Shape C -- nothing (`viz_camera`, `viz_green`, `viz_mxc_cpu`,
`viz_mxc_mandala`).** `still_plan = ()`, explicitly empty.

Differences WITHIN Shape A are columns, not tables:

| Engine(s) | Difference |
|---|---|
| `humo`, `humo_1.7B` | portrait 832x1216 (`aspect=inherit_engine`) |
| `humo_1.7B_169`, `humo_14B_169` | portrait 832x480 (`aspect=inherit_engine`) |
| the 4 HuMo + `cloud_kling_avatar` | portrait `required=always` (today keyed on family `audio_driven_face` inside the validator) |
| `ltx_video` | scene still `required=when_ltx_i2v_enabled` |
| `ltx_audio_in` | + WIDE radio face per bookend role, + cast portrait on character beats, both `required=when_engine_talking` |
| `still_*`, `viz_*` | requiredness DECLARED, never derived from `required_inputs` |

## 4. THE PROMPT LAYER LAW (operator, hard)

Every composed prompt is three layers and stays three layers:

1. **SUBJECT** -- meta brief / story-pack `open_subjects` / cast appearance.
2. **FRAMING + GEOMETRY** -- the ONLY layer a plan owns.
3. **STYLE TAIL** -- the visual-style authority, resolved once at the
   image-prompt entry.

A plan may never replace the subject or the style, and never decides style.
`prompt_field_source` provenance stamps survive. The mesh minimal tail is
expressed as `style_tail_policy=minimal_clean`, which the style authority
honours by emitting a reduced tail -- the plan names a policy, never tokens.

Fix while transplanting: the `portrait` kind is the one kind carrying
`prompt_field_source = None` and `visual_style = None`.

## 5. THE StillPlanRow SCHEMA (closed)

```
StillPlanRow = (
    kind,               # enum: portrait | scene_open | scene_beat |
                        #       scene_character | mesh_fodder |
                        #       scene_background_plate
    cardinality,        # enum: per_beat | per_subject |
                        #       per_recurring_subject | per_bookend_role
    target_class,       # enum: scene | portrait | mesh
    aspect,             # enum: wide | portrait | inherit_engine
    required,           # enum: always | never |
                        #       when_engine_talking | when_ltx_i2v_enabled
    framing_geometry,   # authored TEXT -- the ONE free field (layer 2 only)
    style_tail_policy,  # enum: full | minimal_clean      (default: full)
)
```

`required` is a closed activation enum, never a boolean and never an
expression. `when_engine_talking` evaluates the engine's own
`wants_talking_prompt()` hook; `when_ltx_i2v_enabled` evaluates the FROZEN
`routing_state.enable_ltx_i2v`. Adding a fifth token is an operator decision.

`aspect="inherit_engine"` is never compared directly. Every reader resolves
through one pure helper exported from `nodes/_otr_shared/still_plan_helpers.py`:

```
resolve_row_aspect(row, engine_facts) -> "wide" | "portrait"
```

`target_class == "scene"` is what satisfies the validator's scene slot, and
`scene_background_plate` carries it. Grounded: `_still_spine_row_for_beat`
(`render_driver.py:586-597`) matches `kind.startswith("scene_")` then applies
explicit plate-over-scene precedence -- `plates = [k ==
"scene_background_plate"]; return (plates or matches)[-1]`. A cutover matching
only the three cinematic scene kinds breaks every mesh beat.

## 6. DECLARATION AND AUDIT

`still_plan: tuple[StillPlanRow, ...]` is a CLASS ATTRIBUTE on each adapter in
`nodes/_otr_video_engines/eng_*.py`, adjacent to `family` / `render_aspect` /
`accepts_still`. Plan keys are INTERNAL engine ids; normalize through
`nodes/_otr_shared/public_engines.py` first (4 public ids + 4 legacy aliases
resolve INTO the 31 and never carry their own plan).

**Validation is a POST-REGISTRATION AUDIT, never decorator-time.**
`nodes/_otr_video_engines/__init__.py` wraps every adapter import in
`try: ... except Exception: pass` (`:94-209`), so a `ValueError` raised in a
class body or the `@register` decorator would SILENTLY DELETE the engine from
the menu -- a plan typo would present as "that model disappeared".

The audit runs outside those guards and compares THREE sets for exact
equality: `registry.CAPABILITIES` keys, `all_engine_names()`, and the set of
ids owning a valid `still_plan`. Comparing registered-to-registered proves
nothing, because an adapter whose import was swallowed never reaches
`all_engine_names()`. `tests/test_capability_profiles.py:384-386` already
exercises the independent-roster invariant.

**A missing plan is UNKNOWN and fails closed. An explicit `()` means "needs no
images".** Collapsing the two would make a forgotten declaration look like a
visualizer.

## 7. THE v3 ROUTING STATE (closed schema)

```
policy_version: 3
video_models:   dict[slot, {...}]      # PICKED -- retained for audit
routing_state: {
    force_engine_map:       dict[str, str],   # normalized, parsed ONCE
    enable_humo_hosts:      bool,
    enable_ltx_i2v:         bool,
    ltx_resolved:           {resolved_recipe: str, unet_identity: str} | null,
    effective_video_models: dict[role, internal_engine_id],   # AUTHORITATIVE
    state_sha256:           str,   # lowercase hex
}
```

- `state_sha256` = SHA-256 over canonical sorted JSON of `routing_state`
  EXCLUDING `state_sha256` itself. It carries no widget selections: downstream
  consumers receive only `routing_state` and must be able to recompute it, and
  ComfyUI already invalidates on widget change.
- Top-level `video_models` stays the PICKED map (audit value);
  `routing_state.effective_video_models` is authoritative for every consumer.
- `ltx_resolved` is populated only when `ltx_audio_in` is effective.
  `unet_identity` uses one shared normalization of `_unet_name()`.
- Unknown keys are rejected. Every consumer rehashes and fails closed on
  missing, malformed, unknown-key, or mismatched state.

`OTR_VideoDirector` is the SOLE capture boundary.

`OTR_VideoDirector.IS_CHANGED` is a `@classmethod` returning `state_sha256`.
It must read the environment INSIDE the method body -- ComfyUI passes only
widget values, so an env flip between queued prompts would otherwise serve a
cached `video_policy_json`. It samples `OTR_FORCE_ENGINE_MAP`,
`OTR_ENABLE_HUMO_HOSTS`, `OTR_ENABLE_LTX_I2V`, `OTR_LTX_AV_RECIPE`,
`OTR_LTX_AV_UNET`.

## 8. THE SINGLE RESOLVER

One pure function in `nodes/_otr_shared/` (stdlib-only, cold-import clean,
NO module-scope import of `render_driver` -- cycle risk):

```
resolve_effective_engine_for_role(role, picked_id, routing_state,
                                  engine_facts) -> internal_engine_id
```

`engine_facts` is ONE closed descriptor `{engine_id, family, provider_side}`
produced by ONE registry helper, because `_radio_is_host_redirect_applies`
depends on family and `provider_side` (`render_driver.py:1376-1389`) and those
cannot be derived from an id without re-creating the HuMo/cloud classification
table. Never duplicate that table.

All five sites in section 2 become verified lookups against
`routing_state.effective_video_models`, not re-derivations.
`apply_engine_override` and `_enforce_radio_is_host` read the frozen state,
not `os.environ`. The force map is parsed once at the capture boundary; a
malformed map yields the PICKED engine plus one captured error receipt --
preserving today's behaviour at `render_driver.py:2792-2799` and
`otr_image_gen_dispatcher.py:387-396` -- and is never re-parsed downstream.

**Indirect live reads count.** `wants_talking_prompt()` is called at
`otr_meta_brief_image_prompt.py:529-572` and `render_driver.py:969-978`, and
reaches `eng_ltx_av.py:390-412`, which re-reads recipe and UNET from the
environment by deliberate design. Downstream prompt, ShotLock and
request-building decisions consume `routing_state.ltx_resolved`; live
recipe/UNET reads are permitted ONLY at the capture boundary and at the
adapter mismatch gate, which fails closed before model load or GPU
allocation.

## 9. SITE INVENTORY

**Still-plan consumers (seven, all cut over atomically in S2):**

| # | Site | Disposition |
|---|---|---|
| 1 | `otr_video_director.py:399-440` | read the plan; aspect/talking stop being authorities |
| 2 | `otr_image_director.py:169-205` | `mesh_fodder_roles_from_video_policy` -> `any row.target_class == "mesh"` |
| 3 | `otr_image_gen_dispatcher.py:357-497` | keep the TRI-STATE; swap its basis from `accepts_still` to the plan |
| 4 | `otr_meta_brief_image_prompt.py:476-715, 1493-2013` | mesh fork + aspect reads become plan reads |
| 5 | `render_driver.py:635-766` | `_still_spine_requires_scene` becomes a plan read |
| 6 | `render_driver.py:1528-1853` | the init-selection branch: the LTX-I2V gate (`:1801-1817`) and the IA2V portrait gate (`:1709-1721`) become `required` enum evaluations |
| 7 | `otr_shot_lock.py:723-724` | follows site 3 |

`derive_scene_still_targets` (`otr_meta_brief_image_prompt.py:1002-1106`)
stays ENGINE-BLIND. The architecture is ENUMERATE-then-FILTER: the producer
enumerates from LINES + `SPEAKER_TO_VIDEO_ROLE`, and the dispatcher decides
what is minted. `tests/test_still_spine_engine_coverage.py` (landed
`9d1874f1`) already proved enumeration is not the hole. **The plan is applied
at the FILTER, not at the enumerator.**

**`_SCENE_INIT_FAMILIES` -- six call sites, all replaced in S2 before S3
deletes the frozenset:** `render_driver.py:639, 1570, 1623, 1637, 1641, 1660`.
Several are inside `build_request_from_shot` -- the render path, not the
validator. Deleting without replacing all six is a `NameError` at render.

**`policy_version` -> 3 (five production sites):**
`otr_video_director.py:353-354` (constructs),
`otr_image_director.py:375` (forwards),
`otr_shot_lock.py:1249-1252, 1306`,
`otr_image_gen_dispatcher.py:531-534, 542`,
`render_driver.py:2516` (`build_episode_render_policy`, currently hardcodes 2).

**TRAP: three UNRELATED `POLICY_VERSION` constants must NOT be touched** --
`nodes/_otr_casting.py:1652`, `nodes/_otr_voice_bank.py:43`,
`nodes/cast_lock.py:494`. They version the casting and voice-bank policies,
not the video policy. A careless global replace breaks them.

**Test policy literals to migrate (~31):** `tests/test_image_platform_c1.py`
(19), `tests/test_remaining_video_contracts.py` (5),
`tests/test_still_spine_helpers.py` (2), `tests/test_video_platform_aseam.py`
(2), `tests/test_credits_s2_durable_stamps.py` (1),
`tests/test_hybrid_voice_fit.py` (1),
`tests/test_still_spine_engine_coverage.py` (1). This is the bulk of S0b's
diff.

## 10. THE PARITY CONTRACT

**Three outputs per engine, because they disagree today:**

1. **Authored** -- `derive_image_prompts` objects + `required_scene_targets`.
2. **Materialized** -- `still_consumer_capabilities`, `roles_requiring_stills`.
3. **Render-validated** -- what the render path validates or RAISES on:
   `_still_spine_requires_scene`, the `requires_mesh_fodder` branch, the
   `family == "audio_driven_face"` portrait branch, the LTX-I2V gate, and the
   IA2V portrait gate.

**Configuration matrix, not just the default env** -- the change is a ROUTING
change: hosts off/on; `*=viz_*`; forced `ltx_audio_in`; cloud avatar excluded
from the local redirect; LTX-I2V off/on; IA2V vs a non-talking recipe;
malformed force map; incomplete policy.

The fixture must carry a real `required_scene_targets` receipt so it never
inherits the `OTR_TEST_MODE` validator skip at
`otr_video_render_batch.py:311-321`.

**Expectation transitions are explicit per chunk. No baseline is ever
silently rewritten.**

- S0a asserts HEAD `aa2d4a15` exactly, HuMo before-state included.
- S0b flips ONLY the policy-v3 / routing deltas. **S0b does not change any
  still dimension.**
- S2 flips ONLY the four named HuMo rows.

Parity is defined over normalized authored/materialized/render DECISIONS,
never whole policy or ledger bytes.

## 11. CHUNKS

| Chunk | Content | Proves |
|---|---|---|
| **S0a** | Characterization tests only: 3 outputs x 31 engines x the configuration matrix, asserting HEAD exactly. No production code. Files: `tests/test_still_plan_parity.py`, `tests/fixtures/still_plan_head_parity.json`. Seed: `tmp/_kbA_sp_parity.py` (default env only -- extend it). | The pre-change truth is committed before anything moves. |
| **S0b** | The routing freeze: closed v3 state; `IS_CHANGED`; forwarding VideoDirector -> ImageDirector -> MetaBrief/Dispatcher and VideoDirector -> ShotLock -> ledger -> VideoRenderBatch -> render policy; ONE resolver + engine facts; shared force-map parser; all five resolvers reduced to verified lookups; `apply_engine_override` and `_enforce_radio_is_host` on frozen state; the frozen-routing prepass before `otr_video_render_batch.py:322`; the LTX adapter mismatch gate; ~31 test literals migrated; AST/source audit for stray env reads. | Forced/redirected routes validate against the engine that renders them. Fail-closed rejection of missing/v2/incomplete policy is an INTENTIONAL, named change. |
| **S1** | Schema + 31 `still_plan` declarations + `resolve_row_aspect` + the post-registration audit. Nothing reads the plan. | `CAPABILITIES == all_engine_names() == valid-plan owners`; missing differs from `()`. |
| **S2** | Atomic cutover of all seven sites + all six `_SCENE_INIT_FAMILIES` references. HuMo expectations flip. **OPERATOR EYEBALL: 832x1216 -> 832x480 on HuMo announcer/music.** | Bug fixed; parity holds elsewhere. |
| **S3** | Delete the shims and the stale degrade-chain prose. | Nothing reads the old authorities. |
| **S4 (LIVE)** | Two SEPARATE reset/boot cycles -- the force map is process environment. Boot A: no routing overrides. Reset, Boot B: `OTR_FORCE_ENGINE_MAP=announcer_visual=humo,music_visual=humo` with `OTR_ENABLE_HUMO_HOSTS=0`, LTX recipe/UNET pinned. Run `workflows/otr_canonical.json` in both. | Effective bookend engine `ltx_audio_in`, 832x480 stills, the still-spine receipt captured, `Prompt executed`, `obs_publish OK`, episode/OBS assets on disk. Unit parity does not prove this. |

**Stale degrade-chain prose to fix in S3** (the NO-FALLBACKS rip of
2026-07-02 left these behind): `eng_humo.py:487-498, 540-552, 563-576`,
`_otr_video_engines/__init__.py` (the still_parallax paragraph),
`cheap_families.py:123-131`, `otr_meta_brief_image_prompt.py:1065-1072`,
`render_driver.py:1685-1696`.

## 12. HARD ASSERTIONS

- S0a-S3 change no `INPUT_TYPES`, `RETURN_TYPES`, node mapping, widget, link,
  socket or `widgets_values`. `workflows/otr_canonical.json` stays
  byte-identical at SHA-256
  `5377914B14911B7362D2516BAD3008BB6EF6ACB87C6E13C77C3D4C0D9D8A8C39`.
  Discovering a new socket requirement REOPENS THE PLAN; it does not
  authorize an opportunistic interface change.
- **THE LAW holds:** an audit may improve a story, never fail one for length,
  language, style, visual vocabulary or quality. A missing REQUIRED image is
  structural and stays fail-closed.
- **NO FALLBACKS:** no substitute asset, no scene still as mesh fodder, no
  text-only or dark-floor degradation, no silent resize.
- New shared helpers stay stdlib-only and cold-import clean.
- Every chunk: focused tests + full Windows suite + Bug Bible +
  AST/BOM/zero-byte/UTF-8 + canonical hash, `git commit -F`, pathspec only,
  push to `v2.0-alpha`, verify `HEAD == origin`.
- UTF-8 no BOM, SFW, never the word "dummy".

## 13. VERIFY AT BUILD

- [ ] S0a captured from `aa2d4a15` before any production edit; exercises a
      real `required_scene_targets` receipt; bypasses no validator.
- [ ] Named post-S0b and post-S2 transitions explicit; no silent rebaseline.
- [ ] v3 required and forwarded through both chains; no production consumer
      still asserts `policy_version == 2`.
- [ ] `IS_CHANGED` changes for each routing env input and returns the exact
      forwarded `state_sha256`; identical state is hash-stable.
- [ ] Every consumer rehashes and fails closed on missing/malformed/unknown-key
      /mismatched state.
- [ ] No direct OR indirect routing/activation env read survives outside the
      capture boundary and the adapter mismatch gate.
- [ ] Changing recipe/UNET after capture leaves upstream decisions frozen and
      fails the adapter before GPU work.
- [ ] All seven sites and all six `_SCENE_INIT_FAMILIES` references cut over
      atomically before any old authority is removed.
- [ ] Three-output parity across the full configuration matrix; only named
      deltas move.
- [ ] HuMo correction is exactly 832x1216 -> 832x480 on announcer/music; the
      `_169` siblings unchanged.
- [ ] Canonical workflow byte-identical; `OTR_WorkflowValidator`, JSON
      round-trip, link integrity and widget-vector audits pass.
- [ ] Both fresh-boot live legs publish real assets within the 16 GiB budget.
