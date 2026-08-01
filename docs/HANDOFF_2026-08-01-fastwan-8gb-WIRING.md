# HANDOFF -- `fastwan_8gb` engine wiring, ready for /kibitz

**Written:** 2026-08-01. **Repo:** `ComfyUI-OldTimeRadio`, branch `v2.0-alpha`.
**Prereq reading:** `docs/GO_FORWARD_PLAN.md` MEASURED items 7, 8, 9;
`kibitz-runs/2026-07-31-fastwan-8gb-engine/r1/final.md` (r1 rulings + grounded
claim ledger); `docs/2026-07-31-arm-c-fastwan-BUILD-SPEC.md`.

**Operator rulings already made -- these are NOT open:**

1. Arm C wins on the renders. The build is authorized.
2. **`wan_ti2v` STAYS** (public menu id `wan_8gb`). `fastwan_8gb` is **additive**.
   Not a replacement, not a deprecation.
3. Public menu id: `fastwan_8gb`.

Everything below is either GROUNDED (verified against this box, with file:line)
or flagged OPEN. Nothing is assumed.

---

## 1. THE ONE THING THAT WILL BREAK THIS BUILD IF MISSED

`fastwan_8gb` **must declare its render canvas.** Grounded:
`render_driver.declared_render_canvas` (`render_driver.py:231-303`) reads
`engine.render_canvas`, and `build_request_from_shot`
(`render_driver.py:2552-2557`) applies it LAST -- but its own comment reads:

> Today exactly one adapter declares one (`ltx_8gb`, 512x288)

`wan_ti2v` declares nothing, so its canvas is overwritten to the shared
landscape default. Second file, same conclusion: `motion_common.py:254-256` sets
`_FRAME_COST_REF_PIXELS = 1472 * 832` and documents it as "wan_ti2v render-phase
peak 10277 MB @ 17 frames @ 1472x832."

**So the incumbent renders production at 1472x832, while the campaign measured
it at 832x480.** That is the explanation for the live-vs-bench VRAM gap logged
as unexplained in GO_FORWARD item 8 (live ledger 8251-9811 MB vs bench
6563.1 MiB; 1472x832 is 3.07x the pixels of 832x480).

    render_canvas = (832, 480)      # 26 x 15 latent cells; /32-legal both axes

Without this line every VRAM number from the four-arm campaign is fiction for
the production path.

**Corollary for the record:** "A and C are VRAM-identical" is a bench-graph
fact at a shared canvas. It does not transfer to a production `wan_ti2v` running
3x the area. A separate defect ticket is owed against `wan_ti2v` for rendering
its low-VRAM tier at the shared landscape default; it is NOT this build's job.

---

## 2. ADAPTER SHAPE (r1 Fork 1, ruled)

    # nodes/_otr_video_engines/eng_fastwan_8gb.py
    class FastWan8gbEngine(_WT.WanTi2vEngine):
        ...

**Subclass the incumbent. Do NOT re-parent to `WanInitImageMixin`.** Grounded:
`WanInitImageMixin` (`wan_shared.py:303`) is 15 pure helpers with zero
lifecycle; `WanTi2vEngine` (`eng_wan_ti2v.py:235`) owns `prepare` (beat-scoped
UNET hoist, `on_result` patcher registration, `hoisted_vram_mb` cost
correction), `teardown`, `session_identity`, `_floor_length`, `_planned_length`,
`_build_graph`, `render_clip`, `canonicalize`, `_node_candidates`,
`_loader_names`, `_resolve_render_config`, `_recipe_receipt`, `_tiled_vae`,
`_tile_geometry`, `_negative_prompt`, `assert_usable`, `_aux_loader_files`,
`resolve_isolation`. Re-parenting inherits none of that -- roughly 800 lines of
duplication.

Subclassing touches `eng_wan_ti2v.py` **zero times**, so the freeze holds.

Override set (and nothing else):

| member | change |
|---|---|
| `name` | `"fastwan_8gb"` |
| `render_canvas` | `(832, 480)` -- see section 1 |
| `RECIPE_FASTWAN_8GB` | `"fastwan22_ti2v_5b_dmd3_i2v_v1"` |
| `_node_candidates` | + `LoraLoaderModelOnly`, `ManualSigmas`, `SamplerCustom`, `OTR_DMDRestartSamplerSelect`; drop `KSampler` |
| `_loader_names` | + the LoRA file |
| `_aux_loader_files` | + the LoRA, so `assert_usable` fails closed when it is absent |
| `_build_graph` | route UNET through the LoRA loader; replace `KSampler` with `OTR_DMDRestartSamplerSelect` + `ManualSigmas` + `SamplerCustom` |
| `_resolve_render_config` | steps 3, cfg 1.0, sigmas `(1.0, 0.757, 0.522, 0.0)`, `lora_strength` 1.0 |
| `session_identity` | + the LoRA file receipt, so a base/LoRA swap invalidates the beat session |
| `_recipe_receipt` | the FastWan frozen string + provenance (section 6) |

**BUILD STEP 1, before any logic:** write the subclass with `pass` bodies, run
the incumbent's tests, and confirm every hook FastWan needs is an overridable
method rather than a module-level constant or private free function. If any is
not, that is the moment to raise it -- not after the graph is written.

---

## 3. THE SAMPLER (r1 Fork 2, ruled)

Move `_dmd_restart_sampler` and `OTR_DMDRestartSamplerSelect` out of the
diagnostic bench helper into:

    nodes/_otr_video_engines/dmd_sampler.py

Register in the pack's root `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS`.
`scripts/bench_helper/otr_bakeoff_helper/__init__.py` imports from there; its own
copy is **deleted**, and a source-level guard forbids a second definition.
Lazy `comfy` / torch / tqdm imports (V-12 cold-import).

Why (a) and not (b): both are mechanically possible --
`wrapper_bridge._resolve_value` (`wrapper_bridge.py:154-181`) passes non-Wire
literals through untouched, so an in-process `KSAMPLER` object CAN be handed to
`SamplerCustom.sampler` (r1 refuted the panel claim that this is impossible).
But the **bench** runs an API-format JSON graph through real ComfyUI, which
resolves by registered class name. One registered node consumed both ways means
the pinned transition has exactly one implementation and the recipe contract is
enforceable on both paths.

The bench VRAM probes (`OTR_BakeoffVramReset` / `Probe` / `Reclaim`) stay OUT of
the shipped pack. The sampler is production machinery; the probes are
measurement machinery. That asymmetry is deliberate.

### The fallback must become a refusal

`otr_bakeoff_helper/__init__.py:326-370`: when
`model.inner_model.model_patcher.get_model_object("model_sampling")` fails, the
sampler logs `noise_scaling=FALLBACK-flow` and **proceeds** with an inline
CONST approximation. It is loud, but it still renders, and the bench would grade
that render as a PASS for a transition that was approximated rather than taken
from the loaded model.

Production: raise NAMED before step 1 if `model_sampling` is missing or has no
`noise_scaling`. Carry the same refusal into the bench copy so the two cannot
diverge.

---

## 4. FRAME MATH -- min frames, quantum, long beats

All grounded.

**Minimum frames: 17.** `FRAME_MOTION_FLOOR = {"wan_ti2v": 17}` with
`_DEFAULT_MOTION_FLOOR = 1` (`motion_common.py:270-274`). A new engine id with no
row gets **1**, which lets a length below the 5B VAE quantum reach the model.

    FRAME_MOTION_FLOOR["fastwan_8gb"] = 17          # REQUIRED

**Quantum: 4n+1.** `wrapper_bridge.quantize_frames_4n1`
(`wrapper_bridge.py:447-461`) -- the Wan 2.1 VAE folds 4 frames into 1 latent, so
`(length - 1) % 4 == 0`. It returns the smallest `4n+1 >= target`, snapping the
floor UP and any `max_frames` DOWN. Legal rungs: 17, 21, 25, ... 49, ... 81.

**VRAM cost row.**

    FRAME_COST_MODEL["fastwan_8gb"] = (7000.0, 185.0)

Add it for explicitness, but know what it does and does not do:
`_DEFAULT_FRAME_COST` is `(7000.0, 185.0)` -- **byte-identical to the `wan_ti2v`
row** (`motion_common.py:262-268`). So omitting the row changes no number today;
the row only protects against a future default change. `FRAME_COST_MODEL`
predicts **VRAM only** -- no consumer reads step count or plans wall time
(`motion_common.py:322-370`), so an identical row cannot mis-plan FastWan's
3-step execution. Do NOT refit it (standing ruling, GO_FORWARD item 7 NEXT 3).

**The uncovered quantity is host RAM.** FastWan measured 14.7-15.2 GB against
the incumbent's 10.5-11.1 GB -- roughly +4 GB for LoRA patch data and dequant
staging. Nothing models this. Anyone running a resident writer LLM beside a
FastWan render needs to know.

**Long beats: ping-pong, not planning cap.** Do NOT add `fastwan_8gb` to
`frame_contract.PLANNING_CAP_ENGINES` -- it is "a deliberate allowlist of ONE,
not a rollout" containing only `ltx_8gb` (`frame_contract.py:274-292`), and
adding WAN would turn every beat into a pile of 17-frame renders.

The two length paths FastWan inherits:

* **Single-clip beat** -> `_floor_length` predicts an affordable render length
  from live free VRAM, renders short, and fills the beat with
  `wrapper_bridge.extend_frames_to_target` (`wrapper_bridge.py:499-527`) -- a
  tiled mirror cycle `[0,1,..,N-1,N-2,..,1]`, period `2N-2`, seamless at the
  join, tiled and trimmed to target. A 1-frame render cannot mirror and is
  repeated (surfaced LOUD).
* **Coverage-planned segment** -> `_planned_length`
  (`eng_wan_ti2v.py:745-804`) renders the planned length **whole**, consults no
  predictor, and refuses by name in two cases: a length off the declared ladder
  (build error), or a tier ceiling below the planned length (a real
  contradiction -- the planner never saw the ceiling because WAN is excluded
  from `PLANNING_CAP_ENGINES`).

**OPEN FOR KIBITZ:** FastWan is 10x cheaper per clip in wall time. Does the
`_floor_length` VRAM prediction still want to render short and mirror, or should
a 3-step engine render the beat's full native length and skip the ping-pong
entirely? The mirror exists because a 30-step render could not afford the frames
in *time as well as VRAM*. The VRAM half is unchanged; the time half is not.
**This is the most consequential open question in this handoff** -- it decides
whether FastWan's output is native motion or mirrored motion for a typical beat.

---

## 5. SEEDS AND DETERMINISM

Grounded: the incumbent's `_build_graph` (`eng_wan_ti2v.py:866`) wires
`"seed": int(plan.get("seed", 0))` into `KSampler`. The seed comes off the plan,
defaulting to 0.

**THE TRAP, and it is silent:** `KSampler` takes `seed`. **`SamplerCustom` takes
`noise_seed`.** If FastWan's `_build_graph` copies the incumbent's input dict,
`seed` lands on a node that does not read it, and every clip renders on
`SamplerCustom`'s own default. No error, no log line -- just a beat whose
segments do not vary the way the plan says. Pin this with a test that asserts
the built graph carries `noise_seed` equal to `plan["seed"]`.

The bench pinned `noise_seed 42` with `add_noise true`.

**OPEN FOR KIBITZ:** the 3-step restart transition draws FRESH noise at each
re-noise step via `noise_sampler(sigmas[i], s_next)`. Whether that path is
seed-stable across a re-render -- i.e. whether the same `noise_seed` reproduces
the same clip byte-for-byte -- is **unverified**. A deterministic same-seed
replay test is owed before the long-haul run. This matters more for a 3-step
engine than a 30-step one: with three steps, one differently-seeded re-noise is
a third of the trajectory.

**On "temperatures":** there is no temperature on the video path. FastWan's
knobs are `cfg` (pinned 1.0 -- one forward pass per step, which is half of why
it is fast), the sigma schedule `(1.0, 0.757, 0.522, 0.0)`, and
`lora_strength` 1.0. If the intent was the writer LLM's sampling temperature,
that is a different subsystem and does not touch this build -- **confirm which
was meant.**

---

## 6. RECIPE, PREQUALIFICATION, AND PROVENANCE

    RECIPE_FASTWAN_8GB   = "fastwan22_ti2v_5b_dmd3_i2v_v1"
    PREQUALIFICATION_ENV = "OTR_FASTWAN_8GB_PREQUALIFICATION"

`_RECIPE_ENV_KEYS` maps every frozen knob to its own env name -- steps, cfg,
sigmas, shift, sampler, scheduler, negative, tiled_vae, vae_tile, vae_overlap,
vae_temporal, vae_temporal_overlap, lora_strength. Its own consent act, never
shared with `wan_ti2v` (`wan_recipe.py:76-90`: one shared switch would open both
tiers and stamp a `+prequalification` receipt on a clip that rendered frozen).

**Provenance gap, grounded:** `wan_recipe.recipe_receipt`
(`wan_recipe.py:105-124`) returns the frozen string plus departure suffixes and
**nothing else**. `fastwan22_ti2v_5b_dmd3_i2v_v1` identifies no FastVideo
commit, no source blob, no sigma digest. Pin the upstream revision and a content
hash of the sigma list in a durable manifest, and bind that identity into the
receipt. Otherwise a future FastVideo change to `denoising_step_list` is
undetectable in a shipped ledger.

**Licence:** FastVideo model and Wan base both declare apache-2.0, but the exact
loaded Kijai extraction comes from a repo with **no repo-level licence file**
(`docs/2026-07-31-arm-c-fastwan-BUILD-SPEC.md` s6A). Record that exact chain and
the notice-compliance gap in the manifest rather than writing "both verified".

---

## 7. LEDGER INTEGRATION -- and a gap worth closing while we are here

Measured 2026-08-01 across all 1474 episode ledgers: the video engine is
recorded at `meta.render_engines`, with `histogram`, `by_role`, `by_engine`,
`vram_peak_mb`, and a `per_clip` array. Each `per_clip` row carries:

    shot_id, role, delivered_engine, recipe, quant, use_lora,
    render_canvas, vram_peak_mb, family

**On every `wan_ti2v` row shipped to date, `recipe`, `quant`, `use_lora` and
`render_canvas` are all `null`.** That is why the live-vs-bench VRAM gap took a
separate investigation to explain -- the field that would have answered it
immediately (`render_canvas`) exists and is empty.

`fastwan_8gb` must populate all four from the start:

* `recipe` -> `recipe_receipt(...)` -- the frozen string or its departure form
* `quant` -> the GGUF quant token (`Q5_K_M`)
* `use_lora` -> the LoRA filename, not a bool (a bool cannot tell you *which*)
* `render_canvas` -> the declared `(832, 480)`, so a future reader can tell a
  bench cell from a production clip without reading the adapter

**OPEN FOR KIBITZ:** whether backfilling those four for `wan_ti2v` is in scope.
It is a small change to a frozen adapter's *telemetry*, not its recipe -- but
"frozen" was stated without that distinction, so it is the operator's call.

---

## 8. REGISTRATION SET (half-registration fails CI)

Grounded: `registry.audit_engine_roster()` (`registry.py:575-609`) compares
`CAPABILITIES` against `all_engine_names()` and returns `unexpected` for a
registered-but-undeclared engine; `tests/test_frame_contract.py` turns a
non-empty result into a CI failure. So:

1. `nodes/_otr_video_engines/eng_fastwan_8gb.py` -- the adapter.
2. `nodes/_otr_video_engines/dmd_sampler.py` -- the sampler, registered.
3. `_otr_video_engines/__init__.py` -- the import (wrapped like every other, so
   a packaging quirk cannot break the namespace).
4. `registry.CAPABILITIES["fastwan_8gb"]` -- **required**; base/encoder/VAE/LoRA
   in `model_requirements`, `requires_vendor` / `needs_fp8_te` / etc. per the row
   format at `registry.py:230-262`.
5. `motion_common.FRAME_COST_MODEL["fastwan_8gb"]` and
   `FRAME_MOTION_FLOOR["fastwan_8gb"] = 17`.
6. `_otr_shared/public_engines.py`:
   `_PUBLIC_ENGINES["fastwan_8gb"] = "fastwan_8gb"` (identity, like `ltx_8gb`)
   and a `_PUBLIC_LABEL` row. **The bijection assert at `public_engines.py:60-64`
   must still hold** -- unique internals, no collapse.
7. A profile with the FastWan assets in preflight.
8. Roster + contract tests.

**Label expectation, corrected:** `_PUBLIC_LABEL` is "TOOLTIP / DOCS ONLY, never
the combo/saved value" (`public_engines.py:19-21, 51-58`). The dropdown will
read **`fastwan_8gb (16:9)`**, not "FastWan 2.2 TI2V 5B - 8GB (3-step)". If the
prose label should appear in the menu, that is a separate, explicitly scoped
change to the label builder -- do not assume adding the row achieves it.

**On the `_8gb` suffix:** `wan_8gb` and `ltx_8gb` already ship under this
convention with no physical-8 GB-card qualification either. Either the suffix
means "8 GB tier" and `fastwan_8gb` fits, or two shipped engines are already
mislabelled. Not a FastWan-specific defect; flagged for the operator, not
blocking.

---

## 9. LORA LIFECYCLE (V-4)

Grounded: `MotionEngineBase._detach_patchers` (`motion_common.py:490-506`) walks
`prepared["patchers"]` and detaches each with `detach(unpatch_all=True)`. An
untracked patcher is never detached -- it is VRAM nothing will reclaim.

The incumbent registers **during** the graph, not after:
`eng_wan_ti2v.prepare` passes `on_result=_register` to `run_graph`
(`eng_wan_ti2v.py:428-447`) precisely so "if a later node raised, `run_graph`
never returns a results dict and an unregistered patcher is VRAM nothing will
ever detach."

FastWan must:

1. Hoist base **and** LoRA once per beat through the same `on_result` path.
2. Track both handles in `prepared["patchers"]`.
3. Include the LoRA in `session_identity` so a strength or file change opens a
   new beat session.
4. Account for the larger hoist in `hoisted_vram_mb` -- `prepare` measures free
   VRAM across the loader graph and hands the delta back to the budget so
   `_floor_length` does not charge the resident model twice. FastWan's hoist is
   bigger; the mechanism is inherited, but the number must be checked, not
   assumed.

---

## 10. TEST SET

* Subclass override surface: FastWan changes only the intended members;
  `wan_ti2v`'s behaviour is pinned unmoved.
* `noise_seed` (not `seed`) is wired from `plan["seed"]`. Section 5.
* Declared `render_canvas == (832, 480)` and equals the profile's canvas
  (the profile channel owes a DRIFT GUARD, not authority --
  `render_driver.py:246-258`).
* Missing LoRA -> `assert_usable` refuses by name.
* Missing / incompatible `model_sampling` -> the sampler refuses by name.
* Sigma literal and step count agree; the recipe contract cannot drift.
* Roster audit clean (`missing` and `unexpected` both empty).
* Repeated-session teardown: no VRAM or host-RAM growth across clips, including
  switching between `wan_ti2v` and `fastwan_8gb` in one session.
* Deterministic same-seed replay -- **owed, see section 5**.
* Canonical workflow round-trip through the registered adapter with
  `RESULT SUCCESS` + `obs_publish OK`. A bench cell does not qualify an engine
  (CLAUDE.md s0A; four-arm SPEC s12-13).

---

## 11. WHAT THE KIBITZ SHOULD RULE ON

1. **Ping-pong or native length for a 3-step engine** (section 4). The highest
   consequence question here.
2. **Seed stability of the restart transition** (section 5) -- and whether an
   unverified answer blocks the long-haul run or merely annotates it.
3. **Backfilling the four null ledger fields for `wan_ti2v`** (section 7) --
   telemetry-only change to a frozen adapter: in scope or not.
4. **Whether "temperature" in the long-haul brief meant the writer LLM** or the
   video sampler knobs (section 5).
5. **The `wan_ti2v` production-canvas defect** (section 1) -- log it, fix it in
   this build, or fix it separately. It is the reason the incumbent's live peak
   sits near 10 GB, so a long-haul comparison that ignores it compares a
   832x480 FastWan against a 1472x832 Wan.
