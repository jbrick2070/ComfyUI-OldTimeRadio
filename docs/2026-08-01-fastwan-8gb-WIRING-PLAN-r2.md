# fastwan_8gb -- WIRING PLAN (r2 input)

> **SUPERSEDED IN PART -- READ THE RULING FIRST:**
> `docs/2026-08-01-fastwan-8gb-r2-KIBITZ-FINAL.md`.
> This file is the INPUT the r2 panel attacked, preserved as the historical record.
> **Section 3's premise is REFUTED:** the incumbent does NOT render production at
> 1472x832. The 2026-07-23 leg's seven surviving clips all ffprobe at 832x480, and
> the live-vs-bench VRAM spread is a measurement-scope change (`2b095143`,
> 2026-07-20), not canvas. **Section 5's ping-pong mechanism is also wrong:**
> `compute_real_frame_budget` never shortens a render -- it raises
> `MotionBudgetError`. Do not build from this file.

**Status:** post-r1, pre-code. No code written. Branch `v2.0-alpha`, HEAD `bb8a19b1`.
**Predecessor:** `kibitz-runs/2026-07-31-fastwan-8gb-engine/r1/final.md` (rulings F1,
F2, F2b, F3 + a 16-row grounded claim ledger).
**This round:** implementability. Three open questions, section 12.

Every claim below carries the file and line it was verified against on this box
(Windows, RTX 5080 Laptop 16 GB, torch 2.10, CUDA 13, sm_120). Anything not
verified is marked OPEN or [ASSUMPTION].

---

## 1. SETTLED BY THE OPERATOR -- do not re-litigate

1. Arm C wins on the renders. The build is authorized.
2. **`wan_ti2v` STAYS** (public menu id `wan_8gb`). `fastwan_8gb` is **additive** --
   not a replacement, not a deprecation.
3. Public menu id: `fastwan_8gb`.
4. `FastWan8gbEngine(WanTi2vEngine)` -- subclass the incumbent (r1 F1).
5. The DMD sampler moves to `nodes/_otr_video_engines/dmd_sampler.py`, registered in
   the pack; the bench helper imports it and deletes its copy (r1 F2).
6. **Ledger telemetry backfill for `wan_ti2v` is AUTHORIZED** (operator, 2026-08-01):
   populate `recipe` / `quant` / `use_lora` / `render_canvas`. This is what the
   adapter REPORTS, never what it renders. The frozen recipe is untouched.
7. **"Temperature" means the video sampler knobs** -- `cfg`, the sigma schedule,
   `lora_strength` -- all recipe-pinned. Not the writer LLM.

## 2. ADAPTER SHAPE

    # nodes/_otr_video_engines/eng_fastwan_8gb.py
    class FastWan8gbEngine(_WT.WanTi2vEngine):
        ...

Grounded: `WanInitImageMixin` (`wan_shared.py:303`) is 15 pure helpers with zero
lifecycle. `WanTi2vEngine` (`eng_wan_ti2v.py:235`) owns `prepare` (beat-scoped UNET
hoist, `on_result` patcher registration, `hoisted_vram_mb` cost correction),
`teardown`, `session_identity`, `_floor_length`, `_planned_length`, `_build_graph`,
`render_clip`, `canonicalize`, `_node_candidates`, `_loader_names`,
`_resolve_render_config`, `_recipe_receipt`, `_tiled_vae`, `_tile_geometry`,
`_negative_prompt`, `assert_usable`, `_aux_loader_files`, `resolve_isolation`.
Re-parenting to the mixin inherits none of that -- roughly 800 lines of duplication.
Subclassing touches `eng_wan_ti2v.py` zero times, so the freeze holds.

Override set:

| member | change |
|---|---|
| `name` | `"fastwan_8gb"` |
| `render_canvas` | see section 3 -- OPEN |
| `RECIPE_FASTWAN_8GB` | `"fastwan22_ti2v_5b_dmd3_i2v_v1"` |
| `_node_candidates` | + `LoraLoaderModelOnly`, `ManualSigmas`, `SamplerCustom`, `OTR_DMDRestartSamplerSelect`; drop `KSampler` |
| `_loader_names` | + the LoRA file |
| `_aux_loader_files` | + the LoRA, so `assert_usable` fails closed when absent |
| `_build_graph` | route UNET through the LoRA loader; replace `KSampler` with `OTR_DMDRestartSamplerSelect` + `ManualSigmas` + `SamplerCustom` |
| `_resolve_render_config` | steps 3, cfg 1.0, sigmas `(1.0, 0.757, 0.522, 0.0)`, `lora_strength` 1.0 |
| `session_identity` | + the LoRA file receipt |
| `_recipe_receipt` | the FastWan frozen string + provenance (section 8) |

**BUILD STEP 1, before any logic:** write the subclass with `pass` bodies, run the
incumbent's tests, and confirm every hook FastWan needs is an overridable method
rather than a module-level constant or a private free function.

## 3. THE CANVAS -- TWO CHANNELS, AND ONLY ONE IS IN THE HANDOFF

**Channel A, the declaration.** `render_driver.declared_render_canvas`
(`render_driver.py:231`) reads `engine.render_canvas` and is applied LAST in
`build_request_from_shot` (`render_driver.py:2555`). Grounded: `ltx_8gb` is the ONLY
adapter in the pack that declares one -- `render_canvas = (512, 288)`
(`eng_ltx_8gb.py:518`). `wan_ti2v` declares nothing, so its canvas is overwritten to
the shared landscape default. Corroborated by `motion_common.py:255`:
`_FRAME_COST_REF_PIXELS = 1472 * 832`, documented as "wan_ti2v render-phase peak
10277 MB @ 17 frames @ 1472x832."

**So the incumbent renders production at 1472x832 while the campaign measured it at
832x480** -- 3.07x the pixels. That is the explanation for the live-vs-bench VRAM gap
logged as unexplained in GO_FORWARD item 8 (live ledger 8251-9811 MB vs bench
6563.1 MiB).

**Channel B, the budget's own fallback dims -- NEW THIS ROUND, not in the handoff.**
`eng_wan_ti2v.py:110-111` hardcodes:

    _TI2V_COST_REF_W = 1472
    _TI2V_COST_REF_H = 832

`_floor_length` uses these when called without explicit dims
(`eng_wan_ti2v.py:735-736`), and passes them to
`_MC.compute_real_frame_budget(free_mb, target, width, height, self.name)`, which
prices frames as `per_frame * (pixels / _FRAME_COST_REF_PIXELS)`
(`motion_common.py:347`).

**Consequence: declaring `render_canvas` on the subclass does NOT change the budget
arithmetic.** A subclass that declares (832, 480) but inherits these constants prices
every frame at 3.07x its true cost, refuses frames it could afford, renders short,
and hands the difference to the ping-pong mirror. Two independent "what canvas is
this" channels; the r1 plan named one. Both must move together, or the engine is
mis-budgeted in the safe direction and silently mirror-heavy.

## 4. FRAME MATH

**Minimum frames: 17.** `FRAME_MOTION_FLOOR = {"wan_ti2v": 17}` with
`_DEFAULT_MOTION_FLOOR = 1` (`motion_common.py:274-275`). A new engine id with no row
gets 1, which lets a length below the 5B VAE quantum reach the model.

    FRAME_MOTION_FLOOR["fastwan_8gb"] = 17          # REQUIRED

**Quantum: 4n+1.** `wrapper_bridge.quantize_frames_4n1` (`wrapper_bridge.py:447-461`)
-- the Wan 2.1 VAE folds 4 frames into 1 latent. Legal rungs: 17, 21, 25 ... 177.
`frame_contract` declares min 17, max 177, quantum 4 (`eng_wan_ti2v.py:259-262`).

**VRAM cost row.** `FRAME_COST_MODEL["fastwan_8gb"] = (7000.0, 185.0)`. Add for
explicitness, but grounded: `_DEFAULT_FRAME_COST` is `(7000.0, 185.0)`, byte-identical
to the `wan_ti2v` row (`motion_common.py:263-268`), so omitting it changes no number
today. `FRAME_COST_MODEL` predicts VRAM only -- no consumer reads step count or plans
wall time -- so an identical row cannot mis-plan FastWan's 3-step execution. Do NOT
refit it (standing ruling).

**Uncovered: host RAM.** FastWan measured 14.7-15.2 GB vs the incumbent's 10.5-11.1 GB
-- roughly +4 GB for LoRA patch data and dequant staging. Nothing models this.

**Do NOT add `fastwan_8gb` to `frame_contract.PLANNING_CAP_ENGINES`** -- "a deliberate
allowlist of ONE, not a rollout", containing only `ltx_8gb`
(`frame_contract.py:274-292`). Adding WAN would turn every beat into a pile of
17-frame renders.

## 5. THE TWO LENGTH PATHS FASTWAN INHERITS

* **Single-clip beat** -> `_floor_length` (`eng_wan_ti2v.py:~690-743`) predicts an
  affordable length from live free VRAM, renders short, and fills the beat with
  `wrapper_bridge.extend_frames_to_target` (`wrapper_bridge.py:499-527`) -- a tiled
  mirror cycle `[0,1,..,N-1,N-2,..,1]`, period `2N-2`, seamless at the join. A
  1-frame render cannot mirror and is repeated (surfaced LOUD).
* **Coverage-planned segment** -> `_planned_length` (`eng_wan_ti2v.py:745-804`)
  renders the planned length whole, consults no predictor, and refuses by name in two
  cases: a length off the declared ladder, or a tier ceiling below the planned length.

## 6. THE SAMPLER

`_dmd_restart_sampler` and `OTR_DMDRestartSamplerSelect` move to
`nodes/_otr_video_engines/dmd_sampler.py`, registered in the pack's root
`NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS`. The bench helper
(`scripts/bench_helper/otr_bakeoff_helper/__init__.py`) imports from there; its copy
is deleted; a source-level guard forbids a second definition. Lazy `comfy`/torch/tqdm
imports (cold-import test).

Grounded on why a registered node and not an in-process object:
`wrapper_bridge._resolve_value` (`wrapper_bridge.py:154-181`) passes non-Wire literals
through untouched, so an in-process `KSAMPLER` COULD be handed to
`SamplerCustom.sampler`. But the bench runs an API-format JSON graph through real
ComfyUI, which resolves by registered class name. One registered node consumed both
ways means the pinned transition has exactly one implementation.

**The fallback must become a refusal.** `otr_bakeoff_helper/__init__.py:326-370`: when
`model.inner_model.model_patcher.get_model_object("model_sampling")` fails, the sampler
logs `noise_scaling=FALLBACK-flow` and **proceeds** with an inline CONST approximation.
Loud, but it still renders, and the bench would grade that as a PASS for a transition
that was approximated. Production: raise NAMED before step 1. Carry the same refusal
into the bench copy.

## 7. SEEDS

Grounded: the incumbent wires `"seed": int(plan.get("seed", 0))` into `KSampler`
(`eng_wan_ti2v.py:866`).

**THE TRAP: `KSampler` takes `seed`; `SamplerCustom` takes `noise_seed`.** If
FastWan's `_build_graph` copies the incumbent's input dict, `seed` lands on a node
that does not read it and every clip renders on `SamplerCustom`'s default. No error,
no log line. The bench pinned `noise_seed: 42` with `add_noise: true`
(`scripts/bench_graphs/arm_c_fastwan_lora_gguf.json:151-152`).

This is a KNOWN repo hazard, not a prediction: `run_video_arm_bakeoff.py:673-682`
already carries a hand-written dual-key reader -- "LTX's SamplerCustom calls it
`noise_seed`; both are recorded" -- and raises if neither key is present. The bench
has a guard. Production has none on the path we are about to build.

## 8. RECIPE, PREQUALIFICATION, PROVENANCE

    RECIPE_FASTWAN_8GB   = "fastwan22_ti2v_5b_dmd3_i2v_v1"
    PREQUALIFICATION_ENV = "OTR_FASTWAN_8GB_PREQUALIFICATION"

Its own `_RECIPE_ENV_KEYS` mapping every frozen knob to its own env name, never shared
with `wan_ti2v` (`wan_recipe.py:76-90`: one shared switch would open both tiers and
stamp `+prequalification` on a clip that rendered frozen).

**Provenance gap, grounded:** `wan_recipe.recipe_receipt` (`wan_recipe.py:105-124`)
returns the frozen string plus departure suffixes and nothing else.
`fastwan22_ti2v_5b_dmd3_i2v_v1` identifies no FastVideo commit, no source blob, no
sigma digest. Pin the upstream revision and a content hash of the sigma list in a
durable manifest and bind that identity into the receipt.

**Licence:** FastVideo and the Wan base both declare apache-2.0, but the exact loaded
Kijai extraction comes from a repo with no repo-level licence file
(`docs/2026-07-31-arm-c-fastwan-BUILD-SPEC.md` s6A). Record the exact chain and the
notice-compliance gap rather than writing "both verified".

## 9. LEDGER INTEGRATION

Measured across all 1474 episode ledgers: the video engine is recorded at
`meta.render_engines` with `histogram`, `by_role`, `by_engine`, `vram_peak_mb`, and a
`per_clip` array carrying `shot_id`, `role`, `delivered_engine`, `recipe`, `quant`,
`use_lora`, `render_canvas`, `vram_peak_mb`, `family`
(`nodes/otr_video_render_batch.py:62,148`).

**On every `wan_ti2v` row shipped to date, `recipe`, `quant`, `use_lora` and
`render_canvas` are all null.** `fastwan_8gb` populates all four from the start:
`recipe` -> `recipe_receipt(...)`; `quant` -> the GGUF quant token (`Q5_K_M`);
`use_lora` -> the LoRA FILENAME, not a bool; `render_canvas` -> the declared canvas.
Per operator ruling 1.6, `wan_ti2v` is backfilled the same way (telemetry only).

## 10. REGISTRATION SET (half-registration fails CI)

Grounded: `registry.audit_engine_roster()` (`registry.py:575-609`) compares
`CAPABILITIES` against `all_engine_names()` and returns `unexpected` for a
registered-but-undeclared engine; `tests/test_frame_contract.py` turns a non-empty
result into a CI failure.

1. `nodes/_otr_video_engines/eng_fastwan_8gb.py` -- the adapter.
2. `nodes/_otr_video_engines/dmd_sampler.py` -- the sampler, registered.
3. `_otr_video_engines/__init__.py` -- the import, wrapped like every other.
4. `registry.CAPABILITIES["fastwan_8gb"]` -- base/encoder/VAE/LoRA in
   `model_requirements`, `requires_vendor` / `needs_fp8_te` per the row format
   (`registry.py:230-262`).
5. `motion_common.FRAME_COST_MODEL` + `FRAME_MOTION_FLOOR` rows.
6. `_otr_shared/public_engines.py`: `_PUBLIC_ENGINES["fastwan_8gb"] = "fastwan_8gb"`
   (identity, like `ltx_8gb`) + a `_PUBLIC_LABEL` row. The bijection assert
   (`public_engines.py:60-64`) must still hold.
7. A profile with the FastWan assets in preflight.
8. `workflows/otr_canonical.json` -- the dropdown row, SAME commit as the code.
9. Roster + contract tests.

**Label expectation, corrected:** `_PUBLIC_LABEL` is "TOOLTIP / DOCS ONLY, never the
combo/saved value" (`public_engines.py:19-21, 51-58`). The dropdown will read
`fastwan_8gb (16:9)`, not the prose label. Making prose appear in the menu is a
separate, explicitly scoped change to the label builder.

## 11. LORA LIFECYCLE

Grounded: `MotionEngineBase._detach_patchers` (`motion_common.py:490-506`) walks
`prepared["patchers"]` and detaches each with `detach(unpatch_all=True)`. An untracked
patcher is never detached -- VRAM nothing will reclaim. The incumbent registers DURING
the graph via `on_result=_register` (`eng_wan_ti2v.py:428-447`) precisely so a later
node raising cannot orphan a patcher.

FastWan must: hoist base AND LoRA once per beat through the same `on_result` path;
track both in `prepared["patchers"]`; include the LoRA in `session_identity`; and
check (not assume) the larger hoist's `hoisted_vram_mb`.

## 12. THE THREE QUESTIONS FOR THIS ROUND

### Q1 -- LOW-VRAM CANVAS (operator-directed)

**Operator directive 2026-08-01:** decide what render canvas is most efficient for the
8 GB / low-VRAM tier, for BOTH `wan_ti2v` and `fastwan_8gb`. The operator's stated
preference: **render small and upscale later -- that is the better low-VRAM path.**
The upscale path is wired, not hypothetical: `nodes/rtx_upscale.py` and
`nodes/otr_post_upscale_procgen_blend.py` are both in the pack.

Rule on, with the code as evidence:
- What canvas the 8 GB tier should actually render at, given 4n+1, /32-legal axes, and
  the 5B VAE's behaviour.
- Whether BOTH canvas channels (section 3, A and B) are covered by the proposal, and
  exactly which constants move. A proposal that moves only `render_canvas` is wrong.
- Whether changing `wan_ti2v`'s canvas invalidates its shipped VRAM figures and tier
  contract (`config/profiles/otr_8gb_wan.json`, `video.max_render_frames`,
  `OTR_WAN_TI2V_MAX_FRAMES`), and what re-proving that costs.
- Whether render-small-then-upscale is actually better here or whether the VAE decode
  and the upscaler's own VRAM eat the saving.

### Q2 -- PING-PONG OR NATIVE LENGTH for a 3-step engine

`_floor_length` renders short and mirror-fills. That design assumed a 30-step render
could not afford the frames in TIME as well as VRAM. FastWan is ~10x cheaper per clip
in wall time (81f in 65.3 s vs 175.7 s). The VRAM half of the argument is unchanged;
the time half is not.

Does a 3-step engine still want to render short and mirror, or render the beat's full
native length? This decides whether FastWan's output is native motion or mirrored
motion for a typical beat. Note the interaction with Q1: if the budget is priced at
3.07x (section 3, channel B), FastWan renders far shorter than it needs to and mirrors
far more than it should -- so Q1 and Q2 are the same defect seen twice.

### Q3 -- SEED STABILITY

Two parts:
1. The mechanical one (section 7): `noise_seed` vs `seed`. Confirm the fix and the
   test that pins it.
2. The open one: the 3-step restart transition draws FRESH noise at each re-noise step
   via `noise_sampler(sigmas[i], s_next)`. Is that path seed-stable across a
   re-render -- does the same `noise_seed` reproduce the same clip? **Unverified.**
   With three steps, one differently-seeded re-noise is a third of the trajectory.
   Rule on whether an unverified answer BLOCKS the long-haul run or merely annotates
   it, and specify the exact test that would settle it.

## 13. CONSTRAINTS THAT DO NOT BEND

- `wan_ti2v`'s frozen RECIPE does not move. (Its TELEMETRY may -- operator ruling 1.6.
  Its CANVAS is Q1's subject and moves only if Q1 says so, with re-proving costed.)
- No fallbacks, no silent degrade. Missing LoRA or transition -> REFUSE by name.
- One owner for the transition.
- Unwired code is dead code -- workflow and code in one commit.
- A bench cell never qualifies an engine. The gate is a canonical-workflow render with
  `RESULT SUCCESS` + `obs_publish OK`.
