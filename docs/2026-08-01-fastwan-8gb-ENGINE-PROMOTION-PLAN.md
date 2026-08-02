# fastwan_8gb -- promoting arm C from bench arm to an EQUAL video engine

**Status:** DRAFT for kibitz. No code written. Written 2026-08-01 at HEAD
`e4040897`, branch `v2.0-alpha`.
**Owns:** making FastWan a first-class, dropdown-selectable video engine that is
a peer of `wan_ti2v` / `ltx_8gb` -- not a bench arm, not a special case.
**Does NOT own:** the estimator refit, the physical-8GB qualification, or any
change to `wan_ti2v`'s frozen recipe.

---

## 0. WHY THIS IS ON THE TABLE

The four-arm clamped bench (`docs/GO_FORWARD_PLAN.md`, MEASURED item 7, campaign
at `1a49fdb0`) measured FastWan-as-arm-C against the incumbent:

| | A (incumbent) | C (FastWan) |
|---|---:|---:|
| canvas | 832x480 | 832x480 |
| steps | 30 | **3** |
| worst peak delta | 6563.1 MiB | **6563.1 MiB** |
| 81f wall | 175.7 s | **65.3 s** |
| host RAM | ~10.8 GB | ~14.9 GB |
| licence | apache-2.0 | apache-2.0 |

Identical VRAM to the decimal, 2.76x faster, same canvas, same base weights.
**The operator has now reviewed the renders and picked C.** That was the one gate
the bench could not close, and it is closed.

So the question is no longer "does it work" -- it is "what does it take for
FastWan to be an EQUAL PARTNER in the workflow dropdown", with every receipt,
freeze, contract and profile a shipped engine carries.

## 1. WHAT "EQUAL PARTNER" ACTUALLY MEANS HERE

Grounded from `nodes/_otr_video_engines/registry.py` and the two shipped
image-to-video peers. An engine is a peer when ALL of this is true, and the
value of writing it down is that a half-registered engine is worse than none:

1. **A registered adapter** satisfying `VideoEngine(Protocol)` -- `name`,
   `roles`, `default_roles`, `commercial_clean`, `requires_flag`, `load`,
   `unload`, `family`, `required_inputs`, `invocable`, `invocability_reason`,
   plus the render lifecycle `assert_usable` / `prepare` / `render_clip` /
   `canonicalize` / `teardown`.
2. **A public menu id** in `nodes/_otr_shared/public_engines.py` --
   `_PUBLIC_ENGINES` (which carries a bijection assert), `_PUBLIC_LABEL`, and
   whatever `otr_video_director` builds from them.
3. **A FROZEN recipe with a receipt** -- its own dict + its own env key names +
   its own receipt string, through the ONE mechanism in `wan_recipe.py`
   (each adapter owns its DATA; there is deliberately no central recipe).
4. **A frame contract** (`frame_contract.py`) declaring legal render lengths and
   continuity mode, or it silently resolves to `SINGLE_ONLY`.
5. **A cost declaration** the admission estimator reads (`motion_common`).
6. **Model manifest + profile + variant entries** so the weights are
   discoverable, licence-receipted and tier-selectable.
7. **The canonical workflow updated IN THE SAME COMMIT** -- CLAUDE.md is
   explicit: unwired code is dead code.
8. **Cold-import cleanliness** -- `registry.py` and the adapter module must not
   pull torch/diffusers at module scope (`test_cold_import_no_heavy_libs`).

## 2. THE THREE REAL FORKS -- this is what the panel is for

### FORK 1. Its own adapter, or a second recipe inside `eng_wan_ti2v`?

FastWan is the SAME base GGUF, encoder, VAE, canvas and frame contract as
`wan_ti2v`. It differs by: a rank-128 LoRA, 3 steps not 30, cfg 1.0 not 5.0, and
a restart transition instead of euler.

**Case for a recipe variant inside `eng_wan_ti2v`:** the weights are literally
identical, so a second adapter duplicates loader/VAE/init-image logic that
`WanInitImageMixin` already shares. `eng_ltx_av.py:299-312` has in-tree
precedent for a per-beat recipe switch.

**Case for its own adapter (`eng_fastwan_8gb.py`), which is my lead:**
`eng_ltx_8gb.py` is exactly this precedent one level up -- same family, same
mixin, its OWN adapter and recipe, registered as a normal selectable row. More
decisively, `wan_ti2v`'s recipe is FROZEN (the 2026-07-27 freeze) and the
operator's standing rule is that `wan_ti2v` stays sacred; threading a second
sampling contract through a frozen adapter puts the incumbent's receipt at risk
for the benefit of a newcomer. The still-plans R1 also ruled for per-adapter
ownership over central authority.

**Ask the panel:** is the shared-weights argument strong enough to override the
freeze, and if we DO fork the adapter, exactly which of
`WanInitImageMixin` / `MotionEngineBase` / `wan_shared` is reusable as-is versus
needs a seam?

### FORK 2. Where does the DMD restart sampler live? (the load-bearing one)

The transition is not expressible in stock ComfyUI. `sample_euler` is
deterministic; `sample_euler_ancestral_RF` retains a fraction of the previous
latent. The reference predicts x0 and re-noises to the next timestep with FRESH
noise and zero carry-over. Today that lives in
`scripts/bench_helper/otr_bakeoff_helper/__init__.py` as
`OTR_DMDRestartSamplerSelect` + `_dmd_restart_sampler`.

**That package is a DIAGNOSTIC vendored under the CLAUDE.md s0A bench carve-out
and installed into `custom_nodes` by the bench runner. A production engine may
not depend on it.** Options:

- **(a) Move the transition into the OTR pack** as the one owner (e.g.
  `nodes/_otr_video_engines/dmd_sampler.py`), and have the bench helper IMPORT
  it. One implementation, two consumers, bench keeps proving production's code.
  Risk: the bench helper must stay installable standalone -- check whether it
  can import from the OTR pack at ComfyUI load time.
- **(b) The engine never needs a NODE at all** -- if the adapter drives sampling
  in-process it can build `comfy.samplers.KSAMPLER(fn)` directly and the
  function can live in the engine module. Then the bench helper's node is a thin
  wrapper over the same function.
- **(c) Duplicate the function in both.** Rejected on sight -- two copies of a
  sampling transition is exactly how a silent divergence ships.

**Ask the panel:** does `wrapper_bridge` / `render_driver` drive ComfyUI nodes
by class name (forcing a registered NODE) or can an adapter pass a SAMPLER
object in-process (allowing (b))? That answer picks the option.

### FORK 3. Cost model and frame contract -- inherit or declare?

Same weights as `wan_ti2v`, so resident cost is the same (measured: identical
peak delta). But 3 steps not 30, and the bench measured a HIGHER host-RAM delta
(~14.9 GB vs ~10.8) from LoRA patch staging.

**Lead:** inherit `wan_ti2v`'s frame contract verbatim (min 17, quantum 4, max
177 -- it is the same model) and inherit its VRAM cost declaration, but do NOT
inherit any per-step time assumption. GO_FORWARD is explicit that
`FRAME_COST_MODEL` is wrong in both scaling terms and must not be refit off this
bench, so the honest move is to declare the same numbers with a comment naming
the measured host-RAM delta as an open risk rather than inventing a new fit.

**Ask the panel:** is there any consumer that would silently mis-plan if two
engines declare identical cost but one runs a tenth the steps?

## 3. PROPOSED SHAPE (attack this)

- **Internal engine id:** `fastwan_8gb`. **Public menu id:** `fastwan_8gb`
  (identity, exactly like `ltx_8gb`; keeps the `_PUBLIC_ENGINES` bijection
  assert satisfied). **Label:** `FastWan 2.2 TI2V 5B - 8GB (3-step)`.
- **Module:** `nodes/_otr_video_engines/eng_fastwan_8gb.py`, class
  `FastWan8gbEngine(_WS.WanInitImageMixin, _MC.MotionEngineBase)`,
  `family = "image_to_video"`, `commercial_clean = True`,
  `requires_flag = None` (the registry IS the menu).
- **Frozen recipe**, its own data through `wan_recipe`:
  `RECIPE_FASTWAN_8GB = "fastwan22_ti2v_5b_dmd3_i2v_v1"`, dict pinning
  `steps 3`, `cfg 1.0`, `shift 5.0`,
  `sigmas (1.0, 0.757, 0.522, 0.0)`, `transition "dmd_restart"`,
  `lora_strength 1.0`, plus the tiled-decode geometry inherited from the bench
  graph. Its own `_RECIPE_ENV_KEYS` and its own `PREQUALIFICATION_ENV`.
- **Provenance carried in the receipt, not in a comment:** the recipe is pinned
  from the FastVideo CODE PATH (`DmdDenoisingStage`), never a model card, and
  never the civitai usage advice that circulates for these models.
- **Weights:** base `Wan2.2-TI2V-5B-Q5_K_M.gguf` (already shipped for
  `wan_ti2v`) + `Wan2_2_5B_FastWanFullAttn_lora_rank_128_bf16.safetensors`
  (660,874,456 bytes, Kijai/WanVideo_comfy). Licence chain: apache-2.0 at
  FastVideo and at the Wan base, both verified at the source.
- **Canonical workflow:** the dropdown row lands in
  `workflows/otr_canonical.json` in the SAME commit as the code.

## 4. THE CHANGE SET (enumerate it so nothing lands half-wired)

1. `nodes/_otr_video_engines/eng_fastwan_8gb.py` -- the adapter + frozen recipe.
2. The DMD transition's ONE home (Fork 2) + the bench helper reduced to a
   consumer of it.
3. `nodes/_otr_shared/public_engines.py` -- `_PUBLIC_ENGINES`, `_PUBLIC_LABEL`.
4. Registration wherever the other adapters self-register.
5. `workflows/otr_canonical.json` -- the dropdown row + any widget default.
6. `config/profiles/` -- an 8GB profile row (mirror `otr_8gb_wan.json`) and the
   `otr_w45_*` sweep profile if that family is still live.
7. `workflows/variants/` -- the tier variant if `otr_8gb_wan` gets a FastWan peer.
8. MODEL_MANIFEST / model-download docs -- the LoRA with size, SHA-256, source
   repo and licence.
9. Tests: registry membership, public-id bijection, recipe freeze + receipt,
   departure reporting, frame contract, cold-import cleanliness, and a
   sampler-transition test that FAILS if the sampler degrades to euler.
10. Suite + Bug Bible green; canonical workflow SHA re-pinned if it is pinned.

## 5. CONSTRAINTS THAT DO NOT BEND

- **`wan_ti2v` stays sacred.** Its frozen recipe, its receipt and its numbers do
  not move. If the panel's answer to Fork 1 is "recipe variant", it must explain
  how the freeze survives.
- **No fallbacks, no silent degrade.** If the LoRA or the transition is
  unavailable the engine REFUSES; it never falls back to 30-step euler wearing a
  FastWan receipt.
- **One owner for the transition.** No duplicated sampling code.
- **The recipe is pinned from the reference code path**, not a model card and
  not community usage advice.
- **Unwired code is dead code** -- workflow and code in one commit.
- No engine module touched other than the new one (plus the shared seam Fork 2
  chooses).

## 6. WHAT I WANT BACK, RANKED

1. **Fork 2 first** -- it is the one that changes the file layout, and it
   depends on a fact about `wrapper_bridge` / `render_driver` I have not yet
   established. Answer it from the code.
2. Fork 1, with the freeze argument addressed rather than waved at.
3. Fork 3, specifically whether any planner mis-reads two engines with identical
   declared cost and 10x different step counts.
4. Anything in section 4 that is missing a seam -- I expect the profile /
   variant / manifest rows to be under-specified, and I would rather be told
   than discover it at wiring time.
5. Anything in this plan that is simply wrong.
