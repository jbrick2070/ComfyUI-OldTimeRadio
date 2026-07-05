# RIP qwen_image (image ENGINE) -- 100% clean-break plan

**Status: STAGED, NOT STARTED.** Execution GATED on the operator's other coder
finishing (one-window-at-a-time via docs/GO_FORWARD_PLAN.md). Do NOT edit code
until the operator says go.

## Why (operator 2026-07-05)
qwen_image is the worst VRAM-risk-to-install-base ratio in the menu:
- **VRAM:** 20B core, Q3/Q4 GGUF ~8-11 GB + VAE resident, ~13-15 GB peak -- TIGHT
  on the 14.5 GB single-resident ceiling, and the per-quant peak was NEVER
  measured on the 5080 (qwen_image.py: "must be confirmed first, 20B is tight").
- **Not actually ubiquitous on consumer HW:** its size is exactly why most 16 GB-
  and-under users don't run it. Prestige != install base.
- **Never worked locally here:** no GGUF/TE/VAE installed -> it only ever produced
  a deep fail-closed error (the bake-off FileNotFoundError). A selectable engine
  that always deep-fails is an adoption liability (the dispatcher-preflight gap).
- Adoption strategy: local menu = ubiquitous low-VRAM models (Flux family,
  z_image, lumina, flux2_klein). Prestige/heavy models ride the CLOUD lane.

## CRITICAL -- DO NOT TOUCH (these are NOT the engine)
The token "qwen" appears in TWO unrelated, load-bearing places that MUST survive:
1. `z_image_turbo.py:79` `_DEFAULT_CLIP_TYPE = "qwen_image"` -- this is the ComfyUI
   **CLIPLoader `type` string** for Z-Image's Qwen3-4B text encoder (verified
   2026-06-18: the live /object_info enum has no "z_image"; "qwen_image" is the
   type that renders). Ripping the ENGINE must not touch this string. Also
   `z_image_turbo.py:242` `"type": params["clip_type"]`.
2. The **`qwen_3_4b*.safetensors` encoder files** on disk -- z_image_turbo AND
   flux2_klein both use `qwen_3_4b.safetensors` as their TE (flux2_klein.py:
   "text encoder qwen_3_4b.safetensors"). KEEP the files; KEEP those references.

## Site map (grounded 2026-07-05)
ENGINE (rip):
- `nodes/_otr_image_engines/qwen_image.py` -- the whole adapter (DELETE file).
- `nodes/_otr_image_engines/__init__.py:44-48` -- `from . import qwen_image ...`
  import + its comment (REMOVE).
- `nodes/_otr_image_engines/registry.py:124` -- the `"qwen_image": {...}`
  CAPABILITIES row (REMOVE).
- `nodes/_otr_image_engines/hidream_i1.py:5` -- docstring mentions qwen_image as a
  peer (cosmetic; reword, non-blocking).
- Env vars die with the file: OTR_ENABLE_QWEN_IMAGE, OTR_QWEN_IMAGE_GGUF/CLIP/VAE/
  STEPS/CFG/SHIFT/SAMPLER/SCHEDULER/WIDTH/HEIGHT/NEGATIVE.

TESTS (rip / update):
- `tests/test_image_engine_c3.py` (41 refs) -- the qwen_image dedicated suite;
  DELETE if qwen_image-only (verify at execution), else strip its qwen sections.
- `tests/test_image_dep_pilot.py` (4) -- remove qwen_image from the expected
  engine-name set + isolation-classification lists.
- `tests/test_capability_profiles.py` (1), `tests/test_tested_only_dropdown_gate.py`
  (1), `tests/test_workflow_apply.py` (1) -- drop qwen_image from expected lists.

WORKFLOW JSON: clean -- grep of workflows/*.json + config/profiles/*.json found
ZERO qwen_image references, so no saved widget value points at it. The image-model
combos are DYNAMIC from `all_engine_names()`, so removing the engine auto-drops it
from the dropdown -- NO JSON edit needed (confirm with a round-trip + widget audit).

## Execution order (when un-gated)
1. Rip the engine (delete qwen_image.py; remove __init__ import + registry row;
   reword hidream_i1 docstring). 
2. Rip/patch the tests (delete c3 if qwen-only; strip qwen from the 4 shared tests).
3. Add a PRESERVE guard test: assert z_image_turbo still resolves
   `clip_type == "qwen_image"` and flux2_klein still names `qwen_3_4b.safetensors`
   -- proves the rip didn't collateral-damage the shared Qwen3 TE / clip_type.
4. Validate: `qwen_image not in all_engine_names()`; z_image + flux2_klein still
   register + build their graphs; full suite + Bug Bible + B7 green; workflow JSON
   round-trip + link/widget audit clean; NOBOM; AST parse. Commit AND push per the
   green chunk to v2.0-alpha.

## Acceptance
- `qwen_image` gone from the registry, the dropdown, and the tests.
- z_image_turbo + flux2_klein UNAFFECTED (still render; clip_type/TE intact).
- Suite + Bug Bible + B7 green; no JSON change (or a validated one if the audit
  surfaces a stray saved value). Related queue item: the generic dispatcher
  engine-level assert_usable preflight (separate) still stands.
