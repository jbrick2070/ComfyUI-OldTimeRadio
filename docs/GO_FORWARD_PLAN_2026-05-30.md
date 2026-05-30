# OTR v2.0-alpha -- Go-Forward Plan (2026-05-30)

**Branch:** `v2.0-alpha` | **HEAD at authoring:** `f4d05b9`
**Stance:** stabilize, document, lightly harden. **Do NOT redesign.** The pack is past the big-cleanup stage.
**Supersedes nothing** -- this is the working backlog that merges the external go-forward consult (9 items) with `docs/ARCH_AUDIT_2026-05-30.md` and the git-history LoRA finding below.

---

## 0. Status gate -- why this plan is unblocked now

- **BUG-291 (HuMo/FLUX VRAM thrash) -- FIXED via NORMAL_VRAM.** Removing `--highvram` from `_otr_launch.bat` lets ComfyUI's DynamicVRAM stream the big models instead of pinning them. Live, this session:
  - FLUX bookend + portraits: **1.1-1.3 s/it** (was 107 / 6.9 s/it pinned)
  - HuMo Phase C: **14-18 s/it** (was 193 s/it pinned) -- ~12x recovery
  - `PHASE-C-VRAM-PROBE`: free **14817 MB** before HuMo loads; Lever-1 reclaim reserved 1760 -> 160 MB; models staged not pinned.
- **API node-load clean:** 35 nodes, no import errors. `OTR_WorkflowValidator` reports `widget_vector_drift=0` and now hard-raises on drift.
- **BUG-292 (output split) -- FIXED.** All output lands under one `Documents\ComfyUI\output\otr` tree (`otr/episodes/<id>/...` + final `otr/obs/`).

**Tagging trigger (Jeffrey directive 2026-05-30):** when the in-flight baseline smoke completes end-to-end to `otr/obs`, **tag the current HEAD `v2.0-alpha-stable`**. Then -- and only then -- run the optimized-LoRA test in section 4, *if the code warrants it* (it does for consolidation; see the finding).

---

## 1. Architecture verdict: keep the cascade intact

The intended shape is a full cascade and stays that way:

`audio -> procgen video -> FLUX stills/portraits -> HuMo -> LTX -> composite/blend -> upscale -> final procgen blend -> otr/obs`

This is **not** a broken architecture needing redesign. The main risk now is **over-cleaning**.

### Do NOT remove (intentional keep-surfaces)

| Surface | Why it stays |
|---|---|
| `_otr_model_runtime`, `_voice_backends/` | model-agnostic / provider plumbing in a model-agnostic project; tested, wire-ready |
| Sidecar nodes (`Visual*` subprocess), `VRAMGuardian`, `VRAMContextTest`, `ProjectStateLoader` | spawned-not-wired / topology / state by design |
| Branch gates (`OTR_FluxBranchGate`, `OTR_LtxBranchGate`) | runtime branch control |
| VRAM levers (`free_otr_pipeline_residue`, `_flush_vram_keep_llm`) | the BUG-291 fix depends on these |
| Mirrored helpers (`_resolve_radio_still_path`, `_load_ledger`) | intentional mirrors, pinned by constraint (BUG-LOCAL-121/076) |
| `_legacy_audio/_legacy_stills/_legacy_portraits` fallbacks | defensive defaults for auto-pick callers |

Do **not** hard-lock to one LLM or one model family.

---

## 2. Serialization fault line (read before touching any node surface)

`widgets_values` is **positional**. The live risks, all now gated by `OTR_WorkflowValidator` (node 63):

- Adding/removing/reordering a widget shifts every later saved value silently (BUG-281 was exactly this on node 14).
- `forceInput: True` widget-inputs occupy **no** slot (socket-only).
- INT `seed`/`noise_seed` add a hidden `control_after_generate` companion slot.
- COMBO choices can be a **list OR tuple** -- both count as one widget (the BUG-293 false-positive gap, now fixed).

**Keep `OTR_WorkflowValidator` wired permanently.** It is the only automated guard against the position-drift class and must gate every JSON change.

---

## 3. Remaining work -- small, isolated, test-gated

| # | Item | Type | Action | Gate |
|---|---|---|---|---|
| 1 | Retire duplicate AppData install | operator | Disable whichever copy is NOT the headless-runtime copy; relaunch | "35 nodes" prints once; output under one tree |
| 2 | Keep `OTR_WorkflowValidator` wired | policy | Never unwire; gates every JSON edit | validator POST drift=0 |
| 3 | Clarify HuMo->LTX edge | rename/doc | Treat `51.clips_dir -> 55.humo_clips_dir` as a completion/wait edge unless code proves LTX consumes the clips as source media; rename `wait_for_humo_clips_dir` / `humo_done_gate` or document | validator + smoke if renamed |
| 4 | LoRA structure (nodes 60/61) | JSON + smoke | See section 4 -- consolidate (proven) vs mix-different (gated) | validator + fixed-seed A/B smoke |
| 5 | Second *different* LoRA | creative experiment | Only for a specific visual weakness seen in the baseline (motion / CRT-period / noir / character / first-frame). Do not add to fill an empty slot | A/B vs consolidated baseline |
| 6 | `closing_audio` double-mix check | audit | Confirm node 12 uses `closing_audio` for timing/visual only; it must NOT mix into final audio if `episode_audio` already contains it (Prime Directive 1) | audio byte-identical regression |
| 7 | `LowVRAMCheckpointLoader` unused CLIP | audit | LTX CLIP comes from the deferred Gemma encoder (node 57); confirm the checkpoint loader is not also loading CLIP internally (hidden VRAM/time) | inspect; smoke if changed |
| 8 | `scene_sequencer.py:152 DEFAULT_OUT` | code + smoke | Route the hardcoded `~/Documents/.../otr/audio` fallback through `_otr_paths.otr_audio_dir` (same class as the fixed BUG-292) | full suite + smoke; output stays one-tree |
| 9 | Stale metadata/version labels | doc | Make workflow metadata match `v2.0-alpha` reality so future agents don't assume from stale labels | n/a |

Nothing here is urgent. Items 4 and 8 are the small code/JSON wins; item 1 removes the dual-install fragility that caused the output split.

---

## 4. LoRA decision -- the headline question ("one LoRA, or mix two different?")

### Git-history finding (settled)

Across **all branches and all history**, workflow nodes 60 and 61 have only **ever** referenced the same file:
`ltxv\ltx2\ltx-2.3-22b-distilled-lora-384-1.1.safetensors` (node 60 @ 0.5, node 61 @ 0.2).
The only other LoRA anywhere in the graph is the HuMo lane's `lightx2v_I2V_14B_480p_cfg_step_distill` (node 46) -- a separate lane.

=> Node 61 was **never** intended to be a different LoRA. This is a **true duplicate**, not a stub for a second adapter. Per the consult decision tree, this is the *consolidate* branch, not the *restore-the-intended-LoRA* branch.

### The math

`LoraLoaderModelOnly` is an additive ModelOnly weight patch. Chain is `54 -> 60(@0.5) -> 61(@0.2) -> 55`, same delta both times:

```
W_eff = W + 0.5*delta + 0.2*delta = W + 0.7*delta
```

One loader @ 0.7 is weight-equivalent (within fp accumulation order). This is also the LTX **distillation/acceleration** adapter -- its strength is a speed/quality budget, and 0.7 is a deliberate partial-distillation choice (do not assume 1.0).

### Two mutually-exclusive end-states

**PATH A -- consolidate (the "one LoRA" path). Code warrants it now.**
1. Edit node 60 to `0.7`; delete node 61; rewire `60 -> 55` (drop link to 61).
2. `OTR_WorkflowValidator` POST -> `widget_vector_drift=0`, no raise.
3. Fixed-seed A/B smoke: consolidated vs the baseline render. Expect visual parity (near bit-identical LTX frames).
4. Keep only if parity holds. One commit.

**PATH B -- mix a *different* LoRA (only if the baseline shows a targeted weakness).**
- Keep distillation @ **0.7 on node 60** (do NOT rob the distillation budget -- under-applying it under-converges few-step sampling).
- Repurpose node 61 to a genuinely different adapter at **0.15-0.3**, additive on top of full distillation:
  - LTX motion jitter / camera drift -> a motion/control LoRA
  - generic look -> a CRT / analog-broadcast / period style LoRA (matches the OTR aesthetic)
  - character drift across clips -> a low-strength character LoRA
  - weak first-frame/composition adherence -> a composition LoRA
- A/B vs the Path-A consolidated baseline. Re-verify step convergence + VRAM (a second adapter shifts both).

**Decision rule:** review the baseline `otr/obs` output first. If LTX motion/style/character look good -> **Path A** (consolidate, ship). If a specific weakness is visible -> **Path B**, choosing the adapter that targets *that* weakness only.

---

## 5. Execution rule (every item above)

```text
One change per commit.
Run validator after every workflow JSON edit.
Run smoke after every topology/path/model-load change.
Do not change widget surfaces unless absolutely necessary.
Do not remove provider/fallback plumbing.
Do not hard-lock to one LLM or one model family.
```

Plus the standing OTR gates: Bug Bible + core + audio regression after every code change; verify the workflow JSON is re-wired to the current node surfaces before calling a change done.
