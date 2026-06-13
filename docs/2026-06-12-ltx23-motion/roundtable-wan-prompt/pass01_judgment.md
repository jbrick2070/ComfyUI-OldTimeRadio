# Roundtable judgment -- Wan video coder spec, pass01 (GPT-5.5 frontier-only)

Panel: `openai/gpt-5.5-20260423` (1 model, operator-requested GPT-frontier-only).
Spend: ~$0.11 total (2000-tok truncated run + 6000-tok full run). Judge: Claude,
grounded against `eng_wan_i2v.py` + `wrapper_bridge.py` (run_graph / _topo_order).

## ACCEPTED (confirmed against code -> folded into the embedded spec)
- **MF1 aspect_plan is dead code.** `render_clip` computes `plan["aspect_plan"]`
  then stages the RAW init via `stage_into_comfy_input(plan["init_image"])` and
  `_build_graph` LoadImages it directly -- the pad/crop transform is never applied.
  My "supply init_w/init_h so the policy applies" claim was FALSE. FIX folded.
- **MF3 render-phase NVML is not actually captured.** `assert_vram_within_ceiling`
  fires AFTER `encode_frames_to_silent_mp4` (post-GPU, instantaneous) -> misses the
  sampler peak. Need NVML polling across the render window. FIX folded.
- **MF4 GGUF loader switch unspecified.** `_node_candidates` resolves `UNETLoader`
  only; `_build_graph` always emits `unet_name`/`weight_dtype`. GGUF needs a loader
  mode + `UnetLoaderGGUF` + its installed input names. FIX folded.
- **MF5 TI2V engine is entirely unspecified.** Only `WanI2VEngine` exists; "clone
  the pattern" is too thin. Need a real `wan_ti2v` contract. FIX folded.
- **MF6 Phase1/Phase2 assertions still mixed** (Task 2 said "same asserts as task 1
  (engine-in-trace)"). Rewrote tasks into explicit Phase 1 / Phase 2 substeps.
- **MF7 absolute ckpt path won't load in a /prompt graph.** Comfy loaders take a
  name relative to the registered `diffusion_models` folder, not an absolute path.
  Phase 1 must pass the basename + verify the headless yaml exposes the folder.
- **MF8 "in-process is safe" was wrong.** `resolve_isolation()` escalates on
  `sageattention_patched()` regardless of which graph nodes are used. For the fast
  path: disable Sage + assert `resolve_isolation()==in_process`. FIX folded.
- **SF1 stale KJ docstrings** in `eng_wan_i2v` ("install the Wan wrapper + KJNodes
  pin audit") contradict the core-nodes decision. Flag to update.
- **SF/CUT:** camera LoRA cut from smoke scope; MAD demoted to optional; Path B
  high-noise not fetched until Path A is eyeball-rejected; kill the Phase-1 server
  before the Phase-2 engine leg; wan-specific clip dir.
- **OPT:** per-clip sidecar JSON (seed/model/loader/vae/nodes/steps/cfg/git) +
  a preflight that prints resolved node signatures + model visibility. Folded as
  nice-to-haves.

## TEMPERED (real observation, contingency not a hard rebuild)
- **MF2 topo order loads UNETLoader before clip is freed.** CONFIRMED: `_topo_order`
  is wave-by-wave Kahn (alpha ties), so `unet` runs in wave 0, `pos`/`neg` (and the
  clip free) in wave 1. BUT GPU co-residency is gated by Comfy's lazy
  `load_models_gpu` at the sampler, so this may be harmless -- the render-phase NVML
  (MF3) is what proves it. Folded as: measure first; IF the peak busts, split text
  encoding into a pre-pass (encode -> free umt5 -> then the sampler graph).

## REJECTED / NONE
- No hallucinated claims this pass; the panel stayed grounded.

## CONVERGENCE
- One pass, single frontier model, operator-scoped. The accepted items are folded
  into the embedded spec (GO_FORWARD_PLAN section 1A). Not looping further (operator
  scope = GPT-frontier-only, and the catches were all code-confirmable).
