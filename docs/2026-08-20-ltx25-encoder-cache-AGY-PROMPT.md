# PASTE THIS INTO ANTIGRAVITY -- please pick **Gemini 3.1 Pro (High)**

The CLI lane is already running on Gemini 3.7 Flash (High), so 3.1 Pro adds a
genuinely different reviewer instead of a second copy of the same one.

Open the repo `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`
and paste everything below the line.

---

You are an independent reviewer with read access to this real repo. READ THE
ACTUAL FILES before asserting anything, and cite `file:line` for every claim.
Do not trust my summaries -- check them.

## The change being reviewed (NO CODE WRITTEN YET -- this is a pre-code review)

The LTX 2.5 video lane (`nodes/_otr_video_engines/eng_ltx25.py`) re-reads an
**8.86 GiB Gemma-4 12B Q5 GGUF text encoder from disk on every single shot**.
Counted on a live episode log: 15 shot renders, 13 encoder loads, ratio 1:1.
That is ~63 s of pure disk read per shot, on top of a 54.2 s CPU encode.

I want to add two caches:

* **(A) cache the loaded CLIP** so the encoder is read once, not once per shot;
* **(B) cache the empty NEGATIVE conditioning**, which is byte-identical for
  every shot of every episode.

Both would be injected through `run_graph`'s existing `external_results`
parameter (`nodes/_otr_video_engines/wrapper_bridge.py:322`), with the
corresponding node omitted from the graph. That mechanism already exists and is
already used by `eng_ltx_8gb.py:1278` and `eng_humo.py:698`.

## What I have already grounded MYSELF -- verify or refute these, do not repeat them

1. **A CPU-pinned CLIP does register in ComfyUI's model registry.**
   `comfy/model_management.py::load_models_gpu` ends with
   `current_loaded_models.insert(0, loaded_model)` unconditionally; a CPU
   `load_device` only sets `vram_set_state = VRAMState.DISABLED`. So the
   cached CLIP WILL be walked by our `reclaim_idle_models`.
2. **The engine is a process-lifetime singleton, not per-episode.**
   `nodes/_otr_shared/engine_registry_base.py:149` stores one instance at
   registration; `get_engine` (`:152`) returns that same object forever. So a
   cache on `self` pins 8.86 GiB of system RAM for the life of the ComfyUI
   server -- across episodes AND across lane switches. The box has 63.4 GB.
3. **`LTXVConditioning` does not mutate its inputs.** `comfy_extras/nodes_lt.py:561`
   calls `node_helpers.conditioning_set_values`, which at `node_helpers.py:9-23`
   builds a new list with `n = [t[0], t[1].copy()]` -- BUT the tensor `t[0]` is
   shared by reference, not copied.

## THE QUESTIONS I ACTUALLY NEED ANSWERED

**Q1 (the big one). Does our own post-render reclaim destroy the cached CLIP?**
`render_clip` calls `wrapper_bridge.reclaim_idle_models()` in a `finally`.
That function (`wrapper_bridge.py:266`) walks `current_loaded_models` and calls
`patcher.detach(unpatch_all=True)` on EVERY entry, indiscriminately.
Chase what that actually does to a GGUF CLIP:
`ModelPatcher.detach` (`comfy/model_patcher.py:1295`) ->
`unpatch_model` (`:1130`) -> and the GGUF override
(`custom_nodes/ComfyUI-GGUF/nodes.py:69`) which sets `p.patches = []`.
**Is the cached CLIP still usable on the next beat, or am I caching a corpse?**
My belief is that it survives -- no LoRA means `self.backup` is empty, the
weights stay in RAM on the GGMLTensor, `mmap_released` is already True, and a
re-`load()` re-patches from RAM without touching disk. I am NOT confident.
If I am wrong, the symptom would be garbage conditioning, not a crash.

**Q2. Should the cache instead be EXCLUDED from that reclaim walk?**
Detaching a `load_device=cpu` model reclaims zero VRAM, so skipping it looks
free. But `reclaim_idle_models` is shared by other engines. Is adding an
exclusion parameter to a shared helper the right call, or worse than the
detach-and-reload cycle?

**Q3. Given finding 2 above -- process-lifetime singleton -- what should
release the 8.86 GiB, and when?** Options I see: (a) accept it, 63 GB box;
(b) drop the cache in `teardown()`, which costs a reload per beat and mostly
defeats the purpose; (c) evict when a different video engine renders.
Which, and why?

**Q4. Is caching the negative conditioning actually safe on THIS graph?**
The tensor is shared by reference (finding 3). Does anything downstream --
`LTXVDualCFGGuider`, `SamplerCustomAdvanced`, `LTXVModalityGuidance`,
`LTXVConcatAVLatent` -- write into a conditioning tensor IN PLACE? If yes,
beat 2 onward silently renders with poisoned conditioning.

**Q5. What is the cheapest LOUD check** that would catch a stale-or-wrong
cache, given the failure mode is a wrong render that still looks plausible?

## DO NOT RE-PROPOSE THESE -- all three are already disproven with evidence

1. **`free_after_use` tuning.** The lab's own peak decomposition puts the text
   encoder and both VAEs at 0.0 GiB at the peak; freeing a non-resident
   encoder buys nothing.
2. **Load-order / phase splits for VRAM reasons.** `_topo_order` is Kahn with
   ties on sorted node id, so a split loader still lands in the first batch.
3. **Deleting the negative encode.** The locked sampler `euler_ancestral_cfg_pp`
   forces `disable_cfg1_optimization=True` and consumes `uncond_denoised`, so
   the "empty" negative is LIVE and steers every step. Deleting it would
   silently change every render.

## Hard constraints

* The render recipe is frozen -- no graph parameter may change. This is a
  residency/sequencing change only.
* Root-cause fixes only, no shims.
* Suite baseline that must hold: 11146 passed / 114 skipped / 1 xfailed.

Rank your findings by what would actually break, and name any file I have to
touch that I have not listed.
