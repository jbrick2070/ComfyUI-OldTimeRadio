# Round B -- Gemini (gemini-3-pro-preview) elapsed=27.4s

**1. Verdict**
PARTIALLY AGREE: Union Pro 2.0 and Option C are the correct architectural choices, but ChatGPT completely missed the VRAM threat posed by the control-image *preprocessors* (Depth/Canny extraction).

**2. Factual Errors in ChatGPT's Answer**
*   **Union Pro 2.0 Multi-Control:** ChatGPT states "No simultaneous multi-control by default" implying Union Pro can't do it without stacking. Union Pro 2.0 *natively* supports simultaneous multi-control (e.g., Depth + Canny) in a single pass by combining the control embeddings. You don't need two adapters to do multi-control with it.
*   **Mode Switch Mechanics:** ChatGPT calls the mode switch a "runtime branch". It is actually a specific learned embedding vector added to the conditioning based on an integer ID (e.g., Depth = 2). It costs zero extra time (< 5ms), not "sub-500 ms".

**3. Important Omissions**
*   **Preprocessor VRAM Spikes:** Extracting depth requires a model (like DepthAnything V2). If you load DepthAnything into VRAM *while* FLUX FP8 is sitting there, you will blow past 13.5 GB. 
*   **FP8/BF16 Casting Crashes:** In `diffusers`, passing inputs into a `bfloat16` ControlNet attached to a `float8_e4m3fn` base model frequently triggers `RuntimeError: expected scalar type BFloat16 but found Float8_e4m3fn`. 
*   **Caching the Control Image:** If you generate 3 variations, you shouldn't extract the Depth map 3 times. 

**4. My Recommendation**
*   **Use Union Pro 2.0 (bf16) + FLUX (FP8).** It adds ~3.5 GB VRAM. Stacking separate ControlNets is a waste of disk and VRAM.
*   **Strict Preprocessor Sequencing:** In `flux_keyframe.py`, if deriving from the Day 2 anchor, load the Depth estimator, extract the image, save it to disk (`shot_XXX/depth.png`), **`del` the depth model, and call `torch.cuda.empty_cache()`** *before* loading the FLUX pipeline.
*   **Adopt Option C (Both), but read from disk:** The LLM shotlist provides `control_image`. If missing, look for `shot_XXX/depth.png`. If missing, generate it from `render.png`. This keeps the generation loop completely decoupled from the extraction logic.
*   **Explicit Dtype Casting:** Force your control images and ControlNet conditioning scales to `torch.bfloat16` explicitly in the pipeline call to prevent diffusers from downcasting them to match the FP8 base model.
*   **Isolate Audio:** To preserve C7 (byte-identical audio), ensure `flux_keyframe.py` runs in a completely isolated `spawn` context and does not import `torchaudio` or touch the RNG states used by the audio generator.

**5. Uncertainties to Verify**
*   **Diffusers 0.37.0 Union Support:** Shakker-Labs Union Pro 2.0 requires passing a `control_mode` integer array. Verify that `diffusers 0.37.0` `FluxControlNetPipeline` natively accepts the `control_mode` kwarg without requiring a custom pipeline wrapper.
*   **Torch 2.10 / sm_120 / SageAttention:** You are on a bleeding-edge (beta) stack. SageAttention + SDPA on Blackwell might have undocumented kernel panics when routing `bfloat16` ControlNet residuals into `float8_e4m3fn` transformer blocks. If you get CUDA illegal memory accesses, disable SageAttention for the ControlNet blocks specifically.
