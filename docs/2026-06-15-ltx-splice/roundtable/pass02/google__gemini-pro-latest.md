<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The splice correctly targets the model/loader scope and preserves the sampler/infra logic, but misses a required projection checkpoint and will crash during VRAM reclaim if the base UNET is not explicitly kept.

MUST-FIX BEFORE BUILD:
1. [§4A / §4B] Missing text projection checkpoint. `LTXAVTextEncoderLoader` requires the base `ltx-2.3-22b-dev.safetensors` file for its `ckpt_name` input to load the text projection weights (as seen in `eng_ltx_av.py`). 
   - Fix: Add the projection `.safetensors` file to the required models list in `_installed()`, `assert_usable()`, and `CAPABILITIES["ltx_video"]`. Pass it to the `te` node in `_build_graph`.
2. [§4A] VRAM reclaim crash (`free_after_use` bug). `eng_ltx_video.py` calls `run_graph(..., free_after_use=True)`. If you update the `keep` set to only `"lora"` and `self._TERMINAL`, the bridge will evict the