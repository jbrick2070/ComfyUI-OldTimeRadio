<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The primary fix relies on invented/undocumented ComfyUI globals to hack around a graph topology bug that is actively pinning ~1.4 GB of VRAM during the denoise loop.

MUST-FIX BEFORE BUILD:

1. [Diagnosis / Proposed fix] Graph Topology VRAM Leak (Answers Open Question 4)
   - **Defect:** The plan misses the root cause of the missing ~2 GB headroom. In `eng_ltx_av.py::_build_graph`, `"videovae"` is wired to BOTH `"i2v"` (pre-sampler) and `"decode"` (post-sampler). Because `wrapper_bridge.run_graph`'s `free_after_use` only deletes a result when its *last* consumer runs, the 1384 MB `VideoVAE` is kept in the `results` dict and pinned in VRAM during the entire `sampler` execution.
   - **Fix:** Split the VideoVAE into two declarative nodes in `_build_graph`: `"videovae_enc"` (wired to `"i2v"`) and `"videovae_dec"` (wired to `"decode"`). This allows `free_after_use` to delete `"videovae_enc"` immediately after `"i2v"`, natively reclaiming 1.38 GB of VRAM before the `sampler` activations spike.

2. [Proposed fix (primary)] API Invention for VRAM Reserve
   - **Defect:** "raise ComfyUI's reserved-inference VRAM (e.g. model_management reserved-vram global / a minimum_memory_required hint)" is pseudo-code. There is no public `reserved-vram` global in ComfyUI. Mutating undocumented internals mid-flight is brittle, and `ComfyUI-GGUF`'s custom `UnetLoaderGGUF` may completely ignore it if the model is already loading.
   - **Fix:** Implement Fix #1 first. If a forced reserve is still strictly required, you must explicitly use `comfy.model_management.minimum_vram_retain` (saving and restoring its original value in a `finally` block), and verify that `ComfyUI-GGUF` actually respects this variable during execution.

SHOULD-FIX:

1. [Secondary / complementary] PyTorch Version Typo
   - **Defect:** The plan references "torch 2.10.0+cu130" and "torch 2.10 Windows". PyTorch 2.10 does not exist (latest is 2.6).
   - **Fix:** Correct the version assumption to PyTorch 2.1+ (which is when `expandable_segments:True` was introduced for Windows). Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the pre-startup environment.

CUT THESE (over-engineering):

1. [Fallback] Q2_K unet download
   - **Why it is safe to cut:** Downloading a 6.5 GB Q2_K model violates the "no new download if avoidable" goal and severely degrades quality on a 22B model. Fixing the `VideoVAE` residency (Fix #1) reclaims ~1.4 GB natively, making this lossy fallback unnecessary.

2. [Invariants to guard] Workflow JSON wiring
   - **Why it is safe to cut:** "The fix WIRED + ON in workflows/otr_scifi_16gb_full.json". [ASSUMPTION] This JSON file is not in the provided grounding and is out of scope for a Python engine code fix. The fix must be self-contained in `eng_ltx_av.py`.