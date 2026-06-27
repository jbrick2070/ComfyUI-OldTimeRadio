<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: no. The primary fix rests on an API ("reserved-inference VRAM global / minimum_memory_required hint") that is not described with a concrete function, class, or variable name; an implementor would be blocked at the keyboard.

MUST-FIX BEFORE BUILD:
1. [Proposed fix (primary)] The plan provides no specific ComfyUI call to limit GPU memory during unet load. It mentions `model_management` reserved‑vram global and `minimum_memory_required` hint but neither is a documented setter. Without an exact API (e.g., monkey‑patching `comfy.model_management.get_free_memory`, setting `comfy.model_management.vram_state`, or passing a budget to `load_models_gpu`) the instruction cannot be coded. Fix: research the real mechanism—likely temporary reduction of free memory reported by `model_management` or injection of a `minimum_memory_required` budget into the model options—and supply a concrete snippet that captures and restores the original value around `run_graph`.

2. [Proposed fix (primary)] The plan says “restore it in a finally block” but never shows how to capture the initial state (if any) or guarantees the state is per‑engine and isolate. Without a defined save/restore pair a failed render will leave the system in a throttled state for subsequent beats. Fix: implement a context manager or exactly describe the pair (`old = …` / `finally: … = old`).

3. [Proposed fix (primary)] The headroom magnitude “~2–3 GB” is hard‑coded in prose, not tunable. The implementor has no way to adjust without editing `eng_ltx_av.py`. Fix: expose an environment variable (e.g., `OTR_LTX_AV_HEADROOM_MB`) with a safe default, so the value can be changed per box without code changes.

4. [Secondary / complementary – B. CUDA allocator config] The plan lists `PYTORCH_CUDA_ALLOC_CONF` knobs but does not specify the exact value for Windows nor the point of injection (OTR prestartup, but that location is not described). Without a concrete config string and a named launch script entry, the implementation is incomplete. Fix: provide a recommended config line (e.g., `max_split_size_mb:256,garbage_collection_threshold:0.8`) and state exactly which file/script must set it before `torch` imports.

SHOULD-FIX:
1. [Proposed fix (primary)] The plan assumes that “forcing partial load” will trade steady ~10‑12 s/it for safety, but offers no evidence or calculation. A validate‑only step should estimate the overhead of layer‑streaming for a Q3_K_M GGUF on this GPU. If the overhead is unacceptable (e.g., >20 s/it) the fix may need a different headroom target or a fallback to Q4_K_S.

2. [Proposed fix (primary)] The fix must be gated by an env flag (`OTR_LTX_AV_RESERVE_HEADROOM=1`) so it can be disabled quickly if it degrades performance or breaks on a new ComfyUI version.

3. [Secondary – D] Setting “CUDA - Sysmem Fallback Policy” to prefer‑no‑fallback is an NVIDIA driver setting, not something the code can guarantee. The plan should note that it is a manual operator action, not an OTR code change.

4. [General] `run_graph` in `wrapper_bridge` receives a `keep` set of node IDs, not model patchers. The plan’s statement that the unet patcher is “kept” relies on the node ID `"unet"`. If the GGUF loader node returns a model patcher and the graph wires it into the guider, freeing the unet node’s output may still free the underlying tensor if the patcher is referenced elsewhere. Verify that the `keep` semantics correctly preserve the patcher’s weights in GPU memory when only the node output tuple key is retained.

OPTIONAL / NICE-TO-HAVE:
- Provide a script that measures actual activation peak after applying the headroom to tune `OTR_LTX_AV_HEADROOM_MB` automatically.

CUT THESE:
- The plan mentions a fallback Q2_K unet as “last resort”. Since the primary fix aims to avoid additional downloads and quality loss, the Q2_K discussion is premature. Cut it from this pass00; it can reappear if the primary fix fails.

[ASSUMPTION] The document assumes `comfy.model_management` exposes a way to artificially reduce the free memory seen by loaders; this is not confirmed against the torch 2.10.0+cu130 ComfyUI tree.