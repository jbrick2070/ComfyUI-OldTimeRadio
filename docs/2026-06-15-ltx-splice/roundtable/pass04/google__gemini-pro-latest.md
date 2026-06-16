<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The core graph replacement is solid, but misses wiring updates for the text encoders and misunderstands the candidate inheritance chain.

MUST-FIX BEFORE BUILD:
1. [4C] Gaps in wiring: The plan drops `"encoder"` but forgets to rewire the text conditioning nodes. You must explicitly update `"pos"` and `"neg"` in `_build_graph` to use `"clip": W("te", 0)` instead of `W("encoder", 0)`.
2. [4D] Correctness on VRAM reclaim: The plan says `results.get(<proven key>,(None,))[0] accordingly`. Since the LoRA node wraps the UNET, the final patched model resides at `"lora"`, not `"unet"`. You must explicitly use `results.get("lora", (None,))[0]` to retrieve the model for the teardown patcher bucket.
3. [4B] Over-engineering / Correctness: The plan says to add the new candidates `"in BOTH the base and the i2v candidate dicts"`. Do not do this. `_node_candidates_i2v` inherits from `_node_candidates_sampling`, which inherits from `_node_candidates`. Adding them to the base dict automatically propagates them; adding them to both is redundant and risks desync.

SHOULD-FIX:
1. [4A] The plan updates `_distilled_lora_file()` to return a single string (`ltxv\ltx2\...`). Currently, this method returns a tuple `(name, path)`. Ensure the method continues to return the tuple, or update the unpacking in `_build_graph` (`lora_name, lora_path = self._distilled_lora_file()`) so it doesn't crash.
2. [4E] The plan says `assert_usable()` gains a "node-class gate". `assert_usable()` runs at validate time on the CPU box where ComfyUI nodes cannot be imported (Invariant V-12). Do not import `wrapper_bridge` or resolve classes in `assert_usable()`; keep the node-class gate in `load()` where it belongs.

OPTIONAL / NICE-TO-HAVE:
- [4F] `commercial_clean`: Since the GGUF recipe uses Apache + LTX-2 Community licenses, it is safe to set `commercial_clean = True`.

CUT THESE:
1. [4B] "add ... in BOTH the base and the i2v candidate dicts" — safe to cut per MUST-FIX #3. Inheritance handles this.