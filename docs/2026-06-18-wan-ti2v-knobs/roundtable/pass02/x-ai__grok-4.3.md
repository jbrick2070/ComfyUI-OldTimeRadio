<!-- requested_model: x-ai/grok-4.3 | resolved_model: x-ai/grok-4.3-20260430 -->

VERDICT: no. Plan claims + code invariants contradict on every P1/P2 item; floor cannot ship.

MUST-FIX BEFORE BUILD:
1. [P1 + P2] VAEDecodeTiled path, frame lowering to 17, _TI2V_DEFAULT_FRAMES, and OTR_WAN_TI2V_SAMPLER whitelist are described as required but absent from eng_wan_ti2v.py. Fix: implement the four changes in _node_candidates, _loader_names, assert_usable, render_clip (and add the env var + validation logic) before any build.
2. [P2] Default CLIP remains "umt5_xxl_fp8_e4m3fn_scaled.safetensors" with CLIPLoader (lines 140-141, 109). Plan states this is Mac-broken and requires GGUF/fp16 + CLIPLoaderGGUF. Fix: change default + add GGUF branch in _loader_mode/_node_candidates/_loader_names exactly as done for unet.
3. [P2] Sampler defaults to "uni_pc" (line 118) and assert_usable has no whitelist. Fix: add the OTR_WAN_TI2V_SAMPLER assert (fail-closed) and change default to "euler".
4. [P1] _TI2V_MIN_FRAMES=33 (line 48), render_clip falls back to target_fps (line 164), VramPeakProbe uses interval_s=1.0 (line 170). Fix: set MIN=17, add DEFAULT_FRAMES, switch fallback, change probe to 0.1 s, and re-measure.
5. [P3] CFG guard, OOM pre-flight/retry, and license guard are listed as required but missing from assert_usable and render_clip. Fix: add the three guards.

SHOULD-FIX:
1. [P2] Plan asserts "ComfyUI-GGUF dequant runs on MPS/ROCm" but grounding only shows torch path; verify: actual MPS/ROCm execution of UnetLoaderGGUF + CLIPLoaderGGUF before claiming cross-platform.
2. [P1] "engine already frees the encoder before decode" is stated but free_after_use=True (line 169) is the only evidence; add explicit comment tying it to the umt5 encoder node.
3. [A/B] A/B test still references old defaults (33f, uni_pc, shift 5); update to match the hardened E config.

OPTIONAL / NICE-TO-HAVE:
- Add explicit env var for VAEDecodeTiled toggle inside _node_candidates rather than pure env-gate comment.
- Document the exact 4n+1 math used by quantize_frames_4n1 for the new 17-frame floor.

CUT THESE (over-engineering):
- None. Every item listed is required to close the 8 GB + Mac gap; removing any re-opens the fail-closed case.

[ASSUMPTION] Plan's "VramPeakProbe on a memory-constrained config" assumes the probe implementation exists in motion_common and accepts 0.1 s; verify against source.
[ASSUMPTION] "OTR_WAN_TI2V_SAMPLER whitelist" assumes the env var and assert_usable hook will be added; not present in grounding.