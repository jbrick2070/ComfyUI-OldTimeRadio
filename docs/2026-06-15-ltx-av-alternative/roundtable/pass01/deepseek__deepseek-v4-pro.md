<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: build-ready as-is? no. Critical design decisions unresolved; missing concrete wiring for audio staging, VRAM guard, offload strategy, and boomerang disable. The plan is a brainstorming draft, not a code-ready spec.

MUST-FIX BEFORE BUILD
1. [A.1] Engine structure not decided: The plan asks “1-engine-vs-2-adapter” but does not settle it. A code‑ready plan must define exactly one engine (e.g., `ltx_av`) with roles, family, required_inputs (including `audio_ref`), and fallback chain. Without this, the engine cannot be registered. Fix: choose a single `ltx_av` engine serving all three roles, or the two‑adapter split from the stale plan, with explicit `required_inputs` and fallback logic.
2. [B.3] Audio slice integration missing: The plan says “per‑beat audio ... fed as the A2V conditioning input” but never describes how `audio_ref` is extracted from the request, staged into ComfyUI, or connected to the LTX‑2.3 conditioning node. The existing `wrapper_bridge` only handles image staging; audio staging is a new gap that needs a concrete design. Fix: document the audio staging path (load audio file, convert to expected sample rate/tensor, feed into a ComfyUI audio‑load node or conditioning node), and ensure it reuses the existing per‑beat audio slice plumbing.
3. [C.1] VRAM pre‑flight guard missing: The plan says “NVML REQUIRED (fail‑closed) for the heaviest lane” but does not specify where or how. The engine’s `assert_usable` must probe available VRAM (via NVML) and fail closed if the chosen model variant cannot fit, *before* any heavy import. Without this, a 16 GB‑capped box will OOM mid‑render. Fix: add a VRAM check in `assert_usable` that estimates peak memory for the selected variant and raises `EngineUnusable` if insufficient.
4. [C.1/C.4] Gemma‑3‑12B offload strategy not defined: The text encoder is too large to coexist with the DiT on 16 GB. The plan mentions “CPU/offload” but provides no mechanism. The current `free_after_use` frees intermediates but does not offload model weights to CPU. Fix: design explicit offload (e.g., using ComfyUI’s `model_management` or a custom load‑unload cycle) and test that the encoder can be freed before the DiT loads; otherwise VRAM will be exceeded.
5. [B.4] Boomerang must be hard‑disabled: The plan hypothesises “boomerang OFF” but does not enforce it. The existing `ltx_video` engine honours the env `OTR_LTX_LOOP_VIA_REVERSE` and could loop even for an AV clip, breaking audio‑video sync. Fix: in the AV engine, override `_loop_via_reverse` to return `False` unconditionally, and never apply the mirror in `render_clip`, regardless of env.
6. [General] Fallback chain verification: The stale plan’s fallback (`humo→humo_1.7B→latentsync→still_kenburns`) may be obsolete after the HEAD move to `9633e1e`. Without verifying the current humo/latentsync engines, the fallback for `audio_driven_face` could be broken. Fix: ground the fallback against the present codebase and adjust `fallback_engine` accordingly before coding the AV engine.

SHOULD-FIX
1. [A.2] Audio discard mechanism: The plan asks “decode video‑only … or generate‑then‑drop?” The answer is clear—use the existing silent‑encoding path and never wire any LTX audio‑output node. Document this explicitly in the engine spec and ensure `test_audio_byte_identical` covers the full composition.
2. [C.2] GGUF loader gating: If the plan ever reaches the GGUF fallback, the engine must detect `UnetLoaderGGUF` availability and fail closed with a clear message, not silently fall back. That gating should be part of `assert_usable`.
3. [B.2] JSON defaults: Confirm that the new engine is not included in default profiles (dark) and that the dropdown‑only selection does not accidentally activate it in CI. Already stated but verify during wiring.
4. [C.4] Offload‑thrash heartbeat: The plan’s “~60s/heartbeat watchdog” needs a concrete interface to kill stalled renders; define how it interacts with the ComfyUI process.

OPTIONAL / NICE-TO-HAVE
- The Lane‑A audio‑reactive prompt injection (ledger tempo/energy→motion verbs) as a low‑risk motion win before Lane B.
- Concrete graduation criteria for Lane B (optical‑flow / framediff vs b005 gold, no OOM over N clips) phrased as a CI‑style gate.

CUT THESE (over‑engineering)
1. Two‑stage upscale (Stage2) for initial Lane B: The plan already says “Spatial upscaler only AFTER the base pass is stable”—defer entirely. Implementing it now adds VRAM pressure and complexity that could mask base‑stage problems. Safe to cut until Stage1 proves viable.
2. Simultaneous support for multiple model variants (distilled v1.1, FP8, GGUF): Start with a single variant (preferably distilled v1.1) and a fixed checkpoint path. The “probe‑best‑variant” logic can be added later. Attempting all three at once is speculative and risks multiple VRAM‑related failures.

Marked [ASSUMPTION] where the plan relies on the stale SPRINT_PLAN.md’s fallback chain without current grounding; the judge must verify that chain against the live code. (ASSUMPTION: the humo→latentsync cascade still exists as described.)