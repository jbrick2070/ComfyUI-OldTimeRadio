<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The plan is solid, but the GATE-A sweep is currently blind to silent fallbacks, and Claude's proposed fix for it will cause false failures on random seeds.

MUST-FIX BEFORE BUILD:
1. [Section 5 / Q1] The coverage sweep is blind to silent fallbacks because `expect_engine=""` bypasses the trace check. However, Claude's fix (`expect_engine=engine`) will falsely fail legs that randomly get 0 beats (e.g., `music_visual` on a random 30w seed).
   *Fix:* In `_otr_soak_capstone.py`, remove the `elif expect_engine:` branching entirely. Instead, unconditionally assert that the trace contains zero runtime fallbacks for ALL engines: `fell_off = [t for t in trace if t.get("final_engine") != (t.get("attempts") or [""])[0]]; if fell_off: raise SoakFail(...)`. This catches all silent fallbacks without false-failing on 0-beat slots.
2. [Section 4] The `wan_ti2v` (8GB tier) engine is planned but missing from the S1 capability declarations.
   *Fix:* Add a `"wan_ti2v"` row to `CAPABILITIES` in `registry.py` with `vram_class: "medium"`, `vram_estimate_mb: 8000`, and `model_requirements: ["wan2.2-ti2v-5b"]` so the S5 wizard and availability planner can see it.

SHOULD-FIX:
1. [Section 4 / Q4] Single-expert MoE (Path A) may produce near-static motion, failing the eyeball gate.
   *Fix:* Surface this explicitly in the plan's Section 4 risks. If the 14B low-noise expert fails the motion bar, Path B (two-expert high/low handoff) must be wired before promoting Wan.
2. [Section 4 / Q8] The item-4 matrix claims to run writer-LLM and voice-variation leg-sets, but `otr_coverage_