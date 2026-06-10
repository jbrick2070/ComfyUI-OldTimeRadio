# Pass01 judgment log (judge: claude-fable-5; panel: gpt-5.5, gemini-3.1-pro, grok-4.3, deepseek-v4-pro + claude-fable as independent panelist)
Spend: $0.2237 (manifest.json). All 4 panel calls ok. Claude's panelist review written BEFORE reading panel output.

## ACCEPTED (grounded against the repo)
1. **Node 63 validator self-check defect** (GPT#1). VERIFIED: workflow JSON node 63 `widgets_values` starts `""` -> validator falls back to the canonical 16gb path; snapshots would validate the wrong file. Folded: generator sets artifact self-path + CI validates every .gen.json.
2. **Non-Director engine widgets exist** (GPT#5). VERIFIED in JSON: node 81 indextts2, 82 kokoro, 83 stable_audio_3, 92 humo. Folded: profile-managed widget map covers them; "Director widgets are canonical" framing corrected to "profile-applied widgets".
3. **One applier consumed by generator + headless + UI; delete hard-coded patch lists; whitelist creative inputs** (consensus: GPT#2, Gemini#1, Grok#1/#3, DeepSeek#4, Claude#4). VERIFIED: queue_smoke.py hard-codes patches. This IS the drift kill. Folded with parity test + punch-list acceptance demo.
4. **VRAM ceiling hardcoded breaks 8gb tier** (Gemini#2, DeepSeek#3, Claude#6). VERIFIED: wrapper_bridge.py:37 constant. Folded: OTR_VRAM_CEILING_MB env > profile stamp > 14500; 8gb soak under simulated ceiling.
5. **No Mac-MPS tier in v1** (consensus: GPT#3, Gemini#4, Grok S2, Claude scope-call). VERIFIED: zero mps references. Folded: Mac ships as cpu_floor, labeled; MPS = parked sprint w/ device_routing.py.
6. **Profile schema must be committed first; reject unknown keys/impossible combos** (GPT#4, DeepSeek#1, Claude#3). Folded: capability-based schema, S0.
7. **Derived enable-set from registry metadata, not hand-listed flags** (Claude#3, DeepSeek#3, GPT#8 capability metadata). Folded: vram_class/required_toolchain/model_requirements per engine.
8. **Generator contract: patch-by-NAME with real schemas; validate every artifact; fail on out-of-schema COMBO** (Gemini#3, Grok#2, DeepSeek#2-half, GPT#6). Folded -- with schemas from direct NODE_CLASS_MAPPINGS import (no live server needed; tests already do this), live /object_info cross-check kept in soak lane.
9. **OTR_FORCE_ENGINE_MAP precedence documented** (Gemini S1). VERIFIED: render_driver.py:607-653, marathon uses it. Folded into precedence spec (top dev override).
10. **request_seed naming trap** (Gemini S2). Folded into applier widget map (never patch `seed`).
11. **Seed/determinism key = (profile_id, seed); ledger gets profile_id+hashes; profile carries seed policy** (GPT#10, Claude). Folded.
12. **Generated-manifest-from-registry instead of hand-written manifest** (DeepSeek CUT, reconciled with GPT#9): manifest format exists but is GENERATED from engine model_requirements. Folded into S5.
13. **Problem framing correction** (GPT S1): one production JSON + patch lists, not "multiple JSONs". Folded (final doc will restate the mechanism; pass00's framing was stale).
14. **Cuts**: Gradio (GPT CUT1), videotoolbox now (GPT S7), cartesian testing (GPT CUT3), custom HF_HOME parsing (Gemini CUT -- hf hub honors it natively), 3D-as-v1-switch (GPT CUT2; already platform-deferred). All folded as explicit CUT list.
15. **Startup profile-vs-reality assertion** (Claude#2; GPT#8 pre-launch rejection is the same class). Folded: stamp + hard stop + downshift suggestion.
16. **In-graph profile carrier reduced to a STAMP** (Claude#1 evolved by panel input): env doesn't serialize, but full in-graph profile DATA plumbing is unnecessary -- apply-time widget dialing + identity stamp + server-side stamp->profile resolution for runtime knobs (ceiling, assertion). This resolves snapshot self-sufficiency without new graph wiring.

## REJECTED (with grounds)
- **DeepSeek#2 alt: drop pre-generation, runtime-apply only.** Rejected: loses click-Run shipped files (operator requirement, time-to-first-value criterion). The live-server objection it was based on is mooted by direct NODE_CLASS_MAPPINGS schema import.
- **Grok CUT2: ban all runtime patching at submit.** Too broad: applier-driven patching IS the headless path; the defect was ad-hoc per-script engine patches (GPT S2 made the same distinction). Folded as whitelist instead.
- **Grok CUT1: cut the installer/manifest layer entirely.** Partial reject: deferred to S5, not cut -- onboarding time-to-first-value is a stated decision criterion; but slimmed (generated manifest, no custom HF parsing).
- **Gemini#4: refactor all ~35 cuda checks into device_routing.py NOW.** Out of v1 scope: no MPS tier ships in v1; the refactor lands with the parked MPS sprint. v1 only needs cold-import + cpu_floor smoke.
- **GPT#3 option "implement MPS now"**: same as above; the rename option (mac=cpu_floor) was accepted instead.

## UNVERIFIABLE -> verify-at-build (carried in plan)
- Node 92 engine-widget exact schema name (GPT flagged own assumption).
- Adapter cold-import without CUDA sidecars (GPT S4).
- Saved-COMBO load behavior in lite envs (GPT#7).

## Convergence call
Pass01 produced material must-fixes (node 63, widget map breadth, VRAM ceiling, applier design). NOT converged. Run pass02 on pass01_plan.md.
