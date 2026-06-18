<!-- model: gpt-4.1 (direct API) -->

- VERDICT: **no** — The spec leaves correctness traps and critical gaps unresolved, especially around 3D mesh/still distinction, silent fallback, and ltx_video behavior.

---

## MUST-FIX BEFORE BUILD

1. **[3 / Candidate A, B, Decision 1, 2] — 3D mesh vs. still granularity is not fully specified or wired.**
   - **Defect:** The spec admits "the 3D mesh-portrait-vs-init kind distinction + the per_beat mesh-rebuild lock still need SOMETHING to express granularity (can't be pure convention)". But neither A nor B gives a concrete, checkable, central wiring for this. The 3D lock is enforced in several places (see grounded facts), but how new engines or episode configs declare mesh vs. still, and how the dispatcher/director use this, is not nailed down.
   - **Fix:** Specify a single, explicit field (e.g., `still_kind in {init_image, mesh_portrait}`) on all video/3D engines, and update both dispatcher and directors to use this for both still dispatch and mesh lock enforcement. Document exactly how and where this is checked.

2. **[Candidate A, Decision 2] — ltx_video “optional image” bug not closed.**
   - **Defect:** The spec notes ltx_video can consume an init image but does not declare it in required_inputs, so the dispatcher skips generating a still for it. The fix is implied but not explicit: "how ltx_video starts consuming the still" is left vague.
   - **Fix:** Explicitly require ltx_video to declare its optional still consumption (either by adding "init_image" to required_inputs when the env OTR_ENABLE_LTX_I2V is on, or by adding a new `accepts_still` property checked by the dispatcher). Document the gating logic in both registry and dispatcher.

3. **[Candidate A, Decision 2, 3] — No explicit handling of silent fallback/skip/degrade.**
   - **Defect:** The current dispatcher code fails safe (generates a still) on any unknown or error, but the spec does not require all skips/degrades to be LOUD (i.e., raise/log visibly). This violates the "no silent fallback" invariant.
   - **Fix:** Require all fallback paths in dispatcher/gatekeeper to log or raise a clear, user-facing error/warning, not just return True silently.

4. **[Candidate B, Decision 2] — Fixed-path overwrite risk not mitigated.**
   - **Defect:** Candidate B's "always write to <episode>/stills/<role>.png" risks overwriting stills for same-dim beats or in concurrent runs (see S7 in spec), but no mitigation is specified.
   - **Fix:** Specify a unique, collision-resistant naming convention for still files (e.g., including timestamp, UUID, or beat index), and require all consumers to match.

---

## SHOULD-FIX

1. **[Candidate A] — Overlap/duplication in capability flag and required_inputs.**
   - **Defect:** Candidate A proposes both a new `accepts_still` flag and fallback to "init_image" in required_inputs, risking drift between them.
   - **Fix:** Mandate a single source of truth: either only use `accepts_still`, or require all engines to declare "init_image" in required_inputs if they consume it, and drop the flag.

2. **[Candidate B] — Unclear opt-out for pure floors.**
   - **Defect:** The "tiny consumes_still=False opt-out" for floors is not specified—where does it live, how is it checked?
   - **Fix:** Specify the exact attribute or config key, and the code path in dispatcher/director that checks it.

3. **[Decision 3] — Role_compat interaction not fully spelled out.**
   - **Defect:** The spec says role_compat stays the filter, but does not specify how new engines with new still/mesh needs interact with role_compat. Risk of misfit or silent skip.
   - **Fix:** Require that all engines with new still/mesh needs update ROLE_AVAILABLE_INPUTS and role_compat.engine_fits_role logic, and document this in the build order.

---

## OPTIONAL / NICE-TO-HAVE

- Add tests for all edge cases: pure-procedural, ltx_video with and without OTR_ENABLE_LTX_I2V, 3D mesh talkers.
- Document the “one place” for image engine selection in code comments and operator docs.

---

## CUT THESE (over-engineering)

1. **[Candidate A] — `still_input_name` and `still_kind` if only one is ever used per engine.**
   - **Why:** If all engines use "init_image" except the 3D mesh ones (which always use "mesh_portrait"), just a single `still_kind` is sufficient; `still_input_name` is redundant.

2. **[Candidate A] — Dual logic for capability flag and required_inputs.**
   - **Why:** If you enforce a single flag or required_inputs convention, the fallback/dual-read logic is unnecessary.

---

**[ASSUMPTION]:** No grounding for the exact structure of the engine registry or how new engines are registered; verify that adding flags/fields is viable.

**[ASSUMPTION]:** No code grounding for ltx_video's env flag handling or mesh lock naming conventions; verify details before implementation.