<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: build-ready as-is? **no**. The architecture has multiple gaps: the new family token must be registered in schemas.py; music_visual role requires either adding audio_ref to role_compat or dropping the role; and there is no node-availability check for required ComfyUI nodes, risking silent failures.

MUST-FIX BEFORE BUILD:
1. [schemas.py] Add `"audio_conditioned_video"` to the `FAMILIES` tuple and define its required inputs in `FAMILY_REQUIRED_INPUTS` (e.g., `("text_prompt", "audio_ref")`). Without this, any `VideoRequest` with `family_hint = "audio_conditioned_video"` will fail validation. Also update the assertion that `FAMILIES` and `FAMILY_REQUIRED_INPUTS` match.
2. [role_compat.py] Either (a) add `"audio_ref"` to `ROLE_AVAILABLE_INPUTS[Role.MUSIC_VISUAL.value]` so the engine’s `required_inputs` (`text_prompt`, `audio_ref`) are satisfied for that role, or (b) remove `"music_visual"` from the engine’s `roles` tuple to limit v1 to announcer/character. The plan requires music_visual but does not provide the role-compat change; without it, the engine will be excluded for music visuals.
3. [eng_ltx_av.py / architecture] Add a pre‑flight check in `assert_usable` (or `load`) that verifies all ComfyUI nodes the graph depends on (e.g., `LTXVReferenceAudio`, `A2VidPipelineTwoStage` wrappers) exist in the installed build. This must fail closed with a clear message if nodes are missing – the P0 probe alone is insufficient for a runtime guarantee.

SHOULD-CONSIDER:
1. Limit the engine’s roles to `("announcer_visual", "character_video")` to avoid the `role_compat` change entirely; music_visual can be deferred.
2. Drop the Yvann‑Nodes parallel lane for this sprint. It adds a new custom‑node dependency (b7 sweep, V‑12 review) and only addresses music_visual, which is already the riskiest role.
3. Add `"audio_conditioned_video"` to the `registry.py` docstring list of families for discoverability.
4. Define a clamp policy for beats longer than 20 s (see Q9); the document mentions it but does not describe how the engine/request will enforce it.

OPTIONAL / NICE‑TO‑HAVE:
- Document the exact required ComfyUI node names in the adapter docstring so that the pre‑flight check is explicit.
- Include a unit‑test that validates the fallback chain `ltx_av -> humo -> humo_1.7B -> latentsync -> still_kenburns` is acyclic and terminates at the floor.

CUT THESE (over‑engineering):
1. The Yvann‑Nodes lane as a parallel music‑visual path – it introduces a new external dependency, adds complexity, and does not address the core lip‑sync roles. Safe to cut until after the LTX‑AV lane is proven.

[ASSUMPTION] The driver (`render_driver.py`) populates `audio_ref` for music beats; if not, the plan’s music_visual support cannot work. Verify at build.