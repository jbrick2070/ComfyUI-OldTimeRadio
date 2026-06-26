<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The unified engine's family mapping fundamentally breaks character-beat routing, overwriting portraits with scene stills and feeding ambient audio to lip-sync beats.

MUST-FIX BEFORE BUILD:
1. [Part B] **Portrait Clobbering (Identity Crisis)**: `ltx_audio_in` handles both scene and character beats, but is assigned the `audio_conditioned_video` family. The proposed scene-still routing rule excludes only `engine_family(eng) != "audio_driven_face"`. Consequently, `ltx_audio_in` character beats will enter the scene-still block and overwrite the character's `init_image` portrait with the wide scene still. 
   *Fix*: The scene-still routing exclusion must check the shot role, not just the family. Exclude `shot.get("role") in ("character_video", "announcer_visual")` from receiving the scene still.
2. [Part C.3] **Ambient Audio Leak to Face Beats**: `_uses_ambient_master_audio` routes the ambient master mix to the `audio_conditioned_video` family when per-line timing is missing. Because `ltx_audio_in` uses this family for *all* beats, a character beat lacking timing will receive the ambient mix and lip-sync to the wrong audio (violating the exact protection the docstring warns about).
   *Fix*: Update `_uses_ambient_master_audio` to return `False` if the shot role is `character_video` or `announcer_visual`, regardless of family.
3. [Part B / F4] **Kill-switch Scope Bleed**: Applying the `OTR_ENABLE_LTX_I2V` kill-switch to the proposed unified capability branch will inadvertently disable scene stills for `wan_i2v`, `flux_still`, and `still_kenburns` if the flag is toggled off.
   *Fix*: Restrict the `OTR_ENABLE_LTX_I2V` check strictly to `eng == "ltx_video"`.

SHOULD-FIX:
1. [Part C.3] **Canvas Clamp Logic**: The plan says to replace the canvas clamp name-set with `requires_flag == OTR_ENABLE_LTX_AV`. The driver does not instantiate the engine class here, it only has the `engine_id` string. Reading `requires_flag` requires a registry lookup (`_vreg.get_engine(eng)`), which must be guarded against unregistered engines.
   *Fix*: Just update the explicit name tuple to `("ltx_audio_in",)` or safely fetch the engine instance before checking `requires_flag`.

OPTIONAL / NICE-TO-HAVE:
- [Part A / F3] Keep the name `ltx_audio_in`. Renaming to `ltx_av` creates unnecessary JSON and test churn for zero architectural benefit.
- [Part B / F2] Gating on `"init_image" in engine.required_inputs` is much safer than `accepts_still`, as `accepts_still` is a coverage-arch hint (whether to mint a still), not a render-driver requirement.

CUT THESE (scope / over-engineering):
1. [Part B / F1] **Capability-driven still routing unification**. [ASSUMPTION] This assumes all existing scene engines (`flux_still`, `flat_still`, `wan_i2v`, etc.) properly declare `accepts_still=True`. If they don't, they will silently lose their scene stills. It also conflates the `flux_still` portrait-clearing logic with standard I2V logic. 
   *Why it is safe to cut*: Take the F1 "narrow safe-mirror" path. Leave the patchwork alone and simply add `ltx_audio_in` to the existing `ltx_video` branch. It is vastly lower risk and guarantees no regressions for non-LTX engines.