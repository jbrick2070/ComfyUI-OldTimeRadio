VERDICT: build-ready as-is? no.
Why: The proposed implementation for must-have 1 only updates render_driver.py to look for the "radio" subject but fails to update the image-generation dispatch phase (otr_meta_brief_image_prompt.py), which will cause a LOUD missing-fodder failure during rendering. Additionally, must-have 2 (headroom) cannot be cleanly adjusted in the compositor and must be resolved at the source in the Blender camera settings.

MUST-FIX BEFORE BUILD:
1. [must-have 1] Pipeline Mismatch on Radio Fodder Generation:
   Defect: Changing [render_driver.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/render_driver.py) to map the music-open beat (char_id "") to _mesh_subject_id = "radio" without modifying the image dispatch pipeline will result in the image-generation phase ([otr_meta_brief_image_prompt.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/otr_meta_brief_image_prompt.py)) still minting the fodder under "obj_<beat>" and prompting for a generic story object. The render driver will fail to find the "radio" fodder still in the ledger and raise a LOUD missing-fodder warning.
   Concrete Fix: 
   - In [otr_meta_brief_image_prompt.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/otr_meta_brief_image_prompt.py#L887), if the target role is "music_visual", set _subj_id = "radio".
   - In [otr_meta_brief_image_prompt.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/otr_meta_brief_image_prompt.py#L569), add an explicit branch in _mesh_fodder_subject: if str(role) == "music_visual": return "a vintage 1930s tabletop vacuum tube radio, wood cabinet, glowing dials".
   - In [render_driver.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/render_driver.py#L867), check if _role_of_shot(shot) == "music_visual", and if so, set _mesh_subject_id = "radio" and resolve _fodder from _fidx.get("radio").

2. [must-have 2] Headroom Lever Incompatibility in Compositor:
   Defect: Adjusting headroom in the compositor is not a viable lever. The compositor [otr_silent_composite.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/otr_silent_composite.py#L624) overlays the foreground frames (which Blender already renders at full canvas size) using a centered overlay with no rescaling. Adjusting fit-mode in the compositor would require adding an extra resolution-degrading scale filter in ffmpeg, which wastes GPU/CPU cycles and creates antialiasing artifacts on alpha edges.
   Concrete Fix: Standardize on the Blender stage camera distance/radius as the single lever. Increase the default --radius value in [otr_mesh_stage_blender.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/scripts/otr_mesh_stage_blender.py#L57) from 2.5 to 3.5 (or make it configurable via an environment variable in eng_mesh_stage.py). Update the parser test assertion in [test_video_mesh_stage.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_video_mesh_stage.py#L364) to expect the new default radius.

SHOULD-FIX:
1. [optional kibitz] Variable Turntable Angular Velocity:
   Defect: In [otr_mesh_stage_blender.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/scripts/otr_mesh_stage_blender.py#L384), the turntable keys sweep a fixed arc (arc_degrees, defaulting to 45 degrees) over the entire shot duration (frames). For a 25-frame shot, this rotates at 1.8°/frame; for a 250-frame shot, it rotates at 0.18°/frame. This causes visible turntable speed inconsistency across scenes.
   Concrete Fix: Calculate arc_degrees dynamically based on a fixed angular velocity (e.g. 1.0 degrees per frame or 25 degrees per second), up to the maximum safe limit MAX_ARC_DEGREES = 45.0 to avoid revealing the unpainted back of the projection.

OPTIONAL / NICE-TO-HAVE:
1. Ground Shadow for 3D Integration:
   Defect: The 3D model rendered by Blender has no ground or contact shadow projected onto the background plate, causing the mesh to appear floating in space.
   Concrete Fix: Add a shadow catcher plane beneath the normalized mesh at Z = -0.5 in [otr_mesh_stage_blender.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/scripts/otr_mesh_stage_blender.py) and configure a sun light source to cast a soft contact shadow onto the plate.

CUT THESE (scope / over-engineering):
1. [must-have 1] Canned Radio Mesh Asset:
   Why it is safe to cut: Shipping a static canned GLB radio asset in the repository increases package bloat and introduces asset-loading logic complexity. Generative image-to-3D via the vintage radio prompt for the mesh_fodder is already fully supported by the pipeline and has zero extra asset footprint.

[ASSUMPTION] We infer that Hunyuan3D-2mv is capable of successfully meshing a rectangular 3D vintage radio from a single generated 2D image without creating a warped or rounded "plaster blob" shape. In practice, single-view meshers can struggle with hard-surface rectangular geometry.
[ASSUMPTION] We infer that systems with low VRAM will successfully manage the transition from the torch mesher to headless Blender without CUDA context collision, based on the effectiveness of the _vram_barrier() reclaim mechanism.
