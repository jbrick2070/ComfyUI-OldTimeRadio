<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The proposed `FLAT` baseline destroys the smooth 3D form, reducing the mesh to a flat 2D silhouette.

MUST-FIX BEFORE BUILD:
1. [My proposed direction] `light=FLAT` completely ignores surface normals for shading. While it shows pure vertex colors, it will violate your hard constraint to "Keep the smooth form" because there will be zero diffuse shading to define the 3D volume (only cavity lines will remain). FIX: Abandon `FLAT`. Use `STUDIO` but disable the specular highlights that caused the v1.2 blowout.
2. [Q2] A Matcap maps colors based on the *view normal* (camera-relative), not world/object Z. If you use a vertical gradient Matcap, the gradient will "swim" across the surface as the turntable rotates (e.g., faces pointing up relative to the camera always get the top color, regardless of their actual height on the model). FIX: Reject the custom Matcap idea for a Z-axis gradient.

SHOULD-FIX:
1. [Q3] Yes, WORKBENCH STUDIO specular can be disabled to fix the v1.2 blowout while keeping smooth 3D diffuse shading. FIX: In `_configure_render` under the `WORKBENCH` block, add `shading.show_specular_highlight = False` (wrap in a `try/except` block exactly like you did for `show_cavity` on line 420 to handle Blender API variance).

OPTIONAL / NICE-TO-HAVE:
- [Q5] To lift it from a "flat ramp" to "pleasing", `STUDIO` (with `show_specular_highlight = False`) + `show_cavity = True` + your existing `gradient_color` vertex ramp is exactly the right combination. It provides soft, directional diffuse shading over a clean duotone without any busy textures.