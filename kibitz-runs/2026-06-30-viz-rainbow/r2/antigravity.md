VERDICT: yes-with-fixes. The plan is highly solid and aligned with existing visualizer patterns, but misses updating ambient audio slicing conditions, engine-to-family mappings, and capability profile tests.

MUST-FIX BEFORE BUILD:
1. [WIRING] Ambient Audio Slicing Gap:
   - Defect: In [nodes/_otr_video_engines/render_driver.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/render_driver.py#L1055-L1059), `_uses_ambient_master_audio` determines which beats get ambient audio slices when per-line timing is missing (like `music_visual` beats). It hardcodes `str(engine_id) == "visualizer"`. Since the new engine ID is `"viz_mxc_cpu"`, it will return `False`. This leaves the music visualizer audio-starved (`audio_ref = None`), forcing it to render the silent/idle rainbow path and failing to pulse to audio.
   - Concrete Fix: Modify `_uses_ambient_master_audio` in [nodes/_otr_video_engines/render_driver.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/render_driver.py) to check `engine_id in ("visualizer", "viz_mxc_cpu")`.

2. [WIRING] Missing Fallback Map Entries:
   - Defect: The engine-to-family map `ENGINE_FAMILY` in [nodes/_otr_video_engines/render_driver.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/render_driver.py#L64) and `_FAMILY_FALLBACK` in [nodes/_otr_shared/content_oracle.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_shared/content_oracle.py#L42) do not list the new `"viz_mxc_cpu"` engine. These fallback maps are critical when the video registry is not importable or loaded (e.g., during some CPU/GPU soak testing harnesses).
   - Concrete Fix: Add `"viz_mxc_cpu": "abstract"` to `ENGINE_FAMILY` in [nodes/_otr_video_engines/render_driver.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/render_driver.py) and to `_FAMILY_FALLBACK` in [nodes/_otr_shared/content_oracle.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_shared/content_oracle.py).

3. [DECISIONS] Capabilities Schema and Registry Alignment:
   - Defect: The unit tests in [tests/test_capability_profiles.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/tests/test_capability_profiles.py#L211) strictly assert `set(vreg.CAPABILITIES) == set(vreg.all_engine_names())`. Registering `viz_mxc_cpu` without its capability row will fail the test suite immediately. The plan mentions adding a row to `registry.py` but doesn't define the schema layout.
   - Concrete Fix: Define the capabilities dictionary row in [nodes/_otr_video_engines/registry.py](file:///C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/registry.py#L241) as:
     `"viz_mxc_cpu": {"vram_class": "cpu", "vram_estimate_mb": 0, "required_toolchain": None, "requires_sidecar": False, "cpu_ok": True, "model_requirements": []}`

SHOULD-FIX:
1. [WIRING] Test Suite missing for new engine:
   - Defect: Adding the engine requires a new test file to prevent regressions. The build order doesn't explicitly mandate the creation of a dedicated test file mirroring `tests/test_video_visualizer.py` for testing registration, capability matching, and render contracts under different audio states.
   - Concrete Fix: Add `tests/test_video_viz_rainbow.py` to the build order (C-rb1) to test registration, `required_inputs=()`, and `render_clip` under audio-present and audio-absent paths.

OPTIONAL / NICE-TO-HAVE:
1. [DECISIONS] Single Audio-Optional Engine: The decision to stick to a single audio-optional engine via `required_inputs=()` is highly recommended over split-engines. It minimizes registry clutter. Implementing it only requires simple branching or always-moving background waves driven by time/seed that get modulated by active audio features.

CUT THESE (over-engineering):
1. [DECISIONS] Split-Engine Fallback: Cut any consideration of a separate `viz_mxc_gen` engine. Handling both paths in `viz_mxc_cpu` is trivial (since `eng_visualizer.py` already handles missing/empty audio refs gracefully) and avoids adding unnecessary dropdown noise to the UI.

[ASSUMPTION] We assume that the drawing algorithm details for the "OTR rainbow" (vacuum-tube/radio-dial/magic-eye tuning indicator) will be implemented purely using standard PIL `ImageDraw` primitives and numpy array operations, avoiding any third-party UI/rendering libraries.
