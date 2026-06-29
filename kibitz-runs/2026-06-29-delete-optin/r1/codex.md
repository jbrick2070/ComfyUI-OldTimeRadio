VERDICT: no. The plan removes the flag mechanism before it defines a replacement boundary for hidden registered/dark engines and non-dropdown routing paths.

MUST-FIX BEFORE BUILD:
1. [End goal + Change 6] The curation story contradicts itself: the end goal says untested models are removed from the registry, but Change 6 keeps curation at `validated_engine_names()`. Real code keeps hidden engines registered in `nodes/_otr_video_engines/registry.py` / `nodes/_otr_image_engines/registry.py`, and `nodes/_otr_video_engines/render_driver.py` can still route registered engines via `OTR_FORCE_ENGINE_MAP`; `OTR_LSYNC_BASE_ENGINE` can also name a dynamic base engine. Concrete fix: make one authority explicit. Either unregister non-shipping engines before release, or add a runtime “validated-only unless explicit custom/dev mode” guard for force-map, lsync-base, and custom model paths.

2. [Scope + Change 2/3] The plan clears gates for engines that are not render-ready. `nodes/_otr_video_engines/eng_triposr.py` is explicitly a dark scaffold and its `prepare` / `render_clip` / `canonicalize` raise `NotImplementedError`. `nodes/_otr_video_engines/eng_character_3d.py` registers `triposg_talk`, `hunyuan3d_talk`, and `trellis_talk`, also dark scaffolds. Removing their flags violates the end goal that selecting a model “just renders” and that the only runtime gate is files-on-disk. Concrete fix: remove these dark engines from this deletion scope or unregister them until their forwards are implemented and validated.

3. [Current state + Change 2] The “two flag gates, both keyed on `requires_flag == os.getenv(flag)=="1"`” claim is false. Image adapters such as `nodes/_otr_image_engines/flux2_klein.py` and `nodes/_otr_image_engines/z_image_turbo.py` do not re-check the enable flag in adapter `assert_usable`; they only rely on registry gating and then check model paths. Some video adapters are opt-out/default-on (`nodes/_otr_video_engines/eng_ltx_video.py`, `eng_ltx_av.py`, `eng_visualizer.py`) rather than strict opt-in. Concrete fix: add a real gate inventory by engine: registry gate, adapter gate, default-off vs default-on, remaining model/env gates. Build from that inventory, not the current blanket statement.

4. [Change 5] Deriving dep/GPU verification solely from `vram_class != cpu` or non-empty `model_requirements` is too weak. `scripts/otr_video_dep_pilot.py` currently encodes non-capability metadata: import module, adapter class, forward method, sidecar/in-stack assumptions, banned-dep posture, and `assumed_call`. `nodes/_otr_video_engines/registry.py` CAPABILITIES does not contain enough data to recreate that. Concrete fix: introduce a separate verification manifest or extend metadata explicitly; do not infer dep-pilot coverage from VRAM/model requirement rows alone.

SHOULD-FIX:
1. [Change 3] Removing `requires_flag` from shared `EngineCore` while `VideoEngine` and `ImageEngine` still document structural parity with audio will leave the protocol story incoherent. `nodes/_otr_shared/engine_registry_base.py`, `nodes/_otr_video_engines/registry.py`, and `nodes/_otr_image_engines/registry.py` all describe `requires_flag` as part of the core contract. Fix the protocol narrative and tests as part of the same change.

2. [Open question 2 + Change 4] The audio registry owns its own enum in `nodes/_otr_audio_engines/registry.py`; it does not import the shared enum from `nodes/_otr_shared/engine_registry_base.py`. Keeping `GATED_BY_FLAG` in the shared enum after video/image deletion is harmless but misleading. Prefer either deleting it from shared video/image code or marking it legacy-test-only until the taxonomy is cleaned.

3. [Tests] “Update the ~20 tests” understates the contract surface. `git grep` shows flag/gated expectations across video, image, dep-pilot, GPU-smoke, coverage, and dropdown tests under `tests/`. Fix: list test families by behavior contract: registry usability, adapter disk gates, dropdown curation, force-map/lsync routing, dep-pilot, and coverage acceptance.

OPTIONAL / NICE-TO-HAVE:
- Add one short release invariant: “A non-validated engine may exist in source, but cannot be selected or forced in production paths.” That would make the dropdown/registry split defensible.

CUT THESE (scope / over-engineering):
1. [Change 4] Cut enum deletion from the first build chunk. It is cleanup, not the product behavior, and keeping the enum briefly reduces churn while gates/tests are converted.

2. [Change 5] Cut “derive verification from CAPABILITIES” as a broad redesign. Preserve an explicit verification manifest first; later consolidate metadata only after the no-flag behavior is green.

3. [Scope] Cut dark 3D scaffolds (`triposr`, `triposg_talk`, `hunyuan3d_talk`, `trellis_talk`) from this early deletion pass. They are safe to cut because they are already excluded from validated dropdowns in `tests/test_tested_only_dropdown_gate.py` and do not serve the stated “picked model renders” goal yet.