VERDICT: yes-with-fixes. The plan misses one live visual seam and needs tighter test/constant migration wiring before build.

MUST-FIX BEFORE BUILD:
1. [2] `_compose_mesh_fodder_prompt` is omitted from the 3A routing list. It currently appends `get_era_tail(meta, profile="still")` inside a broad `except Exception: pass` at `nodes/otr_meta_brief_image_prompt.py:1223-1243`. If `get_era_tail` becomes pack-routed, this seam can either keep the old fallback tail or swallow `VisualStyleError`. Concrete fix: include `_compose_mesh_fodder_prompt` in 3A tail routing and the no-swallowed-style-errors AST pin; catch only `ImportError` around flat-import shims.

2. [2]/[5] Constant ownership migration will break or weaken existing tests unless explicitly rewired. Existing tests import/read `STYLE_TAIL_DEFAULT`, `IMAGE_GRADE_TAIL`, `RADIO_BROADCAST_TAIL`, and `ERA_TAIL_DEFAULT` from `nodes/_otr_story_brief_helpers.py` (`tests/test_brief_prompt_finishing.py:37`, `tests/test_still_spine_helpers.py:157`, `tests/test_talking_portrait_s4b.py:99`, `tests/test_video_platform_aseam.py:478`). If constants move only to `_otr_visual_styles.py`, these tests fail; if aliases remain in helpers, production can keep using the old path. Concrete fix: update tests to read the default `sci_fi_radio` pack or a dedicated extraction fixture, and make the direct-read guard allow only that fixture/test path.

3. [4]/[6] The selector test update list is incomplete unless it includes the existing “last optional” pins. `tests/test_source_bank_widget_2c.py:61-64` currently asserts `source_bank` is last optional; `tests/test_story_scaffold_toggle.py:50-57` asserts `story_scaffold` then `source_bank` are the last two. Concrete fix: update these to `story_scaffold -> -3`, `source_bank -> -2`, `visual_style -> -1`, or they will fail immediately after appending slot 26.

SHOULD-FIX:
1. [1]/[2] Define the concrete helper signatures for “style resolution happens ONCE per composer entry.” Today callers use `finish_visual_prompt(meta, ...)` and `get_era_tail(meta, ...)` directly (`nodes/_otr_story_brief_helpers.py:456-554`, `nodes/_otr_video_engines/render_driver.py:1731-1733`). Concrete fix: specify whether helpers accept `style=None`, `era_tail=None`, or resolve internally. Then pin render_driver’s `style_tail=False` path with a test.

2. [4] Add a gate-order test that `get_style(visual_style)` runs before `_apply_story_scaffold_env`, refine capture, OpenRouter/Comfy budget reset, and `_resolve_inputs`. Current source_bank gate is first at `nodes/OTR_LedgerScriptWriter.py:2561-2567`; the new style gate must sit beside it.

OPTIONAL / NICE-TO-HAVE:
- Add one forced-meta test for `mesh_fodder` specifically, since it is easy to miss and feeds 3D.

CUT THESE (over-engineering):
1. [1] Compose-time forbidden-term warning state. The plan already cuts it; keep it cut. Load-time lint is enough for v1 and avoids resident-server global state.