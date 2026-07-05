VERDICT: yes-with-fixes. The plan is close, but current broad exception blocks and an omitted still_word seam will silently defeat the fail-loud/style-routing contract.

MUST-FIX BEFORE BUILD:
1. [2] Style loader failures will be swallowed in existing prompt seams. `C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/otr_meta_brief_image_prompt.py:352-367`, `:1172-1183`, `:1258-1269`, and `:1533-1555` wrap style/finish/grade work in `except Exception: pass`. If `finish_visual_prompt()` or pack lookup raises `UnknownVisualStyleError` / `VisualStyleValidationError`, these paths silently ship unstyled prompts. Concrete fix: remove those broad catches or re-raise `VisualStyleError`; keep only the package/flat `ImportError` fallback.

2. [0]/[2] The routing list omits `compose_still_word_prompt`, but it directly imports and appends `IMAGE_GRADE_TAIL` and uses `get_era_tail` at `C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/otr_meta_brief_image_prompt.py:823-864`. That contradicts the planned AST guard: “no production module reads STYLE_TAIL_DEFAULT / IMAGE_GRADE_TAIL / RADIO_BROADCAST_TAIL / ERA_TAIL_DEFAULT except the style module”. Concrete fix: include `compose_still_word_prompt` in 3A routing, using pack `era_tail`/`image_grade_tail`, while leaving typography/backdrop maps in Python per section 8.

3. [1] The forbidden-term warning API is underspecified and easy to implement incorrectly. “log once per (style, term, seam) per episode” has no seam parameter on `finish_visual_prompt()` (`_otr_story_brief_helpers.py:524-557`) and no reliable episode id in the `meta` dict at initial stamp time (`OTR_LedgerScriptWriter.py:2799-2808`). Concrete fix: define a helper signature such as `warn_forbidden_terms(meta, style, prompt, seam, episode_id=None, logger=log)` and pass explicit seam names from each composer, or cut compose-time warnings from v1.

SHOULD-FIX:
1. [1]/[2] `get_era_tail()` currently documents “NEVER empty, never raises” at `C:/Users/jeffr/.../nodes/_otr_story_brief_helpers.py:259-333`. Unknown visual style = hard error changes that contract if style resolution happens inside it. Concrete fix: update the doc/tests to “never raises except explicit invalid visual_style”, or keep `get_era_tail()` fail-soft and resolve style in a separate fail-loud wrapper.

2. [1] Row schema does not specify `schema_version` type/value. Existing story packs use an explicit string version and known-version set (`nodes/_otr_story_pack.py:23`, `:127-135`). Concrete fix: pin visual style `schema_version` to a string such as `"v1"` through `KNOWN_STYLE_SCHEMA_VERSIONS`, and reject missing/unknown versions.

3. [1] `style_id == filename` needs a filename-safe id rule, not just equality. Concrete fix: validate `style_id` against a conservative regex like `^[a-z0-9_]+$` before using it as a registry id/dropdown value.

OPTIONAL / NICE-TO-HAVE:
- Add a frozen dataclass for the loaded visual style row so seam code consumes attributes, not raw dict keys.
- Add one test that corrupts a non-default pack and confirms `INPUT_TYPES()` fails loudly.

CUT THESE (over-engineering):
1. [1] Compose-time `forbidden_terms` WARN-ONLY logging in v1. It has no enforcement behavior, requires new per-episode/seam state, and risks global-state bugs in a resident ComfyUI server. Load-time lint is enough for the first build.