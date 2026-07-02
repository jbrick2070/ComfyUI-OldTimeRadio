VERDICT: yes-with-fixes. Core change is buildable, but S2/S3/S4 leave prompt precedence, appearance source, and init-image routing ambiguous.

MUST-FIX BEFORE BUILD:
1. [S2] Announcer prompt replacement is underspecified when `creative.text_prompt` exists. `build_request_from_shot` takes the M4 branch first at `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\nodes\_otr_video_engines\render_driver.py:1402`; the announcer talking swap is only in the later no-text scene branch at `render_driver.py:1532`. Concrete fix: state whether ia2v announcer talking-register overrides M4/env prompts, and add a test for an `announcer_visual` + `ltx_audio_in` shot with `creative.text_prompt`.

2. [S3] “<=120-char appearance fragment” has no defined data source. Current render path has `creative.text_prompt`, `char_id`, and `_portrait_index`, but `_portrait_index` only returns paths (`render_driver.py:401`). Existing cast appearance fields are discoverable in `nodes\_otr_wardrobe.py:71`. Concrete fix: define a helper that resolves appearance from cast fields (`character_description`, `appearance`, `description`, `portrait_prompt`), sanitizes style/camera/text tokens, then falls back to the first M4 clause.

3. [S4] Portrait routing conflicts with current `ltx_audio_in` routing. The driver forces all `ltx_audio_in` shots through `_still_index` at `render_driver.py:1072` and clears `init_image` if no scene still exists at `render_driver.py:1095`; tests currently assert character beats use the wide scene still at `tests\test_ltx_audio_in_routing.py:96` and `:105`. Concrete fix: either exclude S4 until P5 is proven, or add an explicit ia2v-character predicate before that branch: portrait wins, scene still is not used for char-face speech, and missing portrait fails loud.

SHOULD-FIX:
1. [S1] Specify registry-object lookup for the hook. “Driver consults the ENGINE” should mean `_vreg.get_engine(engine_id).wants_talking_prompt()`, not a fresh engine instance; otherwise any future instance state is ignored. Current helper shape is at `render_driver.py:586`.

2. [S1] Do not silently convert recipe misconfiguration into “old prompt register.” `LtxAudioInEngine.wants_talking_prompt()` can derive from `_recipe_config(self._recipe())` (`eng_ltx_av.py:392`); define whether exceptions fail loud or are logged. Silent `False` makes debugging prompt routing harder.

3. [S6] Align acceptance with the existing evaluator. `scripts\otr_talking_radio_probe_eval.py` reports `mouth_motion_mean` and an r/delta criterion, while S6 says `mouth-motion >= 2.0`. Concrete fix: name the script/command and choose one pass condition.

OPTIONAL / NICE-TO-HAVE:
1. [S5] Add a byte-unchanged test for `distilled_native`/`sharp_lora` creative prompts, not only role prompts, so the hook cannot leak into non-ia2v recipes.

CUT THESE (over-engineering):
1. [S4] Cut from the initial build until P5 is complete. S1-S3 are enough to test the prompt-register hypothesis; S4 touches init-image invariants and existing tests, so it is only safe after verify: P5 result.