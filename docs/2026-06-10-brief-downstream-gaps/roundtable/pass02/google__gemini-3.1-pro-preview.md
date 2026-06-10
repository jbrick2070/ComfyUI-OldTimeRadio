<!-- requested_model: google/gemini-3.1-pro-preview | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

core_tokens` is empty and default to `True`, or mirror the logic in `otr_meta_brief_image_prompt.py`.

    *   Let's check `_core_tokens` in `otr_shot_lock.py`:
        ```python
        def _core_tokens(text: str) -> list:
            toks = [t.strip(",.;:").lower() for t in str(text).split() if len(t) > 3]
            return toks[:6]
        ```
        If `text` is `"The cop."`, `toks` is `[]`.
        Fix: `toks = _core_tokens(appearance); appearance_ok = (not toks) or any(tok in low for tok in toks)`.

    *   Let's check `log_story_brief_disposition` again.
        ```python
        n_setting = len(terms.get("setting") or [])
        ```
        If `terms.get("setting")` is `"a vintage studio"`, `len()` is 16.
        This is a minor bug, but I'll add it as a SHOULD-FIX. `isinstance(..., list)` should be checked, or just `len([terms.get("setting")])` if it's a string.

    *   Let's check `finish_visual_prompt` in `_otr_story_brief_helpers.py` again.
        ```python
        base = base.replace(NO_TEXT_CLAUSE, "").replace(", ,", ",")
        base = base.strip().rstrip(",").strip()
        ```
        If `prompt` is `"cinematic, no on-screen text, moody"`.
        `base.replace("no on-screen text", "")` -> `"cinematic, , moody"`.
        `.replace(", ,", ",")` -> `"cinematic, moody"`.
        This works.

    *   Let's check `build_request_from_shot` in `render_driver.py` for `scene_broll` fallback.
        Plan F2: "This branch covers ALL