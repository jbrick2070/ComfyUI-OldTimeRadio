<!-- requested_model: google/gemini-3.1-pro-preview | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. Solid plan, but F1 introduces a fatal NameError and F5's join normalization will be lost without an explicit schema addition.

MUST-FIX BEFORE BUILD:
1. [F1] `_env_int` is undefined in `eng_ltx_video.py`. Using it will cause a `NameError` crash during the render loop. Fix: Parse the environment variable explicitly: `try: cap = int(os.environ.get("OTR_LTX_MAX_FRAMES", 121)) except ValueError: cap = 121`, then clamp it `cap = max(_LTX_MIN_FRAMES, cap)`.
2. [F5] `build_execution_plan` in `otr_shot_lock.py` does not currently write `char_id` into the `shot` dictionary. If the Announcer `char_id` is normalized at the ShotLock JOIN but not saved to the shot row, `render_driver.py` will fall back to reading the empty `char_id` from the frozen line. Fix: Add `"char_id": b["char_id"]` to the shot dict in `build_execution_plan`, and update `render_driver.py::build_request_from_shot` to read `char_id = str(shot.get("char_id") or line.get("char_id") or "")`.

SHOULD-FIX:
1. [F4] [ASSUMPTION] `OTR_LedgerScriptWriter.py` is not provided in the grounding excerpts. Verify: Ensure the pre-freeze self-vocative re-attribution logic correctly identifies the interlocutor's `char_id` from the exchange context and modifies the `lines` array before the freeze cascade locks it.
2. [F5] In `render_driver.py::build_request_from_shot`, when warning about a missing portrait index, check `engine_family(shot.get("engine_id", "")) in ("audio_driven_face", "character_3d")` rather than just the role. This accurately catches talking-head failures even if roles are overridden via `OTR_FORCE_ENGINE_MAP`.

OPTIONAL / NICE-TO-HAVE:
- [F3] `_prompt_is_consistent` does not receive the shot's role, so it implicitly relies on the caller (`derive_creative_directives`) only passing character-bearing beats. Document this assumption in the docstring so future callers don't accidentally fail object-only b-roll prompts.