<!-- requested_model: x-ai/grok-4.3 | resolved_model: x-ai/grok-4.3-20260430 -->

**VERDICT:** no. Root causes in FIX 1 and FIX 2 are incompletely diagnosed or too vague to implement; FIX 3 caller site is unconfirmed; changes would be made without verifying the exact seams against the provided excerpts.

**MUST-FIX BEFORE BUILD:**

1. [FIX 1] `build_clip_manifest` (render_driver.py excerpt, the block that emits `rows.append({... "init_image": _portrait_index(led).get(row_char, ""), ...})`) unconditionally keys the opener's `init_image` by `char_id=""` via `_portrait_index`. The synthetic opener (`beat_id` ending `_OPENING_MUSIC_SUFFIX`, `char_id=""` from `otr_shot_lock.py:derive_opening_music_beat`) has its still recorded under `kind=scene_*` + `beat_id` (see `_still_index`). The proposed "resolve from ledger['images'] when char_id empty" does not name the concrete lookup or guard; it must use `_still_index` (or the same `_SCENE_INIT_FAMILIES` path already present in `build_request_from_shot`) for the opener only. Add the check and the `row["init_image"] = _still_index(led).get(bid, "")` fallback; otherwise the black frame is unchanged.

2. [FIX 2] The section ends with an explicit open question ("Panel: which fork is more likely, and where exactly does the per-role selection get dropped?") and a "verify-at-build" instruction. No concrete file/function or edit is given for either the policy-JSON path or the `gen_fn` path. A build cannot proceed until the exact drop point is identified and a minimal patch is written (e.g., the call site inside `dispatch_images` or the engine registry lookup).

3. [FIX 3] `_resolve_title_timing` (video_engine.py) receives `first_dialogue_f=None` because it is called on pre-audio-timing lines. The proposed fix ("feed it the audio-timed lines (post-EpisodeAssembler)") does not name the caller. Verify: every call site of `_resolve_title_timing` (inside `SignalLostVideoRenderer` or the legacy [Video] path) and confirm whether that caller already has access to the post-`overlay_audio_timing` ledger. Without the caller location the change cannot be made.

**SHOULD-FIX:**

4. [FIX 1 + otr_shot_lock.py] `derive_opening_music_beat` emits the synthetic beat with `char_id=""` and `_synthetic_open=True`. The image phase (`otr_image_gen_dispatcher.py:dispatch_images` + `OTRImageDirector`) must include that beat in the `image_prompts["objects"]` list under `kind=scene_*` / `beat_id=b000_music_open` so the still actually lands in `ledger['images']`. The current plan never confirms this seam; add an explicit check or the opener still will be absent even after the manifest fix.

5. [FIX 3] The `music_open_line=None` path in `_resolve_title_timing` still has an envelope fallback (`_envelope_intro_end`). If the audio-timed-lines change is made, the fallback becomes dead code for normal runs; remove or guard it so it cannot silently produce a ~1 s window again.

**OPTIONAL / NICE-TO-HAVE:**

- Keep the `[BUG-403/404 instr]` log lines only until the three fixes are green; they are temporary scaffolding.

**CUT THESE (over-engineering):**

- The entire "Open questions for the panel" block can be deleted once the three MUST-FIX items above are answered; it does not contain actionable code changes.
- The long "What is already CORRECT" preamble repeats invariants already stated in the render_driver.py and otr_shot_lock.py excerpts; it can be reduced to a one-line reference to the audio-spine and `derive_opening_music_beat` contracts.

**ASSUMPTIONS** (inferred beyond the provided excerpts):
- The opener still is minted by the image dispatcher under `kind=scene_*` + `beat_id` (verify against `otr_image_gen_dispatcher.py` + ShotLock interaction).
- `build_clip_manifest` is the only consumer of the opener's `init_image` that must be changed (no other call sites in the unshown render path).