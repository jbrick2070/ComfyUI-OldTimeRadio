<!-- requested_model: openai/gpt-5.5 | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The plan has build-visible gaps in the grounded code paths: missing helper/API wiring, synthetic-open timeline breakage, and char_id/manifest assumptions that current code does not support.

MUST-FIX BEFORE BUILD:
1. [F1] `eng_ltx_video.py` has no `_env_int` helper in the excerpt. Adding `cap = _env_int(...)` as written will raise `NameError`.
   Concrete fix: add a local helper in `eng_ltx_video.py` before `LtxVideoEngine`, e.g. parse `os.environ[name]`, log warning on invalid, clamp to `_LTX_MIN_FRAMES`, and return default when absent. Then call it in `render_clip`.

2. [F2/F6/Acceptance] The planned “DIVERSITY gate” has no wired data path. `build_request_from_shot()` returns only the request; `run_episode()` discards request prompt metadata; `build_clip_manifest()` currently emits no prompt text/sha. Tests requiring “diversity sha8s present” cannot pass as written.
   Concrete fix: persist text-engine prompt metadata somewhere deterministic, e.g. add to `trace` in `run_episode()` for `ltx_video`/`wan_i2v`: `prompt_sha8`, `prompt_chars`, `prompt_source`, `beat_id`, `role`; or add it to `build_clip_manifest()` rows from shot/request metadata. Then implement the all-equal check against that structure.

3. [F2/Acceptance] The synthetic opening music shot will break positioned timeline assembly unless `build_clip_manifest()` is changed. Grounded code: `build_execution_plan()` stamps synthetic shots with `start_s`/`dur_s` and empty `source_line_ids`; `build_request_from_shot()` falls back to shot timing; but `build_clip_manifest()` only reads `start_s` from `lines[bid]` and ignores `shot["start_s"]`. Then `plan_timeline_segments()` disables positioned mode because not all rows have `start_s`.
   Concrete fix: in `build_clip_manifest()`, when line `start_s` is missing, fallback to `shot.get("start_s")`; same for any needed duration metadata. For synthetic rows, preserve a stable `beat_id` such as `b000_music_open` instead of only `shot_id` if downstream gates expect beat ids.

4. [F5] “Normalize at the ShotLock JOIN (shot rows)” is insufficient because grounded `render_driver.build_request_from_shot()` ignores `shot["char_id"]` and uses only `line.get("char_id")`. A shot-row announcer fix will not affect portrait lookup or HuMo input.
   Concrete fix: stamp resolved `char_id` on the shot row and change render driver to use `char_id = str(line.get("char_id") or shot.get("char_id") or "")`. Apply the missing-portrait warning after this resolved value.

5. [F5/Acceptance] Acceptance requires “manifest char_id == staged portrait == cast table,” but grounded `build_clip_manifest()` rows do not include `char_id`, portrait path, or init image. The gate is not checkable.
   Concrete fix: extend manifest rows with resolved `char_id` and `init_image`/portrait path, using the same resolution logic as `build_request_from_shot()`.

6. [F3] The anchor change only explicitly covers `_deterministic_template`; current `derive_creative_directives()` also has LLM/full-text and LLM-directives composition paths. Those paths can still produce prompts where the subject is not leading with “face visible, speaking to camera.”
   Concrete fix: create one common `subject_anchor = "<appearance>, face visible, speaking to camera"` and prepend it to every character-bearing final prompt path before `finish_visual_prompt()`: LLM `text_prompt`, LLM-directives composed prompt, and deterministic fallback. Then run `_prompt_is_consistent()` against that anchored text.

7. [F3] `_prompt_is_consistent()` currently has no role parameter and is only `text_prompt, appearance, setting`. The plan says “for `CHARACTER_BEARING_ROLES` additionally require a person-anchor,” but the function cannot distinguish role unless changed or only called for character beats.
   Concrete fix: either add `role`/`require_person_anchor` parameter, or document and enforce that the stricter person-anchor check is always applied inside `derive_creative_directives()` because it only iterates `char_beats`.

8. [F4/Acceptance] F4 says ambiguous self-vocative cases “LOUD-log and keep,” but Acceptance says “no line ships whose text opens with its own speaker’s vocative name.” These conflict.
   Concrete fix: either scope Acceptance to the production target where no ambiguous cases remain, or make unresolved self-vocatives a pre-freeze blocking/error condition for the fresh render. If fail-soft is mandatory, change the acceptance wording to allow logged unresolved ambiguities.

9. [F4] [ASSUMPTION] The repair says “re-attribute the line’s `char_id`,” but writer/voice selection may also depend on speaker display/name fields, role fields, or cast reference fields not shown here. Changing only `char_id` may leave audio/video incoherent.
   Concrete fix: verify: in `nodes/OTR_LedgerScriptWriter.py`, identify every line field used downstream for TTS voice, speaker display, and cast lookup. Update all speaker identity fields consistently before freeze, not just `char_id`.

SHOULD-FIX:
1. [F2] The plan distinguishes `music_visual (synthetic open)` but does not specify the detector. Grounded synthetic shots have empty `source_line_ids` and `shot_id == "shot_b000_music_open"` from `OPENING_MUSIC_BEAT_ID`.
   Concrete fix: in `build_request_from_shot()`, detect synthetic open via empty `source_line_ids` or `shot_id.endswith("b000_music_open")`, not just role.

2. [F2] The requested INFO line format says `beat=bNNN`, but grounded `build_request_from_shot()` uses `shot.get("shot_id")` in current logging. Synthetic shots and ordinary shot ids are `shot_b...`, not raw beat ids.
   Concrete fix: log `_beat_id_for_shot(shot)` and normalize synthetic open explicitly.

3. [F2] Current fallback branch still honors `OTR_LTX_RADIO_PROMPT` verbatim for open roles. If that override is set to the same string across LTX shots, the diversity gate may fail despite intentional override.
   Concrete fix: either exempt operator override from diversity enforcement, or include `prompt_source=env_override` and make the diversity gate warn-only when override is active.

4. [F1] The plan says “Same cap wired into `eng_wan_i2v` if it shares the ask path,” but no `eng_wan_i2v` grounding is provided.
   Concrete fix: verify: inspect `nodes/_otr_video_engines/eng_wan_i2v.py` for frame-count snapping/length constraints before adding any cap. Do not add speculative code.

5. [Acceptance] `stddev(per-second YAVG) > 2.0` is not guaranteed by F1. Capping LTX length plus composite hold-last-frame can still produce a low-luma or low-motion scene depending on generated content.
   Concrete fix: make this a diagnostic, not a hard build gate, or gate on the concrete fixes: cap log present, non-identical prompt sha8s, and operator frame spot-check.

6. [F6] The LTX cap test should cover non-`8n+1` overrides. With the grounded snap formula `((length - 1) // 8) * 8 + 1`, `cap=120` should produce `113`, not `120`.
   Concrete fix: add a test asserting post-snap length is `<= cap` and satisfies `8n+1`.

7. [F3] The proposed person-anchor token list includes broad words like `person`, `man`, `woman`. That can pass prompts that mention a person in the background rather than the cast member.
   Concrete fix: require at least one core appearance token plus one explicit face/speaking/camera token near the beginning of the prompt, e.g. within first 160 chars.

OPTIONAL / NICE-TO-HAVE:
- Add a single helper in `render_driver` for prompt sha/log construction so tests do not duplicate hashing format.
- Include `prompt_source` in the clip manifest for easier postmortems.
- Add one fixture with missing `beat_intent` and present `arc_phase`, and one with present `beat_intent` and missing `arc_phase`, to lock silent degradation behavior.

CUT THESE (over-engineering):
1. [F1] Cut “same cap wired into `eng_wan_i2v`” from this round unless verification shows the same long-frame failure. It is explicitly operator-gated today and no grounding excerpt proves the shared path.
2. [Acceptance] Cut hard `stddev(per-second YAVG) > 2.0` as a release gate. It is content-sensitive and not directly guaranteed by the code changes; keep it as a diagnostic metric.
3. [Invariants] Cut “7 gated commits stay unpushed” from the build spec. It is release procedure, not a code invariant, and cannot be validated by suite or runtime tests.