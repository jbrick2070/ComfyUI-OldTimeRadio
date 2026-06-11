<!-- requested_model: openai/gpt-5.5 | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: yes-with-fixes. The plan is close, but several sections specify behavior that cannot be implemented from the shown call signatures/data flow as written.

MUST-FIX BEFORE BUILD:
1. [F5] Announcer `char_id` resolution is assigned to `build_execution_plan`, but the shown signature is `build_execution_plan(beats, budget, creative, policy)` and it has no `ledger`/`cast` access. It cannot “resolve from the CAST table by name match” there. Concrete fix: either resolve/stamp announcer `char_id` in `extract_beats(ledger)` where `ledger["cast"]` is available, or change `build_execution_plan(..., ledger)` and update its only caller in `OTRShotLock.lock`. Then stamp `"char_id": b["char_id"]` on every shot row.

2. [F2] Trace prompt observability has no defined data path. `run_episode` currently receives only `request` and appends `{"shot_id","attempts","final_engine"}`; `build_request_from_shot` currently does not return `prompt_source`, `prompt_chars`, or `prompt_sha8`. Concrete fix: in `build_request_from_shot`, add request metadata such as `req["_prompt_source"]`, `req["_prompt_sha8"]`, `req["_prompt_chars"]` after final prompt selection. In `run_episode`, copy those keys into trace rows for `ltx_video`/`wan_i2v` attempts/final engines. Define fallback values for the soak/default `build_request` path, or explicitly gate the fields to the real episode request builder.

3. [F2] The diversity gate is not attached to any concrete execution point. The plan says “Diversity gate reads the trace” but does not say whether this runs in `run_episode`, node-92 report generation, acceptance script, or tests. Concrete fix: add a named pure helper, e.g. `assert_ltx_prompt_diversity(trace)` or `ltx_prompt_diversity_status(trace)`, and call it from the durable node-92/history report path. [ASSUMPTION] If node-92 code is elsewhere, wire it there and make failure/warn behavior explicit.

4. [F3] The proposed anchor can fail the new first-160-character guard when `appearance` is long. `_appearance_for_char` can return unrestricted `character_description`; `subject_anchor = f"{appearance}, face visible, speaking to camera"` may put `face`/`speaking`/`camera` after char 160. Concrete fix: put the face/camera tokens first or truncate appearance in the anchor, e.g. `subject_anchor = f"face visible, speaking to camera, {appearance[:120].rstrip()}"`, and make the guard check the same bounded prefix.

5. [F3] “EVERY talking-head prompt path” conflicts with the shown ShotLock scope. `CHARACTER_BEARING_ROLES = {character_video}` only, while `render_driver` has `announcer_visual` audio-driven profiles (`humo`, `latentsync`) and the workflow can select talking-head announcer. Concrete fix: either extend creative derivation/anchoring to announcer talking-head shots when the selected engine family is `audio_driven_face`/`lipsync_overlay`, or explicitly constrain the saved workflow so no `announcer_visual` talking-head path exists this round. Do not leave the statement broader than the code path.

6. [F4] “update `char_id` AND every speaker-identity field (`speaker_role` etc.; enumerate at build)” is not build-ready. It leaves the exact mutation set undefined, which risks fixing `char_id` while leaving stale speaker/voice fields. Concrete fix: before coding, enumerate the actual line keys from `OTR_LedgerScriptWriter.py`/ledger schema and list them in the spec/test, e.g. `char_id`, `speaker_role`, display-name field(s), voice/cast reference field(s). verify: exact writer line schema and pre-freeze scrub location.

7. [F1] `eng_ltx_video.py` currently has no logger import or `_LOG` in the shown excerpt, but F1 requires LOUD warnings. Concrete fix: add `import logging` and `_LOG = logging.getLogger("OTR.video.eng_ltx_video")` or use the project’s established logger naming before adding `_env_int`.

8. [F6/F1] The CPU test requirement for “238->121->121 snap; cap=120 -> 113” is not supported by the proposed implementation because the length logic remains embedded inside `render_clip`, which requires wrapper graph classes/GPU-adjacent setup. Concrete fix: extract a pure helper such as `_ltx_frame_length(target_frame_count)` that performs min/cap/snap and unit-test that helper. `render_clip` should call it.

SHOULD-FIX:
1. [F2] Define diversity behavior for zero or one brief-composed LTX prompt. “Must not all be equal” is invalid for `n < 2` and can create false failures in small fixtures. Concrete fix: only enforce diversity when `len(brief_composed_ltx_sha8s) >= 2`; otherwise emit info/warn only.

2. [F2] Existing `render_driver.build_request_from_shot` uses `_is_open = role in ("announcer_visual", "music_visual")`. F2 says synthetic-open detection must be empty `source_line_ids` or `shot_id.endswith(OPENING_MUSIC_BEAT_ID)`, “never role alone.” Concrete fix: replace role-based open detection entirely; keep role only for role-clause wording, not for synthetic-open/env-override detection.

3. [F2] Importing `OPENING_MUSIC_BEAT_ID` from `otr_shot_lock.py` inside `render_driver.py` creates a hidden module dependency from render to ShotLock. Concrete fix: either move the constant to a shared lightweight module or duplicate a local string constant with a comment tying it to ShotLock. Avoid importing ShotLock if that risks Comfy/node registration side effects. verify: import side effects of `otr_shot_lock.py`.

4. [F2] Prompt source enum is underspecified for M4 creative prompts. Existing ShotLock stores `creative["source"]` as `"llm"`, `"template"`, or `"template_consistency"`, while F2 wants trace source `m4|env|brief+beat`. Concrete fix: map any existing creative text prompt to `prompt_source="m4"` and optionally preserve the detailed ShotLock source separately as `prompt_subsource`.

5. [F5] `build_clip_manifest` currently computes `beat_id` as `shot_id` when `source_line_ids` is empty. For the synthetic opening shot, this yields `"shot_b000_music_open"` instead of `"b000_music_open"`. Concrete fix: use the same `_beat_id_for_shot(shot)` logic or strip `shot_` for no-source synthetic shots.

6. [F5] “rows gain `char_id` + `init_image` so the face-acceptance check is mechanical” needs exact derivation. Concrete fix: in `build_clip_manifest`, resolve `char_id = shot.get("char_id") or line.get("char_id") or ""`, then `init_image = _portrait_index(led).get(char_id, "")`, and include both in each row. If the acceptance check needs cast-table equality, also include the cast display/name or portrait object id. [ASSUMPTION] Depends on what the face-acceptance checker consumes.

7. [F3] The new `_prompt_is_consistent` requirement says “at least one core appearance token” but `_appearance_for_char` can still return `""` if the cast lookup fails. Concrete fix: when `appearance` is empty for a character beat, log a warning and force deterministic fallback with the cast name if available; otherwise mark the row unverifiable rather than passing an object/scenery prompt.

8. [F4] The detector only handles `name,` vocatives. If the known failure includes `Name:` or dash punctuation, it will ship. Concrete fix: either keep the comma-only scope and pin it in tests, or expand normalized punctuation to comma/colon/dash consistently. Do not imply broader “case/punct-normalized” behavior while only matching comma.

9. [Acceptance] “audio byte-identical” is ambiguous because this round may re-run a fresh episode with newly generated audio unless the frozen audio spine refers to intra-run immutability only. Concrete fix: define the comparison target: byte-identical to the pre-render master audio file for the same fresh run, not byte-identical to a previous episode render. [ASSUMPTION] Based on “Each accept30 run renders a FRESH episode.”

OPTIONAL / NICE-TO-HAVE:
- [F2] Include the first 80-120 chars of the final prompt in the INFO log in addition to sha8/chars/source; this makes operator log review faster without relying only on hashes.
- [F5] Include `prompt_sha8` in manifest rows too, not just trace, if node-92/history is not always easy to correlate with clips.
- [F4] Add a single summary count of self-vocative repairs/warnings to the writer report.

CUT THESE (over-engineering):
1. [F1] The “Wan: NO cap this round” one-line code comment is not functional and is safe to cut. The absence of Wan cap code is sufficient; comments pointing to a round plan add maintenance noise.

2. [F3] “Document that the char_beats loop is the only caller of the guard” is safe to cut if the tests cover all guard callers. It does not affect runtime behavior and may go stale as soon as announcer/talking-head scope is fixed.

3. [F6] Separate “both-absent” plus “one-absent” beat-intent/arc-phase fixtures can be reduced to a table-driven single test covering `(intent, phase) = both/intent-only/phase-only/neither`. Same coverage, less test bulk.