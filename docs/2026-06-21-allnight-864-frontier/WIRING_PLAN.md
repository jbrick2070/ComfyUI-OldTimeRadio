# Signal Lost -- Workflow Wiring Plan (v2, post roundtable 3 + grounded consumer audit)

How the story-engine fixes touch `workflows\otr_scifi_16gb_full.json`. Conclusion: **v1 needs ZERO workflow-JSON edits**, now VERIFIED against the real downstream consumers (not just asserted).

## Architecture fact
The entire story pipeline is ONE node: **node id 1, `OTR_LedgerScriptWriter`**. Its `script_json` output flows to **node 62 `OTR_LedgerFreezeCascade`** (the freeze/gatekeeper), which re-emits `script_json` and fans out to every consumer: SceneSequencer(3), SignalLostVideo(12), BatchCharacterVoices(81), AnnouncerVoice(82), StableAudioTheme(83), ShotLock(90), MetaBriefImagePromptGen(89). Cast cards reach audio via node 62 -> CastLock(80). All story logic is internal Python of node 1 and the modules it calls (`_otr_line_composer`, `_otr_line_hygiene`, `_otr_story_spine`, `_otr_dramatic_state*`, `_otr_casting`, `_otr_style_picker`) -- none are separate ComfyUI nodes (C4 holds).

## Node 1 widget map (CORRECTED: 23 entries, indices 0-22; next append index = 23)
`widgets_values` (JSON lines 62-84), **23 entries**:
```
[0] episode_title ""          [1] target_words 350         [2] num_characters 2
[3] creative_model "...Nemo"  [4] technical_model "...Nemo" [5] (reserved) ""
[6] include_act_breaks true   [7] act_count "auto"         [8] style_combo "let the story decide"
[9] style_custom ""           [10] creativity "balanced"   [11] (bool) false
[12] (float) 0.05             [13] (float) 1.03            [14] (int) 200
[15] lemmy_cameo "roll..."    [16] (bool) true             [17] (bool) true
[18] (bool) true              [19] openrouter_a "(enable)" [20] openrouter_b "(enable)"
[21] comfy_credits_a "(enable)" [22] comfy_credits_b "(enable)"
```
(Confirm names against `OTR_LedgerScriptWriter.INPUT_TYPES` before any future append. Positions are what matter; BUG-LOCAL-097.)

## Per-fix wiring impact (v1)
F1, F2, F3, F4, F5, F6, F7, F8 (arc_shape in additive `meta`): **ZERO workflow-JSON change.** F9, F10: DEFERRED (not built). No new widget, no INPUT_TYPES change, no output-link change.

## Downstream-consumer safety -- VERIFIED (grounded audit)
The "zero edits" claim is confirmed by reading the real consumers, not assumed:
- **Additive keys are safe.** All consumers parse via the shared `_otr_ledger_consumers.py` (`load_ledger` asserts only "top level is a dict"; `iter_lines`/`cast_lookup` are pure `.get()`); direct `json.loads` sites are try/except + `isinstance`-guarded. `meta.arc_shape` and `cast[].speech_signature` are read by NOBODY -- both strings appear 0 times in the codebase. They break nothing.
- **No fixed-count assumption.** SceneSequencer (`enumerate(lines)`), ShotLock (`for i, ln in enumerate(lines)`, budget from cumulative audio samples), SignalLostVideo (`sum(1 for ln in iter_lines)`), MetaBrief (`len(objs)`) all handle `len(lines)`/act count DYNAMICALLY. Nothing asserts 18 lines or 3 acts. So F8 is NOT required to keep the beat count fixed (we keep it fixed in v1 anyway as the conservative choice -- it is optional, not load-bearing).
- **Outro text / costly slot are invisible downstream.** `costly`/`outro` appear only inside `OTR_LedgerScriptWriter.py`. Consumers key off `line_id`/`char_id`/`speaker_role` + line `text`. F2/F3/F7 are invisible as long as `line_id`s and `speaker_role`s are preserved.
- **Cast prose is free-text downstream.** Extra gender/pronoun text and `speech_signature` reach image nodes only as concatenated prompt strings (never structurally parsed); the freeze does a plain `json.dumps` passthrough, so additive keys survive verbatim.

## The REAL constraints to honor (the freeze's CRITICAL invariants, `_otr_ledger_freeze.py`)
Node 1 already meets these; DO NOT regress them:
1. All 7 top-level lists present and list-typed: `cast, lines, beats, scenes, shots, music, clips` (`:124-133`).
2. Every line: unique, non-empty `line_id`; `speaker_role` in `{character, announcer, music_open, music_close, music_inter, sfx}` (`:91-98, 295-310`). -> F7 recompose keeps the same `line_id`; F2/F8 introduce NO new `speaker_role`.
3. Voiced lines (`character`/`announcer`): non-empty `char_id`; referenced `cast[]` entries keep `char_id`/`name`/`voice_preset` (adding `speech_signature` alongside is fine).
4. A skipped line: `text == ""` AND non-empty `tts_skip_reason` (`:320-345`).
A beat-count or missing-optional-meta mismatch is at most a WARNING in the freeze, never CRITICAL.

## Mandatory v1 gate (panel-required)
Before declaring v1 done: run a full end-to-end render of >=1 episode with the CHANGED node-1 code and the UNCHANGED `otr_scifi_16gb_full.json`, and confirm every downstream node processes the ledger without error (this is already covered by the headless smoke in CODING_SPRINT_PLAN Sprint 0/exits). Optionally diff `script_json` of a golden seed before/after to catch any unintended schema change.

## v1 rule
**v1 forbids any `widgets_values`/`INPUT_TYPES`/link edit.** Arc_shape and anti-repeat stay internal (additive `meta`/constants). The widget-append procedure below is a FUTURE appendix only -- do not touch the JSON for this release.

## Appendix: IF a widget is ever exposed (future, not v1)
Per CLAUDE.md section 0: APPEND ONLY at the end (next index = current length = 23); never insert mid-list (BUG-LOCAL-097). Add the matching optional widget to the END of `OTR_LedgerScriptWriter.INPUT_TYPES` -- note (panel): ComfyUI maps positional `widgets_values` to the order of the `required` dict, so a widget that must hold a saved position belongs at the end of `required`, not `optional`. A widget later converted to an input keeps its value slot AND gains `"widget": {"name": ...}`. Re-validate: `OTR_WorkflowValidator` + JSON round-trip (load->dump->reload, no diff) + link/widget audit (widget-count vs live `INPUT_TYPES`; every wired input-name in `INPUT_TYPES`; link referential integrity `[link_id, src, src_slot, dst, dst_slot, type]`). Same commit as the code; push to `v2.0-alpha`.
