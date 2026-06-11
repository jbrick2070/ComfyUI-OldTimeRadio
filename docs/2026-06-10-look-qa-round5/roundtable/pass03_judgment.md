# Pass 03 judgment -- CONVERGED (3-model panel, ~$0.03; campaign total ~$0.08)

Verdicts: gpt-5.5 yes-with-fixes, gemini-3.1-pro yes-with-fixes, deepseek-v4-pro
yes-with-fixes. No "no" verdicts, no architecture changes, no new defect classes:
every remaining item is implementation-precision. CONVERGED at pass03; the build
folds the precision items below directly (recorded here, not re-paneled).

## Folded into the build (grounded, final)

1. **Announcer char_id resolves in `extract_beats(ledger)`** (GPT-1, Gem-S2) --
   it has the ledger/cast; `build_execution_plan` signature untouched; shot rows
   get `char_id` stamped there. Cast row name "ANNOUNCER" probe-verified tonight;
   absent -> skip silently (no new failure mode).
2. **Prompt metadata rides the request** (GPT-2, Gem-S1): `build_request_from_shot`
   stamps `_prompt_source` (m4|env|brief+beat; ShotLock's llm/template detail =
   `_prompt_subsource`), `_prompt_sha8`, `_prompt_chars`; `run_episode` copies
   them onto trace rows for text engines. Node-92 already embeds the whole trace
   in its report (verified in tonight's /history) -- DS-5's "node-92 must be
   updated" is a MISREAD; verify-at-build only.
3. **Diversity check = named pure helper** (GPT-3, DS-4): warn/fail status from
   trace rows, enforced only when n(brief-composed LTX) >= 2; env-source rows
   exempt; called at the report seam + unit-tested.
4. **Anchor AFTER the guard, tokens first** (Gem-1 guard-defeat + GPT-4, DS-S1):
   `_prompt_is_consistent` runs on the UNANCHORED candidate text; on pass, the
   anchor `"face visible, speaking to camera, {appearance[:120]}"` is prepended
   (face tokens lead, long appearance truncated); on fail, the anchored
   deterministic template replaces it. Guard tokens checked in a bounded prefix.
5. **Scope honesty** (GPT-5): the anchor covers `CHARACTER_BEARING_ROLES`
   (character_video) -- the only talking-head path in the saved workflow today
   (announcer routes to LTX). Announcer-as-talking-head is out of scope; noted.
6. **F4 field enumeration is a hard precondition** (GPT-6, DS-2): read the writer
   line schema + scrub site BEFORE coding; the repair moves char_id +
   speaker_role + any voice/cast ref found; comma-only vocative detector,
   pinned as comma-only in tests (GPT-S8).
7. **Pure `_ltx_frame_length(target)` helper** (GPT-8) -- min/floor/cap/snap in
   one CPU-testable function; `render_clip` calls it; module logger added
   following the project pattern (GPT-7).
8. **Manifest precision** (GPT-S5/S6, DS-6): `_beat_id_for_shot` for beat ids
   (synthetic -> b000_music_open, not shot_...); `char_id = shot or line`;
   `init_image = _portrait_index(led).get(char_id, "")`.
9. **Constant duplication over import** (GPT-S3): render_driver keeps a local
   `_OPENING_MUSIC_SUFFIX` constant with a comment tying it to ShotLock's
   OPENING_MUSIC_BEAT_ID (no node-module import side effects).
10. **beat_intent table lives in code with the loose fallback** (DS-S2 + pass02);
    byte-identical = the SAME fresh run's master (GPT-S9, the existing meaning).

## Standing rejections (unchanged from pass02)

Wan cap (operator-gated), Ken-Burns zoom on holds, LLM repair for attribution,
same-episode video-only char overrides, hard YAVG release gate.

## Campaign record

pass01 ~$0.029, pass02 ~$0.028, pass03 ~$0.026; total ~$0.083 of OpenRouter
credit. Panel: gpt-5.5-20260423 / gemini-3.1-pro-preview-20260219 /
deepseek-v4-pro-20260423 x3 passes; Fable = sole judge/synthesizer, all claims
grounded against render_driver.py, eng_ltx_video.py, otr_shot_lock.py,
otr_silent_composite.py, _otr_story_brief_helpers.py, the legacy plan file, and
live-ledger/history probes. Build spec = `pass02_plan.md` + this judgment's
folded items (no pass03_plan rewrite needed -- precision items only).
