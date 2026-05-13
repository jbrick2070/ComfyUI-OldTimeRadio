# v4 Composer Sprint — Round-Robin Review Handoff

**Date:** 2026-05-11
**Branch:** `v2.0-alpha`
**Range:** `3b74590..bb77062` (3 commits)
**Scope:** prompts, LineRequest fields, sampling knobs, optional polish pass.
**NOT in scope:** outline LLM, cast contract, reviewer, audio/video, ledger I/O.

This is a **review punch-list** — what I changed, where I cut corners, and what to challenge. Tests are green (310/5/0 + Bug Bible 16/7/3xf baseline), so do not re-litigate working pieces. Hunt the new failure modes only.

---

## 1. What's actually new (not previously soaked)

### 1.1 LineRequest grew 7 fields in one shot
Bundled v3 plan Commits 1+2+3+4 into one git commit (`869a5a7`). They share field surface so independent revert is impossible after this lands.

New fields on the frozen dataclass: `allowed_people`, `allowed_things`, `prev_speaker`, `current_beat_block`, `theme`, `all_voice_cards`, `sfx_cue`, `position`. All default empty; every prompt block gates on non-empty.

### 1.2 `_build_user_prompt` was completely rewritten
Static-prefix-first layout for future KV-cache reuse. Order is now:

```
STYLE / THEME / EPISODE CONTEXT / NAMED ENTITIES / CAST / OUTLINE
| CURRENT BEAT / POSITION / SOUND IN THE ROOM /
| LAST SPOKEN (this scene) / WRITE LINE
```

Outline render stays plain. Arrow + "← write this" anchoring was rejected in favor of a separate CURRENT BEAT block in the per-call tail, so the static prefix stays byte-stable across every call in an episode.

### 1.3 New helper `render_current_beat(outline, beat_id) -> str`
Returns one outline row as a `CURRENT BEAT\n  bNNN ...` block. Writer calls it per beat. Empty string when no match.

### 1.4 System prompt rewritten
Five-negative OUTPUT FORMAT block first, then flat CRAFT checklist. Ends with `"Output the single line and stop. Nothing before it, nothing after."` Role induction moved out of the system prompt into the WRITE LINE block at the tail.

### 1.5 Role induction in WRITE LINE
`"You are <SPEAKER>."` always, plus `"You are responding to <PREV_SPEAKER>."` when `prev_speaker` is non-empty and not equal to speaker (case-insensitive). Closes with `"Speak now."` verb-final.

### 1.6 Scene-local `last_lines.clear()` at music markers
Inside the `beat.speaker_role in NON_VOICED_ROLES` branch, an inner check for `{"music_open", "music_inter", "music_close"}` clears the window. The `sfx` role does NOT clear.

### 1.7 `_format_last_lines` placeholder reworded
From `"(no prior dialogue -- this is the first line of the episode)"` to `"(scene just opened - no one has spoken yet)"`. Reachable mid-episode after a music marker now, not just at line 1.

### 1.8 Sampling knobs (Commit 2 / hash `4d19549`)
- `_build_truncating_generate_fn` now closures `min_p` + `repetition_penalty`. Only forwarded to `model.generate` when non-default (0.0 / 1.0).
- Per-call `stop=` kwarg added. Implemented via inline `StoppingCriteria` subclass that decodes the last 64 generated tokens and substring-matches each step.
- Three new widgets appended at end of `INPUT_TYPES.optional`: `min_p` (default 0.0), `repetition_penalty` (default 1.0), `max_new_tokens_cap` (default 200).
- `compose_line` attempt-1 `max_new_tokens = min(cap, max(40, target_words * 4))`; attempt-2 uses full cap.

### 1.9 Optional polish pass (Commit 3 / hash `bb77062`)
- `_NARRATION_LEAK_REGEXES` — 5 patterns: `he/she/they (said|replied|added|asked|whispered|shouted|paused|continued|murmured|exclaimed)`, leading quote (smart or straight), `*asterisk action*`, `[bracket direction]`, `(cue verb: sigh|pause|beat|laughs?|smiles?|gestures?|nods?|shrugs?|cough)`.
- `needs_polish(line) -> bool` — cheap regex gate.
- `polish_line(generate_fn, leaked, voice_card)` — one targeted LLM call at temperature 0.4. Falls back to original on any failure.
- Fires inside `compose_line` after retry ladder closes, BEFORE the phantom-name gate runs. Only when `enable_polish_pass=True` AND `needs_polish(cleaned)` is True.
- New `enable_polish_pass` BOOLEAN widget, default False.

### 1.10 Workflow JSON
`workflows/otr_scifi_16gb_full.json` `widgets_values` extended 14 → 18 (`0.0`, `1.0`, `200`, `false` appended). Existing positions 0–13 unchanged.

### 1.11 Drift-guard slot correction
`tests/test_workflow_json_guardrails.py` had `_WRITER_STYLE_SLOT = 11` but style is at slot 9. Corrected to 9. Confirmed pre-existing failure via stash-and-rerun against pre-Commit-5 working tree.

### 1.12 Incidental: 3 humo-smoke workflow JSONs got `"groups": []`
Pre-existing schema-drift failures from `test_workflow_zod_shape` — unrelated to v4. Fixed in passing.

---

## 2. Pitfalls I'm explicitly handing you to look at

### 2.1 `prev_speaker` derivation lives in writer, not composer
At each LineRequest construction site:
```python
prev_speaker = last_lines[-1][0].strip() if last_lines else ""
```
`last_lines` is `list[tuple[str, str]]` of (speaker, text). The window can contain `ANNOUNCER` entries, so a character beat after an announcer line will get `prev_speaker="ANNOUNCER"` → `"You are ALICE. You are responding to ANNOUNCER."` — which is correct but unusual phrasing. Worth a sanity check on whether announcer-followed-by-character lines read naturally.

### 2.2 Self-talk dropdown
`_build_user_prompt` does:
```python
if req.prev_speaker and req.prev_speaker.strip().upper() != req.speaker.strip().upper():
    parts.append(f"You are {req.speaker}. You are responding to {req.prev_speaker}.")
else:
    parts.append(f"You are {req.speaker}.")
```
Two-line monologue (same speaker twice) drops the "responding to" clause cleanly. But this happens BEFORE the music-marker scene clear — if the previous beat was an announcer or music, `prev_speaker` carries through. Verify behaviour at scene boundaries is what you want.

### 2.3 `position` falls back to legacy ARC PHASE when empty
Back-compat path: if a caller passes `arc_phase` but not `position`, the legacy `ARC PHASE: <phase>\n  <guidance>` block still renders. In production writer always sets both, but a hand-built `LineRequest` (e.g. someone debugging from the REPL) may see both blocks if both are set — actually no, the `if/elif` prevents that. Confirm.

### 2.4 `_position_for(beat)` derivation in the writer
```python
arc_order = (
    list(episode_budget.arc_phases)
    if episode_budget is not None
    else list(dict.fromkeys(...))
)
```
`episode_budget` IS always in scope by the time the per-beat loop runs (it's built one block earlier at `compute_episode_budget`). So the outline-only fallback path is dead code today. Kept it because the v4 plan flagged a NameError risk, but the risk is theoretical. **Decide: trim the dead branch, or keep as defensive?**

### 2.5 `phase_beats.get(this_phase, [])` collision
If two beats have the same `arc_phase` value but the second beat's `beat_id` is not in the first's `ids` list (shouldn't happen, but…), `beat_n` falls back to 1. Silent. Worth a guardrail assertion?

### 2.6 `min_p` kwarg forwarding
```python
if active_min_p > 0.0:
    gen_kwargs["min_p"] = active_min_p
```
Transformers < 4.43 doesn't accept `min_p`. If Jeffrey's env has an older transformers, this will raise from inside `model.generate`. Verify the installed version. If unsafe, wrap in try/except and warn-and-skip per the LLM-agnostic policy.

### 2.7 Stop-string `StoppingCriteria` per-step decode cost
The subclass decodes the last 64 tokens **every generation step**. For a 200-token output that's 200 decode calls. On Mistral-Nemo at ~50 tok/s this adds maybe 1-2ms/step → 200-400ms total per line. Negligible at episode scale, but worth knowing. Also: tokenizer caching means the substring check is O(stop_count * decoded_length) per step. Five-pattern compose default is fine; longer custom stop lists would compound.

### 2.8 `polish_line` re-strips formatting but does NOT re-check `needs_polish`
After polish, if the polish output still trips the regex (rare but possible), we just commit it anyway. Decision was deliberate (polish output is rarely worse than original), but worth confirming with you.

### 2.9 Polish pass happens BEFORE the phantom-name gate
A polished line could introduce a new proper noun. The phantom-name flag runs AFTER, so this is fine — but if you later move the order, polish must stay first. There's no comment in the code asserting this dependency.

### 2.10 New `enable_polish_pass` widget at the end of INPUT_TYPES
Right now `widgets_values` order in the saved workflow is:
```
0..13   legacy positions (unchanged)
14      min_p              (Commit 2)
15      repetition_penalty (Commit 2)
16      max_new_tokens_cap (Commit 2)
17      enable_polish_pass (Commit 3)
```
ComfyUI binds by position. Any user opening their existing workflow gets 0.0/1.0/200/false in slots 14-17. Correct, but **anyone who saved a workflow between Commit 2 and Commit 3 has 17-entry `widgets_values` and the new 18th `enable_polish_pass` slot will be missing → ComfyUI fills with default False.** Today no one has done this; it's a future-state risk if the commits land separately on different machines.

### 2.11 `_DEFAULT_STOP_STRINGS` placement
I had to move it ABOVE `polish_line` because `polish_line` references it as a default argument. Default args evaluate at function-def time. AST parse caught this; no runtime damage. Worth a comment in case someone reshuffles.

### 2.12 `tests/test_arc_check.py` collection error
Pre-existing: imports `LLMScriptWriter` from `nodes.story_orchestrator` which no longer exists (extracted to `_otr_legacy_writer.py`). NOT mine, but I dodged it by excluding from the broad sweep. **Worth fixing in a follow-up commit** — this test has been silently uncollectable since the Phase 3 writer extraction.

### 2.13 Long writer file, dense diff
`nodes/OTR_LedgerScriptWriter.py` got +208 / -8 lines across the 3 commits. Most of it is per-beat field threading (LineRequest construction sites are now ~25 lines each). Hard to audit by skimming. Consider extracting LineRequest construction into a helper if it grows much more.

---

## 3. What I deliberately did NOT do

- No few-shot examples in the prompt — eats tokens, creates stale mimicry.
- No KV-cache reuse implementation — only the block-order precondition.
- No model-specific sampler defaults baked in — defaults preserve current behavior.
- No changes to phantom detection, cast locking, reviewer, Script Doctor, outline LLM.
- No always-on polish call.
- No removal of `ARC_PHASE_GUIDANCE` / `arc_phase` — kept the legacy path live for back-compat.

---

## 4. Concrete review prompts for ChatGPT / Gemini

Round-robin candidates:

1. **System prompt audit:** does the OUTPUT FORMAT block + "Output the single line and stop. Nothing before it, nothing after." actually close the narration-leak hole on Mistral-Nemo / Gemma-2-9b / Qwen2.5-7b/14b? Or does the polish pass become load-bearing in practice?

2. **Sampling defaults:** I left `min_p=0.0` and `repetition_penalty=1.0` as defaults (preserves current behavior). The v4 plan considered flipping to `0.05` / `1.03`. **Which side should production land on?** Argument for current: zero risk, opt-in tuning. Argument for new: documented dialogue-quality lift on every small-LLM I would test.

3. **Block-order soundness:** does `THEME` belong above `EPISODE CONTEXT` (current) or below it (alternative)? The theme is supposed to be tonal flavor; placing it AFTER setting could read more naturally. I picked above because it's a single sentence and reads as a header.

4. **Role induction line wording:** `"You are responding to BOB."` vs `"BOB just spoke. Respond:"` vs `"In reply to BOB:"`. Pick one to A/B against my choice.

5. **Polish prompt:** mine uses `"You are a script editor cleaning one line of radio drama dialogue."` Is "script editor" the right authority frame, or would `"You are the same character, restating this line in clean spoken form."` produce more in-voice cleanups?

---

## 5. Suggested QA recipe (manual)

1. Open `workflows/otr_scifi_16gb_full.json` in ComfyUI Desktop. Confirm the 4 new widgets show at the bottom of the writer node with defaults `0.0 / 1.0 / 200 / false`.
2. Queue a 100-word smoke run with defaults. Compare to a pre-869a5a7 baseline ledger you have on disk: line count, line lengths, named-entity flagging should all match the baseline.
3. Queue a second run with `min_p=0.05`, `repetition_penalty=1.03`. Eyeball the lines side-by-side for quality.
4. Queue a third run with `enable_polish_pass=true`. Check the run log for `polish_line firing on` entries — should be sparse (0-2 per 6-beat smoke). Compare polished line text to the pre-polish version in logs.
5. Try a custom_premise that names a non-cast person (e.g. "...Dr. Patel reports..."). Confirm `compose_flags` still stamps `phantom_name:Dr. Patel` AFTER polish (polish must not erase the phantom flag).

---

End of handoff.
