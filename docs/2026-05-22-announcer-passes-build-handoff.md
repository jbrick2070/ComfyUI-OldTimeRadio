# OTR -- Next Session Handoff: Announcer Dedicated-Pass Build

- **Date:** 2026-05-22
- **Branch:** v2.0-alpha
- **Status:** design LOCKED + code-grounded; BUILD NOT STARTED
- **Do this in a fresh conversation with clean context.** The design is final -- just execute the spec below.

---

## What shipped earlier this session

1. **`9b49598` (committed + pushed)** -- BUG-LOCAL-233 b003: `_VOCATIVE_MID_RE` widened to accept `;` / `:` as mid-sentence closing delimiters (`r",\s*announcer\s*([,;:])"`, `subn(r"\1 ", ...)`); +3 tests in `tests/test_vocative_drift.py` (20 total). Full `tests/` walk 2461 passed / 22 skipped / 0 failed.
2. **BUG-LOCAL-233 soak** -- episode `signal_lost_biolab_breakthrough_20260522_073101` (commit 9b49598): zero "ANNOUNCER" in any character line. Passed. Caveat: that episode induced no vocative drift, so it confirms clean output + no regression, not a live semicolon-drift catch. A future episode that drifts would be the belt-and-suspenders proof.
3. **BUG-LOCAL-255 logged** -- `override_announcer_close` matches the private `_speaker_role` key (stripped at `set_lines`); `news_close_brief` is silently dropped every episode. Fix folded into this build (see below).
4. **Round-robin consult** -- `docs/2026-05-22-announcer-dedicated-passes__0{1,2,3,4}.md` + `transcript.json`.
5. **`scripts/_consult_round_robin.py`** -- `gemini-3.5-flash` added to the Gemini ladder (rung 3).

Items 3-5 plus this handoff are committed together as the "groundwork" commit that precedes this build.

---

## The task

The announcer's intro (first beat, b001) and outro (last beat) currently route through the shared `compose_line` -- the same prompt path as character dialogue, with no framing-specific prompt. The outro is then *supposed* to be overwritten with the news interpreter's `news_close_brief` by `override_announcer_close`, but that override is broken (BUG-255), so the outro is just generic composer text.

Promote the intro and outro into their own dedicated `creative`-slot LLM passes.

---

## Locked design (round-robin synthesis + Claude's own corrections)

- **Two separate passes**, not one combined bookend: intro runs in-loop on the first announcer beat; outro runs post-loop.
- **Outro context = `script_brief` + `news_close_brief` + `intro_text` only.** NOT the full script. (Reason: the announcer close is a thematic/journalistic wrap, not a line-by-line recap -- a tight prompt yields a tight close. The round-robin's "full script OOMs the KV cache" argument is wrong-scale -- OTR episodes are ~5 lines / ~150 tokens -- but the conclusion stands for the simpler reason.)
- **Plain-text LLM output, NOT JSON.** Match `compose_line`'s pattern: return text, `strip_line_formatting`, validate. JSON wrapping only adds a broken-JSON failure mode for a one-line output.
- **Skip the generic `compose_line` call for the final announcer beat** -- drop in the deterministic outro fallback as the in-loop placeholder; the post-loop outro pass overwrites it. Total announcer LLM calls stay at 2 (1 intro + 1 outro), now purpose-built.
- **Retire `override_announcer_close`.** The post-loop outro pass writes directly to the known final-announcer beat row, sidestepping the broken `_speaker_role` row-match entirely -- this closes BUG-255. (No alias-guessing helper needed; BUG-255 is a row-selection bug, not a brief-key problem.)
- **Deterministic fallbacks** for both passes -- string-template the brief into a SIGNAL LOST frame. The narrative bookend must never be missing.
- **Only the FIRST and LAST announcer beats get dedicated passes.** If act-breaks ever insert mid-episode announcer beats, those keep using `compose_line` -- they are not the bookend frame.
- Polish (`polish_line`) lives inside `compose_line`; the dedicated passes bypass `compose_line`, so they are never re-polished -- correct by construction, no special-casing needed.

---

## Implementation spec

### `nodes/_otr_line_composer.py` -- add beside `compose_line`

- `clean_one_line(text, max_chars)` -- collapse whitespace, strip wrapping quotes, hard length cap on a word boundary.
- `validate_announcer_line(text, *, min_chars, max_chars)` -- reject empty, multi-line, speaker-label prefixes (`ANNOUNCER:` etc.), bracket/brace stage directions, out-of-band length. Returns `(ok, cleaned)`.
- `fallback_announcer_intro(script_brief)` / `fallback_announcer_outro(news_close_brief)` -- deterministic SIGNAL LOST frame templates from the brief.
- `compose_announcer_intro(*, creative_fn, script_brief, creative_repo_id=None)` and `compose_announcer_outro(*, creative_fn, script_brief, news_close_brief, intro_text, creative_repo_id=None)`:
  - Build `messages = [{"role":"system",...},{"role":"user",...}]` with an announcer-framing system prompt (intro: orient the listener from `script_brief`; outro: close in announcer voice from `news_close_brief`, lightly echo `intro_text`).
  - Call `creative_fn(messages, temperature=, max_new_tokens=, stop=list(_DEFAULT_STOP_STRINGS))` with the `TypeError` fallback to the no-`stop=` form -- exact convention `compose_line` uses.
  - `strip_line_formatting(raw or "")`, then `validate_announcer_line`. On failure -> the deterministic fallback.
  - Return a `LineResult` (`text=`, `compose_flags=` e.g. `("announcer_intro",)` or `("announcer_intro_fallback",)`).
  - Tag each call site `# LLM slot: creative`.

### `nodes/OTR_LedgerScriptWriter.py` -- per-beat loop (announcer branch ~L2330) + post-loop overlay (~L2419-2460)

- Before the loop: `announcer_ids = [b.beat_id for b in outline.beats if b.speaker_role == "announcer"]`; `first_announcer_id = announcer_ids[0]`, `last_announcer_id = announcer_ids[-1]` (guard the empty / first==last cases).
- In the `elif beat.speaker_role == "announcer"` branch:
  - `beat.beat_id == first_announcer_id` -> `compose_announcer_intro(creative_fn=creative_generate_fn, script_brief=<from meta["news"]>, creative_repo_id=resolved["creative_writing_model"])` inside `with slot_scheduler.helper_context("compose_announcer_intro"):`.
  - `beat.beat_id == last_announcer_id` -> no LLM call; `cleaned = _OTRLC.fallback_announcer_outro(nc_brief)` as placeholder.
  - any other announcer beat -> unchanged `compose_line`.
- Post-loop: replace the `if nc_brief: override_announcer_close(...)` block with: read the first announcer line's text from `led.data["lines"]` as `intro_text`; call `compose_announcer_outro(...)` inside `helper_context("compose_announcer_outro")`; write the result to `last_announcer_id` via `patch_line_text(led.data, last_announcer_id, outro_text)` + `led.save()`.
- Keep the `post_assembly_keyterm_check` block that follows it unchanged.

### `nodes/_otr_news_wiring.py`

- Retire `override_announcer_close` (delete it; it has no other callers -- grep to confirm). Keep `post_assembly_keyterm_check`.

### Tests -- new `tests/test_announcer_passes.py`

- `clean_one_line` / `validate_announcer_line` units (label-prefix, bracket, length-band, multi-line rejects).
- `fallback_announcer_intro` / `fallback_announcer_outro` determinism.
- `compose_announcer_intro` / `compose_announcer_outro` with a mock `creative_fn` -- happy path returns validated text; LLM-failure path returns the deterministic fallback.
- A writer-level test (or extend an existing writer test) asserting the first announcer line came from the intro pass and the last from the outro pass / fallback, and that `news_close_brief` now reaches the closing line (BUG-255 regression).

---

## Verify at build time (do NOT trust line numbers blind)

- `LineResult` definition + `compose_flags` type (tuple) -- near `LineRequest` in `_otr_line_composer.py`.
- `strip_line_formatting` signature/location.
- Module constants: `_DEFAULT_STOP_STRINGS` (= `("\n\n", "\n[", "\n(")`, confirmed), `_BASE_TEMPERATURE`, `_MAX_NEW_TOKENS_PER_LINE`, `_SYSTEM_PROMPT`.
- The exact name/scope of `script_brief` in the writer loop (source is `meta["news"]["script_brief"]`).
- `_otr_outline.py` `_synthesize_outline` -- confirm first + last beats are the only hardcoded announcer beats; check whether act-breaks insert mid ones.
- `led.update_line_text` / `_OTRL.patch_line_text` / `_OTRL.patch_line_fields` -- which is correct for the post-loop outro overwrite. The old override path used `patch_line_text(led.data, line_id, text)`; mirror it. `line_id == beat_id` in the ledger.
- Grep `override_announcer_close` repo-wide before deleting -- confirm the writer is the only caller; update/retire its tests in `_otr_news_wiring`-related test files.

---

## Constraints (CLAUDE.md)

- LLM-call tagging: both new passes are `# LLM slot: creative`, fed `creative_generate_fn` / `resolved["creative_writing_model"]`. NO new `model_id` widget (rule 6). Update the Two-Model Selector routing table + wiring test pin.
- Audio is king: ledger row SHAPE must not change -- only how announcer `text` is generated. `tests/v2/test_audio_byte_identical.py` must stay green (the audio-path code is untouched).
- Run the Bug Bible regression + `test_core.py` + `test_dropdown_guardrails.py` + the full `tests/` walk after the change. Baseline before this build: 2461 passed / 22 skipped / 0 failed (will be higher with the new test file).
- No node-surface change is expected (no new widget, no new node class), so no workflow JSON rewire -- but verify.
- Commit via Desktop Commander cmd (`-F .git\COMMIT_EDITMSG`); the sandbox git view is unreliable in Cowork-on-Windows.

---

## Paste-ready prompt for the fresh conversation

```
OTR (ComfyUI-OldTimeRadio / SIGNAL LOST), branch v2.0-alpha.
First moves: read CLAUDE.md, BUG_LOG.md header, then
docs/2026-05-22-announcer-passes-build-handoff.md in full.

Build the announcer dedicated-pass feature per that handoff's
implementation spec: compose_announcer_intro + compose_announcer_outro
in nodes/_otr_line_composer.py, wire them into OTR_LedgerScriptWriter's
per-beat loop + post-loop section, retire the broken
override_announcer_close (closes BUG-LOCAL-255), add
tests/test_announcer_passes.py, run the full tests/ walk + Bug Bible,
commit + push to v2.0-alpha.

The design is LOCKED -- do not re-litigate it. Verify the line
numbers / signatures listed in the handoff's "Verify at build time"
section against the real code before editing.

Environment notes: work against the real disk via Desktop Commander
(the Cowork sandbox mount can show stale/corrupted views; if it does,
quit Claude Desktop, delete %APPDATA%\Claude\vm_bundles, relaunch).
Git via Desktop Commander cmd, never PowerShell.
```

---

*Design history: `docs/2026-05-22-announcer-dedicated-passes__04_synthesis.md` (round-robin + Claude synthesis). Bug: BUG-LOCAL-255 in BUG_LOG.md.*
