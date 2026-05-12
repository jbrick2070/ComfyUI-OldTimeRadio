# Line Composer — Edit Plan for Cowork (v4)

**Date:** 2026-05-11
**Target:** `ComfyUI-OldTimeRadio` branch `v2.0-alpha`
**Files in scope:** `nodes/_otr_line_composer.py`, `nodes/OTR_LedgerScriptWriter.py`, `nodes/_otr_model_loader.py`, `nodes/_otr_episode_budget.py`
**Out of scope:** reviewer, ledger I/O, audio/video nodes, schemas not listed below.
**Testing scope:** AST + existing Bug Bible regression + new unit tests for prompt-assembly. No live LLM runs.

**Budget:** composer hot-path stays under ~1000 tokens per call.

**v4 changes vs v3** (this revision, not yours-to-execute-anew — v4 is a drop-in replacement for v3):

1. New tightened system prompt — replaces v3 Commit 1a wording.
2. Block order rewritten for KV-cache reuse — static prefix first, per-beat tail last. Arrow extracted out of `render_outline_spine` into a separate `CURRENT BEAT` block so the outline render stays byte-stable across calls in one episode.
3. Role induction ("You are ALICE.") moved to the final WRITE block in the user prompt, not the system prompt. Small instruct-tuned models hold a per-call role better when it sits right above the generation target.
4. One new unused-input added: `prev_speaker` name. Makes "reaction" lines have a target. ~5 tokens; meaningful signal.
5. Sampling defaults proposed for the 7B–14B class with rationale. Still exposed as widgets (v3 Commit 5 unchanged in shape).
6. Two-pass polish answered: **do not run per-line by default.** Optional, gated by a cheap narration-leak regex check. Polish prompt included.

---

## Pre-flight

Same as v3. Tag baseline `pre-composer-rewrite-2026-05-11` before any edit.

---

## Commit 1 — System prompt, role induction in user prompt, block order for KV-cache reuse

**Files:** `nodes/_otr_line_composer.py` only. No schema change.

### 1a. Replace the system prompt

Locate `_SYSTEM_PROMPT` at `_otr_line_composer.py:562`. Replace verbatim with:

```
You write one spoken line for a character in a radio drama.

OUTPUT FORMAT — strict:
- Only the words the character speaks out loud.
- No character name, no colon, no quotation marks.
- No stage directions. No actions in parentheses or brackets.
- No "he said" / "she added" / narration of any kind.
- Output the single line and stop. Nothing before it, nothing after.

CRAFT:
- Imply more than you state. People rarely say what they mean.
- Push the scene forward by one small step.
- Follow naturally from the last thing said.
- Stay in the speaker's voice — their job, their pressure, their habits.
- Inhabit the mood without naming it.
- Use only proper nouns listed under NAMED ENTITIES. Generic roles
  ("the tech", "the lab", "mission control") are fine.

Short and charged beats long and explanatory. Within ±30% of the
requested word count.
```

**Rationale.** Small instruct-tuned models in the 7B–14B class drift to narration ("She paused, then said...") when the system prompt is mostly positive craft direction. Putting OUTPUT FORMAT first, with five concrete negatives, and ending with "Output the single line and stop" gives the model a hard close. Keeping the craft block as a flat list (not paragraphs) reads as a checklist, which small models follow more reliably. Total system prompt is ~155 tokens, comparable to v3.

### 1b. Add `allowed_people`, `allowed_things`, `prev_speaker` to `LineRequest`

`LineRequest` at `_otr_line_composer.py:343`, additive after `allowed_roster`:

```python
# Commit 1 (2026-05-11 v4): split roster + previous-speaker target.
allowed_people: frozenset[str] = field(default_factory=frozenset)
allowed_things: frozenset[str] = field(default_factory=frozenset)
prev_speaker: str = ""   # name of the character who spoke the last line
                         # in the rolling window; "" for scene-head or
                         # narrator-following lines.
```

**Rationale on `prev_speaker`.** "Follows naturally from the last thing said" works better when the model knows *who* said it. The recent-dialogue window does name speakers, but for the immediate previous line a small model benefits from an explicit target ("You are responding to BOB."). 5–8 tokens of cost for a measurable improvement in reaction-line coherence.

### 1c. Block order in `_build_user_prompt` — static prefix first

Restructure `_build_user_prompt` (starts line 580) to emit blocks in this order. Static-prefix blocks are byte-identical across every line call in an episode, which lets the next KV-cache reuse pass land without further restructuring.

```
[STATIC PREFIX — cacheable across an episode]
STYLE: <style>

THEME: <one-sentence theme>

EPISODE CONTEXT
<logline / hook / season setup, whatever is already passed>

NAMED ENTITIES IN THIS WORLD
  People: <sorted cast names>
  Places, agencies, things: <sorted key terms>
Generic roles ("the tech", "the lab", "mission control") are fine.
Do not invent any other proper name.

CAST
ALICE — female, weary forensic engineer in her 40s, dry humor
BOB — male, anxious junior tech, voice cracks under pressure
ANNOUNCER — omniscient narrator

OUTLINE
b001 ALICE (curious): hears unusual signal in lab
b002 BOB (worried): warns about source location
b003 ALICE (determined): decides to investigate
...

[PER-BEAT TAIL — changes every call]
CURRENT BEAT
b003 ALICE (determined): decides to investigate

POSITION: complication, beat 2 of 4. Next phase: climax.

SOUND IN THE ROOM: distant klaxon

LAST SPOKEN (this scene)
BOB: I'm telling you, that signal is not coming from inside the building.
ALICE: Then where.

WRITE LINE
You are ALICE. You are responding to BOB.
Mood: determined.
Beat: decides to investigate.
Word count target: 14.
Speak now.
```

**Rationale on block order.** The user is targeting a future KV-cache reuse pass; static-first ordering is how you make that pass cheap. Everything that does not change between calls within one episode (style, theme, cast, full outline, named entities) sits in the prefix and stays byte-stable. The per-beat tail is what the next prompt will diverge on. v3's Commit 1e put the arrow *inside* `render_outline_spine`, which breaks byte-stability of the outline across calls — v4 pulls the arrow out into the separate `CURRENT BEAT` block instead.

**Rationale on "WRITE LINE" role induction.** Putting "You are ALICE. You are responding to BOB. Speak now." immediately above the generation point is the single most effective small-model intervention I would make here. Role specification in the system prompt wears off across long contexts; role specification one block above the target holds. The directive sentence "Speak now." closes the prompt with a verb the model is meant to enact, not a noun it is meant to describe — small models pattern-match the difference.

### 1d. Render functions

`render_outline_spine` stays plain (no arrow, no `← write this`). Take the parameter `current_beat_id` out of v3's plan — it's not needed.

Add a new helper, `render_current_beat(outline, current_beat_id) -> str`, that returns the single matching outline row in the format above. Called from `_build_user_prompt` only when `current_beat_id` is non-empty.

`compose_line` takes `current_beat_id: str = ""` as a kwarg and forwards to `_build_user_prompt`. Same call-site updates in `OTR_LedgerScriptWriter.py` (~lines 1478, 1508) as v3.

### 1e. Drop the "This line accomplishes" label entirely

v3 renamed it to `Beat:`. v4 folds it into the `WRITE LINE` block as shown above (the line `Beat: decides to investigate.`). One label, one place, last block before generation.

### 1f. Populate `prev_speaker`

In `OTR_LedgerScriptWriter.run()`, at each `LineRequest(...)` construction site, derive from the rolling window:

```python
prev_speaker = ""
if last_lines:
    # last_lines entries are "NAME: text" strings per existing convention.
    head = last_lines[-1].split(":", 1)[0].strip()
    if head:
        prev_speaker = head
```

If the window is empty (right after a music marker — see Commit 3), `prev_speaker` stays `""` and the role-induction line drops the "You are responding to X." sentence.

### Verification

- `pytest tests/test_line_composer.py` passes.
- AST parse: `python -m py_compile nodes/_otr_line_composer.py nodes/OTR_LedgerScriptWriter.py`.
- Modules import cleanly: `python -c "from nodes import _otr_line_composer, OTR_LedgerScriptWriter"`.
- `grep -rn 'ALLOWED NAMES\|do not invent any name outside\|This line accomplishes' tests/` — update any hit to the new shape.
- Unit test: render a full prompt with all blocks populated; assert block order matches the spec above; assert no `None` substitutions; assert no doubled blank lines.
- Unit test: `prev_speaker=""` → role-induction line is just `You are ALICE.` (no "responding to" clause). `prev_speaker="BOB"` → `You are ALICE. You are responding to BOB.`
- Back-compat test: legacy `LineRequest` with only `allowed_roster` populated and new fields all default — prompt renders without NAMED ENTITIES, CAST, THEME, POSITION, SOUND IN THE ROOM blocks (they all gate on non-empty).

### Commit message

```
composer: rewrite system prompt, role-induct in user prompt, KV-friendly block order

- Flip system prompt to OUTPUT FORMAT first (5 concrete negatives) +
  CRAFT block as flat list. Small 7B-14B instruct models drift to
  narration when only told "spoken text only" once; a strict format
  block with explicit "Output the single line and stop." holds.
- Block order: static prefix (style/theme/context/entities/cast/
  outline) then per-beat tail (current beat/position/sfx/last
  spoken/write line). Outline rendered plain — arrow moved to a
  separate CURRENT BEAT block so the outline stays byte-stable across
  every line call in an episode (KV-cache reuse precondition).
- Role induction (You are X. You are responding to Y.) moved out of
  the system prompt into the final WRITE LINE block immediately above
  the generation target. Small models hold a role better when the
  directive sits one block above the response slot, not 800 tokens
  upstream.
- Additive LineRequest fields: allowed_people, allowed_things,
  prev_speaker. allowed_roster unchanged for the phantom gate.

No breaking change. Token budget +25 over v3 baseline.
```

---

## Commit 2 — Thread `sfx_cue`, full cast voice cards, theme into `LineRequest`

**Files:** `nodes/_otr_line_composer.py`, `nodes/OTR_LedgerScriptWriter.py`.

Functionally same as v3 Commit 2. Three additive `LineRequest` fields:

```python
sfx_cue: str = ""
all_voice_cards: str = ""
theme: str = ""
```

Rendering follows the block order from Commit 1c. THEME emits before EPISODE CONTEXT. CAST replaces the single-speaker CHARACTER line, with fallback to the speaker-only card when `all_voice_cards == ""`. SOUND IN THE ROOM emits after LAST SPOKEN, before WRITE LINE.

Population path (reuse `voice_card_by_name` from Phase 1 setup, robust first-sentence theme via `re.split(r"(?<=[.!?])\s+", brief, maxsplit=1)`) — unchanged from v3 Commit 2c.

**Rationale on dropping per-line CHARACTER, keeping full CAST.** Lines about or to other characters are the failure mode v3 already identified. Full-cast cards in the static prefix cost ~110 tokens once per episode (cacheable on a future KV pass), versus the same cost per call now. Worth it. The model needs to know who BOB is before ALICE can react to BOB credibly.

**Rationale on theme as one sentence, not multi-sentence.** Outline absorbs structure. A one-sentence theme carries tonal resonance the outline drops. Multi-sentence theme creeps into prompt budget without adding signal — the model needs a flavor, not a brief.

### Verification

Same as v3 plus:
- Unit test for theme regex: `"Dr. Smith faces a crisis. The signal is fading."` → `"Dr. Smith faces a crisis."` not `"Dr"`.
- Unit test: CAST block present in prompt when `all_voice_cards` set; falls back to speaker-only card otherwise.

### Commit message

```
composer: thread sfx_cue, full cast voice cards, theme into LineRequest

LineRequest was missing three signals already available in the writer:

- Beat.sfx_cue — gives the line awareness of the sound environment so
  it can react.
- Other characters' voice cards — joined from voice_card_by_name built
  in Phase 1 setup. Lines about or to other characters were flat
  because the model only saw the speaker's card.
- One-sentence theme from meta.news.script_brief — outline absorbs
  structure; theme carries resonance. Robust first-sentence extract
  handles abbreviations like "Dr. Smith ...".

Additive LineRequest fields with empty defaults; no breaking change.
Static-prefix placement (per Commit 1c block order) keeps the per-call
cost flat on future KV-cache reuse.
```

---

## Commit 3 — Scene-local `last_lines` window

**Files:** `nodes/OTR_LedgerScriptWriter.py`, `nodes/_otr_line_composer.py`.

Same as v3 Commit 3 with one wording tweak in the prompt: when `last_lines` is empty, emit `(scene just opened — no one has spoken yet)` instead of `(silence — scene just opened)`. The longer phrasing reads cleaner to small models and avoids them treating the literal word "silence" as a cue to write a silent beat.

`last_lines.clear()` at music markers. Header: `LAST SPOKEN (this scene)`. Verification unchanged from v3.

### Commit message

```
composer: clear last_lines window at music markers (scene-local context)

Crossing a music_open/music_inter/music_close boundary left the next
2-3 lines reading the prior scene as recent context — wrong signal.
Clear on music markers; re-label "LAST SPOKEN (this scene)"; emit
"(scene just opened — no one has spoken yet)" when empty so the model
knows it's writing the first spoken line of a scene.
```

---

## Commit 4 — POSITION line replaces generic arc-phase guidance

**Files:** `nodes/_otr_line_composer.py`, `nodes/_otr_episode_budget.py`, `nodes/OTR_LedgerScriptWriter.py`.

Same as v3 Commit 4. `position` field on `LineRequest`, format:
- `<phase>, beat N of M. Next phase: <next>.`
- `<phase>, beat N of M. Final phase.` for the last phase.

Renders in per-beat tail per the Commit 1c block order, between CURRENT BEAT and SOUND IN THE ROOM.

`ARC_PHASE_GUIDANCE` deprecated in place with the note from v3 Commit 4d. Same one-release-cycle policy.

**Rationale unchanged from v3.** Generic per-phase guidance steered every beat in a phase the same way. Position-aware framing lets the model treat beat 1 of 4 differently from beat 4 of 4.

### Commit message

```
composer: replace generic ARC_PHASE_GUIDANCE with position-specific POSITION line

Generic per-phase guidance ("Escalate or introduce conflict") steered
every beat in a phase the same way. Beat 1 of 4 in complication and
beat 4 of 4 in complication should write different lines. Compute
"<phase>, beat N of M. Next phase: <next>." per beat and emit as
POSITION in the per-beat tail of the prompt.

arc_order falls back to outline beat order when EpisodeBudget is not
in scope (guarded by `"budget" in locals()`).

ARC_PHASE_GUIDANCE deprecated in place (one release cycle).
```

---

## Commit 5 — Sampling params, `max_new_tokens` cap, stop strings

**Files:** `nodes/_otr_model_loader.py`, `nodes/_otr_line_composer.py`, `nodes/OTR_LedgerScriptWriter.py`.

Same structure as v3 Commit 5 — all params exposed as widgets, LLM-agnostic. v4 proposes specific defaults below with rationale; Jeffrey overrides per-model from the workflow widget.

### 5a. Proposed defaults for the 7B–14B class

| Param | Proposed default | Useful range | Rationale |
|---|---|---|---|
| `temperature` | 0.7 (attempt 1) / 0.85 (attempt 2) | 0.5–1.0 | Current value preserved. 0.7 is the sweet spot across Mistral-Nemo, Gemma-2, Qwen2.5 for in-character dialogue — lower flattens voice, higher invites narration drift. |
| `top_p` | 0.9 | 0.85–0.95 | Lowered from 0.92 by 0.02 to pair cleanly with `min_p=0.05` (when both are active, the tail cut is the union, so a slightly tighter `top_p` keeps the effective distribution stable). |
| `min_p` | 0.05 | 0.02–0.10 | New. Single biggest small-model dialogue improvement available right now. Cuts the long tail of low-probability tokens that produce the "wait, what?" off-key word in an otherwise good line. 0.05 is conservative; aggressive setting is 0.1. |
| `repetition_penalty` | 1.03 | 1.0–1.08 | New. Small models loop on character names and high-frequency words within short outputs. 1.03 is gentle enough to not damage natural repetition of the speaker's verbal tics. >1.08 commonly produces stilted output on short generations. |
| `max_new_tokens_cap` | 200 (widget) | 40–400 | Per v3. Attempt 1 = `min(cap, target_words * 4)`. Attempt 2 = full cap. |
| `stop` | `["\n\n", "\n[", "\n("]` | — | Per v3. Kills bracketed/parenthesized leak on a new line before `strip_line_formatting` has to clean. |

**Why expose, not lock in.** Per the LLM-agnostic constraint memory: control-plane is strictly agnostic; prose-plane (line composer) tolerates period-specialist as opt-in. Defaults are starting points; Jeffrey tunes per loaded model from the widget. Defaults should preserve current behavior unless the new behavior is strictly better — `temperature` and `top_p` stay near current values. `min_p` and `repetition_penalty` default to "disabled" in v3; v4 proposes flipping the default to `0.05 / 1.03` because both are conservative on the safe side and produce measurable dialogue quality gains on every small model I would test against (Mistral-Nemo, Gemma-2-9b, Qwen2.5-7b/14b).

**If you want the strictest LLM-agnostic stance**, keep v3's defaults (`min_p=0.0`, `repetition_penalty=1.0`) and let Jeffrey opt in per-model. The widget exposes the knobs either way. I would still ship `min_p=0.05` as a default — it is the safest non-trivial improvement available to small models.

### Verification

Unchanged from v3. Add one unit test: `make_generate_fn` called with all v4 default kwargs explicit → underlying generate call receives all four sampling params (mock the backend).

### Commit message

```
composer: expose sampling knobs (min_p, repetition_penalty, max_new_tokens_cap) + stop strings

LLM-agnostic — small local LLMs respond differently to sampler tuning.
Lock in nothing structural; expose everything as widgets.

Proposed defaults for the 7B-14B class:
- temperature 0.7 / 0.85 retry  (unchanged)
- top_p 0.9                     (was 0.92; pairs with min_p)
- min_p 0.05                    (new; cuts long tail)
- repetition_penalty 1.03       (new; gentle on short outputs)
- max_new_tokens_cap 200 widget (default preserves current behavior)
- stop ["\n\n", "\n[", "\n("]   (kills bracketed leak on a new line)

Defaults are starting points; per-model tuning from the writer widget.

Attempt-1 max_new_tokens scales with target_words (min of cap and
target_words * 4); attempt 2 uses full cap. Graceful degradation: if
a model family doesn't accept a given param, warn-and-skip not raise.
```

---

## Commit 6 — Optional polish pass, regex-gated and OFF by default

**Files:** `nodes/_otr_line_composer.py`, `nodes/OTR_LedgerScriptWriter.py`.

**Recommendation: ship this commit but default the widget to OFF.** Per the budget constraint — doubling LLM calls adds 3–5 min per episode without KV reuse — a full per-line polish pass is not worth it. A gated polish pass, run only on lines that fail a cheap regex check, costs ~1–2 extra LLM calls per episode on average and catches the failure modes that the targeted Script Doctor at the end currently catches a half-hour later. Worth shipping.

### 6a. Narration-leak regex check

```python
# Commit 6 (2026-05-11 v4): cheap narration-leak detector.
# These are the patterns small instruct models leak into dialogue
# output despite the system prompt's OUTPUT FORMAT rules.
_NARRATION_LEAK_PATTERNS = (
    r"\b(he|she|they)\s+(said|replied|added|asked|whispered|shouted|paused)\b",
    r'^["“‘]',                # opens with a quote mark
    r"\*[^*]+\*",                       # markdown asterisk action
    r"\[[^\]]+\]",                      # bracket stage direction
    r"\([^)]*(?:sigh|pause|beat|laughs|smiles|gestures)[^)]*\)",
)

def needs_polish(line: str) -> bool:
    return any(re.search(p, line, re.IGNORECASE) for p in _NARRATION_LEAK_PATTERNS)
```

### 6b. Polish prompt

```
You are a script editor cleaning one line of radio drama dialogue.
The line below leaked narration or stage direction. Rewrite it as
pure spoken dialogue.

OUTPUT RULES — strict:
- Only the words the character speaks out loud.
- No name, no colon, no quotes, no brackets, no parentheses.
- No "he said" / "she replied" / narration of any kind.
- Preserve the character's intent. Preserve the speaker's voice.
- Keep within ±20% of the original word count.

CHARACTER: <speaker_voice_card>
ORIGINAL LINE: <leaked_line>

Output the cleaned line and stop. Nothing else.
```

Pass through `make_generate_fn` with `temperature=0.4` (much lower than composer — this is a targeted edit, not a creative call), `top_p=0.9`, `min_p=0.05`, `repetition_penalty=1.0`, `max_new_tokens = target_words * 3`.

### 6c. Workflow widget

Add to `OTR_LedgerScriptWriter.INPUT_TYPES`:

```
enable_polish_pass: BOOLEAN, default False
```

When `True`, `compose_line` calls `needs_polish(generated)` after the existing retry ladder closes; if it returns `True`, run one polish LLM call with the prompt above. Polished line replaces the original. If polish itself fails the regex check, keep the polish output (it is rarely worse than the original).

**Rationale on default OFF.** You said you would accept doubling the cost only if quality gain is meaningful. Gated polish at ~10% of lines per episode is +1–2 calls (~30 sec) not +15 calls (~3–5 min). That is in the "worth it" zone. But it is opt-in because the targeted Script Doctor at the end of the writer already covers these cases at zero per-line cost — polish here is a "fix it before it lands in the ledger" preference, not a correctness requirement.

### Verification

- `pytest tests/test_line_composer.py` passes with `enable_polish_pass=False` (default — back-compat).
- Unit test: `needs_polish("Then we should go now.")` → `False`.
- Unit test: `needs_polish('"Then we should go," she said.')` → `True`.
- Unit test: `needs_polish("*sighs* Then we should go.")` → `True`.
- Unit test: composer with `enable_polish_pass=True`, mock backend returning a leaked line on attempt 1, confirm polish call is issued and the cleaned response is used.

### Commit message

```
composer: optional polish pass (regex-gated, default OFF)

After the retry ladder closes, optionally check the generated line
against a small narration-leak regex (he said / *asterisk action* /
[bracket direction] / opens-with-quote-mark / parenthesized cue
verbs). If the line trips the regex, issue ONE polish LLM call with
a targeted cleanup prompt and replace the line.

Default OFF — keeps the composer hot-path at 1 call per voiced beat.
Opt-in via enable_polish_pass widget on OTR_LedgerScriptWriter.
Expected cost when on: +1-2 calls per 15-line episode (~30s), not
+15 calls (~3-5 min).

Polish prompt uses temperature 0.4 (lower than composer — this is a
targeted edit, not a creative call). Backstops the Script Doctor's
end-of-episode pass for users who want clean lines in the ledger as
they're written.
```

---

## Post-flight (hand-off)

After all 6 commits land:

1. `pytest tests/` — full suite passes.
2. `git log --oneline -6` — six clean commits in order with messages above.
3. AST parse on every changed file: `python -m py_compile <files>`.
4. Push `v2.0-alpha` per CLAUDE.md cmd-shell flow.
5. Runtime testing, line-quality scoring, per-model param tuning, polish-pass opt-in evaluation — Jeffrey's scope.

If any regression suite fails, stop at that commit and ask. Do not fix forward past a failing test.

---

## Answers to the five asks, recap

1. **System prompt tightened.** OUTPUT FORMAT block first with five concrete negatives. Ends with "Output the single line and stop." See Commit 1a.

2. **Block order optimized for KV-cache reuse.** Static prefix (style → theme → episode context → named entities → cast → outline) followed by per-beat tail (current beat → position → sfx → last spoken → write line). Outline render stays byte-stable; arrow moved to separate CURRENT BEAT block. See Commit 1c.

3. **Unused inputs:**
   - **Add:** `sfx_cue`, full `all_voice_cards`, one-sentence `theme`, `position` (all from v3), `prev_speaker` (new in v4).
   - **Drop:** ARC_PHASE_GUIDANCE generic string (replaced by POSITION).
   - **Did not add:** few-shot examples (eats tokens, creates stale mimicry), multi-sentence theme (outline covers structure), full meta dump.

4. **Sampling defaults** for 7B–14B class: `temperature=0.7/0.85 retry`, `top_p=0.9`, `min_p=0.05`, `repetition_penalty=1.03`, `stop=["\n\n", "\n[", "\n("]`, `max_new_tokens_cap=200` widget. All exposed as widgets; defaults overridable per-model. See Commit 5.

5. **Two-pass option.** Recommended OFF by default. Polish pass is regex-gated — fires only on lines that leak narration. Expected cost +30 sec per episode when on, not +3–5 min. Polish prompt in Commit 6b.

---

## What this plan deliberately does NOT do

- No few-shot examples in the prompt.
- No outline-spine trimming.
- No model-specific sampler values baked in (defaults overridable).
- No changes to phantom detection, cast locking, reviewer, or Script Doctor.
- No new always-on LLM calls. Polish pass is opt-in and gated.
- No KV-cache implementation in this plan — only the block-order precondition.

End of plan.
