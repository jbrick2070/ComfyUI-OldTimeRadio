# Per-beat LTX motion_clause -- design spec v2 (roundtable-hardened 2026-06-16)

## Goal
Give every LTX i2v beat *subtle, controlled motion that preserves that beat's character*,
by generating a short per-shot motion clause (driven by the beat's dialogue + meta brief)
with the writer LLM and persisting it in the ledger. Production form of the
2026-06-16 motion finding (`docs/2026-06-16-ltx-variety/MOTION_FINDING.md`).

## Why (grounded)
- Empirically: cfg/sampler/strength are NOT the character-safe motion lever. The lever is
  a prompt that names the actual subject and asks for SMALL motion matched to the line.
- Current code: the LTX prompt's motion half is a STATIC per-role lookup
  `_LTX_MOTION_PROMPT_BY_ROLE` (render_driver.py:481, used :972), blended with
  `get_story_brief_ltx(meta)` + budgeted by `finish_visual_prompt` (:993-1015). Generic,
  identical across beats of the same role.

## Design -- two layers

### Layer 1 -- engine motion defaults (opt-in; flag OFF == byte-identical to today)
- sampler: **euler** hard-coded for i2v (NEVER euler_ancestral -- it morphs the face).
  No `OTR_LTX_SAMPLER` knob in v1 (a knob would let the invariant be broken).
- i2v strength: `OTR_LTX_I2V_STRENGTH` (default = current 0.75; character-safe band 0.6-0.75).
- fps precedence: `OTR_LTX_FRAME_RATE` if set+valid, else `ledger['video']['fps']` (25).
- negative: ONE global anti-deform suffix ("deformed face, different person, extra limbs,
  disfigured, bad anatomy"), appended ONLY when the motion flag is ON (so default output
  is unchanged). Not per-beat (the failure is model-level).

### Layer 2 -- per-beat motion_clause (the real lever)

**Ledger schema (nested under the shot):**
```
ledger['video']['shots'][i]['motion_clause'] = {
    "text":        "<=70-char present-tense clause naming the subject>",
    "model":       "<llm id, or 'static-role-map' on fallback>",
    "fallback":    false,
    "schema_version": 1,
    "source_hash": "<sha256 of canonical(char_id, beat_id, norm_dialogue, schema_version)>"
}
```
`source_hash` makes re-renders deterministic AND dialogue-aware: if the line changes the
hash changes and the clause is regenerated; otherwise the stored clause is reused.
(Dropped the old `action_summary` -- it was never sourced.)

**Generation phase = a SEPARATE post-brief BATCH pass (NOT ShotLock).**
Runs after the brief + lines exist, before render; ONE batched LLM call for all beats
(avoids N sequential calls; matches the OpenRouter/Ollama writer slot). Writes the ledger
once. For EVERY shot it writes a full `motion_clause` object -- including fallbacks -- so
render stays read-only.
- Inputs per shot (grounded paths):
  - subject display name <- cast/character entry via `shot['char_id']`
    (render_driver.py:669; NOT the brief prose, which has no name field);
  - the beat's **dialogue line** <- `ledger['lines']` joined by
    `_line_index(ledger).get(_beat_id_for_shot(shot))` (render_driver.py:425,665);
  - scene context <- `get_story_brief_ltx(meta)` (context only).
- The clause reflects the DELIVERY of the line: livelier gesture for an exclaimed/urgent
  line, a small lean/nod for a quiet one. Dialogue is the signal for "how much" motion.
- Output: ONE present-tense clause, **<= 70 chars** (validated in this phase, BEFORE
  finish_visual_prompt -- which budgets the whole prompt, not the clause).
- HARD constraints, enforced by a strict post-gen **parser/validator** (few-shot to guide;
  GBNF cut for v1 -- the validator is the real gate and works on every lane):
  - allowed motion phrases only (substring match): talks, speaks, gestures slightly,
    leans, nods, glances, tilts head, breathes, shifts, blinks, small hand movement;
  - BANNED: stands up, walks, runs, jumps, turns around, enters, exits, dramatic,
    full-body, camera cut;
  - MUST contain the subject display name (never generic "a person");
  - no scene re-description (that comes from get_story_brief_ltx).
  - reject -> per-shot fallback (static role text, fallback=true).

**Hook point (render_driver.py:972, READ-ONLY):** prefer `shot['motion_clause']['text']`
when present+valid; else use `_LTX_MOTION_PROMPT_BY_ROLE` (legacy/old ledgers). Keep the
EXACT existing `get_story_brief_ltx` blend + `finish_visual_prompt(...)` call params; only
the motion-half string is substituted. Add a `validate_motion_clause()` guard -- never
trust ledger text blindly (non-dict, missing/blank/over-budget/banned -> fallback).

## Fail-closed / invariants
- Any failure (LLM/API error, parse fail, validation fail, save fail) -> static fallback;
  NEVER drop a beat or abort (render_driver "never abort" contract). If the ledger SAVE
  fails, render legacy and do NOT claim persistence.
- Opt-in; default OFF -> prompt text byte-identical (golden-ledger test enforces this).
- Deterministic: generated once, stored, reused unless source_hash changes.
- Per-episode disposition log: generated=N reused=N fallback=N invalid=N.

## v1 non-goals / cut
Aggressive/large motion; t2v; two-stage refine; multi-keyframe (LTXVAddGuideMulti); STG
(installed, deferred -- separate coherence layer); GBNF; per-beat negatives; OTR_LTX_SAMPLER.

## Verify-at-build
- Exact dialogue text key on the `ledger['lines']` dict (present; confirm name).
- Cast/character table lookup from `char_id` -> display name.
- Writer-slot budget for one batched all-beats call.
- `_LTX_MOTION_PROMPT_BY_ROLE` contains only motion phrases (no framing) so substitution
  drops nothing.
