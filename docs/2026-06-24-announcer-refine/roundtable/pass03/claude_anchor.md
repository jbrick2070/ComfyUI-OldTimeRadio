# R3 CLAUDE ANCHOR -- wiring / integration / sequencing (code-grounded)

VERDICT: yes-with-fixes. The flag plumb + refine loop are already centralized, so
the wiring is mostly clean; three integration points must be wired explicitly.

## MUST-FIX BEFORE BUILD
1. [§1/§3 flag] **All new gates hang on the SINGLE existing flag, not a new env.**
   GROUNDED: `_apply_story_scaffold_env` (:1551) runs once at run() top (:2402)
   -> `OTR_ENABLE_STYLE_GRAMMAR`; `_style_grammar_on` (:3216) is the in-run gate;
   "off" is the kill-switch (byte-identical). The contract build + intro
   SafeOpenBrief path + outro news-coda path + KILL-4 map MUST all gate on
   `_style_grammar_on`/`story_scaffold`. Introducing any new env splits the
   auto/on/off control and breaks byte-identity-off. (Concrete: reuse the same
   `if _style_grammar_on:` guard the existing style-grammar block uses.)
2. [§3 reroll] **Thread the new line-level register through the reroll rebuild.**
   GROUNDED: `build_reroll_line_request` rebuilds LineRequest from META ONLY
   (:4080, the documented "otherwise loses all of it" -- how KILL 1 threaded
   `grounded_nouns`). The KILL-2 compact register tag added to LineRequest must be
   stamped on meta and re-applied in `build_reroll_line_request`; otherwise a
   guarded reroll (freeze cascade / KILL-1 body gate) drops the register on the
   re-composed line. `conflict_object` is already covered by KILL 1; the register
   is the NEW field that needs the same treatment.
3. [§1 refine] **Build the contract on EVERY run() pass; do NOT guard it behind
   `not _refine_active`.** GROUNDED: `_refine_loop` (:2137) re-invokes
   `run(_refine_active=True, _refine_forced_cast_seed=seed)` sharing ONE cast_seed
   (:2190); pitch_room IS skipped under `_refine_active` (:3053). The StoryContract
   is deterministic from cast_seed, and each pass regenerates the outline (which
   needs the contract fields), so the contract build must run every pass (place it
   with the existing `_style_grammar_on` block, which is already outside the
   refine-skip guards). Determinism holds: same cast_seed -> same slug/lead-in.

## SHOULD-FIX
1. [§3 outro] **Handle `_climax_beat_id == ""` at the post-loop outro.** It is
   computed pre-loop (:3271) only when `_style_grammar_on` and can be "" (zero
   character beats / no climax-class beat resolved). The climax-line lookup must
   fall back to the existing last-character scan when `_climax_beat_id` is "" (the
   plan says so; make it explicit at the call site).
2. [§2 open] **`opening_status_quo` is per-pass under refine.** The setup beat
   intent is re-derived each refine pass (the outline regenerates). That is
   correct (the open tracks the current story) and deterministic per
   cast_seed+pass; just confirm the "first character beat" selection is stable
   (first `speaker_role=="character"` in `outline.beats` order).
3. [§7 flag] **Confirm the off-path is untouched end-to-end.** With
   `story_scaffold=off`, `_apply_story_scaffold_env` forces the kill-switch ->
   `_style_grammar_on=False` -> none of the new branches execute -> the intro/outro
   run their current code verbatim. Add the off-flag golden tests at the run()
   level (not just unit) to prove the integrated path is byte-identical.

## OPTIONAL / NICE-TO-HAVE
- Single source for the coda_lead_in + contract slug: compute both in the same
  `_style_grammar_on` block where `_style_slug` is already derived (:3218), so the
  outro post-loop just reads locals -- avoids recomputation drift.

## CUT THESE
- None new. (Build-chunk split from R2 stands.)

## ASSUMPTIONS
- [ASSUMPTION] `build_reroll_line_request` is the ONLY path that rebuilds a
  LineRequest for a reroll (no second rebuild site that would also need the
  register). verify by grepping its callers.
- [ASSUMPTION] the post-loop outro block has `_climax_beat_id`, `cast_seed`, and
  the contract slug in scope (same run() frame as :3216-:3271). Confirm no early
  return between.
