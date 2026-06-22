# R2 ANCHOR (Claude, code-grounded) -- coding plan / implementability

VERDICT: yes-with-fixes. The R1 targets are right, but the coding plan must be
re-pointed: 2 of R1's headline items (decouple prose/metadata; add a reroll
correction-instruction) are ALREADY IMPLEMENTED (see grounding_r2.md). The buildable
work is the 5 real gaps; ordered smallest-blast-radius first.

MUST-FIX (coding, in implementation order):

1. [gap 2 -- SMALL, do first] Stop the whack-a-mole. In `_otr_reroll.py` after a
   targeted patch, do NOT hand the critic the whole episode again. Re-judge only the
   patched `line_id`s + their immediate neighbours (continuity window), and add a
   loop invariant: the flagged-target COUNT must strictly decrease each cycle or the
   loop bails to repair-then-ship. Concrete: pass a `scope=line_ids` arg into the
   critic call inside the reroll loop; keep the whole-episode pass only for the
   initial score. Contained change; high leverage (directly fixes cycle1=3->cycle2=3).

2. [gap 4 -- SMALL] Voice fail-closed in `cast_lock.py`. (a) Guarantee `cast_seed` is
   persisted into `meta.cast_contract` at cast time so replay always runs (kill the
   `cast_seed is None -> return` silent path for production). (b) After replay, assert
   every `character`/`announcer` row has a non-empty `voice_preset`; if a `char_id`
   wasn't in the `voices` dict, assign the deterministic picker's fallback rather than
   leaving None. Add a test that no row ships with `voice_preset=None`.

3. [gap 5 -- TRACE then guard] role_mismatch. The engine-name-in-role WRITE is NOT in
   cast_lock (it only reads `speaker_role or role`). R3 must trace the upstream node
   that stamps the role/expected field; the fix is a guard at that source that maps/
   rejects TTS-engine strings so only `allowed_roster` values reach the role field.
   For R2: spec the guard + the test ("no `tts_model` value ever appears in a role
   field"); the exact node lands in R3.

4. [gap 1 -- LARGER] Scene-aware composition. `compose_line()` sees only one beat +
   last N lines. Give it the scene's arc: pass the phase trajectory + the escalation
   target (what state this beat must leave the scene in) + the slot's position in the
   arc, so the line is written toward the scene's rising action, not in a vacuum.
   Cheapest viable version: extend the existing `LineRequest`/prompt with an
   `arc_context` block (prev-beat outcome, this-beat target delta, next-beat setup) --
   no new pass, no new model. Bigger version (defer): a scene-level draft-then-split.
   Start with the prompt-context version; measure flat-rate before adding a pass.

5. [gap 3] Operational "flat" rubric. `FlatLine.reason` is a free LLM string. Tighten
   the CRITIC PROMPT with the explicit test (a line is flat unless it changes
   knowledge / shifts pressure / moves a relationship / forces-or-avoids a decision /
   raises-or-clears an obstacle), AND make the critic's `hint` name which of those the
   line must add. This aligns critic + composer on one target. Prompt change, not an
   algorithm.

SHOULD-FIX:
- Per-`speaker_role` fulfilment rules so announcer/music/sfx aren't held to the
  dialogue-pressure test.
- Per-character dialogue voice bible injected into compose + reroll (distinct from the
  portrait `character_description`).

SEQUENCING: gaps 2 + 4 are small, contained, test-backed -- ship first and re-soak to
measure. Gap 1 (scene context) is the big quality lever but needs measurement. Gap 5
(flat rubric) pairs with gap 1. Gap 3/role_mismatch trace is R3.

[verify in R3] the upstream role-field writer (gap 5); whether `compose_line`'s
`LineRequest` already carries any arc fields I can reuse for gap 1.
