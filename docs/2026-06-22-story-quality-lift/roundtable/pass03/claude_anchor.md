ROUND 3 -- CLAUDE ANCHOR REVIEW (wiring / integration / sequencing; grounded vs real code)

VERDICT: yes-with-fixes. One MAJOR wiring fact forces a scope change (DEFECT 2 auto-repair is
unbuildable as wired -> pivot to generation+detection); the rest are interface/channel corrections.

MUST-FIX BEFORE BUILD:

1. [pass02 sec 4 DEFECT 2 repair] VERIFIED UNBUILDABLE. `needs_full_rerun` is a terminal output string;
   the writer is UPSTREAM of the cascade and ignores it; a rerun calls `new_ledger()` which WIPES meta;
   `meta["regeneration_hint"]` is read by nobody; a rerun is a manual operator re-queue into a fresh
   pending dir. With JSON frozen there is NO channel to carry a coherence hint into the regenerated
   episode. FIX: CUT the auto-repair (and the "coherence_hints re-injected into the new ledger"
   mechanism -- it has no re-injection path). DEFECT 2 v1 becomes:
   (a) GENERATION lever (the ROOT fix, parallel to DEFECT 1 Tier 1): strengthen
       `_otr_outline._build_beat_user_prompt` (1166-1236) to require the antagonist's stance toward the
       protagonist/central object be CONSISTENT across beats -- a reversal needs an explicit turn beat.
       JSON-free, no cross-run state, fixes the arc at generation.
   (b) DETECTION: the critic stance axis + LOUD + telemetry (measurement/backstop). No rerun.

2. [pass02 sec 3 + sec 2 audit channel] VERIFIED: line rows have NO per-line `meta` dict (fixed fields;
   the free-form per-line channel is `compose_flags`, a list of "kind:detail"). FIX: the DEFECT-3
   role-coercion breadcrumb rides per-line `compose_flags` (append "role_coerced:announcer->character")
   + an episode-level `meta["role_coercion"]` list; the DEFECT-1 per-line hygiene breadcrumb likewise
   rides `compose_flags`. Drop every "per-line meta" reference from pass02.

3. [pass02 sec 3 DEFECT 3 sites] VERIFIED complete write-point set (W3). The coercion CATCH-ALL = a
   PRE-FREEZE consistency sweep over the whole ledger (it has `cast`), so we do NOT instrument all 9
   builders. The real culprit is the role_mismatch repair (`_otr_ledger_reviewer.py:1063`); guard it
   (reject expected="announcer" when the row has a cast char_id). CAUTION: `cast_lock.py:473`
   LEGITIMATELY stamps announcer when char_id IS the announcer -- so the helper's cast-id set MUST
   EXCLUDE the "announcer" sentinel + music/sfx roles, or it will fight a correct re-stamp.

4. [pass02 sec 2 Tier 1] VERIFIED: the "speak first person, never narrate your own actions in third
   person" rider ALREADY EXISTS at `_otr_line_composer.py:1307-1315` and the corpus still leaks. FIX:
   present Tier 1 as a STRENGTHENING of that existing rider (necessary, not sufficient); Tiers 2 (reroll)
   and 3 (deterministic floor) remain load-bearing. Do not imply Tier 1 alone closes DEFECT 1.

SHOULD-FIX:

1. [pass02 sec 4 detection] FailedDimension is a Literal (W4); adding "stance" is runtime-safe (lone
   consumer `_otr_reroll.py:591-596` prefixes it), but land the Literal value + the critic SYSTEM-PROMPT
   prose (`_otr_story_critic.py:310-329`, so the model emits it) + the `StanceIssue` model + tests in ONE
   chunk, or the model never produces the new dimension.

2. [sequencing] DEFECT 2(a) (beat-intent prompt) + DEFECT 1 Tier 1 (line prompt) are independent files;
   land the DEFECT-2 generation lever WITH the detection axis so the no-bypass re-smoke can show the
   stance flag firing on a pre-lever episode and confirm the lever reduces it.

3. [pass02 sec 2 Tier 3] `_strip_stage_directions` is gated by `is_spoken_role` (character + announcer)
   -- announcer lines are scrubbed too (correct; b018-class). Keep the idempotence + well-formedness
   asserts; add an announcer-line negative fixture.

CUT THESE (over-engineering):
1. [pass02 sec 4] Auto-repair via needs_full_rerun + the coherence-hint-in-meta re-injection -- cut
   entirely (MUST-FIX 1: no surviving channel; manual re-queue only).

[No material ASSUMPTIONS -- W1..W5 verified against the real source.]
