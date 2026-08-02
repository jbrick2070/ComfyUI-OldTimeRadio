# r2 judgment -- every cast member gets a voice (driver grounding)

Panel: codex `gpt-5.6-sol` (high). Operator ruling served: "every cast member
needs a voice -- it's a radio drama, not a mime show. Either have an LLM write
its lines, or entirely remove the character from the ledger." Plus: "read /
kibitz before doing any coding updates."

## THE PANEL CORRECTED MY LANE MODEL, AND IT WAS RIGHT

My input doc claimed `scifi_news_pro` is a gate-less lane. VERIFIED WRONG:

    nodes/story_packs/banks.json:76   scifi_news_pro -> scifi_news_pro_multipass
    nodes/_otr_lane_specs.py:99-101   scifi_news_pro_multipass -> _otr_scifi_fable2
                                                        run_scifi_fable2_episode
    nodes/story_packs/banks.json:169  scifi_news     -> scifi_news_circuit
    nodes/_otr_lane_specs.py:110-112  scifi_news_circuit -> _otr_scifi_codex

So `scifi_news_pro` runs the FABLE2 runner, which HAS the coverage gate. The
lane that deliberately permits a silent cast member is `scifi_news` -> CODEX,
and it says so in its own comment (`_otr_scifi_codex.py:948-960`):

    # coverage is a COUNT field and must not gate production. A short/reconciled
    # draft may not give every planned cast member a beat; that is ADVISORY
    # (recorded), not fatal ... an uncovered cast member simply carries no lines.

That is the exact design the operator's ruling overturns.

## THE ACTUAL MECHANISM (grounded past both the panel and my own first read)

The failing ledger's c03 rows are not merely empty -- they are SKIPPED:

    shot_001_b2  text=''  speaker_role=character  skip=True
    shot_001_b4  text=''  speaker_role=character  skip=True   (x5 total)

And `build_receipt` selects rows with the SAME predicate the validator uses --
both call `_voiced_rows` (`_otr_content_authorship.py:66` and `:138`). There is
no minting/validation asymmetry.

Therefore: **at proof-minting time those rows WERE voiced, and a later pass
emptied them and set `skip=True`.** The coverage check is correct; it is
reporting a real mutation. This is the "proof of nothing" hazard already noted
in PBUG-20260802-02, now with the mechanism nailed down rather than guessed.

Note the policy in the same ledger is `content_owned_readonly`, whose whole
premise is that content does not change after the cascade entry -- so a pass
emptying rows under it is either a policy violation or a legitimate cleanup that
the proof was minted too early to survive.

## ACCEPTED -- the panel's best structural idea

**A1. Enforce BEFORE `_assemble_ledger`, and the dangerous half of the ruling
disappears.** (codex CUT 1.) My input doc warned that "entirely remove" is
load-bearing and risky -- a char_id has references in line/beat/shot rows, a
portrait, a voice assignment, captions, credits, proofs, and a half-removed
character is a worse ledger than a silent one. codex's answer dissolves that:
enforce or reroll at the PRE-LEDGER candidate stage, and no portrait, caption,
credit, voice, beat, shot or authorship artifact exists to clean up. So
"remove the character" = reroll the cast candidate, NEVER post-ledger surgery.
ACCEPTED. This is the single most valuable thing the panel produced.

**A2. No machine-versus-human classification.** (codex CUT 2.) The schemas carry
free-text `role` / `register` / `role_in_conflict`, not an entity-kind flag, so
a "is it a machine?" heuristic would be another ambiguous policy. And "The
Relay" CAN be voiced. This matches the operator's ruling exactly -- every cast
member gets a voice; the answer is not to detect machines. ACCEPTED.

**A3. There is NO single call site both lanes pass through.** (codex MUST-FIX 3.)
My input doc demanded one, which is not implementable: the writer dispatches
each runner separately (`OTR_LedgerScriptWriter.py:3647-3688`) and both assemble
before returning. Correct shape is ONE PURE VALIDATOR called at two
producer-specific boundaries -- fable2 after parse/before `build_final_draft`,
codex during P3 validation before P5 and `_assemble_ledger`. ACCEPTED.

**A4. "Has a line" needs a real predicate.** Non-empty canonical text, `skip ==
false`, correct spoken role -- and ANNOUNCER matched by its sentinel rather than
raw char_id (`production_ledger.py:114-139`). Also: `CastPlanV4` requires 2-4
rows including ANNOUNCER (`_otr_scifi_codex.py:263-289`), so dropping the sole
non-announcer would violate the schema and forces a full cast regeneration.
ACCEPTED -- this is why "reroll" is the correct removal mechanism.

**A5. Fable2 has no tail finalizer.** Codex has `_CodexTailFinalizer` checking
accepted text before the final save; `Fable2TailParts` has none, so a pre-proof
gate alone does not protect fable2 against a later tail mutation -- which is
precisely the mutation observed. ACCEPTED as the second half of the fix.

**A6. Retry transport must be typed.** `Fable2ScriptError` exposes a string
reason and attempt count, not structured defects; do not parse exception text.
And the failure is in P3, not P0/P5 as my doc said. ACCEPTED, my error.

## VERIFY-AT-BUILD

The live ledgers' `meta.source_bank = scifi_news_pro` sits oddly beside a
freeze-policy note saying that pack "declares NO line_composer_system seam". The
bank->lane mapping above is HEAD and verified; whether the artifact was produced
by a different registry generation is not established. Attach the producing
commit SHA and `story_pipeline` to the artifact before treating its lane label
as authoritative.

## SCOPE

No new ComfyUI node, input, widget or link. This is an internal producer-contract
change in two story producers plus one shared pure validator, so
`workflows/otr_canonical.json` stays byte-identical. Bug Bible promotion waits
for the canonical live verify, per the standing admission rule.
