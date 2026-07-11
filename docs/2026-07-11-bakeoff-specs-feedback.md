# Bake-off spec feedback -- for the Codex final rewrite pass

**From the operator + Claude (Fable) grading review, 2026-07-11.**
Scope: docs/2026-07-11-scifi-{codex,gemini,sonnet}-engine-spec.md.
Rewrite technical content only; every entry's creative voice, pass
topology, and pack prompt wording stay UNTOUCHED. If a fix needs a
creative decision, put it in "Open questions for the operator".

## Global laws (apply to ALL three specs)

1. **LOOSEN WORD-COUNT STRICTNESS.** The operator is NOT chasing exact
   word counts -- creativity and flow win. Replace any exact per-beat or
   per-line integer word target with a BAND (suggested: per-scene target
   +/-25% with an absolute floor, total episode band 720 +/-15%). A roll
   must never die because a beat came in at 71 words instead of 77.
   Validators judge structure and legality, not arithmetic perfection.
2. **THE LEDGER MUST STAY INTACT -- SPOKEN TEXT IS SACRED GROUND:**
   - NO stage directions or action text in any line row. Nothing like
     "(sighs)", "[pause]", "*leans in*", "(static crackles)" may reach
     line.text -- TTS will SPEAK it. Actions/SFX belong in music cue
     descriptions or are cut. Every spec must state its parser strips or
     rejects decoration (markdown emphasis, brackets, parentheticals) on
     spoken rows.
   - NO ALL-CAPS words in spoken text (TTS voices spell them out or
     shout-garble). Proper case always; acronyms written as spoken
     ("NASA" -> "nasa"/"N A S A" per the lane's own rule, but pick one and
     validate it).
   - CHARACTERS LOCKED AND ANCHORED: cast is fixed at casting time; every
     line's char_id resolves to that cast; no mid-episode name drift, no
     new speakers appearing after cast lock, speaker labels normalized
     (one-word, no honorific variants like 'DR. HARRIS' vs 'Harris').
   - All five hierarchies completely filled; sequencer-legal roles only
     (character / announcer / music_open / music_close / music_inter);
     no empty text on non-skip rows; skip rows carry tts_skip_reason.
3. **Dedupe the TailFinalizer.** All three specs independently converged
   on the same additive tail protocol -- extract ONE shared definition
   (single doc section or shared plumbing note) and have each spec
   reference it, so three builders don't implement it three ways.
4. Keep everything additive, fail-loud, no python text surgery, SFW,
   UTF-8 no BOM, never "dummy".

## Per-spec must-fix

**scifi-codex (grade A-):** Your fixed P3 beat table (BT02=77, BT05=174,
exact integers, zero tolerance) is the single biggest re-roll generator in
any spec -- a small local LLM cannot hit exact word counts. Convert the
table to bands per Global Law 1; keep the beat STRUCTURE (that is the
creative spine, untouched). Keep the source-span fact index (quote must
equal exact payload slice) -- that is your best feature; just make its
failure mode a bounded reroll of the offending beat, not a dead run.

**scifi-gemini (grade B):** Purge the "archived v2" runnable-looking
skeleton (`.format()` rendering, 3-pitch schema) that contradicts your v3
corrections -- an implementer copying section 8 instead of 8A builds the
wrong engine. One version of truth per spec. Keep the largest-remainder
word-blueprint algorithm but express its outputs as band centers, not
mandates (Global Law 1).

**scifi-sonnet (grade A):** Strongest creative differentiator (the
on-air Continuity Archive audit) -- do not touch it. Two technical slims:
(a) nine tightly-worded seams is a big surface for small local models --
merge seams where two prompts differ only by framing, without changing
any prompt's voice; (b) the WardenChallenge/WardenSatisfied schema split
fixed a guaranteed-crash bug -- add an explicit regression note in the
spec so the builder writes a test for the clean-path schema first.

## Acceptance bar for the rewrite

Each spec re-stamps "CODE-READY v4" with a revision log. A builder must be
able to implement with zero creative decisions; a 720w episode must be
able to land with imperfect-but-in-band word counts; and nothing a TTS
voice would mispronounce, spell out, or perform-as-text may survive to
line.text.
