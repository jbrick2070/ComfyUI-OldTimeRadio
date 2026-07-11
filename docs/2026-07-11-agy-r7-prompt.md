# AGY R7 -- reconcile your table with reality, then plan the Sonnet fixes

REVIEWER ONLY. Do not edit source, do not git add/commit/push. Write to
`agy_review7.md` and stop. Read the real files. Label every claim CONFIRMED (you
opened it) or [ASSUMPTION]. Show your arithmetic.

REPO: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
BRANCH: v2.0-alpha  HEAD: 88888583 (moving -- pull before you read)

## Where the build actually is

- Codex: PUBLISHES a 30-word episode end to end. Verified asset in `otr\obs\`.
- Gemini: was dying at P3 for four straight rolls. Now the deterministic OutlineV4
  repair is ACCEPTED with NO LLM repair call ("[scifi_gemini:P3] deterministic
  OutlineV4 graph-metadata repair accepted"), P3 passes, and it reached P4 -- where it
  threw away a whole drafted scene over three extra `fact_ids` keys. That is fixed too
  (`repair_forbidden_extra_keys` now prunes forbidden extras for EVERY Gemini pass).
  A roll is in flight as you read this.
- Sonnet: has never completed. It is next.

## JOB 1 -- reconcile your truncation table with the evidence (epistemics)

Your R6 table says Codex P3/P3_rewrite and P5/P7/P9 are at "extreme silent truncation
risk, even at 30w, with no prompt_must_fit protection."

But Codex PUBLISHED at 30 words. And across many rolls, the ONLY `PROMPT_GUARD:
Truncated` line ever emitted was Gemini P3 (5408 -> 4592). If Codex's P5/P7 prompts
truly exceeded their input budget at 30w, that WARNING would have fired for them too --
it did not.

So one of these is true, and I want you to tell me which, with numbers:
(a) your prompt-size estimates for Codex are too high -- recompute them against the
    ACTUAL artifact_inputs each pass packs (read `_script_artifact_inputs`,
    `_script_artifact_context`, `_score_graph_contract`), not a guess;
(b) Codex IS truncating silently and publishing anyway (degraded but not fatal) -- in
    which case show me which pass, and what it is losing;
(c) something else.

Grep the actual server log for every `PROMPT_GUARD` line and use that as ground truth:
`C:\Users\jeffr\Documents\ComfyUI\comfyui_8000.log`. Evidence beats estimation. If your
R6 numbers were wrong, say so plainly -- I would rather have a reviewer who corrects
himself than one who defends a table.

Then: which passes ACTUALLY need `prompt_must_fit=True` and a scaled reservation, in
priority order, for the 720-word run? (At 720w the arithmetic changes, and that is the
run that matters next.)

## JOB 2 -- the Sonnet fix plan (this is the deliverable)

You confirmed three Sonnet bugs. Turn them into a plan I can execute. For each: the
exact patch, the test that proves it, and the MECHANICAL/CREATIVE call.

1. The P5 rewrite loop never writes corrected lines back into `events`, so the re-audit
   re-reads the same text. Show me the loop and the exact missing write-back. What is
   the correct merge -- by index, by beat_id, by line ref? What happens to a line the
   rewrite did NOT touch?
2. `AttestationV4.attestation_cites` allows 4 but `DraftLineV4.cites` allows 3, so a
   4-cite reply raises on construction. Which limit is the real contract? Do not just
   widen the smaller one -- tell me which number is CORRECT and why.
3. The lane hardcodes `fact_0` while the P0 dossier contract is 1-indexed
   (`fact_1`..N). Is the hardcode wrong, or is the contract wrong? Where does the
   announcer/warden line legitimately cite nothing at all -- and if a line cites no
   fact, what SHOULD it carry?

Then the parity question: Gemini now has a deterministic repair that (a) prunes
forbidden extra keys for every pass and (b) derives mechanical graph metadata. Sonnet
has neither -- only the P0 literal-span repair. Which Sonnet passes need which, and
what exactly is MECHANICAL in Sonnet's artifacts (ids, ordering, enums, fixed role
labels, extras) versus CREATIVE (anything authored)?

## JOB 3 -- blast radius of a change landing beside us

Another agent is concurrently restoring multi-clip video per beat and removing per-beat
word-count chasing (word count becomes a post-hoc statistic, never a trimmer). That
will touch the beat/line structure and the render plan.

Tell me what in the Gemini and Sonnet lanes DEPENDS on the beat structure and would
break or silently mis-size if the number or meaning of beats changes. Be specific:
- Gemini's `outline_output_token_budget(words, len(bands))` is sized off the advisory
  band count.
- Codex's `_script_output_token_budget(words, accepted_line_count)` is sized off the
  accepted line count.
- What else? Music-cue anchors, the advisory word plan, the ledger assembly, the
  freeze-cascade per-line invariants, the render plan.

## Output (agy_review7.md)

JOB 1 RECONCILIATION: PROMPT_GUARD evidence, corrected numbers, and an explicit
  retraction if your R6 table was wrong.
JOB 2 SONNET PLAN: three fixes, each with patch + test + MECHANICAL/CREATIVE, then the
  deterministic-repair parity list.
JOB 3 BLAST RADIUS: the dependency list.
CONFIDENCE on every claim.
