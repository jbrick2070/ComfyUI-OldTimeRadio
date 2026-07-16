# Rip the LLM veto gates from original_codex56sol -- is this winnable?

**Date:** 2026-07-13
**Base:** `0c30e143` (v2.0-alpha, HEAD == origin, suite 7878 green)
**Question for the panel:** is the plan below SUFFICIENT to get a green canonical run,
what breaks, and what am I missing? Do not just validate it -- try to kill it.

## Context

`original_codex56sol` is a 9-pass structured-LLM story lane. Five live production bugs
today (PBUG-20260713-10..14, see docs/PROD_BUG_LOG.md), all one shape: **an LLM audit
pass allowed to veto the episode, with no repair route, no stated contract, or grading a
property its artifact cannot express.** Three of the five were caused by my own fixes for
the previous two.

Operator ruling: **only deterministic validators may fail an episode. Rip the LLM
opinions -- if they cannot fail anything and cannot fix anything, they are pure cost.**

Key realization that makes "advisory" insufficient: demoting an audit to warning-only does
NOT remove the kill switch. The pass still makes its LLM call, still runs a post_validator
and a 3-rung repair ladder, and **ladder exhaustion still raises
`OriginalCodex56SolPassError` and kills the episode.** Only removal removes the risk.

## The four LLM verdict gates

| Pass | Artifact | Verdict power | Already proven in Python? |
|---|---|---|---|
| P2 triage | SlateTriage | selects a possibility card; `_validate_triage` proves the id resolves | selection is needed; findings are opinion |
| P4 fair-play | FairPlayReport | corroborated block -> truth-map retake -> fail closed | clue coverage per thread, one resolution link per thread, interpretation balance, lost-object/draw equality: ALL proven by `_validate_truth_map` + `_build_grounding_contract` |
| P7 blind listener | BlindListenerReport | remaining block -> script retake (P8) -> fail closed | **NOT proven anywhere.** This is the only test that the mystery is SOLVABLE from pre-reveal clues -- the show's premise |
| P9 final audit | FinalContractAudit | blocking script finding -> retake -> fail closed | `_assert_script_valid` runs one line later and proves graph, text, safety, grounding |

## The plan as handed to the coder

1. **RIP P9** entirely (schemas, `_audit_blocks`, `_audit_advisories`,
   `_validate_audit_envelope`, P9/P9_rerun/P9_retake call sites, disposition, repair-rule
   branches, seam `codex56_final_contract_audit`).
2. **RIP P4** entirely (schemas, `_corroborated_fair_blocks`, `_fair_play_advisories`,
   `_validate_fair_play_envelope`, `_truth_item_exists`, P4/P4_rerun/P3_rerun call sites,
   disposition, repair-rule branches, seams `codex56_fair_play_audit` and
   `codex56_truth_map_retake`).
3. **P2**: keep the selection, nothing may raise on the findings.
4. **P7**: left as "operator decision pending".
5. Wiring: removing a seam requires removing it from ALL THREE surfaces -- pack
   `prompt_stages`, `pipelines.json` `acoustic_puzzle_v1.declared_seams`, and that
   pipeline's pass `seam_refs`. `_otr_story_routing` enforces three-way parity at registry
   load; miss one and the lane dies on import.

## The problem I now see with my own plan

**The most recent live failure was P7, not P4 or P9.** Prompt `f2b9e40a` cleared the P4
retake, cleared P5/P6/P7/P8, and died at:

```
blind-listener rerun still could not infer the declared mundane cause
```

So ripping P4 + P9 while leaving P7 does **not** produce a green run. The plan as served is
necessary but not sufficient.

**P7's implementation also looks broken on its own terms.** `_listener_blocks` synthesizes a
blocking "Cause grounding" finding when the listener's `understood_cause` does not contain a
>=4-character token from `grounding.device_anchor`. But the grounding contract only requires
`device_anchor` to be spoken **on the reveal line** -- and the blind listener is shown only
the lines BEFORE the reveal (`_preceding_lines`). So the gate demands the listener echo a
word the script is not required to have shown it. That is the same "grades what the artifact
cannot express" defect as PBUG-20260713-14.

## What I want from the panel

1. Is ripping P4 + P9 safe, or do those passes carry a load-bearing invariant I have
   missed? Name the exact invariant and where it would be lost.
2. **P7: rip, or fix?** If fixed, what is the correct gate? Options I see:
   (a) require the pre-reveal clue lines to plant the device, and check the listener's
       `understood_cause` against the CLUE anchors rather than the device anchor;
   (b) drop the Python token heuristic entirely and let the LLM's own `blocking` findings
       stand (they are line-grounded);
   (c) rip P7 and accept that fair-play solvability goes unenforced.
   Which, and why?
3. After the rip, what deterministic validators remain, and are they sufficient to
   guarantee a VALID LEDGER (the operator's stated bar)? Name any gap.
4. Will this actually turn the lane green at 30 and 120 words, or is there a further
   blocker downstream that I have not hit yet because the run kept dying earlier?
5. What is the cheapest sequence to a green run?

## Constraints

- Python may never author story content. Models own prose, causality, clue placement.
- Deterministic validators are the authority and stay fatal.
- No shims. Root-cause only.
- The canonical workflow `workflows/otr_canonical.json` must not change.
