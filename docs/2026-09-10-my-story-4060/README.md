# My Story / Jeffrey 4060 full-canonical failure report

**Status: technical all-stills route proven; end-to-end source qualification not achieved.**

This is the 4060 handoff for the six canonical one-act My Story legs recorded
in [the 4060 drill log](../4060_DRILL_LOG.md) as Steps 116--121. It is a
full-pipeline report: every successful leg used the shipped
`workflows/otr_canonical.json`, generated audio, stills, video, credits, and
both archive and OBS delivery. It is not a claim that the supplied story
survived those stages faithfully.

The supplied source was a personal story about Jeffrey: Los Angeles is a
favorite place, but he misses being a baby and his mother in the Bay Area; a
girlfriend is a distinct relationship rather than a replacement for his mother;
and he intends to keep enjoying life. Later legs made the direct visual facts
even more explicit: Jeffrey and his living mother are together at the same Los
Angeles dinner table and must not be staged as separated, waiting, absent, or
replaced.

## Scope and provenance

| Item | Value |
|---|---|
| Machine and campaign | Physical RTX 4060, MRKT all-stills campaign |
| Tested implementation | `0327850a2d819a7ebed0d3df8e0cda2e90616153` |
| Current code distinction | `f829c920` subsequently fixed invalid supplemental-music parent references. It does **not** fix the P0/P1/P2/visual/cleanup fidelity failures below. |
| Workflow and route | Shipped canonical workflow; `my_story` -> `z_image_turbo` -> `still_pan`; no Registry workflow and no LTX route |
| Act safety boundary | Exactly one selected act was verified immediately before every GUI Run. No run was promoted to multi-act. |
| Shared-source changes on 4060 | None. `nodes/` remains owned by the 5080 core window. |

The report intentionally identifies only durable artifact IDs and committed
evidence. It does not copy private runtime logs, machine paths, or credentials
into the repository.

## Executive result

The stills route itself is healthy: four legs completed and published full media
with materialized 1472x832 Z-Image stills, `still_pan` clips, audio identity,
duration, archive, and OBS checks. Two additional legs stopped correctly after
the bounded P1 structured-output ladder exhausted. Those are useful technical
receipts, but **zero of the six legs is an end-to-end source-qualified My Story
output**.

The blocker is semantic preservation, not renderer availability, VRAM, LTX, or
an authorization issue. Direct listener facts can be contradicted by P0,
replaced by P2, inverted in visual planning, or changed by cleanup while the
technical publication eligibility remains true.

## Six-leg evidence matrix

| Leg / drill step | What completed | Failure that prevents source qualification |
|---|---|---|
| 1 / Step 116 | 17 stills, 15 `still_pan` clips, 133.760-second final, delivery and OBS checks passed. | P0/P1 turned nostalgia about a living mother into death/absence and a girlfriend-as-replacement thread; cleanup also left one `unclean_spoken_text` row. |
| 2 / Step 117 | Correct fail-closed stop before media. | P0 simultaneously retained "alive" and assumed the mother was deceased or unavailable. P1 produced JSON-looking but undecodable treatment output on base, lower-temperature, and typed-repair attempts, then raised `StructuredCallFailedError`. |
| 3 / Step 118 | Correct fail-closed stop before media. | P0 retained face-to-face living-mother facts but invented music "between acts" for one act. P1 repeated the three-attempt malformed-output exhaustion. |
| 4 / Step 119 | 25 stills, 22 clips, 124.680-second OBS delivery, archive/audio identity checks passed. | P0 again invented family absence and a one-act interstitial. P1 only succeeded after two malformed attempts and typed repair. Technical eligibility was true while its fidelity discrepancy list was empty. |
| 5 / Step 120 | 20 stills, 17 clips, 123.120-second final, OBS/archive/audio checks passed. | Authored text retained the pair, but the visual plan staged the mother waiting alone and Jeffrey as not visible. Cleanup also replaced valid coda text, "Until next time," with "That's a wrap." |
| 6 / Step 121 | 15 stills, 13 clips, 92.840-second 1920x1080 archive and matching OBS final, audio/duration/delivery checks passed. | P0 repeated one-act interstitial language; after P1's two malformed attempts, schema-valid P2 replaced the shared dinner with an invented fire and said Jeffrey and his mother "sit alone." Cleanup made six edits after judging five rows dirty, yet publication eligibility had zero blockers. |

The final pair-lock artifact identifiers are
`signal_lost_los_angeles_dinner_pair_lock_visual_leg_20260910_193857` and
`los_angeles_dinner_pair_lock_visual_leg_20260910_193857__vstb__stpa__zimg__koko__unk__q354b__sa3_final.mp4`.
The archive is 62,431,761 bytes; the OBS final is 50,983,194 bytes. Both are
92.840 seconds. Their existence proves the all-stills technical path, not the
story’s fidelity.

## Open defects for the 5080 core window

The production records are `PBUG-20260910-03` (source facts),
`PBUG-20260910-04` (one-act interstitial leakage), and
`PBUG-20260910-05` (P1 schema binding). The Step 120 coda mutation is an
additional live instance of existing `PBUG-20260829-14`; it is not counted as a
fourth new defect.

### A1R-1 -- one-act control leaks into narrative instructions

At the tested and current source, P0 and P1 render `"music between acts: yes"`
solely from `include_act_breaks`, even when `act_count=1`
([`_otr_my_story.py`](../../nodes/_otr_my_story.py) P0/P1). P3 already computes
the real value as `max(0, len(acts) - 1)`.

**Required correction:** calculate the interstitial count once from selected
acts, use it in P0 and P1, and add a narrow P0 semantic check that rejects an
interstitial or between-acts assertion when that count is zero. This is a
configuration-fidelity check; it must not become a prose-quality gate.

**Required regression:** one act + breaks enabled has zero interstitial claims
and no interstitial cue; two acts + breaks enabled retains exactly one;
breaks-off remains zero.

### A1R-2 -- P1 structural output is unreliable despite an available local binding seam

Legs 2 and 3 exhausted all three P1 attempts. Legs 4--6 needed typed repair
after two malformed JSON-looking responses. The writer exposes a lazy
`_otr_bind_schema` seam for local transformers, but My Story’s P1 currently
passes the original creative callable directly to the structured call.

**Proposed narrow correction:** bind only P1's local callable once to
`StoryTreatment` when the seam is callable; retain the original callable for
P2/P3, all current attempt accounting, and the fail-closed three-attempt
ladder. Do not introduce generic regex or natural-language JSON repair.

**Required regression:** prove the binder is invoked exactly once for P1, P1
uses the bound callable, P2/P3 remain unbound, malformed output still fails
closed, and the current full-artifact typed-repair behavior remains intact.

### A1R-3 -- direct listener facts have no end-to-end semantic-preservation contract

Schema validity did not prevent the following direct contradictions:

- P0 calling an explicitly living, present mother deceased or unavailable;
- P2 replacing an explicitly shared dinner with a fire and solitude;
- visual planning staging an explicitly face-to-face pair as separated; and
- cleanup changing valid, source-compatible final copy without a durable
  source-fact comparison.

The current protected-fact flag shields specific Python-owned fact components;
it is not a My Story contract for listener-supplied facts. The existing My Story
fidelity list was empty in a leg whose accepted text was demonstrably wrong.

**Required correction:** compile explicit listener facts into a small,
structured, durable contract before P0. Carry it through P0, P1, P2, P3, visual
planning, cleanup, final TTS text, and publication eligibility. It should cover
only direct facts such as named people, stated relationship, living/present
status, required shared presence, setting, action, and ending. It must reject
or bounded-repair direct removal, replacement, contradiction, major invented
event, or inverse staging. It must not judge tone, creativity, subjective
quality, or lawful user language.

**Required regression:** use the actual living-mother, one-act pair-lock source
shape. Assert that P0/P1/P2/P3, accepted visual prompts, and final spoken text
retain both people together at the Los Angeles table; reject deceased/absent
mother, fire, solitude, waiting-alone, offscreen-arrival, and replacement
variants. Assert cleanup cannot mutate a fact-bearing row without a new
validated fidelity receipt, and require eligibility to consume that receipt.

## Observability follow-ups, not media failures

These signals were real diagnostics, but they do not negate the completed
stills renders and should not be filed as duplicate renderer failures.

| Signal | Observed disposition | Required handling |
|---|---|---|
| Pre-image `MISSING-STILL (LOUD)` | Logged before image generation; all later stills materialized and the render-time gate passed. | Say `CAST-TIME STILL DEFERRED` (or equivalent) before ImageGen; reserve loud absence for post-image/render-time failure. |
| `LTX-OPEN HEALTH` on `still_pan` | Five warnings on the intentional no-LTX leg. This is distinct from the repaired `ltx_8gb` allowlist defect. | Run the LTX-open health check only when manifest/policy says LTX was required. |
| Closing-tail floor fill | Permitted 24/25-frame safe fill, used instead of looping unproven footage. | Preserve as expected behavior. |
| `LOUD re-resolve` after episode rename | Reconciled the same master WAV; byte identity passed. | Emit informational `PATH RECONCILED` for same-file success; retain loud failure for rejected, unresolved, or non-identical paths. |

No new production-bug entry is needed for those warning-only classes. Related
historical records already cover actual LTX misclassification, output identity,
and credits-tail loss; this report preserves the narrower phase/route-intent
follow-up without falsely calling a healthy all-stills output an LTX failure.

Diagnostic maintenance follow-up: O1 and O3 are repaired at their shared owners;
pre-image still deferral and successful active-episode path reconciliation now
log INFO. Actual missing-still and identity/freeze refusals remain unchanged.
The code/tests and independent review are recorded in
`../2026-09-11-my-story-cross-machine/diagnostics_receipt.md`. This does not
retroactively requalify the six source-defective runs. O2 planned-versus-actual
LTX health classification remains in GO_FORWARD.

## Current-code boundary

The report’s six-leg evidence belongs to `0327850a`. The later `f829c920`
change sets My Story supplemental music `beat_id` to null while retaining cue
and line identity. That correctly addresses the earlier music-parent warnings,
and its recorded 1/3/6-act freeze evidence should remain credited to that
commit. It does not retroactively qualify these 4060 stories or prove that the
semantic source-fidelity defects are fixed.

## Exit criteria and next live proof

1. Land the three focused A1R core corrections above with focused, full, and
   applicable Bug Bible coverage.
2. Keep the current canonical workflow and one-act GUI boundary on MRKT.
3. Run one fresh canonical one-act all-stills pair-lock recovery leg on the
   corrected implementation. Require a durable source-fidelity receipt for
   P0/P1/P2/P3, visual plan, cleanup, final TTS text, and eligibility, in
   addition to normal ledger/audio/video/OBS checks.
4. Only then resume the separately planned 3/6-act stress coverage on an
   authorized host. A successful one-act recovery proves the repair route; it
   is not a substitute for the planned stress matrix.

The detailed timestamps, raw attempt receipts, and artifact-level verification
remain in [Steps 116--121](../4060_DRILL_LOG.md) for implementation review.
