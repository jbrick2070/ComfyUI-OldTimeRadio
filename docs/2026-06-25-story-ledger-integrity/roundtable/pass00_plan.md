# STORY-LEDGER INTEGRITY -- problem statement for the roundtable

Operator ask: a full panel on bug-fixes for a BETTER story ledger with NO DRIFT --
emphasising multiple LLMs on the BINARY decisions AND a WHOLE-STORY accuracy
review. "Anything's game." Find where the ledger silently stops faithfully
representing a coherent story, end to end (writer -> freeze -> critic -> canon ->
render), and the minimal durable fixes.

## Already SETTLED today (2026-06-25) -- do NOT re-propose

- **Binary dialogue/stage-direction lane: DROPPED via G1** (638 ledgers / 5,513
  lines; deterministic detectors leave ~0 genuine residual; an LLM binary lane
  would wrongly strip in-character commands). `docs/.../binary/G1_RESULTS.md`.
- **Per-line leak accuracy: converged as `leak-floor-v2`** (narrow structural
  rules + news-bleed fixed at `build_allowed_roster`; LLM cleaner cut).
  `docs/2026-06-25-leaking-words/pass02_plan.md` (build-ready, not yet built).
- **Schema-call tolerance / model-agnostic parsing: SHIPPED** (Lever 1;
  `apply_field_aliases` + tolerant validate). The structured-call layer is robust.
So the panel should NOT re-litigate the binary leak lane, the per-line leak gates,
or structured-call tolerance. Go WIDER: whole-story accuracy + cross-stage drift.

## The grounded surface (what exists + where it can silently fail)

1. **Whole-story critic (`_otr_story_critic.py`).** Checks §1 continuity (factual/
   prop/timeline breaks), §2 voice_drift (per-character), §4 arc_verdict
   (strong/uneven/flat/mid_collapse). BUT: on any failure it returns an
   all-empty report with `arc_verdict="strong"` (lines ~36/580) -- a FAIL-OPEN: a
   broken accuracy check silently passes. It is also LLM-based (runs via
   structured_call), so its accuracy rides the same model ceiling that produces the
   errors it is meant to catch. Open: is a "strong" verdict ever VERIFIED, or is
   absence-of-finding treated as correctness?
2. **Canon <-> ledger <-> contract consistency.** PROVEN drift class: the
   `sound_palette` bug -- `episode_canon.sound_palette` was empty for ~100 styles
   because the value derivable from `StoryContract.sound_world` was never threaded
   into the canon (fixed `2baba3a4`). NOTHING asserted canon matched its upstream
   contract. Open: what OTHER canon/ledger fields are silently derivable-but-
   dropped or allowed to diverge (title/premise/setting/time_of_day vs the
   outline; cast in the ledger vs CastLock; style vs the contract)?
3. **Freeze cascade (`_otr_freeze_cascade.py`) verdicts.** `frozen_with_warns` /
   `frozen_with_doctor_edits` ship the episode with KNOWN residual warnings
   (pre_warns/post_warns). Open: which warn classes are tolerated at freeze that
   are actually accuracy defects (e.g. a continuity break the critic flagged but
   the freeze did not block)? Is there a verdict where the doctor's edits
   themselves can INTRODUCE drift (rewriting a line away from the outline intent)?
4. **Positional widget drift (BUG-LOCAL-097).** `widgets_values` is POSITIONAL --
   only APPEND at the end; a mid-list insert silently shifts every saved value.
   This is a LEDGER-of-the-WORKFLOW drift vector. Open: is there an automated
   guard that the saved `otr_scifi_16gb_full.json` widget order still matches live
   `INPUT_TYPES` (the validator exists -- is it RUN in CI, or only ad hoc)?
5. **Schema version (`l3-2026-05-14`) evolution.** Open: when a new optional field
   is added, is there a migration/compat path, or do old ledgers silently read as
   missing -> default -> wrong?

## The questions for the panel

A. **Whole-story accuracy:** is the critic's coverage (continuity/voice/arc) the
   right set, and how do we close the FAIL-OPEN (a failed or absent critic must not
   read as "strong")? Should "strong" require positive evidence, not absence?
B. **Cross-stage consistency:** what is the minimal deterministic assertion set
   that canon/ledger faithfully reflect their upstream sources (contract, outline,
   CastLock) -- to catch the NEXT sound_palette-class silent drop before it ships?
C. **Binary decisions, done right:** where binary gates remain (freeze verdicts,
   reroll triggers, the leak gates), are any FAIL-OPEN or model-dependent in a way
   that lets an inaccurate ledger through? (Not the dropped dialogue lane.)
D. **Drift guards:** which drift vectors (widget order, schema version, canon
   divergence) need an automated, offline, CI-runnable assertion vs an ad-hoc check?
E. **Minimal & durable:** which of the above is highest-leverage, and which is a
   trap (over-engineering a checker that itself drifts)?

## Invariants
Content + checks only (ledger schema frozen unless a migration is explicitly
designed); deterministic + offline + CI-runnable for the GUARDS (an accuracy
guard must not itself depend on an LLM that can fail-open); model/transport-
agnostic; no workflow-JSON node churn; must not break the byte-identical audio
spine or the canonical happy path; UTF-8 no BOM; SFW.
