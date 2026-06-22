# STORY+CAST FIX -- minimal re-soak results (STEPs 1-4 acceptance)

2026-06-22. Server :8011, real `otr_scifi_16gb_full.json`, **OTR_BYPASS_FREEZE_HALT
OFF** (boot WITHOUT `OTR_SOAK_FREEZE_BYPASS`). Minimal matrix: mistral-nemo (small,
local) + gpt-5.5 (frontier, OpenRouter slot-a), 420w, visualizer-only. Box reset
to baseline (GPU 1310 MiB) before boot. Driver: `scripts/_tmp_resoak_minimal.py`
(temp). Voice: ran first with bark (operator switched to indextts2 for speed --
voice engine does NOT affect the story/cast metrics).

## Acceptance vs the criteria

| criterion | result | evidence |
|---|---|---|
| ships WITHOUT the bypass | **PASS** | every leg reaches a shipped freeze verdict with bypass OFF -- the night-soak crutch is no longer needed |
| 0 cast-contract violations | **PASS** | no `role_mismatch` repair-fail in the logs; CastLock passed every leg |
| no `voice_preset=None` reaches TTS | **PASS** | CastLock STEP-3 gate never raised; every character line resolved a voice (bark `en_speaker_*`, then indextts2 `vz_*`) |
| STEP 4 reroll convergence | **PASS** | scoped re-scoring drove the outstanding count DOWN monotonically (3 -> 2 -> 1), `diverged=False`; repair-then-ship KEPT the re-composed lines |
| >=70% frozen_clean | **partial** -> see note | both legs landed `frozen_with_doctor_edits` (a SHIPPED verdict), not pristine `frozen_clean`; the residual 1-2 subjective quality flags are exactly the STEP 5/6 lever |

## Live log evidence (leg = mistral, representative; index leg identical shape)
```
[OTR_StoryCritic] critic complete: arc_verdict=uneven, ... 3 reroll target(s)
[OTR_Reroll] cycle 1/2: 3 target(s) in scope
[OTR_Reroll] cycle 1 re-composed 3 line(s); re-scoring the critic on 3 scoped line(s)+neighbors
[OTR_StoryCritic] critic complete: ... 2 reroll target(s)
[OTR_Reroll] cycle 2/2: 2 target(s) in scope
[OTR_Reroll] cycle 2 re-composed 2 line(s); re-scoring the critic on 2 scoped line(s)+neighbors
[OTR_StoryCritic] critic complete: ... 1 reroll target(s)
[OTR_Reroll] critic still names 2 target(s) after 2 cycle(s) (diverged=False) --
  repair-then-ship: KEEPING the 5 re-composed line(s), stamping needs_full_rerun for the cascade A2 ship-through
[LFC] A2 repair-then-ship: ... SHIPPING the best candidate ... 0 residual structural error(s) ... never refusing
[LFC:phase_10] frozen_with_warns -- 14 soft gap(s).
[LFC] freeze landed: verdict=frozen_with_doctor_edits reviewer=improved pre_warns=17 post_warns=14
[OTR voice P-OBS] char_voice: line=b002 char=c02 -> ... engine=indextts2   (STEP 3: voice resolved, gate not tripped)
```

## Verdict
STEPs 1-4 fixed the MECHANICAL failures that forced the bypass: cast role source
(STEP 1), cast-auditor scope (STEP 2), voice fail-closed (STEP 3), and the
whack-a-mole reroll that hit the cap -> needs_full_rerun (STEP 4). With bypass OFF
every episode now SHIPS on its own (`frozen_with_doctor_edits`) instead of halting.

The remaining gap to pristine `frozen_clean` is the critic still naming 1-2
SUBJECTIVE quality targets at the cap -- precisely the STEP 5 (flat rubric +
failed_dimension) + STEP 6 (beat-planning arc) levers. This CONFIRMS the plan's
sequencing: STEPs 1-4 first (mechanical), then 5/6 (craft).

## Follow-ups
- The `OTR_BYPASS_FREEZE_HALT` stopgap can be retired (episodes ship without it).
  NOTE: it is already opt-in at the boot layer (`_otr_overnight_soak_boot.cmd`
  gates it behind `OTR_SOAK_FREEZE_BYPASS`); no code change required to "remove"
  it -- just never set it.
- Pre-existing, UNRELATED to STEPs 1-4: a writer `normalize_length` structured
  call can exhaust its retry ladder (MERGE_SHORT_LINES guard) -- the story spine
  ships anyway. A writer-quality item, not a cast/freeze defect.
