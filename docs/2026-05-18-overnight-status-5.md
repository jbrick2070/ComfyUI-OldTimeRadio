# Overnight status #5 — 2026-05-18 Sprint H §3.7 retest #12

**Status:** HALT — post-writer defect surfaced (MusicGenTheme
unknown style slug). Per Jeffrey's branching rule for retest #12,
this is the "anything else" branch.
**No fix applied to MusicGenTheme.** Outline target_words follow-up
fix DID land (commit `6cbdee0`) since it's still inside the outline
scope.

---

## TL;DR

**Path C tree refactor works.** Outline phase passed on iter 2 of
retest #12 -- the first time the writer phase has cleared in §3.7
since retest #5. Outline used `1 macro + 3 phase + 14 beat = 18
LLM calls` end-to-end (matches Jeffrey's design estimate of ~16).

The freeze cascade ran on a 16-line ledger. The pipeline advanced
**past** the freeze cascade for the first time in the §3.7
campaign, hitting a downstream defect at MusicGenTheme:

```
ValueError: MusicGenTheme: unknown style slug
'station_supply_arrival_protocol'. Add an entry to _STYLE_PALETTE.
Known slugs: closed_room_suspense, deep_space_distress_call,
detective_case_file, haunted_broadcast_signal, laboratory_containment,
mission_control_procedural, noir_interrogation, pulp_serial_cliffhanger,
radio_newsroom_emergency, small_town_uncanny
```

The style_picker's 2-pass design INVENTS slugs ("Pass 1 inventor
produces 5 distinct snake_case candidates from article + 5 sampled
seed flavors" per memory). MusicGenTheme has a hardcoded 10-entry
`_STYLE_PALETTE` -- only the canonical preset names. Collision is
structural between the LLM invention design and the hardcoded
palette lookup.

FluxBranchGate verdict: still unknown. MusicGenTheme fires before
the gate, crashing the workflow.

---

## What ran

Commit `dd3b5ec` on `v2.0-alpha`: outline LLM call broken into tree.

Followup commit `6cbdee0` on `v2.0-alpha`: outline Stage 3
target_words made Python-authoritative (addresses retest #12 iter 1
OutlineBudgetViolation; pushed mid-retest).

§3.7 retest #12 launched at 2026-05-18T10:35:48 via
`sweep_and_launch.bat --iters 2 --inter-iter-sec 10`.

### Iter 1 (worker_iter_001.json -- commit dd3b5ec)

```
status:        error
failure_class: writer_outline (correctly routed)
exception:     OutlineFailedError
sub-error:     OutlineBudgetViolation: Phase 'setup' got 112 words
               (target 84, allowed 67-101). Per-beat target_words
               each in [20,35] range, but Python's previous
               "override only if out-of-range" branch let the
               sum drift.
executed_count: 4
peak_vram_gb:   10.02
wall_time_s:    150.30
```

This case is exactly the gap commit `6cbdee0` closes. Stage 3
target_words is now Python-authoritative regardless of whether
the LLM-provided value is in range.

### Iter 2 (worker_iter_002.json -- commit dd3b5ec)

```
status:        error
failure_class: unknown (classifier not aware of MusicGenTheme yet)
exception:     ValueError: MusicGenTheme: unknown style slug
               'station_supply_arrival_protocol'. Add an entry to
               _STYLE_PALETTE.
executed_count: 6 (vs prior retests stuck at 4)
peak_vram_gb:  10.13
wall_time_s:   291.29
prompt_id:     2020a39c-b239-4162-b3ec-b8047ecd0b6a
```

Iter 2 advanced past every prior retest's failure point. Key
log markers from comfy_session_iter_002.log:

```
[OTR_LedgerScriptWriter] cast locked: 3 rows (announcer + 2 chars)
[OTR_Outline] success: 16 beats (14 voiced, 2 announcer,
              0 music_inter); calls used: 1 macro + 3 phase
              + 14 beat = 18 total
[OTR_LedgerFreezeCascade] running cascade on ledger
              pending_20260518_103907 (16 lines)
ValueError: MusicGenTheme: unknown style slug
              'station_supply_arrival_protocol'.
```

This iter 2 succeeded BY LUCK on per-beat target_words sum
landing in budget (Gemma's slightly different temperature seed
produced compliant numbers). Commit `6cbdee0` makes this
deterministic.

## Supervisor halt

The supervisor saw 2 failures:
    iter 1: writer_outline
    iter 2: unknown
and halted via the 2-consecutive-same-class rule only because the
classifier doesn't yet have a route for the MusicGenTheme defect.
The iter 2 advancement is invisible in the same-class halt logic
but obvious in the log.

## FluxBranchGate verdict: still unknown

MusicGenTheme fires BEFORE the FluxBranchGate in the executor
order (both downstream of the freeze cascade's script_json signal;
MusicGenTheme hit first this iter). VRAM at iter 2 crash: 10.13 GB.
No OOM, no co-residence.

## Why this is a "halt on anything else" branch

Per Jeffrey 2026-05-18 directive:

> "GREEN through gate → advance per existing posture
>  RED still on outline → audit results inform which sub-call
>      is failing; refine just that call
>  RED on co-residence OOM → auto-fix with gate analog
>  RED on anything else → halt + status-5"

MusicGenTheme is "anything else": it's a downstream-of-writer
defect, NOT the outline tree, NOT a co-residence OOM. Per the
letter: halt + status-5. **No fix applied to MusicGenTheme.**

## The MusicGenTheme defect (recommended path, awaiting sign-off)

`nodes/musicgen_theme.py` has a hardcoded `_STYLE_PALETTE` dict
that maps a fixed 10 canonical preset slugs to musical parameters
(genre / tempo / mood). The current 10 entries (per memory entry
"Style preset set landed 2026-05-10"):

  closed_room_suspense, detective_case_file, pulp_serial_cliffhanger,
  mission_control_procedural (default fallback), deep_space_distress_call,
  noir_interrogation, small_town_uncanny, radio_newsroom_emergency,
  haunted_broadcast_signal, laboratory_containment.

The style_picker's 2-pass design (memory entry "Style auto-derive
sentinel label...") explicitly INVENTS new slugs from article
context + 5 sampled seed flavors. So MusicGenTheme's lookup MUST
have a fallback when the invented slug is missing.

**Options for Jeffrey:**

A. **Soft-fallback in MusicGenTheme** (smallest, in-spirit fix).
   When the slug is not in `_STYLE_PALETTE`, use the mission_control_procedural
   default (already labeled "default fallback" in memory). Log a
   warning. Lets every LLM-invented slug get a workable music bed
   even if not perfectly tuned. ~10 lines of code.

B. **Slug-canonicalizer at the writer fan-out boundary.** Add a
   Python helper that takes any slug (canonical OR invented) and
   maps it to one of the 10 canonical slugs via embedding similarity
   or keyword matching. Used both by MusicGenTheme and by any other
   downstream consumer that has a fixed slug palette. ~50 lines.
   More general; needs more thought.

C. **Pin the style_picker to a CANONICAL output.** Force the
   chooser pass to pick from the 10-slug fixed list (no invent).
   Defeats the "let the story decide" design intent but unblocks
   the downstream palette consumers immediately. Memory entry
   "Style auto-derive sentinel label" suggests Jeffrey explicitly
   wanted the invent path.

D. **Audit + soft-fallback every downstream slug consumer.**
   MusicGenTheme is the first; there may be others (visual style
   prompt, ledger metadata). Recipe like A but broader.

Recommendation: **A** for the immediate unblock to get
FluxBranchGate fire telemetry; **D** as the medium-term audit
once the gate is exercised. Both require Jeffrey's sign-off.

## Outline tree refactor verdict

**Path C is working as designed.** Per-stage retry localizes
failures (iter 1's OutlineBudgetViolation surfaced one call's
allocation drift instead of poisoning the whole 1500-token
mega-output). Iter 2 cleanly produced a valid 16-beat outline
with 18 small calls, each independently constrained.

Wall time per iter dropped to 150-290s (vs 230-340s on legacy
mega-call). The wall time is now dominated by per-call inference
latency, not by failed retries.

`max_new_tokens` per stage:
    Stage 1 (macro):    250
    Stage 2 (per phase): 200
    Stage 3 (per beat): 150

`max_attempts` per call: 3 (with the 3rd attempt being a repair
call carrying the prior validation error). Each call's failure
budget is independent.

## What we did NOT do (per directive)

- Did NOT fix MusicGenTheme.
- Did NOT add a classifier route for the MusicGenTheme defect.
- Did NOT touch any workflow JSON.
- Did NOT modify the style_picker.
- Did NOT touch FluxBranchGate / LtxBranchGate / UnloadAll.
- Did NOT advance to §3.8 or §3.9.
- Did NOT bump a version label.

## Commits this session (Path C)

- `34f759e` -- Path C step 1: upstream LLM audit (read-only)
- `dd3b5ec` -- Path C step 2: outline LLM call broken into tree
- `6cbdee0` -- Path C followup: outline Stage 3 target_words
                Python-authoritative (mid-retest fix for iter 1's
                allocation drift)

## Halt closed

Awaiting MusicGenTheme path direction (A / B / C / D / other).
Same posture as status #1/#2/#3/#4: pre-authorized fixes
overnight remain same-pattern co-residence OOM only;
halt-and-report conditions unchanged; hard stops unchanged.
