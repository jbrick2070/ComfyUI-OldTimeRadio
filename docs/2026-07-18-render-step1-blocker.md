# RENDER window -- Step 1 wiring-smoke BLOCKER (Mistral-Nemo local bake-off)

**Date:** 2026-07-18
**Window:** RENDER (GPU legs + blind judging)
**Baseline:** HEAD `c507acff` == `origin/v2.0-alpha`, branch `v2.0-alpha` (4-bank
Sonnet-bake-off rip landed). Roster confirmed = the exact 8-id tuple.
**Writer pinned:** creative + technical both `mistralai/Mistral-Nemo-Instruct-2407`.
**Harness:** `scripts/otr_headless_canonical.ps1` -> `otr_canonical_api_run.py`
(loads `workflows/otr_canonical.json`; full pipeline through obs_publish). Selective
CIM reset + fresh boot per leg.

## Bottom line

GREEN (RESULT SUCCESS + obs_publish + asset on disk) was **never reached** for any 30w
canonical leg. I stopped before Step 2 / Step 3 GPU spend, per the operator gate.

There were TWO separate problems. One was mine (fixed). The other is that **30 words is
too little material for these writer lanes** -- a clean 30w run STILL fails.

## Problem A (fixed): a leaked `OTR_TEST_MODE=1` poisoned the first codex leg

The very first `scifi_codex_v4` leg booted the ComfyUI server in TEST MODE -- 4
`test-mode injection (in-memory only, disk save skipped)` lines in its server log,
including `image_dispatcher`. Root cause: I had set `OTR_TEST_MODE=1` /
`CUDA_VISIBLE_DEVICES=""` in a probe shell, and `Start-Process` inherits the parent
environment, so the leg's server inherited them. Test mode stubbed the still/shot
dispatch -> the shots carried no frame budget -> the video node raised
`OTR_SceneAwareScopes: total_target_frames<=0`. **That crash was a test-mode artifact,
NOT a real video-path bug.** (The fable2 leg, launched from a clean shell, had 0
test-mode hits.)

**Fix (landed in the leg runner):** `tmp/_render_leg_run.ps1` now strips
`OTR_TEST_MODE`, `CUDA_VISIBLE_DEVICES`, `OTR_CAST_SEED`, `OTR_STYLE_SEED` from the
environment before every boot, so no leaked env can poison a render. Verified: the
clean re-run logged `OTR_TEST_MODE=[]` and 0 test-mode hits.

## Problem B (the real blocker): clean 30w legs fail in the WRITER

With the env guaranteed clean, both banks still fail a 30w canonical leg -- in the
**writer**, before any media, because 30 words gives the lanes too little to satisfy
their own deterministic script validators:

| Leg (clean) | Bank | Stage | Failure |
|---|---|---|---|
| clean1 | scifi_codex_v4 | writer P5 | P5 script validation fails repeatedly (`invalid accepted-order boundary`, then `spoken text begins with a self-vocative`) -> the accepted retries pile up KV-cache VRAM -> `torch.OutOfMemoryError: P5 failed ... (out of memory)` (`CodexPassError`). 0 test-mode hits. |
| smoke2 | scifi_fable2 | writer script pass | `Fable2ScriptError: pass 'script' failed after 5 attempts; markup ladder exhausted` (BAD_LINE_SHAPE x2 + CAST_MEMBER_SILENT: ISHIKAWA, no fallback). 0 test-mode hits. |

The codex OOM is a *consequence* of the 30w content tripping the validator: 30w ->
malformed/thin script -> P5 rejects -> re-generates -> VRAM accumulates across the
back-to-back retries -> OOM. At 420/720w the validator passes on the first try (as in
every prior successful leg), so there is no retry storm and no OOM.

## 120w fallback (operator-authorized) ALSO fails -- both lanes, on structural gates

Per operator ("smoke all at 120w if 30 is too stingy"), re-ran at 120w clean:

| Leg (clean, 120w) | Bank | Stage | Failure |
|---|---|---|---|
| s120_1 | scifi_codex_v4 | writer P3 | `PostValidationError: draft.beat_count ... flattened draft beat count 6 must equal accepted advisory count 12` -> `CodexPassError` after 2 attempts. The outline locked 12 beats; a 120-word draft can only carry ~6; P3 requires an EXACT match. |
| s120_2 | scifi_fable2 | writer script pass | `Fable2ScriptError: pass 'script' failed after 5 attempts: WORD_BUDGET character words 87 outside 95-145; SCENE_COUNT parsed 2 expected 1; SCENE_WORD_GROSS scene 1 = 45 outside 60-180` (no fallback). |

**Root pattern:** both writer lanes enforce STRICT deterministic structural gates that
HARD-FAIL (raise) with no fallback -- codex `_otr_scifi_codex.py:866-872` (draft beat
count must EXACTLY equal the outline's locked count) and fable2
`_otr_scifi_fable2.py:1953` (`WORD_BUDGET`/`SCENE_COUNT`/`SCENE_WORD_GROSS` bands). The
free local **Mistral-Nemo** satisfies these only stochastically; at 30w and 120w in this
session, both lanes missed, on different gates each time. (These gates are LAWFUL under
THE LAW -- they are deterministic, not LLM verdicts -- but they make short local smokes
unreliable.) The codex advisory beat count is fixed by the outline
(`make_advisory_word_blueprint:2245-2256` distributes words across a FIXED beat list, it
does not reduce the beat count for short episodes), so short codex episodes structurally
cannot pass P3.

## Why 30w is the problem, not the banks or the rip

- Prior full-media successes on disk, SAME banks, larger sizes: `scifi_codex_v4`
  "The Whisker Effect" (obs, 56.6 MB, 133 authored words, 6 shots); `scifi_fable2`
  "The Stone Frequency" (obs, 406 MB, 720w). Both banks CAN produce full media.
- The rip `c507acff` touched only writer/roster/pack/test/doc files -- NOT the video
  path or the fable2 lane. Not a rip regression.
- Every prior OTR bake-off / full-media campaign ran at 320/420/720w. **30w full-media
  was never an exercised config.** The operator's Step-1 "30 words" is below the size
  these writer lanes need to emit a valid script.

## Environment note (headroom) -- NOT the cause

Idle VRAM baseline is only ~2.3 GB (desktop apps incl. Resolume running text art are
negligible; operator confirmed no clips loaded). That leaves ~13.7 GB free -- ample for
Mistral-Nemo. The OOM was NOT a desktop-headroom problem: it was the codex lane
re-generating the P5 draft three times back-to-back (validator rejects at 30w) and
accumulating KV-cache VRAM across the retries without freeing between attempts. A failed
leg does leave its ComfyUI server resident (~8 GB) until reset -- worth killing between
legs -- but that is hygiene, not the root cause. No need to change the Resolume setup.

## Recommendation

Short wiring smokes (30w AND 120w) are not viable on the local writer: the codex and
fable2 structural gates need production-length episodes to be satisfiable. Two honest
paths:

1. **Skip the artificial short smokes; run the bake-off legs directly at 420w** (codex_v4,
   fable2, base codex). At 420w the model has room to satisfy the beat-count / word-budget
   / scene-count bands (the Sonnet arm's codex_v4 720w passed; the local whisker_effect
   passed at 133w). Those legs ARE the wiring proof AND the bake-off data; any structural-
   gate failure at 420w is a recorded robustness outcome (Step-3 axis b), not a blocker.
   Cheapest path to real signal. **Recommended.**
2. **Coder-window fix first:** relax the writer lanes' hard structural gates so a valid-
   but-differently-shaped short draft is accepted (advisory, recorded-not-gated), rather
   than raising -- codex P3 exact-beat-count and fable2 WORD_BUDGET/SCENE_COUNT. This is
   what "make short smokes work" actually requires; it touches story contracts and needs
   the full suite + Bug Bible + a live leg. Not a render-window change.

This render window did NOT edit code (window-roles; no story-contract change mid-campaign).

## Reproduce
```
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\otr_headless_canonical.ps1 `
  -Profile none -Words 30 `
  -Set OTR_LedgerScriptWriter.source_bank=scifi_codex_v4 `
  -Set OTR_LedgerScriptWriter.creative_writing_model=mistralai/Mistral-Nemo-Instruct-2407 `
  -Set OTR_LedgerScriptWriter.technical_model=mistralai/Mistral-Nemo-Instruct-2407
```
Leg logs: `tmp/legs/clean1_scifi_codex_v4.log` (clean, OOM), `tmp/legs/smoke2_scifi_fable2.log`
(clean, markup ladder), `tmp/legs/smoke1_scifi_codex_v4.log` (poisoned, test-mode).
Server logs: `tmp/otr_headless_53371.log` (clean codex), `tmp/otr_headless_56681.log`
(fable2), `tmp/otr_headless_54881.log` (poisoned codex).
