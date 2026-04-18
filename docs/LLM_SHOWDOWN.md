# Jeffrey's ComfyUI Bug Hunt — LLM Showdown

Live scoreboard: which LLM is best at diagnosing real, reproduced bugs in
the OldTimeRadio ComfyUI custom node pack. Paired with the Bug Bible
regression suite.

## Rules
- Each tracked bug gets a round-robin consult (see CLAUDE.md section
  "Round-Robin Consultation").
- Each model sees the SAME prompt: symptom + relevant source + logs.
- Scoring is done AFTER the real root cause is verified (by fix + retest,
  not by which LLM sounds most confident).
- Models lose points for factual hallucinations, gain points for
  questioning bad premises, and get credit for actionable
  instrumentation proposals.

## Scoring rubric (out of 10)
- **Root-cause match** (0-5): how close the top pick is to the verified
  wedge.
- **Premise hygiene** (0-2): did it push back on wrong assumptions in
  the prompt, or just roll with them?
- **Factual accuracy** (0-2): no hallucinated math, API names, file
  paths.
- **Actionability** (0-1): did the instrumentation / fix proposal
  actually localize the wedge on retest?

## Cumulative standings (live)
| Rank | Model | Rounds | Avg score | Signature strength |
|------|-------|--------|-----------|---------------------|
| 1 | gemini-3-pro-preview | 2 | 9.0 | Does the math. Called out gpt-5.4's "smoking crack" hyper-fixation on downstream deadlock when Bark compute alone explains the timeout. Questions premises aggressively. |
| 2 | gpt-4.1 | 1* | 8.0 | *Correct despite Claude's wrong premise. Rewarded for trusting the workflow data over my summary. |
| 3 | gemini-3-flash-preview | 1 | 7.0 | Catches technical hallucinations (called gpt-5.4's CUDA claim "nonsense"). |
| 4 | gpt-5.4 | 2 | 5.3 | Round 4: rebounded with verifiable bridge.py PIPE deadlock claim (line numbers off by ~40 but PIPE fact true). Prior rounds: CUDA-stall hallucination. |
| 5 | gemini-2.5-flash | 1 | 5.0 | Weak premise check; plausible but underspecified pick. |

## Bug log

### BUG-SHOWDOWN-001: RUN 245 TIMEOUT — 2026-04-17
- **Symptom:** Soak RUN 245 TIMEOUT at 1805s. 337 dialogue lines generated.
  Treatment file not written. Empty error field. VRAM peak 10.04 GB.
  ComfyUI queue stuck `running=1` for hours after timeout.
- **Active workflow:** `workflows/otr_scifi_16gb_full.json` (full rig,
  includes HyWorld trio). *Claude incorrectly told rounds 2+3 that
  HyWorld was NOT active. Rounds 2+3 answers should be re-weighted.*
- **Real root cause:** (pending verification) — leading candidates:
  - HyworldPoll infinite loop (round 1 gpt-4.1's pick — now back on the
    table since HyWorld IS in the graph).
  - Bark generation cumulative time > 1800s soak ceiling
    (gemini-3-pro's pick — 337 lines × ~6s = 2022s).
  - VRAM thrash into Windows Shared GPU Memory fallback
    (gemini-3-pro's #3 pick — matches "GPU still churning").
  - SignalLostVideo or HyworldRenderer ffmpeg pipe deadlock
    (gemini-3-flash's pick).
- **Verification plan:** Deploy 4 `[WEDGE_PROBE]` log lines (Bark entry,
  HyworldPoll each cycle, SignalLostVideo entry, HyworldRenderer entry),
  restart ComfyUI, re-run. Read the last probe that printed — that is
  the wedge.
- **Fixes landed:**
  - HyworldRenderer: `-shortest` replaced with `-t audio_duration` and
    tpad padding (C7 safety). Commit 86bfeae.
  - HyworldRenderer: `_run_ffmpeg` redirects stderr to temp file, not
    PIPE (addresses gemini-3-flash's deadlock suspicion). Commit 86bfeae.
- **Round scores:** see table above. Re-score after verification.

### BUG-SHOWDOWN-001 — Round 4 (corrected premise) — 2026-04-17
Round 4 fed both models the CORRECT premise: `otr_scifi_16gb_full.json` is the
live soak workflow, HyWorld trio IS wired in. Prior rounds' wrong premise
invalidated those scores (already reflected above).

**gpt-5.4** — 6.5/10
- Root-cause match: 3.0/5 — picked `otr_v2/hyworld/bridge.py:_spawn_sidecar` Windows PIPE deadlock. Plausible downstream wedge but fails to explain why Bark (upstream) wouldn't have hit the 1800s ceiling first.
- Premise hygiene: 1.0/2 — rolled with the prompt; didn't question whether the graph actually reached HyWorld at all.
- Factual accuracy: 1.5/2 — line numbers off by ~40 (cited 256-261, actual PIPE usage at bridge.py:296-299), but the PIPE-not-drained fact is correct and verified.
- Actionability: 1.0/1 — three clean `[WEDGE_TRACE]` instrumentation lines, exact insertion points, one per candidate.

**gemini-3-pro-preview** — 9.0/10
- Root-cause match: 4.0/5 — picked BatchBark 337-line compute time > 1800s ceiling. Math holds (337 × 5s = 1685s + model loads > 1800). Does NOT explain the "hours after timeout" tail, so not a complete diagnosis.
- Premise hygiene: 2.0/2 — aggressively called gpt-5.4's "downstream deadlock" framing out: "completely failing to do basic math." Flagged that ComfyUI `running=1` doesn't mean deadlock, it means still-computing.
- Factual accuracy: 2.0/2 — Bark timing math is correct; called out gpt-5.4's "treatment file hallucination" (Director wrote nothing because graph never got that far).
- Actionability: 1.0/1 — Bark-loop probe is exactly the single-line instrumentation that would confirm the diagnosis.

**Combined diagnosis (Claude synthesis, round 4):**
Most likely chain: **Bark 337-line batch chews through the 1800s soak ceiling
(gemini-3-pro wins on this) → Bark eventually finishes → HyworldBridge spawns
worker.py with unread PIPE handles (gpt-5.4 wins on this) → bridge/worker
deadlock accounts for the "hours after timeout" tail.**

Both diagnoses may both be correct, applied in sequence. Verification requires
ALL of: (1) Bark loop probe, (2) HyworldBridge spawn probe. Gemini's premise-
hygiene dominance carries the round.

## Re-score policy
- When a bug is root-caused, re-score ALL prior rounds' picks using the
  rubric.
- Prior scores are preliminary. A model can move up or down after
  verification.
- Claude (synthesizer) also earns a meta-score based on whether the
  synthesis identified the right culprit AND whether the prompt framing
  was accurate. RUN 245 meta-score for Claude: **3/10** — correctly
  identified the ffmpeg pipe-deadlock class, but fed the wrong workflow
  premise to rounds 2 and 3.

## Open bets
- Which model flags a root cause that NO OTHER model mentions, AND the
  root cause is verified? (bonus +2 on that round)
- First model to say "the prompt is missing X, can't answer" instead of
  answering on bad premises: +1 per occurrence.
