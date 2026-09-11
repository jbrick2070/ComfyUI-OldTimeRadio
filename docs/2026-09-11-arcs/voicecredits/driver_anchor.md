# Driver anchor -- 2.4-voice-credits: Bark non-speech repair and small-canvas credits

Written by the driver (Claude Opus 5, Cowork, 5080) BEFORE any panel round. The panel
proposes; the driver disposes and verifies every claim against the real Windows files.

**Round shape (operator directive): ONE second opinion per round.** r1 codex, r2
cursor, r3 codex, r4 sonnet. **An arc IS coding.** A no-brainer gets no arc -- this
row is here because grounding found a genuine fork with more than one defensible
answer.

**THE OPERATOR'S BAR:** *"as long as it doesn't crash when it's not supposed to."*
Exactness is NOT the goal -- this is a fun experimental app, and aesthetic drift is
explicitly acceptable. This row is classified **CORRECTNESS**.

---

## 1. What is TRUE TODAY, grounded against current code

Bark: generate_voice (eng_bark.py:156-235) runs exactly one live gate today -- BarkSilentOutputError on empty/nonfinite/peak-below-1e-4 output (eng_bark.py:208-234). B1 prevention (_resolve_bark_speech_only default True, _otr_bark_lib.py:635-639) already strips [music]/[whistles]/etc tokens at the source. The speech-shape scorer that WOULD catch a residual squeal (high_band_edge_ratio/flag_high_band_artifact, _otr_bark_lib.py:684-727) exists and is tested, but is called only by the offline scripts/bark_artifact_scan.py -- never from generate_voice/_generate_single_line. So a non-silent squeal that survives B1 passes the one live gate and ships undetected except by manual post-hoc scan or listening. Credits: otr_credits_roll.roll() sizes col1 from the finished video's actual w/h; _draw_col1_abridged (939-971) is the last rung of a whitespace+row-dropping degradation ladder; when it still doesn't fit it draws anyway (never raises, by explicit design -- "a terminal node never destroys a finished episode") and logs log.error (964) naming the pixel overflow. Confirmed overflow: 65px at 512x288 (ltx_8gb), 12px at 640x360, both drawn with PIL silently clipping the tail past the log line.

**Key files:** `nodes/_otr_audio_engines/eng_bark.py:156-235`, `nodes/_otr_bark_lib.py:614-727`, `scripts/bark_artifact_scan.py`, `tests/test_bark_artifact_metric.py`, `nodes/otr_credits_roll.py:872-971`, `docs/GO_FORWARD_PLAN.md:148`, `docs/GO_FORWARD_ARCHIVE.md:8623`, `docs/2026-08-28-c2-bark-health/ADVERSARIAL_REVIEW.md`, `docs/2026-09-11-four-machine-test-wave/PROMPTS.md:127-129`, `docs/HANDOFF_LOG.md:1518-1536`

**Blast radius:** Bark half: confined to nodes/_otr_audio_engines/eng_bark.py + nodes/_otr_bark_lib.py, inside generate_voice for the "bark" engine only (char_voice/announcer_voice roles; bark is non-default since 2026-06-04, indextts2 is the char_voice default, so episodes that don't select bark are untouched). Row itself says "opt-in" -- zero blast radius elsewhere if gated by env flag; adds GPU wall-clock per retried Bark line only (sequential execution, no concurrency conflict). No workflow JSON/widget change unless the opt-in flag is exposed as a widget rather than an env var (a sub-decision for any arc). Credits half: confined to nodes/otr_credits_roll.py col1 layout, a terminal step-21-of-22 node that already never raises; affects only small-canvas legs (confirmed ltx_8gb 512x288, and 640x360) -- purely visual, bounded, never blocks obs_publish.

## 2. Claims in the existing row description that CURRENT CODE CONTRADICTS

These are why this anchor was rebuilt from the code rather than from the backlog
text. A row description is a dated claim, not a finding.

- "eng_bank.py" does not exist anywhere in the repo (zero grep hits); the actual file is nodes/_otr_audio_engines/eng_bark.py plus nodes/_otr_bark_lib.py -- likely a typo in the row description.
- "no speech-shape scoring today" is false as literally stated: a deterministic, unit-tested speech-shape scorer (high_band_edge_ratio/flag_high_band_artifact) already exists in nodes/_otr_bark_lib.py:684-727. What's actually missing is live wiring -- it's called only by the offline scripts/bark_artifact_scan.py, never from generate_voice. Correct framing: 'not wired into live generation,' not 'does not exist.'
- The row reads as a fresh gap but is a reopening of a 2026-06-21/22 roundtable-converged sprint (docs/2026-06-22-bark-artifact/BUILD_PLAN_DRAFT.md, git history) that shipped B1 (speech-only mode, live default-on) and B2 (seed threading, live) while explicitly deferring the retry/reroll loop as optional future work once B1/B2 'plummeted' the artifact rate.
- A separate, more recent precedent -- the 2026-08-28 C2 review (docs/2026-08-28-c2-bark-health/ADVERSARIAL_REVIEW.md) -- explicitly evaluated and rejected 'RETRY the same line once' as the response to a Bark render defect at this exact seam (generate_voice), which the shipped code documents directly (eng_bark.py:221-224). The row's 'bounded keep-best' proposal is functionally a retry policy and must reckon with this precedent even though the defect class differs (silence vs. non-silent artifact).

## 3. The fork -- this is what the round must pressure-test

Bark: wire the existing, tested nodes/_otr_bark_lib.py:684-727 (high_band_edge_ratio/flag_high_band_artifact) into an opt-in bounded retry-and-keep-best inside generate_voice/_generate_single_line -- un-deferring the 2026-06-21 B3 "reroll loop" that was consciously left as future work once source-side prevention (B1/B2) landed -- VERSUS leave it as the current offline/manual scripts/bark_artifact_scan.py audit tool and accept the residual rate, consistent with (a) the 2026-06-22 sprint's own deferral reasoning and (b) the 2026-08-28 C2 review's explicit rejection of "RETRY the same line once" at this same generate_voice seam for a related Bark defect (docs/2026-08-28-c2-bark-health/ADVERSARIAL_REVIEW.md, shipped in eng_bark.py:221-224). Real precedent exists on both sides; this is not mechanical. Credits: NO fork -- mechanical/visual-design work only if ever picked up (a distinct small-canvas card layout, not "more ladder" per the code's own comment at otr_credits_roll.py:968), and the plan/operator have already ruled it out of scope for now.

## 4. Questions for the reviewer -- answer these; do not restate section 1

1. **Is the mechanism in section 1 complete and correct?** Read the cited files
   yourself. Is there a step the driver missed, or something that already partially
   mitigates this?
2. **Which side of the fork survives contact with the real code?** Name the files and
   functions each choice would actually touch, and say which you would not write.
3. **What is the smallest change that resolves it?** Smallest that is CORRECT -- not
   smallest that compiles.
4. **Blast radius, measured not asserted.** This project runs on a 16 GB 5080 and an
   8 GB 4060 (CLAUDE.md section 0B). If the change touches shared code, what
   measurement proves the machine you are NOT fixing is unchanged?
5. **Is any part render-inert enough to ship before a four-machine test wave, or must
   all of it wait?** Say plainly.
6. **What would make this row WRONG to do at all?** Argue the other side once.

## 5. Hard constraints -- a proposal violating any of these is rejected on sight

- **No new content gates**, story-rejection gates, word/duration/cast-size limits,
  standalone checkers, chunkers, or recursive retry loops.
- **No prompt rewrites.** Prompts are hand-crafted per model and CHARACTER-BUDGETED
  (`motion_registers` 240 chars enforced at load, BUG-LOCAL-112; `_fit_motion_slot`
  truncates to 60). Adding conditional nuance spends budget that does not exist.
- **Story/prose QUALITY work is DONE** by operator directive. Correctness defects are
  still open; better prose is not.
- **Do not make an OOM silent.** An OOM is the only acceptable killer and must stay
  truthful. The goal is to stop CAUSING avoidable ones, never to hide them.
- **Do not reduce how many episodes reach `otr/obs/`.** A fix that refuses more than
  it saves is a worse fix.
- **One canonical graph.** No second workflow JSON, no new output-path owner, no
  reviving deleted code.
- **Never blanket-kill Python processes** -- that kills the agent's own tooling.
- Style: no curse words, never the name "dummy".
