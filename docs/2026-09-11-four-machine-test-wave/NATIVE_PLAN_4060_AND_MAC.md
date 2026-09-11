# Native test plan -- 4060 and Mac, run by Claude/Cowork on the machine itself

Written 2026-09-11. Operator directive: *"craft a test plan for Claude/Cowork to run
on the 4060 and Mac natively -- I find that easier -- and they should phone home their
reports."*

**Division of labour for this wave:**

| Machine | Driven by |
|---|---|
| 5080 | this window, directly |
| RunPod | this window, directly |
| **4060** | **Cowork running natively on that box -- this document** |
| **Virtual Mac** | **Cowork running natively on that box -- this document** |

Paste the whole section for your machine into a Cowork window ON that machine. It is
written to be self-contained: do not assume the reader has seen this conversation.

---

## Rules both machines follow

1. **PULL FIRST, and report the HEAD you actually ran.** A lane that cannot state its
   commit has not qualified anything.
   ```
   git fetch origin v2.0-alpha && git pull --rebase origin v2.0-alpha && git rev-parse --short HEAD
   ```
2. **WRITE FINDINGS TO `docs/` ONLY. DO NOT PUSH.** Commit locally if you like. The
   push belongs to the 5080 so four machines cannot collide on one branch. If you
   believe something must be pushed, say so in your phone-home and wait.
3. **PHONE HOME per leg**, not as a stream: one message when you start (with the
   HEAD), one per leg (pass/fail + the artifact path on disk), one at the end.
4. **Every qualifying run loads `workflows/otr_canonical.json` through
   `scripts/otr_canonical_api_run.py`.** No `--workflow` override, no `--replay-from`,
   no `partial_execution_targets`, no hand-built graph. There is one graph.
5. **Use `--run-label`, NEVER `--title`.** `--title` fills the `episode_title` widget,
   so your harness label becomes the on-screen TITLE CARD and the writer stops naming
   the episode. `--run-label` echoes to the console only.
6. **A leg is not complete until it publishes to `otr/obs/`.** If a leg has run more
   than five minutes with nothing there, treat it as failing and go read the leg log
   rather than waiting it out. Never move, hide, sort or clean anything out of
   `otr/obs/` -- seeing the episodes there is the point.
7. **Reset before every headless run.** Kill SELECTIVELY by command line
   (`Get-CimInstance Win32_Process` on Windows, `ps`/`pgrep` on macOS) -- **never a
   blanket `Stop-Process -Name python` or `pkill python`**, which also kills Cowork's
   own tooling and severs your tools mid-run. Confirm port 8000 is not listening and
   GPU memory is back to the desktop baseline.
8. **Record the full receipt for every attempt, pass OR fail:** source fields, code
   and canonical hashes, actual model + quantization + profile, prompt ID, elapsed
   time, memory and model loads, every repair attempt, requested vs actual acts and
   cast, ledger seals, final asset paths. **Preserve terminal failure evidence before
   asserting anything.** Keep the full denominator -- a failed attempt is part of the
   record, not something to re-roll away.
9. **Report the rolled `visual_style` for every leg.** The style roll is independent
   of the source-bank roll by design, so several free legs give a natural A/B. Do not
   pin it, and do not re-roll to get one you prefer.
10. **Do not edit production code while a run is active**, and do not change story
    controls to hit a time target.

---

## 4060 (MRKT) -- the portability surface

```
You are Cowork running natively on the RTX 4060 (8 GB). Repo:
custom_nodes/ComfyUI-OldTimeRadio. Read CLAUDE.md and AGENTS.md first.

Pull v2.0-alpha and report the HEAD you ran.

You are the only box that can answer "does this work somewhere other than where it
was written." That is your whole value in this wave, so test the 8 GB path.

PROFILE: use `otr_4060_12b_gguf_offload`. It is status "shipping".
DO NOT use `8gb_lite` or `otr_4060_floor` -- both are status "draft", and 8gb_lite is
recorded in PBUG-20260904-05 as REFUSING IN TWENTY SECONDS on two of the three banks,
because the 12B writer's 2,048-token context cannot hold their prompts. Picking a
draft profile burns your evening on an already-diagnosed refusal.

Leg A -- one-act monologue, full canonical. Then a three-act ensemble.
  Record the profile that actually RESOLVED: grep the leg log for the resolved-profile
  line and the applied block. Do not take the requested profile as the applied one.

Leg B -- verify today's fixes on 8 GB. These all shipped 2026-09-11 and none has run
on your hardware yet:
  1. The scopes MP4 must land under output/otr/episodes/<episode_id>/composited/ --
     NOT in episodes/_shared/tmp, and NOT in the system temp dir. Check on disk.
     Older stranded otr_scopes_*.mp4 files in _shared/tmp are prior evidence; leave
     them alone.
  2. The episode must publish to otr/obs/ and the file must be non-zero.
  3. If your box resolves the episode directory somewhere unexpected, that is exactly
     the finding this lane exists to produce -- report the actual resolved path.

Leg C -- KNOWN, report-and-move-on, do not chase:
  - small-canvas credits overflow (otr_credits_roll logs at ERROR that the canvas is
    too small for the card as designed). Open row 2.4 voice/credits.
  - If no monospace font resolves, the credits now DEGRADE to a small bitmap font and
    log a warning instead of killing the episode (operator ruling 2026-09-11). If you
    see an ugly title card, that is the new intended behaviour -- report how it looks,
    and set OTR_CREDITS_FONT to any monospace .ttf if you want it pretty.

Leg D -- fresh-install friction. Anything that needed a manual step the docs do not
mention goes in docs/4060_DRILL_LOG.md, which is yours.

Findings to docs/. Do not push. Phone home per leg.
```

---

## Virtual Mac -- Apple Silicon

```
You are Cowork running natively on the virtual Mac (Apple Silicon). Repo:
custom_nodes/ComfyUI-OldTimeRadio. Read CLAUDE.md and AGENTS.md first.

Pull v2.0-alpha and report the HEAD you ran.

Leg A -- one full canonical episode end to end, published to otr/obs/. The point is
that the one canonical graph runs at all on this platform.

Leg B -- TEXT RENDERING, and look at the delivered FRAMES, not the logs. This platform
has burned us here before: eight published macOS episodes shipped with the hero title
running off the right edge of the frame, because the font MEASUREMENT side resolves a
font FILE by absolute path while the DRAWING side hands libass a family NAME, and when
measurement finds nothing it falls back to a bitmap default that ignores the requested
size -- so the title is drawn full-size at a position computed from a width ten times
too small.
  Check, in the actual frames: the title card, the captions, and the credits roll.
  Report what you SEE, with the frame paths.

Leg C -- the credits font behaviour changed TODAY and your box is the most likely
place to exercise it. Apple Silicon ships no DejaVu and no consola. Previously, if no
monospace truetype resolved, credits raised CreditsDataError and KILLED A FULLY
RENDERED EPISODE at the very last stage. As of 2026-09-11 it degrades to PIL's
built-in font and logs a warning instead (operator ruling: "don't assume people have
fonts installed -- I'm open to some bad formatting as long as it doesn't crash").
  Report which path you hit: did a real font resolve, or did you get the degraded
  bitmap card? If degraded, say how bad it looks. Either answer is useful; the
  failure mode we care about is the episode not publishing at all.

Leg D -- note every place this platform needed something the Windows path did not:
fonts installed, ffmpeg build features, PyAV limitations, missing codecs. This feeds
the open shared font-resolution question.

Findings to docs/. Do not push. Phone home per leg.
```

---

## What neither machine covers

**Listening.** The opening/middle/ending audition on at least two publications,
including a six-act, is the operator's own ear. No agent in this wave can ingest
audio, and reading the TTS text or checking a waveform is NOT listening -- do not
record it as such.

**Visual continuity.** The clothing/era variance seen in canonical09 is CLOSED as a
work item (see `docs/2026-09-11-visual-continuity-diagnosis/ARC_CLOSED.md`). The pack
and the story are a combination and exactness is not the goal. Report the rolled style
per leg as evidence; do not open it as a bug.
