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

## Before anything: the frozen head, and one flag

    WAVE HEAD: *** NOT YET FROZEN -- DO NOT START ***

    This line gets a real commit hash when coding stops. Until then, no lane
    starts. When it is filled in: report the HEAD you ACTUALLY pulled, never
    check out the frozen hash, and if you pull a commit that touches `nodes/`,
    `scripts/` or `workflows/` AFTER the hash on that line, STOP and phone home
    -- that would mean the freeze was broken.

**PASS `--timeout 0` ON EVERY LEG. This is the single most likely way to
manufacture a false failure tonight.** `--timeout` defaults to 5400 seconds --
ninety minutes -- and it bounds how long this PROCESS watches, not how long the
render may take. The 4060's own measured canonical is 121 minutes
(`4060_DRILL_LOG.md:1087`); the Mac has legs past two hours. At minute 90 the
default prints `RESULT TIMEOUT` on a render that is alive and still going to
publish, and a window that then "resets before every headless run" KILLS A
PASSING EPISODE and starts it over. `--timeout 0` waits for a terminal result.

**4060 ONLY -- phone home your `kernel_source` counts.** Grep your leg logs and
the stored Ghost prompts for `kernel_source` and report how many shots resolved
`key_object_in_beat` versus plain `key_object`. Your profile
(`otr_4060_12b_gguf_offload`) renders on `animatediff15_v3_haunted_video`, whose
engine inherits `GHOST_PROMPT_PROFILE`, so **your legs are the first live
exercise of the beat-ranking crux resolver** -- no episode has ever rendered
through that path. That ratio is the first real hit-rate measurement of it, and
it is the number that decides how the next tier gets built. If a leg fails, that
resolver belongs on the suspect list beside the platform itself.

## Rules both machines follow

**YOUR REPORT FILE IS GITIGNORED BY DEFAULT, AND `git add` WILL NOT SAY SO.**
`.gitignore:255` is `docs/2026-*/` -- dated folders are local scratch on purpose, and
this wave folder is only tracked because its two documents were force-added.
`kibitz-runs/` is ignored the same way. So:
  * **The phone-home is the delivery mechanism, not the file.** Put the findings IN
    the message. A report that exists only on your disk did not reach anyone.
  * If you also want the file in the repo, it needs `git add -f <path>` -- and per
    rule 1 you still do not push; say in your phone-home that you force-added it and
    let the 5080 carry it.
  * The 4060's `docs/4060_DRILL_LOG.md` is a normal tracked path and needs none of
    this. Prefer it for anything durable.


0. **WHAT COUNTS AS A FAILURE, and it is much narrower than you will assume.**
   Operator directive 2026-09-11: *"be careful not to fail anything because of a
   verification round check. Only an out of memory should fail."* With his standing
   bar: *"as long as it doesn't crash when it's not supposed to"*, and *"this is a
   fun experimental app, I'm not expecting anything exact."*
   * **A leg FAILS only when the run DIED:** an uncaught traceback that ended the
     prompt, an out-of-memory, a hang with nothing in `otr/obs/` past the five-minute
     rule, or the server going away. That is the whole list.
   * **A leg that PUBLISHED to `otr/obs/` PASSED.** Even if the title card is ugly,
     the font fell back, a caption is mistimed, the cast is smaller than requested,
     the story is thin, the images do not match the pack, or a checker somewhere
     printed a complaint. Those are OBSERVATIONS. They go in the receipt as
     observations, under their own heading, and they do not change the verdict.
   * **Do NOT invent a quality gate.** No "it passed but the images were poor, so I
     am calling it a partial". No scoring, no rubric, no threshold. If you find
     yourself reaching for a qualifier, the answer is PASS plus an observation.
   * **A validator's refusal is a FAILURE OF THE VALIDATOR, and it is reported as a
     defect in the code, not as a failing leg.** If a `verify_*` / `assert_*` /
     preflight check is what ended a 30-minute render, that is the single most
     valuable thing you can phone home tonight -- name the file and line. The rule
     it violated is this repo's own: a guard is legitimate ONLY against a silently
     WRONG render (a wrong voice, a wrong cast, altered source text, an unowned
     ledger field). Anything else should have degraded and shipped.
   * **The one inversion:** if an OOM or a genuine resource death was CAUGHT and
     hidden -- the leg "passed" with a quietly degraded render and no loud log --
     that IS worth flagging, for the opposite reason. He wants OOM to be visible.
   * **THE RUNNER RETURNS 1 FOR THREE DIFFERENT THINGS. Do not map `rc != 0` to
     FAIL.** `scripts/otr_canonical_api_run.py` collapses every non-SUCCESS into
     `return 1`, so the exit code alone cannot tell these apart -- read the printed
     lines, which say which one happened:
     | what the log says | what it is | verdict |
     |---|---|---|
     | `PREFLIGHT FAIL: ... the running server cannot see: <files>` | never started; the weights are absent from the roots this server booted with | **NOT A FAILED LEG** -- report it as "could not start, missing weights", name the files, and move to the next leg |
     | `RESULT TIMEOUT ... BUT THE RENDER IS STILL ALIVE` | this process stopped WATCHING; the server is still rendering and should still publish | **NOT A FAILED LEG** -- say so, re-run with `--timeout 0`, and check `otr/obs/` later |
     | `RESULT TIMEOUT ... the queue is EMPTY` or an uncaught traceback or an OOM | the render died | **FAILED** |
     A leg that publishes to `otr/obs/` passed even if this process already gave up
     watching it. The artifact on disk outranks the exit code.

1. **PULL FIRST, and report the HEAD you actually ran.** A lane that cannot state its
   commit has not qualified anything.
   ```
   On the 4060 (Windows PowerShell -- `&&` is a PARSE ERROR there, and this is
   the first command you run):
       git fetch origin v2.0-alpha; git pull --rebase origin v2.0-alpha; git rev-parse --short HEAD
   On the Mac (bash/zsh):
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
   than five minutes with nothing there -- NO. **The five-minute rule is a STALLED
   HEARTBEAT, not elapsed time, and reading it the other way aborts every leg in
   this wave at minute six.** A canonical episode takes 22 to 121 minutes (the
   4060's own measured canonical is 121 min, `4060_DRILL_LOG.md:1087`) and NOTHING
   reaches `otr/obs/` until the very end, because obs is the last step after the
   mux. What must advance is the leg log's `[soak] t=<N>s` heartbeat, which is what
   `scripts/otr_render_watchdog.ps1` watches (`-StallSeconds`, default 300).
   Heartbeat advancing = alive, leave it alone however long it takes. Frozen for
   five minutes, or `:8000/queue` down = go read the leg log
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

PROFILE for the legs that must land: `otr_4060_12b_gguf_offload`, status
"shipping" -- the same 12B Q4_K_M at `gguf_n_ctx: 4096` with a full 48-layer offload,
measured 7.8 of 8.2 GB on your card. 4,096 holds any bank's prompt with room to answer,
so run the required legs there and get them banked first.

**THEN TRY A DRAFT PROFILE ANYWAY, and this correction is the operator's
(2026-09-11).** An earlier draft of this plan said "DO NOT use `8gb_lite` or
`otr_4060_floor`" and called them an evening burnt on a diagnosed refusal. **The record
does not say that.** PBUG-20260904-05 has `8gb_lite` WRITING, RENDERING AND PUBLISHING
on `media_archive` -- RESULT SUCCESS, obs_publish OK, the mp4 in the watched folder --
and refusing only on `science_news` and `original`, whose prompts exceed its
2,048-token context. That is two banks, not a profile. Forbidding it was a forecast
dressed as a finding, and the operator's standing complaint is exactly that:
*"you kept telling me this won't work."*
  So: after the required legs, run `8gb_lite` on `media_archive`. If you have more
  evening, try it on a bank it is "supposed to" refuse.
  * **A twenty-second `GenerationContextOverflowError` is NOT a failed leg** -- it is a
    refusal, and per rule 0 a refusal that kills an episode is a DEFECT IN THE CODE.
    Report the input-token count and the `context_cap` from the log and move on; that
    pair is the measurement the fix needs.
  * A draft profile that publishes is a real result worth having, even if nothing is
    promoted on the strength of it. Status promotions are the 5080's to make.

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

Leg E -- KEEP GOING. The diagnostic legs above are the floor, not the ceiling.
With evening left, spend it on MORE PUBLISHED EPISODES rather than on more analysis:
rotate the source bank and let `visual_style` roll freely, and bank whatever lands in
`otr/obs/`. Published episodes are how the operator reads success -- *"if I see it in
obs then it's somewhat a success"* -- so five published episodes with rough edges beat
two immaculate ones plus an idle box.
  * Do not re-roll away a homely result. Publish it, note what was homely, move on.
  * Do not stop because something upstream "probably will not work here". That
    forecast is what rule 0 exists to retire. Run it; only a death fails it.
  * One line per extra episode in your phone-home: bank, rolled style, elapsed,
    obs filename. No essay.


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

Leg E -- KEEP GOING. The diagnostic legs above are the floor, not the ceiling.
With evening left, spend it on MORE PUBLISHED EPISODES rather than on more analysis:
rotate the source bank and let `visual_style` roll freely, and bank whatever lands in
`otr/obs/`. Published episodes are how the operator reads success -- *"if I see it in
obs then it's somewhat a success"* -- so five published episodes with rough edges beat
two immaculate ones plus an idle box.
  * Do not re-roll away a homely result. Publish it, note what was homely, move on.
  * Do not stop because something upstream "probably will not work here". That
    forecast is what rule 0 exists to retire. Run it; only a death fails it.
  * One line per extra episode in your phone-home: bank, rolled style, elapsed,
    obs filename. No essay.


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
