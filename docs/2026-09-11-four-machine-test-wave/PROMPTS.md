# Four-machine test wave -- copy-paste prompts

Written 2026-09-11 for the operator's plan: **code all day, test tonight on all four
machines at once** (4060, 5080, RunPod, virtual Mac). Paste one block per machine.

Nothing here starts until the day's coding chunks are pushed and this file's
"Gate" line below is satisfied. Until then these are drafts, not instructions.

**Gate:** every lane pulls `v2.0-alpha` first and reports the HEAD it actually ran.
A lane that cannot state its HEAD has not qualified anything.

**FREEZE ONE HEAD AND QUALIFY IT.** Four machines qualify ONE hash tonight. Every
render-path push stacked on top adds another suspect to the 5080's Leg A, the one leg
that must pass -- and if the wave goes badly, a render-inert delta means the suspect
list is one fix plus the platform, which is a single bisect instead of six. So the
frozen hash is written here before the wave starts, and nothing touching the render
path lands after it:

    WAVE HEAD: 1d8d5131c7a07995b258f7527478accb0fde6136

    FROZEN 2026-09-12 by the 5080 coder window, on the operator's word ("start
    it"), and this is the first freeze that is not premature. The two earlier
    cuts -- 3e692dc5 and f5f40bd4 -- were both withdrawn because arcs were still
    deferred past them, and an ARC IS CODING. His ordering was:

        ARCS  ->  CODING  ->  SHAKESPEARE  ->  TESTING, ABSOLUTELY LAST

    That order is now satisfied. Sections 1 and 2 of the go-forward plan are
    empty, he answered all sixteen blocked rows, and everything those answers
    turned into is built, reviewed and pushed. What is left in section 3 is his
    EAR, which testing does not block on.


    THIS IS A CODE FREEZE, NOT A COMMIT FREEZE. Commits
    after the hash will be documentation only -- this file's own freeze line
    among them -- so the hash you pull will legitimately be LATER than the one
    written here. That is expected: nothing touching the render path lands
    after the frozen hash without a re-freeze, and a docs commit cannot change
    what a render produces.

    So: report the HEAD you ACTUALLY pulled, whatever it is. Never check out
    the frozen hash. If you pull a commit that touches nodes/, scripts/ or
    workflows/ AFTER the hash written on the WAVE HEAD line above, STOP and
    phone home -- that would mean the freeze was broken.


### The suspect list, written before the wave rather than after it

If a leg fails, this is what to look at first -- and the third entry is the one
no document listed until now.

1. **The scopes re-homing.** `OTR_SceneAwareScopes` stopped writing into the
   janitor-swept scratch tier and now resolves through the episode path
   authority. Rendered pixels were proven unchanged across 18 inputs, but it is
   the newest thing on the composited path.
2. **The platform itself.** Three of the four machines are not the box the code
   was written on, which is the entire point of running them.
3. **THE CRUX RANKER, ON THE 4060 SPECIFICALLY.** `resolve_crux_kernel` now
   ranks `meta.key_objects` by whether the beat's own text names one. The 4060
   profile renders on `animatediff15_v3_haunted_video`, whose engine inherits
   `GHOST_PROMPT_PROFILE` -- so the 4060's legs are the FIRST live exercise of
   that ranker, on a code path no episode has ever rendered through. It is
   offline-proven and it is still the least-observed thing in the wave.
   **4060: phone home your `kernel_source` counts** -- how many shots resolved
   `key_object_in_beat` versus plain `key_object`. That ratio is the first real
   hit-rate measurement of the deterministic tier, and it decides how the LLM
   tier gets built.

---

## Rules every lane follows (these are in all four prompts)

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
   * **IF A VALIDATOR ENDED THE RENDER, THE LEG FAILED *AND* THE VALIDATOR IS THE
     FINDING.** Record both; they are not alternatives. The run died, so the verdict
     is honest -- but the thing worth phoning home is WHICH check killed it, by file
     and line. The rule it has to answer to is this repo's own: a guard is
     legitimate ONLY against a silently WRONG render (a wrong voice, a wrong cast,
     altered source text, an unowned ledger field). Anything else should have
     degraded and shipped, and that is a code defect.
   * **DO NOT GO LOOKING FOR VALIDATORS.** Report the one that actually fired. The
     tree has ~160 deliberate `NO FALLBACK` raise sites under a standing 2026-06-18
     directive; auditing them against rule 0 would produce a pile of "defects" that
     are settled rulings, and bury the one that mattered.
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

1. **WRITE FINDINGS TO `docs/` ONLY. DO NOT PUSH.** Operator directive 2026-09-11:
   *"only write their fixes in docs and not to push their fixes."* Commit locally if
   you like; the push is the 5080's call so four machines cannot collide on one
   branch. If you believe something must be pushed, say so in your phone-home and
   wait.
   **THIS OVERRIDES `CLAUDE.md` SECTION 7 FOR THE DURATION OF THIS WAVE, and you will
   notice the conflict because you are told to read CLAUDE.md first.** Section 7 says
   every green commit is pushed immediately, "no exceptions", and that pushing to
   `v2.0-alpha` is "always safe, expected, and required". That is the standing rule and
   it resumes the moment this wave ends. Tonight four machines run the same branch
   against one frozen hash, and a push from any of them moves the hash the other three
   are qualifying. Both are operator directives; this one is dated later and is scoped
   to the wave.

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

2. **Phone home in an orderly fashion** -- one message when you start (with the HEAD
   you pulled), one when each leg finishes (pass/fail + the artifact path), one at
   the end. Not a stream.
3. **Every qualifying run loads `workflows/otr_canonical.json` through
   `scripts/otr_canonical_api_run.py`.** No `--workflow` override, no
   `--replay-from`, no `partial_execution_targets`, no hand-built graph. There is
   one graph.
   **PASS `--timeout 0` ON EVERY LEG. This is not optional and it is the single
   most likely way to manufacture a false failure tonight.** `--timeout` defaults
   to 5400 seconds -- ninety minutes -- and it is the window this PROCESS watches
   for, not a limit on the render. The 4060's own measured canonical is 121
   minutes (`4060_DRILL_LOG.md:1087`) and the Mac has legs past two hours. At
   minute 90 the default prints `RESULT TIMEOUT` on a render that is still alive
   and still going to publish. A window that then obeys rule 6 and "resets before
   every headless run" KILLS A PASSING EPISODE and starts it over. `--timeout 0`
   waits for a terminal result and the problem disappears.
4. **Use `--run-label`, NEVER `--title`.** `--title` fills the `episode_title`
   widget, so your harness label becomes the on-screen TITLE CARD and the writer
   stops naming the episode. `--run-label` echoes to the console only.
5. **A leg is not complete until it publishes to `otr/obs/`.** Never move, hide,
   sort or clean anything out of `otr/obs/`.
   **THE FIVE-MINUTE RULE IS A STALLED HEARTBEAT, NOT ELAPSED TIME, and reading it
   the other way would abort every leg in this wave at minute six.** A canonical
   episode takes 22 to 121 minutes depending on the box (the 4060's own measured
   canonical is 121 min, `4060_DRILL_LOG.md:1087`), and NOTHING reaches `otr/obs/`
   until the very end -- obs is the LAST step, after the mux. What must advance
   every five minutes is the leg log's `[soak] t=<N>s` heartbeat, which is exactly
   what `scripts/otr_render_watchdog.ps1` watches (`-StallSeconds`, default 300).
   So: heartbeat advancing = alive, leave it alone however long it takes. Heartbeat
   frozen for five minutes, or `:8000/queue` down = go read the leg log.
6. **Reset before every headless run.** Kill SELECTIVELY by CommandLine via
   `Get-CimInstance Win32_Process` (or `ps`/`pgrep` on Mac/Linux) -- never a blanket
   `Stop-Process -Name python`, which also kills the agent's own tooling. Confirm
   port 8000 is not listening and GPU memory is back to desktop baseline.
7. **Record the full receipt for every attempt**, pass or fail: source fields, code
   and canonical hashes, actual model + quantization + profile, prompt ID, elapsed
   time, memory and model loads, every repair attempt, requested vs actual acts and
   cast, ledger seals, and final asset paths. **Preserve terminal failure evidence
   before asserting anything.** Keep the full denominator -- a failed attempt is
   part of the record, not something to re-roll away.
8. **Do not edit production code while a run is active**, and do not change story
   controls to hit a time target.
9. **Name the rolled `visual_style` in every phone-home.** The visual-style roll is
   deliberately independent of the source-bank roll
   (`nodes/_otr_rolls.py::resolve_style_selection` -- *"Independent of the bank roll
   in every respect"*), so a present-day story can legitimately draw an era-bearing
   pack. With six-plus legs rolling freely tonight, some will land one and some will
   not. Visual continuity is CLOSED, not open (`f6739500`; see
   `docs/2026-09-11-visual-continuity-diagnosis/ARC_CLOSED.md`) -- the pack and the
   story are a combination and exactness is not the goal, so report the rolled style
   as evidence and do NOT open it as a bug. Report the rolled
   id; do NOT pin it, and do not re-roll to get a style you prefer.

---

## 5080 (IDREAM) -- owns the shipping surface

```
Pull v2.0-alpha and report the HEAD you ran.

Leg A -- PBUG-20260911-03 requalification. This is the one that must pass tonight.
The scopes MP4 was being written into episodes/_shared/tmp (the janitor-swept
scratch tier) with a system-temp fallback; it now writes through
otr_composited_dir(episode_id). Run ONE full canonical episode and then verify on
disk, not from the log alone:
  - the scopes file exists under output/otr/episodes/<episode_id>/composited/
  - nothing new appeared in episodes/_shared/tmp (the five stranded files from
    earlier episodes must still be there -- they are evidence, do not delete them)
  - nothing named otr_scopes_* appeared in the ambient system temp dir
  - the final episode published to otr/obs/ and the file is non-zero
Then run the consumer BOTH ways: node 93 OTR_PostUpscaleProcgenBlend with bypass
TRUE (today's canonical value) and with bypass FALSE so the scopes actually blend.
Confirm the audio and frame contracts are unchanged between them.

  ** NOT EXECUTABLE AS WRITTEN -- measured 2026-09-12, by trying it. **
  The bypass=TRUE half needs no work: True IS the canonical value
  (`widgets_values[5]`), so every leg already runs it. The FALSE half has no
  sanctioned route:
    * `otr_canonical_api_run.py --set 93.bypass=false` is REFUSED --
      "widget 'bypass' is not on the creative whitelist; managed widgets are
      patched ONLY via apply_profile_to_workflow(--profile)".
    * and `config/profiles/widget_mapping.json` carries NO entry for it, so no
      profile can set it either.
  That leaves editing `workflows/otr_canonical.json` itself, which is the
  shipping graph. So this sub-check wants a deliberate decision, not a lane:
  add a `bypass` mapping entry (default-identical, since the canonical keeps
  TRUE) and re-freeze, or drop the sub-check. It is left UNRUN rather than
  faked, and the rest of Leg A passed -- see the receipt.


Leg B -- the six-act Jeffrey/Codex fixture, exactly as specified in
docs/GO_FORWARD_PLAN.md: Jeffrey and Codex getting closer to release during one
continuous evening in Jeffrey's workspace. Jeffrey is the physically present adult;
Codex is a named AI dramatic speaker heard through the computer speakers, with only
an on-screen interface -- no embodied human, age or gender invented. Only those two
dramatic voices. Reports from other machines are displayed text or discussed by
them, never new speaking characters. The house announcer is production framing and
is OUTSIDE the dramatic cast, turns and ending. REQUEST FOUR CHARACTERS on purpose,
to exercise the flexible actual-two cast. Select six acts, normal act breaks.
Record requested vs actual acts and cast.

Findings go in docs/. Do not push. Phone home per the rules above.
```

---

## 4060 (MRKT) -- owns the portability surface

```
Pull v2.0-alpha and report the HEAD you ran.

You are the only box that can answer "does this work somewhere other than where it
was written." That is your whole value here, so test the 8 GB path, not the 16 GB one.

Leg A -- one-act monologue and three-act ensemble, full canonical through
scripts/otr_canonical_api_run.py.

USE `otr_4060_12b_gguf_offload`. That profile is status "shipping". Do NOT use
`8gb_lite` or `otr_4060_floor` for the legs that must land -- both are status
"draft". **But do try them afterwards: the operator struck the blanket forbid on
2026-09-11** ("you kept telling me this won't work"), and the record does not
support it -- PBUG-20260904-05 has `8gb_lite` publishing successfully on
`media_archive` and refusing only on the two banks whose prompts exceed its
2,048-token context. A twenty-second `GenerationContextOverflowError` is a code
defect to report with its token numbers, not a failed leg. 8gb_lite is recorded
in PBUG-20260904-05 as refusing in twenty seconds on two of the three banks, because
the 12B writer's 2,048-token context cannot hold their prompts. Picking a draft
profile would burn your evening on a known, already-diagnosed refusal.

Record the profile that actually RESOLVED (grep the leg log for the resolved-profile
line and the applied block -- do not take the requested profile as the applied one).

Leg B -- confirm the PBUG-20260911-03 scopes fix behaves on 8 GB too: the scopes
file under episodes/<episode_id>/composited/, nothing new in _shared/tmp, nothing
in system temp. If the episode directory resolves somewhere different on your
install shape, that is exactly the kind of finding this lane exists to produce --
report the actual resolved path.

Leg C -- small-canvas credits overflow is a KNOWN open row (2.4 voice/credits):
`otr_credits_roll.py` logs at ERROR that the canvas is too small for the card as
designed. If you see it, REPORT it and move on -- do not chase it, and do not treat
it as a failed leg.

Leg D -- least-friction/fresh-install observations as you go. Anything that needed a
manual step the docs do not mention belongs in docs/4060_DRILL_LOG.md.

Findings go in docs/ (4060_DRILL_LOG.md is yours). Do not push. Phone home.
```

---

## Virtual Mac -- Apple Silicon proving

```
Pull v2.0-alpha and report the HEAD you ran.

Leg A -- one full canonical episode end to end on Apple Silicon through
scripts/otr_canonical_api_run.py. The point is that the canonical graph runs at all
on this platform and publishes to otr/obs/.

Leg B -- look hard at TEXT RENDERING, because this platform has burned us there
before: eight published macOS episodes shipped with the hero title running off the
right edge of the frame, because the font MEASUREMENT side resolves a font FILE by
absolute path while the DRAWING side hands libass a family NAME, and when the
measurement side finds nothing it falls back to a bitmap default that ignores the
requested size. Check the title card, the captions and the credits roll in the
actual published frames -- not in the logs. Report what you SEE.

Leg C -- note every place a Linux/macOS box needed something the Windows path did
not (fonts installed, ffmpeg build features, PyAV limitations). This feeds the
credits-degradation proof. (Row 3.8, a shared font resolver, DISSOLVED on
2026-09-11: the four resolvers have genuinely different jobs and a unified
Python table could not close the gap anyway, because captions and titles are
drawn by libass, which never sees the Python side. Do not report against it.)

Findings go in docs/. Do not push. Phone home.
```

---

## RunPod -- rented compute, second model family

```
Pull v2.0-alpha and report the HEAD you ran. Record the actual image/revision you
rented, not just "a pod."

Leg A -- second installed model family. Run one-act, three-act and six-act full
canonical episodes with a compatible locally installed family that is NOT the
Gemma4-12B/NF4 combination the 5080 has been using. VERIFY THE ACTUAL RUNTIME MODEL
IDs from the leg log and the saved ledger -- a dropdown label is not evidence of
what loaded.

Leg B -- ONE fresh Original-bank full publication, for the credits proof that is
still outstanding: the observed creative model must agree across the wire, the saved
ledger, and the rendered credit. Original attribution and long-title containment are
already implemented in code; what is missing is live proof, so capture all three.

STOP THE POD when the legs finish. Rented compute left running is the failure mode
here.

Findings go in docs/. Do not push. Phone home.
```

---

## What this wave does NOT cover

**Listening.** The opening/middle/ending audition on at least two publications,
including a six-act, is the operator's own ear. No agent in this wave can ingest
audio, and reading the TTS text or checking a waveform is not listening -- do not
let any lane record it as such.

**Visual continuity (Sprint 2).** The clothing/room/seating drift seen in
canonical09 is diagnosed from stored evidence, not from these legs. These legs
produce fresh images that FEED that diagnosis; they do not close it.
