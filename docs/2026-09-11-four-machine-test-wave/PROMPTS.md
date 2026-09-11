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

    WAVE HEAD: <filled in at freeze time -- do not start without it>


---

## Rules every lane follows (these are in all four prompts)

1. **WRITE FINDINGS TO `docs/` ONLY. DO NOT PUSH.** Operator directive 2026-09-11:
   *"only write their fixes in docs and not to push their fixes."* Commit locally if
   you like; the push is the 5080's call so four machines cannot collide on one
   branch. If you believe something must be pushed, say so in your phone-home and
   wait.
2. **Phone home in an orderly fashion** -- one message when you start (with the HEAD
   you pulled), one when each leg finishes (pass/fail + the artifact path), one at
   the end. Not a stream.
3. **Every qualifying run loads `workflows/otr_canonical.json` through
   `scripts/otr_canonical_api_run.py`.** No `--workflow` override, no
   `--replay-from`, no `partial_execution_targets`, no hand-built graph. There is
   one graph.
4. **Use `--run-label`, NEVER `--title`.** `--title` fills the `episode_title`
   widget, so your harness label becomes the on-screen TITLE CARD and the writer
   stops naming the episode. `--run-label` echoes to the console only.
5. **A leg is not complete until it publishes to `otr/obs/`.** If a leg has run more
   than five minutes with nothing in `otr/obs/`, treat it as failing and go read the
   leg log. Never move, hide, sort or clean anything out of `otr/obs/`.
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
   not, which is a free A/B for the open visual-continuity work. Report the rolled
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
`8gb_lite` or `otr_4060_floor` -- both are status "draft", and 8gb_lite is recorded
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
shared font-resolution design row (3.8) which is still open.

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
