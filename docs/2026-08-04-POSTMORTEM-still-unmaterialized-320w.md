# POSTMORTEM: still_b007/b008 unmaterialized at video dispatch (320w Shakespeare leg)

Date: 2026-08-04. Author: Claude (Cowork). Status: OPEN -- root cause narrowed
to one code region, not yet pinned; fix direction below. This document
SUPERSEDES the causal chain in the GO_FORWARD "NEWBUG (2026-08-03 22:27)"
entry, which this investigation disproved.

## Incident

- 2026-08-03 22:11 -- overnight chain leg 1: `otr_w45_still_motion`, 320 words,
  bank pinned `shakespeare`, style rolled `recur_frac`. Episode
  `signal_lost_prisoner_of_persistence_20260803_221905` (Twelfth Night 1.5).
- 22:27 -- FAILED at 16.1 min in `OTR_ImageGenDispatcher` (node 91):
  `ImageRenderError: required scene image targets missing or unmaterialized
  before video dispatch: still_b007, still_b008`. 19 of 21 scene stills
  rendered; two did not; the completion gate failed the episode closed.
- Contributing harness defect (mine, fixed same night): one failed leg unwound
  the whole chain. `tmp/_sh_overnight.ps1` now isolates each leg, retries once,
  and bails to the video phase only if the first two stills fail structurally.
- 23:05 relaunch: **11/11 legs published**, including the same profile at the
  same word count on the first attempt. The failure is stochastic, roughly
  1-in-6 at 320 words on today's sample.

## What the gate proved, and what died with the wire

The completion contract (`otr_image_gen_dispatcher.py:1104-1130`) checks every
`required_scene_targets` row for a ledger row + an on-disk file. It fired
correctly: without it these two beats would have become dark-floor video.

But the raise reports ONLY the missing object ids. The per-object reason lives
in the dispatcher's `warnings` list, which is wire-only -- it is stamped into
`ledger["images"]` at :1132, AFTER the gate raises. So the explanation was
discarded by the very error that needed it. The server log that might have held
it was then truncated by the 23:06 reboot (`_otr_soak_server_launch.cmd` line
139 redirects with `>`, which truncates). The on-disk ledger froze at audio
time (`shots: 0`, no images section). Three independent mechanisms destroyed
the evidence.

## Four hypotheses, each killed by evidence

1. **"The writer whiff strands the still" (last night's NEWBUG chain) -- DEAD.**
   The 11-pass night logged **70** "no usable directive" whiffs and every one
   of those episodes published. The template collapse guard materializes fine.
2. **"The cast-time deferral warnings mark the anomaly" -- DEAD.** The passing
   night logged **69** deferral lines -- one per shot, every beat, every leg
   (`b001, b002, b003, music_opening_001...`). `DeferredImageGapError` at
   `otr_shot_lock.py:1000-1007` is normal operation for scene-init lanes whose
   image phase runs later. Last night's grep filtered on `b007|b008`, saw both
   warnings, and pattern-matched a story. The base rate refutes it.
3. **"Pathy dialogue tripped `_assert_not_path`" -- DEAD ON INPUT.** Every
   spoken line in the failed episode's ledger is free of `\`, `/`, and
   image-extension tails.
4. **"The `recur_frac` style tail is pathy" -- DEAD.** All four tails in
   `nodes/visual_styles/recur_frac.json` are clean prose.

## The surviving mechanism space (branch algebra)

Both objects PRINTED the resolve line (`resolve: object=still_b007 ...
engine=z_image_turbo`), so they entered the loop with an engine. From
`otr_image_gen_dispatcher.py:823-1098`, a resolved object with an engine has
exactly TWO exits that leave no row without raising:

- `:880` no engine selected -- excluded; resolve printed `z_image_turbo`.
- `:893-896` `_assert_not_path(prompt)` ValueError -> wire-only warning ->
  `continue`. **The only branch left standing.**

Every other failure path RAISES naming the object (lease timeout, handoff
timeout, engine unusable, adapter unusable, gen_fn None, render exception),
and none did; a completed render always appends a row; a cache hit appends a
row (a dead cache file falls through to a fresh render, LOUD).

The tension: my reconstruction of the whiffed-beat template prompt --
`_subject_anchor("")` = "face visible, speaking to camera" (clean), empty
appearance, clean dialogue, clean style tails -- contains nothing
`_assert_not_path` refuses. So either the ACTUAL composed prompt differs from
the reconstruction (`otr_meta_brief_image_prompt.py` is "the image-side
MIRROR" of ShotLock's derivation -- it imports the same machinery and shares
the `OTR_ShotLock` logger tag, so even the whiff site is ambiguous between
nodes 89 and 90), or the loop has an exit not on this map. Guessing further
without instrumentation is the third-solo-swing pattern; stop here.

## Fix direction (in order; D1 before anything)

**D1 -- observability, three small changes, no behavior change:**
  a. The completion-gate raise carries its evidence: for each missing target,
     whether a row existed (no-row vs dead-path) and every accumulated
     warning mentioning that object id.
  b. The two silent-skip branches (`:881`, `:895`) `log.warning` to the server
     log at skip time, not only `warnings.append` to the wire.
  c. `_otr_soak_server_launch.cmd` rotates the server log (timestamp suffix)
     instead of `>` truncation, so a reboot stops destroying the prior run's
     evidence.

**D2 -- reproduce cheaply.** 320-word still legs, bank `shakespeare`,
  ~1-in-6. With D1 in place the next occurrence names its own branch and its
  own prompt. No speculative fix before that.

**D3 -- fix the named branch at the root.** If it is `_assert_not_path`, the
  fix belongs where the path-like content ENTERS the prompt -- not loosening
  the guard, which is a real defense (prompt/path socket crossing).

**What NOT to do:**
- Do not build "the collapse guard mints a still" (last night's candidate) --
  its premise is dead (70 passing whiffs).
- Do not revive the portrait-init fallback; the no-fallback rip is deliberate
  and correct (`2026-06-18`).
- Do not weaken the completion gate. It did its job.

## Adjacent real defects found en route (each separate, none the cause)

- **`wan_i2v` was never enabled by the sweep boot** -- `_otr_w45_boot.ps1`
  sets `OTR_ENABLE_WAN_TI2V` but not `OTR_ENABLE_WAN_I2V`/ckpt, so its 30w
  "failure" was `wan_i2v not installed`. Harness gap, engine innocent.
- **The four 30w sweep failures are FOUR different causes**, not one:
  `ltx_audio_in` = rolled-bank fable2 script pass exhausted its markup ladder
  (story-side); `wan_ti2v` = freeze cascade `needs_full_rerun` (story-side;
  section 8 "production-proven" stands); `wan_i2v` = boot gap above;
  `viz_mxc_cpu` = no exception_message at all (still uncharacterized). The
  campaign verdict string "no new file in otr/obs" collapsed all four into
  one bucket -- harness verdicts should quote the leg's exception_message.
- **Shakespeare-bank cast rows carry EMPTY `appearance`** for all three
  characters in the failed episode -- every portrait and whiff-template still
  is composed with no appearance vocabulary. Feeds the roster work.
- Boot-log truncation (D1c) and the raise discarding warnings (D1a) are
  themselves defects this incident surfaced.

## Admission

Live-verified failure (headless leg, RESULT FAIL, asset absent) -- qualifies
for `PROD_BUG_LOG.md` under the admission rule. The entry lands WITH the
confirmed branch after D1/D2, so the log records a mechanism, not a guess.
GO_FORWARD carries it as an open bug until then.
