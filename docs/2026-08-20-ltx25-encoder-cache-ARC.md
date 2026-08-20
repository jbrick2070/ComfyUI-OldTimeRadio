# LTX 2.5 episode-scoped text-encoder cache -- the arc, and what it cost to get right

**Date:** 2026-08-20. **Branch:** `v2.0-alpha`. **Driver:** Claude (Opus 5), Cowork.

Full run artifacts live in `kibitz-runs/2026-08-20-ltx25-encoder-cache/`, which
is **gitignored** (`.gitignore:255`) -- this file is the tracked record, and it
exists because a judgment nobody can read later is not a judgment.

---

## THE PROBLEM, COUNTED ON A LIVE LEG

A canonical episode running on `otr_ltx25_high_video` was audited mid-render:

| signal | count |
|---|---|
| shot renders started | **31** |
| text-encoder GGUF disk reads | **31** |
| encoder CLIPs constructed | **31** |
| **out of memory** | **0** |

(Counted mid-render at 25/25 and re-counted on the completed leg at 31/31 --
18 beats, several of them multi-segment. The ratio never moved.)

**A 1:1 ratio.** Every shot re-read the 8.86 GiB Gemma-4 12B Q5 GGUF text
encoder from disk -- roughly 63 s per shot on top of a 54.2 s CPU encode. Wall
clock is what decides how many episodes reach `otr/obs/` in a night, so this is
the thing worth fixing.

## THE FIX

An **episode-scoped** cache of the loaded CLIP and the empty negative
conditioning, injected through `run_graph`'s pre-existing `external_results`
contract with the corresponding node dropped from the graph.

**Ownership sits in the DRIVER, not the engine, and that is the whole design.**
The registry builds each adapter once at import and returns that same instance
forever (`engine_registry_base.py:148-155`), so a cache kept on the adapter
would never end. Every ENGINE-level hook was measured and all run too often:

| candidate hook | frequency | verdict |
|---|---|---|
| `free_otr_pipeline_residue` | every SHOT (`eng_ltx25.py:889`) | destroys the cache before each use |
| `teardown` / `unload` | every BEAT (`beat_session.py:308`) | reloads per beat |
| gpu_residency lease | every BEAT | same, and it is a cross-process lock |
| **`run_episode`** | **every EPISODE** (`render_driver.py:3877`) | **correct** |

## THE PANEL EARNED ITS KEEP FOUR TIMES

**Roster, stated precisely:** Antigravity `Gemini 3.7 Flash (High)` (r1, r2, r3),
a second Antigravity lane on **Gemini 3.1 Pro** run by the operator by hand (r1),
Codex `gpt-5.6-sol` called directly (r1, r2, r3), and **Sonnet 5** as the
mandated post-coding QA on the finished diff. Four rounds, seven external
reviews. Local, $0.

**1. Codex overturned the driver's own r1 judgment.** I concluded "no
per-episode hook exists, accept process-lifetime residency" -- **false**. I had
searched only engine-level hooks and never looked one layer up at the driver.
`run_episode` is the real boundary. Episode ownership was adopted in full.

**2. A local -> cloud hand-off leaked the entire 8.86 GiB.** I nested the scope
release inside `_should_reclaim_between_engines`, which ends in
`and not _is_cloud_video_engine(this)` (`render_driver.py:1885-1888`) because a
VRAM reclaim only matters before a LOCAL engine loads. Whether to drain VRAM and
who owns a handle are different questions that happened to share a line. Caught
by Codex r2.

**3. A proposed drift checksum would have crashed on EVERY cache hit.** Two
lanes proposed guarding the cached negative with `neg[0][0].sum()`. `neg[0][0]`
is the `[tensor, dict]` ENTRY -- a list -- so that raises `AttributeError` every
time. The tensor is at `out[0][0][0]`. Caught by agy r2, in a fix another
reviewer proposed and I had already decided to adopt.

**4. A test proved the wrong mechanism.** Sonnet found that
`test_a_FAILED_episode_still_closes_the_scope` used two different engines, so
the ENGINE SWITCH closed the scope at the hand-off and the test **never
exercised the `finally` it was named for** -- green either way.

## WHAT WAS CUT, AND WHY

**The conditioning drift digest.** Adopted in r2, cut in r3 on Codex's argument:
a whole-tensor reduction cannot catch cancelling mutations, it re-ran twice a
shot to guard a hazard three independent code reads had already ruled out, and
it silently degraded to "no guard" whenever it returned `None` -- `None == None`
compares equal. **A guard that quietly stops guarding is worse than no guard,
because the receipts still claim protection.** `_copy_conditioning` keeps the
mutable outer list and metadata dicts private; the tensor is shared by
reference, exactly as every other node on this graph treats it.

## THREE FALSE PREMISES CORRECTED IN COMMITTED FILES

The 2026-08-19 window corrected "the negative is inert at CFG 1.0" in three
places and believed it was done. Codex found **three more surviving copies**:

* `tests/test_ltx25_recipe_matches_lab_golden.py` -- *"Inert at CFG 1.0, so it
  is removed rather than carried"*;
* the same file's `test_every_cfg_is_exactly_one` -- *"Above 1.0 forces batch
  size 2"*;
* `tests/test_ltx25_video_lane.py` -- *"a negative is inert at CFG 1.0 anyway"*.

All false under the locked `euler_ancestral_cfg_pp`, which forces
`disable_cfg1_optimization=True` and consumes `uncond_denoised`. The negative is
LIVE and steers every step. Both wordings invite the same wrong "optimisation":
wiring `neg` from `pos` to save a 12B encode, which would silently change every
render.

## THE ACCEPTANCE INSTRUMENT

`scripts/otr_ltx25_encoder_load_audit.py`.

It counts **the loader's own GGUF qtype line**, not the adapter's claim about
itself, because the failure mode here is SILENT: `reclaim_idle_models` runs in
`render_clip`'s `finally` on every shot and detaches every resident patcher, the
cached CLIP included. `_cached_clip_is_live` catches an unusable patcher and
degrades to a full reload -- correct, safe, and indistinguishable from success.
A cache that never hits would look exactly like a working one.

It correlates PER SCOPE -- each OPEN..CLOSE interval is judged on its own
events, unscoped diagnostic renders are reported but never gated, and an empty
scope grants nobody an allowance. **This instrument has been wrong three
separate ways** and each was caught by a review round: failing OPEN on a log
with no matched lines (r3 Codex), failing CLOSED on a healthy mixed-engine leg
(r4 agy), and letting one good scope subsidise a bad one (r4 Codex). It now
carries 22 tests to the feature's 69 -- the gate turned out to be harder to get
right than the thing it guards.

**Proven non-tautological against the pre-cache leg: 31 renders, 31 reads,
ratio 1.00, exit 1 on three separate counts.**

## TESTS

**91 new**, across four files, plus corrections to two existing ones.

Two claims were proved rather than asserted, by breaking the code and watching
the tests fail, then restoring the file and verifying it byte-identical:

* re-nesting the scope release inside the reclaim predicate -> both
  local->cloud tests FAIL (`FF`, rc=2);
* neutering the episode `finally`'s release to `pass` -> both failure-path
  tests FAIL (`FF`, rc=2).

**Full suite on a settled tree: 11237 passed / 114 skipped / 1 xfailed, exit 0.**
The baseline was 11146 / 114 / 1, so the delta is **+91 -- exactly the tests
added**, itemised: 45 cache helpers + ownership, 14 driver wiring, 10
`render_clip` paths, 22 audit gate. No regressions at any step. Bible regression
**20 / 26 / 3** unchanged at 295 entries; `build_variants.py --check`
**51 variants / 0 failures**, both baselines held.

## THE CACHE IS PROVEN ON A PUBLISHED EPISODE

Measured by `otr_ltx25_encoder_load_audit.py` on the real leg, not claimed:

```
cache scopes found        : 1
  scope 0: renders=34 reads=1 pinned=1 hits=33 misses=1 drops=0 closed=True
unscoped (diagnostic)     : renders=0 reads=0
PASS: every scope loaded the encoder at most once and reused it thereafter
```

| | before | after |
|---|---|---|
| shot renders | 31 | **34** |
| encoder disk reads | **31** | **1** |
| cache hits | n/a | **33** |
| cached encoder dropped | n/a | **0** |
| out of memory | 0 | **0** |
| leg wall clock | 7674 s | **5599 s** |
| per render | 247.5 s | **164.7 s** |

**33.5% faster per render, and the leg finished 2075 seconds sooner while
rendering THREE MORE shots.** Published:
`signal_lost_a_midsummer_nights_quarrel_20260820_024524` -- 1920x1080, h264 +
aac, 109.4 s, 51 MB. `RESULT SUCCESS`. `otr/obs/` 81 -> 82.

**WHAT THIS SETTLES, and it was the one real risk.** The r4 Codex lane objected
that a structurally-live-but-unusable cached CLIP would take the HIT path and
then raise inside `run_graph` -- a dead render, not a slow one -- and that the
whole design rested on `detach(unpatch_all=True)` leaving a CPU-pinned GGUF CLIP
re-encodable, which three code reads asserted and **no GPU had confirmed**.
33 consecutive hits with zero placement drops confirms it. That objection is
now answered by measurement rather than by reading.

**Caveat stated precisely:** the leg booted BEFORE the r4 ownership fixes
(generation token, lock, per-scope audit). It therefore proves the CORE
mechanism -- a reclaimed CLIP survives and is reused across an entire episode --
and not the ownership edge cases, which are concurrency and kill-switch paths a
single sequential episode cannot exercise. That is the right priority: the core
mechanism is the part no CPU test can reach.

## WHAT WAS PROVEN EARLIER IN THE SAME SESSION

**PROVEN -- the CPU-encoder fix from the previous window, end to end.** The
canonical leg that was mid-render when this session opened completed and
published:

* `signal_lost_beneath_the_silvery_boughs_20260820_002734_silent_procgen_blended_captioned_with_credits_final.mp4`
* **31 shot renders across 18 beats, ZERO out-of-memory.** The two legs before
  it died at beat 15 and beat 1.
* ffprobe on the published bytes: 1920x1080, h264 + aac, 25 fps, 100.52 s,
  61 MB. `otr/obs/` went 80 -> 81.

**Both halves are now proven on published episodes.** The safety argument that
justified pushing ahead of the proof still holds and is worth keeping: every
cache miss falls through to exactly the pre-cache path, so the failure mode was
always "no faster than before", never "wrong" -- and
`OTR_LTX25_ENCODER_CACHE=0` disables the mechanism without a code change.

## A PROCESS LESSON WORTH KEEPING

Sonnet reported the files changing underneath it mid-review. **There was no
second window -- it was me**, applying r2's fixes while the QA pass read the
tree. The "one coder window" rule was not broken, but its spirit was: a QA pass
should read a FROZEN tree, or it spends its budget on a moving target and
reports findings that are already fixed. **Snapshot the diff, freeze edits, then
run QA.**
