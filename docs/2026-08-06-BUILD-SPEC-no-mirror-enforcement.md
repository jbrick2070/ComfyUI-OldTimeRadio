# BUILD SPEC -- no-mirror enforcement (CODE-READY)

**Date:** 2026-08-06. **HEAD at spec time:** `1759afdd` on `v2.0-alpha`.
**Arc:** `kibitz-plugin:kibitz` r1-r4 complete. **7 external calls** (r1-r3 with
Codex + Antigravity; **r4 single-lane -- Antigravity failed on provider quota**).
Artifacts: `kibitz-runs/2026-08-06-2026-08-06-no-mirror-matrix/` (**LOCAL ONLY,
`kibitz-runs/` is gitignored**).

**Operator ruling (2026-08-06):** *"there is no mirror or ping pong unless for
credits."* The closing loop is CONFIRMED OK.

**STATIC.** Nothing here enters `docs/PROD_BUG_LOG.md` without a retained
artifact.

---

## 1. WHAT IS ALREADY TRUE (do not "fix" this)

The rule is enforced in production and fails LOUD:

| surface | state |
|---|---|
| `wrapper_bridge.extend_frames_to_target` | DELETED (tombstone `:499`) |
| `eng_wan_ti2v` ping-pong | REFUSES, single-clip and coverage-planned |
| `eng_ltx_8gb` mirror-extend | REMOVED; trims REAL frames only |
| composite `_should_loop_fill` | no-op returning False |
| composite clip underrun | RAISES `ClipUnderrunsItsBeat`, any shortfall |
| `eng_ltx_video` boomerang | **DEAD, proven by execution** |

`_loop_via_reverse()` returns False even with `OTR_LTX_LOOP_VIA_REVERSE=on`, and
`_loop_fill_allowed` gates on it. The machinery is still in the tree -- that is a
DELETION task, **not an enforcement hole.**

## 2. BUILD ORDER -- producer-first

An earlier draft put the grader first. That is an armed consumer with a starved
producer, the class PBUG-20260805-04 named.

### Step 1 -- receipts on ELEVEN code surfaces

Add `native_frame_count` (always the **EMITTED** count `n`, never a pre-trim
decoded length) and `extension_mode` to:

**Local:**
1. `eng_humo.py` -- ONE stamp in the shared raw return covers all four
   registered subclasses (`:227`, `:982`, `:1052`, `:1081`), plus `_clip_from_raw`.
2. `eng_ltx_video.py` -- the ordinary raw return.
3. `eng_ltx_video.py` -- the HQ raw return.
4. **`eng_ltx_video._clip_from_raw` (`:1578-1619`)** -- the second producer seam.
   Without it the other two die before `render_beat_coverage`. This is exactly
   how `eng_ltx_8gb` lost its receipt originally.
5. `eng_ltx_av.py` raw return (`:1098-1101`) -- registered lane is `ltx_audio_in`.
6. `eng_ltx_av._clip_from_raw` (`:1183-1200`).

**Provider (all currently at ZERO references):**
7. `_CloudVideoBase.canonicalize` (`eng_cloud_video.py:489-503`)
8. `GoogleOmniVideoEngine.canonicalize` (`eng_google_omni_video.py:434-473`)
9. `GoogleVeoVideoEngine.canonicalize` (`eng_google_veo_video.py:592-631`)
10. `_GoogleVidSfxBase._canonical_video_and_sfx` (`eng_google_vid_sfx.py:397`)

**And 11 -- retrofit `extension_mode="none"` onto every remaining
video-producing adapter**, so the v1 contract in step 3 does not break the
procedural/unbounded lanes.

First: add the optional receipt fields to `CanonicalClip`
(`schemas.py:216-247`), or the declared `extra="forbid"` interface rejects the
enriched shape the day validation arrives.

### Step 2 -- the manifest

* **`frame_receipt_version: 1`** at top level.
* **`closing_theme_frame_window = {"start": S, "end": E}`**, derived from ledger
  rows with `speaker_role == "music_close"` AND `start_s_space == "master_mix"`.
  Accept only: finite non-bool `start_s >= 0`, finite `dur_s > 0`, one
  `music_cue_id`, complete unique chunk indices, contiguous/overlapping
  intervals. Convert with the composite's own convention --
  `round(start_s*fps)` for the start, `ceil((start_s+dur_s)*fps)` for the
  terminal boundary. **Emit nothing unless the manifest is POSITIONED.**
  (`beat_id` is a source LINE id; `music_close` is a speaker_role stamped
  separately at `scene_sequencer.py:1733-1799`; `production_ledger.py:1716`
  already discriminates on `start_s_space != "master_mix"`.)
* **Centralized index**: one total, non-throwing helper replacing the
  overwriting dict comprehensions at `acceptance.py:353-355` and `:418-420`.

### Step 3 -- the grader

* `KNOWN_EXTENSION_MODES = ("none", "ping_pong")` for PARSING legacy receipts;
  `DELIVERABLE_EXTENSION_MODES = ("none",)` for what may ship.
* New `RULE_NO_MIRROR` / `grade_no_mirror(ledger, manifest)`: any delivered row
  whose `extension_mode` is a non-empty string outside
  `DELIVERABLE_EXTENSION_MODES` is a finding. **All beats.** No counts, no
  projection, no segment gate.
* **That branch LEAVES `grade_multiclip_honesty`**, which keeps projection and
  count validation only, and must **stop policy grading** on a known
  non-deliverable mode -- otherwise its `delivered_native != delivered` branch
  (`:488-508`) emits a second, wrongly-worded finding for a violation
  `grade_no_mirror` already owns. It MAY still report an UNKNOWN mode as
  malformed accounting; the tests must pin that distinction.
* New `RULE_MANIFEST_SHAPE`: `clips` must be a list/tuple of dicts with unique
  non-empty string `shot_id`s. On a shape defect, emit deterministic shape
  findings, KEEP `grade_frozen_route` (manifest-independent), and SKIP the
  manifest-dependent semantic rules so one bad row does not cascade.
* **The v1 contract:** in a `frame_receipt_version: 1` manifest, every delivered
  `type == "video"` row must declare the string `extension_mode == "none"`;
  BOUNDED engines additionally owe native-count evidence. Unversioned manifests
  may omit the field, but an explicit non-`none` is still rejected. Without this
  version, "reject explicit non-`none` only" cannot catch an adapter that
  REGRESSES by dropping its receipt -- absence would stay permanently legal.
* Export every new rule in `__all__` and add it to `grade_episode`.

### Step 4 -- the composite classifier (OWN COMMIT; the only behaviour change)

`otr_silent_composite.py:444-469` currently loops the last clip for ANY
unexplained tail. Authorize `loop=True` **only** when
`S <= cursor < target_total_frames <= E` from step 2's window. Otherwise
floor/black. **Never loop on doubt.**

**Blast radius, state it in the commit:** any episode whose tail is not provably
the closing window gets floor/black instead of a loop.

A named constant alone was rejected twice: vocabulary without a manifest-backed
window is not enforcement.

### Step 5 -- delete the dead boomerang

Atomically: `_boomerang_frames`, `_ltx_loop_source_length`,
`_LTX_LOOP_MIN_DECODE_FRAMES_DEFAULT`, `_LOOP_VIA_REVERSE_DEFAULT`,
`_loop_via_reverse`, `_loop_fill_allowed`, both render/HQ branches, both
`ltx_loop_via_reverse` raw fields, and the matching assertions in
`test_ltx_boomerang.py`, `test_video_motion_forward.py`,
`test_session_ctx_ownership.py`.

**CONVERT `test_ltx_boomerang.py`, do not delete it** -- keep a tripwire proving
no environment value or session state can restore the mirror.

### Step 6 -- the fossil sweep

Mechanical search for `boomerang`, `ping_pong`, `mirror`, `loop_via_reverse`,
`stream_loop`; classify each hit as sanctioned closing reuse, historical past
tense, or defect. Known:

* `frame_contract.py:280-286` -- cites the DELETED `extend_frames_to_target` and
  governs `PLANNING_CAP_ENGINES` membership;
* `eng_ltx_8gb.py:32-36,1404-1414`; `beat_session.py:143-154`;
  `wrapper_bridge.py:525-526`; `eng_wan_ti2v.py:1001-1017`;
* **`scene_sequencer.py:1831-1842` -- an INFO log printed into the operator's
  LIVE RUN asserting the boomerang "doubles the rendered half-clip back to full
  audio duration".** A stale comment misleads a reader; a stale log misleads the
  operator mid-render.

## 3. TESTS

Per-adapter conformance for all eleven surfaces (raw AND canonicalize).
Table-driven grader tests: `"none"`, `"ping_pong"`, unknown strings, `None`,
malformed types, missing rows, duplicate rows -- for single- AND multi-segment
plans. Closing-window authorization: in-window loops, out-of-window uses
floor/black, malformed window uses floor/black. Do NOT duplicate
`test_clip_fill.py:346-414`, which already covers ordinary underruns.

## 4. SCOPE DECLARATIONS

* **No workflow topology change**, verified: canonical link 261 already carries
  node 92 `clip_manifest_json` to node 84 `OTR_SilentComposite`. The manifest
  SHAPE changes; the STRING wire does not.
* The grader stays **AUDIT-ONLY**; production enforcement is the adapter and
  composite refusals.

## 5. THE GATE, AND THE PUSH ORDER

**Reviews happen BEFORE a chunk is green, not as a hold on green commits.**
CLAUDE.md section 7 requires every green chunk pushed immediately. So per chunk:
focused tests -> Sonnet QA on the diff -> Fable gate where the change is
structural -> full Windows suite (baseline **8836 passed / 131 skipped /
1 xfailed at `e499b7fc`**) -> Bug Bible 17 -> commit -> **push** -> verify
`HEAD == origin/v2.0-alpha`, AST-parse touched Python, no BOM, no zero-byte.

## 6. THE LIVE LEG -- DISCHARGED 2026-08-07 (and why the obvious one would not count)

**RAN AND PASSED.** `signal_lost_midnights_toll_20260807_085918`, profile
`otr_w45_ltx_video`, 120 words, `Prompt executed in 00:47:43`, `RESULT SUCCESS`
+ `obs_publish OK`. Seven `ltx_video` beats; `music_opening` (250 visible
frames) and `music_closing` (200) each exceeded the 169 ceiling and chained into
TWO segments, with the server log showing the real per-segment audio slices
(`segment 0/2` at +6.760s, `segment 1/2` at +1.280s with a trimmed tail). All
beat receipts and all nine segment receipts `extension_mode="none"`;
`scripts/grade_episode.py` `ACCEPTED: 7 shot(s)` exit 0. The retired switches
were verified PRESENT in the render server's environment block
(`OTR_LTX_LOOP_VIA_REVERSE='on'`, `OTR_LTX_LOOP_MIN_DECODE_FRAMES='97'`) with
`OTR_LTX_MAX_FRAMES` / `OTR_LTX_MIN_DECODE_FRAMES` unset -- otherwise the leg
would only have shown the mirror absent, not that the switch is inert.

The requirement below is kept as the spec of record.



A WAN or `ltx_8gb` leg would PASS while never executing the deleted
`eng_ltx_video` machinery or its newly-fixed passthrough.

**Require a canonical `ltx_video` beat EXCEEDING its 169-frame contract ceiling**
(`eng_ltx_video.py:503-510`) so `render_beat_coverage` runs multiple segments,
**with the retired environment switch SET**. Then: per-segment and beat receipts
all `"none"`, an empty `scripts/grade_episode.py` verdict, `obs_publish OK`, and
the canonical episode/OBS assets on disk. Provider paths get stubbed transport
conformance -- no paid calls, offline-first.

**That single leg also discharges F11** (the mirror deletion has never had a live
proof) and the multi-clip receipt work's outstanding live proof.

## 7. FINAL-GATE FINDINGS -- RESOLVED 2026-08-06, before any code was typed

Found by the Fable co-judge at `65bd6705`, AFTER r4. All three are now answered:
**7.1 is CUT by the operator, 7.3 is DECIDED, 7.4 is VERIFIED.** 7.2 stands and
is built with step 3.

### 7.1 CUT (operator ruling 2026-08-06) -- there is no SFX in production

Operator, asked the question below directly: *"we don't have any sfx in prod, it
was ripped out, I don't want to do sfx now."* **Do not build it, and do not
"tidy" the SFX path in passing.**

Grounded before the ruling was applied: `workflows/otr_canonical.json` contains
**zero** matches for `sfx`, so no shipped episode reaches this code. The two
producers are cloud-only (`eng_cloud_video.py:916`,
`eng_google_vid_sfx.py:439-441`) and cloud lanes cannot run on this box, so the
defect is dormant on a lane that is itself switched off.

The finding stays recorded below, unfixed and correctly described, because it is
real and a future SFX lane will meet it. It is NOT scheduled work.

### 7.1-BIS The finding itself (recorded, NOT scheduled)

`beat_clip = dict(clip or {})` (`render_driver.py:3499`) never overwrites
`sfx_stem_path`, `sfx_duration_s` or `sfx_sha256`. All three are beat-scope
where they are consumed: `persist_episode_clips` moves the stem as the BEAT's
(`render_driver.py:4442-4457`), `build_clip_manifest` publishes it on the beat
row (`:4648-4652`), and `otr_master_audio_mux.py:194-209` lays it into the
master mix. Producers: `eng_cloud_video.py:916`, `eng_google_vid_sfx.py:439-441`.

**Reachable but dormant:** the `google_vid_sfx` contracts
(`eng_google_vid_sfx.py:455-460, 498-504`) are soft-continuity discrete ladders,
so a beat above the top rung takes `JOIN_JUMP` multi-segment and lands in
`render_beat_coverage` -- where the assembled beat's SFX receipt names only the
FINAL segment's stem, hash and duration. Cloud lanes cannot run on this box, so
this is the same dormancy class the `ltx_8gb` receipt had before `e499b7fc`
fixed it.

**THE DECISION THIS OWED:** for a multi-segment beat, is the honest value `None`
(fail-closed -- the beat has no single stem, and a partial one misrepresents it,
at the cost of losing that beat's SFX) or a MERGED stem (correct, but new work in
the assembler)? Publishing the last segment's stem is wrong either way.
**ANSWERED BY 7.1 ABOVE: neither, because the lane is cut.** If SFX is ever
revived, whichever is chosen must be written UNCONDITIONALLY like the other
beat-scope fields -- that is the trap this whole change exists to close, and this
is its fifth, sixth and seventh instance.

### 7.2 The grader crashes with the WRONG EXIT CODE on malformed rows

Probe-proven: a non-dict inside `manifest["clips"]` (`acceptance.py:361-362`,
`:429-430`), a non-dict inside `video.shots` (`:76`, `:365`, `:432`), or a
`coverage_plan` that is a list (`:102`) all raise `AttributeError`. Through
`scripts/grade_episode.py` these escape as tracebacks with **exit 1 -- the code
reserved for "graded, findings found"** (`grade_episode.py:112-116`), not exit 2
("could not grade"). **Automation keying on exit codes reads a crashed grader as
a graded episode.**

Section 3's `RULE_MANIFEST_SHAPE` closes the MANIFEST half. **The LEDGER half is
scheduled nowhere** -- and "KEEP `grade_frozen_route`, it is manifest-independent"
still crashes on a malformed ledger shot row. Both halves, plus a top-level
`except` in the durable script that exits 2 rather than 1.

### 7.3 DECIDED -- boundedness is a ShotLock-minted per-shot ledger stamp

Step 3's v1 contract says "BOUNDED engines additionally owe native-count
evidence", and the grader may import nothing and query no live state, so the fact
has to reach it as DATA. **The call, made 2026-08-06 before any code was typed:
candidate (b) -- ShotLock stamps `shot["frame_bounded"]` at freeze time, derived
from `frame_contract.can_split()`, and `acceptance.py` reads only that field.**

Why, and why the other two are not merely worse but wrong:

* **`can_split()` IS the boundedness predicate, already.** It is literally
  `bool(contract.max_frames or contract.discrete_frames)`
  (`frame_contract.py:248-259`) -- "does this engine have a ceiling to exceed" --
  and `tests/test_frame_contract.py:75` already pins it under that name. Minting
  the stamp from it means the planner and the grader cannot drift about what
  "bounded" means, because there is one definition.
* **It is CLOCK-DOMAIN CORRECT, which is this module's first stated refusal.**
  `acceptance.py`'s own docstring refuses live routing state because "the
  environment has moved on, so a disagreement would report the grader's clock
  rather than the episode's". An adapter's `frame_contract` is exactly such state:
  re-grading a January episode in June must judge it by the contract it was
  rendered under. A stamp records that; a lookup cannot.
* **Candidate (a), `coverage_plan` presence, does not discriminate AT ALL** --
  worse than the "under-covers" this section guessed. `_stamp_coverage_plan`
  (`otr_shot_lock.py:1319`) stamps a plan for EVERY registered engine with
  `target >= 1`, and `partition_beat` returns a one-segment `JOIN_SINGLE` plan for
  an unbounded contract (`coverage_plan.py:313-317`). Every shot carries the key,
  so the test would call the visualizers bounded.
* **Candidate (c), an allowlist inside `acceptance.py`, is a STARVED CONSUMER** --
  register a new bounded engine, forget the list, and the grader silently exempts
  the one engine it was added to catch. That is the same drift class as the
  hand-typed numbers retired on 2026-08-06, and it also puts engine names in a
  module whose whole discipline is that it knows none.

**Producer-first, per section 2:** the ShotLock stamp is itself a producer, so it
lands in step 2 with the other documents -- BEFORE the step 3 rule that reads it.
Absence means "not stated", never "unbounded": a legacy ledger with no
`frame_bounded` key owes no native-count evidence, and only an explicit `true`
demands it.

### 7.4 VERIFIED -- the re-pinned cites are correct at HEAD

This spec was written at `1759afdd`. Re-checked against `acceptance.py` at
`90124cb6`: the cites `:353-355`, `:418-420`, `:488-508` are indeed now
**`:361-362`** (the `grade_delivered` row index), **`:429-430`** (the
`grade_multiclip_honesty` row index) and **`:515-520`** (the
`delivered_native != delivered` branch). All three land on the stated code. No
further drift; use the new numbers.
