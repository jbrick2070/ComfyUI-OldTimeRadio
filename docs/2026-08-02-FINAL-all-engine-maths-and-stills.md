# FINAL DECISION: maths + still logic, every video model, local and cloud

**Operator, 2026-08-02:** "codex and Fable for final decisions, code and test.
Be sure it includes a review of all maths and still logic for all video models,
local and global. Then run a 30-45 word randomizer test on each video model."

**REVISED 2026-08-06 to best practice, on operator instruction.** What changed
and why is section 0. The still-logic analysis, the cloud narrative and the fix
list are preserved; the hand-copied number tables are not, because they had
already started lying.

**Do not launch renders or boot a server** -- the campaign runs after the ruling.

---

## 0. HOW TO READ THIS DOCUMENT (revised 2026-08-06)

### 0.1 This doc does NOT hold live numbers any more

**The single source of truth for every per-model number is
`docs/ENGINE_MATRIX.md`**, which is GENERATED from the live engine registry by
`tools/engine_matrix.py` and DRIFT-GATED (`--check`, wired as a suite test in
`tests/test_engine_matrix_doc.py`). It cannot disagree with the adapters without
the suite failing.

**Why this changed, with the receipt.** This document previously carried its own
hand-typed coverage table. Four days later it was WRONG, and the generated
matrix was right:

| model | this doc said (2026-08-02) | live registry says (verified 2026-08-06) |
|---|---|---|
| `humo (portrait)` | `33-177/q4`, jump, **3 segments** | jump, **5**: 97, 97, 97, 97, 57 |
| `humo_14B_169` | `33-49/q4`, jump, **10** = `[49]x7 + [33]x3` | jump, **5**: 97, 97, 97, 97, 57 |

The cause is `_HUMO_14B_SAFE_RENDER_FRAMES = 97` (`eng_humo.py:106`), which moved
after this doc was written. The doc's own headline finding -- "TEN clips ... NINE
freshly minted stills" -- described a ceiling that no longer exists, and
`tools/engine_matrix.py --check` passes today, so the generated table is the
truthful one.

**Best-practice rule this establishes: a hand-maintained document must never
re-type a number a generated one already owns.** Cite the generated matrix
instead. That is the whole reason the drift gate was built.

### 0.2 What this document is FOR

Only the things a generator cannot derive:

* the **still logic** analysis and the local/cloud re-mint split (section 3);
* the **fix list** with its status and evidence (section 4);
* the **open decisions** that need an operator or a panel (section 5);
* the **campaign** plan (section 6);
* the **padding rule** as it currently stands (section 7).

### 0.3 Every fix item now carries a STATUS

Previously F1..F12 were an undifferentiated list, so a FIXED blocking-safety item
and a still-open one read identically. Each item now carries:

* **DONE** -- verified against the code, with the file:line that closes it.
* **PARTIAL** -- some of it landed; what remains is named.
* **OPEN** -- verified still true.
* **UNVERIFIED** -- not re-checked on 2026-08-06. **An unverified item is not a
  safe item; it is an unread one.** Marking it so is the point.

**Status verified 2026-08-06 against HEAD `e499b7fc`.** Re-verify before relying
on any of it -- and move the stamp when you do.

---

## 1. LOCAL AND CLOUD MATHS -- see the generated matrix

The hand-typed tables were deleted on 2026-08-06 (see 0.1). Read instead:

* `docs/ENGINE_MATRIX.md` -- clip window, ladder, continuity, join mode, segment
  counts at 442 frames, effective canvas and re-mint counts, for every registered
  engine, local and cloud.
* Regenerate: `python tools/engine_matrix.py`
* Verify: `python tools/engine_matrix.py --check`

What survives, because it is ANALYSIS rather than a number, and both remain true:

* **Visible totals are EXACT on every engine, local and cloud.** No drift at any
  tested beat length. The coverage arithmetic itself is sound -- that was the
  first thing to check and it held.
* **Rendering more than you show is NORMAL and is not padding.** `ltx_video`
  renders 507 frames to show 442; `ltx_audio_in` renders 449 to show 442;
  `google_veo_video` renders 450 to show 442. The surplus is a legal tail trim of
  REAL frames, not manufactured ones. Anything that compares "rendered" against
  "delivered" and calls the difference a defect is asking the wrong question --
  which is precisely what `nodes/_otr_video_engines/acceptance.py` did until
  2026-08-06 (`e499b7fc`), reporting every honest multi-segment beat as padded.

## 2. CLOUD ENGINES

Numbers: see the generated matrix. The analysis that is not a number:
`google_veo_video` renders 450 and shows 442 (8-frame tail trim), and visible
totals are exact on every cloud engine.

## 3. STILL LOGIC -- the local/cloud split is the headline

**LOCAL: no engine re-mints a per-segment still.** The re-mint path needs JUMP
*and* a still-consuming lane; locally nothing is both. CHAIN engines overwrite
`asset_refs["init_image"]` with the predecessor's real terminal frame; the four
`humo` variants are JUMP but `audio_driven_face` consumes no scene still, so all
segments share ONE portrait and identity holds by construction. What resets on
humo is POSE, not identity.

**CLOUD: eleven of twelve engines re-mint a still per cut.** They are the live
consumers of `otr_image_gen_dispatcher.py:650-690`, whose clone DELIBERATELY
drops the fixed seed so each segment gets a different image
(`hash(request_seed:object_id:prompt_hash)`, and `object_id` carries the segment
index). Its own comment accepts the tradeoff: "what a bookend loses is only the
shared canonical LOOK across its own segments, which is what cutting means."

**That reasoning was written for a SCENE bookend. It is now governing CHARACTER
beats on every cloud lane.** There is no identity conditioning anywhere in the
local lanes; `reference_images` exists only on the cloud engines. So a cloud
character beat over one segment may change the character's face at each cut.
**This is the still-continuity defect, and it lives in cloud, not local.**

**Open contradiction, flagged 2026-08-06, unresolved:** this document's original
tables recorded `humo` as `stills_minted=2`; the generated matrix's re-mint
column reports **0** for every humo row. The generated number goes through
`_lane_consumes_a_still` -> `_still_spine_requires_scene`, and
`otr_shot_lock.py:1132-1134` says a portrait-only face lane owes no per-segment
stills. They cannot both be current truth, and the disagreement is about exactly
the face-identity-per-cut question this section calls its most important finding.
**UNVERIFIED which is right.**

## 4. THE FIX LIST -- with status, verified 2026-08-06 against `e499b7fc`

**F1 -- WIRE THE ADMISSION GUARD. STATUS: DONE.**
Was: `assert_frame_affordable` (`motion_common.py`) had ZERO call sites while
`PLANNING_CAP_ENGINES` contained three engines, so every coverage-planned segment
rendered with no preflight VRAM check.
Now: called at `render_driver.py:3133`, inside `_assert_beat_affordable`
(`:3061`), which `render_beat_coverage` invokes at `:3303` -- BEFORE
`BeatSession` opens, which is where this item asked for it. The verdict is
stamped on the beat clip as `vram_admission`.

**F2 -- THE HUMO CAP CITES MISSING EVIDENCE. STATUS: PARTIAL.**
The ASYMMETRY is resolved: both 14B variants now sit at
`_HUMO_14B_SAFE_RENDER_FRAMES = 97` (`eng_humo.py:106`), so the "49 wide vs 177
portrait at identical pixel count" contradiction is gone.
**Still open:** the justification still cites `docs/2026-06-27-humo-bakeoff`, and
**that document still does not exist in this repo** (verified 2026-08-06). A cap
whose evidence cannot be read is a number nobody can re-derive.

**F3 -- ONE CAP AUTHORITY. STATUS: UNVERIFIED.**
Declare ledger-stamped `video.max_render_frames` the sole production authority;
require the `launch.env` twin absent-or-equal before planning; `render.frame_budget`
diagnostic-only. Not re-checked 2026-08-06.

**F4 -- `fastwan_8gb` 81 -> 65, or requalify. STATUS: OPEN.**
`max_render_frames: 81` is still pinned in `otr_8gb_fastwan.json`,
`otr_g4_fastwan.json`, `otr_w45_fastwan.json` and `otr_g4_wan_ti2v.json`
(verified 2026-08-06). The original objection stands: 81 was promoted from a
bench cell, and a bench cell never qualifies an engine.

**F5 -- `wan_i2v` must declare a canvas. STATUS: LIKELY DONE, VERIFY.**
The generated matrix now reports `wan_i2v` as `canvas-negotiated (_aspect_plan)`
rather than falling to the shared 1472x832 default. Confirm against the
effective-canvas section before closing this.

**F6 -- Cloud engines declare no canvas. STATUS: UNVERIFIED.**

**F7 -- BUILD THE LIP-SYNC ONSET PAD. STATUS: UNVERIFIED.**
`BUG_BIBLE.yaml` BUG-LOCAL-102: HuMo audio leads the lips by 100-200 ms; the
prescribed fix (pre-pad leading silence, drop the pad frames after decode, stamp
the value in the ledger) was never built as of 2026-08-02. **Note the interaction
with the corrected maths:** it repeats at every cut, and humo is FIVE segments,
not the three this document originally assumed.

**F8 -- Stale rationale in `mouth_policy.py`. STATUS: DONE.**
It no longer claims "the same character is regenerated mid-line from a different
seed". The current text records the correction explicitly -- "(This clause used
to say 'from a different seed', which was not true...)" -- which is the right
shape: the fix and its reason both survive.

**F9 -- Stale mirror-extend comment in `eng_humo.py`. STATUS: DONE.**
`eng_humo.py:61` no longer says beats over the cap "render at the cap then
mirror-extend to the audio target". Verified 2026-08-06.

**F10 -- CLOUD JUMP-STILL IDENTITY. STATUS: OPEN (decision, not code).**
Share one portrait across a character beat's segments as the local face lane
does, or accept per-cut re-minting on cloud. See section 3's contradiction.

**F11 -- THE MIRROR DELETION HAS NO LIVE PROOF. STATUS: CLOSED 2026-08-07.**
No canonical leg had proven capped single- AND multi-segment beats cover their
audio. **Reinforced 2026-08-06:** the acceptance-grader repair (`e499b7fc`)
shipped on code-complete + suite-green for the same reason -- a scan of 4,371
JSON files under `output/otr` found no retained multi-segment artifact anywhere
on this box.

**DISCHARGED** by the 0-BIS live leg `signal_lost_midnights_toll_20260807_085918`
(`otr_w45_ltx_video`, 120 words, `RESULT SUCCESS` + `obs_publish OK`), which is
that missing artifact: seven `ltx_video` beats, two of them past the 169-frame
ceiling and chained into two segments each, every beat and all nine segment
receipts `extension_mode="none"`, `scripts/grade_episode.py` `ACCEPTED: 7
shot(s)` at exit 0 -- with the retired `OTR_LTX_LOOP_VIA_REVERSE=on` switch
proven present in the render server's own environment block. See GO_FORWARD
section 0-BIS and the 2026-08-07 `docs/HANDOFF_LOG.md` entry.

**F12 -- `google_veo_video` contract vs behaviour. STATUS: UNVERIFIED.**
Recorded as `max_frames=0` (the unbounded sentinel) while something caps its
segments at 200. `grep max_frames` in `eng_google_veo_video.py` returned nothing
on 2026-08-06, so the declaration has MOVED and this needs re-reading rather than
re-asserting.

## 5. OPEN DECISIONS

1. **F2:** what requalifies the humo cap without a GPU-hour ladder, now that the
   asymmetry is gone but the evidence document is still missing?
2. **F4:** is 65 defensible, given the cost row it derives from is itself
   suspect, or does `fastwan_8gb` hold at 81 until a canonical calibration exists?
3. **F10:** for a CHARACTER beat on a cloud lane, is per-cut re-minting
   acceptable? The local face lane already answers "share one portrait."
4. **Section 3's contradiction:** is humo `stills_minted` 2 or 0?
5. **Sequencing:** which items must land BEFORE the randomizer campaign. The
   campaign is the proof, so anything that changes render behaviour lands first.

## 6. THE CAMPAIGN (after the ruling)

A 30-45 word randomizer episode per video model. Local engines run on this box.
**Cloud engines cannot run: no API keys, no paid services, offline-first**
(CLAUDE.md scope discipline) -- their maths and still logic are reviewed
statically here and must be qualified separately when a cloud lane is authorised.

## 7. THE PADDING RULE, AS IT STANDS (added 2026-08-06)

**Operator ruling, 2026-08-06:** *"there is no mirror or ping pong unless for
credits."* Confirmed the same day: the closing loop is OK.

Enforcement, verified 2026-08-06:

| surface | state |
|---|---|
| `wrapper_bridge.extend_frames_to_target` (the mirror extender) | DELETED (tombstone, `wrapper_bridge.py:499`) |
| `eng_wan_ti2v` ping-pong | REFUSES, single-clip and coverage-planned (`:1094-1100`, `:1105-1113`) |
| `eng_ltx_8gb` mirror-extend | REMOVED; trims REAL frames only |
| `eng_ltx_video` boomerang | RETIRED |
| composite `_should_loop_fill` | RETIRED 2026-08-02, a named no-op returning False |
| composite clip underrun | TERMINAL -- raises `ClipUnderrunsItsBeat`, on any shortfall |
| credits floor-extend / credits music loop | RIPPED 2026-07-03 |

**The ONE sanctioned reuse** is `otr_silent_composite.py:452-455`, which loops
the last drama clip to fill the CLOSING-THEME region up to the master-audio
boundary -- the operator's 2026-06-17 "credits over the scene" look.

**Name it precisely: this is the CLOSING-THEME BACKDROP, not the credits roll.**
The actual credits roll (`OTR_CreditsRoll`) freezes the body's final frame and
appends a silent tail; it never loops. The two were being described as one thing,
which is how an exception quietly grows a second member.

## CONSTRAINTS

100% local, open source, offline-first. 16 GB RTX 5080, 14.5 GB real-world
ceiling. `wan_8gb`'s sampler recipe is FROZEN. The only workflow JSON is
`workflows/otr_canonical.json`; the section 0A bench carve-out is MEASUREMENT
ONLY and may not authorize a production cost row. Every second of audio gets
ORIGINAL video -- no mirrors, no ping-pong, no held frames, with the single
closing-theme backdrop exception named in section 7. Fail loud, no fallbacks.
