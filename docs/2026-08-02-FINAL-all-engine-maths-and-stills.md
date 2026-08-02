# FINAL DECISION: maths + still logic, every video model, local and cloud

**Operator, 2026-08-02:** "codex and Fable for final decisions, code and test.
Be sure it includes a review of all maths and still logic for all video models,
local and global. Then run a 30-45 word randomizer test on each video model."

This consolidates four panels and an archive sweep. Every table is MEASURED from
the live registry today. Dropdown names throughout. The panel's job is to break
the fix list in section 4 and rule on the open questions in section 5.

**Do not launch renders or boot a server** -- the campaign runs after the ruling.

## 1. LOCAL ENGINES -- corrected maths, 442-frame beat (17.68 s @ 25 fps)

Two rows in my earlier doc were WRONG; both panels caught them independently.

| dropdown name | family | contract | join | segments @442 | canvas |
|---|---|---|---|---|---|
| `wan_8gb (16:9)` | image_to_video | 17-177/q4 | chain | `[177,177,93]` | 832x480 declared |
| `fastwan_8gb (16:9)` | image_to_video | 17-177/q4 | chain | `[177,177,93]` raw; **`[81,81,81,81,81,45]` under the 81 cap** | 832x480 declared |
| `ltx_8gb (16:9)` | image_to_video | 9-161/q8 | chain | 3 segments | 512x288 declared |
| `ltx23_16gb_video (16:9)` | text_to_video | 169-169/q8 | chain | 3 segments | 832x480 declared |
| `ltx23_16gb_audio_in (16:9)` | audio_conditioned_video | 9-497/q8 | single | **renders 449, shows 442** | env branch 832x480 / 512x288 |
| `humo (portrait)` | audio_driven_face | 33-177/q4 | jump | 3 segments | own `_native_dims` 480x832 |
| `humo_1.7B (portrait)` | audio_driven_face | 33-177/q4 | jump | 3 segments | own `_native_dims` 480x832 |
| `humo_1.7B_169 (16:9)` | audio_driven_face | 33-177/q4 | jump | 3 segments | own `_native_dims` 832x480 |
| `humo_14B_169 (16:9)` | audio_driven_face | 33-49/q4 | jump | **`[49]x7 + [33]x3`** | own `_native_dims` 832x480 |
| `wan_i2v (16:9)` | image_to_video | 33-177/q4 | chain | 3 segments | **NONE -> 1472x832** |

Corrections from the panels: `humo_14B_169` is `[49]x7 + [33]x3` (a 2-frame
segment is illegal under a 33 floor); `ltx23_16gb_audio_in` RENDERS 449 frames on
its `9+8k` ladder to show 442, and VRAM admission must be sized against 449 on a
22B model. Visible totals are EXACT on every engine -- the coverage arithmetic
itself is sound.

## 2. CLOUD ENGINES -- measured, never before tabulated

| dropdown name | family | contract | join | segs @442 | consumes still | canvas |
|---|---|---|---|---|---|---|
| `cloud_kling_avatar` | audio_driven_face | 50-7500/q1 | single | 1 | no | NONE |
| `cloud_seedance_2` | audio_conditioned_video | 100-375/q25 | jump | 2 | **yes** | NONE |
| `cloud_vidu_q2_pro_fast_720p` | image_to_video | 25-250/q25 | jump | 2 | **yes** | NONE |
| `cloud_vidu_q2_pro_fast_720p_sfx` | image_to_video | 25-250/q25 | jump | 2 | **yes** | NONE |
| `cloud_wan_i2v` | image_to_video | 50-375/q25 | jump | 2 | **yes** | NONE |
| `cloud_wan_i2v_audio` | audio_conditioned_video | 50-375/q25 | jump | 2 | **yes** | NONE |
| `google_omni_video` | text_to_video | 75-250/q1 | jump | 2 | **yes** | NONE |
| `google_vid_sfx_omni` | text_to_video | 75-250/q1 | jump | 2 | **yes** | NONE |
| `google_veo_video` | text_to_video | **1-0/q1** | jump | 3 = `[200,150,100]` | **yes** | NONE |
| `google_vid_sfx_veo_fast/lite/pro` | text_to_video | **1-0/q1** | jump | 3 | **yes** | NONE |

`google_veo_video` renders 450 and shows 442 (8-frame tail trim). Visible totals
are exact on every cloud engine too.

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

## 4. THE FIX LIST -- ordered, break this

**F1 -- WIRE THE ADMISSION GUARD (safety, blocking).** `assert_frame_affordable`
(`motion_common.py:339`) has ZERO call sites, while `PLANNING_CAP_ENGINES`
contains `wan_ti2v`, `fastwan_8gb`, `ltx_8gb`. Every coverage-planned segment on
all three renders with NO preflight VRAM check. The 2026-08-01 ruling made this
guard a named prerequisite -- "U2 must not ship before this" -- and U2 shipped
anyway. An in-process CUDA OOM corrupts the allocator. Wire it at ONE admission
point after either branch selects its immutable length, computing effective free
ONCE as live free plus measured hoist.

**F2 -- THE HUMO CAP CITES MISSING EVIDENCE (safety).** `eng_humo.py:60`
justifies the 49-frame cap with "docs/2026-06-27-humo-bakeoff: zero OOM at
832x480/<=49f". **That document does not exist in this repo** (only the scripts
that would have produced it). It asserts ~15.9 GB against a 14.5 GB ceiling. And
the SAME 14B checkpoint at the SAME pixel count is capped at 49 wide
(`humo_14B_169`) but UNCAPPED to 177 portrait (`humo`) -- 480x832 and 832x480 are
both 399,360 px. One of those numbers is wrong. Requalify or lower.

**F3 -- ONE CAP AUTHORITY.** Declare ledger-stamped `video.max_render_frames` the
sole production authority; require the `launch.env` twin absent-or-equal before
planning; `render.frame_budget` is diagnostic-only and must never participate.

**F4 -- `fastwan_8gb` 81 -> 65, or requalify.** 81 was promoted from a bench cell
in direct contradiction of "a bench cell never qualifies an engine", and shipped
in two profiles marked `"status": "shipping"`. Under the current row at 832x480
(60.326 MiB/frame) and 13,000 MiB effective free at margin 0.85, 65 is the
highest legal 4n+1 rung. **65 is not an 8 GB qualification** and must be labelled
so.

**F5 -- `wan_i2v` must declare a canvas.** It is the only engine rendering at the
1472x832 default with no opinion of its own. The one measured rung says 17 frames
at that canvas costs 10,720 MiB -- near the ceiling before motion begins.

**F6 -- Cloud engines declare no canvas at all** (12 of 12). Same dead-channel
class that cost `wan_8gb` a 268-minute leg.

**F7 -- BUILD THE LIP-SYNC ONSET PAD.** `BUG_BIBLE.yaml:2343` (BUG-LOCAL-102):
HuMo audio leads the lips by 100-200 ms; the prescribed fix -- pre-pad leading
silence, drop the pad frames after decode, stamp the value in the ledger -- was
never built. It is audible on every episode the face engine has produced, and a
10-segment beat repeats it at every cut.

**F8 -- Stale rationale.** `mouth_policy.py:144-148` justifies tolerating jump
cuts because "the same character is regenerated mid-line from a different seed",
but `_video_seed` is computed once per SHOT, not per segment -- every segment of
one beat renders with the identical seed.

**F9 -- Stale comment.** `eng_humo.py:61` still says beats over the cap
"render at the cap then mirror-extend to the audio target". That behaviour is
gone; it now raises `MirrorExtensionForbidden`.

**F10 -- CLOUD JUMP-STILL IDENTITY** (section 3). Decide: share one portrait
across a character beat's segments, as the local face lane already does, or
accept per-cut re-minting for cloud.

**F11 -- THE MIRROR DELETION HAS NO LIVE PROOF.** `wan_ti2v` joined
`PLANNING_CAP_ENGINES` and its mirror was deleted, but no canonical leg has
proven capped single- AND multi-segment beats cover their audio.

**F12 -- `google_veo_video` declares `max_frames=0`** (the unbounded sentinel)
yet something caps its segments at 200 frames. Contract and behaviour disagree --
the same defect class as the `ltx_video` contract lie fixed this morning.

## 5. WHAT THE PANEL MUST RULE ON

1. **Is F1 the right first move**, and is one admission point after length
   selection correct -- or does it belong before `prepare()`?
2. **F2: which humo number is wrong** -- the 49 cap or the uncapped 177? Same
   checkpoint, same pixel count. What requalifies it without a GPU-hour ladder?
3. **F4: is 65 defensible today**, given the row it derives from is itself
   suspect, or does `fastwan_8gb` stay at 81 until the canonical calibration
   exists?
4. **F10: for a CHARACTER beat on a cloud lane, is per-cut re-minting acceptable?**
   The local face lane already answers "share one portrait." Should cloud match?
5. **Sequencing:** which of F1-F12 must land BEFORE the 30-45 word randomizer
   campaign, and which can follow it? The campaign is the proof, so anything that
   changes render behaviour has to land first.
6. **What have I still not measured** on any engine, local or cloud?

## 6. THE CAMPAIGN (after the ruling)

A 30-45 word randomizer episode per video model. Local engines run on this box.
**Cloud engines cannot run: no API keys, no paid services, offline-first**
(CLAUDE.md scope discipline) -- their maths and still logic are reviewed
statically here and must be qualified separately when a cloud lane is authorised.

## CONSTRAINTS

100% local, open source, offline-first. 16 GB RTX 5080, 14.5 GB real-world
ceiling. `wan_8gb`'s sampler recipe is FROZEN. The only workflow JSON is
`workflows/otr_canonical.json`; the section 0A bench carve-out is MEASUREMENT
ONLY and may not authorize a production cost row. Every second of audio gets
ORIGINAL video -- no mirrors, no ping-pong, no held frames. Fail loud, no
fallbacks. **Do not launch renders or boot a server.**
