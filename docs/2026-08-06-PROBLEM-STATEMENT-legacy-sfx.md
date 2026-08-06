# PROBLEM STATEMENT -- the legacy SFX you thought was ripped out

**Date:** 2026-08-06. **HEAD:** `90124cb6` on `v2.0-alpha`.
**Written because:** the operator asked "I thought we ripped SFX out ages ago but
didn't -- I need a problem statement, I'm really confused."

**STATIC.** Nothing here enters `docs/PROD_BUG_LOG.md` without a live artifact.
No code was changed to write this. Every claim below cites a real file at HEAD.

---

## 0. THE SHORT ANSWER

You are not misremembering, and you are not wrong. **Three different things in
this repo are called "SFX". You ripped one, you parked another, and you never
touched the third -- and the third is the one that is still wired.**

| # | The thing | State | Evidence |
|---|---|---|---|
| 1 | The sfx/b-roll **ROLE** | **RIPPED** 2026-07-01 | `role_slots.py:2,72,82` |
| 2 | The SFX **BED** (audio under the master) | **LIVE AND WIRED** | canonical node 85, link 278 |
| 3 | The SFX **ENGINE LANE** / cue ledger | **PARKED** 2026-08-04 | `315e8afd` |

The confusion is real and it is the repo's fault, not yours: one word covers a
video role, an audio mix path and a whole generation campaign, and their states
are unrelated.

---

## 1. WHAT YOU RIPPED (and it IS gone)

**The sfx/b-roll ROLE, 2026-07-01, tagged `rip-sfx-broll`.**

`nodes/_otr_shared/role_slots.py:2` opens with the tombstone: *"rip-sfx-broll
2026-07-01: retired_role_a / retired_role_b roles REMOVED"*. An unknown role now
RAISES by name with NO FALLBACK (`:72`, `:82`). The `pool_n_loop` still/clip
pooling died with it -- `render_driver.py` records that every beat now renders
per-beat with its own scene still.

This is almost certainly the memory you are working from. It is accurate, it is
complete, and nothing below undoes it.

## 2. WHAT YOU PARKED (designs kept, deliberately unspent)

**The SFX engine lane / cue-ledger campaign, 2026-08-04, commit `315e8afd`.**

Your own commit message states the reasoning: *"The operator doubts SFX works
with the video model and calls it a much bigger lift than imagined; ROADMAP's own
8-15 coder-day estimate agrees. Parked in both docs, designs kept, nothing spends
against it without an explicit revival."*

The designs are still on disk and still good:
`docs/2026-07-31-sfx-engine-lane-SPEC.md`, `docs/2026-07-11-timeline-cue-ledger.md`,
`docs/2026-07-11-cue-ledger-r1-codex-prompt.md`.

Parked is not ripped. Nothing here runs, and nothing here is supposed to.

## 3. WHAT IS STILL WIRED, AND THIS IS THE ACTUAL PROBLEM

**The SFX BED was never removed. It is in the canonical workflow today, armed,
with no off switch.**

Grounded against `workflows/otr_canonical.json` by loading the JSON (the file is
one line, so grep tells you nothing useful -- searching it for the string `sfx`
returns NOTHING, which is exactly why this looked settled and is not):

* `OTR_MasterAudioMux` is **node 85**, `mode: 0` -- active, not muted.
* Its `clip_manifest_json` input is **WIRED**: link **278** from node 92
  `OTR_VideoRenderBatch`. That is the channel the SFX stems travel on.
* `widgets_values` is `[25, "ffmpeg", ""]` -- fps, ffmpeg, output_path. **There
  is no SFX widget. There is no way to turn it off from the graph.**

And the node runs the SFX path unconditionally:

* `otr_master_audio_mux.py:751` -- `mux()` calls
  `compile_sfx_bed_from_manifest(...)` on **every single run**, inside the
  terminal try block.
* `:194-196` -- that function returns `""` early **only because no manifest row
  carries `sfx_stem_path`**. That is the entire reason your episodes have no SFX.
* If any row did carry one: `:207-251` compiles an ffmpeg bed, and
  `mux_master_audio` (`:425-441`) mixes it under the frozen master at
  `DEFAULT_SFX_BED_GAIN = 0.72` (`:33`), behind a PCM integrity gate (`:472`).
* The only knob is the env var `OTR_SFX_BED_GAIN` (`:115`), and it changes the
  **level**, never whether the bed runs.

**So the bed is not dead code. It is a loaded consumer waiting on a producer.**

### 3.1 How close is it to firing? One dropdown pick.

Five registered engines produce a provider SFX stem (measured against the live
registry, not a doc):

    cloud_vidu_q2_pro_fast_720p_sfx
    google_vid_sfx_omni
    google_vid_sfx_veo_fast
    google_vid_sfx_veo_lite
    google_vid_sfx_veo_pro

All five are offered in the **`music_visual`** role dropdown (32 engines offered
there, 5 of them SFX-capable). Every other role offers none.

`registry.py:581-585` is explicit that this is by design: *"there is NO
validated-subset dropdown filter. Every REGISTERED engine is SELECTABLE."*

**Therefore: selecting one of those five on the music_visual role is sufficient,
by itself, to turn the SFX bed on.** No code change, no other widget, no warning
in the log, and no way to decline it.

The one thing standing in the way is that all five are cloud/paid engines, which
cannot run on this box. That is a property of your hardware and your wallet --
not a guard anyone built.

## 4. THE DEFECT THAT IS WAITING THERE

If the bed ever does fire on a **multi-segment** beat, it publishes the WRONG
stem. This is spec 7.1 of `docs/2026-08-06-BUILD-SPEC-no-mirror-enforcement.md`,
recorded there and NOT scheduled, per your ruling.

`render_driver.py:3499` builds the assembled beat as `beat_clip = dict(clip or {})`
-- a copy of the **LAST** segment -- and never overwrites the three SFX fields.
All three are consumed as BEAT-scope:

* `persist_episode_clips` moves the stem as the BEAT's (`:4442-4463`),
* `build_clip_manifest` publishes it on the BEAT row (`:4648-4652`),
* the mux lays it at the beat's `start_s` for `target_frame_count / fps` seconds
  (`otr_master_audio_mux.py:207-224`).

So a three-segment beat would play its **last** segment's foley across the
**whole** beat, starting at the beat's opening. And it is reachable rather than
theoretical: the `google_vid_sfx` frame contracts are discrete ladders
(`eng_google_vid_sfx.py:455-460, 498-504`), so a beat above the top rung takes
multi-segment `JOIN_JUMP` and lands in `render_beat_coverage`.

This is the same trap the no-mirror build exists to close -- a beat-scope field
left unassigned silently becomes a segment-scope one -- in its fifth, sixth and
seventh instance.

## 5. WHAT I DID NOT DO

Per your ruling on 2026-08-06 ("we don't have any sfx in prod, it was ripped out,
I don't want to do sfx now") I changed **no SFX code**. The one place the
no-mirror step-1 work touches `eng_google_vid_sfx.py`, it adds the two VIDEO
honesty receipts to the video clip and explicitly leaves the SFX trio alone, with
a comment saying so.

**Correction to what I told you earlier in that exchange:** I said the canonical
workflow "carries no sfx wiring, so nothing published reaches those three keys."
The second half is true today. The first half was misleading -- I had searched
the one-line JSON for the string `sfx` and found none, but the wiring is real and
travels under the name `clip_manifest_json`. The bed is armed. That comment in
`eng_google_vid_sfx.py` overstates the case and should be softened when that file
is next touched.

## 6-ANSWER. THE OPERATOR RULED: RIP IT, 100% (2026-08-06)

Operator, on reading this document: *"I do really want to rip out SFX 100%,
that's my aim. How it gets done: you, and ask Fable -- you can
`/kibitz-plugin:kibitz`, Codex etc. But don't break the system."*

**So option (c) below is the accepted plan, and options (a) and (b) are closed.**
The rest of section 6 is kept as the reasoning that led here, not as an open
question.

Two standing rules bind the execution and neither is optional:

* **The ledger rule (operator directive 2026-07-14, hard):** removing a pass is
  legitimate, but the ledger must still be filled COMPLETELY for downstream
  consumers, which read FIELDS and not intentions. Before anything is deleted:
  enumerate EVERY field the path writes, give each one exactly one new owner
  (deterministic Python, another pass, or an explicit default), delete only
  then, and prove it on a live leg. A ripped pass with an unowned field is a
  broken render, not a simplification.
* **The kibitz gate (operator directive 2026-08-04, hard):** this is a coding
  item, so it gets the FULL four-round arc before code, plus the Fable gate the
  operator named explicitly.

The work is tracked in its own build spec; this document remains the statement
of the problem and the record of how the decision was reached.

## 6-ORIGINAL. THE DECISION THIS OWED -- kept for the record

**Is the SFX bed's live wiring intentional dormancy, or a leftover?**

Both readings are defensible and they differ in what happens the day someone
picks a music_visual engine:

* **(a) LEAVE IT ARMED.** Costs nothing now. The lane's designs are parked and
  ready, and the bed is the half that already works. Risk: a single dropdown pick
  silently adds an audio bed nobody asked for, on a campaign you deliberately
  parked, carrying the section-4 last-segment defect.
* **(b) GATE IT.** Add one explicit switch -- a widget or a declared off-by-default
  -- so the bed cannot activate without an intentional act. Smallest change that
  makes the current state match your mental model. Roughly an afternoon, and it
  is a widget change, so it needs the `widgets_values` append-only discipline
  (BUG-LOCAL-097) and a canonical-JSON edit in the same commit.
* **(c) RIP THE BED TOO,** matching your memory. Before any deletion, the ledger
  rule applies in full: enumerate EVERY field the path writes
  (`sfx_stem_path`, `sfx_duration_s`, `sfx_sha256`, `audio_mode=sfx_mixed`,
  `sfx_bed_path`, `sfx_gain` in the mux report, plus whatever
  `_otr_ledger_freeze.py` and `production_ledger.py` stamp -- 18 and 15
  references respectively), give each one an owner or an explicit default, and
  only then delete. A ripped pass with an unowned field is a broken render.

My read, offered as a recommendation and nothing more: **(b)**. It is the only
option that makes the code say what you already believe, it is cheap, and it does
not throw away a working half of a campaign you may revive. (c) is the most
honest to your memory but it is the most work and it deletes something you might
want back. (a) is fine right up until the day it is not, and that day arrives via
a dropdown rather than a commit.

## 7. WHAT WOULD SETTLE IT WITH EVIDENCE

None of this needs a render. Two checks, both free:

1. Load `workflows/otr_canonical.json`, confirm node 85's `clip_manifest_json`
   link is still 278 and that no widget gates the bed. (Done above; re-run it if
   the workflow moves.)
2. Ask the registry which engines are offered on `music_visual` and which of them
   set `wants_provider_sfx` or carry `sfx` in the name. (Done above: 5 of 32.)

The thing that CANNOT be settled without a paid cloud leg is whether the bed
actually sounds right under the master -- which is precisely the doubt that got
the lane parked in the first place, and it has not changed.
