# Timeline Cue Ledger -- spotted SFX for OTR

Status: **ROADMAP. No code. Not scheduled.** Do not start before runway/720w + lean-mean-rip land.
Date: 2026-07-11

**OPERATOR DECISION -- RATIFIED 2026-07-11: cue decisions run BEFORE the video chain.**

**The CueDirector sees no picture.** Not frames, not the clip manifest (a node-92 output; link 278
`92 slot 1 -> 85 slot 4`), and **not ShotLock** -- node 90 is gated on the assembler's `audio_done`
(link 253: `7 slot 3 -> 90 slot 1`), so it runs *downstream* of the splicer seam. Wiring it back would be
a literal graph cycle. Its context is the **fiction**: `lines[]`, `beats[]` (scene structure -- `scene_id`
per row, `production_ledger.py:841-892`), `cast[]`, `music[]` spans, and `meta.story_brief_terms`.

**That is sufficient, by shared conditioning.** The picture is generated *after*, and *from*, the same
textual brief the CueDirector reads. There is no independent picture truth to consult -- "syncing to
picture" would mean syncing to a stochastic render of a brief we already have. Cues that match the brief
match the picture because the picture is also made from the brief. The 1940s sound man had no picture
either; that is not a limitation being tolerated, it is the medium.

A second post-video overlay pass was considered and rejected: it doubles the machinery to buy picture
context for `establish`/ambience cues -- the **least** picture-dependent of the three functions.

**Re-open at C4 (VFX), which genuinely needs frames.** Not before.

---

## 1. Problem

Two failed shapes:

- **Pre-render SFX** (`speaker_role="sfx"`): cues authored as pseudo-dialogue *before any audio existed*.
  Ripped 2026-07-01; `tests/test_rip_sfx_broll_guard.py` fails loud if it returns. **Do not resurrect.**
- **Byproduct SFX** (today): the Google video engine returns video-with-audio; the stem is extracted and
  amixed as an episode-length bed (`otr_master_audio_mux.py:130-215`). **Nobody chose these sounds.**
  Ambience by accident.

Root cause of both: **you cannot place a sound against a performance that does not exist yet.**

Invert it. Render, listen, spot, re-cut.

---

## 2. How radio did it

Effects were **performed live** by a sound man working a script marked in rehearsal. Cues were **stage
directions** -- `SOUND: DOOR.` -- **events in the fiction**, not words in a sentence. Three jobs:

- **establish** -- place us somewhere. Once.
- **punctuate** -- the door is radio's paragraph break.
- **story_beat** -- the gunshot. The phone that changes everything.

And the split that governs the whole design: **sustained/transitional effects played *through* dialogue,
masked and all. Spot effects that carried story weight got held space.** `SOUND:` sat on its own script
line; the actors held for the knock. Suspense and Gunsmoke are full of air around the big effects.

---

## 3. The failure mode

> **The noun detector.** An LLM reading a transcript cues a door sound every time someone says "door."

A real editor's door slam lands where the door closes **in the fiction** -- often a line away from the
word, and usually where nobody is speaking. **LLMs are excellent at rationales**, so every noun-detection
cue arrives dressed in dramaturgy. Any gate that reads cues and asks "does this show taste?" passes
vacuously.

Defenses, all load-bearing:

- **The intent pass never sees timestamps.** Timestamped words are salient tokens; show them and the model
  cues the tokens.
- **The LLM never authors a timecode.** It names an event and an anchor. Code resolves the rest.
- **Hard cue budget (ceiling, not quota).** Converts *detection* into *selection*. Selection is taste.
- **Abstention is the signal.** A noun detector never says "this scene wants no sound."

**Honest limit:** timestamp-blindness protects *placement*, not *selection* -- pass 1 still reads the script,
and the nouns are right there. Selection is defended only by the budget, the event framing, and C1.
**C1 therefore carries the entire anti-noun-detector load.**

---

## 4. Splice, don't duck

Overlaying an effect on dialogue masks it. Carving it a **nest** -- a real gap it lives in -- is what the
live sound man had. But surgery on a finished master shifts every downstream timecode (HuMo lip-sync, clip
`start_s`, captions). So splice **upstream**, at the one seam where the audio is still raw and the ledger is
still un-shifted:

    3 SceneSequencer -> [NEW] OTR_CueSplicer -> 4 AudioEnhance -> 7 EpisodeAssembler -> video -> 85 Mux

At that seam the audio is **pre-loudness, pre-theme, pre-enhance**, and ledger rows are still in
`scene_audio` space. Consequences, all free:

- Gap insertion is a prefix-sum shift over `lines[]`/`music[]` **before** the assembler's existing BUG-106
  rebase runs -- every downstream timing surface is then computed **once, on final data, by existing code**.
- The enhance/period chain processes the spliced audio **uniformly** -- no spectral seam at the cut.
- **One master ever exists.** No v1/v2, no provenance to pin, no path collision, no re-pointed links.
- Spliced cue audio is mixed **into the master**, inside `_master_loudness` -- the archival WAV is the actual
  show. Node 85's bed is reserved for `overlay` cues only.

A second full sequencer pass was considered and **rejected**: node 7 fans out to 7 links across 5 consumers
(`85`, `92`, `90`, `91`, `12`, `94`), both assembler runs write the **same** path
(`scene_sequencer.py:1395` = `<ledger_dir>/<ep_id>_master.wav`, no uniquifier) so v2 silently overwrites v1
and a mis-wired consumer is undetectable, and `append_transition` (`:1812`) has no refresh so rows duplicate.
One splicer node at the raw seam does the same job with none of that.

### 4a. There is no pause machinery. Build it.

**Verified:** `breath_ms`, `beat_pause_ms`, `pause_ms`, `scene_transition_ms`, `act_break_ms`
(`scene_sequencer.py:754-758`) are **assigned and never read anywhere in the repo.** No `[BEAT]`/`[PAUSE]`
parser exists. The module docstring at `:9` ("intelligent pacing -- breath buffers, BEAT/PAUSE tags") is
false. Lines are butt-joined (`:1046 np.concatenate`) and `_trim_trailing_silence` (`:725`) actively strips
the natural tails first. **The show has zero inter-line air today.** Rip the dead constants (GO_FORWARD item 5).

**The gap must carry room tone.** `env_timeline` is appended **only** on the dialogue branch
(`:1005`, guarded by `segment_np is not None`), and the bed is painted per-span into a zeros array
(`:1052-1063`). An inserted gap would get **digital zero** -- a transmitter dropout, not a beat. The splicer
must extend `env_timeline` across every nest (or paint the bed over `[0, total_len)`). **Period radio never
had silence. It had a live room and a noise floor.**

### 4b. Nest rules

- **Function-gated.** `story_beat` / `punctuate` -> nest is authentic. **`establish` -> overlay, never nest**
  -- footsteps that stop the dialogue, play alone, then resume read as stop-motion.
- **Line boundaries only.** `splice` is legal with `in_gap_before | in_gap_after | at_scene_boundary`;
  `on_word` / `under_line` **force `overlay`**. A hole mid-performance freezes HuMo's mouth mid-word.
- **Subtract the existing gap.** Insert `max(0, needed_ms - existing_gap_ms)` -- the aligner already has the
  gap map -- or every nest stacks on air the TTS already left.
- **Duration from the stem, not a default.** A door-slam hold is ~250-400ms. 650ms of held air is a
  *dramatic silence*, and exists only if pass 1 authored one.

Upside worth naming: because pacing is currently dead, these nests would be **the first inter-line air the
show has ever had.** At line boundaries, with tone continuing, they should improve pacing, not wound it.

`duck_dialogue` is **cut**. Sidechain has no home in the static-gain amix, and the nest is the better answer.

### 4c. Video length and frame budgets: correct for free -- if one contract holds

Splicing **lengthens the episode**, so every video-side duration calculation must see the post-splice
timeline or the render desyncs and the frame budget is wrong. **The ratified ordering gives this for free.**

`OTR_ShotLock` (node 90) computes its clip budget from **cumulative audio samples on the final timeline**
(`compute_clip_budget`), and it is gated on the assembler's `audio_done` (link 253). It therefore runs
**strictly after** the splicer -> it necessarily reads post-splice, post-rebase timing. Frame budgets, clip
counts, and render length recompute from the lengthened master **by existing code, with zero video-side
changes.** This is not a happy accident of the ordering -- it is the *reason* the ordering is the only viable
one. Inverting it (cues after video) would have broken exactly this calculation.

**HARD CONTRACT (C3a):** `overlay_audio_timing` trusts `find_most_recent_ledger` -- **the newest ledger on
disk.** If the splicer ever persists an intermediate **pre-rebase** ledger, ShotLock overlays stale
pre-splice timing onto the frozen script and **every frame budget drifts.** The on-disk ledger must be
**post-splice and post-rebase before `audio_done` fires.** The splicer follows the same disk contract as
sequencer/enhance/assembler -- but it must be stated, tested, and asserted, not assumed.

A clean timestamped ledger is the load-bearing artifact for the whole video lane. Splice, rebase, persist,
*then* gate. In that order, never another.

---

## 5. Spotting: two passes

**Pass 1 -- INTENT.** In: ledger lines, scene structure, cast, settings, sound grammar, budget. **No
timecodes.** Out: an **event list** -- what happens in the fiction, which events get sound, each one's
function, and **what is deliberately left silent.** The only pass where "the tension lives in the silence
before the shot" is thinkable.

**Pass 2 -- PLACEMENT.** In: pass-1 events + word/gap map + music mask. Out: anchors + nest modes.
**Forbidden from adding events.** Barely needs a frontier model.

**Forbidden, in the contract:** no cue per noun-mention; no cues inside music spans; no cues under
overlapping dialogue except `story_beat`; no ambience loops as spot cues; no budget overrun; no off-script
events (or tag `invented:true` and gate on operator review).

**The music mask is free.** `ledger.music[]` carries interstitial spans, stamped by the sequencer write-back
in `scene_audio` space (`:1131-1151`) -- available at the splicer seam. No-cue zones: a spot effect under the
theme fights the score.

---

## 6. C1 -- the gate that decides whether this ships

Not "does the operator like the cues." **Differential, and instrumented so it can actually fail:**

- **Budget-matched control arm.** A ~50-line noun detector, **top-N hits where N = the LLM's own budget**,
  spread across scenes. If the control emits a different *count* or *format*, the operator will spot it from
  the row count alone and the gate passes vacuously.
- **Identical schema. Rationales stripped from both.** The doc's own premise is that rationales are
  unfalsifiable -- so do not show them to the judge.
- **Randomized arm order, >=4 episodes, preregistered threshold** (identify *and* prefer on 3/4).
- **Negative controls:** a scene whose correct answer is *no cues*. A noun detector never abstains.
- **Judged on the pass-1 event list**, not on timecoded rows.

**Indistinguishable -> STOP.** Runs on paper, before a single wav is generated.

---

## 7. Schema (`cue-1`)

    otr/episodes/<ep>/audio/<ep>_cues.json

```
{
  "schema": "cue-1",
  "episode_id": "...",
  "budget": {"per_scene": 3, "per_minute": 2},      # ceiling, not quota
  "events": [                                        # PASS 1 -- no timecodes
    {"event_id":"ev_003", "what":"Corrigan pulls the office door shut behind him",
     "function":"punctuate", "scene_id":"sc_02", "sounded":true, "silent_rationale":null}
  ],
  "cues": [                                          # PASS 2 -- placement only, adds no events
    {"cue_id":"cue_007", "event_id":"ev_003", "type":"sfx",
     "placement_mode":"in_gap_after",                # on_word | in_gap_before | in_gap_after
                                                     # | under_line | at_scene_boundary
     "anchor":{"line_id":"line_012"},
     "nest":{"mode":"splice","gap_ms":null},         # splice | overlay; gap_ms resolved from stem
     "library_id":"door_office_heavy_01",
     "gain_db":-4.0,
     "onset_tolerance_ms":30}
  ],
  "resolved": {                                      # written BY the splicer, after the cut
    "final_master_sha256":"...",
    "cues":[{"cue_id":"cue_007","start_s":47.83,"dur_s":0.38}]
  }
}
```

**`start_s` is an output, never an input.** A cue row cannot carry a resolved timecode *and* request a
splice -- the splice invalidates its own timecode, and every cue shifts every later cue. Anchors + gaps go
in; the splicer emits final `start_s` and stamps the sha of the **one** master that exists. One authority
for the timeline -- the same law that keeps `start_s`-space confusion (this chain's #1 historical bug class)
dead.

**Cues are a derived artifact** -- never a speaker role, never in `lines[]`. The rip guard stays green.
`type` is a column, so `vfx` rows ride the same table later.

---

## 8. Renderer: curated period library

**Commit to it.** The 1940s vocabulary is a few dozen stylized, **repeated** sounds -- the same door every
slam **is the show's door**. Per-cue generation gives a different door every time: modern sound-design
pastiche, the same register break the announcer-framing work is already fighting. It also produces hyper-real
foley, not the band-limited, stagey sound of the era.

30-60 CC0 one-shots (BBC-archive-era; same CC0 posture as the indextts2 refs), period-processed **once**,
addressed by `library_id` + variant. **Collapses C2 to near-trivial:** no model calls, no VRAM, deterministic
durations, reproducible. That matters near release.

Stable Audio (`OTR_StableAudioTheme`, node 83 -- conditions on `seconds_total` natively) and ElevenLabs SFX
(explicit `duration_seconds`) stay as the **escape hatch** for bespoke cues.

**Onset exact, decay natural.** For a one-shot, `dur_s` is **advisory** -- a slam `atrim`'d to 400ms is an
*amputated slam*. The validator checks **onset**. `dur_s` is a real contract only for beds and loops.

---

## 9. Grounded code gaps

- **G1 -- Replace the byproduct bed, don't stack it.** Cues are spotted against audio containing **neither**
  bed. Summing a deliberate score onto accidental Google-stem ambience = doubled slams, uncurated modern
  texture under period stems. **The byproduct bed retires when the cue lane ships.**
- **G2 -- One bed, one limiter.** Two beds would mean `alimiter=0.98` twice (`otr_master_audio_mux.py:196`)
  summed with a master already pushed to -1 dBFS (`scene_sequencer.py:1357`) through a third limiter ->
  pumping under dialogue on every overlap.
- **G3 -- Overlay cues need an adapter.** `compile_sfx_bed_from_manifest` requires `sfx_stem_path` +
  `target_frame_count` + manifest-level `fps` (`:156-186`) and applies **no per-row gain** (one global 0.72
  at mux). Cue rows carry `gain_db`. Second compile path or adapter.
- **G4 -- `IS_CHANGED` on node 85 hashes only `clip_manifest_json`** (`:605-608`). Add the new input or cue
  edits **silently cache-hit and the remix never runs.** Also: it uses per-process-salted `str.hash` -- make
  it sha256.
- **G5 -- Master-path landmines.** The assembler's WAV save is best-effort; on failure `output_path` is the
  literal string `"(video-only - master WAV save failed)"` (`:1368, 1422`) and is already consumed live by
  node 92 (link 264). New nodes **fail loud** on it. Same for the pending-dir rename race that forced
  `_reresolve_master_audio` (`otr_master_audio_mux.py:400-446`).
- **G6 -- VRAM.** Finished video stages sit resident at ~9-10 GB (CLAUDE.md §5); the tier rip removed the old
  ceilings. Alignment runs in the **audio phase, before video** -- which is where the splicer seam puts it
  anyway. The QA drift diff can run CPU int8.
- **G7 -- C0's two routes are not equivalent.** `FileAudioCache` writes **`.npy`** (`_otr_audio_cache.py:239,
  296-306`), which does **not** satisfy `render_driver.py:531`'s `*wav_path` consumer, and aligners want
  wav/pcm. **Take the wav route:** stop the engines' `os.remove` (`eng_indextts2.py:245`) and stamp
  `<engine>_wav_path` on the line -- the consumer is already pre-wired. Stems are pre-`_level_dialogue_clip`
  and pre-resample; **fine for alignment** (timestamps are gain- and rate-invariant). Do not "fix" that.

---

## 10. Free win, independently shippable

**The drift diff.** Transcribe the master, diff against `lines[].text`. TTS drops words, mangles names,
mispronounces. This is a **QA surface the pipeline has never had** -- the first time it would ever check its
own output. No cue system, no library, no schema. **Can ship on its own, in an earlier window.**

---

## 11. Sprints

| # | Deliverable | Gate |
|---|---|---|
| **C0** | Persist per-line stems (wav route, G7). Ship the drift diff. | clean stems on disk; drift report read |
| **C1** | `cue-1` schema + validator + resolver; two-pass CueDirector; **budget-matched blind control arm** + negative controls. **No audio generated.** | **blind A/B, >=4 eps, preregistered. Indistinguishable -> STOP.** |
| **C2** | Curated CC0 period library + `OTR_CueRender` (onset-exact) | cue stems at canonical paths, onset in tolerance |
| **C3a** | `OTR_CueSplicer` at the 3->4 seam: gaps only, room tone through every nest, prefix-sum ledger rebase. Rip the dead pacing constants. | 30w smoke; ledger timings intact; no dead air |
| **C3b** | Cue audio rendered into the nests; `resolved` block written back | operator eyeball -- does it read as scored? |
| **C3c** | Byproduct bed retired; node 85 single-compile + `overlay` adapter; workflow JSON wired same commit | live smoke -> `obs_publish OK` |
| **C4** | `type=vfx` rows + burn pass (the caption burner already proves the timecodes->picture seam) | later. Not before C3c is green. |

**C4 honesty:** the CueDirector sees **no picture at all** -- only the fiction (script, performance, scene
structure, brief). Sufficient for SFX by shared conditioning. **VFX genuinely needs frames**, so C4 must
re-open the ordering question with a real post-video pass. Do not pretend otherwise.

**C3a must also assert the video-budget contract (§4c):** the on-disk ledger is post-splice and post-rebase
**before** `audio_done` fires, or ShotLock silently budgets frames against pre-splice timing.

---

## 12. The claim

Every previous attempt authored cues **upstream** of the performance and hoped. This authors them
**downstream** -- against what was really said -- and then **re-cuts the timeline to make room for them at
the one seam where re-cutting is free.**

The seam choice is the achievement. **The noun detector is the danger.** C1 exists to kill it.
