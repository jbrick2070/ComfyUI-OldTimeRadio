# FOLEY BED -- operator rulings, 2026-08-26

The LTX 2.5 foley bed: the video model's OWN generated audio, mixed UNDER the
TTS and music.

## TERMINOLOGY -- these are two different things and must never be conflated

* **SFX bed** -- separately GENERATED sound effects from a dedicated SFX model.
  **RIPPED 2026-08-06** (`rip-sfx`). Dead, and staying dead. The retired
  `clip_manifest_json` connector on `OTR_MasterAudioMux` is what fed it.
* **FOLEY bed** -- the audio LTX 2.5 already computes as part of the video
  render and currently DISCARDS at `LTXVSeparateAVLatent`
  (`eng_ltx25.py`; `LTXVAudioVAEDecode` is never wired). Never built. Has
  nothing to do with the rip.

Operator, 2026-08-26: *"sfx bed is different than foley bed, i won't get the
two confused."* Rebuilding the ripped SFX path under the name "foley" would be
resurrecting exactly what was deliberately killed.

## RULING 1 -- FIXED MIX, NO DYNAMIC DUCKING

Operator: *"ducking fixed i would say .20 foley .80 voice."*

**Foley 0.20 / voice 0.80. A fixed ratio, not a sidechain.**

Consequences, stated so the build cannot drift:

* NO sidechain compression, NO per-beat loudness analysis, NO envelope
  following. A static gain is deterministic and reproducible -- the same
  inputs give the same master every time, which a dynamic ducker would not.
* This is NOT the retired SFX bed's `OTR_SFX_BED_GAIN` default of 0.45. The
  foley bed sits lower because it plays under dialogue continuously rather
  than in gaps. Do not inherit that constant or its name.
* The TTS/music master is not attenuated by the foley's presence -- voice
  holds 0.80 whether or not a foley stem exists for that beat, so a beat
  without foley does not get louder.

### Ledger-driven ducking was considered and DEFERRED, not rejected

Raised 2026-08-26 in the same conversation: *"maybe there's a more intelligent
way, auto ducking sounds cool."* The option costed out was NOT envelope
sidechain -- it was **ledger-driven** ducking, which is available here and is
not available to a normal mixer: every line's `line_id`, `start_s` and
duration are already frozen in the ledger BEFORE video renders, so the bed can
duck from the SCRIPT rather than by detecting the voice. Foley 0.20 under a
line, ~0.50-0.60 in the gaps, ~150-250 ms ramps -- ambience swelling between
lines the way radio drama actually does, while staying fully deterministic
(same ledger, same mix, no take-to-take variance, no pumping).

Operator's call: *"let's start simple 80/20."* So the FIXED ratio ships first
and proves the foley path end to end; the ledger-driven envelope is a later
build on top of a working bed, not a prerequisite. **0.20 remains the
speech-duck floor in that later design**, so this ruling is not superseded by
it -- the gaps simply get to open above it.

## WHAT THE LAB ALREADY PROVED (evidence, not design)

`vram-recipe-lab/LTX_2_5_ON_16GB.md`: *"Foley and score are the strong suit.
The audio the model generates for a scene it is also rendering -- footsteps,
room tone, a theremin cue -- is where the joint model earns its keep over
bolting a separate audio pass on afterwards."*

Working recipes exist: `recipes/ltx_2_5_golden_i2v_foley.json` and
`ltx_2_5_golden_t2v_action_foley.json` (93.0 s render, 15.51 GiB). Both wire
`LTXVAudioVAEDecode` against `ltx-2.5-audio-vae-bf16.safetensors` with
`audio_cfg = 1.0`.

**But no design exists for keeping it.** Searched both repos on 2026-08-26:
EVERY mux path in OTR and in the lab discards the model audio and muxes TTS
instead -- `ENVELOPE_LADDERS.md:84,125`, `mux_source_delivery.py`,
`HUMO_BAKEOFF.md:148`, `H3_LAB_CANDIDATE_HANDOFF:147`, and
`OTR_MasterAudioMux` itself. The foley is proven good and then thrown away,
every time. Keeping it is new ground.

## RULING 2 -- NORMALIZE THE WHOLE THING AT THE END (already true; do not rebuild it)

Operator: *"of course normalize the whole thing when done."*

**This already exists and needs no new code.** `scene_sequencer._master_loudness()`
measures integrated LUFS and applies a SINGLE LINEAR GAIN to `-14.0` LUFS
(`_MASTER_TARGET_LUFS`), peak-safe at `-1.0` dBFS.

Why it is safe for the foley bed, and why that is not luck: a single linear
gain moves the finished mix as a whole and therefore **cannot disturb the
0.20/0.80 ratio**. A compressor or limiter at this stage would have
re-balanced foley against voice unpredictably. The ratio is set upstream and
survives delivery levelling unchanged.

**THE LINE THE BUILD MUST NOT CROSS**, in that function's own words: *"THIS
STAGE SETS THE DELIVERY LEVEL. It does NOT balance the mix -- that already
happened per clip, upstream... The two do different jobs and must not be
confused."* So: the foley bed is mixed at 0.20/0.80 UPSTREAM, and
`_master_loudness` stays purely a delivery stage. **The foley stem gets no
normalization pass of its own** -- normalizing it separately would fight the
ratio rather than serve it.

The `-14` target is measured, not chosen: 8 real masters across two months
delivered a mean `-9.87 LUFS (std 0.41)`, ~4 dB hot, so every episode was
attenuated at playback while the limiting used to buy that loudness stayed in
the audio. Do not master hot.

## RULING 3 -- THE DEFAULTS THE ARC LOCKED (r4, and they were FORCED)

These read as open choices until the arc grounded them. They are not choices:
`EpisodeAssembler` folds themes and cues into ONE WAV
(`scene_sequencer.py:1408-1425`) and the sequencer lays room tone at intensity
0.01 (`:1165-1180`). **No separate voice/music stems exist.** So there is no
mechanism to hold music at 1.0 while ducking dialogue.

* **The FULL master is `* 0.80`** -- dialogue + room tone + themes + music
  together. A 20% attenuation of music under a foley bed is a normal mixing
  consequence, not a blocker. The driver's *"if music must stay at 1.0 this
  build cannot start"* was an overstatement and is withdrawn.
* **LTX foley STACKS with the existing 0.01 room-tone bed.** Accepted
  explicitly rather than left implicit.
* **0.20 stays the speech floor**, including in the later ledger-driven build.

### The four questions this file opened, all answered by the arc

1. **Where it mixes** -> `OTR_MasterAudioMux` ONLY. The sequencer cannot see
   foley: `_master_loudness` runs in `OTR_EpisodeAssembler.assemble`
   (`scene_sequencer.py:1472`), FOUR stages before video exists.
2. **Timing mismatch** -> the engine emits an UNTRIMMED rung-length stem; a
   sibling of `assemble_beat_segments` applies `(drop_head, keep_frames)` in
   sample space inside `render_beat_coverage`. Trim, then silence-pad to the
   slot. **Never loop, never clone-hold.**
3. **Beats with no foley** -> SILENCE. Carrying a neighbour's stem would place
   picture-conditioned audio under a different picture.
4. **Engine shape** -> its OWN internal engine (`ltx25_foley_plus`), 1:1 with
   `ltx25_high_foley_plus`, per lesson L5.

## WHAT IS NOT BLOCKED -- corrected twice by the arc

The 2026-08-19 standing ruling says *"Chunk B (the foley bed) remains BLOCKED
on execution order."* That blocker was real only for the ORIGINAL mime design,
where the clip's audio REPLACES the beat audio and therefore had to exist
before the master froze. It does not bite a bed mixed UNDER the master at
`OTR_MasterAudioMux`, which runs after video -- and Ruling 4 below removes it
for mime too, by generating the TTS and discarding it.

**TWO DRIVER CLAIMS IN THIS SECTION WERE WRONG AND ARE CORRECTED:**

* *"no workflow JSON surgery is needed"* -- **FALSE.** `OTR_EpisodeAssembler`
  (node 7, order 12) has no way to know it is on a foley route: ShotLock
  (node 90, order 14) is the first writer of per-shot `engine_id`, and
  `OTR_VideoDirector` (node 87) is not wired into its `INPUT_TYPES`. An
  optional `video_policy_json` must be APPENDED to that node and node 87 wired
  to node 7 in `workflows/otr_canonical.json`, in the same change as the code
  (CLAUDE.md section 0), append-only (BUG-LOCAL-097).
* *"the connector is still wired so the mux is reachable"* -- true, but
  incomplete: `tests/test_rip_sfx_bed_guard.py:262-271` requires that
  connector's tooltip to say **"retired"**. Shipping a live mix without
  rewriting that test fails CI on the first compile.

---

## RULING 4 -- MIME SHIPS AS GENERATE-AND-DISCARD. This SUPERSEDES 2026-08-10.

Operator, 2026-08-26: *"in the MIME, you are going to ignore whatever generated
music. So we're gonna waste some music. I get it. It's not gonna be used. But
we'll just render it anyway to make things simpler."*

**Mime is `foley 1.00 / master 0.00`, role-wide, on the same mechanism as
`foley_plus`.** The TTS and the music cue for a mime beat ARE generated and
then mixed to zero. The waste is accepted deliberately.

### What this buys, and why it is worth the waste

`docs/2026-08-10-DESIGN-BRIEF-mime-overrule.md` required that a mime lane
generate **no** TTS or music. That constraint is what forced everything else in
that brief: audio for a mime beat had to exist BEFORE the master froze, which
required a new pre-audio owner node (`OTR_MimePlanRender`), an
execution-order inversion, and a per-beat ownership ledger.

**Generate-and-discard deletes all of it.** Nothing has to happen before the
freeze, because nothing is being replaced -- the master is simply attenuated to
zero in that window at mux time, exactly as `foley_plus` attenuates it to 0.80.
Same pipeline, same code path, one different constant.

Cost: a few seconds of unused TTS per mime beat plus one unheard music cue. On
a one-act that is noise. Against deleting a whole node and an ordering rework,
the trade is obvious.

### THIS IS A DELIBERATE OVERRIDE, NOT AN OVERSIGHT

**The 2026-08-10 "mime generates no TTS" requirement is SUPERSEDED for this
build.** That spec is not wrong -- it solved a harder problem than the operator
needs solved. Recorded explicitly so a future window does not read the older
brief, conclude the rule was forgotten, and rebuild `OTR_MimePlanRender`.

`kibitz-runs/2026-08-26-foley-bed` r1 CUT mime for two reasons. Only one of
them survives this ruling:
* **"multiplying TTS by zero does not satisfy no-TTS"** -- RESOLVED by this
  ruling. The requirement itself is withdrawn.
* **Role-wide scope** -- STANDS, and is now the accepted shape: engines are
  role-wide director dropdowns, so `ltx25_high_mime` in a role means EVERY beat
  of that role in the episode is a silent performance carrying the video's own
  score. That is a scored-film lane, and it is what the operator is choosing.
  Per-beat mime-cast remains out of scope and would still need the 08-10 node.

### The one real edge case, and it is small

The master is ONE continuous WAV, so zeroing a beat's window cuts whatever else
occupies those samples -- including a theme or cue that spans the beat
boundary. A cue crossing the seam into a mime beat stops mid-phrase rather than
resolving. Equal-power crossfades already exist at
`scene_sequencer.py:1435-1444`; a short splice at mime-window edges is the fix
if it audibly clicks. **Polish, not a blocker** -- and explicitly NOT required
for the first build.

### What the driver got wrong, recorded so the correction sticks

The driver reported that mime at 1.00/0.00 "zeros the music" as though it were
a defect needing an operator decision. **It is the intended behaviour** -- the
video brings its own score, so OTR's music is supposed to be off. Likewise
"if music must stay at 1.0 this build cannot start" overstated a 20%
attenuation under `foley_plus` into a blocker. Both framings were wrong and the
operator corrected them.


---

## RULING 5 -- THE FOLEY RECEIPTS RIDE THEIR OWN CONNECTOR, NOT `clip_manifest_json`

Asked 2026-08-26 at the start of the build, because the r4 spec and BOTH QA
gates refused to let the implementer decide it: *"the implementer must not
pick."* The operator picked **(b), a new dedicated connector.**

**What was at stake.** `clip_manifest_json` on `OTR_MasterAudioMux` is a
deliberately-placed tripwire from the SFX-bed rip.
`tests/test_rip_sfx_bed_guard.py:262-271` requires it to exist, be
connector-only, and *"say plainly that it is retired -- never invent a use"*,
and asserts it is *"accepted, hashed, unused."* Driving the foley mix off that
exact JSON is, precisely, inventing a use. Option (a) would have satisfied the
test's literal string assertions while making its name, its docstring and its
reasoning false -- which this repo treats as a defect, not a fix.

**What shipped instead.** `OTR_MasterAudioMux` gained TWO appended optional
inputs and the canonical graph gained three links:

* `video_policy_json` (node 87 -> node 85) -- the SAME question, off the SAME
  source, that `OTR_EpisodeAssembler` now asks (node 87 -> node 7).
* `foley_receipts_json` (node 92 slot 1 -> node 85) -- the clip manifest,
  carrying the per-beat `foley_path` / `foley_sha256` / `start_s`.

`clip_manifest_json` and its link 278 are **completely untouched**: still
wired, still hashed by `IS_CHANGED`, still unused, still saying "retired".
`tests/test_rip_sfx_bed_guard.py` required no edit at all, which is the whole
value of the option the operator chose.

**Cost, as predicted:** one extra link on a JSON that was being edited for
`video_policy_json` regardless.

---

## RULING 6 -- MIME SHIPS IN THE SAME CHANGE AS FOLEY. This OVERRIDES the spec.

Operator, 2026-08-26, mid-build: *"foley and mime we need this feature for
both."*

**The r4 spec's final gate said the opposite** -- *"This build registers ONLY
`ltx25_foley_plus` ... Mime's 1.00/0.00 envelope is the NEXT item, not this
one"* -- on the reasoning that a per-window master gain is a new fork against
the global `* 0.80` and deserves its own pass. **That deferral is withdrawn.**
It was a scoping judgement, and scoping is the operator's call, not the
panel's or the driver's.

Recorded explicitly because the spec artifact in `kibitz-runs/` still says
"mime CUT" in its title and "ONLY ltx25_foley_plus" in its last section. A
future window reading that file alone would conclude mime was never built.
**Both lanes are registered and public as of this change.**

### What it cost, and the panel's reasoning was sound as far as it went

The fork the gate flagged is real: the two lanes attenuate the master
DIFFERENTLY, and one shared constant could not express both.

* **`ltx25_foley_plus` is GLOBAL 0.80.** RULING 1 is explicit -- *"voice holds
  0.80 whether or not a foley stem exists for that beat, so a beat without
  foley does not get louder"* -- so it is a single scale across the whole
  timeline, including beats that carry no bed at all.
* **`ltx25_mime` is PER-WINDOW 0.00.** Engines are ROLE-WIDE dropdowns, so a
  mime role still leaves the announcer and music roles speaking, and all of
  them share ONE master WAV. A global zero would silence the episode; RULING 4
  describes zeroing *"a beat's window"*.

The resolution was not a special case but a generalisation: the mix carries a
master-gain **envelope** rather than a scalar. It starts at the global gain and
mime rows punch their own windows down to zero. Everything else -- the harvest,
the second-pass decode, the durable stem, the cut in the coverage assembler,
the splat -- is shared, which is what made adding the second lane cheap once
the first existed. `Ltx25MimeEngine` is a subclass with a name and two
constants; the constants live in `foley_stems.FOLEY_LANE_GAINS`.
