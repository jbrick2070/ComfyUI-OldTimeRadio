# GHOST SIGNAL -- Spec of Record

**Lane:** `animatediff15_video` · label "AnimateDiff -- Ghost Signal" · spoken **Ghost Signal**
**Date:** 2026-08-22 · **Status:** FINAL DESIGN, PRE-INSTALL.
**Revision:** 4 (synthesized). **Panel:** Fable (r1 cold, r2, r3 rev.1-3, r4 synthesis),
Codex (blind r1, operator-driven "Ghost Signal" pass), Cursor (grounded review pass).
Driver is sole judge; every claim checked against the real Windows files.

> **Every AnimateDiff-specific number here is a HYPOTHESIS until section 8's
> measurements exist. The design is settled; the numbers are not.**

## 0. Standing operator rulings (binding, not relitigated)

| # | Ruling | Source |
|---|---|---|
| R1 | **No still, ever.** `text_to_video`; prompt plus noise. `accepts_still = False` DECLARED with its reason -- the G3.6 gate polices silence, not the value. | 2026-08-22 |
| R2 | **Golden rule:** the lane fills ALL THREE roles; the prompter differentiates them. No role may be refused. | 2026-08-22 |
| R3 | **No ping-pong, no mirror, no loop-fill.** Original seconds of render for every beat. Already law: `acceptance.py:257` `DELIVERABLE_EXTENSION_MODES = ("none",)`, graded on all beats by `grade_no_mirror`. | 2026-08-06 / 08-22 |
| R4 | **12.5 fps native, uniform hold-2, exact 25 fps delivery.** | 2026-08-22 |
| R5 | **FULL FRAME. No black bars.** *"I hate black bars, that's my problem statement."* Killed the original square-aperture staging. | 2026-08-22 |
| R6 | **Low resolution is ACCEPTED as identity**, not tolerated. *"I know it's not high res, I'm fine with that look."* | 2026-08-22 |
| R7 | **UNDER 4 GB VRAM.** The hardest constraint; outranks every aesthetic preference. | 2026-08-22 |
| R8 | **ENLARGE, not upscale.** Operator's own vocabulary fix: *"enlarge could be more accurate."* | 2026-08-22 |

## 1. Identity

The whole screen is the transmission. One dominant form -- a figure, a radio, a
prop, an emblem -- coalesces from noise across the full 1920x1080 frame, becomes
briefly readable, and partly loses itself again. Contours crawl, folds breathe,
hatching redraws itself, colours misregister, and the amount of instability
follows the dramatic pressure of the audio. The picture is not what happened; it
is what the broadcast stirred up.

The lane's material is the **soft signal surface**: a small render accurately
enlarged to full frame, so detail dissolves before it fully resolves and nothing
in the picture is sharper than what was really rendered. A far-off station is
soft, and softness is what an honest enlargement delivers -- so the delivery
method and the fiction agree. The lane never sharpens, never invents, never
disguises its poverty; it broadcasts it edge to edge.

The name survives every ruling and gets more literal each time: "Ghost Signal"
names the transmission -- not the bezel (R5), not a pixel grid (R8).

**Named failure modes.** (a) *Semi-photorealistic face soup* -- a melting human
close-up that contradicts the trusted voice and reads as failed AI cinema.
(b) *The invented-detail failure* -- any delivered frame carrying sharpness that
was never rendered (super-resolved edges, unsharp halos, restored texture) has
told the exact lie the lane exists not to tell, and fails identity even if it
looks better.

**Acceptance.** (a) *Enlargement integrity:* delivered frame exactly 1920x1080,
full frame, no bars/crop/pad -- **and the enlargement is ACCURATE, proved by the
ROUND-TRIP PROBE:** downscale the delivered frame back to the native canvas and
compare to the really-rendered frame; an unsharp pass or a super-resolver in the
chain diverges hard. Threshold (SSIM/PSNR) set from the clean-chain baseline at
measurement #2. This turns "did we invent detail?" into a number.
(b) *Subject readability:* one dominant form readable at some point in every
beat, judged on silhouette + costume + action, **never facial identity**.
(c) *No face soup:* character beats hold mid-shot or wider. (d) *Original
render:* every delivered frame traces to a really-rendered unique frame under
hold-2; `grade_no_mirror` clean on all beats.

## 2. The prompt composer

The LTX composition is NOT inherited -- its premise (the still carries the
subject, the prompt only moves) is dead here.

**THE INVERSION TABLE.** The repo states the dead premise in its own words at
`render_driver.py:1293-1300`: *"The i2v anchor carries the LOOK from the FLUX
still; the video prompt's ONLY job is MOTION ... every sentence is a motion
verb, NO set-dressing nouns, NO negation, total <= 240 chars, first motion verb
within the first 140."* Those rules were MEASURED for the still-carried case.
For a no-still lane they resolve as follows:

| MAD-verified rule (still-carried) | Ghost Signal (no still) |
|---|---|
| Every sentence is a motion verb | **INVERTS.** The subject is built from nouns; the front half is noun phrases by design. |
| NO set-dressing nouns | **INVERTS, bounded.** The subject IS nouns now -- but exactly one subject, one accent; the M4-wall trap still bans sprawl. |
| Total <= 240 chars | **REPLACED.** 320-char constant, derivation below. |
| First motion verb within 140 | **SURVIVES TRANSFORMED.** The guarantee moves from a char offset to the window budget: the whole prompt, motion included, lands inside the first CLIP window. |
| NO negation | **SURVIVES INTACT** -- and independently corroborates putting suppression in the live negative, never the positive. |

**Budget, with derivation.** The ceiling is the SD1.5 CLIP window: 77 tokens,
~75 usable. Constant **320 characters**, target **260-300**; at ~4.3-4.9
chars/token, 300 chars lands ~62-70 tokens with headroom. Overflow past the
first window degrades soft (chunk-and-concat, unverified -- measurement #7) and
is declared second-class, never load-bearing. `_LTX_MOTION_PROMPT_MAX = 240` is
UNTOUCHED; its name says whose it is. The lane's constant ships with this
derivation as its comment (the G3.5 lesson: never a number without its reason).

**Slot order:**
> pack cue -> subject identity -> framing lock -> beat action -> motion ->
> emotion -> one story accent -> shot law

**THE COMPOSER RULING: fixed order, content-gated presence.** The ORDER is fixed
and never conditional; a slot whose ledger signal is absent contributes zero
characters and is omitted entirely. Presence follows the ledger; order follows
the spec; both are pinned by test, so a given ledger composes BYTE-IDENTICALLY
every time. *(Cursor proposed conditional layer-selection. Refused: an order
that varies with content cannot be golden-pinned, and predictability is worth
more than the characters it saves.)*

| Slot | Character | Announcer | Music |
|---|---|---|---|
| Pack cue | `prefix_style_cue(vstyle, ...)` -- the existing single style authority | same | same |
| Subject | **THE CAST SIGIL** -- a frozen ~40-50 char identity string per (`episode_seed`, `char_id`, `style_id`): four durable tokens (silhouette, one asymmetrical landmark, one costume colour/item, one prop), distilled ONCE from the cast row (`portrait_prompt`/`appearance`/`character_description`, else `name`) at plan time, stamped in the ledger, **byte-identical on every beat for that character**, plus explicit gender from the cast row, never inferred from a name. Distilling fresh from 900-char cast prose per beat would DRIFT; the sigil is the cross-beat anchor a no-still lane otherwise lacks, and byte-identity makes it testable. | `announcer_subject_face` | `announcer_subject_object` / `open_subjects` via `get_open_subject` -- the console, per the "radio IS the host" canon |
| Framing lock | one centred subject, same costume/material, **never closer than mid-shot** (the face-soup guard) | face-forward console filling the frame | centred console/emblem |
| Action + motion | `resolve_motion_clause_text(shot)` -- **the authored clause rides WHOLE under its own <=130 contract**. *(Cursor's ~90-char cap refused: truncating an authored clause to meet a layer cap breaks the clause's contract for a saving the budget does not need.)* | **KINETIC DISTILL** of `motion_registers["announcer"]`: strip the `"Continuous shot, same console throughout."` opener (present in EVERY pack -- `paper_origami.json:27-29` -- it restates the shot-law slot and assumes the anchor a still used to guarantee), keep 2-3 verb phrases, ~90 chars | **KINETIC DISTILL** of `motion_registers[key]` via `_ltx_motion_role_key`, same opener strip, ~110 chars |
| Emotion | `traits`, then `beat_intent`, then `arc_phase` -- first available, one phrase | usually empty | cue mood/tempo |
| Story accent | ONE phrase from `meta.story_brief_terms.setting` / `.atmosphere` | atmosphere | cue `description`/palette |
| Shot law | "one continuous shot, same subject." | same | same |

**The lettering defence lives in the NEGATIVE, not the positive.** Negation in a
CLIP positive can mint the very thing it names. This is the **first lane in the
registry where the negative genuinely binds** (real CFG ~7; the distilled LTX
lanes at CFG 1.0 have inert negatives), so the guard is
`effective_negative(pack)` + a lane hygiene head (text, watermark, caption,
lettering, subtitles) and the positive shot law stays affirmative. Per G3.5 the
binding is verified against the sampler THIS lane selects (measurement #4).

**INTERLOCK, and it prices the backbone fork:** an AnimateLCM backbone at CFG 1-2
makes the negative INERT and forces this defence back into positive phrasing.
That cost enters the backbone choice alongside step count and module size.

**EXISTING-CODE HAZARD -- NEVER WIDEN.** `render_driver.py:2791-2795` appends
`", stable centered subject, full face clearly visible, generous headroom,
comfortably composed"` to non-bookend beats -- literally an invitation to this
lane's named failure mode. It is gated on `engine_id.startswith("ltx")`, which
naturally excludes `animatediff15_video`. **That condition must never be widened
to cover this lane.**

**Trim priority** (whole phrases, in order): story accent -> emotion ->
framing-lock adjectives (never the mid-shot floor) -> distilled register verb
phrases (music before announcer) -> stop. **The sigil, the motion clause and the
shot law are NEVER trimmed.** If those three alone exceed budget, the SIGIL
DISTILLER is too verbose -- fix the distiller, not the law.

**Never enters the prompt:** raw `lines[].text` (mints painted lettering,
metaphor objects, extra people -- the motion clause is the ONLY laundered
channel); `episode_title` and proper-noun meta; the M4 scene wall; a second
described character (SD1.5 bleeds two figures into a hybrid -- the other party
exists only as props and shadows); camera choreography as spent budget;
negation in the positive.

## 3. Coverage

### 3a. Primary -- one beat, one timeline
16 frames is the **context window**, not the clip length. Non-looped sliding
Context Options (Standard Static, length 16, overlap 4) process the whole beat in
ONE render, VRAM bounded by context length rather than beat length: a long beat
costs TIME, not memory. Satisfies R3 natively -- no chaining, no seams.
**UNVERIFIED upstream doc claim about a package that is not installed.
Measurement #1, before any adapter code.**

**Rate (R4).** `unique = ceil(D * 12.5)`, equivalently
`ceil(target_frame_count / 2)` off the audio-derived count; `delivered =
2 * unique`; at most one surplus delivered frame is tail-trimmed (a real
rendered frame, trimmed in order -- the `ltx_8gb` precedent,
`allow_tail_trim = True`). Worked: a 49-frame beat (1.96 s) -> 25 unique -> 50
delivered -> trim 1. Contract declares `native_fps == target_fps == 25` with the
hold-2 conform at delivery -- the Veo/H3 pattern G3.1 blesses, never a relabel.

**CONSEQUENCE THAT PROMOTES MEASUREMENT #1:** at 12.5 native even a standard 2 s
beat is 25 unique frames -- past the 16-frame window. **Every beat exercises
sliding context, not just long ones.** #1 is load-bearing for the whole lane.

`FrameContract` (hypothesis until #1): `min_frames` from #8; `max_frames = 0`
(unbounded -- time, not memory); `quantum = 1`; `allow_tail_trim = True`;
**`continuity = CONTINUITY_NONE` DECLARED EXPLICITLY** (G3.3 -- the lane-10
lesson: six lanes once inherited the right value because nobody decided it).
One deterministic seed per beat, f(`episode_seed`, beat_id).

### 3b. Fallback -- the panel grammar (contingent on #1 failing)
Beats partition into **cells** of one context window each, `CONTINUITY_NONE`,
joined as cuts landing on delivery holds. Across a beat's cells: byte-identical
pack cue + subject + framing lock + story accent; one named light direction; a
deterministic rotating panel directive as the ONLY varying slot (mid-shot ->
detail insert -> profile -- a cut that changes shot size reads as editing; a cut
holding framing while the subject redraws reads as an error). Distinct seeds per
cell, f(`episode_seed`, beat_id, cell_index) -- never the same seed twice in a
beat (same seed + same prompt renders the same motion twice: a loop wearing a
render receipt). Splits prefer punctuation, pauses, music accents, action turns.
Partition literals pinned per G3.4.

## 4. The three roles (all ship -- R2)

**Announcer.** One anthropomorphic radio host from `announcer_subject_face` --
dial-eyes, grille-mouth, era material per pack. The face is MECHANICAL, which is
exactly why it holds under instability: a dial that swims is still a dial.
Moving zones per `motion_registers["announcer"]`. No lip-sync on this lane.

**Music.** The console / signal-sculpture from `open_subjects` /
`announcer_subject_object`. No human. Widest motion range -- this is where the
lane's full instability spends itself. The subject stays the console (canon);
the ether around it dances.

**Character.** One **heraldic actor**: exact silhouette, one asymmetrical
landmark, one costume item+colour, one prop, explicit gender, mid-shot, fixed
angle, one action per beat. Identity judged on silhouette + costume + action; a
stable face is a bonus, never a requirement. One subject per beat, absolutely.
Shadow vocabulary (half in shadow, brim low, profile against the light) available
as face-bandwidth reduction, subordinate to the spine. **The motion aperture**
(masked `scale_multival`/`effect_multival` -- calm centre, mutating perimeter; a
MOTION mask, unrelated to the retired visual aperture) is the insurance policy --
measurement #6, dropped without ceremony if it fails.

## 5. Delivery -- ENLARGE, committed

**THE COMMITMENT: clean LANCZOS, unsharp DECOUPLED, full frame to 1920x1080,
edge to edge -- over a TWO-RUNG canvas whose default is the CHEAP rung.**

| Rung | Canvas | Enlarge | Status |
|---|---|---|---|
| **DEFAULT** | **384x216** | clean Lanczos **x5** | cheapest (82,944 px, 44% fewer than 512x288), true 16:9, /8-legal for the VAE |
| **QUALITY RESERVE** | 512x288 | clean Lanczos **x3.75** | training-scale, cleanest latent (36->18->9), G2.1-clean, `ltx_8gb` precedent |

**CANVAS PRIORITY INVERTED IN REV.4, AND THE REASON IS A REPO NUMBER.** Rev.3
committed to 512x288 arguing "weights dominate, pixel savings buy little." That
holds at 8 GB and gets shaky at 4 GB -- and `public_engines.py:93-94` records
`ltx_8gb` measured at **9,106 MB absolute / 6,835 MB net, cold, at
512x288x161**. Different architecture, not transferable -- but it destroys any
assumption that 512x288 is automatically cheap. Under R7 the cheap canvas
DEFAULTS and the expensive one must EARN its place.

**THE PROMOTION RULE.** Measurement #2 bakes off both rungs. Artifacts or
coherence collapse at 384x216 promote the reserve to default. If 512x288 then
busts the 4 GB budget, **the ceiling claim changes honestly rather than
quietly.** The default carries two declared empirical risks: the odd-latent
class (216/8 = 27 latent rows, odd at the FIRST UNet halving; 288/8 = 36
survives two) and below-training-scale coherence (384 long edge vs SD1.5's
~512). Its height 216 is not /32-legal, so the G2.1 exemption question is back
on the LIVE path (section 9) -- and if that answer is a hard no, this promotion
rule executes early.

**ONE COMMITTED SCALER, TWO CANVAS RUNGS, ONE PROMOTION RULE.** Rev.3's
nearest-grid reserve is RETIRED: the integer-multiple rule was a
nearest-neighbour rule, and under Lanczos x5 and x3.75 are equally legal.

Original rev.3 wording, kept because the scaler decision is unchanged: No bars, no pad, no crop, no stretch, no second
diffusion pass, no interpolation (RIFE excluded from v1; later admissible only
as a duration-preserving A/B, never as coverage).

Native 1080p is off the table as hard fact: SD1.5 duplicates features above ~768
on the long edge, and 1080p is far outside R7. So "full 1920x1080" means
"rendered small, brought up honestly."

**Enlarge, not upscale (R8).** A super-resolver's whole job is restoring
sharpness the signal never had -- precisely the lie this lane exists not to
tell. Invented detail also fails acceptance (a)'s round-trip probe BY
CONSTRUCTION. And a single-image upscaler makes **per-frame edge decisions that
shimmer** -- hallucinated instability stacked on authored instability. SeedVR2
is refused in the same breath: temporal awareness fixes the shimmer, not the
lie, and a heavyweight restorer on a minimum-VRAM lane is off-brief twice.

**Lanczos, and the unsharp decoupling is a HARD REQUIREMENT.** Among accurate
filters Lanczos preserves the most of what was really rendered (bilinear
discards rendered signal -- its own small dishonesty). Today `_scale_filter`
couples Lanczos to an `unsharp` pass whose stated purpose is recovering
softness: it manufactures edge contrast that was never rendered, fights the
identity, and would fail the round-trip probe.

**BUILD REQUIREMENT.** `otr_silent_composite._scale_filter`
(`nodes/otr_silent_composite.py:172`) offers exactly two modes today:
lanczos+unsharp (`sharpen=True`) and bilinear (`sharpen=False`). Ghost Signal
needs a **third: Lanczos, NO unsharp, NO pad** -- shipped in the same change as
the adapter, selected by an explicit lane-declared capability, NEVER inferred
from an engine-id string match. (A full-frame 16:9 source pads to nothing
anyway, so R5 is satisfied at the source.) Note this is a SMALLER change than
adding a nearest mode: it decouples two existing flags rather than adding a
filter.

**The activation arithmetic behind the inversion:** 512x288 costs ~1.78x the
temporal-attention memory of 384x216 (64x36 = 2304 latent tokens per frame vs
48x27 = 1296, times 16 context frames). With ~3.8 GB of weights against a 4 GB
ceiling there is almost no headroom, block-offloading is forced, and activations
bind.

**NO UPSCALER.** The upscale engine stays `off`. Beyond R8, two repo-grounded
reasons: `spandrel_esrgan`'s VRAM is **UNMEASURED** (the `~64 MB` in the adapter
is the CHECKPOINT, not VRAM), and Real-ESRGAN x2plus is a **live suspect in an
unresolved artifact the operator personally reported**
(`docs/GO_FORWARD_PLAN.md:1136` -- mesh at 00:37 of `beneath_the_silvery_boughs`;
the raw 832x480 render was CLEAN, the mesh appeared in the 1920x1080 composite;
still unpinned because the ledger never records which upscale engine ran).

**`RealESRGAN_x4plus_anime_6B` is REFUSED as a per-pack exception**, and the
grounds are worth keeping: it is still invention (it draws edges never
rendered); it is single-image, so its edge decisions shimmer frame to frame on a
deliberately unstable surface; and a lane whose delivered surface changes class
per pack has two identities, which is none. **The operator is not blocked:**
`spandrel_esrgan` is already a registered explicit-pick `upscale_stage` engine,
so he can route an anime_6B pass himself as his own experiment. Reopening this
as a lane default needs a measured duration-preserving A/B that does not fail
the round-trip probe's spirit -- AND the 1136 artifact pinned first.


## 6. The ledger

Priority: (1) role + authoritative `char_id`; (2) **THE CAST SIGIL** -- minted
ONCE per (`episode_seed`, `char_id`, `style_id`) by a plan-time batch pass (the
`generate_motion_clauses` pattern: ledger-stamped, render path read-only),
byte-identical thereafter -- plus gender; (3) **beat duration** -- `target_frame_count` is
the coverage authority; (4) `motion_clause`; (5) `traits` / `beat_intent` /
`arc_phase`; (6) one setting/atmosphere phrase; (7) music mood/tempo/palette;
(8) at most one meaningful prop.

**Motion thermostat.** Dramatic pressure maps to motion scale: calm / setup /
resolution -> restrained; rising tension / discovery -> baseline; conflict /
urgency / climax -> stronger mutation; music -> widest. Numeric range
(~0.95-1.15) is a hypothesis -- measurement #5.

**Traps** (enforced at the composer): raw dialogue; `episode_title` and proper
nouns; the M4 wall; a second described figure; negation in the positive; camera
flourishes as budget spend.

## 7. Style packs -- compose

**Pack owns:** medium, material, palette, lighting, character/radio rendering
vocabulary, surface-movement accent, negative conditioning (`negative_tail` via
`effective_negative` -- never a universal anti-cartoon or anti-photo negative
that fights the chosen style).
**Lane owns:** the full-frame coarse pixel grid, one dominant form, imperfect
coalescence, kinetic intensity tracking the audio.

Paper Origami refolds; Storybook Engraving re-etches; Media Archive loses and
regains emulsion; Anime redraws contours; Video Art blooms through phosphor.
Nine palettes, one transmission. The lane's identity lives in the framing lock,
the delivery chain and the motion system -- **never in an engine-side style
constant** (the PBUG-20260817-01 defect class). The prompt head is mediumless by
construction, so the pack's medium words never fight it.

## 8. MEASUREMENTS -- ordered by KILL POWER

Run order is chronological #3 -> #2 -> #1 -> rest; the numbering is kill power.
The preflight receipt cites these by number.

| # | Measures | Settles | If it fails |
|---|---|---|---|
| 1 | Sliding-context VRAM bound: 25 / 63 / 150 unique frames at the DEFAULT canvas (384x216), Standard Static 16/4 non-looped; peak VRAM + wall clock per rung, **against the 4 GB ceiling (R7)** | 3a vs 3b -- the central structural fork. Also mints the `low`/`high` token (G7.4) | 3b activates as written; no redesign |
| 2 | **Enlargement chain + TWO-CANVAS VRAM bench (R7).** Render at 512x288 AND 384x216; record peak VRAM for both against the 4 GB ceiling. Run the decoupled clean Lanczos on each (x5 from 384x216, x3.75 from 512x288) to 1920x1080; eyeball both on a real display; inspect the odd-latent class (27 vs 36 latent rows) and below-training coherence at 384; run the ROUND-TRIP PROBE (downscale delivered -> native, compare to the rendered frame) and set acceptance (a)'s SSIM/PSNR threshold from the clean-chain baseline | the DECLARED canvas under the section 5 promotion rule, the third `_scale_filter` mode's correctness, and the acceptance threshold | **Artifacts or coherence collapse at 384x216 promote 512x288 to default. If 512x288 then busts the 4 GB budget, the ceiling claim changes honestly rather than quietly.** |
| 3 | Install; capture real class names + input signatures from a live `/object_info` (the wan_ti2v / ltx_8gb pattern -- **no adapter code before this**). Confirm `mm-p-0.5` loads; inventory v2/v3, AnimateLCM, Lightning **and their file sizes (R7)** | the backbone: module, size, steps, CFG, and the licence question's shape | v3, then AnimateLCM -- knowing AnimateLCM's CFG 1-2 kills the live negative (#4) |
| 4 | CFG / sampler / negative liveness (G3.5): read the selected sampler; confirm CFG ~7 fits budget at 8-12 steps; A/B the lettering guard | whether the lettering defence lives in the negative or must move to positive phrasing | harden positive-side traps, reword shot law, record the departure |
| 5 | Subject stability + thermostat range at 2 s and 6 s | the character recipe and the real thermostat numbers | role still ships (R2); hardens to tightest band + shadow vocabulary + #6 |
| 6 | Motion aperture at the declared canvas (mask rectangular for 16:9) | whether it ships in v1 | dropped, no redesign |
| 7 | Tokenize real prompts, all 3 roles x 9 packs, with the installed CLIP tokenizer | the 320 constant and trim aggressiveness | tighten budget + distiller; the window does not move |
| 8 | Short-beat floor (a 0.5 s beat = 7 unique frames) | `FrameContract.min_frames` | min_frames refuses below-floor beats loudly at plan time |
| 9 | Cadence eyeball at hold-2 / 25 fps | nothing structural (R4 is ruled) -- tunes motion-scale defaults | -- |

**R7 note:** estimated fp16 weights are SD1.5 UNet ~1.7 GB + motion module
~1.7 GB + CLIP ~0.25 GB + VAE ~0.16 GB = **~3.8 GB before activations.**
Sequential load/unload is MANDATORY (encode -> free; sample -> free; decode ->
free), `MotionEngineBase`'s V-4 patcher-detach teardown carries more weight here
than on any existing lane, and `unload_all_models` is still never the mechanism.
Measurement #1 replaces every figure in this paragraph.

## 9. Open questions for the wiring round

- **G2.1 exemption -- BACK ON THE LIVE PATH (rev.4 canvas inversion).** The
  DEFAULT canvas 384x216 has height 216, which is not /32-legal. Determine
  whether the enforced matrix admits a declared, documented exemption (G2.3's
  dead-channel language is the nearest precedent), or whether the gate needs
  amending with the operator-facing reason. **Must be answered before the canvas
  is declared; a hard no executes the section 5 promotion rule early.**
- **The Cast Sigil's stamping home:** which plan-time pass mints it and which
  ledger field carries it (cast-row sidecar vs shot row). The
  `generate_motion_clauses` batch-pass pattern is the template; byte-identity
  across beats gets a pinned test.
- **Role key string.** `role_compat.py:34` says `character_video`; profiles
  write `character_visual`. Confirm what the selector keys on. Cheap to check,
  expensive to guess.
- **Base class.** `MotionEngineBase` gives the AS-3 lease and V-4 teardown this
  lane wants -- but ALSO `accepts_still = True` by inheritance (must be
  overridden to a loud `False` with its reason) and `compute_real_frame_budget`'s
  loop/ping-pong extension path, which **must never run** (R3). Subclass-and-
  override vs a leaner parent is a wiring decision; the invariant is that
  neither inherited behaviour survives, provably.
- **Family row.** `text_to_video` per R1 -- confirm `schemas.py` semantics and
  the `content_oracle` row for a no-still, no-lip-sync lane.
- **`still_plan` for a no-still lane (G7.4)** -- what a declared-empty plan
  looks like so the audit passes OUT LOUD rather than by omission; the viz lanes
  are the nearest precedent.
- **Third `_scale_filter` mode selection** -- how the composite learns the lane
  wants CLEAN LANCZOS (no unsharp): capability flag, clip-manifest field, or
  request stamp. Explicit and declared; never an engine-id string match.
- **`render_aspect`** -- both rungs are true 16:9 with full-frame delivery, so the
  plain declaration is `"wide"`, matching the cheap families. Confirm the field's
  consumers agree.
- **Licence.** AnimateDiff-Evolved is permissive; a usable SD1.5 checkpoint is
  CreativeML OpenRAIL-M -- use-restricted, not OSI-open, on the lane a stranger
  with a weak card runs FIRST when the repo goes open source. Attestation doc per
  the H3 pattern, or an Apache/MIT SD1.5 derivative. Blocked on #3.
- **Boot/profile surfaces.** G2.3 (profile canvas matches the DECLARED canvas or the dead
  channel is documented), G6.2 boot-requirement probe, G4 admission envelope key -- all
  mechanical once #1/#2 receipts exist; listed so none ships by default.

## 10. New preflight gate proposed with this lane

`docs/VIDEO_LANE_PREFLIGHT.md` mentions **prompt / style / ledger / motion_clause
ZERO times**; "role" appears once, inside G3.6. Prompt composition is unpoliced
-- tolerable while a still carries the subject, fatal on a no-still lane. Per the
doc's own rule (*"every gate exists because a real lane failed it"*), Gate 9 is
written in the same change:

- **G9.1** A video lane fills all three visual roles, the prompter composing for
  each; fewer is declared with its reason. *(R2.)*
- **G9.2** Subject ownership is DECLARED: the still, or the prompt.
- **G9.3** The prompt budget is DERIVED, not inherited.
  `_LTX_MOTION_PROMPT_MAX = 240` is ours and is motion-only-shaped.
- **G9.4** The style-pack join is declared: compose, override, or ship a pack.
- **G9.5** Ledger reads are read-only and named.
- **G9.6** A no-still lane's receipt DECLARES its **composer id** and
  **`subject_ownership = prompt`**, so `_LTX_MOTION_PROMPT_MAX` and the
  still-carried composition can never be accidentally reused on it.
