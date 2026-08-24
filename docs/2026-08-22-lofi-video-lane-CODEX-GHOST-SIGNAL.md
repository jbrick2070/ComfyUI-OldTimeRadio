# CODEX -- "Ghost Signal" (operator-driven pass, 2026-08-22)

Verbatim record of the Codex lane's developed answer, delivered by the operator
running Codex manually. Preserved as the panel record. The driver's grounding
notes are in square brackets at the end; the body is Codex's own.

Dropdown ID: `animatediff_p05_video`
Display label: **AnimateDiff p0.5 -- Ghost Signal**
Spoken name: **Ghost Signal**

It is not "budget cinema." It is a genuine kinetic option for low-VRAM users:
the radio broadcast appears as an imperfect moving transmission.

## A. The look

A coarse square receiver aperture floats in a black 1920x1080 field. Inside it,
one dominant figure, radio, prop, or emblem coalesces from noise, becomes
briefly readable, and partially loses itself again.

Contours crawl. Paper folds breathe. Ink hatching redraws itself. Colors
misregister. Shadows become objects. The amount of instability follows the
dramatic pressure of the audio.

The dangerous failure is semi-photorealistic face soup. A melting close-up
contradicts the trusted voice, hijacks attention, and looks like failed AI
cinema. Ghost Signal therefore favors a mid-shot "heraldic actor":
recognizable through silhouette, hair, costume, asymmetrical feature, and one
persistent prop. A stable face is a bonus, not a requirement.

## B. Why it belongs to radio

A clean video declares, "This is exactly what happened." Ghost Signal says,
"This is the picture the broadcast stirred up."

That leaves imaginative room around the audio. `still_pan` moves over a
completed illustration; Ghost Signal lets anger, dread, wonder, and musical
energy transform the picture itself.

It belongs as an experimental option, not as the default or a replacement for
`still_pan`. Its acceptance test is expressive transformation -- not facial
realism.

## C. Prompt composition

The existing 240-character budget is wrong for this lane. It assumes a still
already carries the subject. That is visible in the current motion-only role
prompts (`nodes/_otr_video_engines/render_driver.py:1327`).

Use a dedicated provisional limit of **320 characters and approximately 65
encoder tokens**, targeting 260-300. Verify the real tokenizer after AnimateDiff
is installed.

Actual order:

> **Pack cue -> subject identity -> framing lock -> visible beat action ->
> motion -> emotion -> one story accent -> shot law**

Sources:

- Pack cue: existing compact cue derived from the selected visual style.
- Subject:
  - Character: `char_id` -> cast appearance, distilled into one invariant
    identity spine.
  - Announcer: `announcer_subject_face`.
  - Music: `announcer_subject_object` or `open_subjects`.
- Framing: one centered subject, same costume/material, mid-shot or closer.
- Motion:
  - Character: the existing per-beat motion clause
    (`nodes/_otr_motion_clause.py:41`).
  - Announcer/music: the pack's role-specific `motion_registers`.
- Emotion: `traits`, then `beat_intent`, then `arc_phase`.
- Story accent: one setting, atmosphere, palette, or lighting phrase.
- Ending: "one continuous shot, same subject, no text."

Example:

> Storybook engraving. One older woman with silver braid, maroon coat and brass
> key; chest-up, centered, same costume. Rain-dark station platform. She recoils
> and grips the key. Crosshatching crawls as her shadow folds inward. One
> continuous shot; no text.

Protect identity and motion when trimming. Drop extra atmosphere and camera
flourishes first. Do not paste raw dialogue, the entire logline, several
objects, or multiple actions into the video prompt.

The selected pack's `negative_tail` remains separate; never add a universal
anti-cartoon or anti-photograph negative that fights the chosen style.

## D. One animation per beat

**One beat should equal one AnimateDiff timeline.** Sixteen frames is the
context window, not necessarily the clip length.

For a beat lasting `D` seconds:

- Generate approximately `ceil(D x 8)` unique native frames.
- Start with non-looped Standard Static context, length 16, overlap 4.
- Process the complete beat through sliding windows.
- Conform that same duration to the 25 fps delivery timeline.
- Never loop, mirror, hold the tail, or slow the frames down.

AnimateDiff Evolved's official documentation says Context Options extend
generation beyond the usual 16-frame sweet spot while bounding VRAM around the
context length, and explicitly recommends Standard Static when looping is
unwanted. Longer beats should therefore cost more time, not necessarily more
peak VRAM.

If measurement reveals a hard practical ceiling, split only at punctuation,
pauses, music accents, or action turns. Make the seam an honest tuning
jump-cut. Repeat the identical style/identity/setting spine, change only the
action phase, and use distinct deterministic segment seeds.

I would leave RIFE out of v1. The hard 8 fps cadence suits the handmade identity
and costs less. RIFE can later be a duration-preserving A/B -- not a means of
making two seconds cover six.

## E. What the ledger contributes

In priority order:

1. Role and authoritative `char_id`
2. Cast appearance: three durable identifiers
3. Beat duration: determines genuine frame count
4. `motion_clause`: the physical movement
5. `traits`, `beat_intent`, and `arc_phase`
6. One setting or atmosphere phrase
7. Music mood, tempo, or palette when present
8. At most one meaningful prop

The ledger should also act as a **motion thermostat**:

- Calm/setup/resolution: restrained motion
- Rising tension/discovery: baseline motion
- Conflict/urgency/climax: stronger mutation
- Music: widest permissible motion range

The tutorial's approximate 0.95-1.15 range is worth testing, but those numbers
remain hypotheses.

Raw dialogue should influence the already-generated motion clause, not be
serialized again. Literal dialogue invites unwanted lettering, metaphor objects,
and extra people.

## F. Style packs

Ghost Signal composes with the selected pack. It does not override it and is not
a tenth pack.

The pack owns: medium and material; palette and lighting; character/radio
rendering vocabulary; surface movement; negative conditioning.

Ghost Signal owns: square receiver aperture; one dominant form; coarse pixels;
imperfect coalescence; kinetic intensity; black surrounding field.

So Paper Origami folds and refolds; Storybook Engraving continually re-etches
itself; Media Archive loses and regains its emulsion; Anime redraws its ink
contours; Video Art Feedback blooms through phosphor echoes. The available
surfaces are grounded in the existing visual-style schema
(`nodes/_otr_visual_styles.py:173`).

## G. The three roles

| Role | Ghost Signal treatment |
|---|---|
| Announcer | One anthropomorphic radio host. Dial-eyes and grille-mouth may hold surprisingly well because the "face" is mechanical. Dials, grille and tubes are the moving zones. |
| Music | A central radio or signal sculpture. No human required. Palette, music position and pack motion register drive the largest, most abstract mutation. |
| Character | One heraldic actor: exact hair/silhouette, one asymmetrical landmark, one costume item/color, one prop. Mid-shot, fixed angle, one action. Face present when possible, but identity never depends on facial pores. |

A procedural **motion aperture**: calmer motion in the center, stronger mutation
toward the perimeter. That gives a face or silhouette a chance to hold while the
surrounding ether remains lively. AnimateDiff Evolved officially supports masked
`scale_multival` and `effect_multival`; whether this specific treatment works at
256x256 is still unverified.

## H. Cheap 1080p delivery

Use nearest-neighbour as part of the identity:

- 256x256 native
- Exact 4x nearest enlargement to 1024x1024
- Center on 1920x1080
- Black margins: 448 pixels on each side and 28 above/below

That becomes a dark room around the receiver aperture. No crop, stretch,
blur-fill, Lanczos, unsharp, or second diffusion pass. The lane declares its
poverty proudly instead of disguising it.

## Tutorial verdict

Keep: motion separated from aesthetics; motion-scale experimentation;
non-looped sliding context windows; simple motion masks; deterministic queued
generation from every ledger beat.

Reject for this lane: looped Uniform context; zero-prompt/IPAdapter generation;
reference-image randomizers; hi-res second KSampler; mid-render foreign-style
reskinning; RIFE in the initial version.

Everything AnimateDiff-specific -- actual VRAM, viable long-beat length, face
stability, mask behavior, `mm-p-0.5` character, and the 320-character prompt
sweet spot -- remains a hypothesis until installation and a live bakeoff. The
creative design itself is settled.

---

## DRIVER GROUNDING NOTES (checked against the real Windows files)

- `announcer_subject_face`, `announcer_subject_object`,
  `announcer_subject_ltx_mouth` -- ALL THREE EXIST (`_otr_visual_styles.py:89-91`,
  dataclass fields at 189-191, real values in every pack). Codex's choice of
  `_face` for the announcer is CORRECT and better supported than Fable's
  `_object`: no-lip-sync rules out `_ltx_mouth`, it does not rule out `_face`,
  and a face built from dials and a grille is drift-tolerant in a way a human
  face is not.
- `_otr_visual_styles.py:173` really is `class VisualStyle:`. CORRECT.
- `_otr_motion_clause.py:41` really is `CLAUSE_MAX_CHARS = 130`. CORRECT.
- `beat_intent` / `arc_phase` / `traits` / `char_id` -- all real ledger fields
  (7 / 9 / 21 / 43 modules). CORRECT.
- Delivery arithmetic CHECKED: 1920-1024=896 -> 448 a side; 1080-1024=56 -> 28
  top and bottom. CORRECT. 256 is /32-legal (G2.1 passes).
- **`_scale_filter` has NO nearest mode today** -- only lanczos+unsharp
  (`sharpen=True`) or bilinear (`sharpen=False`),
  `nodes/otr_silent_composite.py:172`. Section H REQUIRES a third mode.
- **Context Options VRAM claim is UNVERIFIED** -- an upstream doc statement
  about a package that is not installed. The whole "one beat, one timeline"
  design rests on it. Measurement #1 after install.
- **RATE SUPERSEDED BY OPERATOR RULING:** 12.5 fps native + uniform hold-2 =
  exact 25 fps delivery, NOT 8 fps. `ceil(D * 12.5)` unique frames.
- **NAME AMENDED BY THE DRIVER:** register `animatediff15_video`, keep "Ghost
  Signal" as the label. `animatediff_p05_video` embeds the motion module in the
  public id; swapping `mm-p-0.5` later would make the id lie or need a legacy
  alias, and `_PUBLIC_ENGINES` trips a bijection assert at IMPORT.
