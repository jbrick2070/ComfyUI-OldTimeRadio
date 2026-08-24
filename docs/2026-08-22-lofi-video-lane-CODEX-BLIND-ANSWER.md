# Recommendation: “signal-dream” text-to-video

This should be an opt-in artistic lane, not a replacement for `still_pan`, `still_motion`, or LTX. Those existing lanes optimize coherence and economy; this lane optimizes movement, abstraction, and intentional instability.

The repository grounding is clear: style packs already own look and motion registers, motion clauses are capped at 130 characters, the current 240-character LTX prompt is motion-only because a still carries the subject, and delivered clips must use `extension_mode="none"` with no ping-pong or padding.

AnimateDiff is not installed here. Therefore its node classes, actual context window, legal frame counts, tokenizer behavior, native resolution, seed continuity, and VRAM usage are unverified hypotheses.

## A. The look

Call the visual identity a **signal-dream flipbook**.

On screen, it should resemble a tiny hand-made image transmission: coarse square frames, limited detail, unstable silhouettes, slight subject mutation, color bleed, exposure pulses, and abrupt but expressive changes between short rendered sections. The viewer should feel that the radio drama is projecting fragments of a dream through a failing broadcast monitor.

The image must still have one readable anchor per beat:

- one radio console for announcer beats;
- one physical broadcast object for music beats;
- one simplified character silhouette for character beats;
- one dominant action at a time.

It must never look like polished cinematic text-to-video, clean commercial animation, or a generic AI slideshow. The dangerous failure is not excessive flicker. It is **unanchored randomness**: images that mutate so completely that the viewer cannot tell whether the beat shows a person, a radio, or an object. At that point the lane loses the audio-picture relationship, and `still_pan` wins decisively.

## B. Why it belongs to radio

Radio already supplies the important continuity: voice identity, words, timing, and imagination. The picture does not need to explain every object literally. A dreamlike image can act as a visual echo of the performance—an emotional pulse rather than a second screenplay.

A clean, stable visual often asks the viewer to inspect it as an illustration. This lane should instead glance, smear, pulse, and recede while the listener follows the voices. Its instability becomes analogous to memory: the listener knows what happened, while the image only leaves impressions.

The honest verdict is that `still_pan` remains better for:

- zero VRAM;
- subject coherence;
- faithful visual continuity;
- dependable character identity.

This lane earns its place only when the operator prefers the experimental broadcast texture over those advantages. It should never become the default or silently replace a still lane.

## C. The prompt composition

The current 240-character shape is wrong for this lane, not merely too small. It was designed for prompts where the still supplies the subject. A no-still prompt must carry subject, style, scene, and motion together.

I recommend a separate **420-character positive-prompt budget** for this lane. That is a repository design budget, not an AnimateDiff engine limit. The actual installed graph must later measure its tokenizer and enforce the smaller of the two limits.

The assembly order should be:

1. **Style anchor**  
   A compact cue from the selected style pack, plus one role-specific look fragment. For the default `sci_fi_radio` pack, the lane must use a role look directly because the shared compact-style helper intentionally returns an empty cue for that default.

2. **Subject lock**  
   The actual thing being shown:
   - announcer: the pack’s `announcer_subject_face`;
   - music: the pack’s radio/object subject;
   - character: the cast identity plus a short, stable visual descriptor.

3. **Story core**  
   A short setting or situation fragment from the story brief and `story_brief_terms`. This should describe where the beat is happening, not reproduce the dialogue.

4. **Kinetic action**  
   - character beats: the resolved per-beat `motion_clause`;
   - announcer and music beats: the selected style-pack `motion_register`;
   - beat intent and arc phase may modulate intensity, but should not replace a physical action.

5. **Continuity register**  
   A compact positive instruction such as “single centered subject, continuous action, same figure throughout.” It should describe what is present, not rely on prohibitions.

In prose, the shape is:

“[style and role look]. [stable subject]. [short scene core]. [physical action]. [simple continuity and framing].”

The existing `positive_tail`, `portrait_look`, `announcer_subject_*`, and `motion_registers` fields are source material, not text that should be pasted wholesale. Long style tails would consume the room needed by the subject and action.

The budget should protect the subject and action first. If the prompt is too long, remove atmosphere and extra era wording before shortening the subject lock or motion clause. Never cut a motion sentence in the middle.

A separate negative prompt may use the pack’s `negative_tail` only if the actual AnimateDiff graph has a real negative-conditioning path. That graph is not present, so this is unverified. No negative vocabulary should be inserted into the positive prompt as a substitute.

## D. How many animations per beat

Use chained real renders.

A 2-second beat receives one 16-frame render at 8 fps. A 7.3-second beat receives four real renders: approximately 2 + 2 + 2 + 1.3 seconds. The final section must be requested or trimmed to the actual remaining duration; it must never be filled by cloning, looping, or mirroring.

A long beat should not be one enormous low-VRAM render. Several short renders are safer for memory and more compatible with the proposed 16-frame recipe. They will have visible seams, but those seams are part of the signal-dream identity.

Continuity comes from text, not from an image handoff:

- the same subject lock is repeated in every segment;
- the same setting and framing are repeated;
- the beat’s action is divided into start, continuation, and resolution phases;
- only the phase phrase changes between segments;
- each segment gets a deterministic segment-specific seed.

The last seed rule is a design hypothesis, not an AnimateDiff fact. Reusing one seed may repeat the same opening pose; varying it may produce healthier mutations. That must be measured after installation.

No previous frame, latent, still, last-frame condition, or generated image may cross the segment boundary. Each segment is genuinely text plus noise. The assembled manifest should report `extension_mode="none"` and native frame counts.

## E. What the ledger should contribute

Priority order:

1. **Role and subject identity**  
   `shot.role`, `char_id`, and the resolved cast identity. Without a still, this is the most important information.

2. **The physical motion**  
   `video.shots[i].motion_clause`, when valid. This is more valuable than a general emotional adjective because it tells the model what changes on screen.

3. **Scene and setting**  
   A compact setting fragment from the story brief or `story_brief_terms`.

4. **Beat intent and arc phase**  
   `beat_intent` and `arc_phase` should influence the scale and direction of movement: a revelation might turn or advance; rising tension might recoil, reach, or rush.

5. **Speaker and dialogue semantics**  
   Use one or two visually useful nouns from the line, never the whole line. The spoken text is for the listener, not a caption-generation instruction to the image model.

6. **Atmosphere**  
   One short mood or lighting term only after the subject and action are secure.

The major traps are:

- using a character name without a visual descriptor;
- quoting the dialogue, which invites painted lettering or literal text;
- putting several speakers into one prompt and creating an accidental crowd;
- spending the entire budget on the setting;
- writing “sad,” “angry,” or “urgent” without a body action;
- saying “keep the same face” without repeating concrete clothing, silhouette, or prop anchors;
- using prohibitions whose words may become positive visual conditioning.

The current ledger’s character name alone is not enough for reliable no-still identity. If the existing cast record does not already contain a stable visual descriptor, this lane needs a text-only character lock generated once per episode and reused across that character’s beats. That is a proposed design requirement, not a field I verified in the repository.

## F. Style packs

Compose with the selected pack. Do not override it, and do not create a tenth pack.

The nine packs already define the episode’s visual language. They contain role-specific subjects, portrait looks, open subjects, and motion registers. The new lane should respect those choices:

- `anime` remains anime, but rendered as an unstable low-resolution signal;
- `paper_origami` remains paper, but its folds may flutter and mutate;
- `archival_documentary` remains archival, but its image may slip like damaged film;
- `video_art` remains feedback-driven, but the lane’s coarse segmentation supplies the physical degradation.

The lane’s own identity comes from process:

- text-to-video from noise;
- 256×256 native target;
- 16-frame units;
- 8 fps;
- intentionally loose motion;
- repeated textual subject locks;
- nearest-neighbour delivery scaling.

That process identity is strong enough to make the lane recognizable even when the pack changes. A tenth pack would confuse two different authorities: the episode style pack and the engine’s rendering grammar.

## G. The three roles, by prompt alone

### Announcer

Use the pack’s radio-face subject, not a human announcer. The prompt should repeatedly establish one radio console, its dial-eyes, grille-mouth, cabinet material, and centered position.

Use the pack’s `announcer` motion register. Emotional intensity should affect the dial, grille, light, or camera pressure—not replace the radio with a new subject.

The announcer should feel like the station itself is speaking.

### Music

Use one physical broadcast object: radio console, dial, grille, tubes, meter, or related pack-defined object. Do not make music beats human faces.

Use the appropriate `music_open`, `music_inter`, or `music_close` register. Let the motion respond to the music through visible physical changes: needles, tubes, meters, glow, vibration, or a slow object orbit.

The goal is a visual instrument, not a literal music video.

### Character

This is the hardest role and must be deliberately constrained.

For a two-second pure-noise render, the prompt should contain:

- one named character;
- one stable silhouette;
- one persistent clothing or hair feature;
- one distinctive prop or accessory;
- one simple location;
- one visible action;
- a centered medium or waist-up composition.

For example, an illustrative prompt might describe “one woman with a long dark braid, a brass key at her collar, turning sharply toward the door and raising the key in the same centered medium figure.” The exact identity must come from the episode, not this example.

Use `portrait_look` or `portrait_look_talking` from the chosen pack, then place the motion clause immediately after the identity and scene. Avoid crowds, extreme close-ups, complicated hand actions, multiple simultaneous actions, and face-only shots. At 256×256, the lane should preserve a recognizable silhouette and prop before it attempts a stable photorealistic face.

The lane cannot promise perfect facial identity from noise alone. Its continuity acceptance should therefore judge “same character silhouette, costume cue, and action” rather than demand still-image-level facial identity.

## H. The scale to 1080p

Choose **nearest-neighbour**.

Lanczos is already the repository’s normal real-engine upscale path, with unsharp sharpening. That is useful for clean footage but would soften the defining poverty of this lane. Nearest-neighbour makes the 256×256 origin visible as hard, chunky pixels.

Scaling is part of the lane’s identity, not merely delivery plumbing. The output should look like a small signal window enlarged beyond its means.

The square render will be fitted into the 1920×1080 delivery canvas with black side pillars because the existing aspect-preserving composite pads after fitting. That is acceptable and should be intentional: the viewer sees a centered broadcast window rather than a falsely stretched image.

The 8 fps generation cadence must be conformed to the 25 fps delivery timeline without changing duration. Cadence conversion is not permission to loop, ping-pong, or clone a beat’s tail.

## I. The name

Machine-facing lane identifier:

**`animatediff_mm-p-0.5_t2v`**

Operator dropdown label:

**AnimateDiff mm-p-0.5 — Lo-fi T2V**

Spoken name:

**“the Dream lane.”**

The name identifies the AnimateDiff family, the proposed motion module, and the text-to-video capability while deliberately omitting the low/high tier token. That token should only be added after a real VRAM measurement receipt exists.