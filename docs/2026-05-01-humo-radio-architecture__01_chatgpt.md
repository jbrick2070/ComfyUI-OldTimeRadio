# Round A -- ChatGPT (gpt-5.4) elapsed=62.9s

Short version: **yes, your reading is almost certainly correct**, and I would **commit to Option 1**, with one refinement: **ANNOUNCER becomes a real cast role, and all non-dialogue segments (`music_*`, `sfx`) use deterministic static-video treatment in post, not HuMo**. Keep the radio as a **bookend/interstitial visual motif**, not as a HuMo subject. That is the smallest architectural correction that aligns with what HuMo actually is, preserves C7, stays well inside VRAM, and avoids inventing a second fragile video stack.

---

## 1. Is your reading of `WanHuMoImageToVideo` correct?

Based on the behavior you described and the stock node pattern you cited, **yes, I think your reading is correct**:

- `ref_image` is being **VAE-encoded**
- then attached to the **positive conditioning**
- via something like `conditioning_set_values(..., {"reference_latents": ...}, append=True)`
- while the actual generation latent starts from **zeros**, not from an encoded init frame latent

That is **not** “strict image-to-video where frame 0 is the source image.”  
That is a **reference-conditioning path**.

Your summary matches the architecture of a **reference-guided talking-head model**, not a generic I2V model.

### Why the behavior you saw is exactly what I’d expect
If the model is trained for **audio-driven face animation**, then:

- a face image gives it identity structure to preserve
- a non-face image gives it little or nothing useful in the identity path
- so it falls back to:
  - prompt prior
  - model prior
  - seed variation

That explains why your two ANNOUNCER clips became **two different generic women** despite the ledger claiming the radio still was used.

### What would change my mind?
Only one of these would:

1. **Upstream docs explicitly state** `ref_image` is supported for non-human object animation in HuMo  
2. The node code also injects the image into the **initial denoise latent** or a **first-frame constraint path**
3. Reproducible tests show a non-face object is preserved across seeds and clips

From what you’ve described, I do **not** see evidence of any of those.

### Confidence / uncertainty
- **High confidence** that `ref_image` in stock `WanHuMoImageToVideo` is **not** a first-frame I2V input
- **Moderate confidence** that HuMo cannot reliably animate a radio still as the primary subject
- **Low confidence** that there is any hidden stock-node mode that makes non-face references work well enough for your architecture

So: **I agree with the bug diagnosis.**

---

## 2. Which option would I commit to?

## I would commit to **Option 1**, not 2 or 3.

More bluntly: **the “wall-to-wall HuMo coverage” premise is wrong for non-dialogue visuals and should be retired now**. Don’t try to save it with another generative branch unless you absolutely must.

### Why Option 1 is the right fix

#### A. It matches the actual capability boundary
HuMo appears to be good at:
- face-anchored talking-head clips
- identity-preserving speech animation
- portrait-like source imagery

HuMo appears **not** to be good at:
- object animation
- radio-set-as-performer
- generic still-to-video

Option 1 accepts reality instead of fighting the model.

#### B. It is the smallest architectural correction with the biggest payoff
You asked for the smallest change with the largest payoff. That is Option 1.

You mostly need to:
- treat **ANNOUNCER as a cast member**
- reroute speaker-role resolution in `nodes/_otr_speaker_role.py`
- stop sending radio stills into HuMo
- add a deterministic static-video/bookend path in post

That is much smaller than:
- adding LTX orchestration
- managing sequential model phases
- validating another video model under your VRAM ceiling
- debugging timing and mux edge cases across two render families

#### C. It is safest under C7
Option 2 introduces a second render pipeline and more mux complexity. That raises the chance of accidentally perturbing:
- clip durations
- concat order
- audio stream copy behavior
- sample alignment
- final mux determinism

Option 1 can be implemented so that **audio path remains untouched** and video is just a deterministic companion stream.

That is exactly what you want under **“audio is king”** and **byte-identical audio**.

#### D. It fits the genre better than you may think
Old-time radio drama visually reads well as:
- host at microphone
- cast portraits for dialogue
- occasional radio set / studio cutaways
- static or slow-moving interludes during music/SFX

That is actually **more period-authentic** than trying to make a tabletop radio “perform” every non-dialogue second.

#### E. Option 3 is tempting but architecturally dishonest
Option 3 is clever, but I would not choose it as the main fix.

Why:
- it leaves the routing lie in place: the system still pretends the radio is the performer
- it relies on prompt discipline to keep the radio in frame
- HuMo will still prioritize the **face**
- the radio becomes scenery, not the visual subject
- consistency risk is real

If you want the announcer with a radio in frame, you can still do that **inside Option 1** for the announcer portrait prompt. But don’t use it as the core architectural fix.

#### F. Option 2 is overkill for weak payoff
LTX-2.3 or similar for radio clips is not crazy, but for your constraints it’s the wrong first move:
- more code
- more orchestration
- more wall-clock
- more failure surface
- no true lip sync anyway
- likely “animated wallpaper” effect

If the fallback is just Ken Burns, then you don’t need LTX at all.

---

# Recommended decision

## Commit to:
### **Option 1, with this exact policy**
- **Dialogue characters:** HuMo from cast portraits
- **ANNOUNCER:** also a cast portrait, HuMo like any other speaking role
- **music_inter / music_intro / music_outro / sfx:** no HuMo; use deterministic static-video treatment
- **radio still:** bookend/interstitial motif only, generated once and used in post

That is the cleanest correction.

---

## 3. Top three failure modes for Option 1, and how to test them

## Failure mode 1: Speaker-role routing still leaks non-face images into HuMo
This is the most important one.

### What can go wrong
Some path still resolves:
- `ANNOUNCER` → radio still
- `music_*` → radio still → HuMo
- `sfx` → radio still → HuMo

If even one of those survives, you’ll keep getting prompt-driven strangers.

### What to verify
Add logging at the exact point where HuMo clip jobs are assembled:
- line id
- speaker role
- resolved visual source type
- resolved image path
- whether the source is a cast portrait vs radio still vs fallback frame

You want a run-level invariant like:

- **HuMo jobs only ever receive cast/host portraits**
- **No HuMo job ever receives the radio still**

### Concrete tests
1. **Episode with ANNOUNCER + dialogue + music_inter + sfx**
   - inspect render ledger / logs
   - assert no non-dialogue segment is scheduled for HuMo unless explicitly intended

2. **Regression test on l001 and l021**
   - both ANNOUNCER lines should resolve to the same announcer portrait family
   - no generic blonde drift

3. **Hard assertion**
   - if `speaker_role in {music_*, sfx}` and target renderer == HuMo, fail fast

That last one is worth doing.

---

## Failure mode 2: Video timing changes accidentally perturb audio or mux determinism
Given C7, this is the danger zone.

### What can go wrong
In `_render_master_mix_per_clip_mux_mode` or equivalent post path:
- clip durations get rounded differently
- ffmpeg concat introduces a resample
- audio gets re-encoded
- stream mapping changes
- timestamps drift
- final audio bytes differ from v1.5 baseline

### What to verify
For every gate:
- hash the final audio elementary stream or extracted WAV
- compare byte-for-byte to baseline
- fail immediately on mismatch

### Concrete tests
1. **Golden episode test**
   - render same episode before/after fix
   - extract final audio
   - compare SHA-256 and byte length

2. **Mux-only test**
   - take an already-approved audio master
   - run only the new video assembly path
   - verify extracted audio is byte-identical

3. **Duration consistency test**
   - sum segment durations from ledger
   - compare against final container-reported video duration and audio duration
   - define acceptable tolerance for container metadata, but not for audio bytes

If you don’t already have this, add a **post-mux audio hash check** as a gate.

---

## Failure mode 3: Non-dialogue visuals feel dead or visually awkward
This is the main aesthetic risk of Option 1.

### What can go wrong
If `music_*` and `sfx` are handled poorly, the episode may feel like:
- frozen slideshow
- jarring cuts
- random radio insertions
- visual dead air

### What to verify
You need a small human-review rubric, because this is not purely mechanical.

### Concrete tests
Render a representative episode with:
- announcer intro
- dialogue exchange
- music sting
- sfx-heavy beat
- announcer outro

Then review for:
1. **Visual continuity**
   - do transitions feel intentional?
2. **Role clarity**
   - does the viewer understand who is “on screen” during narration vs interlude?
3. **No uncanny motion**
   - no held HuMo clip with subtle breathing/mouth twitch during non-speech unless that’s intended

I would specifically test:
- `music_inter` between two dialogue clips
- `sfx` between two lines from different speakers
- long ANNOUNCER monologue with no other cast

---

## 4. Best visual treatment for `music_inter` and `sfx`

My recommendation:

## Default policy
### **(c) radio still cutaway** for `music_*`
### **(b) static cast still or previous shot hold** for short `sfx`, depending on context

More specifically:

---

## `music_inter`, `music_intro`, `music_outro`
Use **radio/studio cutaway**, not HuMo.

Why:
- music is naturally compatible with a non-speaking visual
- the radio set as a motif works well here
- a slow pan / subtle zoom is enough
- zero VRAM if done in ffmpeg
- deterministic
- no uncanny “silent talking head”

### Best implementation
Use:
- one deterministic FLUX-generated radio/studio still per episode, or
- one approved static asset if you want maximum consistency

Then make:
- 24 fps CFR
- exact duration to match audio segment
- subtle Ken Burns or very slow push-in
- deterministic ffmpeg filtergraph

This is the cleanest use of the radio visual identity.

---

## `sfx`
I would split by duration and context.

### For very short SFX beats
Use **hold previous frame** or **freeze the current character still**.

Why:
- if a door slam or phone ring lasts 0.5–1.5 s, cutting away to the radio is too editorially loud
- holding the current shot is more natural

### For longer SFX-only spans
Use a **contextual static cutaway**, not necessarily the radio every time.

Examples:
- thunder / exterior ambience → environment still
- newsroom teletype → studio/desk still
- suspense sting → radio/studio still is fine

If you don’t already have environment stills in a deterministic path, then yes, use the radio/studio still as the universal fallback.

### What I would avoid
#### (a) extend previous character’s clip with held mouth-closed animation
I would avoid this as the default.

Why:
- HuMo “idle” often still contains micro-motion
- if the character is visibly alive while no one is speaking, it can look like they are about to talk
- if you freeze the last frame, that’s okay; if you extend generated motion, less okay

So:
- **freeze frame** = acceptable
- **continue HuMo idle motion** = usually not ideal

---

## Practical rule set I’d implement

### `music_*`
- always radio/studio cutaway
- static or slow pan

### `sfx`
- if duration < ~1.5 s: freeze previous shot
- if duration >= ~1.5 s: radio/studio cutaway or contextual still

That gives you a simple deterministic policy.

---

## 5. Is there a fourth option you’re missing?

Yes, but it’s really a refinement of Option 1, not a whole new architecture.

## Option 4 — “Static-video first” architecture for all non-speech segments
Instead of thinking in terms of “every audio second must be a HuMo clip,” define:

- **speech segments** → animated (HuMo)
- **non-speech segments** → deterministic static-video segments assembled in post

That means:
- ANNOUNCER speech: HuMo, because it is speech
- dialogue speech: HuMo
- music/sfx/nonverbal spans: static-video compositor path

This is basically Option 1, but stated as a **clean architectural rule** rather than a workaround.

I think this is the right long-term model.

### Why this matters
Your current architecture is organized around **renderer coverage** (“wall-to-wall HuMo”), but the correct organizing principle is **content type**:

- **spoken human performance** → talking-head model
- **everything else** → editorial/static motion design

That is the real fix.

---

## Can the radio actually animate to audio without LTX/cloud/VRAM dragons?

Honestly: **probably not in a way I would recommend betting the architecture on**.

Within your constraints, I do **not** see a strong local/offline path that gives you:

- radio as primary subject
- meaningful audio-reactive motion
- deterministic output
- under 14.5 GB VRAM
- without adding another substantial model stack

### Possible hacks, but I would not recommend them as the main plan
1. **2D procedural animation in post**
   - animate dial glow, speaker grille pulse, slight camera shake, scanlines
   - driven by audio amplitude envelope
   - deterministic
   - zero VRAM
   - but this is motion graphics, not true image animation

2. **Mouth-region style compositing**
   - fake speaker cloth pulsing or tube glow synced to amplitude
   - again deterministic and cheap
   - but stylized, not realistic

3. **Optical-flow / warp-based image animation**
   - local CPU/GPU post effects
   - but likely ugly and not worth the complexity

If you mean “actually animate the radio” in a tasteful way, the only thing I’d seriously consider is:

### A procedural ffmpeg/post effect:
- subtle zoom/pan
- tiny brightness modulation from audio RMS
- optional warm tube-light flicker
- maybe a slight speaker-cloth pulse mask

That would be:
- local
- deterministic
- zero extra VRAM
- safe under C7 if audio stream is untouched

It’s not generative animation, but it may be enough to make the radio feel alive.

If you want, this could become a **small enhancement after Option 1**, not before.

---

# My concrete recommendation

## Commit this architecture:

### 1. Stop treating the radio still as a HuMo input
That premise is broken.

### 2. Add ANNOUNCER as a real cast member
Use the existing portrait pipeline.

### 3. Change speaker-role resolution
In `nodes/_otr_speaker_role.py`:
- `ANNOUNCER` → portrait chain
- `music_*` → non-HuMo static-video path
- `sfx` → non-HuMo static-video path
- only speaking human roles go to HuMo

### 4. Keep the radio as a motif
Use it for:
- open
- close
- music interludes
- longer non-dialogue cutaways

### 5. Preserve audio path exactly
No audio pipeline changes. Add hash verification after mux.

---

# If you want the most opinionated answer possible

## I would **not** ship Option 2 now.
Too much machinery for too little gain.

## I would **not** rely on Option 3 as the fix.
It’s a prompt trick, not an architectural correction.

## I would ship **Option 1 / Option 4**:
**HuMo only for speaking faces. Static-video editorial treatment for non-speech.**

That is the honest fit to the tools you have.

---

# Suggested acceptance criteria

I’d call the bug fixed only if all of these are true:

1. **No HuMo render job ever receives the radio still**
2. **ANNOUNCER clips resolve to a stable announcer portrait**
3. **music/sfx segments render through a deterministic non-HuMo path**
4. **Final audio extracted from mux is byte-identical to baseline**
5. **Peak VRAM remains below 14.5 GB**
6. **A representative episode no longer shows random generic faces for ANNOUNCER**

---

If you want, I can next turn this into a **surgical implementation plan** with:
- exact routing changes
- suggested guard assertions
- ffmpeg static-video strategy
- and a minimal regression checklist tied to your file layout.
