# Synthesis -- 2026-05-06

**Question:** # Round-robin consult — LTX clips rendering mostly-static (BUG-LOCAL-112)

## Context

OTR (`v2.0-alpha`) is generating procedural sci-fi-radio-drama videos. The visual stack uses LTX 2B v0.9 for the announcer/music/sfx beats (i2v anchored to a FLUX-rendered radio_bookend still). HuMo handles character lipsync separately. Today's runs revealed the LTX clips are mostly-static — the bug Jeffrey reported as "LTX is not showing any radio movement or action."

This is on the heels of BUG-LOCAL-095 (FIXED 2026-05-04, swapped `LTXVAddGuide` → `LTXVImgToVideoConditionOnly` to escape keyframe pinning) and BUG-LOCAL-032 (FIXED 2026-05-03, dropped end-frame anchor that was clamping into ping-pong). Both fixes shipped, but the symptom is still here.

## Quantitative motion analysis (just measured)

Sampled 5 evenly-spaced frames within each LTX announcer clip (l001.mp4, the opening beat) and computed mean absolute pixel difference (MAD, RGB 0-255) between consecutive frames. Real motion (a dolly forward, a dial sweep) should produce sustained 15-30 MAD. Static frames produce <5 MAD.

| Episode | Clip dur | Inter-frame MAD samples |
|---|---|---|
| stellar_echoes | 5.16s | 35.95 → 3.32 → 3.21 → 3.38 (one scene-cut spike, then static) |
| stellar_divide | 6.76s | 2.36 → 15.53 → 9.80 → 32.78 (only "good" one) |
| deserted_space_habitat_spinning | 2.28s | 2.04 → 4.31 → 5.86 → 5.92 (subtle/static) |
| cramped_spaceship_cockpit_humming | 5.16s | 1.86 → 2.05 → 7.01 → 32.29 (STATIC first half, then a scene-cut spike) |

3 of 4 clips are essentially still images with occasional scene-cut artifacts.

## OTR LTX params (current)

```python
LTX_FPS = 25
LTX_WIDTH = 832     # 0.4 MP
LTX_HEIGHT = 480
LTX_CFG = 1.0
LTX_I2V_STRENGTH = 0.75

# 8-step distilled sigma schedule from Goofer/DMM
LTX_DISTILLED_SIGMAS = [
    1.0, 0.99375, 0.9875, 0.98125, 0.975,
    0.909375, 0.725, 0.421875, 0.0,
]

sampler = euler
ltx_length_for_dur(dur_s) -> 25*dur_s frames (capped 257; typical 125-175 for 5-7s announcer)
```

The render path:
```python
positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt_text))
cond_pos, cond_neg = LTXVConditioning(positive, base_negative, frame_rate=25)
empty_latent = EmptyLTXVLatentVideo(width=832, height=480, length=N, batch_size=1)
latent_chunk = LTXVImgToVideoConditionOnly(
    vae=vae, image=ref_image, latent=empty_latent, strength=0.75
)
noise = RandomNoise(noise_seed=shot_seed)
guider = CFGGuider(model=model, positive=cond_pos, negative=cond_neg, cfg=1.0)
samples = SamplerCustomAdvanced(noise, guider, sampler=KSamplerSelect("euler"), sigmas=LTX_DISTILLED_SIGMAS, latent_image=latent_chunk)
```

## DMM reference (`comfyui-data-media-machine/nodes/dmm_batch_video.py`)

DMM uses essentially the SAME knobs and produces visibly dynamic output:

- `fps=35` default (vs OTR 25)
- `width=768, height=512` (~0.39 MP, similar to OTR)
- `cfg=1.0` (same)
- `cond_strength=0.75` (same default i2v strength)
- Sampler choices: `euler / euler_ancestral / dpmpp_2m / dpmpp_sde` (default euler, same as OTR)
- Sigmas: literally the same string `"1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"`
- `length=61` default (1.74s at 35fps; 2.44s at 25fps)
- i2v conditioning is OPTIONAL — defaults to pure t2v unless `image_N` socket is wired

## OTR prompt structure (post BUG-LOCAL-110 + BUG-LOCAL-008)

OTR builds each LTX prompt as:
```
<ltx_style_brief>, <role_template>, scene context: <env>, <style> broadcast tone
```

Concrete example for `cramped_spaceship_cockpit_humming` announcer (~700 chars):
```
Sci-fi radio booth, holographic vacuum tubes pulsating in an obsidian console,
ethereal purple lighting casting long shadows across sleek metallic surfaces,
distant cosmic murmurs resonating from a large, retro-styled brass speaker grille,
Vintage 1940s radio broadcast set, tuning dial needle sweeping rhythmically
across the frequency band, copper vacuum tubes pulsing with warm amber filament
light, brass speaker grille vibrating visibly with the broadcast, dust motes
drifting through volumetric studio beams, breathing handheld dolly forward with
subtle camera shake, dim studio lighting with warm glow halos, unattended
equipment, empty studio, no people in frame, cinematic 35mm film grain,
scene context: <60 chars>, space opera epic broadcast tone
```

DMM uses ONE prompt per clip directly from the user widget; no concatenation, no boilerplate. Typically 50-150 chars of motion-rich text.

## Three hypotheses ranked

**H1: Prompt structure dilution.** Motion verbs ("needle sweeping rhythmically", "tubes pulsating", "speaker grille vibrating", "breathing handheld dolly forward with subtle camera shake") are buried in the middle of a 700-char prompt. Static-noun ballast at the end ("unattended equipment, empty studio, no people in frame, cinematic 35mm film grain") may be drowning the motion. T5 encoder attends most to the front. **Fix:** front-load motion verbs, drop the "no people / unattended" filler since CFG=1.0 means negatives are moot anyway.

**H2: Long-clip degradation.** OTR runs 125-175 frame clips (5-7s @ 25fps) for announcer beats; DMM defaults to 61 frames. LTX 2B v0.9 may degrade past ~80 frames into mostly-static output. **Fix:** cap LTX clip length at ~80 frames (3.2s @ 25fps); split longer announcer beats into chunks (each with its own seed and motion language so the chunks don't look identical at the join).

**H3: i2v strength too high for motion.** `LTX_I2V_STRENGTH=0.75` means 75% of the latent is anchored to the static FLUX still; only 25% is free for motion synthesis. DMM ships 0.75 as default but exposes it as a per-clip widget. **Fix:** drop to 0.55-0.60 to give the sampler more freedom from the locked init image. Trade-off: visual identity may drift further from the FLUX bookend.

## Questions

1. Which hypothesis is most likely the dominant cause? Could be a combination — rank them.

2. **H1 specifics:** is front-loading motion verbs in the prompt actually known to help T5-conditioned diffusion video? Is the static-noun-in-the-back theory real or am I projecting human-attention bias onto a transformer?

3. **H2 specifics:** is there documented degradation curve on LTX 2B v0.9 past 80-100 frames? Lightricks's HF page says 257 frame max; what's the empirical sweet spot for motion quality?

4. **H3 specifics:** at i2v_strength=0.55-0.60, does the visual identity loss outweigh the motion gain? Is there a known good range for "i2v that still moves"? DMM defaults to 0.75 but their reference clips DO move with the same value, suggesting it's not the strength alone.

5. **What we missed:** is there a fourth knob we should be looking at? Sampler choice (euler_ancestral adds noise per step which can add motion)? Negative prompt content even though CFG=1.0 makes it mathematically inert (does the act of computing it influence anything)? Sigma schedule's specific values (the first 5 sigmas are crammed near 1.0 — does this throttle motion sampling)?

6. **Smallest fix to ship FIRST:** if we can only change one knob tonight to test, which one gives the highest probability of unlocking motion?

Goal: get LTX to produce visibly dynamic clips of the radio set animating (dial sweeping, tubes pulsing, dolly forward, dust drifting) instead of mostly-still frames. Be code-review-style, name specific knobs, name specific values. Disagreements between you fine; we'll synthesize.

Branch: `v2.0-alpha`. Standing rule: only Jeffrey merges to main. Don't propose architectural changes (sigma schedule rebuild, model swap, sampler family change) without justification — the smallest possible change that unlocks motion is the goal tonight.

---

## ChatGPT (gpt-5.5)

## Short answer

I would **not** chase sampler/sigma/VRAM-level changes tonight.

My ranking:

1. **H1 prompt structure / prompt over-specification is most likely dominant.**  
   Especially because you have a **57-frame / 2.28s clip that is still mostly static**, while one **169-frame / 6.76s clip moves**. That weakens “long clip length” as the primary explanation.

2. **H3 i2v strength is likely a real contributor, but probably not the only cause.**  
   `0.75` can preserve the FLUX still too strongly, especially when the prompt itself reads like a still-image art direction sheet. I would try `0.60` after or alongside the prompt fix.

3. **H2 long-clip degradation is plausible but not the leading cause from your data.**  
   LTX may well get worse past ~80–120 frames, but your own measurements show short static and long dynamic examples. Length is probably an amplifier, not the root.

If you can change only one thing tonight: **change the LTX prompt template first.**  
If “one thing” must mean a numeric constant: change `LTX_I2V_STRENGTH` from `0.75` to **`0.60`**.

---

## 1. Ranked hypotheses

### Rank 1 — H1: prompt structure dilution / still-image bias

Most likely.

The current prompt starts like this:

```text
Sci-fi radio booth, holographic vacuum tubes pulsating in an obsidian console,
ethereal purple lighting casting long shadows across sleek metallic surfaces,
distant cosmic murmurs resonating from a large, retro-styled brass speaker grille,
Vintage 1940s radio broadcast set, tuning dial needle sweeping rhythmically...
```

This is visually rich, but the first impression is still mostly **composition / set dressing**. The motion language is present, but it is buried among many static image-caption phrases:

```text
obsidian console
ethereal purple lighting
long shadows
sleek metallic surfaces
large retro-styled brass speaker grille
dim studio lighting
warm glow halos
unattended equipment
empty studio
no people in frame
cinematic 35mm film grain
```

That style is very FLUX-friendly, but LTX video often benefits from a caption that reads like:

> “A continuous video shot where X moves, Y moves, Z moves.”

Not:

> “A beautiful radio set with many adjectives, plus some motion somewhere in the middle.”

Also, the prompt appears to contain multiple competing visual concepts: sci-fi booth, holographic tubes, retro brass grille, cosmic murmurs, 1940s broadcast set, purple light, amber tubes, dust motes, dolly forward, empty studio, no people, film grain, scene context, broadcast tone. The model may satisfy this as a mostly static tableau with occasional discontinuous scene changes.

The scene-cut MAD spikes support this. The model may be interpreting the prompt as a set of visual beats rather than one continuous take.

### Rank 2 — H3: `LTX_I2V_STRENGTH=0.75` anchoring to the still

Likely contributor.

I would not phrase it as “75% of the latent is locked and only 25% is free,” because that may not be the exact semantics of `LTXVImgToVideoConditionOnly`. But practically, high i2v conditioning can make the model conservative, especially when the prompt does not strongly demand continuous local motion.

DMM’s default being `0.75` does not fully exonerate this knob because:

- DMM i2v is optional and may often be running pure t2v.
- DMM prompts are usually shorter and more direct.
- DMM default length is shorter.
- DMM user prompts may not contain still-image boilerplate and “no people” clauses.

So `0.75` can work, but in OTR’s exact setup it may be part of the lock.

I would test:

```python
LTX_I2V_STRENGTH = 0.60
```

Then, if identity still holds and motion is still weak:

```python
LTX_I2V_STRENGTH = 0.55
```

I would not go below `0.50` tonight unless you are willing to accept more identity drift from the FLUX bookend.

### Rank 3 — H2: long-clip degradation

Plausible, but your data does not make it the prime suspect.

Evidence against it being dominant:

- `deserted_space_habitat_spinning` is only **2.28s**, likely around **57 frames**, and it is still mostly static:
  ```text
  2.04 → 4.31 → 5.86 → 5.92
  ```
- `stellar_divide` is **6.76s**, likely around **169 frames**, and it is the most dynamic:
  ```text
  2.36 → 15.53 → 9.80 → 32.78
  ```

So length alone cannot explain the symptom.

That said, I do believe LTX 2B can get weaker over longer spans, especially with distilled 8-step sampling. I would still consider a later cap around:

```python
LTX_MAX_FRAMES = 81  # 3.24s at 25 fps, and 4n+1-compatible
```

or:

```python
LTX_MAX_FRAMES = 97  # 3.88s at 25 fps
```

But that implies chunking or duration handling, so it is not the smallest safe fix tonight.

---

## 2. H1 specifics: does front-loading motion help T5-conditioned video?

Yes, but with nuance.

It is not quite “T5 pays attention only to the front.” Transformers are not human readers, and attention is not a simple left-to-right priority system. But front-loading still helps for several practical reasons:

### Why it helps

1. **Token truncation / token budget risk**  
   If the prompt is long enough to approach the encoder’s effective limit, late material can be truncated or degraded. I do not know your exact LTX Comfy node token limit from the provided snippets, so I will not bluff a number. But 700 characters is absolutely long enough to make me care.

2. **Training-caption prior**  
   Video captions often state the main action early:
   ```text
   camera slowly pushes in on a radio console as the dial needle sweeps...
   ```
   The model may respond better when the caption resembles motion-caption data rather than still-image art direction.

3. **Competing concepts**  
   The more nouns and style clauses you add, the more the model can satisfy the prompt with appearance rather than motion.

4. **Negated concepts in the positive prompt are risky**  
   This part is especially important:
   ```text
   unattended equipment, empty studio, no people in frame
   ```
   “No people in frame” is in the positive prompt. Text encoders do not reliably treat negation the way humans do. Sometimes the token “people” still activates people-related concepts. Better phrasing:
   ```text
   empty equipment-only radio booth
   ```
   or:
   ```text
   empty radio console, equipment-only shot
   ```

### Is “static nouns in the back drown the motion” real?

Partly, but I would rephrase it.

The back of the prompt does not magically dominate. The issue is that the whole prompt reads like a still-image prompt with some motion adjectives inserted. The model is probably being asked to render a beautiful static set more strongly than it is being asked to render continuous physical motion.

---

## 3. H2 specifics: documented degradation past 80–100 frames?

I do not know of a clean, official degradation curve for LTX 2B v0.9 that says “motion quality falls off after frame X.” The HF max-frame claim means “supported,” not “equally good at all lengths.”

Empirically, for small local test workflows, I would treat these as sane bands:

```text
49–65 frames:  very safe for motion tests
81 frames:     good practical target
97–121 frames: acceptable if prompt is simple and motion is not ambitious
125–175 frames: may work, but more likely to become static, cutty, or inconsistent
257 frames:    max capability, not my first choice for quality
```

At 25 fps:

```text
61 frames  ≈ 2.44s
81 frames  ≈ 3.24s
97 frames  ≈ 3.88s
121 frames ≈ 4.84s
169 frames ≈ 6.76s
```

If OTR later adds chunking, I would aim for **81-frame chunks** first. But I would not make that tonight’s first fix because it changes duration handling and join behavior.

---

## 4. H3 specifics: good i2v strength range

For “i2v that still moves,” I would test this range:

```python
0.55 <= LTX_I2V_STRENGTH <= 0.65
```

My first value:

```python
LTX_I2V_STRENGTH = 0.60
```

Expected tradeoff:

- `0.75`: stronger identity, higher risk of static output.
- `0.65`: still fairly anchored, maybe enough.
- `0.60`: good first unlock test.
- `0.55`: more motion, moderate drift risk.
- `<0.50`: likely starts behaving more like t2v with a loose visual hint.

For OTR specifically, I think the identity-loss risk is acceptable because the target is not a human face or a specific prop layout. It is a radio set mood: tubes, dial, grille, dust, dolly. The prompt can re-anchor that.

But because DMM can move at `0.75`, I would not assume strength alone is the bug.

---

## 5. Fourth knob / missed issue

### A. Positive prompt contains negative instructions

This is the one I would fix immediately.

Current:

```text
unattended equipment, empty studio, no people in frame
```

Better:

```text
empty equipment-only radio booth
```

Avoid:

```text
no people
no humans
no characters
```

unless you are putting those in a real negative prompt with CFG greater than 1. But with `cfg=1.0`, the negative path is mathematically inert for standard CFG.

### B. Add “single continuous shot” language

The scene-cut spikes suggest LTX is sometimes cutting rather than animating.

Add this near the front:

```text
single continuous shot, same radio console for the entire clip
```

or:

```text
unbroken continuous take
```

I would avoid relying only on “no cuts” because negation can be weak. Prefer positive continuity language:

```text
unbroken continuous take
same radio console throughout
```

### C. Use local object motion before camera motion

A static i2v anchor can resist camera motion like:

```text
breathing handheld dolly forward
```

But it may still allow local motion:

```text
needle sweeps
vacuum tubes pulse
speaker grille trembles
dust motes drift
light flickers
```

So put local motion first, then dolly.

### D. CFG / negative prompt

At:

```python
cfg=1.0
```

standard CFG computes approximately:

```text
negative + cfg * (positive - negative)
```

At `cfg=1.0`, that becomes:

```text
positive
```

So the negative conditioning should not meaningfully affect the output. There can be implementation details, but with `CFGGuider` as named in your path, I would treat the negative prompt as functionally irrelevant at `1.0`.

Do not spend tonight tuning the negative prompt unless you also intend to raise CFG, which I do not recommend as the first move.

### E. Sampler choice

`euler_ancestral` might add more stochastic texture or flicker, but I would not use it as the first fix. It can create motion-like noise rather than meaningful radio-set animation. It may also worsen shot stability.

Keep:

```python
sampler = euler
```

for tonight.

### F. Sigma schedule

Do not change the sigma schedule tonight.

The schedule looks like the known distilled LTX/DMM schedule. The first five values being clustered near 1.0 is not something I would reinterpret casually. Since DMM uses the same schedule, it is unlikely to be the dominant local bug.

### G. FPS

DMM defaulting to 35 fps is interesting, but I would not touch it first. Changing fps changes duration/frame-count math and may create more integration fallout than prompt/strength changes.

---

## 6. Smallest fix to ship first

### My recommended first patch: replace the LTX prompt template

Keep all model/sampler/sigma/VRAM behavior unchanged.

Target prompt length: **~180–300 chars before scene context**, not 700.

Use a motion-first template like:

```text
Unbroken continuous video shot with visible motion throughout: an empty equipment-only 1940s sci-fi radio console, tuning dial needle sweeping, amber vacuum tubes pulsing, brass speaker grille trembling, dust motes drifting, slow handheld dolly push-in. Purple and amber volumetric studio light, 35mm film grain, same radio console throughout.
```

If you need to include episode context, append it late and keep it short:

```text
Scene context: cramped spaceship cockpit ambience. Space opera broadcast tone.
```

Full example:

```text
Unbroken continuous video shot with visible motion throughout: an empty equipment-only 1940s sci-fi radio console, tuning dial needle sweeping, amber vacuum tubes pulsing, brass speaker grille trembling, dust motes drifting, slow handheld dolly push-in. Purple and amber volumetric studio light, 35mm film grain, same radio console throughout. Scene context: cramped spaceship cockpit ambience.
```

Important changes versus current:

- Motion is first.
- “Unbroken continuous video shot” is first.
- “same radio console throughout” discourages cuts.
- “no people” is removed.
- “unattended equipment” is replaced by “empty equipment-only.”
- Static art-direction ballast is reduced.
- Local object motion precedes camera motion.

### If you want one numeric change instead

Change:

```python
LTX_I2V_STRENGTH = 0.75
```

to:

```python
LTX_I2V_STRENGTH = 0.60
```

I would test `0.60` before `0.55`.

---

## Minimal test matrix

To avoid guessing, run the same seed and same FLUX bookend still for one known-bad clip:

### Baseline

```python
prompt = current_700_char_prompt
strength = 0.75
length = current
sampler = euler
sigmas = current
```

### Test A — prompt only

```python
prompt = motion_first_short_prompt
strength = 0.75
length = current
sampler = euler
sigmas = current
```

### Test B — strength only

```python
prompt = current_700_char_prompt
strength = 0.60
length = current
sampler = euler
sigmas = current
```

### Test C — prompt + strength

```python
prompt = motion_first_short_prompt
strength = 0.60
length = current
sampler = euler
sigmas = current
```

Expected result:

- If A improves strongly: H1 confirmed.
- If B improves strongly: H3 confirmed.
- If only C improves: H1 and H3 interact.
- If none improve: then investigate H2/chunk length or workflow/node behavior next.

Use your MAD metric, but also inspect visually. For this kind of radio-console motion, I would want sustained samples roughly:

```text
8–15 MAD: acceptable subtle motion
15–30 MAD: strong visible motion
>30 MAD: may be scene cut or large camera change
```

Scene-cut spikes alone should not count as success.

---

## Code-review-style patch recommendation

I cannot cite exact OTR line numbers because I do not have the branch checkout. Based on the symbols you provided, the relevant edits are wherever these are defined/assembled:

```python
LTX_I2V_STRENGTH = 0.75
```

and wherever OTR builds:

```text
<ltx_style_brief>, <role_template>, scene context: <env>, <style> broadcast tone
```

Do **not** change this tonight:

```python
LTX_DISTILLED_SIGMAS = [
    1.0, 0.99375, 0.9875, 0.98125, 0.975,
    0.909375, 0.725, 0.421875, 0.0,
]
```

Do **not** change this tonight:

```python
sampler = euler
```

Do **not** spend time on negative prompt content while:

```python
LTX_CFG = 1.0
```

First patch should be conceptually:

```python
def build_ltx_radio_prompt(env: str, tone: str) -> str:
    return (
        "Unbroken continuous video shot with visible motion throughout: "
        "an empty equipment-only 1940s sci-fi radio console, "
        "tuning dial needle sweeping, amber vacuum tubes pulsing, "
        "brass speaker grille trembling, dust motes drifting, "
        "slow handheld dolly push-in. "
        "Purple and amber volumetric studio light, 35mm film grain, "
        "same radio console throughout. "
        f"Scene context: {env}. {tone} broadcast tone."
    )
```

If you want the highest-probability combined small fix, I would ship:

```python
LTX_I2V_STRENGTH = 0.60
```

plus the shorter motion-first prompt.

But if Jeffrey wants exactly one change tonight: **prompt template first.**

---

## Gemini (gemini-3.1-pro-preview-customtools)

**1. Core Agreement**
I PARTIALLY AGREE that H1 (Prompt Dilution) is the dominant cause, but I STRONGLY DISAGREE with ChatGPT's recommendation to lower `LTX_I2V_STRENGTH` to 0.60, as your own DMM reference data proves 0.75 is fully capable of motion.

**2. Factual Errors in ChatGPT's Answer**
*   **Token Limit Bluffing:** ChatGPT claims 700 characters "approaches the encoder's effective limit." False. 700 characters is roughly 150-180 tokens. LTX-Video uses T5-XXL with a 512-token context window. Truncation is not happening; *attention dilution* is.
*   **Frame Math:** ChatGPT suggests 81 frames because it is "4n+1 compatible." False. LTX-Video's 3D VAE uses 8x temporal compression, meaning frame counts must be `8n + 1`. *(Note: Looking at your data, 5.16s = 129f, 6.76s = 169f, 2.28s = 57f. These are **already** perfect `8n + 1` integers. Your `ltx_length_for_dur` math is spot-on, which ChatGPT completely missed).*

**3. Important Omissions**
*   **Rule C7 (Audio Sync):** ChatGPT casually suggests capping the clip length at 81 frames (3.24s). If your audio beat is 5.16s, capping the video breaks Rule C7 unless you immediately build a chunking/looping architecture, which violates your "smallest fix tonight" constraint. 
*   **Native Training Distribution:** LTX-Video v0.9 was natively trained on 121 frames (4.84s @ 25fps). Pushing to 169 frames pushes the model out-of-distribution. When diffusion video models go OOD on length, their most common failure mode is to freeze into a static image to prevent temporal collapse.

**4. My Recommendation (Smallest Fix First)**
Ship **ONLY** the prompt structure fix tonight. Do not touch the numeric knobs.

*   **Strip the FLUX Boilerplate:** T5-XXL is highly literal. If 80% of your prompt describes static set dressing ("obsidian console", "purple lighting", "35mm film grain"), the model will optimize for a beautiful, static photograph. 
*   **Use a Motion-Exclusive LTX Prompt:** You already anchored the visual identity with the FLUX still via `LTXVImgToVideoConditionOnly`. You don't need to describe the room again. Change your LTX prompt generation to something brutally direct (< 150 chars):
    `"Continuous dolly shot moving forward. A retro tuning dial needle sweeps rhythmically across the frequency band. Vacuum tubes pulse with light."`
*   **Keep `LTX_I2V_STRENGTH = 0.75`:** DMM uses 0.75 and gets motion. If you lower this to 0.60, you will introduce visual drift from your FLUX bookend for no guaranteed motion gain.
*   **Do not cap length tonight:** Since `stellar_divide` successfully moved at 169 frames, the model *can* handle the duration if the prompt isn't drowning the motion prior. Keep generating the full duration to satisfy Rule C7.

**5. Uncertainties**
I am uncertain *why* `stellar_divide` (169 frames) was the only clip to successfully move. If it used the exact same 700-character boilerplate template as the others, it strongly suggests your current prompt is right on the mathematical edge of temporal failure, and `stellar_divide` just got lucky with the `shot_seed` noise initialization. Stripping the prompt ballast should make that luck permanent.

---

## NVIDIA (nvidia/llama-3.3-nemotron-super-49b-v1.5)

### **1. Agreements and Disagreements with ChatGPT & Gemini**

#### **Agreements:**
- **H1 (Prompt Dilution) is Critical**: Both agree the prompt is overly static and diluted. I concur. The current prompt prioritizes set description over motion, which likely confuses the model into producing static frames.  
- **Negative Prompt Handling**: Both note that `CFG=1.0` nullifies negative prompts, but the *positive* prompt contains negations like "no people," which are risky. Removing these is wise.  
- **Sigma Schedule & Sampler**: Both correctly advise against changing these tonight. The sigma schedule is standardized, and `euler` is stable.  

#### **Disagreements:**
- **H3 (i2v Strength)**:  
  - **ChatGPT**: Recommends lowering to `0.60` to reduce anchoring.  
  - **Gemini**: Argues to keep `0.75` since DMM works with it.  
  - **My Take**: Gemini is more convincing. DMM’s success at `0.75` suggests strength isn’t the root cause. The prompt’s static bias is likely overpowering the motion potential. Lowering strength risks visual drift without guaranteed motion gain.  

- **H2 (Long-Clip Degradation)**:  
  - **ChatGPT**: Downplays length, citing a short static clip (57f) and long dynamic clip (169f).  
  - **Gemini**: Notes LTX was trained on 121f (4.84s @25fps), so 169f is out-of-distribution (OOD), causing freezing.  
  - **My Take**: Gemini’s OOD point is valid. While `stellar_divide` (169f) moved, it may have been lucky. OOD models often freeze to avoid temporal collapse. However, fixing the prompt first is safer than capping length (which breaks Rule C7).  

---

### **2. Factual Errors in Previous Answers**
- **ChatGPT**:  
  - Claimed 700 characters "approach the encoder’s limit" — **False**. T5-XXL has a 512-token window; 700 chars ≈ 150–180 tokens.  
  - Suggested `81 frames` for "4n+1 compatibility" — **False**. LTX uses **8n+1** temporal compression (Gemini corrected this).  

- **Gemini**:  
  - Asserted `169f = 8n+1` (21*8 + 1 = 169) — **Correct**.  
  - Accurately noted LTX’s native training length (121f) — **Correct**.  

---

### **3. Important Omissions by Both**
1. **VAE Temporal Compression Impact**:  
   - LTX’s 8x temporal compression means the model sees fewer latent temporal steps. Long clips (e.g., 169f) may still struggle even if frame counts are 8n+1.  

2. **Prompt Variability Across Clips**:  
   - Why did `stellar_divide` (169f) move while others didn’t? Was its prompt different? If all clips used the same template, this suggests the model is on the edge of failure, and small prompt tweaks could tip it into motion.  

3. **Scene-Cut Spikes in MAD Data**:  
   - High MAD spikes (e.g., 32.78 in `stellar_divide`) suggest the model is cutting scenes rather than animating continuously. The prompt should emphasize "unbroken continuous shot" to mitigate this.  

---

### **4. Prioritized Recommendations**
**Goal**: Unlock motion with minimal changes, respecting Rule C7 (audio sync) and avoiding VRAM optimization.  

1. **Deploy a Minimal Motion-Focused Prompt (Highest Priority)**  
   - **Change**: Replace the 700-character prompt with a **<150-character** motion-centric version.  
   - **Example**:  
     ```text
     Continuous dolly forward. Tuning dial sweeps rhythmically. Amber vacuum tubes pulse. Dust motes drift. Unbroken shot, same console throughout.
     ```  
   - **Rationale**: The i2v anchor (`LTXVImgToVideoConditionOnly`) already provides the visual base. The prompt should focus solely on motion.  
   - **File**: Likely in `prompt_builder.py` or wherever `prompt_text` is constructed.  

2. **Remove Static Descriptions and Negative Language**  
   - **Change**: Strip all static set-dressing terms ("obsidian console," "purple lighting," "film grain") and negations ("no people").  
   - **Rationale**: These dilute motion focus. Use the prompt only for *action*, not appearance.  

3. **Keep `LTX_I2V_STRENGTH = 0.75`**  
   - **Rationale**: DMM works at this value. The issue is prompt structure, not strength. Lowering it risks drift without guaranteed motion.  
   - **File**: `config.py` (e.g., line 20).  

4. **Add "Unbroken Continuous Shot" Language**  
   - **Change**: Explicitly state continuity to reduce scene cuts.  
   - **Example**:  
     ```text
     ... Unbroken continuous take. Same radio console throughout. No cuts.
     ```  

5. **Test with Same Seed for Fair Comparison**  
   - **Action**: Re-render a problematic clip (e.g., `cramped_spaceship_cockpit_humming`) with the new prompt using the **same seed** as the original. Compare MAD scores.  

---

### **5. Uncertainties Requiring Verification**
1. **Why Did `stellar_divide` Work?**  
   - Was its prompt different? If all clips used the same template, this suggests the model is sensitive to small prompt variations.  

2. **Impact of Sigma Schedule on Motion**:  
   - The first five sigmas are clustered near 1.0. Does this throttle motion? Probably not, as DMM uses the same schedule.  

3. **Visual Drift at Lower `LTX_I2V_STRENGTH`**:  
   - If we later lower strength to 0.60, how much drift occurs? Need to test with the new prompt first.  

---

### **Final Verdict**
**Ship the prompt fix tonight.**  
- **Change the prompt to motion-centric, strip static boilerplate, and emphasize continuity.**  
- **Do not alter `LTX_I2V_STRENGTH` or clip length.**  
- **Test with same seed and compare MAD.**  

If motion remains static, revisit `LTX_I2V_STRENGTH` (try 0.65) or consider chunking (but this breaks Rule C7). For now, the prompt is the lowest-risk, highest-reward lever.

---

## To decide (Claude / human)

- [ ] All three agree:
- [ ] Two-vs-one splits:
- [ ] Facts to verify:
- [ ] Final grounded recommendation:
