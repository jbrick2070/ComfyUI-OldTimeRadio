# Round A -- ChatGPT (gpt-5.5) elapsed=152.1s

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
