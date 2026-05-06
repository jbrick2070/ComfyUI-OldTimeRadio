# Question -- 2026-05-06

# Round-robin consult — LTX clips rendering mostly-static (BUG-LOCAL-112)

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
