# Round A -- ChatGPT (gpt-5.4) elapsed=83.0s

Here’s the blunt version:

## Recommendation

**Pick: SDXL 1.0 + a tightly curated VHS/CRT LoRA stack** for Phase B anchor frames.

It is the best fit for your actual constraints, not the most glamorous model on paper.

Why:

- **Fits your VRAM ceiling** without heroics.
- **ComfyUI-native and boring in a good way** on Windows.
- **Best style-control ecosystem** for 1980s broadcast grime, CRT bloom, UHF noise, analog darkness.
- **Deterministic-ish and seedable enough** for anchor-frame use.
- **Fast enough** to keep a 6–15 shot episode under your wall-clock target.
- **Lowest integration risk** with your subprocess rule set.

If you force me to name a runner-up: **FLUX.1-schnell** for selected clean environment shots, but I would **not** make it the Phase B default.

---

# Candidate-by-candidate assessment

I’m going to be candid where numbers are uncertain. Exact VRAM depends on:
- ComfyUI build,
- node implementation,
- whether you keep VAE resident,
- image size,
- batch size,
- fp16 vs fp8 path,
- and whether LoRAs are merged or stacked dynamically.

So treat these as **planning estimates**, not guarantees.

---

## 1) SDXL 1.0 + CRT/VHS LoRA stack

## Verdict
**Best overall choice.**

## 1. VRAM footprint
At your likely anchor resolutions:

- **1024×576 / 1024×640 / 1024×1024**
- **fp16**: typically about **8.5–11.5 GB peak**
- with **1–3 LoRAs** loaded: often still within **~9–12 GB**
- with IP-Adapter env reference and standard VAE handling: can rise toward **~12–13.5 GB**
- still generally under your **14.5 GB ceiling**

If you use any **fp8 model variant/path**, you may claw back another ~1–2 GB, but I would not make fp8 a requirement unless you’ve already validated it on your exact stack.

**Bottom line:** SDXL is the safest “it fits” answer.

## 2. Aesthetic match
Very good for your use case.

Strengths:
- Strong at **moody environments**, architecture, interiors, streets, signage, weather, practical lighting.
- Excellent for “**still from a shot**” if prompted correctly.
- Huge style flexibility for:
  - VHS softness
  - CRT phosphor glow
  - analog bloom
  - sodium-vapor streets
  - late-night LA haze
  - UHF-era broadcast grime
  - underexposed interiors
- Better than SD1.5 at preserving **composition + atmosphere** without collapsing into mush.

Weaknesses:
- Native SDXL tends to drift toward **polished concept art / posterization** unless you constrain it.
- If you over-stack LoRAs, you’ll get “retro aesthetic wallpaper” instead of a believable frame.
- Faces/people remain a liability, but you already know not to lean on that.

For **Miracle Mile / late-night radio / apocalypse-but-cozy LA**, SDXL is the strongest style-control platform among your listed options.

## 3. LoRA / IP-Adapter compatibility in 2026
This is where SDXL wins hard.

- **LoRA ecosystem:** mature, broad, easy to source locally.
- **Period-style LoRAs:** yes, this is the one place where the ecosystem actually exists in useful volume.
- **IP-Adapter:** supported and well understood in ComfyUI for environment references.
- **Control tooling:** strongest practical support.

For your “environment only, never characters” rule, SDXL + env-only IP-Adapter is exactly the sane path.

## 4. Per-frame wall-clock
On a 5080 Laptop-class GPU, assuming sane settings:

- **1024-wide env anchor**
- **20–30 steps**
- **single image**
- likely around **8–20 seconds/frame**
- maybe **20–35 sec** if you pile on adapters and use heavier samplers

A 6-shot episode:
- roughly **1–3 minutes** of pure generation time
- maybe **3–6 minutes total** including worker startup, model load, VAE decode, image save

A 15-shot episode:
- still plausibly **well under 10 minutes**, especially if the worker stays warm for the whole episode

This is comfortably inside your target.

## 5. License + practical risk
- **License:** generally acceptable for local use; low practical risk for your project context.
- **Availability:** excellent; weights are still widely mirrored/distributed.
- **Practical risk:** low.

This matters more than people admit. “Can I still get the weights in six months?” is a real criterion.

## 6. Subprocess load behavior
- **First load:** usually tolerable; often **10–30 seconds** depending on storage and node graph.
- **VRAM spike during load:** usually modest and manageable compared with FLUX-class models.
- Good fit for your **spawned sidecar worker** model.

This is important under C3. SDXL behaves like a normal citizen in a subprocess. It does not require weird lifecycle management.

## Net
**Best default anchor model.**

---

## 2) FLUX.1-schnell

## Verdict
**Viable, but not my top pick for this project.**

## 1. VRAM footprint
This is the first place I’d be careful.

Your own estimate of **~12 GB in fp8** is plausible in some ComfyUI setups, but I would budget more conservatively for real-world peaks:

- **fp8 / optimized path**: roughly **11.5–14 GB**
- depending on text encoder residency, VAE, and implementation details, it can flirt with your ceiling
- **fp16** is much less comfortable and may exceed your preferred real-world headroom

I would not call FLUX.1-schnell “obviously safe” under a hard **14.5 GB peak** rule unless you have already tested the exact workflow.

## 2. Aesthetic match
Mixed.

Strengths:
- Strong prompt adherence
- Good scene coherence
- Can produce very cinematic environments
- 4-step speed is attractive

Weaknesses:
- The output often reads **too clean, too contemporary, too ad-like**
- Harder to get convincing **analog ugliness**
- “still frame from a movie” can drift into “AI key art”
- The **1980s VHS/UHF/CRT grime** look is less turnkey than on SDXL

You can prompt your way part of the way there, but the style ecosystem is thinner, which matters more than raw model quality.

## 3. LoRA / IP-Adapter compatibility in 2026
This is the main reason I would not choose it as default.

- **LoRA ecosystem:** still materially weaker than SDXL
- **Period-style LoRAs:** much thinner
- **IP-Adapter / style-reference tooling:** improving, but not as boringly reliable as SDXL
- Environment-only reference workflows exist, but the ecosystem is not as deep

For your exact aesthetic, the style-control deficit matters.

## 4. Per-frame wall-clock
Potentially very good:

- **4-step generation** can be very fast
- likely around **4–10 seconds/frame** once loaded
- maybe a bit more depending on resolution and graph overhead

So yes, it can beat SDXL on throughput.

But for your use case, **speed is not the bottleneck**. Style control and fit are.

## 5. License + practical risk
- **Apache 2.0** is excellent.
- Availability is generally good.
- Practical risk is moderate, mostly around ecosystem maturity rather than legal issues.

## 6. Subprocess load behavior
- **Load time:** heavier than SDXL
- **VRAM load spike:** more concerning than SDXL
- More likely to create “works most of the time but occasionally tips over” behavior near your ceiling

Given Jeffrey’s “if it fits, fits; if it doesn’t, pick smaller” rule, I don’t love living on the edge here.

## Net
**Good secondary option**, especially for fast, clean environment anchors.  
**Not the best default for SIGNAL LOST’s analog-period look.**

---

## 3) FLUX.1-dev

## Verdict
**Do not pick this for Phase B default.**

## 1. VRAM footprint
For your constraints, this is the killer.

- **12B-class model**
- **fp16**: very likely over your practical ceiling
- even reduced-memory paths may be uncomfortable or require exactly the kind of optimization work you explicitly do not want

I would treat this as **non-viable** under your “no low-level VRAM optimization chasing” rule.

## 2. Aesthetic match
Yes, it can look great. That is not the issue.

## 3. LoRA / IP-Adapter compatibility
Improving, but again not enough to offset the memory problem.

## 4. Per-frame wall-clock
Even if you got it running, startup/load cost and memory pressure make it a poor fit.

## 5. License + practical risk
You already noted the **non-commercial license**. That alone may or may not matter to you now, but it’s an avoidable future constraint.

## 6. Subprocess load behavior
Worst of the listed candidates for your setup.

## Net
**Reject for this role.** Too heavy, too finicky, unnecessary.

---

## 4) SD 1.5 + LoRA stack

## Verdict
**Best fallback / safety valve, not best primary.**

## 1. VRAM footprint
Very comfortable.

- **fp16** at your likely working sizes: roughly **4–7 GB**
- even with LoRAs and adapters, usually still well below ceiling
- easiest model here to run without surprises

## 2. Aesthetic match
This is where it loses to SDXL.

Strengths:
- Can do **grimy analog vibe** surprisingly well
- Huge retro/film/VHS LoRA ecosystem
- Sometimes the lower-fidelity prior actually helps for “broadcast artifact” aesthetics

Weaknesses:
- Environment fidelity is weaker
- Composition is less stable
- Fine scene detail and spatial coherence are worse
- More likely to look “AI-ish” or “muddy” at 1024-wide unless carefully managed
- Can collapse into generic retro sludge

For anchor frames feeding motion, you want a strong still image prior. SD1.5 is usable, but not ideal.

## 3. LoRA / IP-Adapter compatibility in 2026
Excellent.

- Massive LoRA library
- Mature IP-Adapter support
- Very easy to operate locally

## 4. Per-frame wall-clock
Fast.

- likely **3–10 seconds/frame**
- startup/load also light

## 5. License + practical risk
Low practical risk. Widely available.

## 6. Subprocess load behavior
Best-behaved of the bunch.

## Net
If SDXL gives you trouble, **this is the fallback I’d actually trust**.  
But I would not choose it first if SDXL fits cleanly, which it should.

---

## 5) Diffusion360 / panoramic SDXL

## Verdict
**Special-purpose tool, not your default anchor model.**

## 1. VRAM footprint
Usually worse than plain SDXL for the same base family, especially if you actually exploit panoramic widths.

- At pano-friendly resolutions, you can easily push beyond your comfortable ceiling
- At restrained widths, the pano advantage shrinks

## 2. Aesthetic match
Potentially useful for:
- establishing shots
- wide streets
- skyline / city / radio tower / freeway / strip-mall exteriors

But:
- your downstream is **LTX-2.3 clips capped around 10–12 s**
- anchor frames need to read as **shot frames**, not stitched panorama showcases
- pano models often encourage compositions that feel like environment plates, not cinematic singles

## 3. LoRA / IP-Adapter compatibility
Depends on exact implementation, but generally less straightforward than plain SDXL.

## 4. Per-frame wall-clock
Slower than standard SDXL if you use wider canvases.

## 5. License + practical risk
Varies by exact model fork. More practical uncertainty.

## 6. Subprocess load behavior
More likely to stress VRAM due to larger latent sizes.

## Net
Keep this as an **optional specialty model for rare establishing shots**, not the Phase B default.

---

# 6) Anything else relevant in 2026

A few things are worth naming, but I would not necessarily recommend them over SDXL here.

## SDXL Lightning / Turbo variants
These can be useful if you care about speed, but for anchor frames:
- they often trade away too much image richness
- can look more synthetic
- less ideal if you want a strong still image to hand off downstream

I would treat them as experimentation tools, not default Phase B picks.

## Juggernaut / RealVisXL / other SDXL finetunes
This is the one “anything else” I’d actually consider seriously.

A good **SDXL finetune oriented toward cinematic realism/environment work** may outperform base SDXL for your use case, while preserving the same ecosystem advantages.

Caution:
- some finetunes drift toward glossy modern photography
- some are overfit to portrait beauty aesthetics
- licensing/distribution can be murkier than base SDXL
- determinism can vary slightly by implementation, but usually acceptable

If you find a **stable, locally available SDXL finetune that excels at moody environments**, that may be even better than base SDXL 1.0. But I would still frame the family choice as **SDXL-class**, not a different architecture.

## Stable Cascade / similar
Interesting historically, but not the obvious low-risk choice for your current stack.

## Sana / PixArt / other newer open image models
Some may be competitive on paper, but:
- ecosystem maturity in ComfyUI,
- LoRA availability,
- Windows-local boringness,
- and deterministic repeatability

all matter more here than benchmark novelty.

For this project, I would not go off-road unless you have a tested reason.

---

# Ranking for Phase B anchor role

## 1. SDXL 1.0 + curated VHS/CRT LoRA stack
**Top pick.** Best balance of fit, style control, speed, and risk.

## 2. SD 1.5 + LoRA stack
Not the prettiest answer, but the **best fallback** if you want maximum operational safety.

## 3. FLUX.1-schnell
Strong model, fast, but weaker fit for your analog-period style and less comfortable ecosystem-wise.

## 4. Diffusion360 / panoramic SDXL
Useful niche tool for occasional establishing shots only.

## 5. FLUX.1-dev
Not appropriate under your constraints.

---

# My actual recommendation

## Phase B default
**SDXL 1.0 base/refiner-free workflow, single-pass, with a minimal LoRA stack:**
- one **VHS/analog degradation** LoRA
- one **CRT/phosphor/glow** LoRA if truly needed
- maybe one **cinematic 80s / low-light / sodium vapor** style LoRA
- stop there

Do **not** build a five-LoRA Christmas tree. That’s how you get poster sludge.

## Why not FLUX.1-schnell as default?
Because your problem is not “make a pretty image quickly.”  
Your problem is “make a seedable, period-faithful, environment-first shot frame locally, every time, inside a boring subprocess.”

That is an **SDXL problem**, not a FLUX problem.

---

# Practical operating advice

Given C7 and your subprocess model:

- Keep the anchor generator in a **fully separate spawned worker**, as you already require.
- Load model once per episode, generate all shot anchors, then tear down.
- Cache by:
  - model identifier
  - LoRA stack identifiers + weights
  - prompt text
  - negative prompt
  - seed
  - width/height
  - sampler
  - steps
  - CFG
- Save the resulting anchor PNGs and metadata so reruns can skip generation.

That gives you the “same prompt + seed => same anchor” behavior as much as the stack allows.

I’ll be honest: exact bitwise image identity across machines / driver versions is harder than people pretend. But on the **same machine, same build, same graph, same seed**, SDXL is about as sane as you’re going to get.

---

# Where I disagree with likely pro-FLUX arguments

If another reviewer tells you FLUX.1-schnell should win because it is newer, faster, and more prompt-faithful, my pushback is:

1. **Prompt adherence is not your limiting factor.**  
   You already have deterministic shot prompts. You need style control and operational boringness.

2. **The LoRA/style ecosystem matters more than raw model quality.**  
   SIGNAL LOST lives or dies on analog-period texture.

3. **Near-ceiling VRAM behavior is a real product risk.**  
   Jeffrey explicitly does not want optimization-chasing.

4. **Anchor frames are not hero stills.**  
   They are inputs to motion clips. “Good enough, consistent, stylable, fast” beats “state-of-the-art but fussier.”

That’s why I rank SDXL first.

---

# Final answer

## Top pick: **SDXL 1.0 + curated CRT/VHS LoRA stack**

Use it as the default Phase B anchor model.

## Runner-up: **SD 1.5 + LoRA stack**
Use only if you hit unexpected SDXL stability/VRAM issues and want the safest fallback.

## Not recommended as default:
- **FLUX.1-schnell** — viable but not best fit
- **FLUX.1-dev** — too heavy / wrong risk profile
- **Diffusion360** — niche establishing-shot tool only

If you want, I can turn this into a **one-page implementation decision memo** with:
- recommended ComfyUI node pattern for the sidecar worker,
- a minimal SDXL LoRA stack policy,
- and a concrete cache-key schema for deterministic anchor reuse.
