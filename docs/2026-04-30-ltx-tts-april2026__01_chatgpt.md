# Round A -- ChatGPT (gpt-5.4) elapsed=67.4s

Here’s the blunt, architecture-first answer.

---

# Executive take

## Q1: Background video model
**Best 16 GB sidecar choice today:** **LTX-Video 0.9.x / 2B-class**, not the giant 19B/22B quantized variants.

Why:
- Your background layer is **supporting scenery**, not hero motion.
- You care more about **stability, determinism, and operational headroom** than benchmark prestige.
- On a **16 GB laptop Blackwell** with a **14.5 GB ceiling**, the 19B/22B quantized path is exactly the kind of “fits on paper, hurts in production” choice I would avoid.
- The smallest change with biggest payoff is: **stay with the original roadmap pick unless you have a demonstrated quality failure**.

If you insist on a larger LTX:
- **19B distilled Q4_K_M** is the only one I’d even consider before 22B Q5_K_M.
- But I would still treat it as **experimental**, not the default production sidecar.

## Q2: TTS expansion
**Best next addition:** **CosyVoice 2** first.  
**Second:** **Qwen3-TTS** only after you validate determinism, Windows friction, and pronunciation control in your exact pipeline.

Why:
- CosyVoice has the best fit for your constraints: local, permissive, practical, multilingual/voice-cloning capable, and generally better aligned with “production utility” than novelty.
- For radio drama, the biggest payoff is not raw model novelty; it’s **predictable pronunciation, voice consistency, and easy per-character assignment**.

---

# QUESTION 1 — Animated background layer model pick

## Recommendation: default to **LTX 0.9 2B fp16** for production

Given your exact use case:
- **12 fps slow-mo**
- **background only**
- **1–2 clips per scene**
- **HuMo foreground carries the acting**
- **sidecar load/unload means no stacked peak VRAM**
- **stability > max params**

…the 2B-class model is the right systems choice.

### Why I would not default to 19B/22B on this machine
Even if the quantized weights nominally fit:
1. **Quantized model size is not total runtime memory.**
   You still need:
   - activations
   - KV/cache-like internal state depending on implementation
   - VAE
   - scheduler buffers
   - latent tensors
   - ComfyUI node overhead
   - CUDA allocator fragmentation on Windows

2. **Laptop 16 GB is not desktop 24 GB.**
   “Fits” and “runs comfortably in repeated scene batches” are different things.

3. **Your quality bottleneck is probably not background model parameter count.**
   For a radio-drama composite:
   - foreground lip-sync readability
   - scene art direction consistency
   - temporal steadiness
   - compositing quality  
   matter more than squeezing another few points of texture realism out of the background.

4. **Quantized bleeding-edge video inference on sm_120 is still a moving target.**
   You explicitly do **not** want low-level VRAM optimization work. Good. Then don’t choose the path most likely to demand it.

So: **LTX 0.9 2B fp16** remains the best default production pick unless you have side-by-side evidence that it fails your art bar.

---

## If you want one “bigger” option: **19B distilled Q4_K_M**, not 22B Q5_K_M

If you’re determined to exploit the 16 GB tier for better backgrounds:

### Better gamble
- **LTX-2 19B distilled, Q4_K_M**
- Reason: more headroom than 22B Q5_K_M, likely fewer “one scene works, next scene OOMs” surprises.

### Worse gamble
- **LTX 2.3 / 1.1 22B Q5_K_M ~14 GB**
- On a **14.5 GB practical ceiling**, this is too close for comfort.
- Especially on Windows + ComfyUI + sidecar orchestration.

My architecture advice:
- If you test larger LTX at all, make it an **opt-in “high background quality” preset**, not the baseline.

---

## “Is 1.1 the current LTX 2.3 release, or is there something newer?”

I need to be explicit: **I’m not fully confident in the naming/version mapping you’re using** because LTX naming in the wild has been messy across:
- official Lightricks releases
- community ports
- ComfyUI wrappers
- distilled checkpoints
- GGUF conversions

From your wording, you appear to mean:
- **“LTX 2.3 (1.1 release)”** as a community/runtime packaging label, not necessarily the canonical upstream model family name.

### My confidence level
- **Low-to-moderate confidence** that your “1.1” is a packaging/release label around the newer 2.x family rather than a newer official architecture generation.
- I would **not** make an architecture decision based on that label alone.

### What I would do
Pin by:
1. **exact Hugging Face repo**
2. **exact model filename**
3. **exact commit SHA**
4. **exact ComfyUI custom node version**

If you want, I can help you build a **pin list format** for `models.md` / `REPRODUCIBILITY.md` so this doesn’t drift.

---

## Quantization gotchas on Blackwell sm_120

Short version: **assume immature edges until proven otherwise**.

### What I’d expect
- **FP16/BF16**: safest operational path
- **GGUF quantized inference**: may work, but behavior depends heavily on the specific backend and custom node implementation
- **FP8 / NVFP4**: I would **not** plan around these yet for ComfyUI production video inference unless you have a tested, pinned stack

### Why
Blackwell support often arrives in layers:
1. PyTorch recognizes the device
2. CUDA kernels compile/run
3. xFormers / Triton / custom ops catch up
4. ComfyUI custom nodes stop making architecture assumptions
5. quant backends become stable

You are on:
- **Windows**
- **torch 2.10**
- **CUDA 13**
- **sm_120**
- **ComfyUI**
- likely multiple custom nodes

That stack is exactly where “advertised support” and “production-safe support” diverge.

### My recommendation
For this project:
- **Do not make FP8/NVFP4 a dependency**
- Treat them as future upside only
- Prefer **plain fp16/bf16-supported paths** that already work

That aligns with your “smallest change, biggest payoff” rule.

---

## Non-LTX alternatives worth considering

For your use case, I would only consider alternatives if they beat LTX on **operational simplicity**, not just quality demos.

## 1) **HunyuanVideo small/distilled variants** — maybe
Pros:
- often good motion/style quality
- active ecosystem

Cons:
- tends to be heavier operationally
- often less attractive than LTX for constrained local sidecar use
- can drag you into exactly the optimization rabbit hole you don’t want

Verdict:
- **Not my first recommendation** for your 16 GB sidecar background-only role.

## 2) **Wan 2.2 5B** — only if you want stack reuse
You mentioned it already.

Pros:
- reuse one family you already know
- simpler maintenance if you already support it elsewhere

Cons:
- your own architecture says **8 GB tier has no background layer**
- if this is only for 16 GB sidecar, Wan doesn’t obviously beat LTX 2B on “smallest change”
- may be overkill or just a parallel maintenance burden

Verdict:
- **Only choose Wan if operational reuse is more valuable than model specialization**.

## 3) **Frame interpolation + still-image background generation**
This is the strongest non-LTX alternative in pure systems terms.

Pipeline:
1. generate a high-quality still background
2. apply subtle pan/zoom/parallax
3. optionally generate sparse keyframes
4. interpolate to target cadence
5. apply your `setpts=PTS*2,fps=12`

Why this is attractive:
- dramatically lower VRAM
- more deterministic
- often enough for “vintage radio drama” staging
- backgrounds in this genre do not need blockbuster motion

If your goal is “animated enough to feel alive,” this may outperform heavyweight video generation in production reliability.

### My candid opinion
For **OTR SIGNAL LOST**, the most architecture-sane answer may be:
- **Use LTX 0.9 2B only where true background animation matters**
- otherwise use **still/parallax/interpolation**
- reserve larger video models for rare showcase scenes

That’s the biggest payoff per complexity dollar.

---

## Q1 final recommendation

### Production default
- **LTX 0.9 2B fp16**

### Optional experimental high-quality preset
- **LTX-2 19B distilled Q4_K_M**

### Avoid as default on 16 GB laptop
- **22B Q5_K_M** near your ceiling

### Best non-LTX alternative
- **still/parallax + interpolation**, especially for radio-drama staging

---

# QUESTION 2 — TTS model expansion candidates

## Best next addition: **CosyVoice 2**

This is the clear first move.

Why:
- permissive licensing path is usually workable for MIT projects
- practical local deployment
- good voice quality / cloning / multilingual utility
- likely better production maturity than many newer entrants
- useful for per-character palette expansion without replacing Bark/Kokoro

For your pipeline, the key question is not “which model sounds most impressive in a demo?”  
It is:
- can you get **repeatable pronunciation**
- can you get **stable voice identity**
- can you run it locally on **8 GB and 16 GB tiers**
- can you keep integration effort low

CosyVoice 2 is the best fit on that axis.

---

## CosyVoice 2 vs CosyVoice 3

My confidence here is limited by release churn. As of your date, I would **not assume CosyVoice 3 is the safer production choice** unless you have verified:
- official release status
- license
- Windows install path
- local inference maturity
- deterministic behavior in your exact stack

### Architecture advice
- Treat **CosyVoice 2** as the production-grade baseline
- Treat **CosyVoice 3** as “evaluate later” unless it is clearly official, stable, and better documented

In other words:
- **don’t chase version numbers**
- chase **operational maturity**

---

## Qwen3-TTS as second candidate

This is worth evaluating, but not before CosyVoice 2.

### Why it’s interesting
- likely strong expressive quality
- modern instruction/control possibilities
- potentially useful for character differentiation

### Risks
- may be heavier than you want
- may have more moving parts
- pronunciation/phoneme control may be less straightforward than the marketing implies
- Windows + local setup may be rougher
- determinism may be harder to lock down

For a lip-sync-driven pipeline, I would rank:
1. **pronunciation control**
2. **timing consistency**
3. **voice identity**
4. **expressiveness**

A model that sounds amazing but drifts in pronunciation is a bad fit for HuMo-driven production.

---

## Strong local TTS candidates for April 2026

Here’s the shortlist I’d actually spend time on.

## Tier 1: evaluate now

### 1) **CosyVoice 2**
Best first addition.

Use cases:
- narrator voices
- recurring characters
- accent/style variation
- voice cloning where legally/ethically appropriate

Watch for:
- exact license text on model weights vs code
- Windows install friction
- whether inference path is stable on your torch/CUDA combo

### 2) **Qwen3-TTS**
Second choice.

Use cases:
- more expressive or stylized characters
- broader palette if CosyVoice voices feel too “clean” or modern

Watch for:
- memory footprint
- inference latency
- pronunciation controls
- reproducibility

---

## Tier 2: only if you need a specific capability

### 3) **Piper**
Not state-of-the-art in naturalness, but still relevant.

Why mention it:
- tiny
- local
- stable
- deterministic-friendly
- easy deployment

Why it may matter:
- for minor roles, announcers, utility voices, or fallback mode, Piper is operational gold

Why not primary:
- probably not rich enough as your main dramatic voice expansion

### 4) **XTTS-family / Tortoise-descendants / StyleTTS-family**
These can be tempting, but I’d be careful.

Common issues:
- license ambiguity
- Windows friction
- inconsistent maintenance
- slower inference
- less predictable production behavior

Verdict:
- evaluate only if you have a specific gap CosyVoice/Qwen don’t fill

---

## Any newer Apache-2.0 / MIT TTS from the last 6 months?

I can’t responsibly name a “must-have landed in the last 6 months” without checking current upstream repos, because this is exactly where hallucination risk is high:
- many projects launch with code-only permissive licenses but model weights under separate terms
- some are research previews, not production releases
- some have Linux-first assumptions that hurt Windows

### So my candid answer
- **I do not currently have high confidence in a newer permissive TTS entrant that clearly displaces CosyVoice 2 for your use case.**
- If one exists, it still has to beat CosyVoice 2 on:
  - local Windows deployment
  - pronunciation control
  - memory fit
  - reproducibility
  - license clarity

That is a high bar.

---

## Period-style controls: “1940s broadcast”, “mid-century radio aesthetic”

### Explicit built-in controls?
I am **not aware of any mainstream local permissive TTS model with reliable explicit native controls for “1940s radio announcer” as a first-class parameter**.

You’ll likely get better results by combining:
1. **voice selection / cloning**
2. **prompt/style text**
3. **post-processing chain**

### For this project, the real “period sound” should come from post
Use TTS for:
- diction
- cadence
- timbre baseline

Then apply a deterministic mastering chain:
- band-limit
- mild tube/saturation emulation
- room/plate flavor if needed
- noise floor
- wow/flutter only if deterministic and subtle
- AM/radio EQ shaping

That is where the “vintage radio” identity should live.

This is also better for **C7 byte-identical audio**, because:
- style prompting can drift
- DSP can be pinned exactly

So don’t over-index on finding a TTS with magical “1940s” controls.  
Build the aesthetic in the mastering stage.

---

## Candidates to avoid

## 1) **Fish Speech**
You already rejected it. I agree.
- license mismatch for your MIT posture
- stop there

## 2) Any model with unclear weight license
If code is Apache/MIT but weights are:
- research-only
- non-commercial
- custom restricted  
then it’s a no-go for your stated policy.

## 3) Linux-first repos with fragile Windows support
For your workstation:
- avoid projects that require hand-patching CUDA extensions
- avoid repos that depend on rapidly changing Triton kernels
- avoid “works on my Ubuntu 24.04” stacks unless there is clear Windows evidence

## 4) Models that require stochastic decoding for quality
Because of **C7 byte-identical audio**, be very suspicious of:
- sampling-heavy generation paths
- hidden nondeterminism in vocoders
- dynamic batching behavior
- fused kernels that vary numerically across runs

If a TTS stack cannot be made byte-identical on your machine, it is not production-safe for OTR.

That may eliminate some otherwise impressive models.

---

# Determinism note for both questions

You emphasized **audio output must remain byte-identical between runs**.

For TTS this is critical. My advice:

## Safe policy
- keep **Bark + Kokoro** as the known-good baseline
- add new TTS engines only behind a **determinism qualification test**

Test protocol:
1. same prompt
2. same seed if applicable
3. same model revision
4. same driver / torch / CUDA / cuDNN
5. same batch size
6. same output format
7. hash final WAV bytes

Run:
- 10 repeated generations
- across cold start and warm start
- across process restarts

If hashes differ, the engine is **not qualified**.

For video backgrounds, exact byte identity is usually less critical than audio in your description, but if you want reproducible renders there too, apply the same discipline.

---

# Smallest-change / biggest-payoff recommendations

## Decision 1
**Keep LTX 0.9 2B fp16 as the 16 GB sidecar default.**

Add only:
- an optional experimental preset for **19B distilled Q4_K_M**
- strict fallback to 2B on any instability

Do **not** make 22B Q5_K_M your baseline.

## Decision 2
**Add CosyVoice 2 next.**
- Keep Bark + Kokoro
- Add CosyVoice 2 as a selectable per-character voice backend
- Gate release on byte-identical determinism tests

## Decision 3
**Treat Qwen3-TTS as a lab candidate, not immediate production default.**

---

# What I would pin, specifically

Because you asked for repo/tag/SHA style specificity, but I don’t want to bluff exact SHAs I can’t verify here, my recommendation is to pin these fields in your repo immediately:

```yaml
background_video:
  family: ltx-video
  upstream_repo: "<HF repo exact name>"
  model_file: "<exact safetensors/gguf filename>"
  upstream_commit: "<git or HF commit SHA>"
  comfyui_node_repo: "<exact custom node repo>"
  comfyui_node_commit: "<SHA>"
  precision: "fp16|bf16|q4_k_m|q5_k_m"
  frames_rule: "8n+1"
  sampler_steps: 4-8
  postprocess: "setpts=PTS*2,fps=12"

tts:
  engine: "cosyvoice2"
  upstream_repo: "<exact repo>"
  model_revision: "<tag/SHA>"
  tokenizer_revision: "<tag/SHA>"
  vocoder_revision: "<tag/SHA>"
  decode_mode: "<greedy/etc>"
  sample_rate: "<Hz>"
  wav_hash_test: true
```

That will save you pain later.

---

# Final answers, concise

## Q1
- **Best 16 GB sidecar choice today:** **LTX 0.9 2B fp16**
- **If using a larger LTX:** **19B distilled Q4_K_M** before 22B Q5_K_M
- **Is “1.1” current?** Unclear; don’t trust the label alone—pin exact repo/file/SHA
- **Blackwell quant gotchas:** yes; don’t depend on FP8/NVFP4 yet
- **Non-LTX alternatives:** best practical one is **still/parallax + interpolation**; Wan only if stack reuse matters

## Q2
- **Strongest next local TTS candidate:** **CosyVoice 2**
- **CosyVoice 2 vs 3:** use **2** unless 3 is clearly official and production-proven in your stack
- **Newer permissive TTS in last 6 months:** none I can confidently say displaces CosyVoice 2 for your constraints
- **Period-style controls:** not reliably native; do the vintage sound in deterministic post-processing
- **Avoid:** Fish Speech, unclear-weight-license models, Linux-fragile repos, and nondeterministic TTS stacks

If you want, I can turn this into a **go/no-go matrix** with columns for:
- VRAM fit
- Windows friction
- determinism risk
- license risk
- integration effort
- expected payoff

That would make the final pick easier.
