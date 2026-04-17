# 02_round2_fidelity.md -- model=gpt-4.1 elapsed=21.1s

## User prompt

ROUND 2 of 3 -- Graphical fidelity & replacement modules.

You've now seen the full HyWorld stack. Reality check:

- WorldMirror 2.0 is the ONLY shipped HY-World 2.0 model. Pano / Stereo / Nav are all "Coming Soon" with no ETA.
- The current worker.py is in stub mode: it writes solid-color PNGs and uses ffmpeg's `zoompan` filter to make Ken Burns motion clips driven by camera adjectives from the shotlist. That's the visual today.
- The design doc Section 11 lists candidate replacement modules: `Diffusion360_ComfyUI` (text -> 360 pano, SDXL-based), `SPAG4D` (pano -> 3DGS, ~6-8 GB VRAM), `ComfyUI-Sharp` (image -> 3DGS, sub-1s, very low VRAM), `SplaTraj` (semantic camera path planning), with `ComfyUI-3D-Pack` as an umbrella.
- Hardware: RTX 5080 Laptop, 16 GB VRAM, Blackwell sm_120, Windows. Python 3.12, torch 2.10, CUDA 13.0 in main env. Worker runs CPU-only today; GPU work must wait for Bark TTS to finish or coordinate via a VRAM gate.
- HARD CONSTRAINT: audio output must be byte-identical to v1.7 baseline (C7). Audio is king. Worker must never starve the audio pipeline of VRAM during generation. Bark TTS holds the GPU ~12-18 min per episode.

YOUR TASKS:

1. **Lowest-risk highest-payoff visual upgrade.** Of the candidate stand-in modules, which single one should land first? Justify in 3 bullets: visual payoff, risk, integration cost.

2. **Sequencing.** Propose a 3-step ladder from today's Ken Burns stub to "real visual fidelity." For each step, name the module + what visual capability it adds + what blocks it from being step 1.

3. **Missing modules.** What's NOT in the design doc that I should consider? Specifically:
   - Anchor image generation (the design doc waves a hand at "any text-to-image model"). What's the right local SDXL/Flux pickfor SIGNAL LOST's CRT/broadcast-static aesthetic?
   - Splat renderer (we have splat generators; what actually turns a PLY into MP4 frames?)
   - VRAM gate / queue between worker and main ComfyUI (need a coordinator since worker shouldn't fight Bark for VRAM)
   - CRT post-FX pass (renderer mentions `crt_postfx` boolean but nothing implements it)
   - Caching strategy (anchor images per scene? splats per env? hash key shape?)

4. **Replacement module shopping list.** For each missing piece in (3), name a specific local-only open-source repo or model and a 1-sentence reason. Prefer ComfyUI-native nodes when they exist. Be honest about what doesn't have a clean answer yet.

5. **Audio coexistence pattern.** Sketch a coordinator design (dataclass + state machine + 4-5 events) where the worker can request GPU time without crashing Bark or causing OOM. Should it be a file-lock, a ZMQ/Redis token, a simple FIFO queue? What's the simplest thing that's actually safe?

Output as numbered sections matching the 5 tasks. Be specific.

---

## Assistant reply

**1. Lowest-risk highest-payoff visual upgrade**

**Recommended module:** **Diffusion360_ComfyUI** (text-to-360-pano, SDXL-based)

- **Visual payoff:** Instantly upgrades from solid-color or Ken Burns stubs to plausible, richly-detailed panoramic environments for each scene, matching the OTR `[ENV:]` tokens; this is a dramatic step up in immersion and variety.
- **Risk:** Very low—Diffusion360_ComfyUI is SDXL-based, mature, and already used in ComfyUI pipelines; no experimental 3D or VRAM-hungry geometry involved, just text-to-image.
- **Integration cost:** Minimal—can be called as a ComfyUI node or via subprocess, and output is a simple image file; fits into the current pipeline as the anchor image source for each scene.

---

**2. Sequencing: 3-step ladder to real visual fidelity**

**Step 1:**  
**Module:** Diffusion360_ComfyUI  
**Adds:** Scene-specific 360-degree panoramic images, replacing solid-color stubs; enables visually rich backgrounds for each scene.  
**Blocker for step 1:** None—runs on SDXL, can be CPU or GPU, and doesn't require 3D or VRAM coordination if run after Bark.

**Step 2:**  
**Module:** ComfyUI-Sharp (image-to-3DGS)  
**Adds:** Converts anchor images (from Step 1) into navigable 3D Gaussian splats (.ply), enabling simple fly-throughs or camera moves in 3D space.  
**Blocker:** Requires a splat renderer to turn .ply into MP4, and must coordinate VRAM if run on GPU; also, fidelity is limited by input image and splat renderer.

**Step 3:**  
**Module:** SplaTraj (semantic camera path planning) + Splat Renderer (e.g., SplatFusion or custom)  
**Adds:** Camera trajectories through 3DGS splats, matching shotlist camera adjectives, with CRT/broadcast post-processing; enables dynamic, emotionally-driven motion and full scene immersion.  
**Blocker:** Splat rendering on Windows is still experimental, and SplaTraj integration may require custom path mapping; also, VRAM coordination is essential for safe coexistence with Bark.

---

**3. Missing modules**

- **Anchor image generation:** The design doc punts to "any text-to-image model," but for SIGNAL LOST's CRT/broadcast-static look, you want a model that can do grungy, analog, or retro-futurist scenes. SDXL is the baseline, but SDXL-tuned checkpoints or LoRAs are better.
- **Splat renderer:** You have splat generators (ComfyUI-Sharp, SPAG4D), but nothing to render .ply splats to MP4 frames. Need a local renderer that works on Windows, ideally headless.
- **VRAM gate/queue:** No explicit coordinator between worker and main ComfyUI; worker could start a GPU job while Bark is running, risking OOM.
- **CRT post-FX pass:** The renderer exposes a `crt_postfx` boolean, but no actual CRT/analog video effect is implemented.
- **Caching strategy:** No mention of how to cache anchor images or splats per scene/environment; no hash key design to avoid redundant generations.

---

**4. Replacement module shopping list**

- **Anchor image generation:**  
  - **Repo/Model:** [Realistic Vision V6.0](https://civitai.com/models/4201/realistic-vision-v60) (SDXL checkpoint), or [DreamShaper XL](https://civitai.com/models/17188/dreamshaper-xl10) with a CRT/analog LoRA like [CRT TV Effect LoRA](https://civitai.com/models/19207/crt-tv-effect-lora).  
  - **Reason:** Both are SDXL-based, work in ComfyUI, and can be further styled with LoRAs for analog/CRT artifacts.

- **Splat renderer:**  
  - **Repo/Model:** [SplatFusion](https://github.com/ashawkey/splatfusion) (for rendering .ply splats to video), or [3D Gaussian Splatting ComfyUI extension](https://github.com/Acly/comfyui-3dgs) if it matures for Windows.  
  - **Reason:** SplatFusion is the most robust local renderer for Gaussian splats, can be scripted, and outputs MP4s; ComfyUI-3DGS is native but less mature/stable on Windows.

- **VRAM gate/queue:**  
  - **Repo/Model:** [filelock](https://pypi.org/project/filelock/) Python package, or a simple custom file-based lock in the shared `io/` directory.  
  - **Reason:** File-based locking is simple, cross-process, and works on Windows; avoids the complexity of Redis/ZMQ for a solo workstation.

- **CRT post-FX pass:**  
  - **Repo/Model:** [ComfyUI-VideoFX](https://github.com/pythongosssss/ComfyUI-VideoFX) or [ComfyUI-Advanced-PostProcessing](https://github.com/ltdrdata/ComfyUI-Advanced-PostProcessing) for scanlines, chroma bleed, and analog artifacts.  
  - **Reason:** Both provide node-based post-processing, including CRT/analog effects, and are easily inserted after video concat.

- **Caching strategy:**  
  - **Repo/Model:** Use a hash of (`episode_title`, `scene_id`, `env_prompt`) as the cache key for anchor images and splats; store in a local cache dir with filenames like `<anchor_hash>.png` or `<anchor_hash>.ply`.  
  - **Reason:** Deterministic, avoids recomputation, and is trivial to implement with standard Python.

---

**5. Audio coexistence pattern: Coordinator design**

**Coordinator:**  
A simple file-based lock system in the shared `io/` directory, managed by a `VRAMCoordinator` dataclass/state machine. Each process (Bark TTS or worker) requests the GPU by acquiring the lock, and releases it when done. The lock file contains the PID and job info for transparency.

**Dataclass Example:**
```python
@dataclass
class VRAMCoordinator:
    lockfile: Path = Path("io/vram.lock")

    def acquire(self, job_id: str, timeout: int = 1800) -> bool:
        # Try to atomically create lockfile; if exists, wait/retry
        # Write job_id and PID to lockfile
        ...

    def release(self, job_id: str):
        # Remove lockfile if owned by this job_id
        ...

    def status(self) -> dict:
        # Return current owner, PID, job_id, timestamp
        ...
```

**State machine events:**
- **REQUEST_GPU:** Worker or Bark tries to acquire the lock before starting GPU work.
- **GRANT_GPU:** Lock acquired; safe to proceed.
- **RELEASE_GPU:** Job done; lock released.
- **TIMEOUT_WAITING:** Waited too long for lock; fail gracefully or retry.
- **ABORT_OWNER_DEAD:** If lockfile exists but PID is not alive, forcibly release.

**Why file-lock:**  
- Works on Windows, no extra dependencies, no network config.
- No risk of deadlocks if you check PID liveness and timestamp.
- Easy to debug if something goes wrong (just delete the lockfile).

**Summary:**  
This is the simplest robust solution for solo, local, single-GPU coordination—no need for Redis or ZMQ unless you scale up.

---

**TL;DR:**  
- Land Diffusion360_ComfyUI first for huge visual upgrade with minimal risk.
- Ladder: Ken Burns stub → Diffusion360 anchor images → 3DGS splats + splat renderer + camera path.
- Missing: CRT LoRA/FX, splat renderer, VRAM gate, caching.
- Use Realistic Vision/DreamShaper XL + CRT LoRA, SplatFusion, file-based lock, ComfyUI-VideoFX, and hash-based cache.
- File-lock is the safest, simplest VRAM coordinator for your use case.
