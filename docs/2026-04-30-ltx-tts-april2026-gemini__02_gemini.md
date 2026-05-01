# Round B -- Gemini (gemini-3-pro-preview) elapsed=43.0s

Here is my architectural assessment for OTR SIGNAL LOST, grounded in your constraints for the 16 GB Blackwell tier. 

I will be blunt: your VRAM ceiling and your "no low-level optimization" rule are on a collision course if we don't pick the right models. Let's de-risk this.

---

### QUESTION 1: Animated Background Layer (16 GB Tier)

**The Verdict: Reject the 22B and 19B models. Use LTX 0.9 2B (FP8/FP16) or Wan 2.2 5B.**

Here is the VRAM math that kills the 14 GB and 12 GB GGUF candidates:
You have a 14.5 GB ceiling. ComfyUI base overhead is ~1 GB. If you load a 12 GB Q4_K_M model, you are at 13 GB. **You have forgotten the Temporal VAE.** 
Video VAE decoding is a massive memory spiker. In ComfyUI (`comfy/sd.py` and `comfy/model_management.py`), decoding an 8n+1 latent sequence to pixel space will easily consume 2-3 GB of VRAM unless you are doing aggressive tiled decoding (which violates your "no low-level VRAM optimization" rule). A 12 GB model *will* OOM during the VAE decode step on a 16 GB card.

*   **Which LTX variant TODAY?** Stick to the original roadmap: **LTX 0.9 2B**. It fits trivially (~5 GB). Because this is a *background layer* that will be blurred/composited under a foreground and slowed down via ffmpeg, throwing 22B parameters at it is a massive waste of compute and a severe stability risk. 
*   **Non-LTX Alternative:** **Wan 2.2 5B** (Assuming an FP8 or BF16 native load). If you already have this cached for the 8 GB tier, reusing it for the 16 GB tier's background layer is the definition of "smallest change, largest payoff." You get cache reuse, guaranteed VRAM safety (~8 GB), and plenty of headroom for the VAE.
*   **Blackwell sm_120 / Quantization Gotchas:** Blackwell natively accelerates FP8 and NVFP4. GGUF (via `llama.cpp` or custom Kijai nodes) often has to dequantize weights to FP16 in VRAM just-in-time for attention layers, causing hidden VRAM spikes. Native PyTorch FP8 (`torch.float8_e4m3fn`) is vastly more stable on CUDA 13 / sm_120 than relying on GGUF hacks. 
*   **The "1.1" Release:** Lightricks versioning can be messy, but whether it's 1.1 or 2.3, the parameter count is the physical law we care about. 22B is too big for a sidecar on 16 GB without Flash Attention chasing and weight streaming.

**Recommendation:** Use **Wan 2.2 5B** in native FP8 if you already support it, or **LTX 0.9 2B**. Run your `ffmpeg -i input.mp4 -vf "setpts=2.0*PTS" -r 12` post-process. It will look perfectly fine as a background composite.

---

### QUESTION 2: TTS Model Expansion

**The Verdict: CosyVoice 2 is your safest production bet, but look at Parler-TTS for the vintage aesthetic.**

**Rule C7 (Byte-Identical Audio) is your biggest enemy here.** 
Autoregressive TTS models (like Bark) are notoriously difficult to make 100% deterministic across different GPU architectures or even different runs if the sampler isn't perfectly seeded and `torch.use_deterministic_algorithms(True)` isn't enforced. 

*   **CosyVoice 2 vs 3:** Stick to **CosyVoice 2**. It uses Flow Matching, which relies on ODE solvers. ODE solvers with a fixed number of steps and a fixed noise seed are mathematically much easier to make byte-identical than autoregressive top-k sampling. If CosyVoice 3 is still new/bleeding-edge in your April 2026 timeline, avoid it. You want stability, not the latest paper.
*   **Newer Apache-2.0 Alternatives (The Vintage Aesthetic):** You need to look at **Parler-TTS** (specifically the Mini or Large variants, Apache 2.0). Parler is an acoustic-prompted TTS. You can literally prompt it with: *"A male speaker with a deep voice, speaking through a vintage 1940s radio microphone with heavy static and vinyl crackle."* 
    *   *Payoff:* This eliminates the need for complex DSP/ffmpeg audio post-processing chains to fake the OTR aesthetic. 
    *   *VRAM:* Parler-TTS Mini is tiny (~1-2 GB). Fits both 8 GB and 16 GB tiers flawlessly.
*   **Phoneme Control:** If strict phoneme control (for HuMo lip-sync) is more important than the acoustic aesthetic, **Kokoro** (which you already have) remains the undisputed king of low-VRAM, highly deterministic phoneme-to-audio generation. 
*   **Candidates to AVOID:** 
    *   **Qwen3-TTS:** While Qwen makes great models, their audio models often rely on complex autoregressive + flow-matching hybrids. Ensuring Rule C7 (byte-identical output) on a new Qwen architecture will likely require digging into their generation scripts and patching their sampler. Skip it.
    *   **Any TTS requiring Flash Attention 3:** On CUDA 13 / sm_120, FA3 is incredibly fast but often non-deterministic by default. If a TTS model mandates FA3 to fit in VRAM, it will violate Rule C7 unless you force standard math backends, which will then blow up your VRAM.

**Recommendation:** 
1. Keep **Kokoro** as the default for strict lip-sync.
2. Add **CosyVoice 2** for high-expressivity character voices (ensure the ODE solver seed is hardcoded).
3. Evaluate **Parler-TTS** for characters where the "1940s broadcast" acoustic prompt provides a massive aesthetic payoff without DSP hacking. 

**Smallest Change / Largest Payoff Summary:**
Drop the 14GB GGUF background models immediately to save yourself weeks of VRAM debugging. Use Wan 2.2 5B or LTX 0.9 2B. For audio, integrate Parler-TTS for native vintage acoustics, and rely on CosyVoice 2 for your heavy-hitting expressive voices. Ensure `torch.backends.cudnn.deterministic = True` is set globally in your ComfyUI `__init__.py` or main execution script to protect Rule C7.
