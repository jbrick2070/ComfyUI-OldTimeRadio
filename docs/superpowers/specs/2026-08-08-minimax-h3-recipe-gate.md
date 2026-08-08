# MiniMax H3 — Recipe Gate (H3 Gate 0)

**Date:** 2026-08-08
**Author:** Claude (Cowork mode)
**Status:** Unblocked, not started. Licensing cleared 2026-08-07 — see `docs/licensing/MINIMAX_H3_AUTHORIZATION.md`.
**Verdict up front:** H3 is worth testing, but two things are already known before a byte is downloaded. It is a **15-second shot generator**, not an episode renderer. And **no published variant fits 14.5 GB resident** — the smallest diffusion model on the Hub is 15.6 GB and the smallest text encoder is 14.6 GB, so running H3 here requires the weight-streaming mechanism this project has already discarded. See §2 Q1. That decision is the gate.

This document is the testing protocol. It follows the same discipline as the HyWorld Gate 0 in `ROADMAP.md`: **no integration code until the gate produces measured numbers.** Every figure below marked *(reported)* comes from secondary sources and is a hypothesis to falsify on our own hardware, not a fact to build on.

---

## Platform Pins (ground truth — anything contradicting these is wrong)

- RTX 5080 Laptop, 16 GB VRAM, Blackwell sm_120, single GPU, no cloud.
- Windows, Python 3.12, torch 2.10.0, CUDA 13.0, SageAttention + SDPA.
- Flash Attention 2/3: NOT AVAILABLE. Do not chase.
- 100% local, offline-first, no API keys, no paid services.
- VRAM ceiling: 14.5 GB real-world target.
- **Audio is king. Full narrative output must never break, shorten, or degrade.**
- `trust_remote_code=False` enforced globally on all model loaders.

---

## 1. What H3 actually is

MiniMax H3 (Hailuo 3.0), released 2026-07-31, weights published 2026-08-03. One omni-modal transformer that takes text, images, video and audio and returns **video with native stereo audio, up to 2K, up to 15 seconds per clip**. 33B parameters. ComfyUI shipped day-0 support.

Two access paths exist. Only one is legal *and* pin-compliant for us:

| Path | Pin-compliant? | Licensed for us? |
|---|---|---|
| MiniMax platform API (`MiniMax-H3`) | ❌ — violates "no API keys, no paid services" | n/a |
| **Local open weights** | ✅ | ✅ **only** via the 2026-08-07 authorization |

The US is an Excluded Territory under the H3 Community License. Without the authorization on file, the local path would be unlicensed too. Read `docs/licensing/MINIMAX_H3_AUTHORIZATION.md` §5 before shipping anything derived from this gate.

---

## 2. The four hard questions

These decide whether H3 is viable here. Answer them in order; a failure at #1 or #2 ends the gate.

### Q1 — Does any variant fit 14.5 GB?

**On current evidence, no — not fully resident.** These are actual file sizes from the Hub, not reported figures.

H3 is a four-part stack: a diffusion model (pick **one** of FL2VA or Ref2VA), the Qwen3-VL-32B text encoder, a video VAE, and an audio VAE.

**`Comfy-Org/MiniMax-H3`** — the ComfyUI repackage, 3.1M downloads, the obvious starting point for a ComfyUI node pack:

| Component | Variant | Size |
|---|---|---|
| Diffusion (FL2VA *or* Ref2VA) | bf16 | 66.3 GB |
| | int8_convrot | 34.0 GB |
| | pruned_bf16 | 40.2 GB |
| | **pruned_fp8_scaled** | **21.0 GB** |
| | **pruned_int8_convrot** | **21.0 GB** |
| Text encoder (Qwen3-VL-32B) | bf16 | 51.5 GB |
| | int8_convrot | 27.1 GB |
| | **nvfp4_awq** | **15.7 GB** |
| Video VAE | fp16 | 5.2 GB |
| Audio VAE | fp32 | 0.6 GB |

Smallest viable Comfy-Org stack: 21.0 + 15.7 + 5.2 + 0.6 = **42.5 GB**. That exactly reproduces the widely-quoted "42.5 GB minimum download," which confirms the reading.

**`Abiray/MiniMax-H3-GGUF`** — goes lower, and ships two ComfyUI workflow JSONs plus a NOTICE file:

| Component | Quant | Size |
|---|---|---|
| FL2VA / Ref2VA unet | **Q3_K_M / Q3_K_S** | **15.6 GB** |
| | Q4_0 | 18.6 GB |
| | Q4_K_M | 19.9 GB |
| | Q8_0 | 36.0 GB |
| Text encoder | **Q4_K_M** | **14.6 GB** |
| | int4_convrot | 15.0 GB |

Smallest GGUF stack: 15.6 + 14.6 + 5.2 + 0.6 = **36.0 GB** on disk.

**The verdict.** The smallest diffusion model in existence today is **15.6 GB (Q3_K_M)** — already above our 14.5 GB ceiling and above the card's 16 GB before a single activation, VAE tensor, or context frame. The smallest text encoder is **14.6 GB**, which by itself consumes the entire budget. Nothing here fits.

That does not make H3 impossible, but it forces a specific architecture:

1. **Sequential residency** — load encoder, encode the prompt, fully evict, then load the diffusion model. The two can never coexist.
2. **Block-swap / layer streaming from system RAM** for the diffusion model, since even Q3_K_M overflows.

Point 2 is the problem. `ROADMAP.md` lists **"Weight streaming from system RAM via ComfyUI-Manager"** and **"Asynchronous weight streamer as a fallback for 16 GB OOM"** under *discarded ideas — do not revisit*. H3 on this card requires exactly the class of mechanism we already rejected. Either that decision gets revisited deliberately, with the wall-clock cost measured, or H3 doesn't run here. **Do not let this get re-litigated silently inside an integration PR.**

The "~8 GB minimum via 4-bit DiffSynth-Studio" figure circulating in coverage is layer-by-layer streaming — the same discarded mechanism, described optimistically. It is not a resident-weights number.

**Download traps:**
- `Abiray/MiniMax-H3-GGUF` lists `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` at **27.1 GB**, which is byte-for-byte the size of Comfy-Org's *int8_convrot*, not its 15.7 GB nvfp4_awq. The file looks mislabeled. **Pull the nvfp4 encoder from `Comfy-Org`,** not from there.
- FL2VA and Ref2VA are the same size. Download **one**. Picking both doubles a 21 GB line item for nothing.
- Confirm ≥ 60 GB free first. `models/` is gitignored, so none of this touches the repo.

### Where the files go

The models root is derived, not hardcoded. `prestartup_script.py:42-45` walks three directories up from this custom-node folder and sets `HF_HOME` beneath it:

```
C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\prestartup_script.py
                ↑ three up
HF_HOME = C:\Users\jeffr\Documents\ComfyUI\models\huggingface
```

So the root is `C:\Users\jeffr\Documents\ComfyUI\models\`. The `Comfy-Org/MiniMax-H3` repo layout deliberately mirrors ComfyUI's own subfolder names — each file drops into the same-named directory:

| File | Destination |
|---|---|
| `minimax_h3_fl2va_pruned_fp8_scaled.safetensors` | `models\diffusion_models\` |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `models\text_encoders\` |
| `minimax_h3_video_vae_fp16.safetensors` | `models\vae\` |
| `minimax_h3_audio_vae_fp32.safetensors` | `models\vae\` |

GGUF path (`Abiray`): the unet goes in `models\unet\` per the ComfyUI-GGUF convention; encoder still `models\text_encoders\`; VAEs unchanged.

**The trap.** `HF_HOME` is set, so a bare `hf download Comfy-Org/MiniMax-H3` dumps ~42 GB into `models\huggingface\hub\` in blob-plus-symlink layout — which ComfyUI's `diffusion_models` / `text_encoders` / `vae` loaders do not scan. The weights would be on disk, invisible, and Windows symlink handling makes it worse. Download **individual files to explicit destinations** instead:

```powershell
hf download Comfy-Org/MiniMax-H3 diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors `
  --local-dir "C:\Users\jeffr\Documents\ComfyUI\models"
```

That preserves the repo's `diffusion_models/…` prefix under the models root, landing the file exactly where the loader looks. Repeat per file; skip whichever of FL2VA/Ref2VA you are not using.

**Verify the root before downloading 42 GB.** The ComfyUI desktop app splits its install directory (`C:\Users\jeffr\AppData\Local\Programs\ComfyUI\resources\ComfyUI`) from the user directory (`C:\Users\jeffr\Documents\ComfyUI`, passed as `--user-directory` in `scripts/soak_operator.py:259`). `folder_paths.models_dir` follows the *install* path unless an `extra_model_paths.yaml` redirects it, while `HF_HOME` above follows the *user* path. Those can differ. Confirm which root is live before committing the download:

```powershell
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -c "import folder_paths; print(folder_paths.models_dir)"
```

If C: is tight, `extra_model_paths.yaml` can point these four entries at the E: drive already in use for archives (`soak_operator.py:31`). 42 GB is a lot to put on a user-profile volume.

### Q2 — Do the quantized kernels exist for sm_120 + torch 2.10 + CUDA 13 on Windows?

H3's memory reductions lean on int8 convrot quantization and custom kernels. **This is the exact trap Flash Attention 2 set on this platform**: the technique exists, the wheel does not. Verify wheel availability for Blackwell laptop + Windows *before* downloading 42.5 GB. Cheapest possible check, highest possible cost if skipped.

### Q3 — Does it load under `trust_remote_code=False`?

README standing rule #7 enforces this globally and says it may not change without explicit security review. If the H3 loaders require remote code, that is a **hard stop and an escalation**, not a flag to flip in passing. Native ComfyUI day-0 support may sidestep this entirely — check the ComfyUI path first.

### Q4 — Where does a 15-second clip fit in a 30+ minute episode?

It does not, directly. H3 is a shot generator. The only sane placement is the **v2 visual sidecar** — see `2026-04-12-otr-v2-visual-sidecar-design.md` — generating per-scene clips against Director output, cut to the existing audio master.

Two consequences follow immediately:

- **Mute H3's native audio.** OTR masters its own 48 kHz stereo from Bark, Kokoro, MusicGen and procedural SFX. H3's native stereo track fights that master and cannot be allowed to shorten or displace narrative audio. Video track only. "Audio is king" is not negotiable for a visual experiment.
- **H3 overlaps HyWorld.** ROADMAP items 3, 4, 6 and 8 are all blocked on HyWorld Gate 0, which is itself unrun. Running two visual bets in parallel on one 16 GB GPU splits attention and VRAM budget. **Decide explicitly which bet leads** before either consumes real effort.

---

## 3. Gate protocol

Run in order. Record peak VRAM and wall time for every step. Stop at the first hard failure and record why.

0. **Settle the weight-streaming question first.** Q1 shows nothing fits resident. Decide, explicitly and on the record, whether the discarded "streaming from system RAM" mechanism is back on the table for H3 only. If the answer is no, the gate ends here and costs nothing. *(Do this before any download.)*
1. **Kernel/wheel survey.** Confirm quantized-inference support for sm_120 + torch 2.10 + CUDA 13 + Windows. Pass/fail. *(Also before any download.)*
2. **Disk check.** Confirm ≥ 60 GB free before pulling weights.
3. **Acquire the smallest viable stack.** GGUF Q3_K_M unet + Q4_K_M encoder (36.0 GB) is the floor; Comfy-Org pruned_fp8_scaled + nvfp4_awq (42.5 GB) is the ComfyUI-native option. Pick one path, one of FL2VA/Ref2VA, and pin the exact revision hash — §VII.1 of the license disclaims any support or update obligation, so upstream can move without notice. Destinations and the `HF_HOME` download trap: §2 Q1, "Where the files go". Verify the live models root *first*.
4. **Loader smoke test.** Load under `trust_remote_code=False`. Record peak VRAM at load, before any generation.
5. **Minimum-viable generation.** One short clip at the lowest supported resolution and frame count. Record peak VRAM, wall time, and whether the OTR pipeline's own weights can coexist or must be fully evicted first.
6. **Target-shape generation.** One clip at the resolution and duration a real scene sidecar would need. Record the same three numbers.
7. **Handoff test.** Confirm `_flush_vram_keep_llm()` discipline holds — Director weights cleared before H3 runs, H3 cleared before Bark runs. This is where a 16 GB card actually dies.
8. **Audio isolation.** Confirm the native stereo track can be discarded cleanly without affecting video output.

Write results to `docs/superpowers/specs/h3_gate0_results.md`. No integration code before that file exists.

---

## 4. Gate decision

- **All pass** → H3 becomes a candidate for the v2 visual sidecar. Write the recipe, then re-open the H3-vs-HyWorld decision with real numbers on both sides.
- **Fits VRAM but too slow** → park it. Record the wall-clock number; revisit when quantization improves. A visual pass that triples episode render time fails the "queue it and walk away" promise in the README.
- **Q1 or Q2 fails** → stop. Record the measurement, add H3 to the discarded-ideas list with the reason, and let HyWorld Gate 0 proceed unblocked. This is the most likely outcome given 26.9 GB peak on a larger card.
- **Q3 fails** → escalate to explicit security review. Do not flip `trust_remote_code`.

---

## 5. Out of scope for this gate

- Any public release of an H3 node. That is a separate decision with its own licensing review — `MINIMAX_H3_AUTHORIZATION.md` §5 explains why the MIT repo cannot carry H3 rights to downstream users.
- Distributing weights or Model Derivatives through this repo. Never.
- The MiniMax platform API. Violates the offline-first pin.
- Using H3 or its Outputs to train or improve any other model. Prohibited by license §V.3.

---

## 6. References

- `docs/licensing/MINIMAX_H3_AUTHORIZATION.md` — the clearance and its limits
- `2026-04-12-otr-v2-visual-sidecar-design.md` — where H3 would slot in
- `ROADMAP.md` — platform pins, HyWorld Gate 0 protocol this gate mirrors
- MiniMax H3 Community License — `https://huggingface.co/MiniMaxAI/MiniMax-H3/LICENSE`
- ComfyUI day-0 H3 support — `https://blog.comfy.org/p/minimax-h3-day-0-support-in-comfyui`
- `Comfy-Org/MiniMax-H3` — ComfyUI repackage, safetensors, source of the §2 Q1 sizes
- `Abiray/MiniMax-H3-GGUF` — GGUF quants down to Q3_K_M, plus two ComfyUI workflow JSONs worth reading as recipe starting points
