# HuMo Daisy-Chain — Honest Path Forward (A/B Split)

**TL;DR:** HuMo doesn't have native long-video support yet. The flashy `WanVideo Long I2V Multi/InfiniteTalk` mechanism is built for InfiniteTalk + Wan 2.1/2.2 base, not for HuMo's TIA conditioning. Two paths below.

---

## What's actually true about HuMo today

| Fact | Source |
|---|---|
| HuMo trained on 97-frame clips at 25fps | HuMo README |
| "Checkpoint for Longer Generation" is a TODO, not released | HuMo README roadmap |
| Issue #1250 "HUMO infinite talk" still OPEN since Sep 2025 | kijai repo |
| `WanVideo Long I2V Multi/InfiniteTalk` node = InfiniteTalk + Wan base | Issue #1941 thread |
| HuMo + InfiniteTalk can be stacked (workflow exists) | RunningHub 1968348721056501761 |
| In stacked workflow: InfiniteTalk drives long-video lip-sync; HuMo drives character/identity | Same workflow's node graph |

**Translation:** the long-video mechanism doesn't natively run on HuMo. You either work within HuMo's window, stack it with InfiniteTalk, or use a different model.

---

## A — What works with HuMo right now

### A1. Single-window generation (97 frames max, ~3.9 sec at 25fps)

Use HuMo as designed. Window-bounded clips, no long-video tricks.

```
[FLUX portrait] + [audio ≤4 sec] + [text prompt]
  → HuMo 17B fp8 (TIA mode)
  → 97-frame clip
```

Quality ceiling for your 5080 / 16GB / lightx2v stack. Reliable.

### A2. RGB-space chain across separate runs

The original approach we discussed. Works, but accumulates drift.

- Generate clip → extract last frame → use as next ref
- ADAIN/mkl color match **back to the original FLUX portrait** (never to previous clip's last frame)
- Optional: IP-Adapter Identity Wash anchored to clean FLUX portrait
- Expect visible micro-jump at each seam (per HuMo's end-of-sequence anchor mechanism)
- Practical limit: 3–4 hops before drift becomes unacceptable

### A3. HuMo + InfiniteTalk stack (verified workflow)

Get long-video continuity by pairing HuMo with InfiniteTalk in one graph. RunningHub workflow `1968348721056501761` documents this — they use it for 15-sec music videos.

```
[FLUX portrait] → HuMo (character/identity, 480p)
              ↓
       [InfiniteTalk] (long-video lip-sync via motion_frame)
              ↓
         [stitched output]
```

Required models (Kijai mirrors):
- `Wan2_1-HuMo-14B_fp16.safetensors` → `models/diffusion_models`
- `whisper_large_v3_encoder_fp16.safetensors` → `models/audio_encoders`
- InfiniteTalk weights from `MeiGen-AI/InfiniteTalk` → `models/diffusion_models`

This is the closest thing to a verified long-video HuMo path that exists today.

### A4. CivitAI Wan.Humo Music Video Automation (community workflow)

Model 2058189 on CivitAI. Multi-scene HuMo with auto-stitching. Adapt for narrative use, not just music videos. Community-tested but unverified by me.

---

## B — Parking lot for v2 (different models, different pipeline)

When you're ready to evaluate alternatives that have native long-video support, these are the candidates. Each requires a separate model swap and pipeline rebuild.

### B1. InfiniteTalk + Wan 2.1/2.2 (closest to HuMo's purpose)

- Talking-head / audio-driven lip-sync
- Native long-video via `WanVideo Long I2V Multi/InfiniteTalk` node
- `motion_frame` mechanism with built-in color correction
- Trade-off vs HuMo: weaker character identity preservation, stronger continuity
- Fits your 16GB budget (GGUF variants available)

### B2. Stable Video Infinity 2.0 (Wan 2.2 I2V A14B)

- LoRA on Wan 2.2 I2V designed specifically for long sequences
- 5-pass chained sampling with motion latent forwarding
- Built-in tail-frame blending between passes
- No audio conditioning — text/image only. Not a HuMo replacement, but a long-shot solution
- Source: Kijai/WanVideo_comfy → Stable-Video-Infinity v2.0
- Fits 16GB with GGUF Wan 2.2 base

### B3. LTX 2.3 (parking — too big for current hardware)

- 22B params, native audio-video sync at model level
- 4K/50fps capable
- Requires 24GB VRAM minimum — **does not fit your 16GB 5080**
- Park until next hardware refresh or quantized variant

### B4. HuMo "Longer Generation" checkpoint (vapor as of Apr 2026)

- Listed as TODO in HuMo README since Sep 2025
- Promised October 2025, not yet shipped
- Worth setting a watch on Phantom-video/HuMo repo
- If/when released, supersedes everything in section A

---

## Recommendation

1. **Today (Tier A):** Build on A3 (HuMo + InfiniteTalk stack). Verified to work, fits your hardware, gets you to ~15-second clips with reasonable continuity.
2. **Bridge:** A2 (RGB chain) for narrative segments where the InfiniteTalk lip-sync isn't needed (cutaways, environment shots).
3. **Parking lot for v2:** B1 (pure InfiniteTalk/Wan) when HuMo's character preservation is not the bottleneck. B2 for non-talking long shots.
4. **Watch:** Phantom-video/HuMo repo for the longer-gen checkpoint. If it ships, revisit everything.

---

## Cowork next steps (lean)

- [ ] Pull RunningHub workflow 1968348721056501761, verify it loads on your stack
- [ ] Test 15-sec HuMo+InfiniteTalk output, measure identity stability vs A2 baseline
- [ ] If A3 works: parse its console output for window-boundary timing, feed to Cowork drift detector
- [ ] If A3 fails on 5080: fall back to A1/A2, document what broke
- [ ] Set GitHub watch on Phantom-video/HuMo for longer-gen release
- [ ] Park B1/B2 evaluation until A3 ceiling is known

---

## What changed from the previous MD

The previous version implied `WanVideo Long I2V Multi/InfiniteTalk` would just work with HuMo. It won't — that node was built for InfiniteTalk-as-primary, not HuMo-as-primary. The verified path is the **stacked** workflow (A3), not feeding HuMo into the long-video node directly.

---

## Cowork fact-check (added 2026-04-25 by Claude)

Verified against current sources before adopting into ROADMAP:

- ✓ HuMo trained on 97 frames @ 25 fps, longer-gen checkpoint TODO — confirmed in [Phantom-video/HuMo README](https://github.com/Phantom-video/HuMo/blob/main/README.md)
- ✓ Issue [#1250](https://github.com/kijai/ComfyUI-WanVideoWrapper/issues/1250) "HUMO infinite talk" opened Sep 18 2025 by Maelstrom2014, still open
- ✓ Issue [#1941](https://github.com/kijai/ComfyUI-WanVideoWrapper/issues/1941) about motion_frame on `WanVideo Long I2V Multi/InfiniteTalk` opened Feb 18 2026 — confirms node was built for InfiniteTalk-as-primary
- ✓ [MeiGen-AI/InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk) — real repo, audio-driven unlimited-length talking video
- ✓ [RunningHub workflow 1968348721056501761](https://www.runninghub.ai/post/1968348721056501761) — "Humo+InfiniteTalk+Character MV production", real, 15-sec MV via one character ref + audio + 3 MV descriptions
- ✓ [Stable Video Infinity 2.0](https://github.com/vita-epfl/Stable-Video-Infinity) — vita-epfl, ICLR 26 Oral, Wan 2.2 I2V A14B base, 5-pass chained sampling, "Error Recycling Fine Tuning"
- ⚠️ **HuMo 14B vs 17B caveat (REAL):** the brief's A3 workflow specifies `Wan2_1-HuMo-14B_fp16.safetensors`, but the actual Kijai mirror at [Kijai/WanVideo_comfy_fp8_scaled](https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled) is **fp8_e5m2 / fp8_e4m3fn scaled**, not fp16, and is **14B** parameters not 17B. OTR currently runs HuMo 17B GGUF from [VeryAladeen/Wan2_1-HuMo_17B-GGUF](https://huggingface.co/VeryAladeen/Wan2_1-HuMo_17B-GGUF). Adopting A3 means downloading the 14B Kijai variant; the two are not drop-in interchangeable.
- ⚠️ **whisper encoder filename caveat:** brief specifies `whisper_large_v3_encoder_fp16.safetensors`, OTR has `whisper_large_v3_fp16.safetensors` — verify these are the same weights before swapping.
- ⚠️ **CivitAI Model 2058189 (A4):** community-unverified by me — flag as "tested by community, unverified locally" if pulled.
- ⚠️ **LTX 2.3 22B fit claim:** consistent with OTR memory `reference_ltx_keep_only_2b.md` (19B/22B variants too big for 16 GB; only 2B v0.9 retained). No conflict.
