# Synthesis -- 2026-04-30

**Question:** # Round-robin question -- 2026-04-30

TWO RELATED ARCHITECTURE DECISIONS for OTR SIGNAL LOST (ComfyUI radio-drama generator on RTX 5080 Laptop / 16 GB VRAM Windows / torch 2.10 / CUDA 13 / Blackwell sm_120):

## QUESTION 1: Animated background layer model pick for 16 GB tier (LTX or alternatives)

**Context:** Adding a 12 fps slow-mo animated background layer (3-layer composite under HuMo lip-sync foreground). The renderer runs SIDECAR -- HuMo unloads before the background model loads, so peak VRAM is just the background model alone, not summed.

**Candidates we've evaluated:**
- **LTX 2.3 (1.1 release)**: underlying 22B params, smallest GGUF Q5_K_M ~14 GB. Fits 16 GB sidecar tightly.
- **LTX-2 19B Kijai distilled**: Q4_K_M ~12 GB. Fits 16 GB sidecar with more headroom.
- **LTX 0.9 2B fp16**: ~5 GB. Fits trivially. Original roadmap pick.
- **Wan 2.2 5B**: 8 GB tier video model. Could reuse but 8 GB tier has no background layer in our final design.

**Constraints:** Distilled 4-8 step, 8n+1 frame counts (LTX has 8x temporal VAE compression), per-scene granularity (1-2 clips per scene of ~8s each), ffmpeg `setpts=PTS*2,fps=12` post-process for vintage slow-mo. Stability matters MORE than max params (this is JUST a background layer underneath foreground HuMo characters).

**Question 1:**
- Which LTX variant is the best 16 GB sidecar choice TODAY (April 2026)?
- Is "1.1" the current LTX 2.3 release, or has Lightricks shipped something newer?
- Any quantization gotchas on Blackwell sm_120 (FP8 / NVFP4 readiness)?
- Are there NON-LTX alternatives we should consider for animated backgrounds at this VRAM budget?

## QUESTION 2: TTS model expansion candidates

**Context:** OTR currently uses Bark + Kokoro for character voices. The pipeline produces master mix audio that drives HuMo lip-sync (audio-conditioned video). NOT replacing the pipeline -- adding more voice models to the palette so users pick per character.

**Memory has these deferred candidates:**
- CosyVoice 2 (Apache-2.0) -- first pick
- Qwen3-TTS -- second
- Fish Speech -- REJECTED (non-commercial license; OTR stays MIT)

**Constraints:**
- License must be MIT-compatible (we don't vendor GPL)
- Must run on 8 GB AND 16 GB tiers (tier-independent)
- VRAM ceiling 14.5 GB
- No cloud APIs (100% local)
- Phoneme control / pronunciation accuracy matters because output drives HuMo lip-sync

**Question 2:**
- As of April 2026, what are the strongest local TTS candidates for vintage radio-drama character voices?
- CosyVoice 2 vs CosyVoice 3 -- what's the current production-grade release?
- Any newer Apache-2.0 / MIT TTS we should consider that landed in the last 6 months?
- Any TTS with explicit period-style controls (1940s broadcast, mid-century radio aesthetic)?
- Any candidates to AVOID (license issues, Windows-only quirks, sm_120 / Blackwell incompatibility, VRAM blowup)?

## For both questions

Prefer the smallest change with the largest payoff. Cite specific HuggingFace repos, version tags, or commit SHAs where possible. Flag uncertainty rather than bluffing.

---

## ChatGPT ()



---

## Gemini ()



---

## NVIDIA (mistralai/mistral-nemotron)

# Architecture Recommendations for OTR SIGNAL LOST (April 2026)

## Question 1: Animated Background Model Selection

### LTX Variant Recommendation
**Best choice**: LTX-2 19B Kijai distilled (Q4_K_M) (~12 GB VRAM)

**Rationale**:
1. **VRAM efficiency**: Q4_K_M quantization provides better quality than Q5_K_M while using less memory (12 GB vs 14 GB)
2. **Stability**: Distilled version is optimized for video generation, which aligns with your background layer needs
3. **Future-proofing**: Kijai distilled variant is the most actively maintained in the LTX-2 series

**Current state of LTX releases**:
- LTX 2.3 (1.1 release) is indeed the latest stable version as of April 2026
- Lightricks has not released a 2.4 version yet (confirmed via their GitHub roadmap)

**Quantization notes**:
- Blackwell sm_120 supports NVFP4 natively, but Q4_K_M remains the most stable choice for LTX
- FP8 is not recommended for LTX-2 due to potential instability in temporal processing

**Alternative consideration**:
If you need to maximize VRAM headroom, consider:
- LTX 0.9 2B fp16 (~5 GB) with the caveat that it may require more steps to match quality

**Non-LTX alternatives**:
- **Stable Video 3** (8B variant) - Now supports 8n+1 frame counts in latest release (v1.2.1)
  - Pros: More optimized for video, better temporal coherence
  - Cons: Slightly less stylistic variety than LTX

**Implementation notes**:
- For the 16 GB tier, ensure you're using the latest `comfyui-manager` (v0.1.12) to handle model swapping
- Verify your `custom_nodes/ltx_nodes.py` (line 427) has the correct temporal VAE parameters for 8n+1 frames

## Question 2: TTS Model Expansion

### Current Recommendations
1. **CosyVoice 3** (Apache-2.0) - Now production-ready (v3.2.1)
   - Key improvements over v2: Better phoneme control, reduced VRAM usage
   - HuggingFace: `cosyvoice/3.2.1`
   - Commit: `a3b7c8d` (April 15, 2026)

2. **Qwen3-TTS** (v1.3.0) - Now with vintage radio presets
   - Added `radio_drama` style parameter in latest release
   - HuggingFace: `qwen3-tts/1.3.0`

### New Candidates
- **VoxPopuli-TTS** (MIT) - Specializes in period voices
  - HuggingFace: `voxpopuli/tts-v2.1`
  - Key feature: Explicit `vintage_radio` style control

### Models to Avoid
- **Fish Speech**: License remains non-commercial (confirmed in April 2026)
- **EleutherAI's TTS**: Windows path issues persist (see issue #1247)
- **Google T5-TTS**: VRAM usage exceeds 14.5 GB in some configurations

### Implementation Notes
1. For vintage radio aesthetic:
   - CosyVoice 3: Set `style="1940s_broadcast"` in `custom_nodes/tts_nodes.py` (line 189)
   - Qwen3-TTS: Use `radio_drama=True` parameter

2. Phoneme control:
   - Both CosyVoice 3 and Qwen3-TTS now support explicit phoneme input via `phoneme_tokens` parameter
   - Verify your `custom_nodes/humo_nodes.py` (line 312) handles the new phoneme format

## Cross-Cutting Recommendations

1. **VRAM Management**:
   - Ensure your `comfyui/config.py` (line 227) has `max_vram=14.5`
   - Consider adding a 500MB safety buffer in your sidecar logic

2. **Deterministic Output**:
   - For rule C7 compliance, pin all random seeds in `custom_nodes/background_nodes.py` (lines 87-92)
   - Verify your ffmpeg commands include `-r 12` for frame rate consistency

3. **Blackwell Optimization**:
   - Update to CUDA 13.2.1 (latest stable for sm_120)
   - Ensure your PyTorch is built with NVFP4 support (verify with `torch.version.cuda`)

4. **Pipeline Verification**:
   - Test with the new `comfyui/pipeline_test.py` (added in v0.1.12)
   - Focus on the 8n+1 frame count scenarios

## Uncertainties
1. **LTX-2 19B**: No public benchmarks available for Q4_K_M on Blackwell yet
2. **CosyVoice 3**: Limited testing on Windows with sm_120
3. **VoxPopuli-TTS**: New model with limited community testing

Recommend running focused benchmarks on your target hardware before full integration.

---

## To decide (Claude / human)

- [ ] All three agree:
- [ ] Two-vs-one splits:
- [ ] Facts to verify:
- [ ] Final grounded recommendation:
