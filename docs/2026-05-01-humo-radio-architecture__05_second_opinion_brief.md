# OTR v2.0 — Second-opinion brief (HuMo + radio still architectural fix)

**Author:** Jeffrey Brick, with Claude  
**Date:** 2026-05-01  
**Project:** ComfyUI-OldTimeRadio (v2.0-alpha)  
**Constraints:** RTX 5080 Laptop, 16 GB VRAM (14.5 GB ceiling), Blackwell sm_120, Windows, torch 2.10.0, CUDA 13.0, SDPA + SageAttention only, 100% local/offline. Final episode audio must be byte-identical to the v1.5 baseline at every gate (rule "C7"). Audio is king — video work that breaks audio gets reverted.

---

## Problem statement

The v2.0 architecture sets a "wall-to-wall HuMo coverage" rule: every audio second of every episode is rendered as a HuMo (audio-driven talking-head, ByteDance, fp8) lip-sync clip. To cover non-dialogue events (announcer narration, intro/outro/interstitial music windows, standalone SFX), the system was wired to feed HuMo a separate FLUX-rendered "radio still" image (`output/otr/stills/radio_bookend_<episode>.png`) as the I2V reference for those lines. The premise: "people speaking get people lip-syncing, the radio is the visual performer for everything else."

In a clean run today (episode `signal_lost_..._20260501_110019`, commit 620013e), FLUX correctly rendered a 1940s broadcast-radio still (1.04 MB PNG, vintage console with knobs and oscilloscope, no faces). The render ledger correctly stamped both ANNOUNCER lines (l001, l021) with `ref_source = "radio-still (announcer)"` and `ref_png_name = radio_bookend_<episode>.png`. But ffmpeg-extracting frame 0 of `l001.mp4` and `l021.mp4` shows two *different* generic blonde women in domestic interiors. Zero radio imagery. The radio still was never visible.

Tracing the dispatch end-to-end:
- `nodes/batch_humo_render.py:1195-1199` correctly resolves `ref_png = radio_still_path` for ANNOUNCER lines.
- `_png_to_tensor` (line 375-385) correctly loads the radio PNG into `[1,H,W,C]` float32 [0,1].
- `WanHuMoImageToVideo` is correctly invoked with `ref_image=entry["ref_image"]` (line 1476).

The OTR-side code is doing exactly what it claims. The mismatch is in what HuMo's `ref_image` parameter actually does. Reading the upstream stock node body in `comfy_extras/nodes_wan.py:1070-1108`:

```python
if ref_image is not None:
    ref_image = comfy.utils.common_upscale(ref_image[:1].movedim(-1,1), W, H, "bilinear", "center").movedim(1,-1)
    ref_latent = vae.encode(ref_image[:, :, :, :3])
    positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [ref_latent]}, append=True)
    negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [torch.zeros_like(ref_latent)]}, append=True)
...
latent = torch.zeros([batch_size, 16, latent_t, H//8, W//8], ...)   # generation latent starts ZEROS
```

`ref_image` is VAE-encoded into a soft `reference_latents` field appended to the positive conditioning. The diffusion latent starts from zeros — KSampler denoises from random noise. This is the standard generic Wan2.1 I2V conditioning path; it is not specific to HuMo. What restricts the actual rendered output to face-only animation is the HuMo finetuned weights (trained on talking-head data), not the conditioning code. With per-line seeds `(seed + idx*1009) & 0x7FFFFFFFFFFFFFFF`, idx=0 and idx=20 get different seeds, the prompt builder produces "A announcer character speaks calmly with subtle facial expressions, dimly lit interior, ambient cinematic lighting, 35mm film grain, shallow depth of field," and HuMo generates two unrelated generic faces. The radio still is encoded and attached but contributes no face identity, so the model defaults to text-prompt-driven generation with no identity lock.

Net: the architectural premise that "the radio is the visual performer for non-dialogue lines" is incompatible with HuMo specifically as the renderer. HuMo will not animate a non-human reference; passing one produces an unconstrained face anyway. The wall-to-wall HuMo coverage rule has to give for non-dialogue lines.

---

## Three fix candidates already considered

**Option 1.** ANNOUNCER becomes a real cast member with a 1940s-radio-host portrait rendered by FLUX, routed through the existing portrait chain to HuMo like any character. `music_*` and standalone `sfx` lines stop going to HuMo entirely — they render as deterministic static-video clips assembled in post via ffmpeg (`-loop 1 -i radio.png -t <frame-rounded-dur> -r 25 -c:v libx264 -pix_fmt yuv420p -an`). The radio still becomes a bookend / interlude motif (open + close, plus longer non-dialogue cutaways). Zero VRAM cost for the static path, smallest code change, matches the OTR aesthetic of "1940s radio drama with a host at a microphone."

**Option 2.** Two-pipeline split. HuMo for character dialogue. LTX-2.3 (already in the v2 architecture for 10-12 s clips, 257 frames @ 24fps, fp8_e4m3fn) animates the radio still for non-dialogue lines, attempting some audio-reactive motion. Or fall back to ffmpeg Ken-Burns pan over the radio still if LTX is too heavy. More code, more orchestration, more wall-clock. LTX output is not lip-sync-aligned to audio — generic motion model. Failure mode: looks like a screensaver overlaid on dialogue audio.

**Option 3.** Hybrid composite. FLUX renders an ANNOUNCER+radio composite (host sitting AT the radio set with microphone). HuMo gets that as `ref_image`, identity-locks to the host's face, radio appears as in-frame background. Smallest code change. Fragile — depends on prompt discipline to keep the radio in frame across renders. Routing premise (radio is the performer) stays nominally intact but is structurally a prompt trick, not an architectural correction.

---

## Round-robin convergence (already run)

A round-robin consult was run today — gpt-5.4, gemini-3-pro-preview, mistral-nemotron — independently. All three converged on **Option 1** with refinements. ChatGPT additionally framed this as "Option 4 — static-video first": speaking faces go to HuMo, everything else goes to a deterministic static-video editorial path. Same fix, cleaner architectural rule.

The consult also corrected two things in the original framing:
1. The `reference_latents` mechanism is generic Wan2.1 I2V — not a HuMo-specific quirk. The face-only behavior comes from the **weights**, not the node code.
2. ffmpeg `-c:a copy` is bit-exact and does not resample; an earlier "ffmpeg concat introduces a resample" worry was wrong.

The consult added two implementation specifics worth flagging here:
- **C7 mux pattern** (Gemini's catch): when mixing HuMo clips and static clips, do NOT concatenate audio+video chunks. Container repacketization can perturb audio bytes even with stream-copy. Pattern that preserves byte-identity: concat all clips video-only (`-an`) into a `silent_combined.mp4`, then a single final mux of the pristine master audio over the concat'd video using `-c:v copy -c:a copy`. This is also the structural fix for a sibling bug (BUG-LOCAL-128) where the current per-clip-mux pipeline truncates ~70 s of audio via `-shortest`.
- **Frame rate** (NVIDIA's catch): HuMo runs at 25fps (`HUMO_FPS = 25`); ffprobe of today's mp4 confirmed `r_frame_rate=25/1`. Lock everything to 25fps, not 24, to avoid concat-time frame-rate mismatches.

---

## My opinion

Commit Option 1. The reasons in order of importance:

1. **It accepts what HuMo is.** The model's weights are trained for face animation. Trying to keep the radio as the visual performer for non-dialogue lines means fighting the model. Option 2 doesn't fix that — it routes around HuMo to a different model that also doesn't truly lip-sync. Option 3 papers over it with prompt discipline. Only Option 1 collapses the problem to a single, honest rule: "speaking faces → HuMo, everything else → editorial static-video."
2. **It's the smallest code change.** ANNOUNCER becomes another cast entry. The existing FLUX portrait pipeline produces the still. The existing HuMo dispatch is unchanged for character lines. The new static-video helper is a few lines of `subprocess.run` around an ffmpeg invocation. No new model loaded, no new VRAM contention.
3. **It's the safest under C7.** Option 2 doubles the rendering pipelines, which doubles the surface where audio-byte-identity can break. Option 1 keeps audio path completely untouched and adds video as a deterministic companion stream.
4. **It matches the OTR genre identity better than the original premise did.** Old-time radio drama is a host-at-microphone medium. The radio set is a motif; the announcer is a person. Trying to make the tabletop radio "perform" every non-dialogue second was always going to feel like a trick.
5. **The static-video path solves a real cumulative-drift risk.** With strict frame rounding (`round(audio_duration * 25)` per clip) and 25fps lockstep, drift is bounded per-clip rather than accumulating across an episode.

I am not confident enough to bet the architecture without a second pair of eyes on two specific things — see below.

---

## Specific questions for the second-opinion reader

Please be opinionated and direct. If you disagree with the recommendation, explain where and why; "the consensus is correct" is also a useful answer if backed by reasoning.

1. **Is my read of `WanHuMoImageToVideo` correct?** Specifically: that `ref_image` is encoded into `reference_latents` as a soft conditioning attention hint, the actual generation latent starts from zeros, and the model's restriction to face animation is in the HuMo weights rather than the I2V node's code. If you believe HuMo CAN animate a non-face reference reliably enough to ship, please cite — paper, model card, GitHub thread, or reproducible test. We checked the stock node source at `comfy_extras/nodes_wan.py:1070-1108` and the OTR dispatch at `nodes/batch_humo_render.py:1195-1199, 1243, 1466-1478` and saw nothing that would change the conclusion, but a fresh look may.

2. **Is Option 1 the right architectural commit, or are we leaving value on the table?** In particular: is there a fourth path we missed? Anything that would let the radio still genuinely animate to audio (not just static or Ken-Burns), within 14.5 GB VRAM, no cloud, no Flash Attention, no quantization hacks, no weight streamers? We are explicitly NOT chasing those (they have failed before in this codebase). If there is a genuinely good local path we haven't considered, please describe it. If there isn't, please confirm that and say so.

3. **Is the C7 mux pattern (concat video-only with `-an`, single final stream-copy mux of pristine master audio) safe to commit as the fix for both BUG-LOCAL-129 and BUG-LOCAL-128?** Container-format edge cases worth flagging: HuMo clips are h264/AAC/mp4 at 1920x1080 25fps; static ffmpeg clips will be h264/no-audio/mp4 at the same spec; master mix is a 48 kHz mono WAV. Are there gotchas with `-c:v copy -c:a copy -map 0:v:0 -map 1:a:0` across this combination, container timestamps, B-frame ordering at concat boundaries, or AAC-bitstream-vs-WAV mux that we should pre-empt with a specific ffmpeg invocation, a remuxing pre-step, or an `-fflags +genpts`?

4. **For SFX specifically:** how should SFX concurrent with dialogue (door slam during a line of dialogue, ambient texture under speech) be handled visually? The plan currently is: stay on the speaking character's HuMo clip, do not cut to a separate visual, and mark the SFX line `is_concurrent_with_dialogue=True` so the static-video path skips it. Is there a better policy that preserves SFX as a distinct ledger event for downstream mixing while not introducing a visual cut?

5. **Sanity check on sequencing:** plan is to ship the speaker-role rewrite + ANNOUNCER cast schema + static-video helper first (BUG-LOCAL-129), then ship the C7 mux rewrite second (BUG-LOCAL-128) once the routing is verified clean. Reasonable, or does landing the mux rewrite first reduce risk? The mux rewrite is the larger blast-radius change but it also happens to fix the audio truncation regardless of routing.

Thanks. Please cite specific files / line numbers when possible.
