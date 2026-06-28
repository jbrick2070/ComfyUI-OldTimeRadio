# HuMo r2-HARDENED -- the decision-gate implementation (Codex + AntiGravity + Claude judge)

Both agents converged; Claude verified the ids/loader and caught a known-bad fix. This answers the
operator gate: "fit the 5/21 14B safely (<= ~13.5 GB real headroom) -> promote; else keep 1.7B."

## 0. The 5/21 quality target (locked)
14B fp8 + lightx2v 480p distill LoRA + ModelSamplingSD3 shift 8 @ 6 steps / cfg 1.0. The gate is FIT,
NOT settings -- do NOT bring no-LoRA-25-step into the promotion path (not the 5/21 look; costs runtime;
doesn't solve the real defect, which is umt5-TE co-residency).

## 1. Target engine id (VERIFIED -- there is no `humo_14B`)
Registered ids: `humo` (portrait 480x832), `humo_14B_169` (16:9), `humo_1.7B`, `humo_1.7B_169`
(eng_humo.py:80/479/533/557). Promotion target = **`humo_14B_169`** (matches the 16:9 other-beats path;
workflow node 92 `engine` + node 87; pick portrait `humo` only if portrait is intended). Promote =
flip `config/profiles/16gb_full.json` role+slot 1.7B->the chosen id AND patch the workflow node(s) in
the SAME change (CLAUDE.md S0) + re-validate.

## 2. THE FIT LEVER -- two-stage graph (NOT free_after_use)
REJECT `run_graph(free_after_use=True)`: eng_humo.py:344-348 documents it was tried and removed --
"inter-node model eviction only fragmented the allocator into an OOM." Instead, split
`HuMoEngine.render_clip` (eng_humo.py:343-349) into TWO graphs:
  (a) conditioning: clip/pos/neg/loadaudio/audioenc/loadimage/humo -> then
      `reclaim_idle_models(reason="humo post-conditioning")` (the PROVEN BUG-291 path already used
      post-decode at :361) to evict umt5 + whisper (~5 GB) BEFORE the sampler;
  (b) sampling/decode: unet/lora/modelsampling/ksampler/vaedecode, fed the stage-(a) conditioning +
      latents as literals.
This frees the 5 GB TE block during the heavy 14B forward without the failed inter-node eviction.

## 3. MEASURE the gate (HuMo has no peak probe today)
HuMo only does an instantaneous post-reclaim guard (assert_vram_within_ceiling, eng_humo.py:361-365).
Add `motion_common.VramPeakProbe` around the stage-(b) run_graph+encode (mirror eng_ltx_av.py:701),
return `vram_peak_mb`, thread it through canonicalize. PROMOTION GATE = measured render-window peak
<= 13500 MB (run the smoke with `OTR_VRAM_CEILING_MB=13500`, not the 14.5 default).

## 4. HONEST RISK (the gate may FAIL -> keep 1.7B)
[ASSUMPTION, both agents] the fp8 14B UNET weights alone are ~14 GB resident; even after evicting the
TE, the sampler phase (14B + VAE + latents) may still ride >13.5 GB. If the measured peak can't hold
<= 13.5 with headroom, KEEP 1.7B (reliability wins) and harden 1.7B instead. The bakeoff Phase B is the
measurement that decides.

## 5. Settings gotcha (agy, grounded)
The 16:9 path `HuMo17BLandscapeEngine` overrides cfg to 2.5 (eng_humo.py:538) vs portrait 1.7B cfg 1.0
(the de-blue). If promoting `humo_14B_169`, FORCE cfg 1.0 (distill) or verify the 2.5 doesn't
re-introduce the blue/colour cast. Portrait 1.7B de-blue cfg 1.0 is correct (eng_humo.py:515).

## 6. CUT from this build (both agents) -> research probe only
- GGUF HuMo UNET/TE: HuMo uses `UNETLoader`/`CLIPLoader` only (eng_humo.py:170,216); LTX's
  `UnetLoaderGGUF` is a DIFFERENT graph. A GGUF HuMo-17B (VeryAladeen/calcuis/Alissonerdx repos exist)
  would need HuMo-specific GGUF loader candidates + audio-cross-attn mapping + /object_info verify +
  tests. Keep it as a SEPARATE research probe -- it is the only lever that lowers the ~14 GB fp8 UNET
  weight itself, so revisit it IF the two-stage fix can't hit 13.5 GB.
- Newer talking-face model swap: separate dependency/license/wrapper validation project, not this build.

## Build order
1. Add VramPeakProbe to HuMo + thread vram_peak_mb (measurement first -- nothing to promote without it).
2. Bakeoff Phase A/B legs incl. the two-stage-graph variant of `humo_14B_169`; record measured peak.
3. If peak <= 13.5 with headroom: implement the two-stage split, flip profile+workflow, re-validate,
   suite+Bug Bible, leave clips for the operator eyeball BEFORE the profile flip lands.
4. Else: keep 1.7B; harden 1.7B settings; log the GGUF-17B probe as the future lever.
