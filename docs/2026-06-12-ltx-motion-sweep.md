# LTX motion sweep -- bare-bones smoke (2026-06-12)

Operator directive: stop chasing LTX motion through 24-min episodes; isolate it.
Harness: `scripts/otr_ltx_motion_smoke.py` (one still -> one short webm via the
REAL engine i2v topology: CheckpointLoaderSimple v0.9 + CLIPLoader t5xxl ->
CLIPTextEncode x2 + LoadImage + EmptyLTXVLatentVideo -> LTXVImgToVideoConditionOnly
-> LTXVConditioning -> KSampler -> VAEDecode -> SaveWEBM). ~30-60s/render at
768x448. Motion scored by `scripts/otr_ltx_mad.py` (mean inter-frame MAD; the 6/5
static-motion metric). Fixed still + announcer motion prompt (179 chars), strength
1.0, cfg 3.0, 30 steps, seed 42. Clips in `docs/ltx_motion_clips/`.

## Results

| sampler        | length | MAD_mean | MAD_p90 | motion       |
|----------------|--------|----------|---------|--------------|
| euler (TODAY)  | 97     | 0.59     | 0.72    | none/freeze  |
| euler_cfg_pp   | 97     | 0.88     | 1.43    | pan          |
| euler_cfg_pp   | 169 (TODAY's len) | 1.17 | 2.33 | pan        |
| **euler_cfg_pp** | **257** | **4.21** | **7.15** | **REAL**   |
| **euler_cfg_pp** | **305** | **5.30** | **6.86** | **REAL**   |

(MAD verdict bands: <0.6 freeze, 0.6-2.0 pan, >2.0 REAL.)

## Conclusion -- the 6/1 recipe is reproduced

Two drifts, BOTH needed:
1. **Sampler:** `euler` (today, hardcoded) -> **`euler_cfg_pp`** (CFG++; the 6/1 +
   6/5 ledgers). Alone it lifts MAD ~0.59 -> ~0.88 (still a pan).
2. **Length:** today's **169 STARVES** the motion (MAD 1.17, pan). **257-305**
   unlocks REAL dynamic motion (MAD 4.2-5.3). The model needs the longer window
   to develop motion from the i2v-anchored first frame.

i2v ConditionOnly stays ON at strength 1.0 throughout (the still IS the look);
strength was NOT touched -- euler_cfg_pp + length already brought motion back.

## Recommended LTX default (to lock + prove in one episode)

- `OTR_LTX_SAMPLER_NAME = euler_cfg_pp`  (sampler; @23aca22 made it swappable)
- LTX i2v length default **257** (from 169).

### OPEN CAVEAT before locking the length default
The smoke ran at **768x448** (decodes cleanly at 257/305). The EPISODE renders at
**1472x832**, where the installed wrapper's VAEDecode has a documented tiled-band
constraint (169 + 233 decode clean; 121/137 raised the 256-vs-128 tensor mismatch
-- BUG-LOCAL note in eng_ltx_video). **257 at 1472x832 must be decode-validated**
before raising the episode length default; 233 may be the safe ceiling there.
euler_cfg_pp is safe to lock at any resolution.
