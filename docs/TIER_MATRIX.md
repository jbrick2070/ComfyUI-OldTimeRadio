# The shipping set

Read from the generated graphs in `workflows/variants/`. Kokoro voices,
MusicGen and upscaler off throughout, inherited from the canonical.

## 8 GB NVIDIA

| tier | JSON | writer | quant | video lane | image | acts | chars |
|---|---|---|---|---|---|---|---|
| **low** | `otr_8gb_low` | Qwen3.5-4B | bnb_nf4 | viz_camera | none (dormant) | 1 | 2 |
| **still** | `otr_8gb_still` | Qwen3.5-4B | bnb_nf4 | still_motion | sd15 | 1 | 2 |
| **video** | `otr_8gb_video` | Qwen3.5-4B | bnb_nf4 | ltx098_low_video | sd15 | 1 | 2 |
| **foley** | `otr_8gb_foley` | Qwen3.5-4B | bnb_nf4 | ltx25_high_foley_plus | sd15 | 3 | 3 |
| **mime** | `otr_8gb_mime` | Qwen3.5-4B | bnb_nf4 | ltx25_high_mime | sd15 | 3 | 3 |
| **animatediff** | `otr_8gb_animatediff` | Qwen3.5-4B | bnb_nf4 | animatediff15_v3_haunted_video | sd15 | 1 | 2 |

## 16 GB NVIDIA

| tier | JSON | writer | quant | video lane | image | acts | chars |
|---|---|---|---|---|---|---|---|
| **low** | `otr_16gb_low` | gemma-4-12b-it | bnb_nf4 | viz_camera | none (dormant) | 1 | 2 |
| **still** | `otr_16gb_still` | gemma-4-12b-it | bnb_nf4 | still_motion | z_image_turbo | 1 | 2 |
| **video** | `otr_16gb_video` | gemma-4-12b-it | bnb_nf4 | ltx25_high_video | z_image_turbo | 1 | 2 |
| **foley** | `otr_16gb_foley` | gemma-4-12b-it | bnb_nf4 | ltx25_high_foley_plus | z_image_turbo | 3 | 3 |
| **mime** | `otr_16gb_mime` | gemma-4-12b-it | bnb_nf4 | ltx25_high_mime | z_image_turbo | 3 | 3 |
| **animatediff** | `otr_16gb_animatediff` | gemma-4-12b-it | bnb_nf4 | animatediff15_v3_haunted_video | z_image_turbo | 1 | 2 |

## Apple Silicon

| tier | JSON | writer | quant | video lane | image | acts | chars |
|---|---|---|---|---|---|---|---|
| **low** | `otr_mac_low` | Qwen3.5-4B | none | viz_camera | none (dormant) | 1 | 2 |
| **still** | `otr_mac_still` | Qwen3.5-4B | none | still_motion | sd15 | 1 | 2 |
| **video** | `otr_mac_video` | Qwen3.5-4B | none | ltx098_low_video | sd15 | 1 | 2 |
| foley | _not built_ | | | | | | |
| mime | _not built_ | | | | | | |
| **animatediff** | `otr_mac_animatediff` | Qwen3.5-4B | none | animatediff15_lightning_video | sd15 | 1 | 2 |

## AMD ROCm (experimental -- no receipts)

| tier | JSON | writer | quant | video lane | image | acts | chars |
|---|---|---|---|---|---|---|---|
| low | _not built_ | | | | | | |
| **still** | `otr_amd_still` | Qwen3.5-4B | none | still_motion | z_image_turbo | 1 | 2 |
| video | _not built_ | | | | | | |
| foley | _not built_ | | | | | | |
| mime | _not built_ | | | | | | |
| animatediff | _not built_ | | | | | | |

## CPU only

| tier | JSON | writer | quant | video lane | image | acts | chars |
|---|---|---|---|---|---|---|---|
| **low** | `otr_cpu_low` | Qwen3.5-4B | none | viz_camera | none (dormant) | 1 | 2 |
| still | _not built_ | | | | | | |
| video | _not built_ | | | | | | |
| foley | _not built_ | | | | | | |
| mime | _not built_ | | | | | | |
| animatediff | _not built_ | | | | | | |

