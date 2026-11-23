# Qwen-Image 2.1 -- licence attestation

Records the terms this build honors for the `qwen_image_21` still engine. Modelled
on [FLUX3_KLEIN_LICENSE_ATTESTATION.md](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/FLUX3_KLEIN_LICENSE_ATTESTATION.md). **This is NOT a grant**: Alibaba's
Qwen team published public weights under a standing public licence and no individual
grant exists or was sought. What is recorded is the PUBLIC agreement and the narrow
reading this build adopts. Not legal advice.

## 1. What is fetched, and from where

Qwen's own repository is `Qwen/Qwen-Image-2.1` (Hub API `gated: false`, licence
metadata `qwen-research`, checked 2026-11-22). The pack fetches from Comfy-Org's
public repackage **`Comfy-Org/Qwen-Image-2.1`** (Hub API `gated: false`), at the
exact pinned revision `aacf31a6d48e707f352ecba714cee0680da3479b`, with no token:

| file | bytes | SHA-256 |
| :-- | --: | :-- |
| `diffusion_models/qwen_image_2.1_int8_convrot.safetensors` | 7,256,783,064 | `cb74113cb03faecd79611b01fd7fd642f0aa60d6f0b95086abee214d75eaa57d` |
| `text_encoders/qwen3vl_8b_int8_convrot.safetensors` | 9,350,798,360 | `8bfd0f6e12abf2d2d697ecc888e5e90b0d6741d6708f05799f53afa560452e8f` |
| `vae/qwen_image_2.1_vae_bf16.safetensors` | 675,509,688 | `bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9` |

All three hashes match what the Hub publishes for them and were recomputed from the
downloaded bytes on 2026-11-22. The pack's fetch refuses any other bytes
(`_pin_metadata`). No GGUF is used anywhere. The repackage carries no `LICENSE`
file of its own (tree read 2026-11-22); its README links Qwen's.

## 2. The terms (Qwen RESEARCH LICENSE AGREEMENT, release date 2026-09-20), as read

Read in full from `Qwen/Qwen-Image-2.1`'s `LICENSE` on 2026-11-22.

* **Grant (s2.a):** a limited, non-transferable, royalty-free licence to use,
  reproduce, distribute, copy and modify the Materials **for non-commercial
  purposes only**.
* **"Non-Commercial" (s1.i)** means "for research or evaluation purposes only".
* **Commercial use (s2.b)** needs a separate licence from Qwen (the agreement names
  `model-business@notice.qwencloud.com`).
* **Redistribution (s3):** allowed with a copy of the agreement, a notice on
  modified files, and the attribution notice in a Notice text file. The pack does
  not redistribute weights; the user's machine fetches them.
* **Rules of use (s4):** follow export controls; if the Materials or their
  outputs are used to create, train, fine-tune or improve an AI model that is
  distributed, display "Built with Qwen" or "Improved using Qwen"; do not use
  "Qwen" as the primary name of a derivative product.
* **s5.c:** suing Qwen over the Materials or their outputs ends the licence.
* **s6.d:** the user defends and indemnifies Qwen against third-party claims
  arising from their use.
* **s7:** Qwen may terminate on breach, and the user must then delete the
  Materials. **s8:** Chinese law, Hangzhou courts.
* The agreement says nothing separate about who owns generated **outputs**, and
  unlike the FLUX Kommunity License it has no clause that lets a small business
  use outputs commercially.
* The text encoder's own licence was NOT read. Treat the whole fetched set as under
  this research licence.

## 3. The reading this build adopts

* `commercial_clean = False` -- the same shelf as `flux_gen1` and `flux3_klein`.
  Opt-in only (`default_roles = ()`); never a default, and no shipped workflow
  selects it.
* Using this engine for episodes you publish commercially needs a separate licence
  from Qwen. Whether a free, public, unmonetized publication counts as "research or
  evaluation" is NOT decided here; that is the publisher's reading of s1.i. The
  pack does not check.
* The pack's rights receipt is advisory (standing ruling: always publish to
  otr/obs).
* The pack does not use outputs to train or fine-tune any model, so s4.b does not
  apply to it.
* Image-edit / multi-reference mode and the model's native transparent output are
  NOT wired: this engine is text-to-image only and returns plain RGB.

## 4. Re-check triggers

Re-read the licence and re-verify the hashes if Qwen updates its `LICENSE`,
Comfy-Org's repackage moves off the pinned revision, or the licence name changes
from `qwen-research`.
