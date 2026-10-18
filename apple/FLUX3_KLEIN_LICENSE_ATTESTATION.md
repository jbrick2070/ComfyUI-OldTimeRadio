# FLUX 3 Image Klein -- licence attestation

Records the terms this build honors for the `flux3_klein` still engine. Modelled
on `IDEOGRAM4_LICENSE_ATTESTATION.md`. **This is NOT a grant**: Black Forest Labs
published public weights under a standing public licence and no individual grant
exists or was sought. What is recorded is the PUBLIC agreement and the narrow
reading this build adopts. Not legal advice.

## 1. What is fetched, and from where

Black Forest Labs' own repositories (`black-forest-labs/flux-3-image-klein`,
`-base`, `-comfy`) are GATED: a Hugging Face login and a click-through are
required, and the pack is tokenless by design. The pack therefore fetches, at the
exact pinned revision `3ba0f7064cbb1666a71e2820d3cc5c2273bb6cd0`, from the
ungated third-party mirror **`ml-resources/flux-3-image-klein-comfy`** (Hub API
`gated: false`, anonymous 302, checked 2026-10-18):

| file | bytes | SHA-256 |
| :-- | --: | :-- |
| `diffusion_models/flux3-image-klein-int8-convrot.safetensors` | 5,827,874,779 | `847d8933548afd93df4af2cac2498e70891bf28d9dae06d695f536992e168fe2` |
| `text_encoders/qwen3vl_4b_int8_convrot.safetensors` | 4,858,559,240 | `32de113cbf8282b02cbff0794b637b9d6676da09314a1db5a30bcea9bf4ee01c` |
| `vae/flux2-vae.safetensors` (the Ideogram 4 row) | 336,211,292 | `868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3` |

**The mirror is a straight copy of BFL's `-comfy` repo.** BFL hides LFS SHA-256s
on its gated repo, but the git blob id of an LFS file is the hash of its pointer
text (which embeds the SHA-256 and size). For every file under
`diffusion_models/`, `text_encoders/` and `vae/` (12 files), plus `LICENSE.md`
(blob `044cf485...`, 19,593 B) and `NOTICE.md`, the mirror's blob id and size
EQUAL BFL's. Compared 2026-10-18 through the public tree API. Unofficial, but
byte-identical; the pack's fetch refuses any other bytes (`_pin_metadata`).

No GGUF is used anywhere.

## 2. The terms (FLUX Kommunity License v1.0), as read

Read in full from the mirror's `LICENSE.md` (same blob as BFL's).

* **Non-Commercial Purposes** are free: personal use, research, hobby, and
  testing/evaluation by a business in a non-production environment.
* **Qualifying User** (s1.e, s2.a): you "together with your affiliates' gross
  annualized revenue is less than US$5,000,000". A Qualifying User may use
  **Outputs** commercially and run the model in production solely to generate
  them.
* **s2.e conditions for that commercial use:** run content filters OR review
  Outputs for unlawful/infringing content before distribution; include AI
  disclosures where the law requires.
* **Outputs are not Derivatives**; BFL claims no ownership of them (s1.a, s2.d).
* **Restrictions (s4):** no military, surveillance or biometric-processing use; no
  infringement of publicity / "digital replica" rights; no using the model, its
  Outputs or results to improve another AI model; no commercial fine-tune or
  distillation without a BFL licence (https://bfl.ai/licensing).
* **Redistribution (s3):** allowed with the licence text and the attribution
  notice. The licence is personal and non-sublicensable: each user is bound
  directly. The pack does not redistribute weights; the user's machine fetches
  them.
* The Qwen3-VL-4B text encoder and BFL's prompt upsampler are Apache-2.0.

## 3. The reading this build adopts

* `commercial_clean = False` -- the same shelf as `flux_gen1`. Opt-in only
  (`default_roles = ()`); never a default.
* A publisher over the US$5M line, or any commercial fine-tune, needs BFL's
  licence. The pack does not check revenue.
* The pack's rights receipt is advisory (standing ruling: always publish to
  otr/obs). Whether it satisfies s2.e's "content filter or output review" for a
  commercial user is that user's responsibility.
* Image-edit / multi-reference mode (`ReferenceMultiLatent`) is NOT wired.
  Face-identity across beats is unmeasured; see the adapter docstring.

## 4. Re-check triggers

Re-read the licence and re-run the blob-id comparison if BFL updates the
`-comfy` repo, the mirror's revision changes, or the licence version moves off
v1.0.
