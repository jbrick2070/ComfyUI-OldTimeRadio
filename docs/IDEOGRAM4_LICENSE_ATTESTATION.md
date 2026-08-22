# Ideogram 4 -- licence attestation

Records the terms this build must honor for the locally-installed Ideogram 4
weights. Modelled on `docs/H3_LICENSE_ATTESTATION.md`, which is the established
shape for a restrictively-licensed model in this repo.

**This is NOT a grant.** Unlike H3 -- where MiniMax issued a named written
authorization to the operator's entity -- Ideogram published public weights under
a standing public licence and no individual grant exists or was sought. What is
recorded here is the PUBLIC agreement and the narrow reading this build adopts.

## 1. What was installed, and from where

Downloaded 2026-08-21 on the operator's explicit authorization ("ideo 4"), from
the ComfyUI repackage **`Comfy-Org/Ideogram-4`** (public). The upstream
first-party repo `ideogram-ai/ideogram-4-fp8` is **GATED** -- access is
restricted and was not requested, so the licence text below was read from the
ungated `ideogram-oss/ideogram4` mirror
(`model_licenses/LICENSE-IDEOGRAM-4-NON-COMMERCIAL`), not from behind the gate.

| file | bytes |
| :-- | --: |
| `diffusion_models/ideogram4_nvfp4_mixed.safetensors` | 5,490,550,037 |
| `diffusion_models/ideogram4_unconditional_nvfp4_mixed.safetensors` | 5,490,550,037 |
| `text_encoders/qwen3vl_8b_nvfp4.safetensors` | 6,305,221,764 |

Installed into `C:/ComfyUI-Models/` (the box's real model store, per
`extra_model_paths.yaml`). The `flux2-vae.safetensors` the graph also needs was
**already present** and was deliberately NOT re-fetched -- the repo would have
laid its own copy down on the identical path and overwritten the file
`flux2_klein` depends on. The two are 2,264 bytes apart and are not the same file.

## 2. The terms, quoted

* **Non-Commercial Purposes** -- *"activity or use that fits in any of the
  following categories: (i) use that does not directly or indirectly generate
  revenue and is not otherwise intended for or directed towards commercial
  advantage or monetary compensation, (ii) use by a for-profit entity solely for
  testing, evaluation, or research and development in a 'non-production
  environment'..."*
* **Outputs** -- *"We claim no rights in outputs you generate using the Model.
  You are responsible for outputs and their subsequent uses."*
* **Redistribution** -- permitted only *"on terms that are no less restrictive
  than those set forth in this Agreement"*, with a copy of the Agreement provided
  to recipients and an attribution notice retained in a `Notice` file.
* **Competing models** -- *"You may not use any Output to develop, train,
  fine-tune or distill a model or other product or services that is competitive
  with the Model or any of Company's other products or services."*

There is **no revenue threshold and no small-business exemption.**

## 3. Operating constraints this build honors

Deliberately the NARROW reading, so the build cannot exceed what the public
licence actually permits:

- Ideogram 4 inference runs only on the operator's own hardware, offline. No
  hosted service, no shared or public inference endpoint.
- The weights are never redistributed, republished, mirrored, or bundled into
  any release artifact of this project -- in any form, quantized included.
  **Open-sourcing the OTR code is unaffected: the code ships, the weights do
  not.**
- Outputs are used within the operator's own productions. Ideogram claims no
  rights in them.
- **Commercial scope: NOT granted.** Treat all use as non-commercial. If OTR
  ever generates revenue directly or indirectly, this lane requires a separate
  commercial licence from Ideogram and must be disabled until one exists.
- **This build relies on the "no revenue" prong, NOT the "non-production
  environment" carve-out** (r1 review, cursor lane -- a sharp reading and it is
  right). The licence's second prong covers for-profit use *"solely for testing,
  evaluation, or research and development in a non-production environment"*.
  OTR renders and publishes episodes; that is a production path by any honest
  reading, so the carve-out is not available and must not be leaned on. The
  permission this build stands on is prong (i): use that does not directly or
  indirectly generate revenue.
- No Ideogram output is used to train, fine-tune or distil any model. OTR trains
  nothing, so this is satisfied by construction.
- If any constraint above ceases to hold, the Ideogram lane is disabled until
  the position is re-established.

## 4. Why this is not a new class of risk for this repo

`commercial_clean = False` is an established, populated declaration here, and it
is already carried by the **shipping hero video engine**:

| engine | file:line | note |
| :-- | :-- | :-- |
| `eng_ltx25` | `nodes/_otr_video_engines/eng_ltx25.py:396` | **ships episodes today** |
| `eng_minimax_h3` | `nodes/_otr_video_engines/eng_minimax_h3.py:461` | attested separately |
| `flux_gen1` | `nodes/_otr_image_engines/flux_gen1.py:96` | BFL non-commercial, **registered** |
| `eng_musicgen` | `nodes/_otr_audio_engines/eng_musicgen.py:26` | CC-BY-NC-4.0 |

**Correction (r1 review, cursor lane, verified):** an earlier draft also listed
`sd35_large.py:127` here. It does declare `commercial_clean = False`, but the
class is **not registered** -- `nodes/_otr_image_engines/__init__.py` never
imports it, and its CAPABILITIES row was removed 2026-06-29. Citing it as a
shipping precedent was misleading, so it is dropped. The authoritative count is
**11 registered image engines with 11 matching CAPABILITIES rows**
(`cloud_flux_pro`, `cloud_krea_2_turbo`, `cloud_luma_photon_flash`,
`cloud_nano_banana_2`, `cloud_seedream_2`, `flux2_klein`, `flux_gen1`,
`google_image`, `ideo`, `lumina_image`, `z_image_turbo`).

**Note the name collision:** `ideo` in that list is the **paid cloud** Ideogram
arm. The local engine must use the distinct id `ideogram4`.

**CORRECTION, made after the r1 review caught it (Codex, grounded and confirmed
by the driver against the files).** An earlier draft of this section claimed
enforcement is *"declare-and-record, non-blocking"* across the stack. **That is
true for AUDIO and false for IMAGES**, and the distinction matters:

* **Audio:** `nodes/_otr_audio_cache.py` carries `commercial_clean` into the
  release-gate sidecar and `nodes/cast_lock.py:985` emits a *"non-blocking
  warning (I-8)"*. The gate's three-state rule is real: `True` ships silently,
  `False` warns and still renders, **missing/null fails closed stop-ship**.
* **Images:** `nodes/_otr_release_gate.py:3-6` scans *"roles, voice-bank
  entries, audio cache sidecars, and `audio_meta`"* -- **images are not in that
  list**, and the image ledger row built in
  `nodes/otr_image_gen_dispatcher.py:1606-1621` carries `engine_id`,
  `engine_version` and hashes but **no `commercial_clean` field at all**.

So on the image side the flag is an **adapter-level declaration only**; nothing
scans it. **This is a pre-existing gap, not one Ideogram creates** --
`flux_gen1` and `sd35_large` are already `commercial_clean = False` image
engines whose ledger rows carry no flag. But it means this attestation is
**documentation-only for the image lane**, and it must not be read as an
enforced control. Adding image provenance to the release scan is a separate,
explicitly-scoped change; note that doing so naively would trip the fail-closed
rule on every historical image row.

The Ideogram adapter still declares `commercial_clean = False`, exactly as its
image-engine peers do.

## 5. A SEPARATE issue that is NOT a licence matter, recorded here so it is not lost

Ideogram 4 ships a **built-in safety filter** that can refuse to return an
image ("Image blocked by safety filter"), and the official ComfyUI template
notes that plain-text prompts have a higher false-positive rate than the
structured JSON the model was trained on.

That is a **reliability and product-fit** question, not a licensing one, and it
sits against a hard operator directive (`CLAUDE.md`: no content guardrails on
generated episodes) on a pipeline that adapts Macbeth and King Lear. It is being
decided on its own merits in
`docs/2026-08-21-ideogram4-local-still-engine/driver_anchor.md` (question D8).
It is flagged here only so that a future reader does not mistake the licence
sign-off for a sign-off on that question.

## 6. Scope of this attestation

This records terms; it is not legal advice and expands nothing. Where the
Agreement's actual text conflicts with anything here, the Agreement governs.
Prepared 2026-08-21 from the public licence text.

**Operator sign-off: PENDING.** The download was authorized; the operating
constraints in section 3 are the driver's narrow reading and have not yet been
reviewed by the operator. Section 3's commercial line is the one that wants his
eye, because only he knows whether OTR is or will be revenue-generating.
