# fastwan_8gb -- asset manifest and licence chain

Durable identity for every file `fastwan_8gb` loads. Written 2026-08-01, verified
on this box (Windows, RTX 5080 Laptop 16 GB). Hashes are of the files that
actually rendered the four-arm bench's arm C.

**Scope note, so absence is not read as clearance:** the
`docs/model-license-audit-targets.txt` framework currently covers **LLM repos
only** (Mistral / Gemma / Qwen). No video model -- not the Wan base, not the umt5
encoder, not this LoRA -- has a row in it. This file is the video-side record; it
is NOT a pass through that gate.

---

## Assets

| role | file | bytes | store |
|---|---|---:|---|
| UNET (base) | `Wan2.2-TI2V-5B-Q5_K_M.gguf` | 3,810,603,360 | `C:/ComfyUI-Models/unet/` |
| LoRA (the distillation) | `Wan2_2_5B_FastWanFullAttn_lora_rank_128_bf16.safetensors` | 660,874,456 | `C:/ComfyUI-Models/loras/` |
| Text encoder | `umt5-xxl-encoder-Q5_K_M.gguf` | 4,145,878,880 | `C:/ComfyUI-Models/text_encoders/` |
| VAE | `wan2.2_vae.safetensors` | 1,409,400,960 | `C:/ComfyUI-Models/vae/` |

### LoRA digest

    sha256  79290493711b022e1c6e655d803715cd8a91a75cdb139856cad46f354e2f681c
    size    660874456 bytes
    source  Kijai/WanVideo_comfy

The size matches the figure recorded in
`docs/2026-07-31-arm-c-fastwan-BUILD-SPEC.md` exactly, so the file hashed here is
the file that arm C measured.

**The base UNET is BIT-IDENTICAL to `wan_ti2v`'s.** That is not a coincidence to
note in passing -- it is why the two engines measure the same VRAM peak, and it is
the reason FastWan is a throughput tier rather than a different model.

## Recipe identity

    recipe_id   fastwan22_ti2v_5b_dmd3_i2v_v1
    steps       3          (asserted == len(sigmas) - 1 at import)
    cfg         1.0
    shift       5.0
    sigmas      "1.0, 0.757, 0.522, 0.0"      <- STRING; ManualSigmas' wire format
    sampler     dmd_restart                    <- OTR_DMDRestartSamplerSelect
    scheduler   manual_sigmas
    lora_strength 1.0

The sigma list is FastVideo's `denoising_step_list` divided by 1000, pinned from
the **code path** (`DmdDenoisingStage`) -- never a model card, never community
usage advice.

**Still owed (r4):** the exact FastVideo upstream revision. `recipe_receipt`
returns the frozen string plus departure suffixes and nothing else, so a future
FastVideo change to `denoising_step_list` would be undetectable in a shipped
ledger. The sigma digest is pinned here; the upstream commit is not, because it
was not recorded when the recipe was lifted. Record it before treating any
published clip's receipt as reproducible provenance.

## Licence chain -- RECORDED WITH ITS GAP, not summarized as "verified"

| link | declared licence | verified at |
|---|---|---|
| FastVideo (the method + `denoising_step_list`) | apache-2.0 | upstream repo |
| Wan 2.2 TI2V-5B (base weights) | apache-2.0 | upstream repo |
| **`Kijai/WanVideo_comfy` (the extraction actually loaded)** | **no repo-level licence file** | **GAP** |

Both upstreams declare apache-2.0. The artifact this engine actually loads is a
third-party extraction from a repo that carries **no repo-level licence file**
(`docs/2026-07-31-arm-c-fastwan-BUILD-SPEC.md` s6A).

"Both upstreams say apache-2.0" and "this artifact is notice-compliant" are
different claims, and a `commercial_clean` flag is read as the second. So:

    FastWan8gbEngine.commercial_clean = False

That is deliberate and is NOT inherited from `wan_ti2v` (which declares `True`
because its GGUF and VAE are Apache-2.0 at the source). Flipping it is a one-line
change once the notice chain is resolved -- either a licence file appears upstream,
or the extraction is reproduced locally from the apache-2.0 sources and that
provenance is recorded here.

Until then `fastwan_8gb` renders and ships, but its output must not be described
as commercially clear.

## Preflight

`registry.CAPABILITIES["fastwan_8gb"].model_requirements` is
`["wan2.2-ti2v-5b", "fastwan-2.2-5b-lora"]`, and `_aux_loader_files()` adds the
LoRA row so `assert_usable` **fails closed by name** when it is absent.

That row is load-bearing: without the LoRA the graph still builds and still runs,
rendering 3 steps through the UN-distilled base model. There is no error and no
log line -- just ruined output wearing a FastWan receipt. The preflight row is the
only thing standing between a missing file and a false receipt.
