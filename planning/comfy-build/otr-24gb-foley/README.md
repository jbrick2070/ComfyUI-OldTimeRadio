# Comfy Build: otr-24gb-foley

The Comfy developer-platform build of `workflows/otr_24gb_foley.json`,
made with comfy-cli 1.24.0 (`comfy build`). `comfy-build.yaml` here is the
working copy: every later `comfy build` / `comfy deploy` command reads it.

| | |
|---|---|
| Build | `971bd986-c1ef-4b2f-a91f-96659b0755c6` |
| Release v1 | `7f0bcc94-9acd-49f2-a6e7-30dedf728e1c`, linux/nvidia, deployable |
| Cut | 2026-10-31, first cut green, about 4 minutes of build |
| Pack | ComfyUI-OldTimeRadio at main `fa6f469e` (git pin, not the registry) |
| ComfyUI | v0.43.2, base image cuda130-py312 (torch 2.12.1) |
| pip pins | none; the pack's requirements.txt resolves on its own |
| Models | 11 weights, 61.5 GB, fetched by the builder from Hugging Face by pinned revision and sha256 |

## How the spec was made

`comfy build init --from-workflow` returns an empty definition for this pack:
the registry cannot attribute any `OTR_*` class, and its "maybe" guesses name
unrelated packs. The spec is hand-authored instead. The model rows come from
`scripts/otr_fetch_lane_weights.py` (lanes `ltx25_24gb`, `flux3_klein`,
`stable_audio_3`) plus the 12B writer pin in `nodes/_otr_visual_assets.py`;
Stable Audio 3 carries no pin in the fetcher, so its sha256 is the one
`comfy build refs resolve` returned.

## What the build proved, and what it did not

* Proved: the pack installs on Linux (kokoro 0.9.4, spacy 3.8.18,
  nvidia-vfx 0.2.0.0 all installed), imports in 0.6 s, and ComfyUI boots.
* Not proved: an episode. The builder ran no node tests for this pack
  ("validated by boot and import only"), and the OTR node count is not in
  the manifest.
* Open risk: spaCy's `en_core_web_sm` is not installed, and the torch
  Kokoro refuses by name without it. If the voices refuse on the endpoint,
  submit the workflow with both `OTR_CastLock` engine widgets set to `bark`;
  Bark fetches `suno/bark` through the HF cache on a cold worker.
* Not carried in the image: Kokoro voices and Bark weights live in the HF
  cache, outside `models/`, so a cold worker downloads them.

## Next

Deploying is `comfy deploy up planning/comfy-build/otr-24gb-foley --gpu <class>
--region <region> --min 0 --max 1`, which bills continuously. Read
`comfy deploy refs compute` first and get the operator's yes.
