# Comfy Build: otr-24gb-foley

The Comfy developer-platform build of `workflows/otr_24gb_foley.json`,
made with comfy-cli 1.24.0 (`comfy build`). `comfy-build.yaml` here is the
working copy: every later `comfy build` / `comfy deploy` command reads it.

| | |
|---|---|
| Build | `971bd986-c1ef-4b2f-a91f-96659b0755c6` |
| Release v1 | `7f0bcc94-9acd-49f2-a6e7-30dedf728e1c`, linux/nvidia, deployable (pack at `fa6f469e`) |
| Release v2 | `d908f8bb-fb8c-4fc6-bff4-34764137557d`, linux/nvidia, deployable (pack at `e3885726`, 2.3.49) |
| Release v3 | `33213b10-a779-442c-b199-a31fc9570801`, linux/nvidia, deployable (pack at `a31f954f`, 2.3.49 + fontless-host fonts) |
| Deployment | `dep-3b611093-3d7f-4047-87c5-00fc632663bf` "production", rtx-pro-6000-mig-2g-48gb, min 0 / max 1, on release v3 since 2026-10-31 |
| Cut | 2026-10-31, first cut green, about 4 minutes of build |
| Pack | ComfyUI-OldTimeRadio at main `a31f954f` (git pin, not the registry) |
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

## What the endpoint proved (2026-10-31, release v1)

Four episodes rendered end to end on the MIG 48 GB worker with Bark voices
(Kokoro refused for the missing `en_core_web_sm`): three one-acts at 73, 82
and 183 minutes and a five-character three-act at 132 minutes, all published.
The three one-acts came back `failed` because node 12 reported a file the
mux's asset cleanup deletes (PBUG-20261031-01); the three-act, run with
cleanup off, came back `succeeded`. Release v2 carries the fix.

Release v2, same day: a one-act with cleanup on came back `succeeded`
(job `d9e8a234`), so the node 12 fix holds on the endpoint.

## What release v3 fixed (2026-10-31)

The runtime image carries no font files at all. Every v1 and v2 episode
published with no captions and no title card (libass found nothing to draw
with and exited 0), and the credits fell back to Pillow's built-in face,
squashed with the third column off the frame. The pack now ships DejaVu Sans
and DejaVu Sans Mono in `nodes/fonts` and uses them only when the machine has
no Latin face of its own. Job `af276e53` on release v3 (one-act, 88 s,
cleanup on) came back `succeeded`, with burned captions, the hero title and
the credits drawn in DejaVu Sans Mono.

## Languages on the endpoint (2026-11-01, release v3)

Job `eaef5ad2`, a Hindi one-act with both voice engines on Kokoro, came back
`succeeded` (211 s, 2.5 hours on the worker): Kokoro voiced every line
(hf_alpha, hm_psi, hm_omega) through the espeak phonemizer, which needs no
spaCy model, and the shipped Devanagari face drew the title, the caption
speaker names and the credits headings. So Kokoro works on the endpoint for
Spanish, French, Italian, Portuguese and Hindi; only English needs
`en_core_web_sm`. Mandarin and Japanese cannot voice here: their phonemizer
libraries carry a Python 3.13 marker and the image is 3.12.

Seen in that episode, not cloud-specific: the credits draw Devanagari
without shaping (the i-matra lands after its consonant) because Pillow has
no raqm on this box or the endpoint, and some dialogue lines came back from
the writer in romanized Hindi.

## Next

* Kokoro on the endpoint needs `en_core_web_sm` in the image.
* `comfy deploy stop planning/comfy-build/otr-24gb-foley` when idle time
  should not hold compute; storage still bills while stopped.
