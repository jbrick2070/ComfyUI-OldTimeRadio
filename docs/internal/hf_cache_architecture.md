# HuggingFace Cache Architecture in ComfyUI

**Status: locked-in after 2026-04-19 reorg.** Reclaimed ~146 GB and consolidated all HF-library cache into one canonical path.

## The Reality of HuggingFace in ComfyUI

Hugging Face (the platform) is just a registry like GitHub, but the `huggingface_hub` Python library enforces a very specific, rigid cache layout: `blobs/` + `models--org--name/snapshots/HASH/` (tied together with symlinks).

Many ComfyUI custom nodes — especially for audio and vision models like Bark, MusicGen, Kokoro, Depth-Anything, or Gemma-style LLMs — rely entirely on this library. They simply call `snapshot_download()` and expect that exact structure to exist. Because they don't offer easy custom path overrides, trying to force them into ComfyUI's standard folder tree usually breaks them.

Therefore, creating a dedicated `models/huggingface/hub/` folder isn't an arbitrary choice; it is dictated by the cache format. It's the pragmatic equivalent of a `pip_cache` or `npm_cache`. It keeps everything that relies on the HF library's internal logic in one canonical place without breaking node graphs.

## Why This Setup Is Best Practice

The layout is clean, efficient, and production-ready:

- **Typed flat files** — native assets go in their dedicated ComfyUI folders (`checkpoints/`, `loras/`, `controlnet/`, `vae/`, `text_encoders/`, etc.)
- **Diffusers** — full Diffusers pipelines stay isolated in `models/diffusers/MODEL_NAME/`
- **The hub cache** — anything pulled via `huggingface_hub` lands in `models/huggingface/hub/`

Windows junctions at `~/.cache/huggingface/hub/` point back into the ComfyUI tree. Anything hardcoded to look in the default user-profile location still works perfectly, but the actual data lives where we want it.

## Could It Be More Elegant?

In a perfect world, yes — Bark in `models/tts/bark/`, MusicGen in `models/audio/musicgen/`, neatly aligned with ComfyUI's "typed folder" philosophy.

In practice, most HF-dependent nodes won't respect those paths. Moving files out of the `snapshots/HASH/` structure usually breaks the node because it loses references to `config.json`, tokenizer files, or model weights. Until node developers consistently support custom pathing, centralizing the HF cache is the realistic sweet spot for stability.

## Maintenance

1. **Keep `HF_HOME` locked.** Set to `C:\Users\jeffr\Documents\ComfyUI\models\huggingface` at User scope. Future `snapshot_download()` calls land there automatically.
2. **Scan the cache.** Run `huggingface-cli scan-cache` periodically to see what's eating disk. Old, unused snapshots can be purged safely.
3. **Watch the junctions.** Directory junctions (`mklink /J` or `New-Item -ItemType Junction`) are rock solid, but if the ComfyUI root folder ever moves to a new drive, recreate them against the new absolute path.
4. **Handle exceptions.** If a node supports custom paths (some Diffusers wrappers do), extract the `.safetensors` into typed folders and map via `extra_model_paths.yaml`.

## Current Junction Map

Created 2026-04-19. All links live at `C:\Users\jeffr\.cache\huggingface\hub\` and point into `C:\Users\jeffr\Documents\ComfyUI\models\huggingface\hub\`:

| Junction source (legacy HF default) | Target (canonical) |
|---|---|
| `models--suno--bark` | `models--suno--bark` |
| `models--facebook--musicgen-medium` | `models--facebook--musicgen-medium` |
| `models--hexgrad--Kokoro-82M` | `models--hexgrad--Kokoro-82M` |
| `models--depth-anything--Depth-Anything-V2-Large-hf` | `models--depth-anything--Depth-Anything-V2-Large-hf` |
| `models--google--gemma-4-E4B-it` | `models--google--gemma-4-E4B-it` |

`models--tencent--HunyuanWorld-Mirror` was left untouched — Jeffrey's HY-World project uses it and the canonical path wasn't determined.
