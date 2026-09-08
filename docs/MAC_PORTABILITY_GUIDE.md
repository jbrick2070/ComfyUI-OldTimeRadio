# Running ComfyUI-OldTimeRadio on a Mac (Apple Silicon)

**Status: WORKING.** The shipped canonical renders end to end on Apple Silicon
and publishes to `otr/obs/` (first proven 2026-09-07 on a Mac mini M4, 16 GB,
macOS 26.6.2, ComfyUI Desktop 0.34.6, Python 3.13, torch 2.12.1).

This is the practical guide. The measurements behind it are in
`MAC_LESSONS_LEARNED.md`; the engine-by-engine survey is in
`MAC_ENGINE_TEST_PLAN.md`.

---

## 1. Press Run -- what works with no extra steps

Install the pack, load `otr_canonical` from
**Workflow > Browse Templates > EXTENSIONS > comfyui-old-time-radio**, press Run.

The shipped graph is deliberately the **only fully-local configuration Apple
Silicon has**: writer on Metal, Kokoro voices, Stable Audio 3 music, and three
visualizer video lanes that mint no still, so **no image weights and no API key
are needed**.

| stage | engine | notes |
| --- | --- | --- |
| writer | `Qwen/Qwen3.5-4B` on `mps` | ~6.5 tok/s. **Keep `llm_quant_policy` = `none`** |
| voices | `kokoro` | runs through kokoro-onnx on CPU on Python 3.13, by design -- that is what makes it auto-install |
| music | `stable_audio_3` | ungated, commercially clean, self-fetching |
| video | `viz_mxc_cpu`, `viz_green`, `viz_camera` | zero dependencies, no stills |

### Two settings that will ruin your day if you change them

* **`llm_quant_policy` must stay `none`.** NF4 *works* on Metal but runs at
  0.3-0.5 tok/s against 14.47 on CUDA -- there is no Metal kernel, so
  bitsandbytes falls back to a Python dequantise loop. It is slow enough that
  the 40 s `NewsCurationDeep` budget expires and the run dies with a confusing
  `_LLMTimeoutWorkflowPause`. It does not look like "quantisation is slow".
* **Do not select `viz_mxc_mandala`.** It needs `pycairo`, which has no macOS
  wheel (see section 3).

### Memory: 16 GB is the floor, and it is tight

The writer alone peaks near 14 GB inside ComfyUI on a 16 GB machine. It works,
but close other applications first -- a remote-desktop session competing for RAM
is enough to trigger an OOM kill. Note `ps` under-reports badly on Apple Silicon
(0.33 GB for a 14 GB process); use `footprint -p <pid>` and read
`phys_footprint`.

---

## 2. What CANNOT work locally on Apple Silicon, and why

**CORRECTED 2026-09-07 (this entry previously said Apple Silicon has no local
image engine -- that was wrong).** `z_image_turbo` LOADS AND EXECUTES ON METAL.
It is blocked by MEMORY, not by the device:

```
Requested to load ZImageTEModel_   loaded completely;  7672.25 MB   full load: True
Requested to load Lumina2          loaded completely; 11739.54 MB   full load: True
...
RuntimeError: MPS backend out of memory (MPS allocated: 13.01 GiB,
  other allocations: 5.96 GiB, max allowed: 20.13 GiB). Tried to allocate 1.46 GiB
```

Both models load fully; it dies in the KSampler needing about 20.4 GiB against a
20.13 GiB ceiling -- it misses by roughly 300 MB. The checkpoint alone is
12.3 GB.

**So the honest statement is about RAM, not about Metal: no local image engine
fits in 16 GB. A 32 GB Mac would very likely have one**, and that is a hardware
question the operator can act on rather than a portability defect anyone can
patch. The `["cuda"]` declaration is wrong in KIND -- the engine is not
CUDA-only, it is large.

The same applies to image-to-video: `ltx_8gb` passed its engine gate on mps and
then failed at the still its own lane needed, for exactly this reason.

**The rest of the picture, unchanged:** everything below is still declared
`["cuda"]` and untested on Metal:

* **Image** -- `z_image_turbo`, `flux_gen1`, `flux2_klein`, `lumina_image`,
  `ideogram4_local` are all `["cuda"]`. Only `cloud_*` / `google_image` / `ideo`
  list `mps`, and those are paid APIs.
* **Video diffusion** -- every LTX, `wan_ti2v`, `fastwan_8gb`, `humo`,
  `mesh_stage`, both `animatediff15_v3_*` and both `minimax_*` are `["cuda"]`.

**The knock-on that surprises people:** the four `still_*` video lanes DO declare
`mps` and would run -- but they *consume* a still, and nothing local can mint one
here. Same for image-to-video engines. So selecting them on a Mac fails for a
reason one step removed from the engine you picked.

**Consequence:** the visualizer lanes are the only fully-local video path Apple
Silicon has. The shipped canonical is not a conservative choice, it is the only
one that runs.

---

## 3. Engines that need manual steps -- and exactly what to do

### `viz_mxc_mandala` -- Windows-only in practice

`pycairo` publishes Windows wheels, an sdist, and **no macOS wheel**. On a stock
Mac `pip install pycairo` falls back to the sdist and fails to build, because
libcairo headers are not present. Measured 2026-09-07.

```bash
brew install cairo pkg-config
pip install pycairo
```

Without Homebrew this engine cannot run. **Use `viz_mxc_cpu`, `viz_green` or
`viz_camera` instead** -- zero dependencies, no cairo. The engine now says this
itself when you select it on a Mac.

Linux has the same problem for a different reason (no Linux wheels):
`apt install libcairo2-dev pkg-config` first.

### `animatediff15_v3_*` -- weights have NO auto-fetcher

The two files are named in the CUDA profiles' `model_requirements` but nothing
downloads them. On any platform you must place them by hand:

| file | source repo | destination |
| --- | --- | --- |
| `v1-5-pruned-emaonly-fp16.safetensors` (1.99 GB) | `Comfy-Org/stable-diffusion-v1-5-archive` | `models/checkpoints/` |
| `v3_sd15_mm.ckpt` (1.56 GB) | `guoyww/animatediff` | `models/animatediff_models/` |

```bash
python -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('Comfy-Org/stable-diffusion-v1-5-archive','v1-5-pruned-emaonly-fp16.safetensors'))
print(hf_hub_download('guoyww/animatediff','v3_sd15_mm.ckpt'))"
```

then copy each to the destination above. **Mac status: UNVERIFIED** -- the
engine declares `["cuda"]` and contains no NVIDIA-specific code, so the
declaration may simply be untested. Being tested; see `MAC_ENGINE_TEST_PLAN.md`.

### Cloud / API engines

`elevenlabs`, `google_tts`, `google_lyria`, `sonilo`, every `cloud_*`,
`google_image`, `ideo` all already declare `mps`. They need **credentials, not a
port** -- testing them measures an API key rather than Apple Silicon. A cloud
image key is the single highest-leverage addition for a Mac, because it unlocks
the `still_*` video lanes that are otherwise unreachable.

---

## 4. If something breaks, read this first

| symptom | cause | fix |
| --- | --- | --- |
| **ComfyUI will not start at all**, exit 1, `ImportError: tokenizers>=0.23.1 ... found 0.22.2` | the shipped `tokenizers` pin in versions `.24`-`.28` | `pip install 'tokenizers>=0.23.1,<0.24'`, and use `2.0.0-alpha.29` or later. **Do NOT reinstall from Manager** -- it serves the broken version, and a bricked boot cannot be repaired from the UI because Manager IS a ComfyUI extension |
| music is noise, "like a broken cassette" | ComfyUI's sub-quadratic attention is wrong on MPS | fixed automatically by the pack's prestartup. If you disabled it, pass `--use-pytorch-cross-attention` |
| render dies at the mp4 encode, `winget install ffmpeg` | ffmpeg/ffprobe were not declared dependencies | fixed; `pip install -r requirements.txt` brings both |
| `NewsCurationDeep exceeded 40s` | `llm_quant_policy` is not `none` | set it to `none` (section 1) |
| OOM / the machine freezes mid-render | 16 GB shared with macOS and everything else | close other apps; the writer peaks near 14 GB |
| a video/image engine refuses with `EngineUnusable` | it declares `["cuda"]` | expected on Mac -- see section 2 |

**Error messages that lie on a Mac** (cosmetic, being cleaned up):
`[StoryOrchestrator] CUDA warmup complete` and
`orphan worker still on GPU ... racing the orphan's CUDA kernels` both print on
machines with no CUDA. Neither indicates a CUDA code path.
