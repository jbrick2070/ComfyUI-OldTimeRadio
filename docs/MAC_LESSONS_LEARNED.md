# Mac (Apple Silicon) -- lessons learned

**READ THE COUNT IN CONTEXT: 2026-09-07 WAS DAY ONE.** Operator: *"today was the
first day we tried a Mac render."* Six defects in a single session is not decay,
neglect, or a pack that "used to work on Mac" -- **this path had never been
executed on Apple Silicon by anyone**. Every one of these is a first-contact
finding, which is exactly what a first run is for, and four of the six turned out
not to be Mac problems at all. A platform nobody has run is not a platform that
is broken; it is a platform that is unmeasured. It is measured now.

**WHAT THIS SESSION WAS TESTING.** Operator, 2026-09-07: *"that's what we are
testing for -- the lowest friction path to renders using my canonical json."*
Not tuning, not quality, not benchmarking: **can a person install this pack and
press Run on the shipped graph, and get an episode?**

Read the findings below through that lens. Every one of the six defects was
FRICTION ON THE PRESS-RUN PATH, and each stopped a render outright:

| # | defect | where it stopped you |
| --- | --- | --- |
| 1 | `tokenizers` pin | ComfyUI would not boot AT ALL |
| 2 | `viz_mxc_mandala` needs Windows-only `pycairo` | first music shot, no fallback |
| 3 | ffmpeg never declared | the mp4 encode |
| 4 | ffprobe was the other half | the clip-contract check after encoding |
| 5 | SA3 100% NaN | crashed the render at minute 14 |
| 6 | sub-quadratic attention on MPS | rendered, but the music was noise |

**Four of the six are not Mac-specific** -- 1 breaks any host with transformers
>= 5.11, and 2, 3 and 4 break Linux too. Apple Silicon was the machine that
happened to walk the whole path first.

**Not in scope, deliberately:** sampling-quality tuning, story quality, and
benchmark comparisons. Where the evidence pointed at a tuning change (the SA3
step count, section 9) it is recorded and NOT applied, because a change that
does not reduce friction is not what this was for.

**First real Apple Silicon hardware session, 2026-09-07.** Mac mini M4 (10-core,
16 GB unified), macOS 26.6.2, ComfyUI Desktop 0.34.6 standalone `mac-mps`,
Python 3.13.12, torch 2.12.1, `Device: mps`.

Everything below was MEASURED on that box. Where something is inferred or still
open, it says so. Companion entries: PBUG-20260907-05 through -08 in
`PROD_BUG_LOG.md`; the plan-level version is the 2026-09-07 section of
`GO_FORWARD_PLAN.md`.

**SCOPE WARNING: these are results for this named M4 / 16 GB host, not a claim
about every Mac.** A model marked OOM, unsafe or not offered here may work on a
Mac with more unified memory, a different macOS/PyTorch generation, or a future
optimized backend. Keep platform capability separate from this machine's
qualification status: `failed on this Mac` does not mean `cannot run on
macOS`. The removed local caches are likewise a storage choice for this host,
not a cross-platform retirement of those models.

---

## 1. Does MPS actually execute? YES -- for the writer.

The standing worry was that `nodes/` carries ~40 `torch.cuda.is_available()`
checks and zero `torch.backends.mps.is_available()` checks (confirmed: 40 cuda
sites, and the single `torch.backends.mps` hit is a docstring), so selecting
`mps` might silently route to CPU.

**It does not, for the writer.** `llm_device` flows
`OTR_LedgerScriptWriter` -> `_policy.device` -> `_otr_model_loader.py`:

```python
if quant_config is None and max_memory is None:
    model = model.to(device)          # device == "mps"
```

The canonical runs quant `none` with `max_memory=None`, so that branch is taken.
Verified on the live process rather than inferred: `footprint -p <pid>` showed
**14 GB phys_footprint** with `AGXMetalG16G_B0.bundle` (the Apple GPU Metal
driver) mapped into the address space and `IOAccelerator` regions resident.

**Where an mps selection IS ignored:**

* **Kokoro TTS** -- `voice_device="mps"` is accepted by the widget and unused by
  the engine. On Python 3.13 the backend is `kokoro-onnx` (the torch `kokoro`
  line is gated `python_version < "3.13"`), and its onnxruntime session is built
  from an explicit CPU-by-default provider list. **This is deliberate: the ONNX
  backend is what makes Kokoro auto-install on 3.13 at all.** The code logs the
  device stamp as unused. The defect is only that the dropdown still offers
  `mps` as if it did something.
* **Upscale lane** -- `_otr_upscale_engines/_resolve.py` rejects `mps` outright,
  by documented intent, and raises rather than degrading.
* **VRAM budgeter** -- `total_vram` comes from `torch.cuda.get_device_properties`
  behind a cuda guard, so it is `0` here and the ceiling widget enforces nothing
  on Mac.

**The one already-correct device-aware line** is the GGUF backend:
`default_layers = DEFAULT_N_GPU_LAYERS if policy.device in ("cuda", "mps") else 0`.
That is the idiom the rest of the tree should converge on.

---

## 2. NF4 QUANTIZATION IS DEAD ON METAL. This is the biggest trap.

`bitsandbytes` installs fine on macOS arm64 (0.50.2 has a wheel and ships an
`mps` backend module), and NF4 genuinely computes -- a `Linear4bit` moved to
`mps` kept a real `Params4bit` weight with a live `quant_state` and returned
finite bf16 output. **So it works, and you must still not use it.**

| writer config | speed | peak footprint |
| --- | --- | --- |
| Qwen3.5-4B, quant `none` (bf16) | **6.6 tok/s** | ~9 GB standalone / ~14 GB in ComfyUI |
| Qwen3.5-4B, `bnb_nf4` on mps | **0.3-0.5 tok/s** | ~5.9 GB |
| Qwen3.5-4B, `bnb_nf4` on CUDA (for contrast) | 14.47 tok/s | 2.99 GiB |

There is no optimized Metal kernel for NF4. bitsandbytes falls back to
`backends/default/ops.py::_dequantize_4bit_compute`, a pure-PyTorch dequantize
loop that blows past `torch._dynamo`'s recompile limit. Same quantization, **48x
apart** between CUDA and Metal.

**This is not a speed-vs-memory tradeoff you get to choose.** At 0.5 tok/s the
`NewsCurationDeep` phase cannot finish inside its 40 s budget, so the run fails
100 % of the time regardless of fitting in RAM. It surfaces as a confusing
`_LLMTimeoutWorkflowPause`, not as "quantization is slow".

**On Apple Silicon, `llm_quant_policy` must be `none`.**

---

## 3. On this host, 16 GB unified is the wall and no tested dropdown gets under it

| | |
| --- | --- |
| writer `phys_footprint` while generating | **14 GB** |
| machine total | 16 GB unified, shared with macOS |
| swap at peak | 0.00 M -> **5.6 GB** |
| free memory at peak | **1 %** |
| outcome | ComfyUI **OOM-killed by macOS**, twice |

Levers that do NOT work, all measured:

* **NF4** -- fits at 5.9 GB but is 13x too slow to clear a timeout (section 2).
* **`gemma-4-E2B` instead of Qwen** -- measured **~10 GB at bf16**, *larger* than
  Qwen's ~9 GB. The catalog's "~3 GB resident" for E2B is its NF4 figure, and NF4
  is unusable here. **Qwen3.5-4B at quant `none` is the smallest viable Mac
  config**, which is why the canonical ships exactly that.

The lever that should work and is not wired up: **GGUF via llama.cpp.**
llama.cpp has real Metal kernels, `Q4_K_M` on a 4B writer is roughly 2.5 GB, and
`_otr_gguf_backend.py` already honours `mps` for `n_gpu_layers`. The README
documents the build
(`CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python==0.3.33`), but
`llama-cpp-python` is declared in neither `requirements.txt` nor
`pyproject.toml`, so a clean install cannot reach it. See PBUG-20260907-07.

---

## 4. `ps rss` lies on Apple Silicon. Use `footprint`.

`ps` reported **0.33 GB** for a process actually holding **14 GB** -- unified and
GPU memory do not land in RSS. A 40x error. Size anything on this platform with
`/usr/bin/footprint -p <pid>` and read `phys_footprint`.

Watch the units: `footprint` switches between MB and GB, and a parser that
assumes MB will silently report `10 GB` as `10`. That bug briefly made E2B look
20x smaller than it is during this very session.

---

## 5. Error messages say CUDA on a machine with no CUDA

Two seen in one run, both misleading during Mac diagnosis:

* `[StoryOrchestrator] CUDA warmup complete (8.9s)` -- a hardcoded string in the
  warmup block. The warmup runs on whatever device the model is on. It is NOT
  evidence that anything routed to CUDA, and it cost real diagnosis time here.
* `NewsCurationDeep exceeded 40s; orphan worker still on GPU ... racing the
  orphan's CUDA kernels` -- there are no CUDA kernels. It is a plain timeout;
  nothing is orphaned or stuck.

Neither is a functional fault. Both should be reworded to name the actual device.

---

## 6. Operational traps that cost time in this session

1. **The install can brick ComfyUI, and the UI cannot recover it.** The shipped
   `tokenizers>=0.22,<=0.23` pin conflicts with the bundled transformers and
   ComfyUI Manager installs requirements ONE LINE AT A TIME, so the resolver
   silently downgrades and ComfyUI then exits 1 on every launch. Because Manager
   *is* a ComfyUI extension, a bricked boot means you cannot install the fix from
   the UI -- it needs a terminal. Fixed in `2.0.0-alpha.29`; **`.24` through `.28`
   all carry it.** Full detail: PBUG-20260907-05.
2. **Reinstalling from the registry undoes everything.** It restores the bad pin
   *and* replaces the canonical with the CUDA-targeted one
   (`cuda / bnb_nf4 / 14.5`), which then fails on hardware you do not have. Two
   bugs stacked. Do not reinstall until a fixed version is the served one.
3. **The ComfyUI canvas outlives the file on disk.** After the install was
   repaired, Queue Prompt still ran the *previous* graph's widget values
   (`gemma-4-E2B-it` + NF4 + `viz_green`) because the frontend keeps the loaded
   graph. Always re-load from
   `Workflow > Browse Templates > EXTENSIONS > comfyui-old-time-radio >
   otr_canonical` rather than reusing the canvas.
4. **The template gallery is behind a button named "Extensions", not "Manager".**
   ComfyUI 0.34.6 has no button labelled "Manager" anywhere; it opens a panel
   headed "Nodes Manager". Docs saying "open ComfyUI Manager" dead-end. If the
   template seems to vanish, check the server first --
   `GET /api/workflow_templates` -- it is usually a stale frontend cache, cured
   by a hard refresh.
5. **A render starves the remote desktop.** On a rented Mac the box appears to
   freeze mid-run; that is memory pressure on shared 16 GB, not a hang.

---

## 7. PROVEN on Apple Silicon -- an episode reached `otr/obs/`

**2026-09-07, 20:21 -- the first OTR episode ever produced on Apple Silicon.**

```
otr/obs/magnetic_pulse_20260907_201810__rfrc__vcam__none__koko__news__q354b__sa3_final.mp4
  duration  135.1 s
  video     h264 1920x1080 @ 25 fps, 3378 frames (encoded 2695 frames in 24.5 s, 110.2 fps)
  audio     aac 48000 Hz stereo
  RMS       -53 -> -37 dBFS across the timeline (varying: real content, not the
            silent fallback the NaN guard would have produced)
  SA3       ZERO non-finite warnings
```

The whole path, fully local, no image weights and no API keys: Qwen3.5-4B writer
on Metal -> Kokoro voices -> Stable Audio 3 music -> visualizer video -> ffmpeg
encode -> published. By the operator's own standard -- *a leg that does not reach
`otr/obs/` did not pass* -- **Apple Silicon passes.**

### What it took, in order

Five defects stood between a fresh Mac install and that file. Each was real, and
none was "Apple Silicon can't do this":

1. **`tokenizers>=0.22,<=0.23`** -- excluded 0.23.1 and bricked ComfyUI's boot
   entirely (PBUG-20260907-05). Fixed in alpha.29.
2. **The canonical shipped `viz_mxc_mandala`**, whose `pycairo` dependency is
   Windows-only, as the music lane -- so the shipped default could never render
   on macOS *or Linux* (PBUG-20260907-10). Now `viz_green`.
3. **ffmpeg was never a declared dependency** -- the run died at the mp4 encode
   telling a Mac user to run `winget install ffmpeg`. Now `imageio-ffmpeg`.
4. **ffprobe was the other half** -- imageio ships ffmpeg alone, and the
   visualizer probes back every clip it encodes. Now `ffmpeg-downloader`, which
   installs a matched pair.
5. **SA3 returned 100% NaN** -- and the cause was OTR's own determinism wrapper
   meeting an MPS `baddbmm` bug, not the model (PBUG-20260907-09b).

### Still unmeasured

* **Wall clock.** This run took roughly 24 minutes end to end; no clean
  before/after comparison against a CUDA box has been made.
* **Memory headroom.** It completed at ~44% free, but earlier runs OOM-killed at
  quant `none`. The margin is thin and not characterised.
* **Repeatability.** ONE episode. Nothing here proves the second one lands.
* **Audio quality.** OTR drives `stable_audio_3_small_music` -- the
  consistency-distilled member of Comfy-Org's base/non-base pair -- at
  `steps=100, cfg=7.0`, while Comfy-Org's own template for a non-base checkpoint
  uses `steps=8, cfg=1` (verified locally against both shipped templates). The
  source comment claiming "cfg=7.0 (SA3 native default)" is not supported by
  Comfy-Org's own default for the checkpoint this pack loads. The music is
  audible and the episode publishes; whether it SOUNDS better or worse at the
  intended recipe is untested, and the operator has explicitly deprioritised
  chasing it.
* Every lane outside the canonical: no local image or video-diffusion engine
  exists on this platform at all (section 8).


---

## 9. WHY stable_audio_3 -- the selection criterion is licensing and auto-install, NOT quality

**Operator, 2026-09-07:** *"the reason why we chose stable audio three was that
it was ungated and it was auto installed. That's the main [thing]."*

Record this before someone "improves" it away. The music engine was chosen on
two hard requirements, and quality was never one of them:

1. **A friendlier COMMERCIAL LICENCE than MusicGen.** Operator, same day: *"it
   had a friendlier commercial licence than musicgen."* This is the reason
   `musicgen` was replaced: MusicGen is **CC-BY-NC**, so the shipped default was
   silently producing non-commercially-licensed music beds for every user who
   never touched the dropdown (PBUG-20260907-04).
2. **Ungated** -- no licence click, no HF token, nothing to accept before the
   weights will download.
3. **Auto-installs** -- the weights fetch themselves at boot with no manual step.
   On a pack whose premise is *press Run*, an engine needing a human to go accept
   something is not a candidate.

**(1) and (2) are INDEPENDENT tests and it is worth keeping them apart.**
MusicGen is *ungated but non-commercial* -- freely downloadable, and still
unusable for a commercial episode. A model can pass either check and fail the
other, so both must be applied.

**What this rules out, permanently, regardless of how good it sounds:**

* Any **gated** model (a licence click, an HF token, a manual download).
* Any **non-commercial** licence -- that is what `musicgen` failed on.
* Any engine whose weights cannot be fetched by the boot prefetch.

**And what it means for the SA3 sampling config.** The A/B (2026-09-07) showed
Comfy-Org's own recipe for this checkpoint is 10.6x faster with no measurable
quality difference, and the source comment claiming `cfg=7.0` is the "SA3 native
default" is false. **None of that is a reason to switch engines** -- it is at
most a reason to retune the one we have. The engine choice is settled on
licensing and installability, and those have not changed.

Operator on scope, the same day: *"we just need it to run and produce music and
not fail the episode."* It does, once the attention backend is right (section 8
/ PBUG-20260907-11).

---

## 10. A pulled checkout is not the Comfy Desktop runtime copy

**Observed 2026-09-10 on the M4 / 16 GB My Story trial.** Pulling the Git
checkout under `Documents/otr-mac/repo` did not update the separate extension
copy loaded by Comfy Desktop under its instance `custom_nodes` directory.  The
first canonical API preflight failed closed with a useful mismatch:
`OTR_LedgerScriptWriter` had 37 saved widget values but the live server exposed
only 33 serialized slots.  No prompt was queued.

For this installation layout, update the runtime copy as well and restart
ComfyUI before using the new canonical graph.  Confirm the restart by fetching
the live object schema or by running the canonical API preflight; a successful
My Story generation must report 37 writer widgets and zero workflow drift.
Refreshing only the browser canvas is not sufficient because Python node
classes are imported at server startup.

Do not work around a stale schema by editing the canonical JSON or trimming its
widget vector.  The mismatch is deployment state, not workflow corruption.

### My Story retry evidence from the same trial

The first generated act invented a fourth speaker, `Voice (Female)`, outside
the supplied three-person cast (Ada Mercer, Eli Ward and Jonah Pike).  The
typed act validator rejected attempt 1 and the bounded repair attempt rewrote
the act using only the admitted cast.  This is expected repair-path evidence,
not permission to weaken speaker validation: without the rejection, CastLock
would receive an unowned speaking role and could not guarantee a distinct
available voice.

The run also showed why publication remains the only PASS signal.  Its first
server process ended after ledger cleanup but before TTS, still rendering or an
artifact in `otr/obs/`; the saved input and pending ledger are recovery
evidence, not a delivered episode.  A retry with one updated server reproduced
the same boundary.  The macOS kernel then supplied the missing evidence:
`memorystatus` killed `python3.13` as the largest compressed process at
19,163 MB.  This was an operating-system memory kill, which explains the bare
process exit and absent Python traceback.

Qwen3.5-4B on MPS therefore is not safe for this particular My Story plus
per-line cleanup workload on the measured M4 / 16 GB host, even though shorter
canonical episodes previously published with the same writer.  This does not
establish an all-Mac limit: a higher-memory Apple Silicon machine may complete
the same route, and no such machine was tested here.  Do not repeat the
identical route on this host after this receipt; select a smaller Mac-offered
writer and keep the remaining proven Mac dropdowns, then require a real
`otr/obs/` publication before changing the matrix.  Process residency across
many local model calls matters more here than the model's static weight size.

The immediate smaller-writer fallback also failed closed, for a different and
useful reason.  `unsloth/Llama-3.2-3B-Instruct` was not already cached; its
auto-fetch requires the 6.4 GB payload plus the loader's 5 GB safety margin,
while this host had only 8.5 GB free.  The loader raised
`InsufficientDiskSpaceError` before downloading anything.  At this point a
continued local trial requires an explicit storage-cleanup decision; do not
silently delete model caches merely to make a qualification run proceed.
