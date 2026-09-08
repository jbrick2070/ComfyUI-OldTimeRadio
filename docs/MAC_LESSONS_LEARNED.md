# Mac (Apple Silicon) -- lessons learned

**First real Apple Silicon hardware session, 2026-09-07.** Mac mini M4 (10-core,
16 GB unified), macOS 26.6.2, ComfyUI Desktop 0.34.6 standalone `mac-mps`,
Python 3.13.12, torch 2.12.1, `Device: mps`.

Everything below was MEASURED on that box. Where something is inferred or still
open, it says so. Companion entries: PBUG-20260907-05 through -08 in
`PROD_BUG_LOG.md`; the plan-level version is the 2026-09-07 section of
`GO_FORWARD_PLAN.md`.

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

## 3. 16 GB unified is the wall, and no dropdown gets you under it

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

## 7. What is proven on Apple Silicon, and what is not

**Proven:** the pack installs, boots, registers all 25 nodes with zero import
failures, resolves `Device: mps`, passes the visual-asset preflight
(`READY engines=stable_audio_3`, with all three `z_image_turbo` slots correctly
refused as provably unused), fetches Stable Audio 3 weights, and drives the
writer on Metal at **~6.5 tok/s** through news curation, outline beats and
character descriptions.

**Not proven:** everything downstream of the writer. **No episode has reached
`otr/obs/` on this machine.** TTS, music, video and publish are all unmeasured on
Apple Silicon. By the operator's own standard -- *a leg that does not reach
`otr/obs/` did not pass* -- Mac has not passed.

`otr_obs_dir()` resolves to
`<comfy output>/otr/obs` (`OTR_OBS_DIR` unset). The directory does not exist
until the first successful publish, so its absence on a fresh box is expected,
not a fault.


---

## 8. There is NO local image or video-diffusion engine on Apple Silicon

Not a runtime discovery -- it is **declared in the registries**, so it can be
read without spending a render. Every engine carries a `device_backends` list.

**Video** (`nodes/_otr_video_engines/registry.py`):

| runs on mps | cuda only |
| --- | --- |
| `viz_green`, `viz_mxc_cpu`, `viz_mxc_mandala`, `viz_camera` | every LTX (`ltx_video`, `ltx_8gb`, `ltx25_video`, `ltx25_mime`, `ltx25_foley_plus`, `ltx_audio_in`) |
| `still_motion`, `still_flat`, `still_pan`, `still_word` | `wan_ti2v`, `fastwan_8gb`, `humo`, `mesh_stage` |
| `word_razzle` | `animatediff15_v3_haunted_video`, `animatediff15_v3_stillin_lab_video` |
| every `cloud_*` and `google_*` (API keys) | `minimax_h3_video`, `minimax_h3_audio_in` |

**Image** (`nodes/_otr_image_engines/registry.py`): `flux_gen1`, `flux2_klein`,
`lumina_image` and `z_image_turbo` are **all `[cuda]`**. Only `cloud_flux_pro`
and `google_image` list `mps`, and both are paid APIs.

**The consequence, and it is the shape of Mac support rather than a bug.** The
four `still_*` video lanes DO declare `mps` -- but on Apple Silicon nothing
local can produce the still they consume, so they are unreachable without an
API key. That leaves the visualizer lanes as **the only fully-local video path
that exists on this platform**.

So the shipped Mac canonical (`viz_mxc_cpu` / `viz_mxc_mandala` / `viz_camera`,
minting no stills, downloading no image weights) is not a conservative choice.
It is the only local configuration Apple Silicon can run at all.

**Practical:** a declaration is enforced before any work happens, so selecting
one of the cuda-only rows on a Mac raises `EngineUnusable` immediately and the
failure teaches nothing. Read `device_backends` first.

### But at least one of those declarations looks untested rather than measured

**`ltx_8gb` -- "LTX 0.9.8" -- deserves a real measurement before it is written
off.** It is a different engine from `ltx_video`, and the distinction matters:

| row | weights | plausible on Mac? |
| --- | --- | --- |
| `ltx_video` | `ltx-2.3-22b-dev-gguf` + `gemma-3-12b` encoder | no -- 22B, never a 16 GB model |
| `ltx_8gb` | `ltxv-2b-0.9.8-distilled` | **maybe** -- 2B distilled, all-in-one |

Grepping the 0.9.8 adapter (`eng_ltx_8gb.py`) for hard NVIDIA dependencies --
`nvenc`, `nvml`, `triton`, `flash_attn`, `torch.cuda`, `.cuda()`, `sm_*`, `fp8`,
`nvfp4` -- returns **nothing**. It drives stock ComfyUI nodes (`CLIPLoader`,
`CLIPTextEncode`), pins its text encoder to CPU (`t5_device="cpu"`) and diffuses
on whatever device ComfyUI resolved, which is `mps` here.

So its `["cuda"]` row is very likely the same shape as the stale
`bitsandbytes ... sys_platform != 'darwin'` marker: **a policy nobody has
retested, not a hardware fact.** The honest blocker is more likely MEMORY than
device -- the recipe's `t5xxl_fp16` encoder is roughly 9 GB by itself, on top of
the 2B diffusion model, which is the same 16 GB wall as section 3.

**This is unmeasured either way.** Flipping the declaration to test it is a real
experiment worth running on Apple Silicon hardware; asserting it works, or that
it cannot, would both be guesses. What IS established is that nothing in the
adapter code forbids it.
