# Apple Silicon compliance matrix

**GENERATED 2026-09-09** by a 41-agent audit of all 61 registered engines, each verdict grounded in the registries and adapter code, and every load-bearing verdict (PROVEN / WILL NOT RUN / OOM RISK) adversarially re-checked. **5 were overturned** and are marked below.

Target: **Apple M4 / 16 GB unified memory**. An OOM there is a HARD MACHINE REBOOT, not a process kill, which is why OOM risk is a first-class column.

## How to read a verdict

| verdict | means |
|---|---|
| **PROVEN** | has actually rendered on this machine, with a receipt |
| **LIKELY** | no Metal blocker found and it fits 16 GB, but nobody has run it here |
| **OOM RISK @16GB** | would likely exhaust unified memory -- expect a reboot, not an error |
| **WILL NOT RUN** | an unconditional blocker on any Mac |

**A missing `mps` in `device_backends` does NOT mean an engine cannot run.** In this repo that row is a claim about PROVEN EXECUTION, so its absence means nobody has run it. The two questions are separate and this matrix answers both.

**Gated or manual weights are friction, not failure** -- recorded so you know what a first run costs, never as a mark against the lane.

## Summary

| verdict | count |
|---|---|
| WILL NOT RUN | **5** |
| OOM RISK @16GB | **13** |
| PROVEN | **16** |
| LIKELY | **27** |
| | **61 total** |

## WILL NOT RUN (5)

| engine | ns | weights | ~GB | why |
|---|---|---|---|---|
| `ideogram4_local` | image | manual | 17.3 | It is the one local image row that declares needs_fp4_te True and requires_vendor "nvidia", and every escape from fp4 is also closed on Metal: int8_convrot would hit the same unimplemented aten::_int_mm that killed z_image's int8  |
| `fastwan_8gb` | video | manual | 21.8 | **[OVERTURNED from OOM RISK @16GB]** A subclass of wan_ti2v that hoists MORE (it adds the rank-128 DMD LoRA to the session nodes) at a ~21.8 GB peak, so it is further from fitting than the incumbent that already rebooted this mach |
| `ltx_audio_in` | video | manual | 15.2 | assert_usable hard-refuses when NVML is unavailable -- a real code gate, not a registry row -- and NVML is an NVIDIA-only library, so this lane fails closed on every Mac regardless of memory. |
| `minimax_h3_audio_in` | video | manual | 42.5 | The REF2VA sibling shares the same NVFP4 encoder and int8-convrot DiT class, adds the audio VAE for reference encoding, and totals 42.5 GB -- so both the fp4/int8 Metal blocker and the size wall apply unchanged. |
| `minimax_h3_video` | video | manual | 41.9 | It conditions on an NVFP4-AWQ Qwen3-VL encoder and an int8-convrot DiT -- Metal executes neither fp4 nor int8 matmul (the measured z_image_turbo int8 build died on `aten::_int_mm` not implemented for MPS) -- and the three artifact |

**Blockers, verbatim:**

* `ideogram4_local` -- nvfp4 (fp4) text encoder + dual-expert diffusion; the non-fp4 ladders are int8 (aten::_int_mm missing on MPS) or ~27 GB of fp8 — no Metal-viable precision exists
* `ltx_audio_in` -- Hard NVML/pynvml gate in assert_usable (requires_vendor "nvidia") -- unavailable on Apple Silicon; the LTX-2.3 22B stack would also not fit 16 GB.
* `minimax_h3_audio_in` -- NVFP4-AWQ text encoder (needs_fp4_te True) plus an int8-convrot DiT; MPS has no fp4 path and no aten::_int_mm.
* `minimax_h3_video` -- NVFP4-AWQ text encoder (needs_fp4_te True) plus an int8-convrot DiT; MPS has no fp4 path and no aten::_int_mm.

## OOM RISK @16GB (13)

| engine | ns | weights | ~GB | why |
|---|---|---|---|---|
| `indextts2` | audio | manual | 11.1 | **[OVERTURNED from WILL NOT RUN]** Two independent hard blockers: its uv-locked sidecar venv is pinned to Python 3.10 + torch 2.8.0+cu128 behind a PowerShell-only installer, and the checkpoints alone are 11.06 GB inside an 18.93 G |
| `flux_gen1` | image | manual | 13.0 | One all-in-one FLUX.1-dev fp8 checkpoint the adapter itself sizes at ~13 GiB (17.3 GB per the engine survey) with no split/offload structure, and its default request is 832x1216 — z_image's 12.3 GB checkpoint already needed 20.4 G |
| `z_image_turbo` | image | auto, ungated | 19.3 | Measured on this box: bf16 (12.31 GB) loads fully on Metal and then dies in the KSampler needing ~20.4 GiB of a 20.13 GiB ceiling; the int8_convrot build hits NotImplementedError aten::_int_mm (no int8 matmul on Metal at all); nvf |
| `animatediff15_v3_haunted_video` | video | manual | 3.8 | Only 3.77 GB of weights and no Metal blocker (its Lightning sibling published on this M4 with the same node pack and SD1.5 substrate), but it declares adaptive_hold_for_memory=False, so it has NO guard against the 136-latent ceili |
| `animatediff15_v3_stillin_lab_video` | video | manual | 3.8 | It subclasses the haunted lane and therefore inherits hold_factor=2 with adaptive_hold_for_memory=False at the same 512x288 canvas, so the same audio-driven ~160-latent beat that rebooted this M4 will reach it unguarded. |
| `humo` | video | manual | 26.7 | HuMo renders FULLY RESIDENT by contract (no free_after_use -- forcing eviction fragmented the allocator into an OOM), so its 17.89 GB fp8 UNET, 6.27 GB umt5, 2.88 GB whisper, VAE and LoRA are co-resident at 26.74 GiB against an ~1 |
| `humo_1.7B` | video | manual | 12.6 | Its own UNET is only 3.24 GB, but HuMo17BEngine._loader_names overrides ONLY unet and lora, so it still inherits the 6.27 GB umt5_xxl_fp8 encoder plus whisper 2.88 GB and the VAE, all fully resident -- about 12.6 GB against a ~10. |
| `humo_14B_169` | video | manual | 26.7 | Same 14B fp8 checkpoint and same fully-resident loader set as humo, rendered 16:9 instead of portrait -- identical heavy residency class, so 26.74 GiB of weights against an ~11.8 GiB Metal ceiling. |
| `ltx25_foley_plus` | video | manual | 19.6 | Registry-identical to ltx25_video -- the same five artifacts and the same graph, differing only in a two-node audio decode after the DiT is reclaimed -- so it carries the same ~19.6 GB residency and the same MPS black-video defect |
| `ltx25_mime` | video | manual | 19.6 | The third LTX 2.5 sibling on the same graph and the same five artifacts -- the only difference is what happens to the model's audio after the render -- so ~19.6 GB of weights against an ~11.8 GiB Metal ceiling, plus the same all-b |
| `ltx25_video` | video | manual | 19.6 | A 10.73 GB LTX 2.5 DiT plus an 8.86 GB Gemma-4 12B encoder is ~19.6 GB, and the adapter's CPU-pinning of that encoder buys nothing on unified memory where 'cpu' is the same physical RAM; on top of that, MPS BF16 attention NaNs are |
| `ltx_video` | video | manual | 14.8 | The LTX-2.3 22B stack: a 10.03 GB Q3_K_M GGUF plus an 8.80 GB Gemma-3 encoder, a 42.98 GB projection checkpoint and a 7.08 GB LoRA, with a measured per-clip peak of ~14.8 GB on a 16 GB CUDA card -- past the ~11.8 GiB Metal working |
| `wan_ti2v` | video | manual | 10.6 | It is the one lane MEASURED fatal here: it loaded fully on Metal ('WAN22 ... loaded completely; 9536.40 MB, full load: True') and then took the whole machine down, and the pack's unified-memory guard now refuses it at 10.6 GiB of  |

## PROVEN (16)

| engine | ns | weights | ~GB | why |
|---|---|---|---|---|
| `bark` | audio | auto, ungated | 4.2 | Ran on Metal on this exact host: 40.8 s for 4.6 s of structured speech (flatness 0.070, finite); the row gained "mps" on that measurement and _otr_bark_lib now probes cuda->mps->cpu instead of the old hardcoded `cuda if available  |
| `kokoro` | audio | auto, ungated | 0.3 | Shipped voice engine of the otr_mac_mps profile and part of six published M4 episodes; on Python 3.13 it runs kokoro-onnx on CPU by design, so the mps voice_device stamp is accepted-and-unused rather than a failure. |
| `musicgen` | audio | auto, ungated | 2.2 | Measured on the M4 2026-09-07 at mps 14.1 s vs cpu 14.7 s for 256 tokens, flatness 0.134/0.103, finite; it is a transformers model so the ComfyUI MPS attention fault does not touch it. Licence (CC-BY-NC) is why it is not the defau |
| `stable_audio_3` | audio | auto, ungated | 3.5 | Declares mps, is the mac profile's music engine, and published on the M4 - but only because the pack's prestartup forces PyTorch/SDPA attention on MPS (sub-quadratic gives noise) and _otr_determinism disables fill_uninitialized_me |
| `flux2_klein` | image | manual | 11.0 | It actually rendered on this M4 — klein_run.log shows Flux2TEModel_ (7672.25 MB) and Flux2 (2591.64 MB) both loaded and sampling, and it minted a clean 1472x832 still at ~8 min/still — but it survived on swap (20 GB phys_footprint |
| `sd15` | image | auto, ungated | 2.0 | The only image engine whose CAPABILITIES row declares mps on purpose, and it has published real episodes on this exact M4/16 GB — 1.99 GB single CheckpointLoaderSimple file, no split loaders, no TE download, long side clamped to 7 |
| `off` | upscale | none | - | Pure pass-through that never resolves or loads a device — the composite only calls resolve_device when engine.name != "off" — and it is the shipped widget value ('off','cpu') on node 84 of the Mac lightning variant that published  |
| `animatediff15_lightning_video` | video | manual | 3.1 | A complete 23-beat episode rendered through the real OTR adapter path on the M4/16 GB and published, and it is the only Ghost lane that sets adaptive_hold_for_memory=True so the 136-latent reboot ceiling is guarded. |
| `ltx_8gb` | video | auto, ungated | 9.1 | Published real 1080p episodes on this exact Mac mini M4/16 GB; the registry row was corrected from ["cuda"] to ["cuda","mps"] only after the receipt landed, and its two weights self-fetch. |
| `still_flat` | video | none | - | Dead-flat ffmpeg hold of a minted still, no weights; it drew beats b001/b006 in the published three-lane still run on this M4. |
| `still_motion` | video | none | - | Pure CPU/ffmpeg hold-and-pan over a still, zero weights and zero VRAM; it published on this M4 with sd15 supplying the still. |
| `still_pan` | video | none | - | CPU/ffmpeg pan over a minted still, no weights and no VRAM; covered by the published two-run sweep of all four still_* lanes on this M4. |
| `still_word` | video | none | - | Same ffmpeg flat-hold mechanics as still_flat with a word-card prompt; it is the DELIVERED engine in the published 20260908_042821 receipt (stwo shortcode). |
| `viz_camera` | video | none | - | First-party numpy/PIL/ffmpeg procgen visualizer whose assert_usable probes only ffmpeg; it is the shipped Mac profile's video engine and published episodes on this M4. |
| `viz_green` | video | none | - | Torch-free CRT scope drawn with numpy/PIL and encoded by ffmpeg; assert_usable gates on ffmpeg alone and it is one of the three lanes in the proven zero-download Mac canonical. |
| `viz_mxc_cpu` | video | none | - | Pure numpy/PIL rainbow scope, no GPU and no shaders; assert_usable checks ffmpeg only, and it is the announcer lane in the shipped Mac profile that published on this M4. |

## LIKELY (27)

| engine | ns | weights | ~GB | why |
|---|---|---|---|---|
| `chatterbox` | audio | manual | 3.0 | **[OVERTURNED from WILL NOT RUN]** There is no macOS install path: the only installer is PowerShell and the adapter's default sidecar interpreter is the Windows `.venv/Scripts/python.exe`, while the Mac profile sets allow_sidecars |
| `dia` | audio | manual | 6.0 | **[OVERTURNED from WILL NOT RUN]** Its installer force-reinstalls a nightly cu128 torch as the final step, a wheel index that publishes nothing for macOS, and the worker never accepts a device at all - it picks fp16 only when torc |
| `elevenlabs` | audio | none | - | No local model, sidecar or GPU at all - it runs on Comfy Cloud through invoke_partner_node, so the cost is credits and auth rather than unified memory; declares mps and has simply never been exercised from this machine. |
| `google_lyria` | audio | none | - | Direct Lyria-3-clip BYO-key lane with no local weights; the only host dependency is an ffmpeg binary to decode the returned MP3, and this Mac has ffmpeg 9.0 (the pack already probes it rather than hardcoding flags). |
| `google_tts` | audio | none | - | Direct Gemini BYO-key REST lane with no Google SDK, torch, numpy or CUDA import at module scope and model_requirements: [] - nothing about it touches Metal; untested here only because testing it measures an API key. |
| `sonilo` | audio | none | - | Cloud music engine over invoke_partner_node with model_requirements: [] - zero local residency, so 16 GB unified is irrelevant; declares mps and needs only a Comfy Cloud key, but no Mac run is on record. |
| `stable_audio_music` | audio | auto, GATED | 4.5 | Never run on a Mac and declares only ["cuda"], but the adapter's sole device code is `.to(dev)` plus `torch.Generator(device=dev)`, and a seeded torch.Generator("mps") is recorded as returning finite noise on this exact host - so  |
| `cloud_flux_pro` | image | none | - | No local compute at all — the provider renders and the only host-side work is a PIL sRGB-PNG canonicalize; the row already declares mps with practical_without_gpu True and assert_usable checks nothing but Pillow and a healthy part |
| `cloud_krea_2_turbo` | image | none | - | Cloud partner row: no local weights, no VRAM, mps declared and practical_without_gpu True, with the only usability check being Pillow plus a healthy cloud_krea_2_turbo pin row — nothing here touches Metal. |
| `cloud_luma_photon_flash` | image | none | - | Cloud partner row with empty model_requirements and mps already declared; the compute is the provider's and the host only transcodes to sRGB PNG, so the only thing between this and a still on an M4 is an API key. |
| `cloud_nano_banana_2` | image | none | - | Same CPU-side partner-API shape as every cloud still row — mps declared, practical_without_gpu True, zero model_requirements, and the only structural check is Pillow plus the cloud_nano_banana_2 pin row; untested here purely becau |
| `cloud_seedream_2` | image | none | - | Cloud partner row with no weights and no accelerator use; mps is already declared and practical_without_gpu is True, and assert_usable only requires Pillow and an OK cloud_seedream_2 pin row — credentials, not Metal, are the gate. |
| `google_image` | image | none | - | Direct Gemini BYO-key adapter that never invokes a local model — module scope imports no torch/PIL/SDK and assert_usable does nothing but resolve_api_key(), so the sole failure mode on an M4 is a missing Google API key, not Metal. |
| `ideo` | image | none | - | The cloud Ideogram v4 scene-still row (node_key cloud_ideogram_v4) — not to be confused with local ideogram4_local: no weights, no fp4, mps declared, practical_without_gpu True, gated only by credentials and its partner pin row. |
| `lumina_image` | image | manual | 10.4 | No Metal blocker exists anywhere in it — bf16 native flow model through stock UNETLoader/CLIPLoader(lumina2)/VAELoader, no fp8, no fp4, no GGUF pack, no sidecar, no device check in assert_usable — and its ~10.4 GB concurrent artif |
| `spandrel_esrgan` | upscale | auto, ungated | 0.1 | No Metal blocker exists (needs_fp8_te/needs_fp4_te both False, requires_sidecar False, required_toolchain None, plain ESRGAN/RRDBNet convs through spandrel) and the checkpoint is only 67 MB, but _resolve.resolve_device rejects the |
| `cloud_kling_avatar` | video | none | - | The render happens provider-side with zero local weights; assert_usable checks only ffmpeg/ffprobe and the partner pin row, so what it measures is a Comfy Cloud credential, not Apple Silicon (it does need an image engine for its i |
| `cloud_seedance_2` | video | none | - | Provider-side render, no local weights and no device gate in assert_usable; the only requirement beyond ffmpeg/ffprobe is OTR_COMFY_API_KEY plus an init_image from a working image engine. |
| `cloud_vidu_q2_pro_fast_720p` | video | none | - | Fixed 720p Vidu Q2 image-to-video rendered provider-side; local device genuinely does not matter, only ffmpeg/ffprobe, the partner pin and credentials. |
| `cloud_wan_i2v` | video | none | - | Provider-side Wan i2v -- none of the local Wan MPS problems apply because no Wan weights load here; assert_usable wants ffmpeg/ffprobe and a pinned partner row only. |
| `cloud_wan_i2v_audio` | video | none | - | Same provider-side row as cloud_wan_i2v with an audio_ref added; zero local VRAM and no device check anywhere in its gate, so the variable is the Comfy Cloud credential. |
| `google_omni_video` | video | none | - | Direct Gemini Omni BYO-API text-to-video; assert_usable resolves a Google API key plus ffmpeg/ffprobe and nothing else, and the render is entirely provider-side. |
| `google_veo_video` | video | none | - | Direct Veo 3.1 predictLongRunning BYO-API lane with no local weights; the gate is a Google key and ffmpeg/ffprobe, both satisfiable on this host. |
| `humo_1.7B_169` | video | manual | 12.6 | **[OVERTURNED from OOM RISK @16GB]** The same 1.7B checkpoint and the same inherited fp8 umt5 + whisper fully-resident loader set as humo_1.7B, just at 832x480 -- roughly the same pixel budget, so the same ~12.6 GB residency again |
| `mesh_stage` | video | manual | 4.6 | assert_usable gates only on an OTR_BLENDER_EXE path (a plain os.path.exists, no .exe enforcement), ComfyUI CORE hy3d node classes and a 4.59 GB checkpoint; its reclaim barrier already branches to torch.mps, and nothing NVIDIA-spec |
| `viz_mxc_mandala` | video | none | - | Zero weights and a pure CPU cairo painter with no Metal path at all, but pycairo publishes no macOS wheel, so it needs `brew install cairo pkg-config` before pip can build it -- a friction step, not a device blocker, and nothing h |
| `word_razzle` | video | none | - | It is the Pixverse cloud_pixverse_i2v row, not a local lane -- on this Mac it failed with 'no credentials: set OTR_COMFY_API_KEY', which the guide records explicitly as measuring a credential rather than Apple Silicon. |

## What the adversarial pass overturned

Five verdicts did not survive being attacked. Each is a case where the first reading was plausible and wrong, which is the whole reason the pass exists.

* **`chatterbox`**: WILL NOT RUN -> **LIKELY**. The blocker is packaging + an unproven registry row, not hardware, and the claim's own note concedes it ("the blocker is packaging, not Metal"). Three specific refutations. (1) THE cu128 PIN IS FALSE FOR CHATTERBOX. scripts/_otr_chatterbox_install.ps1:21 has the cu128 line COMMENTED OUT; the live step is `pip install chatterbox-tts soundfile` (line 15), vendor-neutral. Contrast scripts/_otr_dia_in
* **`dia`**: WILL NOT RUN -> **LIKELY**. The blocker is a missing install, not an unconditional Metal blocker, and the device half of the claimed reason is factually wrong. (1) The cu128 pin lives in scripts/_otr_dia_install.ps1:21, a PowerShell script that also hardcodes C:\Users\jeffr\Documents\ComfyUI\dia at line 6 and therefore cannot execute on macOS at all - so it is the absence of a mac installer, not a runtime pin. Nothing in the
* **`fastwan_8gb`**: OOM RISK @16GB -> **WILL NOT RUN**. The size half of the claim is wrong and the deciding fact is the wrong one. (1) The ~21.8 GB peak is ungrounded: docs/2026-08-01-fastwan-8gb-MODEL-MANIFEST.md:17-22 pins the shipped files in bytes -- UNET Q5_K_M 3,810,603,360 + LoRA 660,874,456 + umt5 Q5_K_M encoder 4,145,878,880 + wan2.2 VAE 1,409,400,960 = 10.03 GB on disk, and 5.88 GB peak-concurrent under the pack's own floor rule (docs/MAC_PO
* **`humo_1.7B_169`**: OOM RISK @16GB -> **LIKELY**. REFUTED on the deciding fact. The 12.6 GB half is fine — HuMo really is fully-resident-by-contract (motion_common.py:702-723 names HuMo explicitly as the engine that does NOT evict; eng_humo.py:604-608 "HuMo renders FULLY RESIDENT by contract (BUG-265)"), and on unified memory motion_common.py:895-897 charges every artifact, so the sum is real, not guessed: humo_1.7B_fp16 3,483,511,088 + umt5_xxl_
* **`indextts2`**: WILL NOT RUN -> **OOM RISK @16GB**. Both claimed blockers fail. (1) The cu128 pin is refuted by the very file cited: scripts/_otr_indextts2_install.ps1:4-6 states the stack is "python 3.10 + torch 2.8 -- cu128 on Windows/Linux, Metal-capable default wheels on Mac, selected automatically by the repo's [tool.uv.sources]"; the "+cu128" at :40-42 is explicitly scoped to "this box" (the Windows reference machine), and :116-120 already ca

## Method and limits

One agent per namespace read the CAPABILITIES table and each adapter module for facts the table does not carry -- `required_toolchain`, `requires_sidecar`, `needs_fp8_te`/`needs_fp4_te`, `model_requirements`, and any hard device check in `assert_usable`. Every PROVEN / WILL NOT RUN / OOM RISK verdict then went to a second agent told to refute it.

**LIKELY is a prediction, not a receipt.** It means the code shows no Metal blocker and the weights fit; it does not mean anyone ran it. Only PROVEN rows carry an artifact. Treat OOM RISK as advisory too: sizes are estimates from artifact bytes, and the one measured ceiling on this host (136 AnimateDiff latents at 512x288, PBUG-20260909-01) is a single bracket rather than a memory model.

