# Mac engine survey -- selected test plan

**Purpose.** The canonical renders on Apple Silicon (2026-09-07). This plan
covers what else does. Operator's framing: *"they may not work, that's okay. We
just have to document what we need to do to get it running for people."*
**So auto-install stops being a gate and becomes a property we record.**

Every row's `device_backends` comes from the shipped registries. This document
originally said a declaration "is enforced before any work runs, so a `[cuda]`
row raises `EngineUnusable` on a Mac". **That was wrong** and is corrected in
`docs/ADDING_IMAGE_AND_VIDEO_LANES.md`: outside the registries and the tests the
row is read only by `capability_profiles._fit_reason()` (profile admission, via
`scripts/build_variants.py`) and by the machine-matrix generator. The render path
never consults it, so a `[cuda]` engine picked by hand IS attempted on a Mac and
fails -- if it fails -- on its own `assert_usable`.

That makes these tests MORE useful, not less. A wrong row does not stop anything;
it just misinforms every reader and every profile. So "is the declaration
honest?" is exactly the question, and only a render can answer it.

Status keys: **PASS** measured working / **FAIL** measured broken / **BLOCKED**
cannot run and why / **UNTESTED**.

---

## Tier 1 -- free or near-free (declares mps, or already on disk)

| engine | kind | declares | status | notes |
| --- | --- | --- | --- | --- |
| `kokoro` | TTS | cuda,cpu,mps | **PASS** | the shipped default. On py3.13 runs kokoro-onnx, CPU by design; `voice_device` is accepted and unused (PBUG-20260907-08) |
| `stable_audio_3` | music | cuda,mps | **PASS** | needs PyTorch attention on MPS or output is noise (PBUG-20260907-11) |
| `musicgen` | music | cuda,cpu,mps | **PASS** | measured 2026-09-07: mps 14.1 s vs cpu 14.7 s for 256 tokens, flatness 0.134/0.103, finite. Metal buys nothing at this size. **Unaffected by the attention bug** -- transformers model, not ComfyUI-native. Licence remains CC-BY-NC, so not a default candidate |
| `bark` | TTS | cuda,cpu,**mps** | **PASS on mps -- declaration FIXED 2026-09-07** | measured 2026-09-07: mps 40.8 s / flatness 0.070 / 4.6 s audio; cpu 27.8 s / 0.064 / 3.0 s. Structured speech both ways. The registry row now carries `mps` (fixed on this measurement); it previously omitted it AND `_otr_bark_lib.py` hardcoded `"cuda" if torch.cuda.is_available() else "cpu"` -- **the exact `cuda ... else cpu` pattern the original MPS audit was looking for**, and a real one. Note mps is not FASTER here (roughly a wash per second of audio), so the win is correctness of the declaration, not speed |
| `viz_green` / `viz_mxc_cpu` / `viz_camera` | video | cuda,cpu,mps | **PASS** | all three render; `viz_camera` is the operator's preference |
| `viz_mxc_mandala` | video | cuda,cpu,mps | **BLOCKED** | needs `pycairo`, which has no macOS wheel and needs libcairo headers. Declaration is right, the canonical was wrong to select it (PBUG-20260907-10) |
| `word_razzle` | video | cuda,cpu,mps | UNTESTED | cheap, no weights |
| `still_motion` / `still_flat` / `still_pan` / `still_word` | video | cuda,cpu,mps | **ALL FOUR PASS** (2026-09-08) | this row said BLOCKED (upstream) -- "they CONSUME a still and every local image engine is `[cuda]`, only reachable with a cloud image key". `sd15` removed that block. `still_motion` published at 00:55 (`__arch__stmo__`), the other three at 04:28 (`__cart__stwo__`); every still minted locally on Metal. See MAC_PORTABILITY_GUIDE section 9 |

## Tier 2 -- low-VRAM video, the real gap

Weights for `ltx_8gb` are ALREADY on disk: `ltxv-2b-0.9.8-distilled` (5.91 GB) +
`t5xxl_fp16` (9.12 GB) = **15 GB on a 16 GB machine.** Memory, not the device
declaration, is the likely blocker -- the T5 is loaded to CPU by the recipe
specifically to manage this.

| engine | declares | test | why |
| --- | --- | --- | --- |
| **`ltx_8gb` (LTX 0.9.8)** | cuda | flip the declaration, run one shot | **Highest value.** Its adapter contains NO NVIDIA-specific code (no nvenc/nvml/triton/flash_attn/torch.cuda), drives stock ComfyUI nodes, and pins its T5 to CPU. The `[cuda]` row looks untested rather than measured |
| `fastwan_8gb` | cuda | after LTX | 8 GB-class |
| `animatediff15_v3_haunted_video` | cuda | after LTX | 8 GB-class, proven on the 4060 |
| `ltx25_*`, `wan_ti2v`, `humo`, `mesh_stage`, `minimax_*` | cuda | LOW priority | large models; 16 GB unified is already the wall for a 4B writer |

## Tier 3 -- sidecar TTS (expensive, least likely to pay)

`indextts2`, `chatterbox`, `dia` -- all `[cuda]` AND `requires_sidecar: True`,
meaning an isolated venv per engine. Two costs (a Mac port AND a sidecar) for a
lane `kokoro` already fills. **Recommend deferring** unless a specific voice is
wanted.

## Tier 4 -- cloud / API (not a Mac question at all)

`elevenlabs`, `google_tts`, `google_lyria`, `sonilo`, every `cloud_*`,
`google_image`, `ideo` -- all already declare mps. They need CREDENTIALS, not a
port. Testing them measures an API key, not Apple Silicon. **Only worth doing to
unblock the `still_*` video lanes**, which is the one thing a cloud image key
would genuinely add to this platform.

---

## Method (keep it cheap)

1. **Read the declaration first.** A `[cuda]` row that the adapter contradicts is
   a one-line experiment; a `[cuda]` row backed by real CUDA-only code is not.
2. **Test the engine in isolation before a full episode.** A 12 s cue or one
   video shot answers the question for ~2 minutes of compute instead of ~25.
3. **Measure STRUCTURE, not level.** dBFS could not tell real music from noise
   here -- spectral flatness and a human ear could. See PBUG-20260907-11.
4. **Record what a user must DO**, not just pass/fail: the pip install, the
   manual download, the launch flag. That is the deliverable.


---

## Measured 2026-09-07 -- Tier 2 results

| engine | result | evidence |
| --- | --- | --- |
| `z_image_turbo` (image) | **RUNS ON METAL, OOMs at 16 GB** | text encoder 7.67 GB + Lumina2 11.74 GB both `full load: True`; KSampler needed ~20.4 GiB against a 20.13 GiB ceiling. Checkpoint is 12.3 GB. **Memory-blocked, not device-blocked** -- the `["cuda"]` row is wrong in kind |
| `ltx_8gb` (LTX 0.9.8) | **PASS -- PROVEN on Metal** (2026-09-08) | this row read INCONCLUSIVE: it passed the engine gate on mps but failed at the still it consumes, because that still came from `z_image_turbo`. `sd15` fixed the supply and the adapter then ran for real. Receipts: `otr/obs/..._20260908_012201__vart__lx8g__...` (one LTX lane, 39:17) and `..._20260908_030426__sbke__lx8g__...` (all three roles on LTX, 1:07:27, ~14 GB peak, zero errors). The `["cuda"]` row was untested policy and is now `["cuda","mps"]` |
| `animatediff15_v3_*` | weights fetched, UNTESTED | `v1-5-pruned-emaonly-fp16` (1.99 GB) + `v3_sd15_mm.ckpt` (1.56 GB) now on disk. Both adapters contain zero NVIDIA-specific code. **These are SD1.5-class and far smaller than z_image, so they are the best remaining candidate for local video on 16 GB** |

**What this changes about the plan:** the blocker for local image AND
image-to-video on this box is a 12.3 GB image model on 16 GB of shared RAM.
Testing more large models on this machine measures the RAM, not the port. The
useful remaining tests are the SMALL ones -- animatediff (3.5 GB of weights) --
and, if a bigger Mac ever appears, a re-run of z_image_turbo which is ~300 MB
short of fitting here.


---

## PROVEN 2026-09-08 -- a local image lane on Apple Silicon

```
otr/obs/the_tick_that_breaks_reason_20260908_005546__arch__stmo__unk__koko__pubd__q354b__sa3_final.mp4
  h264 1920x1080 + aac | 82.7 s | 79 MB | rendered in 22:22
  lane token `stmo` = still_motion -- the video lane that CONSUMES a still
  4 stills minted by sd15, zero image errors, memory ended at 86% free
```

`[OTR.image.sd15] minted still 768x432 seed=... steps=20 cfg=7.00
sampler=dpmpp_2m/karras ckpt=v1-5-pruned-emaonly-fp16` -- and 768x432 IS the
`_fit_native` clamp doing its job on an 832-wide request, so the two-headed
failure mode never had a chance to appear.

| engine | status |
| --- | --- |
| `sd15` | **PASS -- proven end to end, published to `otr/obs/`** |
| `still_motion` | **PASS** -- consumed the stills, encoded, published |
| `z_image_turbo` | FAIL on Mac in every variant (section above) |

**What this unblocks.** `ltx_8gb` (LTX 0.9.8) passed its engine gate on `mps`
and then failed only because no still could be minted. With `sd15` supplying
stills, image-to-video on Apple Silicon is testable for the first time.

**One cosmetic thing to note, not a fault:** the inter-beat reclaim logs
`free_gb_after=nan` on Mac. That is `_vram_snapshot` reading CUDA-only memory
APIs, exactly as recorded in the 2026-09-07 portability sweep -- the reclaim
itself ran fine (`unload_llm, _unload_bark, gc.collect, soft_empty_cache`), only
its telemetry is blind. Harmless; the shared-resolver fix for it was
deliberately deprioritised as tidiness rather than capability.


---

## 2026-09-08 -- ALL THREE BEAT CLASSES on local video diffusion

```
otr/obs/the_weight_of_the_dead_wifes_phantom_20260908_030426__sbke__lx8g__unk__koko__sspr__q354b__sa3_final.mp4
  h264 1920x1080 + aac | 108.0 s | 81 MB | rendered in 1:07:27
  ALL THREE video lanes = ltx098_low_video (announcer, music, character)
  ALL THREE image models = sd15
  12 stills minted (3 lanes x 4 beats) | ZERO errors
```

This is the heaviest configuration the pack has: every beat mints a still AND
runs video diffusion, where the shipped canonical gives two of three roles to
free procedural visualizers.

**Cost on a 16 GB Mac:** it swapped to ~14 GB and held 27% free, and it took
1:07:27 against 22:22 for the single-lane still_motion run. It completed with no
errors, but this is the ceiling rather than a comfortable setting.

### Apple Silicon engine status, consolidated

| engine | kind | status |
| --- | --- | --- |
| `sd15` | image | **PROVEN** -- 12 stills across 3 lanes, no errors |
| `ltx_8gb` (LTX 0.9.8) | video diffusion | **PROVEN** on all three beat classes |
| `still_motion` | video (still-consuming) | **PROVEN** |
| `viz_mxc_cpu` / `viz_green` / `viz_camera` | video (procedural) | **PROVEN** (shipped canonical) |
| `kokoro`, `stable_audio_3` | audio | **PROVEN** |
| `musicgen`, `bark` | audio | **PROVEN on mps** (bark's row was wrong) |
| `animatediff15_v3_*` | video diffusion | **BLOCKED, not failed** -- needs the ComfyUI-AnimateDiff-Evolved node pack plus a domain-adapter LoRA; never reached its own code. Not a Mac issue |
| `z_image_turbo` | image | **FAIL** -- too large at bf16, `aten::_int_mm` unimplemented at int8 |
| `viz_mxc_mandala` | video | **BLOCKED** -- pycairo has no macOS wheel |

**Yesterday this platform had procedural visualizers and nothing else.** It now
has a local image engine and a local video-diffusion engine, and neither needed
a code fix to the engines themselves -- only honest `device_backends` rows and,
for LTX, a still it had never been able to obtain.
