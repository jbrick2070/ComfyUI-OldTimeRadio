# Mac engine survey -- selected test plan

**Purpose.** The canonical renders on Apple Silicon (2026-09-07). This plan
covers what else does. Operator's framing: *"they may not work, that's okay. We
just have to document what we need to do to get it running for people."*
**So auto-install stops being a gate and becomes a property we record.**

Every row's `device_backends` comes from the shipped registries. **A declaration
is enforced before any work runs**, so a `[cuda]` row raises `EngineUnusable` on
a Mac regardless of whether the code would work -- which means some of these
tests are really "is the declaration honest?"

Status keys: **PASS** measured working / **FAIL** measured broken / **BLOCKED**
cannot run and why / **UNTESTED**.

---

## Tier 1 -- free or near-free (declares mps, or already on disk)

| engine | kind | declares | status | notes |
| --- | --- | --- | --- | --- |
| `kokoro` | TTS | cuda,cpu,mps | **PASS** | the shipped default. On py3.13 runs kokoro-onnx, CPU by design; `voice_device` is accepted and unused (PBUG-20260907-08) |
| `stable_audio_3` | music | cuda,mps | **PASS** | needs PyTorch attention on MPS or output is noise (PBUG-20260907-11) |
| `musicgen` | music | cuda,cpu,mps | **PASS** | measured 2026-09-07: mps 14.1 s vs cpu 14.7 s for 256 tokens, flatness 0.134/0.103, finite. Metal buys nothing at this size. **Unaffected by the attention bug** -- transformers model, not ComfyUI-native. Licence remains CC-BY-NC, so not a default candidate |
| `bark` | TTS | cuda,**cpu** | IN PROGRESS | `_otr_bark_lib.py:132` is a literal `cuda if available else cpu`, so mps is never selected. musicgen proves transformers models work on mps, so the declaration is likely wrong |
| `viz_green` / `viz_mxc_cpu` / `viz_camera` | video | cuda,cpu,mps | **PASS** | all three render; `viz_camera` is the operator's preference |
| `viz_mxc_mandala` | video | cuda,cpu,mps | **BLOCKED** | needs `pycairo`, which has no macOS wheel and needs libcairo headers. Declaration is right, the canonical was wrong to select it (PBUG-20260907-10) |
| `word_razzle` | video | cuda,cpu,mps | UNTESTED | cheap, no weights |
| `still_motion` / `still_flat` / `still_pan` / `still_word` | video | cuda,cpu,mps | **BLOCKED (upstream)** | they declare mps and would run -- but they CONSUME a still, and every local image engine is `[cuda]`. Only reachable with a cloud image key |

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
