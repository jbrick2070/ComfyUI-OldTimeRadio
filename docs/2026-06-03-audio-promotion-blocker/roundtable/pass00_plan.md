# OTR audio-engine PROMOTION blocker -- harden the path (2026-06-03)

## Objective
Promote the OTR (ComfyUI custom-node) audio stack per the EXECUTION-PLAN engine
matrix: shipped out-of-box defaults should become **character voice = IndexTTS2
(#1) > Chatterbox (#2) > Bark (#3)**; **music = Stable Audio 3 (#1, ComfyUI-native)
> MusicGen (#2) > Stable Audio Open (#3)**; announcer = Kokoro|Chatterbox. Today
the live defaults are Bark/Kokoro/MusicGen and they render clean end-to-end. We
want the newer engines as defaults WITHOUT breaking the working stack.

## The hard stack (must protect -- it currently renders bug-free)
- Windows 11, single RTX 5080 Laptop, 16 GB VRAM, **sm_120 (Blackwell)**.
- **torch 2.10.0+cu130, torchaudio 2.10.0+cu130, CUDA 13.0** (bleeding-edge
  nightly; sm_120 support needs this).
- **numpy 2.4.4, transformers 5.5.0**, ComfyUI Desktop on this venv.
- xformers / flash_attn / tensorrt all ABSENT (kept absent on purpose -- brittle
  on sm_120 and break byte-identical determinism).
- Hard rules: 100% local + offline-first at RUN time (no network during execute),
  byte-identical render-twice determinism is a release gate, VRAM peak ceiling
  14.5 GB, SFW. The audio engines are in-process ComfyUI node adapters: today an
  adapter does `import <lib>` INSIDE the ComfyUI python process.

## The blocker (measured today, non-destructively via pip --dry-run + git reqs)
Installing the new VOICE engines into the main venv would DOWNGRADE the stack and
brick ComfyUI + the OTR writer (which uses transformers 5.5):

| engine | source | pins torch | pins numpy | pins transformers | other |
|---|---|---|---|---|---|
| chatterbox-tts 0.1.7 | PyPI | ==2.6.0 (generic, NO sm_120) | <2.0 (->1.26.4) | ==5.2.0 | gradio/diffusers tree |
| IndexTTS2 (indextts 2.0.0) | git only | ==2.8.* (cu128) | ==1.26.2 | ==4.52.1 | keras2.9, librosa0.10, optional deepspeed |
| Stable Audio Open (stable-audio-tools) | PyPI | won't resolve | -- | -- | old pinned pandas fails to build |

- torch 2.6 cannot drive sm_120 at all; torch 2.8-cu128 MIGHT, but the numpy 2->1
  and transformers 5->4 downgrades alone break ComfyUI (built for numpy 2.x) and
  the OTR writer. Verdict: neither voice engine can be a main-venv default here.
- Stable Audio 3 is DIFFERENT: ComfyUI-NATIVE (`Comfy-Org/stable-audio-3`,
  ComfyUI >= v0.22.0). It uses ComfyUI's OWN torch -- **no pip dep, no conflict**.
  It needs only: ComfyUI new enough, the native SA3 nodes present, HF-gated
  weights (license + HF_TOKEN), and a not-yet-written `eng_stable_audio_3.py`
  adapter. This is the one conflict-free promotion.

## Options currently on the table
- **A. Keep Bark voice default; promote only SA3 music.** Safe, ships now. Loses
  the IndexTTS2 voice-quality bump.
- **B. Isolated-venv SIDECAR for IndexTTS2 (and/or Chatterbox).** A separate venv
  (own torch 2.8-cu128 / numpy 1.26 / transformers 4.52) runs the engine as a
  subprocess; ComfyUI passes text in, gets audio back via IPC. Main stack never
  touched. New architecture -- current adapters are in-process.
- **C. Downgrade the whole main venv to the engines' pins.** Un-does cu130 +
  breaks the writer. Rejected; listed for completeness.
- **D. SA3-native music now (headless adapter, fail-closed until weights), defer
  voice.** A subset of A.

## What we want from this panel (think outside the box)
1. Are we MISSING a cleaner path to IndexTTS2/Chatterbox-quality voice on a
   torch-2.10+cu130 / numpy-2.x / sm_120 box? e.g. ONNX/torch-export the voice
   model + run under onnxruntime-gpu with NO torch pin; the engines' own newer/
   un-pinned releases; vendoring just the inference module and relaxing pins; a
   model server in a container; CPU-only sidecar for voice; or a different SOTA
   open voice engine that is torch-2.10/numpy-2 compatible AND commercial-clean.
2. If the SIDECAR (B) is right: the LEAN, robust IPC design for a per-line TTS
   sidecar that preserves byte-identical render-twice (engine must bind an
   external torch.Generator), stays offline at run time, tears down to respect
   14.5 GB, and fails closed with a NAMED error when its venv/weights are absent.
   Sharp edges? (per-line startup cost, model residency across lines, seed
   plumbing across the process boundary, Windows subprocess quirks.)
3. SA3-native: any gotcha making `Comfy-Org/stable-audio-3` a poor default on a
   single 16 GB Blackwell card (VRAM with the video branch co-resident, the HF
   license/token UX, determinism of the native sampler)?
4. Sequencing/risk: given "remove legacy in lockstep" + "F-pilot gates
   promotion", what is the smallest correct first sprint, and what will we regret?

## Invariants the answer must NOT break
Do not propose anything that: adds a runtime network call during execute; pulls
xformers/flash_attn/tensorrt; swaps/downgrades the main-venv torch/numpy/
transformers; breaks byte-identical render-twice; exceeds 14.5 GB VRAM peak;
introduces a paid/cloud runtime dependency; or makes a box-fresh ComfyUI install
fail to load (INPUT_TYPES import-safe, no module-scope IO).
