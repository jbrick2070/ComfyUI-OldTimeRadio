# Chatterbox live test on the RTX 5080 -- results (2026-06-05)

Ran a live chatterbox install + GPU render on Jeffrey's desktop while he was away.

## PROVEN (live, on the box)
- **Isolated venv installed**: `C:\Users\jeffr\Documents\ComfyUI\chatterbox\.venv`
  via the main 3.12 Python. `pip install chatterbox-tts soundfile` -> chatterbox-tts
  0.1.7 (with torch 2.6.0 **CPU-only** from PyPI -- no CUDA).
- **Blackwell torch override**: installed `torch 2.11.0+cu128` + `torchaudio
  2.11.0+cu128` into the venv (PyTorch cu128 index). GPU verified:
  `NVIDIA GeForce RTX 5080 Laptop GPU`, capability **(12, 0) = sm_120**, CUDA op
  OK. chatterbox-tts imports cleanly under torch 2.11 (its `torch==2.6.0` pin is
  metadata-only; a pip dependency-conflict warning is expected + harmless).
- **Worker renders end-to-end on the GPU**: the OTR chatterbox worker loaded the
  model (first run downloaded it) and rendered a clip. **RENDER_OK frames=91200
  sr=24000 dur=3.80s.**
- **Full OTR adapter path renders**: with `OTR_ENABLE_CHATTERBOX=1`, the real
  adapter (main venv) spawned the isolated worker, rendered, and returned the
  AUDIO dict: **ADAPTER_OK shape=(1, 96000) sr=24000 dur=4.00s.** Flag gating,
  cross-venv Path-B bridge, and soundfile load all confirmed.

## BUG FOUND + FIXED (live)
- `torchaudio.save` fails in the sidecar venv: **torch 2.11 routes save through
  torchcodec** (not installed) -> "TorchCodec is required". Fixed:
  `scripts/_otr_chatterbox_worker.py` now saves via `soundfile.write` (committed
  e5d22a4). Re-validated green after the fix.

## NOT done (externally gated -- needs Jeffrey at the keyboard)
The **full in-ComfyUI-graph 30-word render** did not run. ComfyUI Desktop is an
Electron app; its python server cannot be (re)started from this non-interactive
shell, and computer-use needs an on-screen approval Jeffrey wasn't there to give.
The running server also could not see the new flag/code without an app relaunch.

## State left on the box (clean)
- ComfyUI Desktop is CLOSED (0 processes, :8000 free) -- a normal pre-launch state.
- `OTR_ENABLE_CHATTERBOX=1` and `OTR_CHATTERBOX_VENV=...chatterbox\.venv\Scripts\
  python.exe` are persisted in the USER environment (HKCU; set via the .NET API
  because `setx` was not on this shell's PATH).
- All code committed + pushed (HEAD e5d22a4 on v2.0-alpha).

## To finish the in-graph render (one sitting, ~2 min of clicks + render time)
1. Open **Comfy Desktop** normally (it now reads the flag + loads the new code).
2. Confirm `http://127.0.0.1:8000` is up.
3. Run the ready smoke driver (queues the 30-word workflow with engine=chatterbox
   on node 81, voice_bank=default on node 80):
   `& "C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe" "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\scripts\_otr_chatterbox_smoke.py"`
4. Watch `output\otr\episodes\...` + the node-81 render log; expect chatterbox
   character voices, `Bark loaded = 0`.
