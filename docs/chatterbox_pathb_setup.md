# Chatterbox Path-B setup (isolated-venv character-voice sidecar)

`chatterbox` (Resemble AI, MIT -> commercial-clean ENGINE) is an opt-in clone
voice. It runs OUT OF PROCESS in an isolated venv ("Path B") because it pins its
own torch / numpy that would brick the main torch-2.10 / cu130 Blackwell venv.
`nodes/_otr_audio_engines/eng_chatterbox.py` launches a subprocess worker and
talks to it over stdin/stdout JSON -- ZERO shared torch. The lane fails CLOSED
with a named error (C-7) if any piece below is missing.

## Required pieces
1. **Isolated venv** with `chatterbox-tts` + a Blackwell-capable torch (sm_120).
   Box default: `C:\Users\jeffr\Documents\ComfyUI\chatterbox\.venv`.
   Build it: `scripts\_otr_chatterbox_install.ps1`.
2. **Model weights** download automatically on the first `from_pretrained`
   (HF cache); no manual weights step.
3. **The OTR worker bridge:** `scripts/_otr_chatterbox_worker.py` (THIS repo,
   committed). Side-effect-free at import; spawns the model only when driven.

## Enable it (opt-in; needs a ComfyUI RESTART to load env)
- `setx OTR_ENABLE_CHATTERBOX 1`  (required -- the engine is flag-gated)
- `setx OTR_CHATTERBOX_VENV  "...\chatterbox\.venv\Scripts\python.exe"`  (optional; box default used otherwise)
- `setx OTR_CHATTERBOX_WORKER "...\ComfyUI-OldTimeRadio\scripts\_otr_chatterbox_worker.py"`  (optional)

## Worker protocol (matches eng_chatterbox.py)
Launched `python _otr_chatterbox_worker.py`.
- ready:    one JSON line `{"ready": true}` (else `{"ready": false, "error": ...}`)
- request (stdin):  `{"text","ref_clip","exaggeration","cfg_weight","temperature","seed","out_path","verbose"}`
- response (stdout): `{"ok": true, "out_path": <wav>, "sample_rate": 24000}`
- stop:     `{"stop": true}`
The worker redirects fd1 -> fd2 so model / torch / tqdm prints can never corrupt
the JSON channel. It introspects the real `generate()` signature (`supported_kwargs`)
and resamples to 24000 so the batch packs single-rate.

## Voices
chatterbox is zero-shot: it reuses the CC0 reference bank. The bank ships 36
`engine=chatterbox` char rows (`cb_*`) mirroring the indextts2 CC0 refs, plus one
announcer ref (`cb_announcer_male`). No extra download.

## License / watermark
Engine: MIT (commercial-clean). Every output carries Resemble AI's imperceptible
PerTh watermark. Effective commercial-clean = engine AND ref; with the CC0 refs
the chatterbox lane is commercial-clean.

## Verify-at-build (RTX 5080, sm_120)
Confirm the chatterbox venv's torch supports sm_120 (the install smoke prints
cuda=); if not, reinstall a cu128 torch INTO THE CHATTERBOX VENV ONLY (see the
commented line in the install script). Confirm whether `generate()` binds an
external `torch.Generator` before enabling bit_exact (G1) -- until then the
adapter keeps `supports_external_generator` off.
