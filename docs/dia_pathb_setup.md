# Dia Path-B setup (isolated-venv character-voice sidecar)

`dia` (Nari Labs `Dia-1.6B-0626`, Apache-2.0 -> COMMERCIAL-CLEAN) is an opt-in
clone voice and the recommended commercial-safe character engine (unlike the
bilibili-licensed IndexTTS2 default). It runs OUT OF PROCESS in an isolated venv
("Path B"): on Blackwell (RTX 5080, sm_120) Dia needs torch 2.8 nightly cu128
(Nari issue #26), which would conflict with the main torch-2.10 / cu130 venv.
`nodes/_otr_audio_engines/eng_dia.py` drives a subprocess worker over stdin/stdout
JSON -- ZERO shared torch. Fails CLOSED with a named error (C-7) if a piece is
missing. char_voice only this pass (no Dia announcer yet).

## Required pieces
1. **Isolated venv** with torch 2.8 nightly cu128 + `dia` (from GitHub).
   Box default: `C:\Users\jeffr\Documents\ComfyUI\dia\.venv`.
   Build it: `scripts\_otr_dia_install.ps1`.
2. **Model weights** (`Dia-1.6B-0626` + Descript Audio Codec) download on the
   first `from_pretrained` (HF cache); no manual weights step.
3. **The OTR worker bridge:** `scripts/_otr_dia_worker.py` (THIS repo, committed).

## Enable it (opt-in; needs a ComfyUI RESTART to load env)
- `setx OTR_ENABLE_DIA 1`  (required -- flag-gated)
- `setx OTR_DIA_VENV  "...\dia\.venv\Scripts\python.exe"`  (optional; box default otherwise)
- `setx OTR_DIA_WORKER "...\ComfyUI-OldTimeRadio\scripts\_otr_dia_worker.py"`  (optional)
- `setx OTR_DIA_MODEL "nari-labs/Dia-1.6B-0626"`  (optional; the default)

## Worker protocol (matches eng_dia.py)
Launched `python _otr_dia_worker.py --model <hf_id>`.
- ready:    one JSON line `{"ready": true}` (else `{"ready": false, "error": ...}`)
- request (stdin):  `{"text","ref_clip","ref_transcript","seed","out_path","verbose"}`
- response (stdout): `{"ok": true, "out_path": <wav>, "sample_rate": 44100}`
- stop:     `{"stop": true}`
Same fd1 -> fd2 discipline as the indextts2 worker. Each per-line render is one
`[S1]` turn; `save_audio` writes the WAV (soundfile fallback for a raw-array build).

## Voices + the transcript wrinkle
Dia is zero-shot: it reuses the CC0 bank (36 `engine=dia` `dia_*` char rows mirror
the indextts2 refs). Dia's BEST clone prepends the transcript of the reference
clip; the CC0 refs ship without transcripts, so the official path here is
**audio_prompt-only** (condition on the clip, no transcript). Optional quality
upgrade: add `config/dia_ref_transcripts.json` keyed by reference WAV BASENAME
(e.g. `{"vz_caro_davy.wav": "the spoken words in that clip"}`) -- the adapter
sends the matching transcript and the worker prepends it, no code change.

## License
Engine: Apache-2.0 (commercial-clean). With the CC0 refs the Dia lane is
commercial-clean end to end -- the intended path for shipping films.

## Verify-at-build (RTX 5080, sm_120)
Confirm the dia venv's torch 2.8 nightly cu128 runs on sm_120 (install smoke
prints cuda=). Confirm the exact `Dia.generate()` signature + that audio_prompt-only
clone quality is acceptable; if not, add the transcript map. Evaluate Dia2
(released 2025-11-19) as a later swap -- target 0626 for now (proven, documented).
