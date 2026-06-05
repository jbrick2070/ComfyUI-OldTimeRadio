# IndexTTS2 Path-B setup (isolated-venv character-voice sidecar)

`indextts2` is the default `char_voice`, but it runs OUT OF PROCESS in an isolated
venv ("Path B") because it hard-pins old torch that would brick the main
torch-2.10 / cu130 Blackwell venv. `nodes/_otr_audio_engines/eng_indextts2.py`
launches a subprocess worker and talks to it over stdin/stdout JSON. The lane
fails CLOSED with a named error (C-7: absent default-ON model -> named error,
never auto-fetch) if any piece below is missing.

## Required pieces
1. **Isolated venv** with the `indextts` package + a Blackwell-capable torch
   (>= 2.7 cu128 for sm_120). Here: the upstream clone at
   `C:\Users\jeffr\Documents\ComfyUI\index-tts\.venv` (`uv sync`).
2. **Weights dir** with `config.yaml` + `gpt.pth` + `s2mel.pth` + the qwen emo
   model + `w2v-bert-2.0` (under `checkpoints/hf_cache`). Here:
   `C:\Users\jeffr\Documents\ComfyUI\index-tts\checkpoints`.
3. **The OTR worker bridge:** `scripts/_otr_indextts2_worker.py` (THIS repo,
   committed). It was scratch-only before and went missing, which silently broke
   the default char voice -- committing it is the root-cause fix.

## Env vars (set as USER env; ComfyUI must restart to load them)
- `OTR_INDEXTTS2_VENV`   = `...\index-tts\.venv\Scripts\python.exe`
- `OTR_INDEXTTS2_DIR`    = `...\index-tts\checkpoints`
- `OTR_INDEXTTS2_WORKER` = `...\ComfyUI-OldTimeRadio\scripts\_otr_indextts2_worker.py`

## Worker protocol (matches eng_indextts2.py)
Launched `python _otr_indextts2_worker.py --model-dir <ckpt> [--fp16]`,
cwd = parent of model_dir.
- ready:    one JSON line `{"ready": true}` (else `{"ready": false, "error": ...}`)
- request (stdin):  `{"text","ref_clip","emo_vector":[8],"emo_alpha","seed","out_path","verbose"}`
- response (stdout): `{"ok": true, "out_path": <wav>, "sample_rate": 22050}`
- stop:     `{"stop": true}`
The worker redirects fd1 -> fd2 so model / torch / tqdm prints can never corrupt
the JSON channel; the protocol JSON is written to the saved real-stdout fd.

## License
IndexTTS2 is under the bilibili Model Use License -- NON-COMMERCIAL / personal use.

## Status
Verified working on RTX 5080 (Blackwell sm_120) 2026-06-05: standalone worker
render OK (22.05 kHz WAV), and a full 30-word episode rendered end to end with
gemma-4-12b as writer -> indextts2 voice -> video.
