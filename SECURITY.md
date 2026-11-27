# Security

What this pack does on your machine, what it never does, and how to report a
problem. Every statement here describes the code in the version that carries
this file, and each one names where to check it.

ComfyUI's own HTTP API has no login, so anything a node returns, raises or
logs can be read by whoever can reach the server. The pack is written with
that in mind.

## What the pack never does

- **It registers no HTTP route.** Nothing here adds an endpoint to ComfyUI's
  server (`tests/test_pack_registers_no_http_route.py`). ComfyUI serves the
  pack's browser files from `js/`, as it does for every pack that has them.
- **It runs no `eval` or `exec`, and it calls no `pickle`.** The pack's
  direct `torch.load` calls (the Kokoro voice files) pass
  `weights_only=True`. No load turns `trust_remote_code` on: the writer's
  loads pass `False`, and the Bark and MusicGen loads leave it unset, which
  is off. Most weights are safetensors. The others, and what opens each: the
  Kokoro model by the `kokoro` library, or its ONNX copy by `onnxruntime`;
  Bark's `.bin` by `transformers`; the ESRGAN upscaler's `.pth`, checked
  against a pinned SHA-256 first, by `spandrel`; the Silero voice-activity
  model, a TorchScript file that comes inside the `silero-vad` package, with
  `torch.jit.load`; the Whisper speech-detection model, a CTranslate2 file,
  by `faster-whisper`; and the AnimateDiff motion modules (`.ckpt`, `.pth`)
  by the AnimateDiff-Evolved pack, not this one.
- **It installs nothing at run time.** No `pip`, no wheel, no archive is
  downloaded and unpacked or imported. Python dependencies are the ones in
  `requirements.txt`, installed by ComfyUI Manager or by you.
- **It runs no shell.** In the package the registry distributes, every
  process the pack starts goes through one module
  (`nodes/_otr_shared/proc.py`) that takes an argument list, refuses
  `shell=True`, and refuses any program not on its list.
- **It sends no telemetry.** It collects nothing about you and reports nothing
  to the author. It sets `HF_HUB_DISABLE_TELEMETRY=1` if you have not set it.
- **It does not go looking for credentials.** No token or key is read at
  startup, and none is copied into the process environment
  (`tests/test_hf_token_is_never_read_from_the_registry.py`). The voice
  files fetched at startup are requested with no token
  (`tests/test_kokoro_voice_prefetch.py`).
- **It does not change other packs.** At startup it replaces no module of
  another package and leaves nothing of its own on the import path
  (`tests/test_prestartup_leaves_the_process_alone.py`).

## Programs it starts

| Program | Why | When |
|---|---|---|
| `ffmpeg`, `ffprobe` | every encode, mux, caption burn and probe | every episode |
| `git` | the commit stamp in an episode's record | when the pack is a git checkout |
| `nvidia-smi` | GPU name and driver version | when torch cannot answer |
| `vm_stat`, `sysctl` | free and total memory | macOS only |
| `blender` | the `mesh_stage` video engine | only if you select that engine and point `OTR_BLENDER_EXE` at Blender |
| `python` | the Chatterbox voice engine's worker, run by the interpreter in that engine's own folder (`chatterbox`, in ComfyUI's root, outside this pack) | only if you select that engine and that folder exists; `OTR_CHATTERBOX_VENV` names a different interpreter |

The ffmpeg used is the one on your machine or the one the `imageio-ffmpeg`
dependency ships. No node takes a program path as an input.

The IndexTTS2 voice engine is in the GitHub tree and is left out of the
registry package. It starts its own worker process directly, outside the
module above, and only if you select that engine.

## Network access

- **Hugging Face (`huggingface.co`).** Model weights are downloaded when a run
  needs them and they are not on disk, each from a repository named in the
  code. Pinned to one commit, with every downloaded file checked against
  its SHA-256: most image and video weights (`nodes/_otr_visual_assets.py`),
  and Kokoro's model, config and voice files, its ONNX export and the
  Whisper speech-detection model (`nodes/_otr_model_pins.py`). A pinned
  file already on disk is used with no network call; Kokoro's voice files
  and ONNX model, once in the models folder, are used as they are, and a
  copy found in the Hugging Face cache is checked first. From the
  repository's current revision: the writer
  models, Bark, MusicGen and the AnimateDiff motion modules. The Kokoro voice
  files, and on Python 3.13 the Kokoro ONNX model, are fetched once when
  ComfyUI starts so the default voice works on a fresh install; set
  `OTR_SKIP_KOKORO_PREFETCH=1` or `HF_HUB_OFFLINE=1` to stop that.
- **News feeds.** Only when you pick a source bank that reads feeds. The
  reader (`nodes/_otr_feed_fetch.py`) uses https only, refuses any address
  that is not publicly routable, connects to the address it validated, and
  caps size and time.
- **Cloud lanes (off by default).** OpenRouter, the Google API and Comfy
  credits are used only when you enable a cloud slot in the workflow and
  supply your own key or sign-in. A key is sent only to its provider's own
  host and is dropped if a response redirects elsewhere. The OpenRouter and
  Comfy credits requests carry a title header naming this project, and the
  OpenRouter ones a referrer header pointing at this repository; neither
  says anything about you. A provider's result is downloaded over https, and
  only from a host whose addresses are all publicly routable. That is
  checked when the name is resolved and is not pinned for the connection
  that follows. `OTR_CLOUD_RESULT_ANY_HOST=1` turns the address check off,
  for networks that map every host to a private address. The result's URL is
  kept out of error text.

With the models on disk, no cloud slot enabled and a source bank that does not
read feeds, an episode can render with the network unplugged (set
`HF_HUB_OFFLINE=1`). Online, the Hugging Face libraries may still check cached
files against the Hub.

## Files

- **Reads.** A path that arrives through a node input is confined to ComfyUI's
  input, output and temp folders, its saved-workflow folders and the pack's
  own folders before it is opened (`nodes/_otr_paths.py`).
- **Writes.** Episodes and their working files go under ComfyUI's output
  folder, in `otr/`. Models go under ComfyUI's models folder and the Hugging
  Face cache. Short-lived scratch files (an ffmpeg error log, a probe
  sample, an assembly folder, a lock folder for GPU use) go in the system
  temp folder; `OTR_GPU_LEASE_DIR` moves the lock folder. With the Google
  lane on, a cache of that provider's model list goes in ComfyUI's user
  folder, under `old_time_radio/`. If the pack cannot resolve ComfyUI's
  output folder when it loads, its news-history file falls back to
  `.otr_state` in your home folder.
- **At startup.** If `HF_HOME` is not set, the pack picks one cache folder for
  Hugging Face downloads and checks that it can write there. On Windows that
  reads the `HF_HOME` and `LongPathsEnabled` registry values and no other.

## What it shows and records

- **Errors, log lines and on-canvas text carry no absolute paths.** A path is
  cut to its file name before it leaves a node, in the whole exception chain,
  in the pack's log records and in the text a node draws on the canvas
  (`nodes/_otr_error_text.py`). The type name ComfyUI reports beside an
  error is cut the same way (`tests/test_error_type_names_carry_no_path.py`).
  No value a node declares to ComfyUI is an absolute path
  (`tests/test_object_info_has_no_absolute_paths.py`).
- **What ComfyUI adds by itself.** When any node fails, ComfyUI attaches a
  traceback that names each source file by its full path, its own files
  included, and echoes the failing node's inputs. That is ComfyUI's behavior
  for every pack, and this pack does not alter it.
- **An episode's record.** Each episode writes a ledger under the output
  folder. It names the files of that episode by absolute path, because later
  steps open them, and it stamps the software it ran on: ComfyUI's version,
  this pack's version, Python, torch, CUDA, the GPU and its driver, and the
  operating system. It does not list your other custom node packs unless you
  start ComfyUI with `OTR_ENV_FINGERPRINT_PACKS=1`. Whether the ledger is
  removed once the episode is published depends on the workflow's cleanup
  setting, and a cleanup that is refused keeps it. Read a ledger before you
  attach it to a public issue.
- **The credits.** The finished video's credits include a SYSTEM block naming
  the machine's CPU, GPU, memory, operating system and library versions. It
  carries no host name and no account name. It is part of the video you
  publish, so know that it is there.

## Credentials

API keys for the OpenRouter and Google lanes come from environment variables
or from a key file you create (`api_key.location.example` shows the layout).
The Comfy credits lane uses the sign-in ComfyUI itself hands to the node; the
pack stores nothing for it. A key is read when a cloud call is made, attached
to that provider's host only, and kept out of the pack's error text and log
lines. Never paste a key or a token into a workflow widget: a value saved in
a workflow travels with it.

A Hugging Face token is not needed for any model the pack ships a choice for.
If you have one (`HF_TOKEN` in the environment that started ComfyUI, or your
`hf auth login`), the writer, image and video model downloads send it to
huggingface.co with the download, which is what lets a gated model of your
own work. The startup voice fetch and the speech-detection model fetch send
none. Other loads go through the Hugging Face libraries, which attach a
saved login to their own requests by default.

## Third-party code and data inside the pack

- `nodes/_otr_audio_engines/_misaki/` -- phonemizer modules from misaki
  (Apache-2.0).
- `nodes/_otr_audio_engines/_spacy_model/` -- spaCy's small English pipeline
  `en_core_web_sm` (MIT), a byte-for-byte copy of the published release.
- `nodes/fonts/` -- DejaVu Sans and DejaVu Sans Mono (the DejaVu licence),
  Noto Sans SC (SIL Open Font License), and a Devanagari and Latin font
  merged from two unmodified Noto fonts (SIL Open Font License).
- `config/source_banks/` -- public-domain story texts, and Shakespeare
  sources and translations, that the adaptation lanes read.

The misaki and spaCy folders each carry a `PROVENANCE.json` naming the source
and the SHA-256 of every file, and their licence files (spaCy's sit inside
the model folder, as published). The fonts folder carries each font's
licence. Each public-domain story text and each Shakespeare source scene has
a provenance record beside it, and
`config/source_banks/shakespeare/translations/manifest.json` names the
translator, the edition and the source of every translated scene.

The GitHub tree also holds files the registry package leaves out, among them
`assets/` (art, and the Inter Display font with its SIL Open Font License)
and `contest/`, the author's own companion app.

## Reporting a problem

Please report a security problem privately through GitHub's
"Report a vulnerability" on this repository's Security tab. For anything that
is not sensitive, open an issue. Say which version you are on and how to
reproduce it.
