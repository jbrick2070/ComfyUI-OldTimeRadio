# OTR RunPod playbook: install, run, qualify, troubleshoot

This is the **single canonical RunPod playbook**: first deployment, template,
models, HuMo/LTX 2.5/H3 boundaries, canonical qualification, unattended runs,
and the failure atlas all live here. `RUNPOD_PORTABILITY_LAB.md` is retained
only as a compatibility redirect and owns no procedure. Every claim below is
either MEASURED on real hardware and labelled so, or marked UNPROVEN.

**The diary is in git history.** This file used to record what was learned in
what order, which meant a reader met 625 lines of superseded conclusions --
starting with a section titled "Read this first" whose conclusion the file
itself called formally dead -- before reaching the procedure. Restructured
2026-08-31. To see how a fact was found: `git log docs/RUNPOD_INSTALL.md`.

**Most of this is NOT RunPod-specific.** Only playbook steps 1-2, section 2
(template spec) and the volume-speed material are particular to RunPod;
everything else -- storage roots, warming models, the index-tts build, verifying
by `/object_info`, and nearly every atlas entry -- applies to ANY fresh Linux
host. A second cloud provider gets a deploy subsection here, not a rival
document.

**Its own rule, now applied to the whole file:** this is a LIVING document. When
something changes, EDIT THE CELL. Do not append a v3.

| you want to | read |
|---|---|
| stand up a pod, start to finished episode | 1. The playbook |
| build or edit the template | 2. Template spec |
| get models onto the machine | 3. Models and assets |
| run a sweep overnight | 4. Running unattended |
| qualify HuMo, LTX 2.5, or understand H3 | 4A. Heavy-engine qualification |
| something is broken | 5. Failure atlas, by symptom |

---

## 1. The playbook -- what actually works, in order

This is the procedure.

### Step 1 — add an SSH key to the ACCOUNT, once, ever

**This is the answer to "how do we stop typing into a terminal", and none of the
HTTP routes were.** Generate a keypair on the workstation, paste the PUBLIC half
into RunPod's account -> SSH Public Keys. RunPod injects it into every pod's
`authorized_keys` **at pod creation** -- a key added to a RUNNING pod does not
reach it, so add it before deploying.

    ssh-keygen -t ed25519 -f ~/.ssh/runpod_otr -N ""
    # paste ~/.ssh/runpod_otr.pub into the account, then deploy

Everything after this is scriptable and needs no human.

**USE THE DIRECT TCP FORM, NOT THE PROXY.**

    ssh root@<ip> -p <port> -i ~/.ssh/runpod_otr      # takes commands. USE THIS.
    ssh <pod>-<hash>@ssh.runpod.io -i ...             # forces an interactive
                                                      # shell, IGNORES the command
                                                      # argument, and its PTY
                                                      # CORRUPTS long input:
                                                      # `memory.total` arrived as
                                                      # `memory.totaal`.

If only the proxy is available, have the REMOTE side fetch what it needs
(`curl` the script from the repo) rather than typing it across the PTY.

### Step 2 — deploy WITH the network volume

Select the network volume explicitly at deploy. RunPod states it *"Replaces
volume disk"*, so it does not conflict with a template's storage block -- the
earlier pods simply did not have it selected. Verify from inside:

    df -h /workspace
    mfs#euro-3.runpod.net:9421  2.0P  ...  /workspace    <- volume IS mounted
    overlay                      55G  ...  /             <- it is NOT

**CHECK WHICH GPU YOU ACTUALLY GOT.** A deploy summary saying "RTX 5090 32 GB"
produced an **RTX PRO 4000 Blackwell, 24 GB**. One call, no cost:

    GET /system_stats

### Step 3 — set BOTH storage roots, or the volume only half works

Two separate caches, and missing either wastes the drive:

| what | variable | why |
|---|---|---|
| video weights | `OTR_COMFYUI_MODELS_ROOT=/workspace/runpod-slim/ComfyUI/models` | set in the TEMPLATE so fetcher and server share one explicit persistent root; otherwise the current resolver uses `folder_paths.models_dir`, which is template-dependent and may be disposable |
| writer / voice / music | `HF_HOME=<models_root>/huggingface` | **defaults to `/root/.cache`, which is on the 55 GB CONTAINER disk and is erased on stop.** Without this the 24 GB writer re-downloads every session even with a volume attached. **Use the models root, not an invented path -- see below.** |

**The expensive half is the one that defaults to the disposable place.** The
current 16 GB+ starter's three automatic model lanes total about 32.1 GB before
writer and voice caches; the HF cache with both writers is **38 GB**.

**THE VALUE IS `<models_root>/huggingface`, AND PICKING ANY OTHER PATH COSTS
YOU THE CACHE TWICE.** That is OTR's convention, not a preference: the reference
machine runs `HF_HOME=C:\ComfyUI-Models\huggingface`, and
`nodes/_otr_hf_env.py` exists specifically to resolve and re-export it. So on a
pod the value is:

    HF_HOME=/workspace/runpod-slim/ComfyUI/models/huggingface

**This session got that wrong and paid 84 GB for it.** The template set no
`HF_HOME`, so 71 GB had already accumulated at the convention; the session then
symlinked the default cache path to a NEW directory, `/workspace/hf`, and warmed
it -- fetching Mistral-Nemo (46 GB) and gemma-4-12b (23 GB) a second time. Two
roots, both live, 69 GB of exact duplicate, and a 200 GB volume that then hit
`[Errno 122] Disk quota exceeded` mid-download. The duplication is invisible
until you `du` both roots, because each one looks correct on its own.

ComfyUI reads `HF_HOME` at start, so a cache created afterwards is invisible to
it. Symlink the default path at the convention rather than inventing a new one:

    ln -sfn /workspace/runpod-slim/ComfyUI/models/huggingface /root/.cache/huggingface

### Step 4 — provision, in one command, from the repo

The 16 GB+ starter uses IndexTTS2, whose authorized reference WAVs cannot ship
with the project. On a clean pod, use an honest two-pass provision: the first
pass clones everything and fetches the public lanes, then exits **INCOMPLETE**
at the named voice-bank boundary.

    ssh <pod> 'cat > /root/provision.sh' < scripts/otr_pod_provision.sh
    ssh <pod> 'OTR_PROVISION_PROFILE=otr_runpod_starter \
      nohup bash /root/provision.sh > /root/prov.log 2>&1 &'

After that pass, complete section 3's reference-WAV and portable-bank recipe.
It writes `/workspace/otr-config/otr-runtime.env`. The completing pass is:

    ssh <pod> '. /workspace/otr-config/otr-runtime.env; \
      OTR_PROVISION_PROFILE=otr_runpod_starter OTR_WITH_INDEXTTS2=1 \
      nohup bash /root/provision.sh > /root/prov.log 2>&1 &'

**Run it DETACHED.** A foreground SSH session dies on any client-side timeout
and takes the install with it -- this cost one full run mid-`pip install`.

The script locates the tree ComfyUI actually scans (**do not assume
`/workspace/ComfyUI`; this image uses `/workspace/runpod-slim/ComfyUI`**),
recovers `OTR_COMFYUI_MODELS_ROOT` from `/proc/1/environ` because **SSH sessions
do not inherit container env**, pins ComfyUI core and both packs, installs deps,
and routes the exact selected profile. The starter automatically fetches
`wan_ti2v_gguf`, `z_image`, and `stable_audio_3`; an incomplete voice bank makes
the command return nonzero rather than pretending the pod can render.

### Step 5 — warm the HF models BEFORE rendering

Skipping this fails the first render at **73 seconds**:

    ERROR node 1 (OTR_LedgerScriptWriter) raised _LLMTimeoutWorkflowPause

transformers fetches the writer on first use and the download outlasts the
writer timeout. **A developer box never sees this** -- its HF cache has been warm
for months -- so it only ever appears on a fresh machine, wearing the name of a
model problem.

Set an `HF_TOKEN` (a RunPod **Secret**, not a plain env var -- the console warns
"Environment variables are not encrypted") for rate limits and gated lanes.

### Step 6 — restart, verify, then render

`POST /api/manager/reboot` returns **502 and works**; poll `/system_stats` for a
200. Then the only check that proves anything:

    GET /object_info   ->   OTR_ non-zero, ADE_ non-zero, ckpt_name populated

**Install verification does not imply render capability.** A pod passed every
install check -- 25 `OTR_`, 143 `ADE_`, all models registered -- and still could
not produce a frame, because `h264_nvenc` cannot initialise in a container
without `libnvidia-encode.so.1`. Fixed in the pack (the probe now encodes a test
frame instead of grepping `ffmpeg -codecs`), but the shape of that failure is
the lesson: it landed **7-18 minutes in**, after the script and audio were done.

### What a fresh pod costs, measured

    RTX PRO 4000 Blackwell 24 GB   $0.58/hr
    RTX 5090 32 GB                 $0.69/hr
    network volume 200 GB          $0.019/hr  (~$14/mo, billed regardless)
    CUDA13/driver-570 repair       ~8 min     (only on that measured mismatch)
    starter automatic lanes        ~32.1 GB   (Wan + Z-Image + Stable Audio)
    HF warm, both writers          38 GB, once per VOLUME -- not per pod

---

## 2. Template spec

**A LIVING spec.** Edit the cell; do not append a version. What matters is what
a template needs TODAY.

### Config

| field | value | why |
|---|---|---|
| Region | **EU-RO-1** | where GPUs actually schedule; US-CA-2 never gave one |
| Network volume | mounted at `/workspace` | weights re-download every boot without it. Selecting a network volume REPLACES the volume-disk setting -- do not set both |
| Container disk | **70 GB tested** | pack, torch deps and a native index-tts venv land here, not on the volume |
| Effective cgroup RAM | **at least 100 GiB for the heavy-engine lab** | read the cgroup limit, not `free`; this is conservative rental headroom, not an engine minimum |
| Free model-volume space | **at least 150 GiB for the heavy-engine lab** | complete weights, HF caches, voices, images, and receipts must coexist |
| HTTP port | `8188` **ComfyUI** | the only load-bearing port; everything drives through it |
| HTTP port | `8888` JupyterLab | browser shell |
| HTTP port | `8080` FileBrowser | optional; returned 401 all session |
| HTTP port | `19123` gotty web terminal | optional. SSH on 22 is the better path -- the web terminal is what forced the copy-pasting the account SSH key replaced |
| TCP port | `22` **SSH** | the frictionless path; see playbook step 1 |
| `--highvram` | **NOT in a shared template** | right on 31.4 GiB, WRONG on 8 GB |

### Environment -- all four

| name | value |
|---|---|
| `OTR_COMFYUI_MODELS_ROOT` | `/workspace/runpod-slim/ComfyUI/models` |
| `HF_HOME` | `/workspace/runpod-slim/ComfyUI/models/huggingface` |
| `HF_TOKEN` | `{{ RUNPOD_SECRET_HF_TOKEN }}` |
| `PYTHONUNBUFFERED` | `1` |

**Set `OTR_COMFYUI_MODELS_ROOT` explicitly even though the current resolver has
a safe fallback.** Without it, `_models_root()` uses ComfyUI's
`folder_paths.models_dir`; that avoids the historical literal-Windows-path bug,
but the location remains template-dependent and may be disposable or different
from the mounted volume you intend to preserve.

**`HF_HOME` must be the MODELS ROOT, not a path of your choosing.** That is
OTR's convention: the reference machine runs
`HF_HOME=C:\ComfyUI-Models\huggingface`, and `nodes/_otr_hf_env.py` exists to
resolve and re-export it. Any other value does not relocate the cache, it ADDS
one -- see the atlas entry "quota exceeded on a half-empty volume".

**`HF_TOKEN` goes in as a SECRET reference, never the token itself.** RunPod's
console states plainly that environment variables are not encrypted. Create the
secret under the account; its detail page prints the exact
`{{ RUNPOD_SECRET_<NAME> }}` string -- the secret's name upper-cased behind that
prefix. Only gated repos need it: writer, voices, music, the haunted lane,
`minimax_h3` and `wan_ti2v` are all ungated. For H3, "ungated" means only
that Hugging Face does not require a token; it grants no authorization. OTR
never selects H3 in public provisioning.

**SSH keys and secrets both inject at POD CREATION.** Adding either to a running
pod does not reach it; redeploy. (A token can still be delivered to a live pod --
section 3.)

### Start command

**Leave it blank.** The image's `/start.sh` brings up ComfyUI and Jupyter
correctly, and provisioning is a separate detached step (playbook step 4) rather
than an entrypoint override.

If you do override it, it REPLACES the entrypoint and must end by exec'ing the
image's real start -- get that wrong and the pod boots and serves nothing,
silently. Use `python3`, never `python`: this image has no `python` on PATH.

### Set-up cost, MEASURED

    restart from EXITED to serving ComfyUI      16 s
    model download to the volume                ~36 GB in minutes (datacenter link)
    RTX PRO 4000 Blackwell 24 GB                $0.58/hr
    RTX 5090 32 GB                              $0.69/hr
    network volume 200 GB                       ~$0.019/hr, billed regardless

A 200 GB volume cannot hold every lane. 400 GB, with the cache stored once, is
comfortable.

### What is actually PROVEN on rented hardware, and what is not

Read this before concluding the pod works.

    PROVEN   animatediff15_v3_haunted_video, 1 act, 8 clips
             RTX PRO 4000 Blackwell 24 GB
             2058 s (34.3 min), VRAM peak 15,990 MB, published to obs
             writer gemma-4-12b, voices kokoro, music musicgen

    PROVEN   ltx_8gb, 1 act, 9 beats / 1100 frames
             RTX A4500 20 GB, Ampere sm_86
             1925 s, 63.8 s final 1920x1080 h264+aac, obs_publish OK

    PROVEN   wan_ti2v, 1 act
             RTX A4500 20 GB, Ampere sm_86
             3949 s, RESULT SUCCESS, obs_publish OK

    NEGATIVE 2026-09-01: RTX 4090 24 GB, LTX 2.5 high-video canonical episode.
             CUDA execution, all weights/packs, IndexTTS, Stable Audio,
             image dispatch, canonical validation, and queue submission passed.
             Node 92 decode GPU-OOMed after 39:56 at the shipped 1664x960
             two-stage canvas; the 116.42 GiB cgroup recorded no OOM event.

    UNPROVEN on a pod: HuMo and a completed LTX 2.5 canonical episode.
             H3 is excluded from this operator's cloud policy rather than
             classified as a GPU refusal.

AnimateDiff SD1.5 remains the 8 GB floor, while the A4500 receipts prove two
meatier public lanes on rented Ampere: LTX-2b and Wan 2.2 TI2V. Those receipts
do not automatically transfer to LTX 2.5, HuMo, a different RAM cap, or a new
starter combination. Qualify each exact profile and tuple rather than treating
"RunPod works" as one global fact.

---

## 3. Models and assets

Sections 1-2 are about standing a pod up. This one is about filling it, which
turned out to be the harder half.

### THE REAL PORTABILITY GAP: engines name FILES, not SOURCES

The runtime-downloaded audio engines carry their own Hugging Face repo ids, but
most video and image engines resolve a FILENAME and know nothing about its
source. The reference machine once hid that gap because manually placed files
had always been present. Today `otr_fetch_lane_weights.py` owns closed public
lanes and `otr_provision.py` owns exact gated/manual tiers; a filename without
one of those provenance owners remains an install gap.

`docs/MODEL_ASSET_INDEX.md` (generated by `scripts/otr_asset_index.py`) now
answers "to use engine X, download Z". Closing a lane means resolving its files
on the Hub, verifying repo AND path AND size, reading the destination folder off
the engine's own resolver, and adding a row to the fetcher.

**Do not guess a repo to close a lane faster.** HuMo 14B is now closed by the
five-file, commit-pinned, SHA-256-verified `humo` lane; it downloads the exact
Kijai `...scaled_KJ` UNET the engine resolves. HuMo 1.7B stays separate and its
exact rename/destination is printed by
`python scripts/otr_provision.py --profile otr_w45_humo_1_7b --list`. A 1.7B
profile never downloads the 14B DiT or LoRA.

### THE HF CACHE CAN EXIST TWICE, AND EACH COPY LOOKS CORRECT

`HF_HOME` **must** be `<models_root>/huggingface` -- see section 2. Setting it
anywhere else does not relocate the cache, it ADDS one:

    /workspace/hf/hub                       84 GB   <- the invented path
    <models_root>/huggingface/hub           71 GB   <- the convention

Mistral-Nemo (46 GB) and gemma-4-12b (23 GB) were stored in both, byte for byte.
Nothing warns you: each root is internally valid, the server reads whichever
`HF_HOME` points at, and the waste only appears if you `du` both.

**How it surfaced was misleading, which is the part worth remembering.** The
symptom was `[Errno 122] Disk quota exceeded` partway through a lane fetch, on a
200 GB volume showing 185 GB used. That reads as "the volume is too small" and
invites buying more storage. It was "the cache was written twice". Merging the
roots reclaimed **70 GB** and took minutes.

Merge, never move -- the two roots are not identical. Verify before deleting:
A held gemma-4-E2B that B did not, A's musicgen was more complete than B's, and
B's gemma-4-12b was short one small blob. A blind `mv` either direction loses
something.

    cp -an /workspace/hf/hub/. <models_root>/huggingface/hub/

**`rsync` is NOT on `runpod/comfyui:cuda13.0`** -- the first merge attempt failed
on that, and the script's own "every model in A exists in B" check refused to
delete anything. Write that check; it is the difference between a failed step and
a lost model.

### VOICE-CLONING ENGINES ARE A SEPARATE PROJECT, NOT A DOWNLOAD

`indextts2`, `chatterbox` and `dia` shell out to their own Python interpreter.
That is deliberate -- their dependencies conflict with ComfyUI's -- and it means
they cannot be bundled into the pack without reintroducing the clash the
isolation prevents.

`indextts2` measured on the reference machine:

    index-tts/          18.93 GB
      checkpoints/      11.06 GB   portable, fetch on the target
      .venv/             7.78 GB   39,759 files of WINDOWS wheels -- rebuild

It also needs a **registered portable voice bank**, not merely WAVs copied into
a directory. `scripts/otr_make_portable_voice_bank.py` takes two distinct,
authorized PCM speech WAVs (male and female, each at least one second),
preserves every shipped non-Index row,
replaces the unavailable operator-local Index rows, copies the references under
`<models_root>/TTS/refs/indextts2/`, and writes their exact SHA-256 values. The
bank's absolute path must be exported as `OTR_VOICE_REFERENCE_BANK` for both
provisioning and the ComfyUI launch. The complete command is below; this file
is its sole documentation owner.

**This is what stops the w45 campaign on a fresh machine:** those profiles cast
`indextts2`, so every lane fails identically at `OTR_BatchCharacterVoices`,
roughly nine minutes in, for a reason that has nothing to do with the video lane
being measured. The haunted profile works because it pins `kokoro`.

### Sizes, read from the Hub rather than estimated

    ltxv-2b-0.9.8-distilled.safetensors      6.34 GB  Lightricks/LTX-Video
    t5xxl_fp16.safetensors                   9.79 GB  comfyanonymous/flux_text_encoders
    Wan2.2-TI2V-5B-Q5_K_M.gguf               3.81 GB  QuantStack/Wan2.2-TI2V-5B-GGUF
    umt5-xxl-encoder-Q5_K_M.gguf             4.15 GB  city96/umt5-xxl-encoder-gguf
    wan2.2_vae.safetensors                   1.41 GB  Comfy-Org/Wan_2.2_ComfyUI_Repackaged
    Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ    17.89 GB  Kijai/WanVideo_comfy_fp8_scaled
    lightx2v_I2V_14B_480p_..._rank64_bf16    0.74 GB  Kijai/WanVideo_comfy

A 200 GB volume is not enough to hold every lane. 400 GB, with the cache stored
once, is comfortable.

### Getting the token onto a RUNNING pod, without a redeploy

A RunPod secret injects at pod CREATION, so a pod that predates the secret
cannot see it and no amount of editing the template reaches it. That is worth
knowing, but it is not the only route: the token can be sent to a live pod
directly, and it never has to pass through an assistant's context.

    # value goes registry -> ssh stdin -> file. Never echoed, never on a
    # command line (where it would land in shell history and the process list).
    $t = [Environment]::GetEnvironmentVariable('HF_TOKEN','User')
    $t | ssh -p <port> -i <key> root@<ip> "cat > /root/.hf_token && chmod 600 /root/.hf_token"

**PUT IT ON THE CONTAINER DISK (`/root`), NOT THE NETWORK VOLUME.** `/root` dies
with the pod. `<models_root>/huggingface/token` sits on shared storage and is
inherited by every future pod that mounts that volume, long after the pod that
wrote it is gone.

**POWERSHELL WILL CORRUPT IT, SILENTLY.** The transfer above arrived as **41
bytes for a 37-character token** -- a UTF-8 BOM (3 bytes) plus a CR. The file
looks right, `cat` prints something that looks like a token, and authentication
fails with a message about the token being invalid rather than about encoding.
This is the same Windows trap the project rules already call out for source
files, and it applies just as well to a credential. Always verify the LENGTH,
never the value:

    sed -i "1s/^\xEF\xBB\xBF//" /root/.hf_token
    tr -d "\r\n" < /root/.hf_token > /root/.hf_token.clean
    mv /root/.hf_token.clean /root/.hf_token
    wc -c < /root/.hf_token          # must equal the local length
    head -c3 /root/.hf_token | od -An -tx1   # 68 66 5f = "hf_", NOT ef bb bf

**Then prove it before spending a download on it.** Two calls, no weights moved:

    GET /api/whoami-v2                         -> your username
    GET /api/models/Lightricks/LTX-2.5         -> OK means the token works AND
                                                  the licence terms are accepted

That distinction matters: `gated: "auto"` fails identically for a bad token and
for un-accepted terms, so a single probe that returns OK rules out both at once.

**A running ComfyUI still will not see it.** The server read its environment at
start, so a gated model loaded during a RENDER needs the process restarted with
`HF_TOKEN` exported -- fetching weights from a shell is a separate matter.

### index-tts: use the pinned owner, then keep runtime and provision paths equal

Do not reconstruct this environment with `uv venv`, `uv pip install .`, or the
pod's system Python. The audited provisioner checks out IndexTTS2 commit
`830f6f8f94a51fea23ab1d639027a86200075a4e`, resolves Python 3.10, runs
`uv sync --frozen`, downloads the exact pinned model manifest, warms and pins
all four runtime repositories under `checkpoints/hf_cache`, validates every
artifact, then boots the real worker offline and requires its ready response.
It explicitly installs a managed Python 3.10 with `only-managed` discovery and
places it under `<index-root>/.uv-python`; the venv and the interpreter it links
to therefore survive together when that root is on a network volume. If you set
`UV_PYTHON_INSTALL_DIR` yourself, it must also name a persistent-volume
directory.

On Linux, export the runtime layout explicitly. `OTR_INDEXTTS2_ROOT` is a
provisioner source-root setting; the qualified engine adapter intentionally
uses its own `_VENV`, `_DIR`, and `_WORKER` paths, so setting ROOT alone does
not redirect the running engine. Keep the source beside the pinned ComfyUI
checkout, not inside it, so a second provision pass does not correctly reject
the core as dirty:

```bash
export OTR_COMFY_ROOT=/workspace/runpod-slim/ComfyUI
export OTR_COMFYUI_MODELS_ROOT="$OTR_COMFY_ROOT/models"
export HF_HOME="$OTR_COMFYUI_MODELS_ROOT/huggingface"
export COMFY_PY="$OTR_COMFY_ROOT/.venv-cu128/bin/python"
export OTR_INDEXTTS2_ROOT=/workspace/index-tts
export OTR_INDEXTTS2_VENV="$OTR_INDEXTTS2_ROOT/.venv/Scripts/python.exe"
export OTR_INDEXTTS2_DIR="$OTR_INDEXTTS2_ROOT/checkpoints"
export OTR_INDEXTTS2_WORKER="$OTR_COMFY_ROOT/custom_nodes/ComfyUI-OldTimeRadio/scripts/_otr_indextts2_worker.py"

mkdir -p /workspace/otr-config
cd "$OTR_COMFY_ROOT/custom_nodes/ComfyUI-OldTimeRadio"
"$COMFY_PY" \
  scripts/otr_make_portable_voice_bank.py \
  --models-root "$OTR_COMFYUI_MODELS_ROOT" \
  --male-wav /absolute/path/to/authorized-male.wav \
  --female-wav /absolute/path/to/authorized-female.wav \
  --output /workspace/otr-config/voice_reference_bank.portable.json
export OTR_VOICE_REFERENCE_BANK=/workspace/otr-config/voice_reference_bank.portable.json

cat > /workspace/otr-config/otr-runtime.env <<EOF
export OTR_COMFY_ROOT="$OTR_COMFY_ROOT"
export OTR_COMFYUI_MODELS_ROOT="$OTR_COMFYUI_MODELS_ROOT"
export HF_HOME="$HF_HOME"
export COMFY_PY="$COMFY_PY"
export OTR_INDEXTTS2_ROOT="$OTR_INDEXTTS2_ROOT"
export OTR_INDEXTTS2_VENV="$OTR_INDEXTTS2_VENV"
export OTR_INDEXTTS2_DIR="$OTR_INDEXTTS2_DIR"
export OTR_INDEXTTS2_WORKER="$OTR_INDEXTTS2_WORKER"
export OTR_VOICE_REFERENCE_BANK="$OTR_VOICE_REFERENCE_BANK"
EOF
chmod 600 /workspace/otr-config/otr-runtime.env

"$COMFY_PY" scripts/otr_provision.py \
  --profile otr_runpod_starter --with-indextts2
```

Use the exact `comfy python:` path printed by the first provision pass for
`COMFY_PY`, and use the exact profile being installed in the last command.
Source `otr-runtime.env` before every later provisioner and ComfyUI launch. The
generated bank contains
one lower-case `male` and one lower-case `female` Index character reference
while retaining Kokoro and the other non-Index rows. The private
Lemmy-specific qualified route is deliberately unavailable on a stranger's
bank; generic character casting remains available. `--commercial-clean`
describes only verified rights in the supplied recordings and does not change
the non-commercial IndexTTS2 model profile. The generated bank names that one
unavailable qualified route exactly; it does not waive invalid evidence,
revoked rights, corrupt bytes, or any other character route.

If a persistent volume layout requires `/workspace/index-tts`, set ROOT, VENV,
and DIR to that same tree (or symlink it to `<comfy_root>/index-tts`) and keep
WORKER pointed at this repository. The final `checkpoints` path component is
mandatory because pinned vendor code resolves `./checkpoints/hf_cache` from
the worker's launch directory. A first render must never download these caches.

---

---

## 4. Running unattended

`scripts/otr_pod_overnight_sweep.sh`. Copy it to the pod and launch it there.

    ssh <pod> 'cat > /root/overnight.sh' < scripts/otr_pod_overnight_sweep.sh
    ssh <pod> 'setsid nohup bash /root/overnight.sh > /root/driver.log 2>&1 < /dev/null &'

### RUN IT ON THE POD, NOT FROM THE WORKSTATION

Driving a multi-hour sweep over SSH from Windows ties it to a laptop staying
awake and a tunnel staying up. A dropped connection ends the night's work while
the meter keeps running. On the pod it survives anything happening at the other
end -- and `setsid` is the part that matters, because plain `nohup` still dies
with the session's process group on some images.

### NO LANE MAY STOP THE RUN

A failing lane is recorded and the sweep moves on. "Does this work on rented
hardware" is the question being asked, so a failure IS a result. The earlier
sweep that aborted on the first bad lane produced one data point for the price
of twenty-five.

### READINESS IS CHECKED BEFORE EVERY LEG, NOT ONCE

An empty queue is not readiness -- a dead server reports an empty queue too. The
gate is `/object_info` returning 200 **and** containing `OTR_` classes **and** an
idle queue, re-checked before each leg so one wedged lane cannot poison the rest.

### THE TOKEN MUST BE IN THE SERVER'S ENVIRONMENT, NOT JUST ON DISK

ComfyUI reads its environment once, at start. A token written to the pod after
boot is invisible to every render until the server is restarted with it
exported. Fetching weights from a shell is a separate matter and works
immediately -- which makes this easy to miss, because the downloads succeed and
only the renders fail.

### WARM IN PARALLEL WITH THE BOOT

Model warming and server startup are independent, so they overlap. On a machine
billed by the second, serialising them is money spent on waiting.

### THE NETWORK VOLUME IS 6.5x SLOWER THAN LOCAL DISK, AND MODEL SWAPS PAY FOR IT

Measured on the pod, 2026-08-31, same 1 GB read both ways:

    network volume (MooseFS)   277 MB/s
    container disk           1,800 MB/s

The volume is the right place for weights -- it is the only thing that survives
a pod -- but it changes the arithmetic on model SWAPPING, and the pipeline swaps
a lot. `Mistral-Nemo-Instruct-2407` is 46 GB on disk and had been loaded **48
times** in one server lifetime; at 277 MB/s a single load is roughly **2.8
minutes of pure disk read**, before any compute.

**This is invisible on the dev box.** A local NVMe makes the same swap cost
seconds, so a pipeline that reloads a large technical model between passes looks
free there and is expensive on rented hardware.

**Watch for it in the obvious tell:** `nvidia-smi` showing 3% utilisation with
several GB resident, while the ComfyUI log scrolls
`Loading weights: n/363`. That is not a stall and not a hung render -- it is the
volume. Do not go looking for a deadlock.

**What NOT to do about it.** Do not quietly swap the technical model for a
smaller one to make the number look better: `technical_model` changes the script
the writer produces, so that is a content change wearing a performance costume,
and it invalidates any same-seed comparison against previous runs. If the swap
cost is worth attacking, it is an operator decision about which model to run, or
a change to how often the model is unloaded -- not a silent substitution.

---

## 4A. Heavy-engine canonical qualification: HuMo, LTX 2.5, and H3

This is the one complete heavy-engine procedure. Do not create a second lab
document or replace the canonical graph with a hand-built workflow.

### Evidence boundary and tested high-RAM tuple

HuMo 14B and LTX 2.5 have published canonical receipts on the local RTX 5080.
The first high-RAM remote qualification used this exact tuple:

- secure RunPod `jaq27diu24jt1p`, RTX 4090, 24,564 MiB VRAM, driver
  `570.172.08`;
- template `runpod/comfyui:cuda13.0`, ComfyUI at
  `/workspace/runpod-slim/ComfyUI`;
- 125 GB API RAM allocation, cgroup-v1 limit `124999999488` bytes
  (116.42 GiB effective), 70 GB container disk, 437 GB network volume;
- pinned ComfyUI core `169fcf35a2fc163fec31338b816503ddac0d3fcf`.

CUDA execution, all five LTX weights, both pinned node packs, IndexTTS worker,
all 25 OTR nodes, the real canonical validator, and queue submission passed.
After 39:56 the shipped 1664x960 two-stage decode failed at node 92 with a GPU
`OutOfMemoryError`; cgroup `memory.failcnt` and `oom_kill` remained zero. This is
a negative receipt for the exact 24 GB RTX 4090 tuple, not a host-RAM failure
and not a global claim that LTX 2.5 cannot run on RunPod. A clean 48 GB L40S
qualification is the next tuple. Promotion still requires `RESULT SUCCESS`,
`obs_publish OK`, delivered-engine evidence, zero new cgroup OOM events, and
the final file under `otr/obs/`.

The lab requires at least 100 GiB effective cgroup RAM and 150 GiB free on the
resolved model volume. Those are conservative rental requirements for the
whole episode and its caches, not claims that either video engine intrinsically
uses 100 GiB. The previous Ampere pod reached LTX's two-stage 1664x960 decode
and was SIGKILLed at a 57.7 GiB cgroup cap; that is a negative receipt for that
RAM cap, not for Ampere or RunPod.

### Starter versus qualification profiles

The newcomer starter remains `wan22_high_video` (`wan_ti2v`). It is proven on a
rented RTX A4500, completely fetchable, and honest about its 832x480 canvas.
LTX 2.5 does not enter the starter until a clean pod publishes a canonical
episode. Heavy qualification is explicit:

| lane | profile | download policy |
|---|---|---|
| HuMo 14B | `otr_w45_humo_14b_169` | automatic pinned five-file `humo` lane |
| LTX 2.5 high video | `otr_ltx25_high_video` | terms click, then exact five-file manual tier |
| H3 | `otr_w45_minimax_h3_video` | operator-owned offline hardware only |

Before HuMo or LTX provision can claim render-ready, complete the IndexTTS
section above and export all five IndexTTS variables plus
`OTR_VOICE_REFERENCE_BANK` in both the provisioning and server environments.

### CUDA 13 template on a 570-series driver

An image tag, venv name, and successful `import torch` are not runtime receipts.
On the tested pod, `.venv-cu128` inherited system site packages and actually
imported torch `2.10.0+cu130`. Driver 570 exposed CUDA capability 12.8 to that
wheel, so CUDA initialisation failed with `found version 12080`. The template
also exported `/opt/comfyui-runtime-constraints.txt`, pinning the CUDA 13 trio
and silently undoing a manual repair during later requirements work.

`scripts/otr_pod_provision.sh` now runs a real CUDA tensor matrix multiply in
the exact ComfyUI Python. It has one measured automatic repair: torch
`2.10.0+cu130` on a 570-579 driver receives this exact cu128 trio from PyTorch's
cu128 index, outside the conflicting constraint:

```text
torch==2.10.0+cu128
torchvision==0.25.0+cu128
torchaudio==2.10.0+cu128
```

It probes real CUDA work again and prevents the same template constraint from
reverting the repair during later installs. After every core, pack, and profile
dependency is installed, it repeats the device matmul before printing
`provision complete`. Unknown tuples fail closed. The measured wheel
replacement took about eight minutes. Prefer an otherwise
equivalent CUDA 12.8 image on a 570-series host to avoid that setup time; the
CUDA 13 template remains supported through this narrow repair. No VRAM reserve,
frame reduction, or workflow change is involved.

### Clone once, then provision the chosen profile

```bash
export OTR_COMFY_ROOT=/workspace/runpod-slim/ComfyUI
export OTR_COMFYUI_MODELS_ROOT="$OTR_COMFY_ROOT/models"
export HF_HOME="$OTR_COMFYUI_MODELS_ROOT/huggingface"
mkdir -p "$OTR_COMFY_ROOT/custom_nodes" "$HF_HOME"

if [ ! -d "$OTR_COMFY_ROOT/custom_nodes/ComfyUI-OldTimeRadio/.git" ]; then
  git clone -b v2.0-alpha \
    https://github.com/jbrick2070/ComfyUI-OldTimeRadio.git \
    "$OTR_COMFY_ROOT/custom_nodes/ComfyUI-OldTimeRadio"
fi
cd "$OTR_COMFY_ROOT/custom_nodes/ComfyUI-OldTimeRadio"
```

For LTX 2.5, accept the terms while signed in at
<https://huggingface.co/Lightricks/LTX-2.5>. Set `HF_TOKEN` as a RunPod secret,
or use the no-echo live-pod recipe in section 3. HuMo is public.

HuMo 14B:

```bash
export OTR_PROVISION_PROFILE=otr_w45_humo_14b_169
export OTR_WITH_INDEXTTS2=1
bash scripts/otr_pod_provision.sh
```

The `humo` lane pins every revision, destination, byte count and SHA-256,
downloads through `.part`, and fetches the engine's real
`Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors`, not Comfy-Org's unrelated
`humo_17B` lookalike. The five files total 28,707,153,033 bytes (26.7356 GiB).
The local 5080 receipt measured 13.06 GiB VRAM and 27.53 GiB host RAM at
832x480x97; use at least 32 GiB host RAM outside this roomier lab.

LTX 2.5:

```bash
export OTR_PROVISION_PROFILE=otr_ltx25_high_video
export OTR_WITH_INDEXTTS2=1
bash scripts/otr_pod_provision.sh
```

The provisioner pins ComfyUI-GGUF and applies the exact Gemma-4/BF16 loader
patch. It also pins ComfyUI-LTXVideo at
`3b9c5cde4700917074823d45e25401d81049f8fc` and applies
`patches/ComfyUI-LTXVideo-kornia-pad.patch`. Kornia 0.8.3 removed the `pad`
re-export used by that commit; the patch calls the file's existing `F.pad` at
three sites. Clean preimage, patch, patched postimage and the sole allowed
dirty path are hash-pinned. Do not downgrade Kornia or hand-edit the pack.

The first LTX provision may exit nonzero while the gated weights are absent.
That is an honest incomplete receipt. After the terms click, use this exact
manual helper; `.part` never counts as installed:

```bash
fetch_exact () {
  repo=$1; revision=$2; remote=$3; relative_dest=$4
  expected_bytes=$5; expected_sha=$6
  dest="$OTR_COMFYUI_MODELS_ROOT/$relative_dest"
  part="$dest.part"
  mkdir -p "$(dirname "$dest")"

  if [ -f "$dest" ] \
     && [ "$(stat -Lc '%s' "$dest")" = "$expected_bytes" ] \
     && printf '%s  %s\n' "$expected_sha" "$dest" | sha256sum -c -; then
    echo "PRESENT $relative_dest"
    return 0
  fi

  auth=()
  [ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")
  if ! curl -fL --retry 4 "${auth[@]}" -o "$part" \
      "https://huggingface.co/$repo/resolve/$revision/$remote"; then
    rm -f "$part"
    return 1
  fi
  [ "$(stat -Lc '%s' "$part")" = "$expected_bytes" ] || {
    rm -f "$part"
    return 1
  }
  printf '%s  %s\n' "$expected_sha" "$part" | sha256sum -c - || {
    rm -f "$part"
    return 1
  }
  mv -f "$part" "$dest" || {
    rm -f "$part"
    return 1
  }
}

fetch_exact realrebelai/LTX-2.5_GGUFs \
  112436f97aaf99ce13ecb7b7eca7e2f6c128d3ec \
  LTX-2.5-Distilled-Q3_K_M.gguf \
  diffusion_models/LTX-2.5-Distilled-Q3_K_M.gguf \
  11525623808 4286f8de1074c0c4fddfb92f38bd7df9161782b53c1717ebd69f1189c7933265

fetch_exact elix3r/gemma4-12b-with-proj-ltx-2.5-GGUF \
  085ceddbbac3c0370de7f59ebec8bef4763f04b5 \
  gemma4-12b-with-proj-ltx-2.5-Q5_K_M.gguf \
  text_encoders/gemma4-12b-with-proj-ltx-2.5-Q5_K_M.gguf \
  9514920864 1d35d4fbfa34cca1513d8e9fdd77c0573778b21ffdcbe4ca9c906f37a8c502f9

fetch_exact Lightricks/LTX-2.5 \
  5e6e71018ee1756ed329b697a7b4aedc934dfce9 \
  vae/ltx-2.5-video-vae-bf16.safetensors \
  vae/ltx-2.5-video-vae-bf16.safetensors \
  1472223346 847e14ca7f3355debca0cea4eaa24ac0fbcdf0061da054ac89ca638a869ddba3

fetch_exact Lightricks/LTX-2.5 \
  5e6e71018ee1756ed329b697a7b4aedc934dfce9 \
  vae/ltx-2.5-audio-vae-bf16.safetensors \
  vae/ltx-2.5-audio-vae-bf16.safetensors \
  364866540 c52733d37f6a7fb7949c3dc0fb468c6cb2169e4d836983a73babb9f0d54837a5

fetch_exact Lightricks/LTX-2.5 \
  5e6e71018ee1756ed329b697a7b4aedc934dfce9 \
  latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  995778752 eb5a71fe4068ee87ccdb1c3aa635e547ca76bd2d30ae20ae889f2c325c0677e8
```

The LTX tier totals 23,873,413,310 bytes. Rerun the same profile provision
until every final file and dependency verifies and the command returns zero.

### H3 authorization and hardware boundary

This operator does **not** put H3 weights on RunPod. Under
`docs/H3_LICENSE_ATTESTATION.md`, H3 stays on owned offline hardware and its
weights are never redistributed. That is an authorization boundary, not a GPU
limitation. An authorized third party can inspect the same exact five-file lane:

```bash
python scripts/otr_provision.py --profile otr_w45_minimax_h3_video --list
python scripts/otr_fetch_lane_weights.py minimax_h3
```

It totals 63,440,965,087 bytes (59.084 GiB) and is never auto-selected by a
public machine class. Current ComfyUI core supplies H3 nodes; the public
`mkhamra/quibble-h3` repository is a Ref2VA workflow/case study, not an OTR node
provider. Legal local 5080 receipts at 124 model / 129 canvas frames measured
6,315 MB FL2VA and 6,678 MB REF2VA. The physical RTX 4060 has isolated H3 clip
receipts, including a retained 864x480x124 Ref2VA artifact; its full canonical
H3 episode remains unqualified, not proven impossible.

### Launch the exact profile contract and canonical graph

Use the Python path printed as `comfy python:` by the provisioner. On the tested
template it is `/workspace/runpod-slim/ComfyUI/.venv-cu128/bin/python`. Stop only
the existing ComfyUI listener; never `pkill python`. Export the IndexTTS values
from section 3 and the token before boot.

LTX has no artificial VRAM reserve:

```bash
export COMFY_PY=/workspace/runpod-slim/ComfyUI/.venv-cu128/bin/python
export HF_TOKEN="$(tr -d '\r\n' < /root/.hf_token)"
env -u PIP_CONSTRAINT "$COMFY_PY" "$OTR_COMFY_ROOT/main.py" \
  --listen 127.0.0.1 --port 8000 \
  --output-directory "$OTR_COMFY_ROOT/output" \
  > /workspace/otr-ltx25-server.log 2>&1 &
SERVER_PID=$!

until curl -fsS http://127.0.0.1:8000/object_info >/dev/null; do sleep 2; done
curl -fsS http://127.0.0.1:8000/queue >/dev/null

COMFYUI_URL=http://127.0.0.1:8000 \
  "$COMFY_PY" scripts/otr_canonical_api_run.py \
  --profile otr_ltx25_high_video --act-count 1 \
  --source-bank original --visual-style sci_fi_radio --timeout 0
```

HuMo wide uses its shipped `humo_diet` launch contract. These are runtime
offload controls for the proven HuMo recipe, not a pretend smaller-GPU test:

```bash
env -u PIP_CONSTRAINT "$COMFY_PY" "$OTR_COMFY_ROOT/main.py" \
  --listen 127.0.0.1 --port 8000 \
  --output-directory "$OTR_COMFY_ROOT/output" \
  --reserve-vram 2.921 --disable-pinned-memory \
  > /workspace/otr-humo-server.log 2>&1 &

until curl -fsS http://127.0.0.1:8000/object_info >/dev/null; do sleep 2; done
COMFYUI_URL=http://127.0.0.1:8000 \
  "$COMFY_PY" scripts/otr_canonical_api_run.py \
  --profile otr_w45_humo_14b_169 --act-count 1 \
  --source-bank original --visual-style sci_fi_radio --timeout 0
```

Omit `--workflow`: the runner must load `workflows/otr_canonical.json` itself.
Do not reuse one resident server between LTX and HuMo. A finished render leaves
the server resident with VRAM allocated; `Prompt executed`, `obs_publish OK`,
and the final file distinguish completion from a crash.

### Qualification receipt and telemetry

Success requires all of these:

- `RESULT SUCCESS` and `obs_publish OK`;
- final file under the actual `$OTR_COMFY_ROOT/output/otr/obs/`;
- delivered engine matches the selected profile;
- zero new cgroup `memory.failcnt` or `oom_kill` events;
- exact GPU, driver, torch/CUDA build, core/pack commits, patch hashes, cgroup
  limit/peak, GPU peak, elapsed time, server log, runner log, and artifact path;
- LTX's shipped 1664x960 tiled decode and upsample/refine evidence remains
  present. Do not shrink the canonical canvas to manufacture a pass.

For cgroup v1 telemetry:

```bash
while kill -0 "$RUNNER_PID" 2>/dev/null; do
  printf '%s,' "$(date -u +%FT%TZ)"
  paste -sd, \
    /sys/fs/cgroup/memory/memory.usage_in_bytes \
    /sys/fs/cgroup/memory/memory.max_usage_in_bytes \
    /sys/fs/cgroup/memory/memory.failcnt
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits
  sleep 1
done
awk '/oom_kill/{print}' /sys/fs/cgroup/memory/memory.oom_control
```

Large downloads and wheel installs can leave tens of GiB in reclaimable page
cache. `memory.usage_in_bytes` alone is not an engine working-set measurement;
judge it together with process survival, fail counters, queue/history, and the
artifact. The active pod exceeded 100 GB headline usage with zero failcnt and
OOM kills, so the healthy run continued.

### Lessons that must remain in this playbook

1. Record the full software/hardware tuple. A GPU name or venv directory name
   cannot substitute for executable CUDA evidence.
2. Require real device work before downloading tens of GiB. `import torch` is
   not a CUDA test.
3. Inspect inherited pip constraints; they can silently undo a correct repair.
4. A pinned pack commit does not pin every dependency API. Own the smallest
   exact patch with preimage, patch, postimage, and dirty-path guards.
5. Do not use VRAM clamps or a reduced canvas as a physical-card receipt. Rent
   enough host RAM and run the canonical workflow unchanged.
6. Installation success is not render success. Preserve provisioning, server,
   runner, telemetry, cgroup, delivered-engine, and final-artifact evidence.

## 5. Failure atlas -- indexed by SYMPTOM

Find what you are seeing, not when it was discovered.

### "The pod restarted, the volume is intact, and the venv is broken"

**A stopped pod can lose its GPU.** RunPod offers to migrate: it builds a NEW
pod with identical GPUs and moves your data. Take it -- "start using CPUs" is
useless for rendering and "do nothing" gambles on the same hardware freeing up.
But understand what migrate actually preserves:

    NETWORK VOLUME   kept entirely       167 GB models, 90 GB cache, refs, index-tts
    CONTAINER DISK   gone                /root/.hf_token, uv, any scripts in /root

**Older installs split the venv from its interpreter.** Measured on a real
migration, 2026-08-31:

    /workspace/index-tts/.venv/lib          14 GB   survived
    /workspace/index-tts/checkpoints        25 GB   survived
    /workspace/index-tts/.venv/bin/python   ->  /root/.local/share/uv/python/
                                                cpython-3.11-.../bin/python3.11
                                            DANGLING

`uv` does not copy an interpreter into the venv; it symlinks to its managed
Python. The old installer left that interpreter on container disk, so 39 GB of
expensive install survived on the volume while the small interpreter vanished.
The current provisioner defaults `UV_PYTHON_INSTALL_DIR` to `.uv-python` inside
each persistent engine root, preventing that split for new installs.

**For a legacy dangling venv, repair the interpreter version named by its
symlink; do not rebuild the packages:**

    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="/root/.local/bin:$PATH"
    readlink .venv/bin/python
    uv python install 3.10        # current pinned IndexTTS2 runtime
    uv python install 3.11        # chatterbox/dia; legacy Index only if its link says 3.11

Do NOT rebuild the venv. `--allow-existing` is the fallback if the version no
longer matches; reinstalling from scratch would re-download 14 GB of wheels to
replace a file that was never the problem.

**Also re-send the token** -- it lived on the container disk too. See "Getting
the token onto a RUNNING pod" above; verify by LENGTH, and strip the BOM.

### "Install reported success, and the pack contributes ZERO nodes"

ComfyUI is scanning a different directory than the one you installed into. Do
not chase dependencies: `__init__.py` loads each node in its own try/except, so
a missing library SKIPS that node and prints `Skipped '<name>': <reason>` -- 32
of 34 nodes still registered with every requirement blocked. **A true zero is
never a dependency problem.** Find the tree ComfyUI actually scans:

    python3 -c "import folder_paths,os;print(os.path.dirname(folder_paths.__file__))"

Do NOT assume `/workspace/ComfyUI`; this image uses `/workspace/runpod-slim/ComfyUI`.

### "Manager returns HTTP 200 and nothing installs"

The registry entry is **Flagged**, not Pending. Flagged does not self-clear, and
with zero Active versions `latest_version` resolves to null -- `@latest` has no
target, and the `nightly` git path is refused by Manager's `security_level` on
any network-exposed instance. **Nobody can install OTR through ComfyUI-Manager
by any route** until that is resolved. Install with `git clone -b v2.0-alpha`.

### "The render dies 7-18 minutes in, at the video stage"

NVENC. Most containers expose CUDA for compute but not `libnvidia-encode.so.1`,
so `h264_nvenc` cannot initialise. **ffmpeg being installed proves nothing** --
this was misdiagnosed once as a missing ffmpeg. OTR now probes by ENCODING a
test frame rather than grepping `ffmpeg -codecs`, and falls back to CPU. The
probe uses 256x256 deliberately: NVENC rejects tiny frames, and a 64x64 probe
reported a healthy card as unavailable.

### "Disk quota exceeded on a volume that is half empty"

The HF cache exists TWICE and each copy looks correct on its own. It reads as
"buy more storage" and is really "written twice". Measured: 84 GB at an invented
path plus 71 GB at OTR's convention, with Mistral-Nemo (46 GB) and gemma-4-12b
(23 GB) byte-identical in both. Merging reclaimed **70 GB in minutes**.

Set `HF_HOME` to `<models_root>/huggingface`, then MERGE, never move -- the two
roots are not identical in either direction:

    cp -an /workspace/hf/hub/. <models_root>/huggingface/hub/

`rsync` is NOT on this image. Write an "every model in A exists in B" check
before deleting anything; that guard is what turned a failed merge into a safe
one.

### "The first render fails at ~73 s with `_LLMTimeoutWorkflowPause`"

A model-shaped message for a missing DOWNLOAD. transformers fetches the writer
on first use and the download outlasts the writer timeout. Warm the cache before
rendering. A developer box never sees this -- its cache has been warm for months.

### "3% GPU, several GB resident, log scrolling `Loading weights: n/363`"

Not a stall. The network volume reads at **277 MB/s** against the container
disk's **1.8 GB/s**, so a 46 GB technical model is roughly 2.8 minutes of pure
disk read per swap. Do not hunt a deadlock -- and do NOT quietly swap
`technical_model` for something smaller, which changes the script the writer
produces: a content change wearing a performance costume.

### "`pgrep`/`pkill` found nothing, or killed the wrong thing"

The pattern matches the SSH command line containing it -- itself.
`pgrep -f "ComfyUI/main.py"` returns your own shell, and killing it then seeing
an empty list reads exactly like "the server is dead" when it has been up all
along. **Identify the server by its listening port:**

    ss -lptn | grep 8188        ->  LISTEN  353/python

It bites `pkill` harder: `ssh pod 'pkill -f overnight.sh; ...; nohup bash
overnight.sh &'` kills the session executing that line, so everything after the
`pkill` never runs -- no output, no relaunch, and a pod that bills while looking
fine. **Put kill patterns in a script FILE on the pod and run the file.** This
cost the run twice in one session.

### "The queue is empty but nothing renders"

An empty queue is also what a DEAD server reports. Readiness is three things
together: `/object_info` returns 200, it contains `OTR_` classes, and the queue
is idle. Gating on queue-empty alone started a sweep while the proxy still
served its 502 page; all 25 lanes failed in about a second and looked like 25
broken lanes.

### "`POST /api/manager/reboot` returned 502"

It worked. The server drops the connection going down. Poll `/system_stats` for
a 200 rather than trusting the status code. A Manager reboot does not re-read
the image's argument file -- stop and start the pod for argument changes.

### "RESULT SUCCESS, but the pull tool finds no episode"

`/history` never records the published episode. `OTR_MasterAudioMux` publishes
it and returns a bare tuple with no `ui` payload, so the only video the API
advertises is the pre-post-processing intermediate in the episode's `audio/`
folder (PBUG-20260830-24). Pull by listing the obs DIRECTORY over SSH:

    python scripts/otr_pod_obs_bridge.py <podId> --host <ip> --port <port>

### "Nothing restarts ComfyUI, and there is no log to tail"

There is no supervisor on this image, and its ComfyUI logs to a PIPE
(`/proc/<pid>/fd/1 -> pipe:`). Relaunch it yourself, redirecting to a file --
worth doing for the log alone. **An SSH session does not inherit the container
environment**, so recover it from pid 1:

    cd /workspace/runpod-slim/ComfyUI
    eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^(OTR_COMFYUI_MODELS_ROOT|HF_HOME|HF_TOKEN)=' | sed 's/^/export /')"
    nohup .venv-cu128/bin/python main.py --listen 0.0.0.0 --port 8188 \
          --enable-cors-header > /workspace/comfyui.log 2>&1 &

### "The token is on disk and every render still fails on a gated model"

ComfyUI read its environment at start. Fetching weights from a shell works
immediately, which is exactly what hides this. Restart the server with
`HF_TOKEN` exported.

### "A sweep produced one data point for the price of twenty-five"

It aborted on the first bad lane, or walked the roster once. A soak must LOOP:
ten lanes failed on a missing model, the model was installed twenty minutes
later, and none was retried -- their recorded result described a machine state
that had stopped being true. Number the rounds, so a lane that flips reads as
flapping rather than being averaged away.

### "Every lane fails identically, minutes in, at the same node"

One missing asset on the SHARED path. `OTR_StableAudioTheme` and
`OTR_BatchCharacterVoices` both sit there, so a machine missing Stable Audio 3
or the indextts2 install fails EVERY profile that reaches them -- 7 to 13
minutes into each, after the script, cast and voices are already done. Install
everything up front (playbook step 4). Do not add a checker that merely reports
it sooner: that moves the discovery earlier without removing any of the work.


### "DRAM canary TRIPPED -- but `free -g` says 92 GB available"

Both readings are correct; they measure different things. The canary reports
`psutil.virtual_memory().available`, and on Linux that diverges from the `free`
column by the entire page cache. A pod that had just pulled 74 GB of weights
read `free=19.8 GB` against `available=92.4 GB` -- a 4.7x gap that does not
exist on a fresh Windows box, which is why this never surfaced on the dev
machine. Compare like with like before calling it a false alarm.

Then read what the message does NOT say: it is advisory. The blend is the only
caller, and it deliberately degrades OPEN -- it warns and proceeds, because a
transient dip during a `filter_complex` blend that buffers frames from two 1080p
mp4s is expected, not fatal. A tripped canary in the log therefore does not mean
a degraded episode. The 2026-08-31 sci-fi leg tripped at 1.64 GB and still
published a full-quality 187 s / 1920x1080 episode. Confirm with the artifact:
`_captioned_with_credits_final.mp4` present in `otr/obs/` and a `RESULT SUCCESS`
line is proof the blend ran; the canary line alone proves only that RAM was
briefly tight.

The wording that caused this misread has been fixed at the source -- the message
now names `available` and says ADVISORY -- but old logs still carry the original
sentence, which told the reader an abort should happen at the one call site that
never aborts.

### "I restarted the pod and everything survived, so the template is fine"

It is not fine, and that restart proved nothing. A RESTART leaves the container
disk intact -- apt packages, `/root`, the token, everything. Only a RECREATE
(terminate, then create fresh from the template) discards it, and that is the
only version of the test that tells you whether the documented install path can
build a working machine. On 2026-08-31 a restart was read as proof; the recreate
that followed immediately found five separate holes the restart had hidden.

Keep the two words apart when reporting, too. "The pod came back with everything"
after a restart is a statement about container persistence, not about the repo.

### "The pod has no models, no repo, no ComfyUI -- but the template is right"

Check `RUNPOD_VOLUME_ID` before anything else:

    tr '\0' '\n' < /proc/1/environ | grep RUNPOD_VOLUME_ID

If it is ABSENT, no network volume attached, and `/workspace` is a container
overlay that merely looks like the real thing -- `df -h /workspace` shows
`overlay 70G` where an attached volume shows `mfs#<region>.runpod.net:9421`.
Everything then appears "missing" while nothing is actually lost.

The usual cause is the DATACENTER. Network volumes are region-locked, so a GPU
outside the volume's region is simply never offered the attach, and the pod
boots without it. Re-creating in the volume's region is the only fix; a 70 GB
overlay cannot hold a 176 GB model tree, so "download it again" is not an option.

### "Three isolated voice venvs report missing, but their packages are all there"

This describes venvs created by the older installer. Their site-packages live
on the network volume while `.venv/bin/python` points to a managed interpreter
under `/root`, which is container disk. A recreate wipes it and leaves the
packages intact but unreachable. Repair both required Python lines, not the
packages:

    uv python install 3.10        # IndexTTS2
    uv python install 3.11        # chatterbox and dia

New installs made by `scripts/otr_provision.py` keep each managed interpreter
inside that engine's `.uv-python` directory on the same persistent root, so a
recreate does not need this repair. Before assuming a 33 GB rebuild, check
whether `readlink -f .venv/bin/python` resolves at all.

### "chatterbox and dia die with 'no kernel image is available'"

A CUDA ARCHITECTURE mismatch, not a code fault. Left to resolve torch on its
own, pip installed cu124 for chatterbox and cu126 for dia; neither ships kernels
for Blackwell (sm_120), so both imported fine and died at the first kernel
launch, while index-tts happened to draw cu128 and worked throughout.

Proven by running the SAME venv bytes on an RTX A4500 (sm_86), where all three
launch kernels cleanly. The installer now takes its CUDA build from ComfyUI's
own `torch.version.cuda` -- already proven on that machine -- and verifies with a
real kernel launch, because `import torch` succeeds on a build with no kernels
and reported OK for both venvs that later failed mid-episode.

### "The image model is installed and the render still says it is not"

Check WHICH file. `z_image_turbo` ships in three precisions and the adapter ranks
installed candidates `nvfp4 > fp8 > bf16`, so nvfp4 wins whenever present. nvfp4
needs hardware fp4 (sm_120); an Ampere or Ada card cannot execute it, and the
ranking will still choose it -- picking the one file that fails.

Fetch by architecture: `z_image_blackwell` on sm_120, `z_image` (bf16) below it.
The provisioner reads `compute_cap` and picks. If a shared volume already carries
nvfp4, move it OUT of the models tree -- a subfolder is not enough, because the
ranking matches on basename.

### "Commands over ssh.runpod.io return garbage or nothing"

The `ssh.runpod.io` proxy only opens interactive shells: it discards a remote
command, and demands a PTY (`Your SSH client doesn't support PTY`). With `-tt`
you get a shell whose output is interleaved with escape codes and whose long
lines are re-wrapped mid-path -- which produced a FALSE reading of a directory
test, and a wrong "the volume attached" conclusion.

Use the DIRECT form from the Connect tab, `ssh root@<ip> -p <port>`, for anything
scripted. Keep the proxy for typing at a shell by hand.

### "WrapperNodeMissing, seventeen minutes into a meaty video lane"

A required CUSTOM NODE PACK is not installed. OTR's video engines resolve node
CLASSES by name out of the ComfyUI registry at render time, so a missing pack
cannot fail at startup -- it fails deep inside the episode, after the script,
the cast, the voices and the stills have all completed successfully. That is
what makes it read like a code fault instead of a missing install.

    wan_ti2v   needs ComfyUI-GGUF                      (UnetLoaderGGUF, CLIPLoaderGGUF)
    ltx25      needs ComfyUI-GGUF + ComfyUI-LTXVideo   (the advanced LTXV nodes)
    ltx_8gb    needs neither -- core ComfyUI carries the basic LTXV three

Both GGUF-dependent lanes default to GGUF weights, so ComfyUI-GGUF is not
optional for them. `scripts/otr_provision.py` now clones all three packs; before
2026-08-31 it cloned only AnimateDiff, which is why a fully provisioned machine
could render the AnimateDiff lane and nothing else. Restart ComfyUI after
adding a pack -- node classes register at startup.

Diagnose it by reading the traceback rather than the runtime: the exception is
raised from `wrapper_bridge`, names no file of yours, and the elapsed time will
be long enough to look like an OOM that never happened.

### "Which z_image precision should this machine fetch?"

z_image_turbo is the LOW-VRAM lane and it is proven at 8 GB -- the 4060 has
published nine episodes on it. The adapter offloads the text encoder before the
diffusion peak, so precision is a question of download size and offload
pressure, NOT of whether the lane will run.

    z_image_blackwell   nvfp4  4.20 GB   sm_120 only -- hardware fp4
    z_image             bf16  11.46 GB   any NVIDIA
    z_image_int8        int8   5.78 GB   any NVIDIA, smallest universal option

A CAUTION LEARNED BY GETTING IT WRONG HERE: do not read a card's memory.used
during the image stage as the lane's requirement. On a 20 GB A4500 that figure
reached 19.3 GB, which was written up as "bf16 does not fit 16 GB" -- and that
was false. ComfyUI expands into whatever memory is free; the 8 GB proof already
existed and contradicted it. Measure a floor by running on the small card, never
by watching the big one fill up.

The ranking is `nvfp4 > fp8 > bf16 > other`, and `z_image_turbo_int8_convrot`
matches none of the first three, so it sorts LAST. Fetch ONE precision. If both
int8 and bf16 are installed the ranking takes bf16 and the smaller download was
wasted.

### "h3 fails with WrapperNodeMissing and no amount of weights fixes it"

OTR resolves H3 from `comfy_extras/nodes_minimax_h3.py` in current ComfyUI core
and does not use `ComfyUI-MiniMax-H3-Turbo`. If
`MiniMaxH3ImageToVideo` or `MiniMaxH3ReferenceToVideo` is absent, update the
ComfyUI core, restart it, and verify the live `/object_info`; do not install an
unrelated Turbo pack. A missing wrapper class is not evidence of an NVFP4
architecture failure.

### "wan_ti2v gets past the missing pack and then says sentencepiece"

    Error: Please make sure sentencepiece and protobuf are installed.

ComfyUI-GGUF's `requirements.txt` lists three packages, and the last two sit
under a comment reading `# optional - tokenizer`:

    # main
    gguf>=0.13.0
    # optional - tokenizer
    sentencepiece
    protobuf

They are not optional for this pack's use here -- wan_ti2v loads a GGUF text
encoder and fails without them, roughly eighteen minutes in, after the script,
cast, voices and stills are done. "Optional" describes the upstream project's
view, not OTR's.

Install the WHOLE file, never the line that looks required:

    <comfy-python> -m pip install -r custom_nodes/ComfyUI-GGUF/requirements.txt

`scripts/otr_provision.py` does exactly that for every pack it clones. This
entry exists because a hand-install picked out `gguf` alone and reproduced the
failure the provisioner was written to prevent -- the tool was right and the
shortcut was wrong.

### "The obs bridge says the pod has published 0 episodes, and it has 3"

Git Bash rewrote the path. MSYS converts an argument that looks like a POSIX
path into a Windows one before the program sees it, so

    --pod-obs /workspace/runpod-slim/ComfyUI/output/otr/obs

arrives as `C:/Program Files/Git/workspace/runpod-slim/...`, the remote `ls`
matches nothing, and the bridge truthfully reports zero. Nothing is broken; the
question asked was about a directory that does not exist.

    export MSYS_NO_PATHCONV=1
    export MSYS2_ARG_CONV_EXCL="*"

Set both before any command that passes a REMOTE absolute path through Git Bash
-- the obs bridge, ssh one-liners, docker, kubectl. Running the same command
from PowerShell also avoids it. The tell is that the identical ssh command works
when you type it and fails inside the script.

### "ltx25 says WrapperNodeMissing and every pack and weight is installed"

Check ComfyUI's own version before anything else:

    cd <comfy-root> && git describe --tags

**The ltx25 lane needs ComfyUI >= v0.32.0.** `LTXVDualCFGGuider` and
`LTXVModalityGuidance` live in ComfyUI CORE -- `comfy_extras/nodes_lt.py` -- not
in ComfyUI-LTXVideo, and they arrived in commit 57ce8e1a, *Add support for LTX
2.5 (#15499)*, on 2026-08-11. No node pack can supply them.

Measured on the rented pod: its image shipped **v0.26.2 (2026-06-30)**, two
months and eight minor versions behind the development box's v0.34.2. Updating
ComfyUI-LTXVideo changed nothing because the pack was already current and never
had those classes.

This is the most expensive shape of failure in the whole atlas, because every
signal points the wrong way: the pack is installed, the weights are on disk, the
node pack is at its latest commit, and the error names a node rather than a
version. It surfaces at render time, seventeen minutes in.

A pod image is a snapshot. Treat its ComfyUI version as a fact to check, not a
given -- the template that boots fastest is often the one furthest behind.

### "upscale_engine is off -- is that a mistake somebody should fix?"

No. It is OFF ON PURPOSE in `workflows/otr_canonical.json`, and it should stay
that way until somebody deliberately turns it on for a measured test.

Operator, 2026-09-01: an upscaler once tried to **infinitely upscale stills**.
That is why the default is `off` rather than an oversight nobody got round to.

There are only two engines -- `off` and `spandrel_esrgan` -- so the untested
surface here is one option behind a default-off switch, not a family. Turning it
on also needs ESRGAN weights, which are not on disk on the reference machine, so
flipping the widget alone would fail at load rather than upscale anything.

If it is ever tested: bound it. A runaway is not a quality problem, it is a
resource problem, and the thing to measure first is whether it terminates.

## Appendix -- a fresh pod with NO persistent storage

The playbook above deploys with a network volume, which is the right default:
weights survive, and a recreate costs two apt packages and a token. This
appendix covers the other case -- a pod with nothing but its container disk,
where everything is downloaded every time and lost when the pod dies.

It works, and the provisioner does all of it. The constraint is disk, and it is
tighter than it looks.

### Measured sizes, on a 70 GB container disk

    ComfyUI itself                 0.7 GB
    ComfyUI venv                   3.1 GB
    node packs (3)                 0.9 GB
    haunted lane (AnimateDiff)     3.7 GB
    z_image_int8 (image)          13.6 GB
    stable_audio_3 (music)         3.5 GB
    writer model                3 - 12 GB   depending on the class
    ------------------------------------------------------------
    a working install          ~29 - 38 GB   comfortable on 70 GB

    index-tts venv                39 GB   <-- does NOT fit alongside the above
    chatterbox venv               10 GB
    dia venv                      8.9 GB

**THE VOICE DECISION IS THE WHOLE BUDGET.** index-tts is 39 GB on its own, so on
a 70 GB disk it cannot coexist with the models. A no-volume pod should run
kokoro, which loads in ComfyUI's own venv and needs no isolated environment at
all; chatterbox (10 GB) fits if a cloning voice is required. Reaching for
index-tts here is what fills the disk, and the failure arrives as
`Errno 122` mid-download rather than as a clear refusal.

### The sequence

    # 1. clone the pack into the tree ComfyUI actually scans
    cd <comfy>/custom_nodes
    git clone -b v2.0-alpha https://github.com/jbrick2070/ComfyUI-OldTimeRadio

    # 2. the token -- USUALLY NOTHING TO DO HERE
    #    Set HF_TOKEN once in the RunPod template UI and the provisioner finds
    #    it by itself. Only needed if you did not:
    printf '%s' "$HF_TOKEN" > /root/.hf_token && chmod 600 /root/.hf_token

    # 3. first pass: public packs/lanes; expected INCOMPLETE at IndexTTS2
    cd ComfyUI-OldTimeRadio
    OTR_PROVISION_PROFILE=otr_runpod_starter bash scripts/otr_pod_provision.sh

    # 4. complete section 3's authorized voice-bank recipe, then finish
    . /workspace/otr-config/otr-runtime.env
    OTR_PROVISION_PROFILE=otr_runpod_starter OTR_WITH_INDEXTTS2=1 \
        bash scripts/otr_pod_provision.sh

That single command pins ComfyUI to the audited core commit, proves real CUDA
execution, installs the system libraries, clones the exact node-pack commits
and their requirements, repairs supported runtime drift, routes the selected
profile, and fetches its automatic lanes. Every one of those steps exists
because skipping it cost a leg.

Choose the weights explicitly if the defaults are wrong for the box:

    OTR_PROVISION_LANES="haunted" OTR_PROVISION_IMAGE_LANE=z_image_int8 \
        bash scripts/otr_pod_provision.sh

`python scripts/otr_fetch_lane_weights.py --list` prints every lane with its
size and how much is already on disk.

### What you give up

Nothing about correctness -- a no-volume pod renders the same episodes. What you
lose is the ability to stop and resume: every model is re-downloaded on the next
pod. The current starter's automatic lanes alone total about 32.1 GB, before
writer, IndexTTS2, and other Hugging Face caches. If you expect to run more than
one pod, the volume pays for itself quickly.

### The starter workflow -- one good everything, dropdowns already set

`workflows/variants/otr_runpod_starter.json` is the from-scratch answer for a
RunPod box with **16 GB of VRAM or more**. It is generated from the current
profile and ships these exact selectors:

    video    wan22_high_video (16:9)   Wan 2.2 TI2V at 832x480
    image    z_image_turbo             bf16 automatic lane
    voice    indextts2 (characters)    authorized reference bank required
             kokoro (announcer)
    music    stable_audio_3
    writer   Mistral-Nemo-Instruct
    bank     default                   mixed cast; kokoro_builtin would force
                                       every voice to kokoro and contradict the
                                       character engine above

Wan is the starter because it has a published rented-A4500 receipt and every
weight is public and automatically fetchable. The complete generated starter
combination is still a qualification target on a new pod; individual selector
history is not a substitute for `RESULT SUCCESS`, `obs_publish OK`, and a final
artifact from this exact profile.

**It is GENERATED, never hand-edited.** Change `config/profiles/
otr_runpod_starter.json` and re-run:

    python scripts/build_variants.py --profiles otr_runpod_starter

Editing the variant directly survives a diff and then fails regeneration.

The provisioner fetches these three lanes automatically. These are the exact
manual retry commands if a transfer was interrupted:

    python scripts/otr_fetch_lane_weights.py wan_ti2v_gguf
    python scripts/otr_fetch_lane_weights.py z_image
    python scripts/otr_fetch_lane_weights.py stable_audio_3

LTX 2.5 is an explicit heavy qualification profile, not the newcomer starter.
Its terms click, exact five-file tier, pack patches, and canonical command are
all in section 4A of this same playbook.

### Giving the pod your Hugging Face token, the easy way

**Set `HF_TOKEN` in the RunPod template UI, once.** Every pod from that template
then has it, and `otr_pod_provision.sh` picks it up with no further action --
there is no file to write and no command to remember.

The reason that is not obvious: RunPod sets template variables on the CONTAINER
(pid 1), and an SSH session gets a fresh environment that does NOT inherit them,
so `echo $HF_TOKEN` over SSH prints nothing even when the template set it. The
provisioner reads pid 1 directly, which is the same trick it already uses for
`OTR_COMFYUI_MODELS_ROOT`.

It searches, in order:

    1. an existing /root/.hf_token
    2. HF_TOKEN in the shell running the provisioner
    3. HF_TOKEN on the container (pid 1) -- the template variable
    4. ~/.cache/huggingface/token, from a previous `hf auth login`

and writes whatever it finds to `/root/.hf_token`, chmod 600. It prints only
WHERE the token came from and HOW LONG it is -- never the value. That length is
worth reading: a token pasted through PowerShell once arrived 41 characters
instead of 37, carrying a BOM and a carriage return, and the only symptom was a
401 much later. Whitespace is now stripped, and a value not starting with `hf_`
is called out.

With no token, ungated lanes still work; gated ones 401. The provisioner says so
rather than failing silently.

### "Unexpected text model architecture type in GGUF file: 'gemma4'"

**The ltx25 lane needs a PATCHED ComfyUI-GGUF. Upstream does not work.**

LTX 2.5's text encoder is `gemma4-12b-with-proj-ltx-2.5-Q5_K_M.gguf`, and
city96's ComfyUI-GGUF does not list `gemma4` among the architectures it will
load. Cloned fresh from upstream on 2026-09-01, HEAD `6ea2651` dated
2026-01-12 -- that IS the current release; the pack simply has not gained the
architecture.

OTR pins that pack at `6ea2651e7df66d7585f6ffee804b20e92fb38b8a`, then
applies one hash-pinned in-repo patch to `loader.py`: `gemma4` enters
`TXT_ARCH_LIST`, the three raw LTXAV BF16 parameters are named, and their byte
storage is dequantized on load. Clean preimage, patch, patched postimage, and
the sole allowed dirty path are all verified; unknown drift fails closed.

So this is not merely "install a node pack" -- the pinned ComfyUI-GGUF checkout
also needs OTR's in-repo loader patch. This portability limit is specific to
LTX's GGUF text encoder. H3's required node classes are already in ComfyUI core.

**The symptom arrives late and blames the file.** The error names the GGUF, so
the instinct is to suspect a bad download or the wrong quant. The weights are
fine; the loader does not know the architecture. Check
`grep gemma4 <comfy>/custom_nodes/ComfyUI-GGUF/loader.py` before re-downloading
anything.

**And the patch must reach the PROCESS.** ComfyUI imports loader.py at boot and
Python will not reload it, so copying the file onto a running server changes
nothing until a restart. Verify with:

    python -c "import sys;sys.path.insert(0,'<comfy>/custom_nodes/ComfyUI-GGUF');
               import loader;print('gemma4' in loader.TXT_ARCH_LIST)"

### "ComfyUI was Killed, and free said there were 193 GiB available"

`free` INSIDE A CONTAINER REPORTS THE HOST'S MEMORY, NOT YOUR LIMIT. Measured
on a RunPod pod, 2026-09-01:

    free -g              251 GiB total, 193 GiB available
    cgroup memory.max     61999996928 bytes = 57.7 GiB   <- the real ceiling

ComfyUI ran until it crossed 57.7 GiB and the kernel killed it. The log ends
mid-render with no error, the shell prints `Killed`, and nothing anywhere
mentions memory. There is no CUDA OOM because it was never a VRAM problem.

Read the cgroup, not `free`:

    cat /sys/fs/cgroup/memory.max            # cgroup v2
    cat /sys/fs/cgroup/memory/memory.limit_in_bytes   # v1

**This invalidates casual host-RAM reasoning on any pod**, including OTR's own
DRAM canary, which reads psutil and therefore sees the host. A canary that
believes it has 193 GiB will never warn before a kill at 57.7.

What crossed it here: `ltx25_foley_plus` on its TWO-STAGE PASS, decoding at
1664x960. The video and mime lanes at the same canvas did not. So the two-stage
decode is the expensive step, and 57.7 GiB is not enough for it -- a real
number for the matrix. The legal-length H3 receipts did not capture host RAM
and provide no H3 host-memory sizing signal.

If a pod must run that lane, size the CONTAINER's memory, not the instance's.
