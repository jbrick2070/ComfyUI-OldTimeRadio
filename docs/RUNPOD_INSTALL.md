# Standing up OTR on a rented GPU

For the reproducible HuMo/LTX 2.5 high-RAM qualification sequence, use
[`RUNPOD_PORTABILITY_LAB.md`](RUNPOD_PORTABILITY_LAB.md). This file is the
supporting RunPod history, template guide, and failure atlas. Every claim is
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
| video weights | `OTR_COMFYUI_MODELS_ROOT=/workspace/runpod-slim/ComfyUI/models` | set in the TEMPLATE; without it `_models_root()` falls back to `C:\ComfyUI-Models`, which on Linux becomes a literal directory nothing scans -- and reports success |
| writer / voice / music | `HF_HOME=<models_root>/huggingface` | **defaults to `/root/.cache`, which is on the 55 GB CONTAINER disk and is erased on stop.** Without this the 24 GB writer re-downloads every session even with a volume attached. **Use the models root, not an invented path -- see below.** |

**The expensive half is the one that defaults to the disposable place.** Video
weights are 3.7 GB; the HF cache with both writers is **38 GB**.

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

    ssh <pod> 'cat > /root/provision.sh' < scripts/otr_pod_provision.sh
    ssh <pod> 'nohup bash /root/provision.sh > /root/prov.log 2>&1 &'

**Run it DETACHED.** A foreground SSH session dies on any client-side timeout
and takes the install with it -- this cost one full run mid-`pip install`.

The script locates the tree ComfyUI actually scans (**do not assume
`/workspace/ComfyUI`; this image uses `/workspace/runpod-slim/ComfyUI`**),
recovers `OTR_COMFYUI_MODELS_ROOT` from `/proc/1/environ` because **SSH sessions
do not inherit container env**, clones both packs, installs deps, and fetches
the ungated bundle.

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
    first provision                ~6 min     (packs + 3.7 GB weights)
    HF warm, both writers          38 GB, once per VOLUME -- not per pod

---

---

## 2. Template spec

**A LIVING spec.** Edit the cell; do not append a version. What matters is what
a template needs TODAY.

### Config

| field | value | why |
|---|---|---|
| Region | **EU-RO-1** | where GPUs actually schedule; US-CA-2 never gave one |
| Network volume | mounted at `/workspace` | weights re-download every boot without it. Selecting a network volume REPLACES the volume-disk setting -- do not set both |
| Container disk | **60 GB** | pack, torch deps and a native index-tts venv land here, not on the volume |
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

**`OTR_COMFYUI_MODELS_ROOT` is not optional.** Without it `_models_root()` falls
through to its Windows default and the fetcher writes gigabytes into a literal
directory named `C:\ComfyUI-Models`, which ComfyUI never scans -- reporting
success the whole way.

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

    UNPROVEN everything else. No episode has completed on a pod using
             LTX 2.5, wan_ti2v, ltx_8gb, HuMo, indextts2 voice cloning,
             Stable Audio 3, or ANY image engine.

**And the proven one is the lane you would not rent a big card for.**
AnimateDiff SD1.5 is the 8 GB FLOOR lane -- it exists so a small card can
produce anything at all, and a 16 GB dev box runs it overnight for the cost of
electricity. Renting 24 GB to run it is paying for headroom and then not using
it. A rented tier earns its cost by running what the local box CANNOT: the large
video lanes, voice cloning rather than the cheap fallback, and the better image
engines. **None of that is proven yet**, so do not read the PROVEN line above as
"the pod works" -- read it as "the pod works for the one case that needed it
least."

---

## 3. Models and assets

Sections 1-2 are about standing a pod up. This one is about filling it, which
turned out to be the harder half.

### THE REAL PORTABILITY GAP: engines name FILES, not SOURCES

Exactly **three** Hugging Face repo ids exist in all of OTR's engine code
(`hexgrad/Kokoro-82M`, `facebook/musicgen-small`,
`stabilityai/stable-audio-open-1.0`). Every video and image engine names the
FILENAME it wants and nothing about where to get it. Only the three lanes in
`scripts/otr_fetch_lane_weights.py` had a recorded provenance.

**So every other weight on the reference machine was placed there by hand, and
nobody else could obtain it from anything in this repository.** That is the
single biggest barrier to somebody else running OTR -- larger than the registry
being flagged, larger than any install step -- and it is invisible from the dev
box, where the files have simply always been present.

`docs/MODEL_ASSET_INDEX.md` (generated by `scripts/otr_asset_index.py`) now
answers "to use engine X, download Z". Closing a lane means resolving its files
on the Hub, verifying repo AND path AND size, reading the destination folder off
the engine's own resolver, and adding a row to the fetcher.

**Do not guess a repo to close a lane faster.** HuMo 14B is now closed by the
five-file, commit-pinned, SHA-256-verified `humo` lane; it downloads the exact
Kijai `...scaled_KJ` UNET the engine resolves. HuMo 1.7B stays separate and its
exact rename/destination is documented in the portability lab. A 1.7B profile
never downloads the 14B DiT or LoRA.

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
provisioning and the ComfyUI launch. The complete command is below and in
[`RUNPOD_PORTABILITY_LAB.md`](RUNPOD_PORTABILITY_LAB.md#5-indextts2-reference-voices).

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
export OTR_INDEXTTS2_ROOT=/workspace/index-tts
export OTR_INDEXTTS2_VENV="$OTR_INDEXTTS2_ROOT/.venv/Scripts/python.exe"
export OTR_INDEXTTS2_DIR="$OTR_INDEXTTS2_ROOT/checkpoints"
export OTR_INDEXTTS2_WORKER="$OTR_COMFY_ROOT/custom_nodes/ComfyUI-OldTimeRadio/scripts/_otr_indextts2_worker.py"

mkdir -p /workspace/otr-config
"$OTR_COMFY_ROOT/.venv/bin/python" \
  scripts/otr_make_portable_voice_bank.py \
  --models-root "$OTR_COMFYUI_MODELS_ROOT" \
  --male-wav /absolute/path/to/authorized-male.wav \
  --female-wav /absolute/path/to/authorized-female.wav \
  --output /workspace/otr-config/voice_reference_bank.portable.json
export OTR_VOICE_REFERENCE_BANK=/workspace/otr-config/voice_reference_bank.portable.json

"$OTR_COMFY_ROOT/.venv/bin/python" scripts/otr_provision.py \
  --profile otr_w45_humo_14b_169 --with-indextts2
```

Use the exact profile being installed in the last command. Preserve all five
exports in the ComfyUI service environment too. The generated bank contains
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

## 4A. What the starter ships, and why it is not LTX 2.5

**Settled 2026-09-01 by a kibitz round on the three pod blockers.** Everything in
this section is a decision with a receipt, not a preference.

### The starter ships `wan22_high_video`

`config/profiles/otr_runpod_starter.json` selects **`wan22_high_video`**
(internally `wan_ti2v`), because it is the only lane that is simultaneously:

* **proven off the development machine** -- rendered end to end on a rented RTX
  A4500 (Ampere, 20 GB) in 3,949 s, and carrying **eight published episodes** in
  the ledger;
* **completely fetchable by a stranger** -- the `wan_ti2v_gguf` lane is 9.37 GB
  and self-contained (GGUF unet + GGUF encoder + Wan 2.2 VAE);
* **honest about its canvas** -- the engine declares `render_canvas = (832, 480)`
  and the profile declares the same pair, so the config an operator reads is the
  canvas that renders.

The engine's `OTR_ENABLE_WAN_TI2V` flag is a vestigial opt-in, not a hardware
gate. The provisioner now exports it and appends it to `/root/.bashrc`.

### Why LTX 2.5 is NOT in the starter

The clean-machine loader path now exists: the provisioner pins ComfyUI-GGUF and
ComfyUI-LTXVideo, applies OTR's in-repo Gemma-4/BF16 patch, and the engine
semantically verifies the registered loader. The remaining starter gate is a
full remote receipt. The earlier Ampere pod cleared the loader but was SIGKILLed
at its 57.7 GiB container limit during the two-stage 1664x960 decode. That is a
negative receipt for that RAM cap, not proof against Ampere or RunPod.

The five LTX weights are also intentionally not a blind public bundle: four are
gated by the Lightricks terms. `MANUAL_TIERS["ltx25"]` owns their exact paths,
revisions, byte counts, and SHA-256 values, while
`docs/RUNPOD_PORTABILITY_LAB.md` gives the post-terms copy/paste procedure.

LTX 2.5 enters the starter when a clean pod publishes an episode with it, and not
before.

### H3: technically portable, but excluded from this operator's RunPod policy

Under the current signed operating standard in
`docs/H3_LICENSE_ATTESTATION.md`, this operator runs H3 only on owned hardware,
offline, and never redistributes its weights. That is an authorization boundary,
not a GPU limitation; third parties must review and obtain whatever authorization
applies to them.

OTR uses `comfy_extras/nodes_minimax_h3.py` from current ComfyUI core. It does
not require the optional Larry Turbo pack. Comfy-Org explicitly documents its
NVFP4 H3 encoder as usable without Blackwell and also publishes other H3
artifacts. The complete explicit operator-local lane is:

    python scripts/otr_fetch_lane_weights.py minimax_h3

It pins five files totaling 63,440,965,087 bytes (59.084 GiB) and is never
auto-selected by a profile or machine bundle. Legal local 5080 receipts at 124
model / 129 canvas frames measured 6,315 MB FL2VA and 6,678 MB REF2VA absolute
VRAM; host RAM was not captured. The physical RTX 4060 separately produced
receipt-bearing 864x480x90 cold/warm/warm H3 clips and retained 864x480x124
Ref2VA A/V artifacts. That makes isolated H3 clips LAB-PROVEN on the card; the
full canonical OTR episode remains unqualified, not proven impossible. The
public `mkhamra/quibble-h3` repository is a Ref2VA workflow/case study, not an
OTR node provider or qualification receipt.

### humo is not blocked and never was

The ledger records **32 published humo episodes** across four engine variants.
The 14B download gap is now closed by
`python scripts/otr_fetch_lane_weights.py humo`; the remaining portability gate
is a live canonical episode on the rented hardware, not missing download
provenance.
(An earlier draft of the blockers document claimed humo had never produced an
episode. That claim came from grepping the wrong file -- `episode_canon.json`
records no engines at all. The authority is
`meta.render_engines.per_clip[].delivered_engine` in the episode ledger.)

### THE DEFECT THIS ROUND ACTUALLY FOUND -- a provisioned pod could not render

Three faults, all fixed, all invisible to a green log:

1. **The music model was never fetched.** The provisioner fetched one video lane
   and one image precision and called that "the minimum that renders an episode."
   `otr_fetch_lane_weights.py`'s own `MINIMUM_HINT` names
   **haunted + one z_image precision + stable_audio_3**, and the `stable_audio_3`
   lane note reads *"without it EVERY profile fails at the music node."* The
   provisioner now fetches it by default (`OTR_PROVISION_AUDIO_LANE` to override).
2. **A failed required download still exited 0.** The script is `set -uo pipefail`
   with no `-e` -- deliberately, so benign non-zero does not abort a long
   provision. But required weight lanes now increment a failure counter and the
   script exits 1 with `provision INCOMPLETE`.
3. **The starter had no preflight.** `preflight.required_models` was `[]`; it now
   names seven real checkable filenames. The z_image DIFFUSION model is
   deliberately excluded -- the provisioner picks its precision from the card, so
   pinning one variant would fail the check on hardware that correctly chose
   another.

**The provisioner's default video lane now matches the starter profile** (chosen
by VRAM: >= 16 GB gets `wan_ti2v_gguf`, smaller cards get the AnimateDiff floor).
A default that disagrees with the one profile written for newcomers is the same
bug as the missing music model.

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

    # 3. everything else, in one command
    cd ComfyUI-OldTimeRadio && bash scripts/otr_pod_provision.sh

That single command upgrades ComfyUI if it is below v0.32.0, installs the system
libraries, clones the node packs AND their own requirements, repairs any
dangling venv interpreter, picks the image lane from compute capability and
VRAM, and fetches the minimum weights. Every one of those steps exists because
skipping it cost a leg.

Choose the weights explicitly if the defaults are wrong for the box:

    OTR_PROVISION_LANES="haunted" OTR_PROVISION_IMAGE_LANE=z_image_int8 \
        bash scripts/otr_pod_provision.sh

`python scripts/otr_fetch_lane_weights.py --list` prints every lane with its
size and how much is already on disk.

### What you give up

Nothing about correctness -- a no-volume pod renders the same episodes. What you
lose is the ability to stop and resume: every model is re-downloaded on the next
pod, which for the base install is roughly 25 GB of transfer before the first
frame. If you expect to run more than one pod, the volume pays for itself on the
second.

### The starter workflow -- one good everything, dropdowns already set

`workflows/variants/otr_runpod_starter.json` is the from-scratch answer for a
RunPod box with **16 GB of VRAM or more**. Load it and every dropdown is already
on a choice that has produced a real episode -- no picking, no guessing:

    video    ltx25_high_video (16:9)   LTX 2.5, the flagship lane
    image    z_image_turbo             482 episodes
    voice    indextts2 (characters)    1430 episodes
             kokoro (announcer)        64 of the last 64
    music    stable_audio_3            925 episodes
    writer   Mistral-Nemo-Instruct     what the proven LTX 2.5 episodes used
    bank     default                   mixed cast; kokoro_builtin would force
                                       every voice to kokoro and contradict the
                                       character engine above

Nothing there is aspirational. Each value is the one carried by episodes on
disk, which is why the writer is Mistral-Nemo rather than the gemma-4-12b the
machine matrix declares for this class -- the matrix states a class default, the
starter ships what actually rendered.

**It is GENERATED, never hand-edited.** Change `config/profiles/
otr_runpod_starter.json` and re-run:

    python scripts/build_variants.py --profiles otr_runpod_starter

Editing the variant directly survives a diff and then fails regeneration.

Weights for exactly this set, and nothing else:

    python scripts/otr_fetch_lane_weights.py z_image      # or z_image_blackwell
    python scripts/otr_fetch_lane_weights.py stable_audio_3

LTX 2.5's own weights are gated -- accept the terms on Hugging Face, set
HF_TOKEN, and note that ComfyUI must be **v0.32.0 or newer** or the lane cannot
resolve its nodes at all (section 0 of DEPENDENCIES).

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

The reference machine's copy is modified, and the delta is small and specific:

    loader.py   17 lines   "gemma4" added to TXT_ARCH_LIST;
                           LTXV_BF16_PARAMETERS; a BF16 dequant path for raw
                           LTXAV parameters that GGMLOps does not handle
    nodes.py    76 lines
    ops.py, dequant.py     identical to upstream

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
