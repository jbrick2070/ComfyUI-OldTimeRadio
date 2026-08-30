# Running OTR on RunPod (or any rented ComfyUI pod)

**Status: PARTIALLY PROVEN.** Everything marked MEASURED was done on a real
pod on 2026-08-29. Everything marked UNPROVEN is written from the code or from
one observation and has not been demonstrated. The difference is deliberate —
this document exists because the last attempt failed on something no one had
checked, and a confident guide would have cost the reader the same night it
cost us.

---

## 0. Read this first: the failure that eats the evening

A rented ComfyUI pod can report a node pack as **installed and enabled while
contributing zero nodes**. Not a dependency error, not a crash — the pack is
simply never loaded, and every surface you would naturally check says it is
fine.

**MEASURED, on a RunPod H100 with the `comfyui-mcp` vendor template:**

| query | what it reads | result |
|---|---|---|
| `install_custom_node action:"list" mode:"default"` | `custom_nodes/` on disk **now** | **33 packs**, incl. OTR and AnimateDiff-Evolved |
| `install_custom_node action:"list" mode:"imported"` | the same scan **frozen at ComfyUI startup** | **2 packs** — only the two baked into the container image |

`/object_info` corroborated the second exactly: **847 node classes, every one
core.** Zero `OTR_`, zero `ADE_`, zero `VHS_`, zero KJNodes. It survived a full
container stop/start, so it was not stale process state.

**Why no amount of reinstalling fixes it:** ComfyUI-Manager writes to the
directory it reads, which on that image is not the directory ComfyUI scans.
Every "successful" install lands somewhere invisible. If you are installing
over HTTP with no shell, you cannot see this, and the install will keep
reporting success.

**So the first thing you do on any pod is verify that a pack loads — before
you spend an hour on models.**

---

## 1. The verification that must pass before anything else

Install any small, well-known pack through the Manager, restart, then ask the
server what it actually has:

```bash
curl -s "$POD_URL/object_info" | python -c "
import json,sys
oi = json.load(sys.stdin)
print('total node classes:', len(oi))
for prefix in ('OTR_', 'ADE_', 'VHS_'):
    print('  %-6s %d' % (prefix, sum(1 for k in oi if k.startswith(prefix))))
"
```

**Interpretation, and this is the whole test:**

* A count in the hundreds with **zero** third-party prefixes → you have the
  broken layout above. Stop. Fix the scan path (section 4) or change template.
  Do not install models yet.
* Third-party classes present → the pod loads packs. Proceed.

Do this **before** downloading a single weight. Models are the expensive part
and they are worthless on a pod that will not load the nodes that use them.

---

## 2. What is already proven to work on a pod

**MEASURED, same H100:**

* Pod deploy via the RunPod REST API with a network volume attached.
  Restart from EXITED to serving ComfyUI: **16 s**.
* Model downloads to the volume at datacenter speed — ~36 GB in minutes via
  the Manager's server-side fetch.
* Stock video generation is fast: Wan 2.2 14B t2v, 81 frames @ 832×480,
  4-step LoRA — **48.3 s cold, 35–40 s warm**.

So the hardware, the volume and the transfer speeds are not the problem. The
problem is only ever getting ~30 Python classes to register.

---

## 3. Installing OTR

**Do section 1 first.** Then, in order:

1. **Install the pack.** Either the registry id `comfyui-old-time-radio`, or a
   git clone of `https://github.com/jbrick2070/ComfyUI-OldTimeRadio` on branch
   **`v2.0-alpha`**.

   > **`main` is stale.** It sits thousands of commits behind and still
   > advertises `version = "1.0.0"`. A fresh clone must land on `v2.0-alpha`.

   > **Registry caveat, MEASURED 2026-08-29:** `latest_version` resolves to
   > `2.0.0-alpha.8` (dated Aug 25) because the newer versions are flagged
   > pending review. A registry install therefore gets Aug 25 code. Clone from
   > git for anything current.

2. **Restart ComfyUI** — not just the Manager. A pack that is not in the
   startup scan does not exist.

3. **Re-run the section 1 check.** `OTR_` must be non-zero. If it is zero, no
   later step will help.

4. **Install the node packs OTR's video lanes drive.** These are ComfyUI node
   packs, not pip packages, so `pip install` cannot supply them:

   | lane | needs |
   |---|---|
   | `animatediff15_*` (the low-VRAM haunted lane) | [ComfyUI-AnimateDiff-Evolved](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved) |

5. **Fetch weights to the network volume**, not the container. Point
   `OTR_COMFYUI_MODELS_ROOT` at the volume-backed models directory. OTR
   resolves its models root through that variable, then `COMFYUI_MODELS_ROOT`,
   then a Windows-oriented default that will be wrong on a pod — so on Linux,
   **set it explicitly**.

---

## 4. If packs do not load: the fix needs a shell

**UNPROVEN — this is the reasoned repair, not a demonstrated one.**

Everything above can be done over HTTP. This cannot. Open RunPod's console web
terminal (or add an SSH key) and:

```bash
# find where ComfyUI actually scans, from the running process
ps aux | grep -m1 "[m]ain.py"
python -c "import folder_paths, os; print(os.path.dirname(folder_paths.__file__))"
```

Then make the packs visible to that path — a symlink from the scanned
`custom_nodes/` to wherever the Manager put them, or move them outright, and
restart ComfyUI.

**More GPUs do not help.** A second pod has the identical layout. This is one
directory path, not a capacity problem.

---

## 5. Choosing what to run

**The lane that needs no Hugging Face token at all** — verified by anonymous
download of the real weight blobs:

* writer `google/gemma-4-E2B-it`, SD 1.5, `v3_sd15_mm.ckpt`,
  `v3_sd15_adapter.ckpt`, `Kokoro-82M`, `musicgen-small`. All ungated.
* Profile: `otr_nvidia_8gb_haunted` — **proven on real 8 GB hardware**, three
  consecutive published episodes, ~35–55 min each.

**Set a token anyway.** Anonymous pulls are rate-limited and a throttled
multi-gigabyte fetch is a failed install rather than a slow one. See README
section 2c for the safe ways — and never put a token in a node widget.

**Gated:** the LTX 2.5 lanes need Hugging Face access (`gated: "auto"`,
HTTP 401 anonymously). Everything else in the pack is ungated.

---

## 6. What is NOT established, and should be

* **Only ONE template was tested.** Concluding "pods cannot run OTR" from n=1
  is exactly the error this document warns about. A different ComfyUI template
  may load packs correctly, and finding one is the cheapest possible next
  experiment: deploy, install one small pack, run the section 1 check. About
  ten minutes and roughly a dollar.
* **No OTR episode has ever rendered on a rented pod.** Every timing in
  section 2 is a stock workflow.
* **The shell repair in section 4 has not been performed.**

If you get an OTR episode out of a pod, that is new information — the project
would like to hear about it.

---

## 7. Second template, MEASURED 2026-08-30 — the pack-load problem is different here, and the blocker is OURS

A second pod was tested, which is what section 6 asked for. **The n=1 conclusion
in section 0 does not generalise**: this template does not have the invisible-
directory fault. It has a different wall, and that wall is on our side.

### The pod

    pod        gigantic_magenta_sturgeon (w7rggm1x5d3q7x)
    GPU        RTX 5090, 31.4 GiB          <- double the 5080's 16 GiB
    RAM        109 GB
    ComfyUI    0.26.2
    Manager    V3.41
    services   8188 ComfyUI, 8080 FileBrowser, 8888 JupyterLab

### Everything reachable over HTTP, no shell needed

**The proxy URL pattern is `https://<podId>-<port>.proxy.runpod.net`** and it
works without auth for these endpoints:

    /system_stats                 200   <- GPU model + VRAM, ComfyUI version
    /object_info                  200   <- 1.76 MB, the section-1 test
    /api/manager/version          200   <- "V3.41"
    /api/manager/queue/status     200   <- {"total_count":0,...}
    /api/manager/queue/install    POST
    /api/manager/queue/start      POST

`/system_stats` answers "which GPU did I actually get" in one call and costs
nothing — do it first, before assuming a pod matches its template name.

### This template DOES load third-party packs

`/object_info` returned **1036 node classes** against the 847 in section 0, and
30 `LTX*` plus 43 `Wan*` classes — those are ComfyUI core now, not packs, but the
class count alone shows a fuller install than the broken template. There is no
evidence here of the Manager-writes-where-ComfyUI-does-not-scan fault.

### THE TWO REAL WALLS, both hit in one session

**1. Manager refuses a git-URL install over HTTP.**

    POST /api/manager/queue/install  {"repository": "https://github.com/..."}
    -> HTTP 404  "A security error has occurred. Please check the terminal logs"

This is ComfyUI-Manager's `security_level`, not a bug and not a RunPod policy. A
network-exposed instance will not install an arbitrary git URL. Lowering it means
editing Manager's config, which needs a shell — so it is not a way around
needing one.

**2. A CNR install is ACCEPTED and then does nothing, because our registry
version is Pending.**

    POST /api/manager/queue/install  {"id": "comfyui-old-time-radio", ...}
    -> HTTP 200                       (accepted)
    POST /api/manager/queue/start     -> HTTP 200
    queue status, 60 s later          -> total_count 0, is_processing false
    /object_info                      -> OTR_: 0

Nothing was ever queued. The cause is on our side and is already documented in
`CLAUDE.md` section 7A:

    https://api.comfy.org/nodes/comfyui-old-time-radio
    -> latest_version: null

Versions sit `NodeVersionStatusPending` until Comfy-Org's own cron promotes
them, and **while pending, `latest_version` resolves to null**, so Manager has
no target to resolve. The registry install cannot work for us until a version
goes Active. **This is the single thing most likely to waste the next session's
hour**: the POST returns 200, the queue reports healthy, and nothing installs.

### So the install path needs a shell — and getting one is one click

Both HTTP routes are closed, so a shell is required after all. On this pod there
are three ways in, in order of friction:

1. **Web terminal** — a toggle on the pod's Connect tab ("Enable web terminal",
   default Stopped). One click by the operator, no key handling.
2. **JupyterLab on 8888** — already running (302 to its auth page).
3. **Direct TCP SSH** (`ssh -> <ip>:<port> -> :22`) — needs an SSH key added to
   the account first.

With a shell, the install is the ordinary one:

    cd /workspace/ComfyUI/custom_nodes    # confirm the path ComfyUI SCANS first
    git clone -b v2.0-alpha https://github.com/jbrick2070/ComfyUI-OldTimeRadio
    # restart ComfyUI, then re-run the section 1 check

**Re-run the section-1 `/object_info` check after the restart regardless.** It is
the only thing that proves a pack loaded, and it is two seconds.

### What is still unproven on a pod

* **No OTR episode has ever rendered on rented hardware.** Unchanged from
  section 6.
* The **LTX 2.5 foley lane needs a gated Hugging Face model**, so that lane
  additionally needs a token set as a pod env var by the operator. The ungated
  path in section 5 does not have this problem and is the better first render.
* Cost discipline: a running pod bills whether or not it is doing anything. The
  balance at the time of writing was $65.58.

### 7A. IT WORKS. The pack loads on this template — MEASURED 2026-08-30

    before clone   1036 node classes    OTR_: 0
    after  clone   1061 node classes    OTR_: 25   <- all 25, no skips

**Section 0's conclusion is now formally dead.** A rented pod CAN load this
pack. What follows is the exact path that worked, in order, so nobody derives it
again.

#### Step 1 — find the path ComfyUI actually scans. DO NOT ASSUME IT.

This template does **not** use `/workspace/ComfyUI`. It uses:

    /workspace/runpod-slim/ComfyUI/custom_nodes

The obvious guess fails outright:

    bash: cd: /workspace/ComfyUI/custom_nodes: No such file or directory

Costing a round trip to a wrong guess is exactly what section 4 warned about, and
this document handed out the wrong path anyway. **Look before you clone** — the
JupyterLab file browser shows the tree at a glance, or:

    python -c "import folder_paths, os; print(os.path.dirname(folder_paths.__file__))"

#### Step 2 — get a shell

Both HTTP install routes are closed (section 7), so a shell is required:

* **Web terminal** — a toggle on the pod's Connect tab. It lives in RunPod's own
  console, NOT on a pod-proxied port. Probing `<podId>-19123.proxy.runpod.net`
  and friends returns 404; do not go looking for it there.
* **JupyterLab (8888)** — has token auth on. Every API path returns
  `403 {"message": "Forbidden"}` without it, including `/api/terminals`,
  `/api/contents` and `/api/status`. The bare `/lab` URL is NOT enough; the
  console's link carries `?token=...`. Its **Terminal** works fine once open.

#### Step 3 — clone and install, one line

    cd /workspace/runpod-slim/ComfyUI/custom_nodes && \
      git clone -b v2.0-alpha https://github.com/jbrick2070/ComfyUI-OldTimeRadio && \
      pip install -r ComfyUI-OldTimeRadio/requirements.txt

`-b v2.0-alpha` is mandatory. `main` is thousands of commits behind and still
advertises `version = "1.0.0"`.

#### Step 4 — restart, and IGNORE THE 502

    POST /api/manager/reboot   ->   HTTP 502

**That 502 is the restart working, not failing.** The server drops the
connection as it goes down, so curl reports a bad gateway; ComfyUI was serving
again ~40 s later. Poll `/system_stats` for a 200 instead of trusting the reboot
call's status code.

#### Step 5 — prove it, don't assume it

    curl -s "$URL/object_info" | python -c "
    import json,sys; oi=json.load(sys.stdin)
    print(len(oi), sum(1 for k in oi if k.startswith('OTR_')))"

A non-zero `OTR_` count is the only proof a pack loaded. Two seconds, and it is
the difference between this document's section 0 and this section.

#### What this pod is

    pod      gigantic_magenta_sturgeon (w7rggm1x5d3q7x)
    GPU      RTX 5090, 31.4 GiB      <- double the 5080's 16 GiB
    RAM      109 GB
    ComfyUI  0.26.2   Manager V3.41

#### Still not done

**No OTR episode has yet rendered on rented hardware.** Loading 25 node classes
is not rendering an episode. What remains:

1. **Weights.** Nothing is downloaded yet. Use the ungated bundle in section 5
   (`gemma-4-E2B-it`, SD 1.5, `v3_sd15_mm.ckpt`, `v3_sd15_adapter.ckpt`,
   `Kokoro-82M`, `musicgen-small`) for the first render — it needs no token.
2. **`OTR_COMFYUI_MODELS_ROOT`** must be set explicitly; the default is
   Windows-oriented and wrong on a pod.
3. **ComfyUI-AnimateDiff-Evolved** for the haunted lane; the LTX 2.5 foley lane
   additionally needs a gated Hugging Face model, so that one wants a token set
   as a pod env var.

---

## 8. THE SPEEDRUN — the install is zero-terminal, and WE are what breaks it

Everything in section 7A was done by hand in a JupyterLab terminal. **Almost
none of it had to be.** Manager exposes the whole install over HTTP, and every
endpoint answers on this pod:

    POST /api/manager/queue/install         install a node pack   -> 200
    POST /api/manager/queue/install_model   install a model       -> 500 on an
                                                                     empty body,
                                                                     i.e. it EXISTS
    POST /api/manager/queue/start           run the queue         -> 200
    POST /api/manager/reboot                restart               -> 502, and it
                                                                     works anyway
    GET  /object_info                       prove it loaded       -> 200

`install_model` returning **500 rather than 404** is the tell: the endpoint is
present and only rejected a malformed payload. Working out its exact shape is
maybe half an hour, and it removes the second terminal step (weights) the same
way the first POST removes the first (nodes).

### What actually forced a terminal, and it is ours

    GET https://api.comfy.org/nodes/comfyui-old-time-radio
    -> latest_version: null

Our alpha.13/.14 are still `NodeVersionStatusPending`. Manager therefore has no
target to resolve, the CNR install accepts with **HTTP 200 and installs
nothing**, and the only remaining route is a git clone — which Manager refuses
over HTTP with *"A security error has occurred"* because the instance is network
exposed. **Two closed doors, one root cause, and the root cause is on our side
of the fence.**

This is the same Pending state `CLAUDE.md` section 7A already blames for
ComfyUI-Manager reporting "not a CNR node" on nightly installs. It now also
costs a pod install its zero-touch path. **Check the registry FIRST**: if
`latest_version` is non-null, the speedrun below works and nobody types
anything.

### The speedrun, once a version is Active

No SSH key, no Jupyter token, no web terminal. From any machine that can reach
the pod:

    URL=https://<podId>-8188.proxy.runpod.net

    1. GET  $URL/system_stats          # which GPU did I actually get?
    2. POST $URL/api/manager/queue/install
            {"id": "comfyui-old-time-radio", "version": "latest",
             "selected_version": "latest", "channel": "default", "mode": "remote"}
       POST $URL/api/manager/queue/install     # same, for AnimateDiff-Evolved
       POST $URL/api/manager/queue/install_model   # x4, the ungated bundle
    3. POST $URL/api/manager/queue/start
       GET  $URL/api/manager/queue/status      # poll to total_count 0
    4. POST $URL/api/manager/reboot            # expect 502; poll /system_stats for 200
    5. GET  $URL/object_info                   # OTR_ non-zero == it loaded
    6. COMFYUI_URL=$URL python scripts/otr_canonical_api_run.py \
            --profile otr_nvidia_8gb_haunted --act-count 1

Step 6 works remotely because the runner's model preflight validates against the
**server's** `/object_info`, not local disk — so a short fetch is refused in
seconds instead of failing twenty minutes into a render.

### Three tiers of frictionless, in dependency order

1. **Get a registry version Active.** Fixes it for every user, not just pods,
   and turns "clone a repo in a terminal" into clicking Install. Everything else
   is downstream of this one.
2. **Drive the Manager API** as above. Needs the `install_model` payload shape.
3. **A custom RunPod template** with pack and weights baked in — genuinely
   zero-step for someone who is not the author, also the most work, and it goes
   stale the moment the pack moves.

### Terminal gotchas, if a shell is unavoidable anyway

* **`python` is NOT on PATH on this image — only `python3`.** The fetcher died
  on `bash: python: command not found` after the clone in the same one-liner had
  already succeeded, which reads as a total failure and is not.
* **`--highvram` is not in the default `argv`** (`main.py --listen 0.0.0.0
  --port 8188 --enable-cors-header`). Worth adding on a 31.4 GiB card. **It is
  wrong on 8 GB — do not put it in a general recommendation without binding it
  to VRAM.**
* `comfy-aimdo` (DynamicVRAM) ships on this image. That is the component that
  called native `abort()` and killed a 4060 episode in PBUG-20260829-03.

### 8A. CORRECTION — we are FLAGGED, not Pending, and that changes the diagnosis

Section 8 blamed the zero-terminal install failing on our registry versions
being *Pending*. **That was wrong.** Measured 2026-08-30:

    https://api.comfy.org/nodes/comfyui-old-time-radio/versions
    2.0.0-alpha.14   NodeVersionStatusFlagged
    2.0.0-alpha.13   NodeVersionStatusFlagged
    versions: 2      active: 0

**No Active version exists at all** -- alpha.8 and earlier went away with the
listing that was deleted and recreated. So `@latest` has nothing to resolve to.

**The Manager API itself works perfectly.** Proven on the same pod, same
session, zero terminal: ComfyUI-AnimateDiff-Evolved installed over HTTP because
it IS Active (1.5.7), taking the pod from 1036 to 1181 node classes with 143
`ADE_` classes registered.

    POST /api/manager/queue/install
      {"id": "<cnr-id>", "version": "latest", "selected_version": "latest",
       "channel": "default", "mode": "remote"}
    POST /api/manager/queue/start
    POST /api/manager/reboot        # 502, works anyway
    GET  /object_info               # confirm

`version`, `channel` and `mode` are read with `json_data['...']`, not `.get()`,
so all three are MANDATORY -- omitting any is a 500, which is what a malformed
`install_model` body returns too.

**The `nightly` branch cannot substitute.** Setting `selected_version:
"nightly"` makes Manager read `repository` and route through
`get_risky_level(git_url, pip)` then `is_allowed_security_level(...)`, which
refuses an arbitrary git URL on a network-exposed instance:

    HTTP 404  "A security error has occurred. Please check the terminal logs"

**Consequence, and it is bigger than pods.** Flagged does NOT self-resolve the
way Pending does -- Pending waits for Comfy-Org's cron, Flagged means their
scanner objected. Until a version goes Active, **nobody can install OTR through
ComfyUI-Manager by any route**: not `@latest` (nothing to resolve), not
`nightly` (security), not on a pod and not on a desktop. A git clone is the only
path, which is exactly the friction the template's start command exists to
absorb -- a start command runs as container init and never meets Manager's
security policy at all.

---

## 9. THE TEMPLATE SPEC — current requirements, kept in place

**This is a LIVING spec, not a version history.** When something changes, edit
the cell; do not append a v3. What matters is what a template needs *today* —
nobody ever needs to know what it needed last week.

Every item below cost real time to discover and none of it is guessable from
the outside.

### Config

| field | value | why |
|---|---|---|
| Region | **EU-RO-1** | where GPUs actually schedule; US-CA-2 never gave one |
| Network volume | mounted at `/workspace` | 4 GB of weights re-download every boot without it |
| Container disk | > 5 GB | pack + torch deps land here, not on the volume |
| HTTP port | `8188` **ComfyUI** | everything drives through it |
| HTTP port | `8888` **JupyterLab** | the ONLY shell we ever got; keep it |
| HTTP port | `8080` FileBrowser | optional — returned 401 all session, never used |
| UDP ports | none | nothing here uses UDP; SSH is TCP |
| `--highvram` | **NOT in a shared template** | right on 31.4 GiB, WRONG on 8 GB |

### Environment

    OTR_COMFYUI_MODELS_ROOT=/workspace/runpod-slim/ComfyUI/models

**Not optional.** Without it `_models_root()` falls through to its Windows
default and the fetcher writes ~4 GB into a literal directory named
`C:\ComfyUI-Models`, which ComfyUI never scans. It reports success. This burned
a whole diagnosis cycle: the first pod's checkpoint list stayed empty and was
read as a failed download when the bytes were on disk the whole time.

Set it at DEPLOY time. A Manager reboot restarts the ComfyUI process with its
existing environment and argv — it does not re-read the image's config, so
post-hoc changes appear not to take.

### Start command

Runs as container init, which is why it works at all: it never meets
ComfyUI-Manager's `security_level`, and it does not care that our registry entry
is Flagged. Shape:

    CN=/workspace/runpod-slim/ComfyUI/custom_nodes
    [ -d "$CN/ComfyUI-OldTimeRadio" ] || \
        git clone -b v2.0-alpha <repo> "$CN/ComfyUI-OldTimeRadio"
    python3 -m pip install -q -r "$CN/ComfyUI-OldTimeRadio/requirements.txt"
    python3 "$CN/ComfyUI-OldTimeRadio/scripts/otr_fetch_lane_weights.py" haunted
    <exec the image's normal entrypoint>

* **Idempotent guards** so a restart does not re-clone.
* **`python3`, never `python`** — this image has no `python` on PATH, and a
  one-liner whose clone already succeeded dies on it in a way that reads as
  total failure.
* **`-b v2.0-alpha` is mandatory** — `main` is thousands of commits behind.
* **It must end by exec'ing the image's real start.** A start command REPLACES
  the entrypoint; get this wrong and the pod boots and serves nothing, silently
  — the same failure shape as section 0.
* **AnimateDiff-Evolved does NOT need to be in here.** It is Active on the
  registry and installs over plain HTTP with no terminal (proven: 1036 -> 1181
  classes, 143 `ADE_`). Bake it in only if you want the template offline-complete.

### Verify, every time

    GET /object_info   ->  OTR_ non-zero

The only thing that proves a pack loaded. A Manager install reporting success
does not, and neither does the pack appearing on disk.

### README

`docs/RUNPOD_TEMPLATE_README.md` — 4,993 characters, fits RunPod's 5,000 limit
(count decoded characters, not bytes; the em-dashes are multibyte and `wc -c`
overstates by ~60).

### 9A. NVENC, not ffmpeg — a build-time capability test that lies at runtime

**This section previously said the pod image lacked ffmpeg. That was wrong.**
ffmpeg was already installed (`7:6.1.1-3ubuntu5`, `apt-get` reported
`0 newly installed`). The real fault was one level down and far more
interesting.

**What actually happened.** The first OTR render ever submitted to rented
hardware reached `t=404s` — writer finished, script written, audio done — then
died with `BrokenPipeError` at `OTR_SignalLostVideo`. ffmpeg's own log named it
exactly:

    Stream #0:0 -> #0:0 (rawvideo (native) -> h264 (h264_nvenc))
    [h264_nvenc] Cannot load libnvidia-encode.so.1
    [vost#0:0/h264_nvenc] Error while opening encoder
    Conversion failed!

The container exposes CUDA for compute but **not `libnvidia-encode.so.1`**, so
hardware h264 encoding is unavailable. That is normal for rented GPUs and for
most Docker setups; it is not a RunPod defect.

**Why our CPU fallback did not save it.** `video_engine._check_nvenc` decided
with:

    "h264_nvenc" in (ffmpeg -codecs output)

That answers *"was ffmpeg COMPILED with nvenc"*, not *"can nvenc RUN here"*.
Ubuntu builds ffmpeg `--enable-nvenc`, so the string was present, the check said
yes, and the already-existing `libx264` fallback never got a chance to fire.

**Fixed in `bd6bd936`** — the check now encodes one frame to `-f null -` and
believes the result.

**Two things worth carrying away, both about how this failed rather than what
failed:**

* **It failed SEVEN MINUTES IN, after the expensive work was done.** Nothing
  earlier in the pipeline touches the encoder, so a pod passes every install
  check — 25 `OTR_` classes, 143 `ADE_`, all four models registered — and still
  cannot produce a frame. Install verification does not imply render capability.
* **A capability probe can be wrong in the safe-looking direction.** The first
  attempt at the fix probed with a 64x64 canvas, which NVENC refuses outright
  ("Frame Dimension less than the minimum supported value") — so a *healthy*
  5080 probed as unavailable and would have been silently pushed onto CPU
  encoding. Measured before and after on the working machine, caught, raised to
  256x256. **Probe at a realistic size, and always measure the machine that
  already works.**

**A note on PyAV:** `av` is a ComfyUI core dependency and ships its own bundled
ffmpeg libraries, so an in-process probe succeeds regardless. That bundled build
is not on PATH and is not what `subprocess.Popen` finds. PyAV working proves
nothing about the binary or about NVENC.
PyAV working proves nothing about the binary.

---

## 10. THE PLAYBOOK — what actually works, in order (MEASURED 2026-08-30)

Sections 0-9 are the archaeology. This is the procedure.

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
| writer / voice / music | `HF_HOME=/workspace/hf` | **defaults to `/root/.cache`, which is on the 55 GB CONTAINER disk and is erased on stop.** Without this the 24 GB writer re-downloads every session even with a volume attached |

**The expensive half is the one that defaults to the disposable place.** Video
weights are 3.7 GB; the HF cache with both writers is **38 GB**.

ComfyUI reads `HF_HOME` at start, so a cache created afterwards is invisible to
it. Symlink rather than fight the env:

    ln -s /workspace/hf /root/.cache/huggingface

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

## 11. Lessons the hard way -- read this before debugging a pod

Each of these cost real time or real money on 2026-08-30. They are here so the
next session does not re-buy them.

### `pgrep -f "ComfyUI/main.py"` MATCHES ITS OWN SSH COMMAND

    ssh pod 'pgrep -f "ComfyUI/main.py"'   ->  returns the pid of THAT shell

The pattern is present in the command line you just sent, so `pgrep` finds your
own `bash -c`, you kill it, and the follow-up listing comes back empty -- which
reads exactly like "ComfyUI is dead." It is not. **The real process is
`python main.py --listen 0.0.0.0 --port 8188`** -- a RELATIVE path that pattern
never matched to begin with.

**Identify the server by its listening port, never by a name pattern:**

    ss -lptn | grep 8188        ->  LISTEN  353/python

This produced a confident and completely wrong "ComfyUI is not running on the
pod" before the port check corrected it.

### AN EMPTY QUEUE IS NOT READINESS -- it is also what a DEAD server reports

A sweep gated on `queue_running + queue_pending == 0` started the instant
ComfyUI could answer `/queue`, while `/object_info` was still returning the
RunPod **502 "Waiting for service to respond"** page. All 25 lanes failed in
about a second each and the run looked like 25 broken lanes.

**Readiness is three conditions together:**

1. `/object_info` returns 200 -- the SERVER is answering, not the proxy
   placeholder,
2. it contains `OTR_` classes -- the PACK is loaded, not merely ComfyUI,
3. the queue is empty -- so a wall-clock number measures a render and not a wait.

Gate before the sweep AND before every leg; one wedged lane otherwise poisons
every lane after it. `scripts/`-adjacent helper: `pod_ready.py` in the session
scratchpad does exactly this.

### NOTHING RESTARTS COMFYUI FOR YOU on `runpod/comfyui:cuda13.0`

There is no supervisor. Kill it and the port stays dead until you relaunch it
yourself, with the container env recovered from `/proc/1/environ` (an SSH
session does not inherit it):

    cd /workspace/runpod-slim/ComfyUI
    eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^(OTR_COMFYUI_MODELS_ROOT|HF_HOME|HF_TOKEN)=' | sed 's/^/export /')"
    nohup .venv-cu128/bin/python main.py --listen 0.0.0.0 --port 8188 \
          --enable-cors-header > /workspace/comfyui.log 2>&1 &

### THE IMAGE'S COMFYUI LOGS TO A PIPE, SO THERE IS NOTHING TO TAIL

    ls -l /proc/<pid>/fd/1   ->   pipe:[636008536]

Stage has to be read from `/queue` and from files appearing under
`output/otr/episodes/`. **Relaunching it yourself with `> /workspace/comfyui.log`
is worth doing for that reason alone** -- it turns a blind pod into one you can
read.

### A ROLLED SOURCE BANK TURNS A LANE SWEEP INTO BANK ROULETTE

The first sweep rolled `source_bank`, drew `scifi_news_pro`, and died at 82s in
`_llm_rank_news_candidates` with `_LLMTimeoutWorkflowPause`. That number
describes the news bank's LLM ranking call. It says **nothing** about the video
lane that leg was supposed to be measuring.

**Pin the bank and the style so the lane is the only variable.** `original` is
the right control: fully local, no RSS fetch, no source document.

### VERIFY WHAT THE DELIVERABLE ACTUALLY IS -- `/history` will mislead you

See PBUG-20260830-24. A `RESULT SUCCESS` leg publishes an episode that
`/history` does not list; the only video it advertises is the intermediate in
`audio/`. Pull published episodes by listing the obs DIRECTORY over SSH:

    python scripts/otr_pod_obs_bridge.py <podId> --host <ip> --port <port>

### Measured, so it is not re-guessed

    RTX PRO 4000 Blackwell 24 GB   1-act, gemma-4-12b writer, AnimateDiff
                                   haunted lane, 8 clips
                                   2058 s  (34.3 min)  RESULT SUCCESS
                                   VRAM peak 15,990 MB

---

## 12. The template needs updating, and `HF_HOME` is the reason

**Measured on the live pod 2026-08-30.** The template currently injects only:

    OTR_COMFYUI_MODELS_ROOT=/workspace/runpod-slim/ComfyUI/models
    JUPYTER_PASSWORD=<generated>
    PYTHONUNBUFFERED=1

**`HF_HOME` is absent**, and the cache survives on this pod only because of a
symlink made by hand:

    /root/.cache/huggingface -> /workspace/hf        # NOT in the template
    /workspace/hf  =  84 GB

A fresh pod from today's template gets no symlink, so `HF_HOME` falls back to
`/root/.cache/huggingface` on the 55 GB container disk -- which cannot even hold
84 GB, and is erased on stop regardless. **Every new pod would re-download the
whole cache.** That is the single most valuable field to add.

### `HF_HOME` IS the best practice, and the volume proves the layout

One variable, not three. `HF_HOME` is the modern root; `HF_HUB_CACHE` defaults
to `$HF_HOME/hub` and `TRANSFORMERS_CACHE` is deprecated in favour of it. Do not
set the legacy names -- setting several is how a cache ends up split across two
roots with neither complete. The volume confirms the layout is right:

    /workspace/hf/hub     <- model repos
    /workspace/hf/xet     <- xet chunk store

### Template fields to set

| field | value | why |
|---|---|---|
| `OTR_COMFYUI_MODELS_ROOT` | `/workspace/runpod-slim/ComfyUI/models` | already correct, keep |
| `HF_HOME` | `/workspace/hf` | **add** -- 84 GB, re-downloaded every pod without it |
| `HF_TOKEN` | a RunPod **Secret** reference | **add** -- gated lanes only |

### The HF token: use a RunPod Secret, and never a plain env var

RunPod's own console says it plainly -- *"Environment variables are not
encrypted"* -- so a token pasted into the env field is stored in the clear and
is visible to anyone who can read the template. Create it under **Secrets** in
the RunPod account, then reference it from the template's env value; the console
shows the exact `{{ RUNPOD_SECRET_<name> }}` string to paste when the secret is
created. Secrets are injected at pod creation, exactly like SSH keys, so **a
secret added to a RUNNING pod does not reach it** -- redeploy.

**Do NOT run `huggingface-cli login` on the pod.** With `HF_HOME` on the volume
it writes the token in plaintext to `/workspace/hf/token`, where it then
persists on shared storage across every pod that mounts that volume, long after
the pod that typed it is gone. The Secret route keeps the token out of the
volume and out of any terminal history.

**The token is only needed for GATED repos.** Everything the default path uses
is ungated: the writer, voices, music, the haunted lane, `minimax_h3` and
`wan_ti2v` all fetch with no account. The one gated video repo is
`Lightricks/LTX-2.5`, which reports `gated: "auto"` and needs the terms accepted
once, by hand, on the model page -- `scripts/otr_fetch_lane_weights.py`
deliberately refuses to offer it, because a script must never paper over a
licence click.
