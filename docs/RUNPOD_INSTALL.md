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
