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
