# Old Time Radio (OTR) for ComfyUI

Generate complete radio-drama episodes on a rented GPU — script, cast, voices,
music, video and credits — from a single ComfyUI workflow.

## Description

OTR is a ComfyUI node pack that writes and produces a finished audio-drama
episode end to end. A local LLM writes the script and casts it; each character
gets a distinct synthesized voice; music cues are generated per scene; every
beat is rendered to video; and the whole thing is muxed, captioned and given a
credits roll. The output is a single MP4.

It runs **fully local and offline-first** — no API keys, no paid services, no
cloud calls. The default lane needs no Hugging Face token at all.

Episodes are drawn from source banks: public-domain fiction, Shakespeare, an
original bank, and archive-driven lanes. A 3-act episode runs 4–6 minutes; the
pipeline supports 1 to 5 acts.

**Status: the pack installs and registers cleanly on a pod (verified — 25 node
classes). A full episode render on rented hardware has NOT yet been verified.**
It is proven on local 8 GB and 16 GB NVIDIA cards. Treat pod rendering as
unproven until you have your own successful run.

## Getting Started

### Dependencies

* **NVIDIA GPU.** 8 GB is the proven floor (RTX 4060, nine published episodes);
  16 GB is comfortable. More VRAM does not speed up the default lane much — the
  models are small.
* **A network volume, mounted at `/workspace`.** Strongly recommended. Model
  weights are ~4 GB for the default lane and re-download on every pod start
  without persistent storage. Provision it in the **same region** as the pod or
  it will not attach.
* **Disk:** ~20 GB for the pack, its Python dependencies and the default
  weights. More if you enable additional video lanes.
* **No Hugging Face token needed** for the default haunted lane. Some optional
  lanes (LTX 2.5) use gated models and do require one — set `HF_TOKEN` as a pod
  environment variable if you want those.

### Using the template

Deploy the pod and wait for **ComfyUI on port 8188** to report Ready. On first
boot the start command clones the pack and downloads the default weights, so
the first start takes several minutes longer than later ones.

**Verify the pack actually loaded before doing anything else.** This is the
single most useful check, and a pod can report a pack installed while
contributing zero nodes:

```
curl -s "https://<podId>-8188.proxy.runpod.net/object_info" | python3 -c "
import json,sys; oi=json.load(sys.stdin)
print('classes:', len(oi), ' OTR_:', sum(1 for k in oi if k.startswith('OTR_')))"
```

A non-zero `OTR_` count means you are ready. Zero means the pack did not load —
see Help.

Then either open ComfyUI in a browser and load the workflow from
**Workflow → Browse Templates → EXTENSIONS → comfyui-old-time-radio →
otr_canonical**, or drive it headlessly:

```
COMFYUI_URL=https://<podId>-8188.proxy.runpod.net \
  python3 scripts/otr_canonical_api_run.py \
    --profile otr_nvidia_8gb_haunted \
    --act-count 1
```

`--act-count 1` is the fastest meaningful run. The finished episode lands in
ComfyUI's output directory.

## Help

**"No nodes" / the pack seems missing.** Run the `/object_info` check above. It
is the only thing that proves a pack loaded — a Manager install reporting
success does not. If `OTR_` is zero, ComfyUI is scanning a different directory
than the one the pack was installed into. Find the real one:

```
python3 -c "import folder_paths, os; print(os.path.dirname(folder_paths.__file__))"
```

**`python: command not found`.** Many pod images ship only `python3`. Use
`python3` everywhere; a one-line command whose clone succeeded can still die on
this and read as a total failure.

**Restarting.** `POST /api/manager/reboot` returns **HTTP 502 and works anyway** —
the server drops the connection on its way down. Poll `/system_stats` for a 200
rather than trusting the reboot call's status code.

**Changing launch arguments.** A Manager reboot restarts the ComfyUI *process*
with its existing arguments; it does not re-read the image's argument file. Stop
and start the pod for argument changes to take effect.

**`--highvram`.** Reasonable on a 24 GB+ card. **Do not set it on 8 GB** — it is
hardware-specific, not a general improvement. When in doubt leave VRAM handling
on automatic.

**First render is slow.** Model loads dominate. Later runs on the same pod reuse
what is resident.

**Long renders and client timeouts.** The headless runner watches for a bounded
time and then stops *watching* — it does not stop the render. If it reports a
timeout while the queue still shows work in progress, the episode is still going
and will still finish. Use `--timeout 0` to wait for a terminal result.

## Authors

Jeffrey A. Brick — [@jbrick2070](https://github.com/jbrick2070)

Source: <https://github.com/jbrick2070/ComfyUI-OldTimeRadio> (branch
`v2.0-alpha` — `main` is stale and should not be used)

Built on [ComfyUI](https://github.com/comfyanonymous/ComfyUI) and
[ComfyUI-AnimateDiff-Evolved](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved).

## Version History

* 0.1
    * Initial template. Pack install and node registration verified on a pod
      (1036 → 1061 classes, 25 `OTR_` classes). Episode rendering on rented
      hardware not yet verified.
