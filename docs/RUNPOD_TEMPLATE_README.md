# OTR — Old Time Radio for ComfyUI

Generate complete radio-drama episodes on a rented GPU — script, cast, voices,
music, video and credits — from one ComfyUI workflow.

## Description

OTR is a ComfyUI node pack that writes and produces a finished audio drama end
to end. A local LLM writes and casts the script; each character gets a distinct
synthesized voice; music cues are generated per scene; every beat is rendered to
video; and the result is muxed, captioned and given a credits roll. Output is a
single MP4.

It runs **fully local** — no API keys, no paid services, no cloud calls. The
default lane needs no Hugging Face token.

Episodes draw from public-domain fiction, Shakespeare, an original bank and
archive lanes. A 3-act episode runs 4–6 minutes; 1 to 5 acts are supported.

**Status: a full episode has rendered and published on rented hardware.**
Verified 2026-08-30 on an RTX PRO 4000 Blackwell (24 GB): a 1-act episode in
34 min, 8 clips, VRAM peak 15,990 MB. Also proven on local 8 GB and 16 GB
NVIDIA cards.

## Getting Started

### Dependencies

* **NVIDIA GPU.** 8 GB is the proven floor (RTX 4060, nine published episodes);
  16 GB is comfortable. Extra VRAM adds little — the models are small.
* **A network volume mounted at `/workspace`, plus `HF_HOME` pointing INTO
  it.** Set `HF_HOME=<models_root>/huggingface` -- OTR's convention. Any other path gives you two caches and downloads everything twice. Without it the writer/voice/music cache
  (tens of GB) lands on the container disk and is erased on every stop, so a
  volume alone does not save you the download. Create the volume in the **same
  region** as the pod or it will not attach.
* **Disk:** ~20 GB for the pack, its dependencies and default weights.
* **No Hugging Face token needed** for the default haunted lane. Some optional
  lanes (LTX 2.5) use gated models and need `HF_TOKEN` set as a pod environment
  variable.

### Using the template

Wait for **ComfyUI on port 8188** to report Ready, then open a terminal
(JupyterLab on 8888, or the web terminal) and install:

```
cd $(python3 -c "import folder_paths,os;print(os.path.dirname(folder_paths.__file__))")/custom_nodes
git clone -b v2.0-alpha https://github.com/jbrick2070/ComfyUI-OldTimeRadio
cd ComfyUI-OldTimeRadio && python3 -m pip install -r requirements.txt
OTR_COMFYUI_MODELS_ROOT=$(python3 -c "import folder_paths,os;print(os.path.join(os.path.dirname(folder_paths.__file__),'models'))") \
  python3 scripts/otr_fetch_lane_weights.py haunted
```

`-b v2.0-alpha` is **mandatory** — `main` is thousands of commits behind.
Use `python3`; many images have no `python` on PATH.

Restart ComfyUI, then **verify the pack loaded** — this is the only real proof,
and a pod can report a pack installed while contributing zero nodes:

```
curl -s "$URL/object_info" | python3 -c "import json,sys; oi=json.load(sys.stdin); print('OTR_:', sum(1 for k in oi if k.startswith('OTR_')))"
```

Non-zero means ready. Then load the workflow in the browser from
**Workflow → Browse Templates → EXTENSIONS → comfyui-old-time-radio →
otr_canonical**, or drive it headlessly with
`scripts/otr_canonical_api_run.py --profile otr_nvidia_8gb_haunted
--act-count 1`.

## Help

**Weights download but ComfyUI can't see them.** Set
`OTR_COMFYUI_MODELS_ROOT` as shown above. Without it the fetcher falls back to a
Windows path, which on Linux becomes a literal directory of that name — several
GB land somewhere nothing scans, and it reports success.

**"No nodes" / pack seems missing.** Run the `/object_info` check. A Manager
install reporting success does not prove a pack loaded. If `OTR_` is zero,
ComfyUI is scanning a different directory than the one you installed into.

**`python: command not found`.** Use `python3` everywhere.

**Restarting.** `POST /api/manager/reboot` returns **502 and works anyway** —
the server drops the connection going down. Poll `/system_stats` for a 200.
A Manager reboot does not re-read the image's argument file; stop and start
the pod for argument changes.

**Encoding.** OTR probes for NVENC and falls back to CPU automatically. Most
containers expose CUDA for compute but not `libnvidia-encode.so.1`, so CPU
encoding is normal and expected on rented GPUs.

**First render is slow** — model loads dominate. Later runs reuse what's
resident.

**Long renders.** The headless runner stops *watching* after a bounded time; it
does not stop the render. Use `--timeout 0` to wait for a terminal result.

## Authors

Jeffrey A. Brick — [@jbrick2070](https://github.com/jbrick2070)

Source: <https://github.com/jbrick2070/ComfyUI-OldTimeRadio> — branch
`v2.0-alpha`. `main` is stale; do not use it.

Built on ComfyUI and ComfyUI-AnimateDiff-Evolved.

## Version History

* 0.1
    * Initial template on `runpod/comfyui:cuda13.0`. Install and node
      registration verified on an RTX 5090 pod (1036 → 1061 classes, 25
      `OTR_`). Episode rendering on rented hardware not yet verified.
