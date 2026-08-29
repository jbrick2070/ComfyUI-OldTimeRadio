# OTR on RunPod -- the real bring-up log (2026-08-29, first attempt)

Written the night of the first attempt, while every mistake was fresh. This is
the record of what actually happened, what worked, what blocked, and the exact
path a new user (or the next session) should follow. Steps marked PROVEN
happened and worked; steps marked PENDING are the known remaining work.

## What this lane is for

The local 16 GB box is the production daily driver -- `otr/obs` lives there and
episodes must keep landing in it. The cloud lane exists for what the local box
cannot do: the 8-lane simultaneous video test, the unclamped high-end challenge
(BF16 weights, bigger canvas -- as a SANCTIONED EXPERIMENT, never a silent
recipe change), and raw overnight speed. Ownership of the dailies does NOT
transfer until a pod renders a canonical episode end to end and lands it in the
operator's obs indistinguishably.

## The architecture that survived the night

- **One network volume, many pods.** A RunPod network volume (~$0.07/GB/mo)
  mounts at `/workspace` on every pod in its datacenter, read-write, many pods
  at once. Models downloaded once outlive every pod. 200 GB = $14/mo and is
  the right size (models ~80 GB + stack + output room).
- **Pods are disposable; the volume is the house.** GPU class can change per
  night (5090 for battle-testing, H100 for the challenge) against the same
  drive.
- **The 8-lane test does NOT need a cluster.** RunPod "Instant Clusters" are
  8-GPU-per-server training rigs (usually unavailable, $122+/hr). The fleet is
  just N single-GPU pods sharing the volume -- or ONE H100 80GB running lanes
  sequentially, which is simpler and likely cheaper.

## PROVEN steps

1. **API key**: console.runpod.io -> Settings -> API Keys -> create with
   Read & Write. Put it in `~/.comfyui-mcp/.env` as `RUNPOD_API_KEY=...`.
   The comfyui-mcp `runpod` tool (create/start/stop/status/connect) works from
   that moment.
2. **Fund with credits** (prepaid; pods draw from balance, not the card).
   Keep Auto-Pay OFF so the balance is a hard spending ceiling. Enable MFA --
   and note the passkey/MFA console dance below, because it cost real money.
3. **Create the network volume** in a datacenter that HAS the GPUs you want
   TODAY -- availability is per-datacenter and shifts hourly. The volume locks
   its datacenter forever. (First attempt: volume in US-CO-1, which turned out
   to be a GPU desert that night; a twin in US-CA-2 saved the evening and the
   CO-1 twin was deleted.)
4. **Deploy a pod attached to the volume.** Two routes:
   - Console: https://console.runpod.io/deploy?template=bnqtkvcer3 (the
     comfyui-mcp template -- panel on port 3000, ComfyUI on 3001, Manager
     enabled, `/workspace` dirs pre-mapped). Select the network volume (locks
     region), GPU, **container disk >= 30 GB** (template default 20 bricks the
     boot), **CUDA filter 12.8+**.
   - REST API (what actually worked at 3 AM): `POST https://rest.runpod.io/v1/pods`
     with `{name, cloudType: "SECURE", computeType: "GPU", gpuCount: 1,
     gpuTypeIds: [priority list], templateId: "bnqtkvcer3",
     networkVolumeId: "<id>", containerDiskInGb: 40}`. Pass a LONG
     `gpuTypeIds` ladder -- the API takes the first type with stock, and
     "could not find any pods with required specifications" means the ladder
     ran dry in that datacenter, not that the request was malformed.
5. **First boot takes minutes** (container image pull on an uncached host);
   probe readiness with `GET https://<pod-id>-3000.proxy.runpod.net/system_stats`
   until 200. Then `runpod action:"connect"` retargets every comfyui tool at
   the pod.
6. **Stop the pod when idle** (`runpod action:"stop"`): GPU billing ends, the
   container disk AND the volume persist, restart is instant.

## The traps, in the order they bit

- **"H100" is not a GPU type id.** The API wants exact strings:
  `NVIDIA GeForce RTX 5090`, `NVIDIA H100 80GB HBM3`, `NVIDIA H100 PCIe`, ...
- **The volume's datacenter can be dry.** Check GPU stock in a DC BEFORE
  creating the volume there. Deploy attempts tell the truth faster than the
  console's availability badges.
- **Console click-storms create billable accidents.** During the MFA/passkey
  setup the volume-create form re-fired: the account ended up with FIVE
  volumes including a 3,000 GB one billing ~$210/month, names polluted by
  whatever was in the clipboard ("MFA_MAN"). Audit
  `GET /v1/networkvolumes` after any console session and delete strays
  (`DELETE /v1/networkvolumes/<id>`). The cleanup saved $252/month.
- **Idle auto-stop vs long downloads.** `runpod action:"connect"` arms an
  idle-watcher that stops the pod when ComfyUI sits quiet -- which is exactly
  what ComfyUI does during an 80 GB model pull. Unwatch (or plan around it)
  during bring-up; re-arm for render nights.
- **Pods created through the raw REST API have NO dead-man switch** (that is a
  comfyui-mcp `create` feature). A raw-API pod bills until someone stops it --
  keep it on a manual leash.
- **Registry install of OTR is hostage to version status.** The pod Manager
  resolves packs from the Comfy Registry, and a pack whose recent versions are
  Pending/Flagged does not resolve (the alpha.9-11 flag streak, caused by
  shipping `scripts/` -- 135 files of subprocess/PowerShell -- fixed in
  alpha.12). Until an Active version exists, the git-URL route into the pod's
  own Manager is the fallback; the local comfyui-mcp refuses git installs
  against a remote target by design.

## PENDING -- the remaining bring-up work, honestly stated

1. **Install the pack on the pod** at current HEAD: cleanest is the registry
   once alpha.12 goes Active; fallback is the pod Manager's own git-URL
   install (via its web UI or its HTTP API on port 3001), or SSH + git clone.
2. **Models onto `/workspace`**: HF-hosted weights pull at datacenter speed
   (~$1 of pod-time for the whole store). BUT the locally CONVERTED GGUFs
   (`gemma4-12b-with-proj-ltx-2.5-Q5_K_M.gguf`, the Gemma writer GGUF) may not
   exist on HF under those names -- verify, else re-convert on the pod or
   upload from the local box (slow home uplink; last resort).
3. **Linux differences**: the Windows launcher (.cmd) does not apply; the
   template boots ComfyUI its own way. `OTR_COMFYUI_MODELS_ROOT` must point at
   the volume. ffmpeg presence, font paths, and the prestartup banner need a
   first-boot audit.
4. **The IndexTTS2 fingerprint gate**: a fresh Linux IndexTTS2 build will fail
   the pinned Lemmy release gate. That is the gate working as designed --
   voices need the operator's ear before a cloud episode ships.
5. **Getting episodes home**: obs lives on the local box; a cloud render needs
   a transfer step back. Not designed yet.

## Cost record (first night)

- 5090 false start (no volume, image-pull limbo): ~$1
- H100 bring-up session (boot + connect + probing): ~$1-2
- Deleted volume accidents: $0 (caught same night; would have been $252/mo)
- Standing: turtle_martini 200 GB @ $14/mo; two EXITED pods (disk pennies)
