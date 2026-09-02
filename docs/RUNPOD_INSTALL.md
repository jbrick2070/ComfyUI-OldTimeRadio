# RunPod playbook for ComfyUI-OldTimeRadio

This is the only RunPod playbook for OTR. It owns pod selection, installation,
manual downloads, voice references, launch, qualification, unattended runs,
telemetry, and recovery. Other `RUNPOD_*.md` files are redirects to this file.

The objective is the real canonical workflow on real hardware. Do not replace
`workflows/otr_canonical.json`, reduce its shipped canvas, add a VRAM gate, or
use an artificial VRAM clamp as a hardware receipt. Rent enough RAM and disk,
run the graph unchanged, and report the exact tuple.

## 1. Choose the pod and lane

For the complete heavy-engine lab, use:

- one NVIDIA GPU appropriate to the lane;
- at least **100 GiB effective cgroup RAM**;
- at least **150 GiB free on the filesystem that owns the models**;
- a 200 GB or larger container disk, or a network volume mounted under
  `/workspace` with the model tree on that volume;
- ports 8188 (ComfyUI), 8888 (JupyterLab), and SSH exposed as needed.

Those RAM and disk figures cover the whole OTR stack, writer cache, isolated
voice runtime, heavy video weights, page cache, and output room. They are not
claims that one engine intrinsically consumes 100 GiB of RAM.

| Goal | Starting hardware | Status |
|---|---|---|
| Default episode or still lane | 8 GB NVIDIA | Proven on the physical RTX 4060; easiest first run |
| HuMo 14B | 16 GB+ NVIDIA and 32 GB+ host RAM | Proven on the physical RTX 5080; public pinned download |
| LTX 2.5, shipped 1664x960 output | 32-48 GB NVIDIA and 100 GiB+ cgroup RAM | Use a 48 GB L40S first; 24 GB RTX 4090 exact tuple reached decode and GPU-OOMed |
| MiniMax H3 | Authorized owned/offline NVIDIA hardware | Never put operator H3 weights on RunPod |

RTX 5090, RTX 4090, RTX 3090, and RTX 3080 Ti are useful physical-card
candidates, not blanket compatibility claims. A card becomes proven only after
a canonical episode publishes with a complete receipt. Eight GB is not a
supported target for HuMo 14B, LTX 2.5, or a full H3 episode. It remains a good
target for the default and still workflows.

The tested template is `runpod/comfyui:cuda13.0`. On a 570-series NVIDIA
driver, its torch 2.10 cu130 build may import but fail real CUDA work. The OTR
provisioner detects that exact tuple, installs the audited torch 2.10 cu128
trio, runs a CUDA matrix multiply, and verifies CUDA again after all pack
dependencies. Unknown incompatible tuples fail honestly.

Before spending download time, verify the real limits:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [ -f /sys/fs/cgroup/memory.max ]; then
  echo "cgroup v2 max=$(cat /sys/fs/cgroup/memory.max)"
else
  echo "cgroup v1 max=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)"
fi

df -h /workspace /
```

RunPod's UI RAM label is not the receipt; the cgroup file is. Disk and RAM are
different limits. A large network volume does not raise a 62 GB RAM cap.

## 2. One provisioning loop

ComfyUI-Manager cannot install this alpha pack reliably. Provision from the
`v2.0-alpha` branch. The bootstrap below locates the template's real ComfyUI
tree, pins ComfyUI core and partner packs, repairs the measured CUDA mismatch,
clones or fast-forwards OTR, downloads automatic lanes, warms the selected
writer, verifies manual tiers, and prints one receipt.

Choose one profile first:

```bash
# Easy public first episode/stills, no cloned-voice setup:
export OTR_PROVISION_PROFILE=otr_nvidia_8gb_haunted
unset OTR_WITH_INDEXTTS2

# Or HuMo 14B:
# export OTR_PROVISION_PROFILE=otr_w45_humo_14b_169
# export OTR_WITH_INDEXTTS2=1

# Or LTX 2.5 high video:
# export OTR_PROVISION_PROFILE=otr_ltx25_high_video
# export OTR_WITH_INDEXTTS2=1
```

Run the owner:

```bash
curl -fL --retry 4 \
  https://raw.githubusercontent.com/jbrick2070/ComfyUI-OldTimeRadio/v2.0-alpha/scripts/otr_pod_provision.sh \
  -o /tmp/otr_pod_provision.sh
bash /tmp/otr_pod_provision.sh
```

Safe reruns verify completed artifacts and resume only through `.part` files.
The dropdown itself never downloads a lane. The provision command is the
automatic path; sections 3 and 4 give the complete manual work that cannot be
automated.

If no profile is exported, the script selects `otr_runpod_starter` at 16 GB+
and `otr_nvidia_8gb_haunted` below 16 GB. The starter uses IndexTTS2 and
therefore needs section 3; the 8 GB haunted profile uses the public voice path.

Before the profile pass, provisioning writes:

```text
/workspace/otr-config/otr-runtime.env
```

That mode-0600 receipt is the one owner for ComfyUI, repository, Python,
models, Hugging Face cache, IndexTTS2, voice-bank, profile, and port paths. It
does not contain `HF_TOKEN`, `COMFYUI_URL`, or `OTR_INDEXTTS2_VENV`. Do not
replace it with a hand-written environment file.

An initial nonzero exit can be correct: the receipt names every missing manual
file or authorized voice reference. Complete that item and rerun the same
profile until the final line says `provision complete`.

## 3. Authorized IndexTTS2 references

Skip this section when the selected profile does not use IndexTTS2. OTR cannot
ship another person's voice recordings. Supply two distinct authorized PCM WAV
files, one male and one female, each at least one second. Upload them through
JupyterLab to a persistent path such as:

```text
/workspace/otr-upload/authorized-male.wav
/workspace/otr-upload/authorized-female.wav
```

After the first provision pass created the repository and runtime receipt:

```bash
source /workspace/otr-config/otr-runtime.env

"$COMFY_PY" "$OTR_REPO_ROOT/scripts/otr_make_portable_voice_bank.py" \
  --models-root "$OTR_COMFYUI_MODELS_ROOT" \
  --male-wav /workspace/otr-upload/authorized-male.wav \
  --female-wav /workspace/otr-upload/authorized-female.wav \
  --output "$OTR_VOICE_REFERENCE_BANK"

export OTR_WITH_INDEXTTS2=1
bash "$OTR_REPO_ROOT/scripts/otr_pod_provision.sh"
```

The generator preserves every non-Index voice row and replaces the private
Index rows with the two authorized references. `--commercial-clean` may be
used only when the operator has actually verified those recording rights; it
does not change IndexTTS2's own model license.

The provisioner pins IndexTTS2 source, Python 3.10, its lockfile, model
artifacts, four runtime repositories, the two reference hashes, and a real
offline worker-ready handshake. It stores the managed interpreter beside the
persistent engine root. If a legacy migrated volume has a dangling venv
interpreter, install the managed interpreter at the persistent path and rerun
provisioning; its `uv sync --frozen` relinks the venv while reusing cached
wheels:

```bash
export UV_PYTHON_INSTALL_DIR="$OTR_INDEXTTS2_ROOT/.uv-python"
uv python install 3.10        # IndexTTS2
bash "$OTR_REPO_ROOT/scripts/otr_pod_provision.sh"
```

Do not export the Linux offline wrapper as `OTR_INDEXTTS2_VENV` during
provisioning. The runtime adapter finds that wrapper through the audited
ComfyUI `index-tts` link; the online downloader must use the real vendor venv.
Either setup order is safe: running the Index profile before creating the bank
installs the pinned runtime and ends `MISSING` only on the two references; that
work is retained for the rerun.

## 4. Engine-specific weights

### HuMo 14B: automatic public lane

```bash
export OTR_PROVISION_PROFILE=otr_w45_humo_14b_169
export OTR_WITH_INDEXTTS2=1
bash "$OTR_REPO_ROOT/scripts/otr_pod_provision.sh"
```

The `humo` lane pins revisions, destinations, byte counts, and SHA-256 for all
five files. Its engine file is
`Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors`, not the unrelated
Comfy-Org `humo_17B` lookalike. To inspect or fetch it directly:

```bash
"$COMFY_PY" "$OTR_REPO_ROOT/scripts/otr_fetch_lane_weights.py" --list
"$COMFY_PY" "$OTR_REPO_ROOT/scripts/otr_fetch_lane_weights.py" humo
```

The physical RTX 5080 canonical receipt measured 13.06 GiB VRAM and 27.53 GiB
host RAM at 832x480x97. The profile's `humo_diet` boot contract is part of that
real recipe, not an artificial smaller-card simulation.

### LTX 2.5: one terms click, five exact files

While signed in, accept the terms at
<https://huggingface.co/Lightricks/LTX-2.5>. Put the token in the RunPod
template's `HF_TOKEN` secret before boot. For a running pod, a no-echo fallback
is:

```bash
read -rsp 'Hugging Face token: ' OTR_HF_INPUT; echo
printf '%s' "$OTR_HF_INPUT" | tr -d ' \t\r\n' > /root/.hf_token
chmod 600 /root/.hf_token
unset OTR_HF_INPUT
```

The first provision pass installs and patches the required packs, then names
the absent manual tier:

```bash
export OTR_PROVISION_PROFILE=otr_ltx25_high_video
export OTR_WITH_INDEXTTS2=1
bash "$OTR_REPO_ROOT/scripts/otr_pod_provision.sh"
"$COMFY_PY" "$OTR_REPO_ROOT/scripts/otr_provision.py" \
  --profile otr_ltx25_high_video --list
```

Load the generated layout and the token, then fetch the exact manifest. A
final file is published only after byte-count and SHA-256 verification:

```bash
source /workspace/otr-config/otr-runtime.env
[ -s /root/.hf_token ] && export HF_TOKEN="$(tr -d ' \t\r\n' < /root/.hf_token)"

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

bash "$OTR_REPO_ROOT/scripts/otr_pod_provision.sh"
```

The five files total 23,873,413,310 bytes. The provisioner pins and patches
ComfyUI-GGUF for Gemma 4 and pins and patches ComfyUI-LTXVideo for Kornia 0.8.3.
Do not downgrade Kornia, hand-edit those packs, add a VRAM clamp, or shrink the
canonical 1664x960 output to manufacture a pass.

### H3: local-only authorization boundary

This operator's H3 weights never go to RunPod. The boundary is recorded in
`docs/H3_LICENSE_ATTESTATION.md`. On authorized owned/offline hardware only:

```bash
python scripts/otr_provision.py --profile otr_w45_minimax_h3_video --list
python scripts/otr_fetch_lane_weights.py minimax_h3
```

The five-file lane totals 63,440,965,087 bytes and is never automatically
selected by a public machine class or pod roster. The public
`mkhamra/quibble-h3` repository is a Ref2VA workflow/case study, not an OTR
node-pack download source. The physical RTX 4060 has isolated 124-frame H3 clip
receipts; it does not yet have a full canonical H3 episode receipt.

## 5. Launch and qualify one profile

Every pod launch uses port 8188 and the runtime receipt. At boot, the shared
helper stops only exact listeners on the template/selected port, applies the
selected profile's boot contract, carries the token into the new ComfyUI
process without printing it, launches on `0.0.0.0` for the RunPod proxy, and
verifies nonzero OTR classes plus an idle queue.

```bash
source /workspace/otr-config/otr-runtime.env
QUAL_DIR="/workspace/otr-config/qualification/${OTR_PROVISION_PROFILE}-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$QUAL_DIR"
export OTR_SERVER_LOG="$QUAL_DIR/server.log"

source "$OTR_REPO_ROOT/scripts/otr_pod_runtime.sh"
otr_load_runtime
otr_boot_profile "$OTR_PROVISION_PROFILE"

cd "$OTR_REPO_ROOT"
"$COMFY_PY" scripts/otr_canonical_api_run.py \
  --profile "$OTR_PROVISION_PROFILE" --act-count 1 \
  --source-bank original --visual-style sci_fi_radio --timeout 0 \
  > "$QUAL_DIR/runner.log" 2>&1 &
RUNNER_PID=$!

while kill -0 "$RUNNER_PID" 2>/dev/null; do
  printf '%s,' "$(date -u +%FT%TZ)" >> "$QUAL_DIR/gpu.csv"
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits >> "$QUAL_DIR/gpu.csv"
  sleep 1
done
wait "$RUNNER_PID"; RUNNER_RC=$?
tail -n 80 "$QUAL_DIR/runner.log"
test "$RUNNER_RC" -eq 0
```

Omit `--workflow`: the runner itself must load
`workflows/otr_canonical.json`. A finished render leaves ComfyUI resident and
holding VRAM. Completion is proved by the logs and artifact, not by low idle
VRAM.

A qualification pass requires all of these:

- `$QUAL_DIR/runner.log` says `[canonical-api] RESULT SUCCESS`;
- `$QUAL_DIR/server.log` says `obs_publish OK ->`;
- a new final MP4 exists under `$OTR_OUTPUT_ROOT/otr/obs/`;
- the newest episode's `audio/*_ledger.json` contains per-clip
  `delivered_engine` values matching the selected profile rather than a silent
  fallback;
- no new cgroup OOM event occurred;
- `$QUAL_DIR` retains identity, server, runner, cgroup-before/after, and
  `gpu.csv` receipts plus the final artifact path;
- for LTX 2.5, the count of
  `TWO-STAGE PASS nodes=3 decode=1664x960` in the clean server log equals the
  count of its `delivered_engine` clips in the newest ledger.

For `otr_ltx25_high_video`, the exact cross-check is:

```bash
LEDGER=$(find "$OTR_OUTPUT_ROOT/otr/episodes" -type f -name '*_ledger.json' \
  -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
TWO_STAGE_COUNT=$(grep -c 'TWO-STAGE PASS nodes=3 decode=1664x960' "$OTR_SERVER_LOG")
DELIVERED_COUNT=$(grep -c '"delivered_engine": "ltx25_video"' "$LEDGER")
test "$TWO_STAGE_COUNT" -gt 0
test "$TWO_STAGE_COUNT" -eq "$DELIVERED_COUNT"
grep -q 'obs_publish OK ->' "$OTR_SERVER_LOG"
grep -q '\[canonical-api\] RESULT SUCCESS' "$QUAL_DIR/runner.log"
```

Useful identity commands:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$COMFY_PY" - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
PY
git -C "$OTR_COMFY_ROOT" rev-parse HEAD
git -C "$OTR_REPO_ROOT" rev-parse HEAD
find "$OTR_OBS_DIR" -maxdepth 1 -type f -name '*.mp4' -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' | sort
```

Write that output to `$QUAL_DIR/identity.txt`. Copy the applicable cgroup block
below to `$QUAL_DIR/cgroup-before.txt` before Queue and
`$QUAL_DIR/cgroup-after.txt` afterward. GPU peak comes from the maximum first
column in `$QUAL_DIR/gpu.csv`; the engine's own `render-window VRAM peak` line
is a second source for LTX 2.5.

For cgroup v2, capture the values before and after the run:

```bash
cat /sys/fs/cgroup/memory.max
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.peak
cat /sys/fs/cgroup/memory.events
```

For cgroup v1:

```bash
cat /sys/fs/cgroup/memory/memory.limit_in_bytes
cat /sys/fs/cgroup/memory/memory.usage_in_bytes
cat /sys/fs/cgroup/memory/memory.max_usage_in_bytes
cat /sys/fs/cgroup/memory/memory.failcnt
cat /sys/fs/cgroup/memory/memory.oom_control
```

High page-cache usage after 20-60 GB of downloads is not by itself an engine
working-set measurement. Judge it with process survival, cgroup event deltas,
queue/history, server log, and the final artifact.

## 6. Unattended sweep and soak

Run these from the repository on the pod. Do not copy a second script into
`/root`, and do not drive a multi-hour run through a workstation SSH session.
Both scripts source the same runtime helper as the manual qualification.

One-act every public `otr_w45_*` profile, then three acts for passers:

```bash
export OTR_SWEEP_RESULTS=/workspace/otr-config/overnight_results.txt
setsid nohup bash "$OTR_REPO_ROOT/scripts/otr_pod_overnight_sweep.sh" \
  > /root/overnight-driver.log 2>&1 < /dev/null &
```

Continuous one-act soak:

```bash
export OTR_SOAK_RESULTS=/workspace/otr-config/soak_results.txt
setsid nohup bash "$OTR_REPO_ROOT/scripts/otr_pod_lane_soak.sh" \
  > /root/soak-driver.log 2>&1 < /dev/null &
```

To qualify a smaller explicit roster:

```bash
export OTR_POD_PROFILES='otr_w45_still_flat otr_w45_wan_ti2v otr_w45_ltx25_video'
setsid nohup bash "$OTR_REPO_ROOT/scripts/otr_pod_overnight_sweep.sh" \
  > /root/overnight-driver.log 2>&1 < /dev/null &
```

The helper groups identical launch fingerprints, restarts when the full boot
contract changes, and uses the current profile for recovery. A missing model is
recorded as a lane result and does not abort the campaign. H3 is excluded by
its `h3` boot contract; explicitly placing H3 in a cloud roster is an error.
Any roster that uses IndexTTS2 must have the portable bank first.

Read `/root/overnight_results.txt`, `/root/soak_results.txt`, the per-leg logs,
and `$OTR_SERVER_LOG`. Soak keeps only the latest three logs per profile.

## 7. Failure atlas

| Symptom | Cause | Fix and proof |
|---|---|---|
| `torch` imports, but CUDA says driver capability 12.8 for a cu130 wheel | CUDA 13 template on driver 570-579 | Rerun the owner; require its real CUDA matmul and final `torch verified` line |
| Provision says a manual tier is incomplete | Exact gated/private files are absent or wrong | Follow section 4; `.part` never counts; rerun until every size/SHA verifies |
| LTX reports `WrapperNodeMissing` with weights present | Required pinned/patch-owned pack is absent or drifted | Rerun provisioning; do not install an arbitrary latest pack or downgrade Kornia |
| HuMo downloads ~16 GB and still says not installed | Wrong `humo_17B` lookalike | Fetch the pinned `humo` lane; verify the Kijai `Wan2_1-HuMo-14B...KJ` file |
| H3 has weights but no usable nodes | Authorization/source boundary or wrong graph provider | Keep OTR H3 local; `quibble-h3` is not the node-pack owner |
| First render pauses on an LLM timeout | Writer weights were left for first Queue | Rerun current provisioning; writer warm must be `OK` in its receipt |
| Token works in a shell but gated render fails | Resident ComfyUI never inherited the token | Store `/root/.hf_token`, run `otr_load_runtime`, then force the same profile through `otr_boot_profile "$OTR_PROVISION_PROFILE"` |
| IndexTTS2 survives a migration but its Python link is broken | Managed interpreter lived on erased container disk | Set persistent `UV_PYTHON_INSTALL_DIR`; install Python 3.10, then rerun provisioning |
| Port 8188 is ready but a new launch cannot bind | Template server is still resident | Use the shared runtime helper; never `pkill python` or kill every Python process |
| `/queue` is empty but nothing runs | Empty queue was mistaken for readiness | Require `/object_info` with nonzero `OTR_` classes and an idle queue |
| Manager reports success but OTR contributes zero nodes | Installed into a different ComfyUI tree or Manager has no installable alpha | Use the provisioner and its resolved `OTR_COMFY_ROOT`; verify `/object_info` |
| Process is killed while `free` shows RAM | Pod cgroup cap, not host-wide memory | Read `memory.max`/`memory.events` or v1 fail counters; rent more cgroup RAM |
| `CUDA out of memory` while cgroup OOM stays zero | GPU VRAM exhaustion | Record the exact tuple; use a larger physical GPU, not a reduced canonical graph |
| `RESULT SUCCESS` but no deliverable is found | Wrong output root or incomplete publish | Require `obs_publish OK` and the final MP4 under `$OTR_OBS_DIR` |
| Low GPU utilization while hundreds of weight shards scroll | Persistent network volume is loading a large model | Let it finish; use logs/heartbeat, not utilization alone, to decide it is stuck |
| Disk fills despite a large volume | Models or HF cache landed on container disk or in two cache roots | Use the generated receipt; keep `HF_HOME=$OTR_COMFYUI_MODELS_ROOT/huggingface` |
| Provisioner says ComfyUI is not a git checkout or has tracked changes | The template core cannot be safely pinned in place | Follow its printed side-by-side checkout recipe, or restore the exact tracked core deliberately; never overwrite unknown work |

## 8. Evidence ledger

These labels are deliberately narrow:

| Hardware | Lane | Evidence |
|---|---|---|
| Local RTX 5080 16 GB | HuMo 14B | PROVEN canonical published episode; 13.06 GiB VRAM / 27.53 GiB host RAM receipt |
| Local RTX 5080 16 GB | LTX 2.5 | PROVEN canonical published episode |
| RunPod RTX 4090 24 GB, 116.42 GiB cgroup RAM | LTX 2.5 | NEGATIVE exact tuple: reached shipped 1664x960 two-stage decode, then GPU OOM; zero cgroup OOM |
| RunPod L40S 48 GB, 188 GB cgroup RAM | LTX 2.5 | Clean provisioning/CUDA/weights/Index receipt complete; canonical render qualification pending |
| RunPod RTX PRO 4000 Blackwell 24 GB | default starter | PROVEN published one-act episode |
| Local RTX 4060 8 GB | default and still workflows | PROVEN published episodes; H3 only has isolated clip receipts |

The 16 GB 5080 success and 24 GB 4090 failure are not explained by capacity
alone; GPU architecture, driver, allocator state, and residency behavior differ.
The L40S leg must therefore retain both `text encoder pinned to CPU` and
`render-window VRAM peak` lines. Its extra headroom is a qualification choice,
not a claimed diagnosis.

Do not promote a candidate from a model-loader probe, clamp simulation, Reddit
report, or partial clip. A published canonical episode on the named physical
hardware is the proof boundary.
