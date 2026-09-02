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
| Default AnimateDiff episode path | 8 GB NVIDIA | Proven on the physical RTX 4060 for writer/video/voice/music; configured still-image lane was not invoked |
| Klein still/image lane | 8 GB NVIDIA | Physical 4060 lab still rendered pre-fix, but took about 42 minutes and was not an OTR episode; fixed-card retest pending |
| HuMo 14B | 16 GB+ NVIDIA and 32 GB+ host RAM | Proven on the physical RTX 5080; public pinned download |
| LTX 2.5, shipped 1664x960 output | 32-48 GB NVIDIA and 100 GiB+ cgroup RAM | Use a 48 GB L40S first; 24 GB RTX 4090 exact tuple reached decode and GPU-OOMed |
| MiniMax H3 | Authorized owned/offline NVIDIA hardware | Never put operator H3 weights on RunPod |

RTX 5090, RTX 4090, RTX 3090, and RTX 3080 Ti are useful physical-card
candidates, not blanket compatibility claims. A card becomes proven only after
a canonical episode publishes with a complete receipt. Eight GB is not a
supported target for HuMo 14B, LTX 2.5, or a full H3 episode. It remains the
proven floor for the default AnimateDiff episode path. Klein is physically
lab-measured on 8 GB but remains an episode-unproven candidate: its first
pre-fix still took about 42 minutes, and the post-residency-fix card retest is
still pending.

The tested template is `runpod/comfyui:cuda13.0`. On a 570-series NVIDIA
driver, its torch 2.10 cu130 build may import but fail real CUDA work. The OTR
provisioner detects that exact tuple, installs the audited torch 2.10 cu128
trio, runs a CUDA matrix multiply, and verifies CUDA again after all pack
dependencies. Unknown incompatible tuples fail honestly.

The public machine rows select Kokoro voices and therefore require ComfyUI
Python 3.12 or earlier. The provisioner reports the actual interpreter and
fails before model-weight downloads if a selected voice cannot install. On an NVIDIA image
that ships Python 3.13, either use a Python 3.12 ComfyUI image or explicitly
select the Bark procedural floor with
`export OTR_PROVISION_PROFILE=otr_4060_floor` before provisioning.

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
writer, verifies manual tiers, and prints one receipt. Ordinary installs use
the same `--machine` keys as the generated machine matrix.

For automatic NVIDIA selection, clear both overrides:

```bash
unset OTR_PROVISION_PROFILE OTR_PROVISION_MACHINE
unset OTR_WITH_INDEXTTS2

# Optional: force a matrix row instead of VRAM auto-selection.
# export OTR_PROVISION_MACHINE=16gb
```

Use an explicit profile only for a named qualification lane. It overrides a
previously loaded machine selection:

```bash
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

Safe reruns verify completed artifacts. Automatic fetches use resumable
`.part` files; the manual `fetch_exact` helper below stages atomically through
`.part` but restarts an interrupted file.
The dropdown itself never downloads a lane. The provision command is the
automatic path; sections 3 and 4 give the complete manual work that cannot be
automated.

If no override is exported, the script fails honestly when NVIDIA VRAM cannot
be read or is below the supported 8 GB floor. It selects matrix key `8gb` from
8 GB to below 10 GB, `12gb` from 10 through 15 GB, and `16gb` at 16 GB or
more. Those are the exact rows in `config/machine_classes.json`; the old hidden
`otr_runpod_starter` default is gone. A set `OTR_PROVISION_PROFILE` remains the
explicit lab override for HuMo, LTX, or another named experiment.

Before the profile pass, provisioning writes two mode-0600 files:

```text
/workspace/otr-config/otr-runtime.env
/workspace/otr-config/otr-secrets.env
```

The first, non-secret receipt is authoritative for ComfyUI, repository, Python,
models, Hugging Face cache, IndexTTS2, voice-bank, selected machine/profile,
the normalized selector, a unique provision generation, the protected-secret
file path, and the port. It does not contain credential values, `COMFYUI_URL`,
or `OTR_INDEXTTS2_VENV`. The second file carries only the allowlisted
`HF_TOKEN`, `OTR_COMFY_API_KEY`, `OTR_GOOGLE_API_KEY`, and
`OPENROUTER_API_KEY` values that actually exist. Template values are recovered
from PID 1 because an SSH shell does not inherit them; explicit shell values
win, and safe reruns preserve an existing value until it is explicitly
replaced. Runtime callers may override output and log destinations, but not
receipt-owned install/cache paths. Never paste a credential into the
non-secret receipt.

For a running pod whose template did not contain a provider key, enter it once
without terminal echo and rerun provisioning so later sweep shells inherit it:

```bash
source /workspace/otr-config/otr-runtime.env
read -rsp 'Comfy provider key: ' OTR_SECRET_INPUT; echo
export OTR_COMFY_API_KEY="$OTR_SECRET_INPUT"
unset OTR_SECRET_INPUT
bash "$OTR_REPO_ROOT/scripts/otr_pod_provision.sh"
```

Use the corresponding allowlisted variable for Google or OpenRouter. A logged-in
ComfyUI Desktop session may instead supply hidden prompt authentication;
therefore a provider key is a headless RunPod requirement, not a universal
profile-install gate. `otr_load_runtime` sources the protected file without
printing values and preserves an explicit caller override.

An initial nonzero exit can be correct: the receipt names every missing manual
file or authorized voice reference. Complete that item and rerun the same
selection until the final line says `provision complete`.

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

Manual tiers use the provisioner's exact repo, revision, destination, byte
count, and SHA-256 manifest. Define this shared helper once in the current
shell before using any manual recipe below. It writes only `.part` until the
whole file verifies:

```bash
source /workspace/otr-config/otr-runtime.env
source "$OTR_REPO_ROOT/scripts/otr_pod_runtime.sh"
otr_load_runtime

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
```

### Flux.2 Klein: manual public files for AMD and still-consuming profiles

The AMD machine row and any still-consuming Klein profile name
`flux2_klein` as a manual tier. The 8/12 GB NVIDIA rows expose Klein as an
image selection, but their AnimateDiff video path accepts no init still, so
the planner intentionally does not gate that proven episode path on these
optional files. To qualify Klein itself, fetch these three public files
(10,985,506,708 bytes total):

```bash
fetch_exact Latentiq/FLUX.2-klein-4B-GGUF \
  4dc94114f28d56e7b63e7bb624a1c1f20353245b \
  flux-2-klein-4b-Q4_K_M.gguf \
  diffusion_models/flux-2-klein-4b-Q4_K_M.gguf \
  2604311104 0b25d143c8469b342bc5af3bce92b783bf6b0636d285f7b2f75e38af63af9a15

fetch_exact Comfy-Org/flux2-klein \
  5f526678002e43af5551dadb73ce2e8c91b43afe \
  split_files/text_encoders/qwen_3_4b.safetensors \
  text_encoders/qwen_3_4b.safetensors \
  8044982048 6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a

fetch_exact Comfy-Org/flux2-dev \
  ab9055628ea245000e610f2aa2c96f4746093546 \
  split_files/vae/flux2-vae.safetensors \
  vae/flux2-vae.safetensors \
  336213556 d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5

bash "$OTR_REPO_ROOT/scripts/otr_pod_provision.sh"
```

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

### HuMo 1.7B: manual public files

The 1.7B experimental profiles use a separate four-file manual tier totaling
13,560,364,279 bytes. It does not replace the stranger-facing 14B HuMo lane:

```bash
fetch_exact Comfy-Org/HuMo_ComfyUI \
  3a5e6947d865c3910cb2407cf2dac6a8df506b5a \
  split_files/diffusion_models/humo_1.7B_fp16.safetensors \
  diffusion_models/humo_1.7B_fp16.safetensors \
  3483511088 3f8c08e7db17e807397b9a9ed9d9b28a6e42c8083029395674e95544191b1b15

fetch_exact Comfy-Org/Wan_2.1_ComfyUI_repackaged \
  617a7633e636506f850e043bc4605f290a466a8e \
  split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
  text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
  6735906897 c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68

fetch_exact Comfy-Org/HuMo_ComfyUI \
  3a5e6947d865c3910cb2407cf2dac6a8df506b5a \
  split_files/audio_encoders/whisper_large_v3_fp16.safetensors \
  audio_encoders/whisper_large_v3_fp16.safetensors \
  3087130976 a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95

fetch_exact Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
  c4f60d30c55a624e35427060fdd217579a6c1d77 \
  split_files/vae/wan_2.1_vae.safetensors \
  vae/wan_2.1_vae.safetensors \
  253815318 2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b

bash "$OTR_REPO_ROOT/scripts/otr_pod_provision.sh"
```

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

Load the token, then use the shared `fetch_exact` helper from the start of this
section for the exact manifest. A final file is published only after byte-count
and SHA-256 verification:

```bash
[ -s /root/.hf_token ] && export HF_TOKEN="$(tr -d ' \t\r\n' < /root/.hf_token)"

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
"$COMFY_PY" scripts/otr_provision.py --profile otr_w45_minimax_h3_video --list
"$COMFY_PY" scripts/otr_fetch_lane_weights.py minimax_h3
```

The five-file lane totals 63,440,965,087 bytes and is never automatically
selected by a public machine class or pod roster. The public
`mkhamra/quibble-h3` repository is a Ref2VA workflow/case study, not an OTR
node-pack download source. The physical RTX 4060 has isolated 90-frame H3 clip
receipts, below OTR's 124-model-frame floor; it does not yet have a full
canonical H3 episode receipt.

## 5. Launch and qualify one matrix row or profile

Every pod launch uses port 8188 and the runtime receipt. At boot, the shared
helper stops only exact listeners on the template/selected port, applies the
selected recipe's boot contract, carries the token into the new ComfyUI
process without printing it, launches on `0.0.0.0` for the RunPod proxy, and
verifies nonzero OTR classes plus an idle queue.

```bash
source /workspace/otr-config/otr-runtime.env
SAFE_SELECTOR="${OTR_PROVISION_SELECTOR//:/-}"
QUAL_DIR="/workspace/otr-config/qualification/${SAFE_SELECTOR}-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$QUAL_DIR"
export OTR_SERVER_LOG="$QUAL_DIR/server.log"

source "$OTR_REPO_ROOT/scripts/otr_pod_runtime.sh"
otr_load_runtime || exit $?
otr_acquire_campaign_lock "manual qualification" || exit $?
trap 'otr_release_campaign_lock' EXIT
otr_boot_profile "$OTR_PROVISION_SELECTOR" || exit $?

RUN_SELECTION=()
case "$OTR_PROVISION_SELECTOR" in
  machine:*) RUN_SELECTION=(--machine "${OTR_PROVISION_SELECTOR#machine:}") ;;
  *) RUN_SELECTION=(--profile "$OTR_PROVISION_SELECTOR") ;;
esac

cd "$OTR_REPO_ROOT"
"$COMFY_PY" scripts/otr_canonical_api_run.py \
  "${RUN_SELECTION[@]}" --act-count 1 \
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
otr_release_campaign_lock
trap - EXIT
test "$RUNNER_RC" -eq 0
```

Omit `--workflow`: the runner itself must load
`workflows/otr_canonical.json`. A finished render leaves ComfyUI resident and
holding VRAM. Completion is proved by the logs and artifact, not by low idle
VRAM. Manual qualification and the unattended sweep/soak are mutually
exclusive and share the same campaign lock.

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
Both scripts source the same runtime helper as the manual qualification. Run
exactly one sweep or soak at a time: a persistent nonblocking campaign lock
refuses a second driver before it can interrupt a paid render. All evidence
logs default under the persistent `/workspace/otr-config/logs` directory.

Load the receipt once in the launching shell:

```bash
source /workspace/otr-config/otr-runtime.env
source "$OTR_REPO_ROOT/scripts/otr_pod_runtime.sh"
otr_load_runtime
```

One-act every RunPod-provisionable `otr_w45_*` profile, then three acts for
passers:

```bash
setsid nohup bash "$OTR_REPO_ROOT/scripts/otr_pod_overnight_sweep.sh" \
  > "$OTR_POD_LOG_DIR/overnight-driver.log" 2>&1 < /dev/null &
```

Continuous one-act soak:

```bash
setsid nohup bash "$OTR_REPO_ROOT/scripts/otr_pod_lane_soak.sh" \
  > "$OTR_POD_LOG_DIR/soak-driver.log" 2>&1 < /dev/null &
```

On a Python 3.13 NVIDIA template, run the supported Bark procedural fallback
unattended with an explicit one-profile roster:

```bash
export OTR_POD_PROFILES='otr_4060_floor'
setsid nohup bash "$OTR_REPO_ROOT/scripts/otr_pod_overnight_sweep.sh" \
  > "$OTR_POD_LOG_DIR/overnight-driver.log" 2>&1 < /dev/null &

# After that campaign finishes, restore ordinary otr_w45_* discovery.
unset OTR_POD_PROFILES
```

To qualify a smaller explicit roster:

```bash
export OTR_POD_PROFILES='otr_w45_still_flat otr_w45_wan_ti2v otr_w45_ltx25_video'
setsid nohup bash "$OTR_REPO_ROOT/scripts/otr_pod_overnight_sweep.sh" \
  > "$OTR_POD_LOG_DIR/overnight-driver.log" 2>&1 < /dev/null &
```

The helper groups identical launch fingerprints, restarts when the full boot
contract changes, and uses the current profile for recovery. A missing model is
recorded as a lane result and does not abort the campaign. H3 is excluded by
the `h3` boot-contract family (including `h3_8gb_lab`); explicitly placing H3
in a cloud roster is an error.
Any roster that uses IndexTTS2 must have the portable bank first.

Default discovery also asks the provisioner for a complete install plan and
excludes four legacy lab profiles that still lack an exact public weight owner:
`otr_w45_fastwan`, `otr_w45_ltx_audio_in`, `otr_w45_ltx_video`, and
`otr_w45_mesh_stage`. Naming one explicitly is an error rather than a doomed
paid leg. Add it back only when its source revision, destinations, byte counts,
and SHA-256 values have one executable or fully documented owner.

Read `$OTR_SWEEP_RESULTS`, `$OTR_SOAK_RESULTS`, the per-leg and driver logs,
and `$OTR_SERVER_LOG` under `$OTR_POD_LOG_DIR`. Soak keeps only the latest
three logs per profile. These survive a pod stop because they are on the
network volume, unlike `/root` container storage.

To stop the active OTR campaign safely, load the runtime as above and call:

```bash
otr_stop_campaign
```

That validates the recorded PID and process group before signaling the driver,
its current runner, and its managed ComfyUI server. It stops the OTR work, not
RunPod billing; after the logs settle, stop the pod in the RunPod console.

## 7. Failure atlas

| Symptom | Cause | Fix and proof |
|---|---|---|
| Provision refuses Kokoro before downloading | Selected machine row is running under Python 3.13 | Use a Python 3.12 ComfyUI image; on NVIDIA, explicitly select `otr_4060_floor` for the Bark procedural route |
| `torch` imports, but CUDA says driver capability 12.8 for a cu130 wheel | CUDA 13 template on driver 570-579 | Rerun the owner; require its real CUDA matmul and final `torch verified` line |
| Provision says a manual tier is incomplete | Exact gated/private files are absent or wrong | Follow section 4; `.part` never counts; rerun until every size/SHA verifies |
| LTX reports `WrapperNodeMissing` with weights present | Required pinned/patch-owned pack is absent or drifted | Rerun provisioning; do not install an arbitrary latest pack or downgrade Kornia |
| HuMo downloads ~16 GB and still says not installed | Wrong `humo_17B` lookalike | Fetch the pinned `humo` lane; verify the Kijai `Wan2_1-HuMo-14B...KJ` file |
| H3 has weights but no usable nodes | Authorization/source boundary or wrong graph provider | Keep OTR H3 local; `quibble-h3` is not the node-pack owner |
| First render pauses on an LLM timeout | Writer weights were left for first Queue | Rerun current provisioning; writer warm must be `OK` in its receipt |
| Token works in a shell but gated render fails | Resident ComfyUI never inherited the token | Store `/root/.hf_token`, run `otr_load_runtime`, then force the same recipe through `otr_boot_profile "$OTR_PROVISION_SELECTOR"` |
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
| Local RTX 4060 8 GB | default AnimateDiff episode path | PROVEN published episodes for writer/video/voice/music; configured still-image lane was uninvoked |
| Local RTX 4060 8 GB | MiniMax H3 raw FL2VA recipe | LAB-PROVEN three isolated 90-frame clips below OTR's 124-model-frame floor; not an OTR adapter episode |

The 16 GB 5080 success and 24 GB 4090 failure are not explained by capacity
alone; GPU architecture, driver, allocator state, and residency behavior differ.
The L40S leg must therefore retain both `text encoder pinned to CPU` and
`render-window VRAM peak` lines. Its extra headroom is a qualification choice,
not a claimed diagnosis.

Do not promote a candidate from a model-loader probe, clamp simulation, Reddit
report, or partial clip. A published canonical episode on the named physical
hardware is the proof boundary.
