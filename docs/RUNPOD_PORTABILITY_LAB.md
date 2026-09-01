# RunPod portability lab: HuMo, LTX 2.5, and the H3 install boundary

This is the executable qualification recipe for a clean remote Linux machine.
It is intentionally honest about the present evidence boundary:

- HuMo and LTX 2.5 have published OTR receipts on the local RTX 5080.
- Neither engine has completed this exact high-RAM RunPod lab yet.
- A 5090, 4090, 3090, 3080 Ti, or other rental is a **lab candidate**, not a
  proven OTR row, until its own canonical episode reaches `otr/obs/`.
- HuMo 14B is automatic and reproducible. LTX 2.5 remains a gated manual
  weight step after the one-time Lightricks terms click.
- H3 has a complete explicit five-file lane, but this operator's signed policy
  keeps H3 on owned offline hardware. The same clean-machine dependency and
  verification sequence is documented for authorized third parties; it is not
  permission to put this operator's H3 weights on a pod.

The lab deliberately asks for at least 100 GiB of effective container RAM and
150 GiB free on the model volume. Those are honest rental requirements, not
claims that the engines themselves consume that much. The headroom covers the
complete episode path, two-stage decode, model caches, voices, images, and
diagnostic receipts.

## 1. Rent and verify the machine

Choose an NVIDIA pod with:

- at least 16 GB VRAM for the first HuMo/LTX trials;
- at least 100 GiB advertised system RAM (`minRAMPerGPU: 100` in the RunPod
  REST offer filter, or `minMemoryInGb: 100` in the legacy GraphQL deploy
  call);
- at least 150 GiB free persistent/container storage after the template boots.

Inside the pod, record the actual hardware and limits:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
grep '^MemTotal:' /proc/meminfo
df -B1 /workspace

if [ -f /sys/fs/cgroup/memory.max ]; then
  printf 'cgroup_v2_memory_max='; cat /sys/fs/cgroup/memory.max
elif [ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
  printf 'cgroup_v1_memory_limit='; cat /sys/fs/cgroup/memory/memory.limit_in_bytes
else
  echo 'No readable cgroup memory controller; do not qualify this pod.' >&2
  exit 1
fi
```

Use the smaller of `MemTotal` and a numeric cgroup limit. In cgroup v2,
`memory.max=max` means use `MemTotal`; it is not an automatic pass. Do not run
this lab below 107,374,182,400 effective bytes or below 161,061,273,600 free
bytes on the resolved model volume.

## 2. Resolve the real ComfyUI tree and interpreter

The provisioner will discover both, but an explicit template path is safer:

```bash
export OTR_COMFY_ROOT=/workspace/ComfyUI
export OTR_COMFYUI_MODELS_ROOT="$OTR_COMFY_ROOT/models"
export HF_HOME="$OTR_COMFYUI_MODELS_ROOT/huggingface"
mkdir -p "$HF_HOME"
```

If the template uses a different tree, change `OTR_COMFY_ROOT`; do not create a
second guessed models directory. Every Python/pip command must use the
interpreter that proves it imports `folder_paths` from that exact ComfyUI tree.
`scripts/otr_pod_provision.sh` performs that probe and refuses a mismatch.

For LTX 2.5, first accept the terms while signed in at
<https://huggingface.co/Lightricks/LTX-2.5>, then export a read token without
printing it:

```bash
read -rsp 'Hugging Face read token: ' HF_TOKEN; echo
export HF_TOKEN
printf '%s' "$HF_TOKEN" > /root/.hf_token
chmod 600 /root/.hf_token
```

HuMo 14B is public and does not require a token.

## 3. Stop only ComfyUI and prove a clean baseline

Never use `pkill python`. It can kill unrelated services and the control path.

```bash
for port in 8000 8188; do
  pid=$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
  if [ -n "$pid" ]; then
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
    case "$cmd" in
      *ComfyUI*main.py*) kill "$pid" ;;
      *) echo "Port $port belongs to a non-ComfyUI process: $cmd" >&2; exit 1 ;;
    esac
  fi
done

test -z "$(lsof -tiTCP:8000 -sTCP:LISTEN 2>/dev/null || true)"
test -z "$(lsof -tiTCP:8188 -sTCP:LISTEN 2>/dev/null || true)"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

Wait for GPU memory to return to the template's idle baseline before changing
the boot contract.

## 4. Install the pinned core, packs, and selected profile

Clone OTR under the actual ComfyUI tree, then let the audited entry point own
the ComfyUI core pin, partner packs, dependencies, model root, and profile
routing:

```bash
mkdir -p "$OTR_COMFY_ROOT/custom_nodes"
if [ ! -d "$OTR_COMFY_ROOT/custom_nodes/ComfyUI-OldTimeRadio/.git" ]; then
  git clone -b v2.0-alpha \
    https://github.com/jbrick2070/ComfyUI-OldTimeRadio.git \
    "$OTR_COMFY_ROOT/custom_nodes/ComfyUI-OldTimeRadio"
fi
cd "$OTR_COMFY_ROOT/custom_nodes/ComfyUI-OldTimeRadio"
```

**Before either successful profile provision below, complete section 5 and
export `OTR_VOICE_REFERENCE_BANK`.** The profiles use IndexTTS2 for character
voices. Provisioning is intentionally incomplete until both authorized
reference WAVs and the generated full portable bank exist. It may install the
pinned environment on an earlier pass, but it must return nonzero rather than
claiming a render-ready machine.

### HuMo 14B: one-command weight owner

For the wide 14B shipping profile:

```bash
export OTR_PROVISION_PROFILE=otr_w45_humo_14b_169
export OTR_WITH_INDEXTTS2=1
bash scripts/otr_pod_provision.sh
```

The profile router calls this exact lane:

```bash
"$OTR_COMFY_ROOT/.venv/bin/python" scripts/otr_fetch_lane_weights.py humo
```

If the template uses another proven ComfyUI interpreter, substitute that
interpreter; never use an unrelated system Python. `LANES["humo"]` is the one
source manifest for all five 14B artifacts. It pins every repository revision,
destination, byte count, and SHA-256; downloads to `.part`; verifies both size
and hash; and atomically renames only a verified file. It fetches the engine's
actual Kijai `Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors`, never the
wrong Comfy-Org `humo_17B` lookalike.

The HuMo lane is 28,707,153,033 bytes (26.7356 GiB). Its local 5080 receipt is
13.06 GiB VRAM and 27.53 GiB host RAM at 832x480x97. Use at least 32 GiB host
RAM even outside this deliberately roomier lab.

The 1.7B HuMo profiles remain a separate manual tier for now. They never fetch
the 14B DiT or its LoRA. Print their exact paths and hashes without installing:

```bash
"$OTR_COMFY_ROOT/.venv/bin/python" scripts/otr_provision.py \
  --profile otr_w45_humo_1_7b --list
```

### H3: explicit engine weights plus profile dependencies

This operator does not run H3 on RunPod. For an authorized clean machine, first
inspect the complete selected-profile plan; the shipping 45-word video profile
routes Z-Image and Stable Audio in addition to the operator-only H3 tier:

```bash
"$OTR_COMFY_ROOT/.venv/bin/python" scripts/otr_provision.py \
  --profile otr_w45_minimax_h3_video --list
```

Fetch the complete five-file H3-engine lane explicitly:

```bash
"$OTR_COMFY_ROOT/.venv/bin/python" scripts/otr_fetch_lane_weights.py minimax_h3
```

Then rerun the selected profile. The provisioner automatically fetches the
profile's exact Z-Image precision and Stable Audio lane, verifies all five H3
files against the fetcher's pinned byte/SHA manifest, and stays nonzero if any
final file or dependency is missing:

```bash
"$OTR_COMFY_ROOT/.venv/bin/python" scripts/otr_provision.py \
  --profile otr_w45_minimax_h3_video --with-indextts2
```

The H3 command is complete for H3's own weights; it is deliberately not a
profile bundle and never auto-selects itself from a public machine class.

### LTX 2.5: pinned packs plus exact manual weights

```bash
export OTR_PROVISION_PROFILE=otr_ltx25_high_video
export OTR_WITH_INDEXTTS2=1
bash scripts/otr_pod_provision.sh
```

The first run is expected to exit nonzero while the gated manual LTX tier is
absent. That is an honest incomplete receipt. Use this helper to download each
file to `.part`, verify it, and rename it atomically:

```bash
fetch_exact () {
  repo=$1; revision=$2; remote=$3; relative_dest=$4
  expected_bytes=$5; expected_sha=$6
  dest="$OTR_COMFYUI_MODELS_ROOT/$relative_dest"
  part="$dest.part"
  mkdir -p "$(dirname "$dest")"

  if [ -f "$dest" ] \
     && [ "$(stat -c '%s' "$dest")" = "$expected_bytes" ] \
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
  actual_bytes=$(stat -c '%s' "$part") || {
    rm -f "$part"
    return 1
  }
  if [ "$actual_bytes" != "$expected_bytes" ]; then
    echo "SIZE MISMATCH $relative_dest: $actual_bytes != $expected_bytes" >&2
    rm -f "$part"
    return 1
  fi
  if ! printf '%s  %s\n' "$expected_sha" "$part" | sha256sum -c -; then
    rm -f "$part"
    return 1
  fi
  mv -f "$part" "$dest" || {
    rm -f "$part"
    return 1
  }
}
```

Run the five exact LTX 2.5 rows:

```bash
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
command until it verifies every final path and returns zero. A `.part` file
never counts as installed.

If using `otr_ltx25_foley_flux2klein`, first print and install its additional
exact Flux2 Klein manual tier:

```bash
"$OTR_COMFY_ROOT/.venv/bin/python" scripts/otr_provision.py \
  --profile otr_ltx25_foley_flux2klein --list
```

## 5. IndexTTS2 reference voices

The selected HuMo/LTX profiles use IndexTTS2 for character voices. The
provisioner installs its isolated pinned environment and weights when
`OTR_WITH_INDEXTTS2=1`, but it cannot invent or redistribute a person's voice.
Provide two distinct, authorized, uncompressed PCM speech WAVs, one male and
one female, each at least one second long. Then derive a complete portable bank
from the shipped bank:

```bash
mkdir -p /workspace/otr-config

export OTR_INDEXTTS2_ROOT=/workspace/index-tts
export OTR_INDEXTTS2_VENV="$OTR_INDEXTTS2_ROOT/.venv/Scripts/python.exe"
export OTR_INDEXTTS2_DIR="$OTR_INDEXTTS2_ROOT/checkpoints"
export OTR_INDEXTTS2_WORKER="$OTR_COMFY_ROOT/custom_nodes/ComfyUI-OldTimeRadio/scripts/_otr_indextts2_worker.py"

"$OTR_COMFY_ROOT/.venv/bin/python" \
  scripts/otr_make_portable_voice_bank.py \
  --models-root "$OTR_COMFYUI_MODELS_ROOT" \
  --male-wav /absolute/path/to/authorized-male.wav \
  --female-wav /absolute/path/to/authorized-female.wav \
  --output /workspace/otr-config/voice_reference_bank.portable.json

export OTR_VOICE_REFERENCE_BANK=/workspace/otr-config/voice_reference_bank.portable.json
```

The IndexTTS source is deliberately a persistent sibling of the pinned ComfyUI
checkout. Putting it under `$OTR_COMFY_ROOT` creates an untracked nested checkout
and makes the next core-integrity gate refuse the rerun.

The utility validates complete PCM payloads, copies the references under
`models/TTS/refs/indextts2/`, records their exact SHA-256 values, preserves
every non-Index row (including Kokoro announcer choices), removes unavailable
operator-local Index rows, and adds exactly one lower-case `male` and one
lower-case `female` Index character row. An orphan WAV, an altered file, an
uppercase gender label, a missing mapping, or a one-gender bank is incomplete.

Keep all five exports (`OTR_INDEXTTS2_ROOT`, `_VENV`, `_DIR`, `_WORKER`, and
`OTR_VOICE_REFERENCE_BANK`) in the environment for **both** provisioning and
the ComfyUI server launch. The engine's qualified defaults are Windows-shaped,
so a Linux source-root override alone is insufficient. Persist these same
absolute values in the pod template or launch script; writing them only in one
interactive shell does not configure a later service process. Rerun the
selected profile provision command after the bank exists and require a zero
exit.

The generated bank deliberately does not reproduce the private,
Lemmy-specific qualified Index route. Generic male/female casting works; that
specific route remains unavailable until the operator supplies an authorized
replacement and separately re-qualifies it. `--commercial-clean` marks only
the rights status of the two supplied recordings after those rights are
verified. It does not change IndexTTS2's own non-commercial model profile.
The bank records that one absence by its exact qualified route id. It does not
disable character policy generally: a typo, a present-but-invalid row, revoked
rights, changed evidence, or any other route still fails closed.

## 6. Launch the selected boot contract

Do not reuse one server between LTX and HuMo. Stop it selectively and relaunch
for each profile. Profile launch environment maps to ComfyUI arguments as
follows:

- `OTR_HEADLESS_RESERVE_VRAM_GB=<n>` becomes `--reserve-vram <n>`;
- `OTR_HEADLESS_DISABLE_PINNED=1` becomes `--disable-pinned-memory`.

HuMo 14B wide uses the `humo_diet` contract:

```bash
export OTR_HEADLESS_RESERVE_VRAM_GB=2.921
export OTR_HEADLESS_DISABLE_PINNED=1

"$OTR_COMFY_ROOT/.venv/bin/python" "$OTR_COMFY_ROOT/main.py" \
  --listen 127.0.0.1 --port 8000 \
  --output-directory "$OTR_COMFY_ROOT/output" \
  --reserve-vram "$OTR_HEADLESS_RESERVE_VRAM_GB" \
  --disable-pinned-memory \
  > /workspace/otr-comfy-server.log 2>&1 &
```

For LTX, clear the HuMo-only values unless its selected profile explicitly
declares them. Wait for both endpoints:

```bash
export COMFYUI_URL=http://127.0.0.1:8000
until curl -fsS "$COMFYUI_URL/object_info" >/dev/null; do sleep 2; done
curl -fsS "$COMFYUI_URL/queue" >/dev/null
```

If the proven interpreter is not `$OTR_COMFY_ROOT/.venv/bin/python`, use the
exact interpreter printed by `otr_pod_provision.sh` for launch and runner too.

## 7. Run the canonical graph

LTX 2.5:

```bash
COMFYUI_URL=http://127.0.0.1:8000 \
  "$OTR_COMFY_ROOT/.venv/bin/python" scripts/otr_canonical_api_run.py \
  --profile otr_ltx25_high_video --act-count 1 \
  --source-bank original --visual-style sci_fi_radio --timeout 0
```

HuMo 14B wide, after a fresh HuMo-diet launch:

```bash
COMFYUI_URL=http://127.0.0.1:8000 \
  "$OTR_COMFY_ROOT/.venv/bin/python" scripts/otr_canonical_api_run.py \
  --profile otr_w45_humo_14b_169 --act-count 1 \
  --source-bank original --visual-style sci_fi_radio --timeout 0
```

Omit `--workflow`: the runner must load `workflows/otr_canonical.json` itself.
Do not replace these with an ad-hoc graph. Do not use `--machine 16gb_ampere`
to test LTX; that machine class correctly selects Wan.

## 8. Qualification receipt

Success requires all of the following:

- `RESULT SUCCESS` and `obs_publish OK`;
- the final file exists under the pod's actual
  `$OTR_COMFY_ROOT/output/otr/obs/`;
- the ledger's delivered engine matches the profile under test;
- no new cgroup OOM or OOM-kill event;
- server and runner logs, elapsed time, GPU peak, raw cgroup limit and peak,
  and exact artifact paths are retained;
- LTX also passes its encoder-load and two-stage audits, including the shipped
  1664x960 tiled decode and upsample/refine evidence.

A loader success followed by cgroup SIGKILL is a negative RAM-capacity receipt,
not proof that LTX cannot run on RunPod. One cold and one warm success qualify
only the exact returned hardware/software tuple. GPU product names are not
transferable receipts.

`--reserve-vram` is an offload/runtime control in some launch contracts. A run
on a larger GPU with reserved capacity is only a pressure experiment; it never
qualifies or rules out a physical card with that smaller capacity.

After the high-RAM pod work, return to the physical RTX 4060 and rerun the
existing still/AnimateDiff path as the 8 GB regression bench. HuMo and LTX 2.5
are not 8 GB targets. Isolated H3 clips are lab-proven on that physical 4060;
the full canonical H3 episode remains experimental and unqualified there.
