$ErrorActionPreference = "Stop"
$ssh = @(
    "-i", "C:\Users\jeffr\.ssh\runpod_otr",
    "-o", "BatchMode=yes",
    "-o", "IdentitiesOnly=yes",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ConnectTimeout=25",
    "-p", "43125",
    "root@213.173.109.173"
)
$remote = @'
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
source /workspace/otr-config/otr-runtime.env
PY=/workspace/runpod-slim/ComfyUI/.venv-py313/bin/python
"$PY" -V
"$PY" -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'
if ss -lntp | grep -q ':8188'; then
  echo COMFY_ALREADY_UP
else
  export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
  cd "$OTR_COMFY_ROOT"
  nohup "$PY" main.py --listen 0.0.0.0 --port 8188 --disable-metadata \
    > /workspace/otr-config/comfy_8188.log 2>&1 &
  echo BOOT_PID=$!
fi
ok=0
for i in $(seq 1 45); do
  if curl -fsS http://127.0.0.1:8188/system_stats >/dev/null 2>&1; then
    echo SERVER_HEALTHY i=$i
    ok=1
    break
  fi
  sleep 8
done
if [ "$ok" -ne 1 ]; then
  echo SERVER_NOT_UP
  tail -n 80 /workspace/otr-config/comfy_8188.log || true
  exit 4
fi
curl -fsS http://127.0.0.1:8188/object_info/OTR_WorkflowValidator >/dev/null
echo VALIDATOR_OK
cd "$OTR_REPO_ROOT"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
nohup "$PY" scripts/otr_canonical_api_run.py \
  --workflow workflows/variants/otr_16gb_foley.json \
  --act-count 3 \
  --source-bank my_story \
  --comfyui-url http://127.0.0.1:8188 \
  --timeout 0 \
  > /workspace/otr-config/foley_mystory_3act.log 2>&1 &
echo RUNNER_PID=$!
sleep 12
echo "=== runner log ==="
tail -n 50 /workspace/otr-config/foley_mystory_3act.log || true
echo "=== comfy tail ==="
grep -E 'got prompt|house DEFAULT_IDEA|All 25 nodes|Error|Traceback' /workspace/otr-config/comfy_8188.log | tail -n 30 || tail -n 20 /workspace/otr-config/comfy_8188.log
'@
& ssh @ssh $remote
if ($LASTEXITCODE -ne 0) { throw "pod render launch failed rc=$LASTEXITCODE" }
