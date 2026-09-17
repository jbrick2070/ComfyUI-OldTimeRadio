$ErrorActionPreference = "Continue"
$key = Join-Path $env:USERPROFILE ".ssh\runpod_otr"
$ssh = @(
    "-i",$key,
    "-o","BatchMode=yes",
    "-o","IdentitiesOnly=yes",
    "-o","StrictHostKeyChecking=accept-new",
    "-o","ConnectTimeout=20",
    "-p","43125",
    "root@213.173.109.173"
)
$remote = @'
echo "=== queue ==="
curl -fsS http://127.0.0.1:8188/queue || echo QUEUE_DOWN
echo
echo "=== runner tail ==="
tail -n 80 /workspace/otr-config/foley_mystory_1act.log || true
echo "=== comfy errors ==="
grep -E 'ERROR|Traceback|StructuredCallFailed|obs_publish|Prompt executed|house DEFAULT_IDEA|RESULT' /workspace/otr-config/comfy_8188.log | tail -n 40 || true
echo "=== newest obs ==="
ls -1t /workspace/runpod-slim/ComfyUI/output/otr/obs/*_final.mp4 2>/dev/null | head -n 5 || true
'@
& ssh @ssh $remote
