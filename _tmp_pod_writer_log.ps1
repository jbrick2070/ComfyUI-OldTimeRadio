$ErrorActionPreference = "Stop"
$key = Join-Path $env:USERPROFILE ".ssh\runpod_otr"
$log = "/workspace/otr-config/comfy_8188.log"
$ssh = @(
  "-i", $key,
  "-p", "43125",
  "-o", "StrictHostKeyChecking=accept-new",
  "-o", "ConnectTimeout=20",
  "root@213.173.109.173"
)
$remote = @'
set -e
echo "=== grep structured / my_story / JSON ==="
rg -n "StructuredCall|my_story|JSONDecode|ValidationError|lines\.10|no decodable|ActScript|raw head" /workspace/otr-config/comfy_8188.log | tail -n 80
echo "=== foley log tail ==="
tail -n 80 /workspace/otr-config/foley_mystory_3act.log
echo "=== comfy last 40 ==="
tail -n 40 /workspace/otr-config/comfy_8188.log
'@
ssh @ssh $remote
