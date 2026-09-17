$ErrorActionPreference = "Stop"
$repo = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
$key = Join-Path $env:USERPROFILE ".ssh\runpod_otr"
$info = Get-Content -Raw "$repo\_tmp_runpod_ssh.json" | ConvertFrom-Json
$ssh = @(
    "-i", $key,
    "-p", "$([int]$info.port)",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "BatchMode=yes",
    "-o", "IdentitiesOnly=yes",
    "-o", "ConnectTimeout=25",
    "root@$($info.host)"
)
ssh @ssh @'
echo "=== env in comfy pid ==="
pid=$(ss -lntp | awk "/:8188/{print}" | grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2)
echo comfy_pid=$pid
if [ -n "$pid" ]; then
  tr "\0" "\n" < /proc/$pid/environ 2>/dev/null | awk -F= "/^OPENROUTER_API_KEY=/{print \"COMFy_OR_CHARS=\" length(\$2); next} /^OPENROUTER/{print \$1 \"_set=1 chars=\" length(\$2)}"
fi
echo "=== queue ==="
curl -fsS http://127.0.0.1:8188/queue | python3 -c "import sys,json; d=json.load(sys.stdin); print(\"running\",len(d.get(\"queue_running\") or []),\"pending\",len(d.get(\"queue_pending\") or []))"
echo "=== object_info slot ==="
curl -fsS http://127.0.0.1:8188/object_info/OTR_LedgerScriptWriter | python3 - <<'PY'
import json,sys
info=json.load(sys.stdin)["OTR_LedgerScriptWriter"]["input"]
block=info.get("required") or {}
if "openrouter_slot_a_model" not in block:
    block=info.get("optional") or {}
slot=(block.get("openrouter_slot_a_model") or [[]])[0]
print("SLOT_A_N", len(slot) if isinstance(slot,list) else 0)
print("HAS_GPT_LATEST", "~openai/gpt-latest" in (slot or []))
print("HAS_ENABLE", "(enable OpenRouter)" in (slot or []))
print("HEAD", list(slot)[:8] if isinstance(slot,list) else slot)
PY
echo "=== log hits ==="
grep -E "OpenRouter|openrouter|OPENROUTER|401|402|403|credit" /workspace/otr-config/comfy_8188.log | tail -n 40
echo "=== writer last ==="
grep -E "LedgerScriptWriter|StructuredCall|heartbeat|ERROR|Traceback|openrouter" /workspace/otr-config/comfy_8188.log | tail -n 30
'@
