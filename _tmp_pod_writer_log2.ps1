$ErrorActionPreference = "Stop"
$key = Join-Path $env:USERPROFILE ".ssh\runpod_otr"
$sshBase = @(
  "-i", $key,
  "-p", "43125",
  "-o", "StrictHostKeyChecking=accept-new",
  "-o", "ConnectTimeout=20",
  "root@213.173.109.173"
)

$remotePath = "/tmp/_otr_writer_triage.sh"
$script = @'
#!/bin/bash
set -e
LOG=/workspace/otr-config/comfy_8188.log
echo "=== attempt warnings ==="
grep -n "my_story_act_2" "$LOG" | tail -n 40
echo "=== ValidationError / Field required ==="
grep -n "Field required\|lines.10\|input_value" "$LOG" | tail -n 40
echo "=== draft dir ==="
ls -la /workspace/runpod-slim/ComfyUI/output/otr/episodes/_shared/state/story_drafts/af7641e40b8b5b8c8804cc90d1a97ef80658f520f18c75cb48f246d622ff8c32
echo "=== pending ledger ==="
python3 - <<'PY'
from pathlib import Path
p = Path("/workspace/runpod-slim/ComfyUI/output/otr/episodes/pending_20260914_203120/audio/pending_20260914_203120_ledger.json")
print("ledger exists", p.exists(), "size", p.stat().st_size if p.exists() else 0)
draft = Path("/workspace/runpod-slim/ComfyUI/output/otr/episodes/_shared/state/story_drafts/af7641e40b8b5b8c8804cc90d1a97ef80658f520f18c75cb48f246d622ff8c32")
if draft.is_dir():
    for f in sorted(draft.rglob("*")):
        print(f.stat().st_size, f)
elif draft.exists():
    print("file", draft, draft.stat().st_size)
    print(draft.read_text(encoding="utf-8", errors="replace")[:4000])
PY
'@

$local = Join-Path $env:TEMP "_otr_writer_triage.sh"
[System.IO.File]::WriteAllText($local, $script.Replace("`r`n","`n"), (New-Object System.Text.UTF8Encoding $false))
scp -i $key -P 43125 -o StrictHostKeyChecking=accept-new $local "root@213.173.109.173:$remotePath"
ssh @sshBase "bash $remotePath"
