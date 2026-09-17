$ErrorActionPreference = "Stop"
$repo = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
$key = Join-Path $env:USERPROFILE ".ssh\runpod_otr"
$info = Get-Content -Raw "$repo\_tmp_runpod_ssh.json" | ConvertFrom-Json
$ssh = @(
    "-i", $key, "-p", "$([int]$info.port)",
    "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes",
    "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=25",
    "root@$($info.host)"
)
$remote = @'
set +e
echo "=== dangling? ==="
ls -l /workspace/runpod-slim/ComfyUI/.venv-py313/bin/python
readlink -f /workspace/runpod-slim/ComfyUI/.venv-py313/bin/python
ls -l /root/.local/share/uv/python/cpython-3.13-linux-x86_64-gnu/bin/python3.13
echo "=== uv pythons ==="
ls /root/.local/share/uv/python 2>/dev/null
echo "=== venvs ==="
ls -d /workspace/runpod-slim/ComfyUI/.venv* 2>/dev/null
echo "=== torch pythons ==="
find /root /workspace /opt /usr -name python3.13 -type f 2>/dev/null | head
find /workspace -maxdepth 6 -name python -type f 2>/dev/null | head
echo "=== try venv python ==="
/workspace/runpod-slim/ComfyUI/.venv-py313/bin/python -c "import sys; print(sys.executable)"
echo rc=$?
'@
$unix = $remote.Replace("`r`n", "`n")
$tmp = "$repo\_tmp_pod_pycheck.sh.unix"
[System.IO.File]::WriteAllText($tmp, $unix, (New-Object System.Text.UTF8Encoding $false))
scp @("-i", $key, "-P", "$([int]$info.port)", "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes") $tmp "root@$($info.host):/tmp/_pycheck.sh"
ssh @ssh "bash /tmp/_pycheck.sh"
