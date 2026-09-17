$ErrorActionPreference = "Stop"
$sshBase = @(
    "-o", "BatchMode=yes",
    "-o", "IdentitiesOnly=yes",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ConnectTimeout=20",
    "-p", "43125",
    "root@213.173.109.173"
)
$keys = @(
    "C:\Users\jeffr\.ssh\runpod_otr",
    "C:\Users\jeffr\.ssh\id_ed25519"
)
$remote = @'
set -e
echo HOST=$(hostname)
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true
echo "--- df ---"
df -h / /workspace 2>/dev/null | tail -n +1
echo "--- find OTR ---"
for p in \
  /workspace/runpod-slim/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio \
  /workspace/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio \
  /workspace/runpod-slim/custom_nodes/ComfyUI-OldTimeRadio
do
  if [ -d "$p/.git" ]; then echo FOUND "$p"; OTR="$p"; fi
done
if [ -z "${OTR:-}" ]; then
  echo "OTR_NOT_FOUND"
  find /workspace -maxdepth 5 -type d -name ComfyUI-OldTimeRadio 2>/dev/null | head
  exit 3
fi
cd "$OTR"
echo OTR="$OTR"
git rev-parse --abbrev-ref HEAD
git log -1 --oneline
git remote -v | head -n 2
git fetch origin main
echo "--- incoming ---"
git log --oneline HEAD..origin/main | head -n 20
git pull --rebase origin main
echo "--- after pull ---"
git log -1 --oneline
git rev-parse HEAD
echo "--- comfy ports ---"
ss -lntp 2>/dev/null | grep -E ':8188|:8888|:22 ' || netstat -lntp 2>/dev/null | grep -E ':8188|:8888' || true
'@

$used = $null
foreach ($key in $keys) {
    Write-Output ("TRY_KEY " + [IO.Path]::GetFileName($key))
    & ssh -i $key @sshBase $remote
    if ($LASTEXITCODE -eq 0) {
        $used = $key
        break
    }
    Write-Output ("KEY_FAIL rc=" + $LASTEXITCODE + " " + [IO.Path]::GetFileName($key))
}
if (-not $used) { throw "SSH failed with both runpod_otr and id_ed25519" }
Write-Output ("SSH_OK key=" + [IO.Path]::GetFileName($used))
