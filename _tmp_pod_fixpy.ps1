$ErrorActionPreference = "Stop"
$ssh = @(
    "-i", "C:\Users\jeffr\.ssh\runpod_otr",
    "-o", "BatchMode=yes",
    "-o", "IdentitiesOnly=yes",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ConnectTimeout=20",
    "-p", "43125",
    "root@213.173.109.173"
)
$remote = @'
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
VENV=/workspace/runpod-slim/ComfyUI/.venv-py313
echo "=== dangling python ==="
ls -l "$VENV/bin/python"
echo "=== uv ==="
if ! command -v uv >/dev/null 2>&1; then
  curl -fsSL https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version
echo "=== install cpython 3.13 on container disk ==="
uv python install 3.13
echo "=== after install ==="
ls -l "$HOME/.local/share/uv/python" || true
find "$HOME/.local/share/uv/python" -name python3.13 -o -name python | head
if [ ! -x "$VENV/bin/python" ]; then
  echo "symlink still dead; relink"
  REAL=$(uv python find 3.13)
  echo REAL="$REAL"
  ln -sfn "$REAL" "$VENV/bin/python"
  ln -sfn "$REAL" "$VENV/bin/python3"
fi
"$VENV/bin/python" -V
"$VENV/bin/python" -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "ok", torch.cuda.is_available())'
echo PY_OK
'@
& ssh @ssh $remote
if ($LASTEXITCODE -ne 0) { throw "python restore failed rc=$LASTEXITCODE" }
