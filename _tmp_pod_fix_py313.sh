#!/bin/bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo UV=$(command -v uv)
uv python install 3.13
echo "=== uv python dirs ==="
ls -d "$HOME/.local/share/uv/python"/cpython-3.13* 2>/dev/null || true
WANT="$HOME/.local/share/uv/python/cpython-3.13-linux-x86_64-gnu/bin/python3.13"
if [ ! -x "$WANT" ]; then
  FOUND=$(find "$HOME/.local/share/uv/python" -type f -name python3.13 | head -n 1)
  echo "WANT_MISSING found=$FOUND"
  if [ -z "$FOUND" ]; then
    echo NO_PYTHON313
    exit 8
  fi
  mkdir -p "$(dirname "$WANT")"
  ln -sfn "$FOUND" "$WANT"
fi
ls -l "$WANT"
"$WANT" -V
VENV=/workspace/runpod-slim/ComfyUI/.venv-py313/bin/python
ls -l "$VENV"
"$VENV" -V
"$VENV" -c "import torch; print('TORCH', torch.__version__, 'CUDA', torch.cuda.is_available())"
