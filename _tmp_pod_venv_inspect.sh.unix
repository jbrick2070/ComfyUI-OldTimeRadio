#!/bin/bash
set -euo pipefail
V=/workspace/runpod-slim/ComfyUI/.venv-py313
ls -la "$V/bin" | head -40
echo "=== python symlink ==="
ls -l "$V/bin/python" "$V/bin/python3" "$V/bin/python3.13" 2>&1 || true
echo "=== file python ==="
file "$V/bin/python" 2>&1 || true
echo "=== readlink ==="
readlink -f "$V/bin/python" 2>&1 || true
echo "=== host pythons ==="
ls -l /usr/bin/python* 2>&1 | head
echo "=== py313 elsewhere ==="
find /workspace -maxdepth 5 -type f -name 'python3.13' 2>/dev/null | head
find /opt /usr/local -name 'python3.13' 2>/dev/null | head
echo "=== pyvenv.cfg ==="
cat "$V/pyvenv.cfg" 2>&1 || true
echo "=== runtime env full keys ==="
grep -E '^[A-Z_]+=' /workspace/otr-config/otr-runtime.env | cut -d= -f1
