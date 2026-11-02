#!/usr/bin/env sh
# Start Radio Play Studio: installs the app's two packages the first time,
# starts comfy-api-proxy if it is not running, checks the setup, and opens the
# app in your browser. Start ComfyUI first.
#
# Which Python: RPS_PYTHON if you set it, else a .venv in this folder, else python3.
set -e
cd "$(dirname "$0")"

if [ -n "$RPS_PYTHON" ]; then
    PY="$RPS_PYTHON"
elif [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
else
    PY="python3"
fi

if ! "$PY" -c "import comfy_sdk, comfy_api_proxy" >/dev/null 2>&1; then
    echo "Installing the app's packages - comfy-sdk and comfy-api-proxy - this happens once."
    "$PY" -m pip install -r requirements.txt
fi

exec "$PY" -m radio_play_studio start "$@"
