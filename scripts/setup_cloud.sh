#!/usr/bin/env bash
# Compatibility entry point. All cloud setup authority lives in
# scripts/otr_pod_provision.sh.
set -uo pipefail

echo "setup_cloud.sh now delegates to the audited OTR pod provisioner."

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd -P || true)
if [ -n "$HERE" ] && [ -f "$HERE/otr_pod_provision.sh" ]; then
  exec bash "$HERE/otr_pod_provision.sh" "$@"
fi

TEMP_SCRIPT=$(mktemp /tmp/otr_pod_provision.XXXXXX.sh) || exit 1
trap 'rm -f "$TEMP_SCRIPT"' EXIT
curl -fsSL \
  https://raw.githubusercontent.com/jbrick2070/ComfyUI-OldTimeRadio/v2.0-alpha/scripts/otr_pod_provision.sh \
  -o "$TEMP_SCRIPT" || exit 1
bash "$TEMP_SCRIPT" "$@"
