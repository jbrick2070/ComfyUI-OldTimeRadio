#!/usr/bin/env bash
# Historical name retained so old bookmarks fail with useful instructions.
set -u

echo "download_models.sh is retired: its floating model list was not reproducible." >&2
echo "Use one of these current authorities from the OTR checkout:" >&2
echo "  <ComfyUI Python> scripts/otr_fetch_lane_weights.py --list" >&2
echo "  docs/RUNPOD_INSTALL.md  (single RunPod playbook and exact manual tiers)" >&2
exit 2
