# Historical entry point retained so old notes fail with the current authority.
[Console]::Error.WriteLine(@'
download_humo_models.ps1 is retired. It used floating URLs and a personal model
root, so it cannot prove a portable install. The pinned, SHA-256-verified HuMo
14B owner is now the cross-platform fetch lane:

  <ComfyUI Python> scripts/otr_fetch_lane_weights.py humo

For a complete profile install, use:

  <ComfyUI Python> scripts/otr_provision.py --profile <exact-humo-profile>

See docs/RUNPOD_INSTALL.md for the single canonical RunPod playbook.
'@)
exit 2
