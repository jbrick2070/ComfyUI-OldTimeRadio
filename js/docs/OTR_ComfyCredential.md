<!-- Written by scripts/build_node_docs.py from config/node_docs.json and the node's own definition. Edit those, then run the script. -->
# 0 - Comfy Credential

Carries the Comfy API key for this queue, when you are signed in, so the Comfy Cloud writer slots can run. With nothing signed in it passes nothing and the episode stays local.

## What it does

Receives this queue's Comfy API key (from the app when you are signed in with an API key, or from a headless submitter) and hands it to the nodes that spend Comfy credits. It is the only node that sees the key, and it never fails, so the key can never appear in an error report. Keep it wired into the Workflow Validator; a local-only run needs no key and loses nothing.

## Outputs

- `credential` (`STRING`)

## Read more

- [CLOUD.md](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/CLOUD.md)

Class `OTR_ComfyCredential`, category `OldTimeRadio`.
