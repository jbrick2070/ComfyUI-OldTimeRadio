# Pinned portability patches

`ComfyUI-GGUF-ltx25-gemma4.patch` is the narrowly scoped LTX 2.5 compatibility patch used by the RunPod portability installer.

- Upstream pack: `city96/ComfyUI-GGUF`
- Required upstream commit: `6ea2651e7df66d7585f6ffee804b20e92fb38b8a`
- Patch source identity: `jbrick2070/vram-recipe-lab@56311b2d5104524eef1670fa647538d98e805c79`
- Patch SHA-256 (LF bytes): `d9185b7a8129f85b59b4df527488aa396da7c99217d336a3580a4c3d0fd4fa04`
- Clean normalized `loader.py` SHA-256: `b66b5f39a656b1ada80cc452e18cf1e71323cd52b1a61b6852cf90dbf4842345`
- Patched normalized `loader.py` SHA-256: `63f8146be990b557728e5e806547fe6f904b87318ff6c4c87dde3c73f17bdf85`
- Companion `ComfyUI-LTXVideo` commit: `3b9c5cde4700917074823d45e25401d81049f8fc`

The source and patch are Apache-2.0-compatible. The provisioner checks the exact commit, normalized preimage, changed-file set, and normalized postimage. It refuses drift instead of resetting or overwriting an existing installation.

Manual repair is intentionally explicit: stop ComfyUI, move the incompatible pack aside, check out the pinned GGUF commit into `ComfyUI/custom_nodes/ComfyUI-GGUF`, then run:

```bash
git -C ComfyUI/custom_nodes/ComfyUI-GGUF apply --ignore-space-change --ignore-whitespace /absolute/path/to/ComfyUI-OldTimeRadio/patches/ComfyUI-GGUF-ltx25-gemma4.patch
```

Re-run `python scripts/otr_provision.py --packs-only` with ComfyUI's own Python interpreter. That command verifies the postimage and installs the pack requirements.
