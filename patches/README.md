# Pinned portability patches

## `ComfyUI-LTXVideo-kornia-pad.patch`

The load-bearing patch for the LTX 0.9.x lane (`ltx_8gb`): Kornia 0.8.3 removed the `pad` symbol that `pyramid_blending.py` imports, so the pinned pack fails at import until this one-file patch lands. The provisioner checks the exact commit, normalized preimage, changed-file set, and normalized postimage, and refuses drift instead of resetting or overwriting an existing installation.

- Upstream pack: `Lightricks/ComfyUI-LTXVideo`
- Required upstream commit: `3b9c5cde4700917074823d45e25401d81049f8fc`
- Patch SHA-256 (LF bytes): `109fbe2927b9c07d95d431470f7449942094fc6047dcbc9ad4a519a57ac0c993`
- Clean normalized preimage SHA-256: `08d2b18cfd325a3610683abc574e058fd209ddc7453c19b47cc108a8882a7dc1`
- Patched normalized postimage SHA-256: `19ac341bad75f8ea03988aef664924896fc24960accd2a79f415536c2833997e`

Manual repair is intentionally explicit: stop ComfyUI, check out the pinned commit into `ComfyUI/custom_nodes/ComfyUI-LTXVideo`, then run:

```bash
git -C ComfyUI/custom_nodes/ComfyUI-LTXVideo apply --ignore-space-change --ignore-whitespace /absolute/path/to/ComfyUI-OldTimeRadio/patches/ComfyUI-LTXVideo-kornia-pad.patch
```

and re-run `python scripts/otr_provision.py --packs-only` to verify the postimage. The hashes above are the ones the provisioner checks (`scripts/otr_provision.py`, `LTXVIDEO_*`).
